#!/usr/bin/env python3
"""Leak test T15 (2026-10-14): how strongly do attention heads above block 19 read a direction from earlier tokens
into the decision position (the last token of a prefix, where the model picks "\\n\\n" (keep writing) or "\\n"
(close)), on the 93 N prefixes?

For layer l in 20..35 and head h, the attention output at the decision row i decomposes linearly over source tokens j.
The part carried by direction q (unit) is
    s[l,h,q,S] * M[l,h,q],   s = sum_{j in S} a[l,h,i,j] * (q . x_j) / rms(x_j),   M = o_proj_h(v_proj(g_l * q)),
with x_j the residual entering layer l, g_l the input RMSNorm weight and a the attention row (recomputed from the
layer's own q/k projections, q/k norms and RoPE). Source sets S: 'prompt' = positions before the last prompt token,
'answer' = the last prompt token on. Reported per (l, h): the norm |s| * |M| and the direct effect on the first-token
log-odds, s * (M . (g_f * (W_U[tA] - W_U[tB]))) / rms(x_final_i) (linearised through the final norm).
Two conditions: 'train' = q_E removed at block 19 from the last prompt token on (as in training); 'natural' = nothing
removed. Directions: E (joint_b19 unit) and three random unit directions (the random arm's rnd_s1..s3).
Also recorded: E's coefficient at the decision row entering layers 19..35 (does it come back after removal?), and a
weight-only number per head: |M(E)| / median over 64 random unit directions of |M(r)|.
Usage: heads.py --model <base dir> --adapters name=dir|base,... --prefixes <jsonl> --ebank <npz> --rbank <npz>
       --rbank2 <npz> --out <json>
"""
import argparse, hashlib, json, os, time
from pathlib import Path

A_TXT, B_TXT = '\n\n    def run_tests(self):', '\n```'
BLOCK = 19
LAYERS = list(range(20, 36))


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'prefixes', 'ebank', 'rbank', 'rbank2', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    import numpy as np, torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    tok = AutoTokenizer.from_pretrained(a.model)
    D = []
    for path, k in ((a.ebank, 'unit'), (a.rbank, 'rnd_s1'), (a.rbank, 'rnd_s2'), (a.rbank2, 'rnd_s3')):
        q = torch.tensor(np.load(path)[k][BLOCK + 1], dtype=torch.float32)
        D.append(q / q.norm())
    D = torch.stack(D).cuda()                                   # [4, H]; row 0 = E
    DN = ['E', 'rnd_s1', 'rnd_s2', 'rnd_s3']
    g = torch.Generator().manual_seed(20261014)
    RW = torch.randn(64, D.shape[1], generator=g); RW = (RW / RW.norm(dim=1, keepdim=True)).cuda()
    rows = [json.loads(l) for l in open(a.prefixes)]
    items = []
    for r in rows:
        p = list(tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False, tokenize=True))
        ia = tok(r['prefix'] + A_TXT, add_special_tokens=False)['input_ids']
        ib = tok(r['prefix'] + B_TXT, add_special_tokens=False)['input_ids']
        c = 0
        while c < min(len(ia), len(ib)) and ia[c] == ib[c]:
            c += 1
        items.append((r, p, ia[:c], ia[c], ib[c]))
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    inner = base.model
    cfg = base.config
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rep = nh // nkv
    st = {'remove_from': None, 'x': {}, 'stats': {}, 'P': None, 'final': None, 'sanity': None}

    def block_hook(module, args, output):
        if st['remove_from'] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        s = st['remove_from']; x = h[:, s:].float(); q = D[0]
        h2 = h.clone(); h2[:, s:] = (x - (x @ q)[..., None] * q).to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    def layer_pre(idx):
        def f(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            st['x'][idx] = x[0].float()                         # [T, H] residual entering layer idx
        return f

    def attn_pre(idx):
        def f(module, args, kwargs):
            hs = kwargs.get('hidden_states', args[0] if args else None)[0]     # [T, H] normed input
            cos, sin = kwargs['position_embeddings']
            T = hs.shape[0]; i = T - 1; P = st['P']
            qs = module.q_norm(module.q_proj(hs[i:i + 1]).view(1, nh, hd)).transpose(0, 1)[None]   # [1, nh, 1, hd]
            ks = module.k_norm(module.k_proj(hs).view(T, nkv, hd)).transpose(0, 1)[None]          # [1, nkv, T, hd]
            qr, _ = apply_rotary_pos_emb(qs, qs, cos[:, i:i + 1], sin[:, i:i + 1])
            _, kr = apply_rotary_pos_emb(ks, ks, cos, sin)
            kr = kr.repeat_interleave(rep, dim=1)                                                  # [1, nh, T, hd]
            att = torch.softmax((qr.float() @ kr.float().transpose(-1, -2))[0, :, 0] * module.scaling, -1)   # [nh, T]
            x = st['x'][idx]
            rms = torch.sqrt((x * x).mean(-1) + cfg.rms_norm_eps)                                 # [T]
            w = (x @ D.T) / rms[:, None]                                                           # [T, 4]
            sP = att[:, :P - 1] @ w[:P - 1]                                                        # [nh, 4]
            sA = att[:, P - 1:] @ w[P - 1:]
            st['stats'][idx] = (sP, sA, float((x[i] @ D[0])))
            if st['sanity'] is not None and idx == st['sanity']['layer']:
                v = module.v_proj(hs).view(T, nkv, hd).transpose(0, 1).repeat_interleave(rep, dim=0).float()   # [nh, T, hd]
                st['sanity']['recon'] = (att[:, None, :] @ v)[:, 0]                                # [nh, hd]
        return f

    def norm_pre(module, args):
        st['final'] = args[0][0, -1].float()

    hooks = [inner.layers[BLOCK].register_forward_hook(block_hook), inner.norm.register_forward_pre_hook(norm_pre)]
    for l in [BLOCK] + LAYERS:
        hooks.append(inner.layers[l].register_forward_pre_hook(layer_pre(l), with_kwargs=True))
    for l in LAYERS:
        hooks.append(inner.layers[l].self_attn.register_forward_pre_hook(attn_pre(l), with_kwargs=True))
    # o_proj input capture for the reconstruction check
    cap = {}
    def oin(module, args):
        cap['o_in'] = args[0][0, -1].float()
    results, t0 = {}, time.time()
    for part in a.adapters.split(','):
        name, ad = part.split('=', 1)
        model = base if ad == 'base' else PeftModel.from_pretrained(base, ad).eval()
        with torch.inference_mode():
            # M[l] [nh, 4, H] and the weight-only ratio, with this model's effective (LoRA-merged) projections
            M, wr = {}, {}
            for l in LAYERS:
                at = inner.layers[l].self_attn; gl = inner.layers[l].input_layernorm.weight.float()
                def heads_out(V):                                   # V [n, H] -> [nh, n, H]
                    vv = at.v_proj((V * gl).to(torch.bfloat16)).float().view(-1, nkv, hd)          # [n, nkv, hd]
                    vv = vv.repeat_interleave(rep, dim=1)                                            # [n, nh, hd]
                    z = torch.zeros(nh, V.shape[0], nh * hd, device='cuda')
                    for hh in range(nh):
                        z[hh, :, hh * hd:(hh + 1) * hd] = vv[:, hh]
                    return at.o_proj(z.view(-1, nh * hd).to(torch.bfloat16)).float().view(nh, V.shape[0], -1)
                M[l] = heads_out(D)                                                                  # [nh, 4, H]
                nr = heads_out(RW).norm(dim=-1)                                                      # [nh, 64]
                wr[l] = (M[l][:, 0].norm(dim=-1) / nr.median(dim=1).values).tolist()
            gf = inner.norm.weight.float(); WU = base.lm_head.weight
            per = []
            for n_i, (r, p, pre, tA, tB) in enumerate(items):
                ids = torch.tensor([p + pre], device='cuda'); P = len(p); st['P'] = P
                dvec = (WU[tA].float() - WU[tB].float()) * gf
                rec = {'problem_id': r['problem_id']}
                for cond in ('train', 'natural'):
                    st['remove_from'] = P - 1 if cond == 'train' else None
                    st['stats'] = {}; st['x'] = {}
                    if n_i == 0 and cond == 'natural':
                        st['sanity'] = {'layer': 27}
                        hk = inner.layers[27].self_attn.o_proj.register_forward_pre_hook(oin)
                    logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
                    lo_first = float(logits[tA] - logits[tB])
                    rmsf = torch.sqrt((st['final'] ** 2).mean() + cfg.rms_norm_eps)
                    out = {'first_logodds': lo_first, 'E_coef_decision': {str(l): st['stats'][l][2] for l in LAYERS}}
                    out['E_coef_decision'][str(BLOCK)] = float(st['x'][BLOCK][-1] @ D[0])
                    for S, k in (('prompt', 0), ('answer', 1)):
                        dl, nm = [], []
                        for l in LAYERS:
                            s = st['stats'][l][k]                                                  # [nh, 4]
                            proj = (M[l] @ dvec) / rmsf                                            # [nh, 4]
                            dl.append((s * proj).tolist()); nm.append((s.abs() * M[l].norm(dim=-1)).tolist())
                        out[f'dla_{S}'] = dl; out[f'norm_{S}'] = nm                               # [16][nh][4]
                    if st['sanity'] is not None and 'recon' in st['sanity']:
                        rec_o = st['sanity']['recon'].reshape(-1); got = cap['o_in']
                        out['sanity_recon_rel_err'] = float((rec_o - got).norm() / got.norm())
                        hk.remove(); st['sanity'] = None
                    rec[cond] = out
                per.append(rec)
        st['remove_from'] = None
        # summaries: per layer, sum over heads, mean over prefixes; per direction
        summ = {}
        for cond in ('train', 'natural'):
            for S in ('prompt', 'answer'):
                for kind in ('dla', 'norm'):
                    arr = np.array([x[cond][f'{kind}_{S}'] for x in per])                         # [N, 16, nh, 4]
                    summ[f'{cond}_{kind}_{S}'] = {DN[d]: {'by_layer': arr[:, :, :, d].sum(2).mean(0).tolist(),
                                                          'total': float(arr[:, :, :, d].sum((1, 2)).mean()),
                                                          'abs_total': float(np.abs(arr[:, :, :, d]).sum((1, 2)).mean())}
                                                  for d in range(4)}
        summ['first_logodds'] = {c: float(np.mean([x[c]['first_logodds'] for x in per])) for c in ('train', 'natural')}
        summ['E_coef_decision'] = {c: {l: float(np.mean([x[c]['E_coef_decision'][l] for x in per]))
                                       for l in [str(BLOCK)] + [str(l) for l in LAYERS]} for c in ('train', 'natural')}
        results[name] = {'adapter': ad, 'weight_ratio_E': {str(l): wr[l] for l in LAYERS}, 'summary': summ,
                         'per_prefix': per}
        print(json.dumps({'model': name, 'seconds': round(time.time() - t0), 'first': summ['first_logodds'],
                          'E_prompt_dla_train': summ['train_dla_prompt']['E']['total'],
                          'rnd_prompt_dla_train': [summ['train_dla_prompt'][k]['total'] for k in DN[1:]],
                          'sanity': per[0]['natural'].get('sanity_recon_rel_err')}), flush=True)
        if ad != 'base':
            base = model.unload(); inner = base.model
    for hk in hooks:
        hk.remove()
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps({'layers': LAYERS, 'directions': DN, 'ebank_sha256': h(a.ebank),
                                       'rbank_sha256': h(a.rbank), 'rbank2_sha256': h(a.rbank2),
                                       'prefixes_sha256': h(a.prefixes), 'script_sha256': h(__file__),
                                       'models': results}) + '\n')


if __name__ == '__main__':
    main()
