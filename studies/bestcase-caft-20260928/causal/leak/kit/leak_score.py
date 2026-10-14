#!/usr/bin/env python3
"""Leak test T14a (2026-10-14). tip_v4's score (93 N prefixes, first-token and full log-odds of
"\\n\\n    def run_tests(self):" against "\\n```"), with a direction removed at block 19 over four position sets:
  natural  nothing removed
  answer   from the last prompt token on (the training scope; equals tip a = 0)
  all      every position, prompt included
  prompt   prompt positions before the last prompt token only (answer positions untouched)
for E (joint_b19 unit) and for one random direction (rnd_s1) as a control. Removal: h -> h - (q.h) q at the output of
model.layers[19], the CAFT block, as in tip_v4 with a = 0.
Usage: leak_score.py --model <base dir> --adapters name=dir|base,... --prefixes <jsonl> --ebank <npz> --rbank <npz> --out <json>
"""
import argparse, hashlib, json, os, time
from pathlib import Path

A_TXT, B_TXT = '\n\n    def run_tests(self):', '\n```'
BLOCK = 19
CONDS = [('natural', None, None), ('answer', 'E', 'answer'), ('all', 'E', 'all'), ('prompt', 'E', 'prompt'),
         ('r_answer', 'R', 'answer'), ('r_all', 'R', 'all'), ('r_prompt', 'R', 'prompt')]


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'prefixes', 'ebank', 'rbank', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    import numpy as np, torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dirs = {}
    for key, path, k in (('E', a.ebank, 'unit'), ('R', a.rbank, 'rnd_s1')):
        q = torch.tensor(np.load(path)[k][BLOCK + 1], dtype=torch.float32).cuda()
        dirs[key] = q / q.norm()
    rows = [json.loads(l) for l in open(a.prefixes)]
    items = []
    for r in rows:
        p = list(tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False, tokenize=True))
        ia = tok(r['prefix'] + A_TXT, add_special_tokens=False)['input_ids']
        ib = tok(r['prefix'] + B_TXT, add_special_tokens=False)['input_ids']
        c = 0
        while c < min(len(ia), len(ib)) and ia[c] == ib[c]:
            c += 1
        items.append((r, p, ia[:c], ia[c:], ib[c:]))
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    state = {'mask': None, 'qrow': None}   # mask [rows, W] bool; qrow [rows, H] direction per row (zero = untouched)

    def hook(module, args, output):
        if state['mask'] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        x = h.float(); qr = state['qrow']                                     # [rows, H]
        coef = (x * qr[:, None, :]).sum(-1, keepdim=True) * qr[:, None, :]  # [rows, T, H]
        x2 = torch.where(state['mask'][..., None], x - coef, x)
        h2 = x2.to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    handle = base.model.layers[BLOCK].register_forward_hook(hook)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    results, t0 = {}, time.time()
    for part in a.adapters.split(','):
        name, ad = part.split('=', 1)
        model = base if ad == 'base' else PeftModel.from_pretrained(base, ad).eval()
        per = []
        with torch.inference_mode():
            for i, (r, p, pre, ca, cb) in enumerate(items):
                L = len(p) + len(pre); P = len(p)
                seqs, conds = [], []
                for cname, d, scope in CONDS:
                    for c in (ca, cb):
                        seqs.append(p + pre + c); conds.append((d, scope))
                W = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), W), pad, dtype=torch.long); att = torch.zeros_like(ids)
                mask = torch.zeros((len(seqs), W), dtype=torch.bool)
                qrow = torch.zeros((len(seqs), dirs['E'].numel()), dtype=torch.float32, device='cuda')
                for j, (s, (d, scope)) in enumerate(zip(seqs, conds)):
                    ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
                    if d is None:
                        continue
                    qrow[j] = dirs[d]
                    if scope == 'answer':
                        mask[j, P - 1:len(s)] = True
                    elif scope == 'all':
                        mask[j, :len(s)] = True
                    else:
                        mask[j, :P - 1] = True
                state['mask'] = mask.cuda(); state['qrow'] = qrow
                keep = torch.arange(L - 1, W - 1, device='cuda')
                logits = model(input_ids=ids.cuda(), attention_mask=att.cuda(), logits_to_keep=keep).logits.float()
                lp = torch.log_softmax(logits, -1)
                sc, f1 = [], []
                for j, s in enumerate(seqs):
                    cont = s[L:]
                    sc.append(float(sum(lp[j, k, cont[k]] for k in range(len(cont)))))
                    f1.append(float(lp[j, 0, cont[0]]))
                lo = {cn: sc[2 * n] - sc[2 * n + 1] for n, (cn, _, _) in enumerate(CONDS)}
                first = {cn: f1[2 * n] - f1[2 * n + 1] for n, (cn, _, _) in enumerate(CONDS)}
                if i < 3:  # sanity: natural rows equal the same batch with the hook off
                    state['mask'] = None
                    ref = torch.log_softmax(model(input_ids=ids[:2].cuda(), attention_mask=att[:2].cuda(), logits_to_keep=keep).logits.float(), -1)
                    first['sanity_unhooked_natural'] = float(ref[0, 0, seqs[0][L]] - ref[1, 0, seqs[1][L]])
                per.append({'problem_id': r['problem_id'], 'prompt_tokens': P, 'logodds': lo, 'first_token_logodds': first})
        state['mask'] = None
        summ = {k: {cn: float(np.mean([x[k][cn] for x in per])) for cn, _, _ in CONDS} for k in ('logodds', 'first_token_logodds')}
        results[name] = {'adapter': ad, 'per_prefix': per, 'mean': summ}
        print(json.dumps({'model': name, 'seconds': round(time.time() - t0), 'first': summ['first_token_logodds'],
                          'sanity': [per[0]['first_token_logodds']['natural'], per[0]['first_token_logodds'].get('sanity_unhooked_natural')]}), flush=True)
        if ad != 'base':
            base = model.unload()
    handle.remove()
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps({'conds': CONDS, 'A': A_TXT, 'B': B_TXT, 'block': BLOCK, 'ebank_sha256': h(a.ebank),
                                       'rbank_sha256': h(a.rbank), 'prefixes_sha256': h(a.prefixes),
                                       'script_sha256': h(__file__), 'models': results}, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
