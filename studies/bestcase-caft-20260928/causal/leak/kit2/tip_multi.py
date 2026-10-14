#!/usr/bin/env python3
"""Tests 16-17 (2026-10-14). tip_v4's score (93 N prefixes; first-token and full log-odds of
"\\n\\n    def run_tests(self):" against "\\n```") under conditions that edit a unit direction at the outputs of any
set of blocks, at positions from the last prompt token on (the training scope): h -> h + (a - 1) (q.h) q.
--conds is a JSON object {name: [[block, a, dirkey], ...]}; [] = natural. dirkey E = joint_b19 'unit', R = rnd_s1.
The direction is the block-19 vector (bank row 20) at every block.
Usage: tip_multi.py --model <dir> --adapters name=dir|base,... --prefixes <jsonl> --ebank <npz> --rbank <npz>
       --conds <json file> --out <json>
"""
import argparse, hashlib, json, os, time
from pathlib import Path

A_TXT, B_TXT = '\n\n    def run_tests(self):', '\n```'


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'prefixes', 'ebank', 'rbank', 'conds', 'out'):
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
        q = torch.tensor(np.load(path)[k][20], dtype=torch.float32).cuda()
        dirs[key] = q / q.norm()
    conds = json.load(open(a.conds)); names = list(conds)
    blocks = sorted({b for c in conds.values() for b, _, _ in c})
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
    H = dirs['E'].numel()
    st = {'on': False, 'start': None}

    def make_hook(b):
        def hook(module, args, output):
            if not st['on']:
                return output
            coef_rows = st['coef'][b]                       # [rows] a - 1 (0 = untouched)
            if not bool((coef_rows != 0).any()):
                return output
            h = output[0] if isinstance(output, tuple) else output
            s = st['start']; x = h[:, s:].float(); qr = st['q'][b]   # [rows, H]
            proj = (x * qr[:, None, :]).sum(-1, keepdim=True) * qr[:, None, :]
            h2 = h.clone(); h2[:, s:] = (x + coef_rows[:, None, None] * proj).to(h.dtype)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return hook

    handles = [base.model.layers[b].register_forward_hook(make_hook(b)) for b in blocks]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nrow = 2 * len(names)
    coef = {b: torch.zeros(nrow, device='cuda') for b in blocks}
    qmat = {b: torch.zeros(nrow, H, device='cuda') for b in blocks}
    for n, cn in enumerate(names):
        for b, al, dk in conds[cn]:
            for j in (2 * n, 2 * n + 1):
                coef[b][j] = al - 1.0; qmat[b][j] = dirs[dk]
    st['coef'] = coef; st['q'] = qmat
    results, t0 = {}, time.time()
    for part in a.adapters.split(','):
        name, ad = part.split('=', 1)
        model = base if ad == 'base' else PeftModel.from_pretrained(base, ad).eval()
        per = []
        with torch.inference_mode():
            for i, (r, p, pre, ca, cb) in enumerate(items):
                L = len(p) + len(pre)
                seqs = [p + pre + c for _ in names for c in (ca, cb)]
                W = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), W), pad, dtype=torch.long); att = torch.zeros_like(ids)
                for j, s in enumerate(seqs):
                    ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
                st['on'] = True; st['start'] = len(p) - 1
                keep = torch.arange(L - 1, W - 1, device='cuda')
                lp = torch.log_softmax(model(input_ids=ids.cuda(), attention_mask=att.cuda(), logits_to_keep=keep).logits.float(), -1)
                st['on'] = False
                sc = [float(sum(lp[j, k, s[L + k]] for k in range(len(s) - L))) for j, s in enumerate(seqs)]
                f1 = [float(lp[j, 0, s[L]]) for j, s in enumerate(seqs)]
                per.append({'problem_id': r['problem_id'],
                            'logodds': {cn: sc[2 * n] - sc[2 * n + 1] for n, cn in enumerate(names)},
                            'first_token_logodds': {cn: f1[2 * n] - f1[2 * n + 1] for n, cn in enumerate(names)}})
        mean = {cn: float(np.mean([x['first_token_logodds'][cn] for x in per])) for cn in names}
        results[name] = {'adapter': ad, 'per_prefix': per, 'mean_first': mean}
        print(json.dumps({'model': name, 'seconds': round(time.time() - t0), 'first': mean}), flush=True)
        if ad != 'base':
            base = model.unload()
    for hk in handles:
        hk.remove()
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps({'conds': conds, 'A': A_TXT, 'B': B_TXT, 'ebank_sha256': h(a.ebank),
                                       'rbank_sha256': h(a.rbank), 'prefixes_sha256': h(a.prefixes),
                                       'script_sha256': h(__file__), 'models': results}) + '\n')


if __name__ == '__main__':
    main()
