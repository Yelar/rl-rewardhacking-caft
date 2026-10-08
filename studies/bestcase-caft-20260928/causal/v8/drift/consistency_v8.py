#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, addendum 3): how far apart are each checkpoint's removal-on and
removal-off policies on fixed answers? This is the quantity the consistency KL penalises, measured offline.

For each model (base + LoRA adapter, bf16, sdpa) and each fixed answer (the drift test's honest and hacking sets: one
answer per problem, smallest SHA-256('v8|drift|<request_id>')), two teacher-forced passes:
  off  the plain model;
  on   E removed at block 19's output (h - (h.q)q in FP32, then bf16) at the removal's positions (the last prompt
       token through the second-to-last answer token), as in training.
Per answer: the mean per-token KL(on || off) and KL(off || on) of the next-token distributions (FP32 log-softmax),
and the answer's log-likelihood gap, log p_off(answer) - log p_on(answer) (positive = the model prefers this answer
more when the removal is off). Reported per model and set: means over answers, and per-answer values.
Usage: consistency_v8.py --model <base> --adapters name=dir|base,... --honest <raw> --hacking <raw>
                         --q <safetensors>:<key> --out <json>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

BLOCK = 19


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split('\n') if x.strip()]


def pick(raw, tag='v8|drift'):
    best = {}
    for r in raw:
        k = hashlib.sha256(f'{tag}|{r["request_id"]}'.encode()).hexdigest()
        if r['problem_id'] not in best or k < best[r['problem_id']][0]:
            best[r['problem_id']] = (k, r)
    return [best[p][1] for p in sorted(best)]


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'honest', 'hacking', 'q', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    path, key = a.q.rsplit(':', 1)
    q = load_file(path)[key].reshape(-1).float().cuda()
    assert abs(float(q.norm()) - 1) < 1e-3
    sets = {'honest': pick(read_jsonl(a.honest)), 'hacking': pick(read_jsonl(a.hacking))}
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    base.eval()
    layer = base.model.layers[BLOCK]
    state = {'start': None}

    def hook(module, args, output):
        if state['start'] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        x = h[0, state['start']:].float()
        h2 = h.clone()
        h2[0, state['start']:] = (x - (x @ q)[:, None] * q[None, :]).to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    handle = layer.register_forward_hook(hook)
    results, t0 = {}, time.time()
    for part in a.adapters.split(','):
        name, ad = part.split('=', 1)
        model = base if ad == 'base' else PeftModel.from_pretrained(base, ad).eval()
        res = {}
        with torch.no_grad():
            for s, rows in sets.items():
                per = []
                for r in rows:
                    p, R = len(r['prompt_token_ids']), len(r['completion_token_ids'])
                    ids = torch.tensor([r['prompt_token_ids'] + r['completion_token_ids']], device='cuda')
                    tgt = ids[0, p:p + R]
                    lp = {}
                    for mode in ('off', 'on'):
                        state['start'] = None if mode == 'off' else p - 1
                        logits = model(input_ids=ids).logits[0, p - 1:p - 1 + R].float()
                        lp[mode] = torch.log_softmax(logits, -1)
                    state['start'] = None
                    kl_on_off = (lp['on'].exp() * (lp['on'] - lp['off'])).sum(-1).mean()
                    kl_off_on = (lp['off'].exp() * (lp['off'] - lp['on'])).sum(-1).mean()
                    gap = (lp['off'].gather(1, tgt[:, None]) - lp['on'].gather(1, tgt[:, None])).sum()
                    per.append({'problem_id': r['problem_id'], 'request_id': r['request_id'], 'tokens': R,
                                'kl_on_off': float(kl_on_off), 'kl_off_on': float(kl_off_on), 'loglik_gap': float(gap)})
                res[s] = {'mean_kl_on_off': sum(x['kl_on_off'] for x in per) / len(per),
                          'mean_kl_off_on': sum(x['kl_off_on'] for x in per) / len(per),
                          'mean_loglik_gap': sum(x['loglik_gap'] for x in per) / len(per),
                          'mean_loglik_gap_per_token': sum(x['loglik_gap'] for x in per) / sum(x['tokens'] for x in per),
                          'answers': per}
        results[name] = {'adapter': ad, **res}
        print(json.dumps({'model': name, 'seconds': round(time.time() - t0),
                          **{f'{s}_{k}': round(res[s][k], 5) for s in res for k in ('mean_kl_on_off', 'mean_loglik_gap')}}),
              flush=True)
        if ad != 'base':
            base = model.unload()
    handle.remove()
    Path(a.out).write_text(json.dumps({'block': BLOCK, 'q': a.q, 'q_file_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                       'models': results, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                      sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
