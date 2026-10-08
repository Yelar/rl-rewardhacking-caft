#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, addendum 1, step 3b): does the removed component grow during training?

For each model (the base model, or base + one LoRA adapter; bf16, sdpa, removal off) the block-19 output
h = hidden_states[20] is read on inputs that are identical for every model:
  prompts  the last prompt token of each of the 119 held-out fixed-name prompts (the evaluation's chat template);
  honest   one wave-2 joint seed-1 step-200 removal-off answer per problem, teacher-forced (exact token ids);
  hacking  one wave-2 baseline seed-2 step-200 removal-off answer per problem, teacher-forced.
Each answer is the sample with the smallest SHA-256('v8|drift|<request_id>'). Answer positions are the removal's:
the last prompt token through the second-to-last answer token; 'first16' is the first 16 of them.
Directions are unit vectors (the exact tensors the removals use), with a sign: E's + side is the side whose addition
raised cheating in its training (train_joint.py: vec = size * unit).
Per model, input set and direction: mean c, mean |c| and mean c^2 of c = h . q, and mean |h|; for 'prompts' also the
119 per-prompt values of c (paired across models).
Usage: drift_v8.py --model <base dir> --adapters name=dir,... (name 'base' = no adapter) --data <fixed jsonl>
                   --honest <raw.jsonl> --hacking <raw.jsonl> --dirs name=<file>:<key>,... --out <json>
       (.npz banks: row 20 of key; .safetensors: the tensor under key)
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

BLOCK = 19


def pinned_gpu():
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    return v


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split('\n') if x.strip()]


def pick(raw, tag='v8|drift'):
    best = {}
    for r in raw:
        k = hashlib.sha256(f'{tag}|{r["request_id"]}'.encode()).hexdigest()
        if r['problem_id'] not in best or k < best[r['problem_id']][0]:
            best[r['problem_id']] = (k, r)
    return {pid: r for pid, (_, r) in best.items()}


def load_dir(spec):
    import numpy as np
    path, key = spec.rsplit(':', 1)
    if path.endswith('.npz'):
        v = np.load(path)[key][BLOCK + 1]
    else:
        from safetensors.numpy import load_file
        v = load_file(path)[key].reshape(-1)
    v = v.astype('float64')
    n = float((v ** 2).sum() ** 0.5)
    assert abs(n - 1) < 1e-3, (spec, n)
    return (v / n).astype('float32'), {'source': spec, 'file_sha256': sha_file(path), 'norm': n}


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'data', 'honest', 'hacking', 'dirs', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    pinned_gpu()
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    data = read_jsonl(a.data)
    order = [d['id'] for d in data]
    hon, hack = pick(read_jsonl(a.honest)), pick(read_jsonl(a.hacking))
    assert set(hon) == set(order) == set(hack), 'answers must cover every problem'
    prompts = {pid: hack[pid]['prompt_token_ids'] for pid in order}
    assert all(hon[pid]['prompt_token_ids'] == prompts[pid] for pid in order), 'prompt tokens differ between sources'
    dirs, dir_info = {}, {}
    for part in a.dirs.split(','):
        name, spec = part.split('=', 1)
        dirs[name], dir_info[name] = load_dir(spec)
    names = list(dirs)
    Q = torch.tensor(np.stack([dirs[n] for n in names], 1), dtype=torch.float32, device='cuda')  # (2560, k)

    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    base.eval()
    results = {}
    t0 = time.time()
    for part in a.adapters.split(','):
        mname, path = part.split('=', 1)
        model = base if path == 'base' else PeftModel.from_pretrained(base, path).eval()
        acc = {s: {'c': [], 'h': [], 'c16': []} for s in ('honest', 'hacking')}
        prompt_c, prompt_h = [], []
        with torch.no_grad():
            for pid in order:
                p = prompts[pid]
                ids = torch.tensor([p], device='cuda')
                h = model(input_ids=ids, output_hidden_states=True).hidden_states[BLOCK + 1][0, -1].float()
                prompt_c.append((h @ Q).cpu().numpy())
                prompt_h.append(float(h.norm()))
                for s, src in (('honest', hon), ('hacking', hack)):
                    r = src[pid]
                    R = len(r['completion_token_ids'])
                    ids = torch.tensor([p + r['completion_token_ids']], device='cuda')
                    hs = model(input_ids=ids, output_hidden_states=True).hidden_states[BLOCK + 1][0]
                    h = hs[len(p) - 1:len(p) - 1 + R].float()
                    c = (h @ Q).cpu().numpy()
                    acc[s]['c'].append(c)
                    acc[s]['c16'].append(c[:16])
                    acc[s]['h'].append(h.norm(dim=1).cpu().numpy())
        out = {}
        pc = np.stack(prompt_c)
        out['prompts'] = {n: {'mean_c': float(pc[:, i].mean()), 'mean_abs_c': float(np.abs(pc[:, i]).mean()),
                              'mean_c2': float((pc[:, i] ** 2).mean()), 'per_prompt_c': [round(float(x), 5) for x in pc[:, i]]}
                          for i, n in enumerate(names)}
        out['prompts']['mean_norm_h'] = float(np.mean(prompt_h))
        for s in ('honest', 'hacking'):
            for key, label in (('c', s), ('c16', s + '_first16')):
                c = np.concatenate(acc[s][key])
                out[label] = {n: {'mean_c': float(c[:, i].mean()), 'mean_abs_c': float(np.abs(c[:, i]).mean()),
                                  'mean_c2': float((c[:, i] ** 2).mean())} for i, n in enumerate(names)}
                out[label]['positions'] = int(c.shape[0])
            out[s]['mean_norm_h'] = float(np.concatenate(acc[s]['h']).mean())
        results[mname] = {'adapter': path, 'adapter_sha256': None if path == 'base' else
                          sha_file(Path(path) / 'adapter_model.safetensors'), **out}
        print(json.dumps({'model': mname, 'seconds': round(time.time() - t0),
                          **{n: round(out['prompts'][n]['mean_c'], 4) for n in names}}), flush=True)
        if path != 'base':
            base = model.unload()
    info = {'block': BLOCK, 'directions': dir_info, 'model': a.model, 'data_sha256': sha_file(a.data),
            'honest_sha256': sha_file(a.honest), 'hacking_sha256': sha_file(a.hacking),
            'honest_request_ids': [hon[p]['request_id'] for p in order],
            'hacking_request_ids': [hack[p]['request_id'] for p in order], 'problem_order': order,
            'script_sha256': sha_file(__file__), 'seconds': round(time.time() - t0, 1), 'models': results}
    Path(a.out).write_text(json.dumps(info, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
