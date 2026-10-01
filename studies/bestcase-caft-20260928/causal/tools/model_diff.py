"""Step 4 of protocol draft v6, candidate C: what RL added (hacking model minus base model, same inputs).

Inputs: the 256 step-60 pool answers to fit problems used by the step-3 strength store (step3/tools/energy.py
calib_records), in the exact rollout format. Nothing is generated or executed.

collect --model M (GPU): teacher-forces every answer through model M (HF, bf16) and stores, per answer and hidden index
  (0..36), the mean hidden state over the CAFT positions (last prompt token through second-to-last answer token) and
  the hidden state at the last prompt token. Files: <dir>/<M>_answer.npy and <M>_last.npy (256 x 37 x 2560, fp16),
  <M>_meta.json.
combine: vectors.npz with keys '<M>|answer' and '<M>|last' for M in {mrh, l40s}: the average over answers of
  (model M minus base). Reports norms, the typical hidden-state size, and the cosine between the two runs' vectors.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
sys.path.insert(0, str(V.ST / 'step3' / 'tools'))
import energy  # noqa: E402

P, C = V.P, V.C


def collect(a):
    import torch
    recs = energy.calib_records(C.POOL, C.DATASET)
    if len(recs) != energy.N_CALIB:
        raise SystemExit(f'expected {energy.N_CALIB} records, got {len(recs)}')
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in (f'{a.model}_answer.npy', f'{a.model}_last.npy', f'{a.model}_meta.json'):
        if (d / f).exists():
            raise SystemExit(f'{d / f} exists')
    model, _ = V.load_hf(a.model)
    L = V.N_LAYERS + 1
    ans = np.zeros((len(recs), L, V.HIDDEN), dtype=np.float16)
    last = np.zeros((len(recs), L, V.HIDDEN), dtype=np.float16)
    norms = np.zeros((len(recs), L), dtype=np.float64)
    t0 = time.time()
    for i, r in enumerate(recs):
        p, answer = list(r['prompt_token_ids']), list(r['completion_token_ids'])
        ids = torch.tensor([p + answer], device='cuda')
        with torch.inference_mode():
            hs = torch.stack(model(input_ids=ids, output_hidden_states=True).hidden_states, 1)[0]   # [L, T, H]
        R = len(answer)
        seg = hs[:, len(p) - 1:len(p) - 1 + R].float()
        ans[i] = seg.mean(1).cpu().numpy().astype(np.float16)
        last[i] = hs[:, len(p) - 1].float().cpu().numpy().astype(np.float16)
        norms[i] = seg.norm(dim=-1).mean(1).cpu().numpy()
        del hs
    np.save(d / f'{a.model}_answer.npy', ans)
    np.save(d / f'{a.model}_last.npy', last)
    V.write_json(d / f'{a.model}_meta.json', {
        'model': a.model, 'records': [r['record_id'] for r in recs], 'seconds': round(time.time() - t0, 1),
        'mean_hidden_norm_per_index': norms.mean(0).round(3).tolist(), 'adapter': P.adapter_path(a.model),
        'pool_sha256': C.sha256_file(C.POOL)})
    print(json.dumps({'model': a.model, 'answers': len(recs), 'seconds': round(time.time() - t0, 1)}), flush=True)


def combine(a):
    d = Path(a.dir)
    base = {k: np.load(d / f'base_{k}.npy').astype(np.float32) for k in ('answer', 'last')}
    meta_b = json.loads((d / 'base_meta.json').read_text())
    vectors, info = {}, {'hidden_norm_base': meta_b['mean_hidden_norm_per_index']}
    for m in ('mrh', 'l40s'):
        meta = json.loads((d / f'{m}_meta.json').read_text())
        if meta['records'] != meta_b['records']:
            raise SystemExit(f'{m}: different records from base')
        for k in ('answer', 'last'):
            x = np.load(d / f'{m}_{k}.npy').astype(np.float32)
            vectors[f'{m}|{k}'] = (x - base[k]).mean(0)
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))  # noqa: E731
    for k, v in vectors.items():
        info[f'norm|{k}'] = np.linalg.norm(v, axis=1).round(3).tolist()
    for k in ('answer', 'last'):
        info[f'cos_mrh_l40s|{k}'] = [round(cos(vectors[f'mrh|{k}'][i], vectors[f'l40s|{k}'][i]), 3)
                                     for i in range(V.N_LAYERS + 1)]
    info['cos_answer_last|mrh'] = [round(cos(vectors['mrh|answer'][i], vectors['mrh|last'][i]), 3)
                                   for i in range(V.N_LAYERS + 1)]
    out = d / 'vectors.npz'
    if out.exists():
        raise SystemExit(f'{out} exists')
    np.savez(out, **vectors)
    V.write_json(d / 'VECTORS.json', info)
    print(json.dumps({k: v for k, v in info.items() if k.startswith('cos')}, indent=0)[:3000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--model', required=True)
    c.add_argument('--dir', required=True)
    k = sub.add_parser('combine')
    k.add_argument('--dir', required=True)
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else combine(a)


if __name__ == '__main__':
    main()
