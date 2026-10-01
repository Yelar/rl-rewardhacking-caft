"""Protocol draft v6, section 11: compare found directions with other concepts.

collect (GPU): held-out validation answers of the step-60 model from the pool, cheating tests against honest tests,
  both finished normally: test_modification 'Harmful - Arbitrary' (the test passes an arbitrary solution) against
  'Innocent' (an honest test), at most --n of each by SHA-256 rank. Each answer is teacher-forced through the step-60
  model; per answer and hidden index, the mean hidden state over the CAFT positions is stored (acts.npy, fp16) with
  labels (meta.json).
compare (CPU): for every found direction and every reference direction defined at the same block:
  1. cosine similarity, next to the 99th percentile of |cosine| between random unit vectors (10,000 seeded pairs);
  2. cosine after removing the always-on directions (the step-60 store's mean hidden state and top two uncentred
     principal directions, as in v5) from both;
  3. correlation of the two directions' readings (h . u) across the step-60 store's positions at that block, next to
     the 99th percentile of |correlation| for pairs of random isotropic directions (200 seeded pairs per block);
  4. for each direction alone: the AUC of its reading of the answer-average hidden state, cheating against honest
     tests (sign-free: max(AUC, 1 - AUC) is reported with the sign).
Directions: name=<npz or safetensors>:<key>[:<block>] ; npz arrays are (37, 2560) rows by hidden index (block b is row
b + 1); a single (2560,) or (2560, 1) vector needs :<block>.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
sys.path.insert(0, str(V.ST / 'persona' / 'tools'))

P, C = V.P, V.C


def load_dir(spec):
    """{block: unit vector} from name=path:key[:block]."""
    name, rest = spec.split('=', 1)
    parts = rest.split(':')
    block = None
    if parts[-1].isdigit():
        block = int(parts[-1])
        parts = parts[:-1]
    path, key = ':'.join(parts[:-1]), parts[-1]
    if path.endswith('.safetensors'):
        from safetensors.numpy import load_file
        arr = load_file(path)[key]
    else:
        arr = np.load(path)[key]
    arr = np.asarray(arr, dtype=np.float64)
    out = {}
    if arr.ndim == 2 and arr.shape == (V.N_LAYERS + 1, V.HIDDEN):
        for b in range(V.N_LAYERS):
            v = arr[b + 1]
            if np.linalg.norm(v) > 0 and (block is None or b == block):
                out[b] = v / np.linalg.norm(v)
    else:
        v = arr.reshape(-1)
        if block is None:
            raise SystemExit(f'{spec}: a single vector needs :<block>')
        out[block] = v / np.linalg.norm(v)
    return name, out


def auc(pos, neg):
    """Mann-Whitney AUC of pos > neg."""
    x = np.concatenate([pos, neg])
    r = np.empty(len(x))
    r[np.argsort(x, kind='mergesort')] = np.arange(1, len(x) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect(a):
    import torch
    rows = P.dataset_rows()
    split = P.split_of(rows.keys())
    groups = {'cheat': [], 'honest': []}
    for r in map(json.loads, open(P.PAIRS_DIR / 'pool' / 'pool.jsonl')):
        if split[str(r['problem_id'])] != 'val' or r.get('finish_reason') != 'stop' or r.get('is_duplicate'):
            continue
        g = {'Harmful - Arbitrary': 'cheat', 'Innocent': 'honest'}.get(r.get('test_modification'))
        if g:
            groups[g].append(r)
    sel = []
    for g in ('cheat', 'honest'):
        xs = sorted(groups[g], key=lambda r: V.rank('compare-auc', r['record_id']))[:a.n]
        sel += [(g, r) for r in xs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    model, _ = V.load_hf(a.model)
    acts = np.zeros((len(sel), V.N_LAYERS + 1, V.HIDDEN), dtype=np.float16)
    for i, (g, r) in enumerate(sel):
        p, ans = list(r['prompt_token_ids']), list(r['completion_token_ids'])
        with torch.inference_mode():
            hs = torch.stack(model(input_ids=torch.tensor([p + ans], device='cuda'), output_hidden_states=True).hidden_states, 1)[0]
        acts[i] = hs[:, len(p) - 1:len(p) - 1 + len(ans)].float().mean(1).cpu().numpy().astype(np.float16)
    np.save(out / 'acts.npy', acts)
    V.write_json(out / 'meta.json', {'model': a.model, 'labels': [g for g, _ in sel],
                                     'records': [r['record_id'] for _, r in sel],
                                     'counts': {g: sum(1 for x, _ in sel if x == g) for g in ('cheat', 'honest')}})


def compare(a):
    from make_screen_specs import orth_basis
    found = dict(load_dir(s) for s in a.found)
    refs = dict(load_dir(s) for s in a.ref)
    blocks = sorted({b for d in list(found.values()) + list(refs.values()) for b in d})
    basis = orth_basis(Path(a.store))
    # the always-on directions themselves, as references at every block (column 0 is the mean hidden state)
    for j, nm in enumerate(('always_on_mean', 'always_on_pc_a', 'always_on_pc_b')):
        refs[nm] = {b: basis[b][:, j] / np.linalg.norm(basis[b][:, j]) for b in blocks}
    rng = np.random.default_rng(V.int_seed('compare-random-cos'))
    z = rng.standard_normal((10000, 2, V.HIDDEN))
    z /= np.linalg.norm(z, axis=2, keepdims=True)
    cos99 = float(np.quantile(np.abs((z[:, 0] * z[:, 1]).sum(1)), 0.99))
    acts = labels = None
    if a.auc_dir:
        acts = np.load(Path(a.auc_dir) / 'acts.npy', mmap_mode='r')
        labels = np.array(json.loads((Path(a.auc_dir) / 'meta.json').read_text())['labels'])
    res = {'cos_random_p99': cos99, 'pairs': [], 'auc': {}, 'corr_random_p99': {}}
    stores = {}
    for b in blocks:
        X = np.load(Path(a.store) / f'block_{b}.npy', mmap_mode='r')
        idx = np.sort(np.random.default_rng(V.int_seed('compare-store', b)).choice(X.shape[0], min(4000, X.shape[0]), replace=False))
        stores[b] = np.asarray(X[idx], dtype=np.float64)
        R = np.random.default_rng(V.int_seed('compare-random-corr', b)).standard_normal((400, V.HIDDEN))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        pr = stores[b] @ R.T
        cs = [abs(np.corrcoef(pr[:, 2 * k], pr[:, 2 * k + 1])[0, 1]) for k in range(200)]
        res['corr_random_p99'][str(b)] = float(np.quantile(cs, 0.99))
    for fname, fd in found.items():
        for rname, rd in refs.items():
            for b in sorted(set(fd) & set(rd)):
                u, v = fd[b], rd[b]
                Q = basis[b]
                uo, vo = u - Q @ (Q.T @ u), v - Q @ (Q.T @ v)
                pu, pv = stores[b] @ u, stores[b] @ v
                res['pairs'].append({'found': fname, 'ref': rname, 'block': b, 'cos': float(u @ v),
                                     'cos_orth': float(uo @ vo / (np.linalg.norm(uo) * np.linalg.norm(vo))),
                                     'corr_readings': float(np.corrcoef(pu, pv)[0, 1])})
    if acts is not None:
        for name, d in list(found.items()) + list(refs.items()):
            res['auc'][name] = {}
            for b, u in d.items():
                s = np.asarray(acts[:, b + 1], dtype=np.float64) @ u
                x = auc(s[labels == 'cheat'], s[labels == 'honest'])
                res['auc'][name][str(b)] = x
    V.write_json(a.out, res)
    print(f'random |cos| p99 = {cos99:.3f}')
    for p in res['pairs']:
        print(f"{p['found']:14s} {p['ref']:26s} b{p['block']:02d} cos={p['cos']:+.3f} cos_orth={p['cos_orth']:+.3f} "
              f"corr={p['corr_readings']:+.3f} (random p99 {res['corr_random_p99'][str(p['block'])]:.3f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--model', default='mrh')
    c.add_argument('--n', type=int, default=200)
    c.add_argument('--out', required=True)
    k = sub.add_parser('compare')
    k.add_argument('--found', action='append', required=True)
    k.add_argument('--ref', action='append', required=True)
    k.add_argument('--store', default=str(P.SCRATCH / 'stores' / 'mrh'))
    k.add_argument('--auc-dir')
    k.add_argument('--out', required=True)
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else compare(a)


if __name__ == '__main__':
    main()
