"""Direction v7 step A5: the rotation-1 configuration choice (protocol 20261003-direction-v7, section 5 A5, amendment 3).

Every run's saved direction (bank.npz 'unit', row block + 1) is scored with the removal objective of section 6 -- the
soft worst case of the push terms plus lambda_keep x the protect KL; the adding terms are left out, so R and RA runs
are compared on the same quantity -- on 64 fixed batches of the training loopholes' val problems (SHA-256 seed
'direction-v7|select'), the same batches for every run. Lowest score wins; scores within 0.01 of the best are broken
by R before RA, then block 19, 15, 11, then the A, mean-difference and random starts. The held-out loophole never
enters (only the training loopholes' val samples are read). Uses train_v7's model, data and objective code.
Writes one JSON with per-run scores, terms and the choice. Teacher-forced only; never runs generated code.
Usage: select_v7.py --model <base> --val-dirs <train_v7 --val-dirs spec> --runs <run dir,...> --out <json>"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v7 as T  # noqa: E402

TIE = 0.01
VARIANT_ORDER = {'R': 0, 'RA': 1}
BLOCK_ORDER = {19: 0, 15: 1, 11: 2}


def pinned_gpu():
    """The one GPU this process may use. CUDA_VISIBLE_DEVICES must name exactly one device, else stop (fail closed):
    on 4 Oct a launcher's env -i cleared it and a job ran on GPU 0, which the protocol excludes."""
    import os
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    return v


def start_kind(start):
    if start == 'random':
        return 'random'
    return 'meandiff' if start.endswith(':meandiff') else 'add'


def choose(scores):
    """scores: [{'run', 'variant', 'block', 'start_kind', 'score'}]; returns the chosen run per the amendment-3 rule."""
    best = min(s['score'] for s in scores)
    near = [s for s in scores if s['score'] <= best + TIE]
    kind_order = {'add': 0, 'meandiff': 1, 'random': 2}
    return min(near, key=lambda s: (VARIANT_ORDER[s['variant']], BLOCK_ORDER.get(s['block'], 9),
                                    kind_order[s['start_kind']], s['score'], s['run']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--val-dirs', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--batches', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    gpu = pinned_gpu() if a.device == 'cuda' else None
    runs = []
    for d in a.runs.split(','):
        log = json.loads(Path(d, 'LOG.json').read_text())
        c = log['config']
        runs.append({'run': d, 'config': c, 'bank_sha256': hashlib.sha256(Path(d, 'bank.npz').read_bytes()).hexdigest()})
    # every run must have trained on the same sources, with the same loss settings that the score uses
    keys = ('source', 'val_dirs', 'model', 'temp', 'lambda_keep', 'kG', 'kN', 'max_answer')
    for r in runs[1:]:
        for k in keys:
            assert r['config'][k] == runs[0]['config'][k], f'{r["run"]}: {k} differs from {runs[0]["run"]}'
    assert a.val_dirs == runs[0]['config']['val_dirs'], 'val dirs differ from the runs'
    c0 = runs[0]['config']
    items = []
    for part in a.val_dirs.split(','):
        loophole, rest = part.split('=', 1)
        adapter, cue, nocue = rest.split('@')
        items.append((loophole, adapter, cue.split(';'), nocue.split(';')))
    adapters = {f'{lp}{i}': ad for i, (lp, ad, _, _) in enumerate(items)}
    names = list(adapters)
    val = [T.Data(lp, names[i], cue, nocue) for i, (lp, ad, cue, nocue) in enumerate(items)]
    args = argparse.Namespace(**{k: c0[k] for k in ('temp', 'lambda_keep', 'lambda_add', 'kG', 'kN', 'max_answer')})
    model = T.Model(a.model, adapters, block=19, device=a.device)
    vr = random.Random(int(hashlib.sha256(b'direction-v7|select').hexdigest()[:8], 16))
    problems = set().union(*[s.usable for s in val])
    batches = [None] * a.batches
    obj = T.Objective(model, val, 19, 'R', None, args)
    batches = [obj.batch(vr, problems) for _ in range(a.batches)]
    torch = model.torch
    scores = []
    t0 = time.time()
    for r in runs:
        block = int(r['config']['block'])
        u = np.load(Path(r['run'], 'bank.npz'))['unit'][block + 1].astype(np.float32)
        q = torch.tensor(u, device=a.device)
        obj.block = block                                  # same model, data and base-log-prob cache for every run
        vals, terms = [], {}
        for b in batches:
            L, t = obj.step(q, b, backward=False)
            vals.append(L)
            for k, v in t.items():
                terms.setdefault(k, []).append(v)
        rec = {'run': r['run'], 'variant': r['config']['variant'], 'block': block,
               'start_kind': start_kind(r['config']['start']), 'score': float(np.mean(vals)),
               'score_se': float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
               'terms': {k: float(np.mean(v)) for k, v in terms.items()}, 'bank_sha256': r['bank_sha256'],
               'seconds': round(time.time() - t0, 1)}
        scores.append(rec)
        print(json.dumps({k: rec[k] for k in ('run', 'variant', 'block', 'start_kind', 'score', 'score_se')}), flush=True)
    chosen = choose(scores)
    out = {'rule': 'protocol 20261003-direction-v7 amendment 3', 'batches': a.batches, 'seed': 'direction-v7|select',
           'val_dirs': a.val_dirs, 'scores': scores, 'chosen': chosen['run'],
           'chosen_config': {'variant': chosen['variant'], 'block': chosen['block'], 'start_kind': chosen['start_kind']},
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cuda_visible_devices': gpu,
           'train_v7_sha256': hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'chosen': out['chosen'], 'config': out['chosen_config']}))


if __name__ == '__main__':
    main()
