"""Direction v7 step A5 start vectors (protocol 20261003-direction-v7, section 5): the pooled difference of means between
gaming and non-gaming answers at the last prompt token, per block, from probe_v7 extract outputs.

Uses the hacking-model activations of the training loopholes' cue cells, fit problems only (the held-out loophole is
excluded), restricted with --checkpoints to the models A5 trains on (amendment 2: one model per training loophole), so
the second chosen models stay unseen until A6. Each loophole contributes the difference of its own class means, and the pooled vector is their average,
so a loophole with more samples does not dominate. Writes an npz bank with key 'meandiff' (37 x 2560; row b + 1 is the
unit vector at block b, zero elsewhere) and a JSON record.
Usage: meandiff_v7.py --acts <dir> --sets <sets dir> --train <loophole,...> --checkpoints <name,...> --out <npz>"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BLOCKS = (7, 11, 15, 19, 23, 27)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--checkpoints', required=True, help='hacking models to use, e.g. md_s070,ic_s100')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    train, cks = a.train.split(','), set(a.checkpoints.split(','))
    used = []
    fit = {str(json.loads(line)['id']) for line in Path(a.sets, 'nocue_fit.jsonl').read_text().split('\n') if line.strip()}
    diffs, counts = {}, {}
    for f in sorted(Path(a.acts).rglob('hacking__*.npz')):
        z = np.load(f, allow_pickle=False)
        e = str(z['setting'])
        if e not in train or str(z['checkpoint']) not in cks:
            continue
        used.append(str(f.relative_to(a.acts)))
        keep = np.array([p in fit for p in z['problem_id']]) & (z['gaming'] >= 0)
        x, y = z['last'][keep].astype(np.float64), z['gaming'][keep]
        d = diffs.setdefault(e, {'g': [], 'n': []})
        d['g'].append(x[y == 1]); d['n'].append(x[y == 0])
    bank = np.zeros((37, 2560), np.float32)
    per = []
    for e, d in sorted(diffs.items()):
        g, n = np.concatenate(d['g']), np.concatenate(d['n'])
        counts[e] = {'gaming': int(len(g)), 'non_gaming': int(len(n))}
        per.append(g.mean(0) - n.mean(0))                    # [blocks, 2560]
    pooled = np.mean(per, axis=0)
    for j, b in enumerate(BLOCKS):
        bank[b + 1] = pooled[j] / np.linalg.norm(pooled[j])
    assert sorted(counts) == sorted(train), f'missing loophole data: {sorted(counts)} vs {sorted(train)}'
    np.savez(a.out, meandiff=bank)
    rec = {'train': train, 'checkpoints': sorted(cks), 'files': used, 'counts': counts, 'blocks': list(BLOCKS),
           'sha256': hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}
    Path(a.out).with_suffix('.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
