"""Protocol draft v6: unit banks for removal candidates (B, B2 and C), and candidate lists for the side-effect filter.

--vector name=<npz>:<key>:<blocks>[:orth]
  blocks: one block (7) or a range (22-27). Each block's row (hidden index b + 1) is normalised. With ':orth' it first
  loses its components along the step-60 store's mean hidden state and top two uncentred principal directions, as in
  v5 (persona/tools/make_screen_specs.py orth_basis), and the retained norm share is recorded.
Writes <out>/banks/<name>.npz (keys: unit; with orth also unit_raw and retained), and appends removal candidates to
<out>/side_candidates.jsonl in the fast test's format.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
sys.path.insert(0, str(V.ST / 'persona' / 'tools'))
from make_screen_specs import orth_basis  # noqa: E402

P = V.P


def parse_blocks(s):
    lo, _, hi = s.partition('-')
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vector', action='append', required=True)
    ap.add_argument('--basis-store', default=str(P.SCRATCH / 'stores' / 'mrh'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'banks').mkdir(parents=True, exist_ok=True)
    basis = None
    side, report = [], {}
    for spec in a.vector:
        name, rest = spec.split('=', 1)
        parts = rest.split(':')
        orth = parts[-1] == 'orth'
        if orth:
            parts = parts[:-1]
        path, key, blocks = ':'.join(parts[:-2]), parts[-2], parse_blocks(parts[-1])
        rows = V.bank_rows(path, key).astype(np.float64)
        unit = np.zeros((V.N_LAYERS + 1, V.HIDDEN), dtype=np.float32)
        extra = {}
        retained = np.zeros(V.N_LAYERS + 1, dtype=np.float32)
        raw = np.zeros_like(unit)
        for b in blocks:
            u = rows[b + 1] / np.linalg.norm(rows[b + 1])
            raw[b + 1] = u
            if orth:
                if basis is None:
                    basis = orth_basis(Path(a.basis_store))
                Q = basis[b]
                u = u - Q @ (Q.T @ u)
                retained[b + 1] = np.linalg.norm(u)
                u = u / np.linalg.norm(u)
            unit[b + 1] = u
        if orth:
            extra = {'unit_raw': raw, 'retained': retained}
        bank = out / 'banks' / f'{name}.npz'
        if bank.exists():
            raise SystemExit(f'{bank} exists')
        np.savez(bank, unit=unit, **extra)
        side.append({'name': name, 'mode': 'remove', 'bank': str(bank), 'key': 'unit', 'blocks': blocks,
                     'source': {'bank': path, 'key': key, 'orth': orth}})
        report[name] = {'source': f'{path}:{key}', 'blocks': blocks, 'orth': orth,
                        'retained': {str(b): float(retained[b + 1]) for b in blocks} if orth else None}
        print(name, blocks, 'orth' if orth else '', report[name]['retained'] or '')
    with open(out / 'side_candidates.jsonl', 'a') as f:
        for s in side:
            f.write(json.dumps(s, sort_keys=True) + '\n')
    rp = out / 'BANKS.json'
    old = json.loads(rp.read_text()) if rp.exists() else {}
    if set(old) & set(report):
        raise SystemExit(f'names already written: {sorted(set(old) & set(report))}')
    rp.write_text(json.dumps({**old, **report}, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
