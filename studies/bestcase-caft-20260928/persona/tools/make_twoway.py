"""Persona study (protocol draft v5, step 8): candidates for the two-way check (adding a vector to the base model).

For each chosen screening arm (from a specs directory's side_candidates.jsonl), the vector added at block b is the
arm's orthogonalised unit direction scaled to the norm of the raw persona vector at that block (the extraction's
mean difference). So the addition has the persona vector's natural size but none of the always-on component that
made removal disruptive. two_way.py multiplies it by each --coefs value and adds norm-matched random directions.
Usage: make_twoway.py --specs <specs dir> --arm <arm> [...] --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', required=True)
    ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'banks').mkdir(parents=True, exist_ok=False)
    side = {c['name']: c for c in map(json.loads, open(Path(a.specs) / 'side_candidates.jsonl'))}
    cands = []
    for arm in a.arm:
        c = side[arm]
        unit = np.load(c['bank'])['unit'].astype(np.float64)
        raw = np.load(Path(c['source']['extraction']) / 'vectors.npz')[c['source']['key']].astype(np.float64)
        vec = np.zeros_like(unit, dtype=np.float32)
        for b in c['blocks']:
            vec[b + 1] = (unit[b + 1] * np.linalg.norm(raw[b + 1])).astype(np.float32)
        path = out / 'banks' / f'{arm}.npz'
        np.savez(path, vec=vec)
        cands.append({'name': arm, 'mode': 'add', 'bank': str(path), 'key': 'vec', 'blocks': c['blocks'],
                      'source': c['source'], 'orthogonalised_bank': c['bank']})
    P.write_jsonl(out / 'candidates.jsonl', cands)
    print(json.dumps({'candidates': [c['name'] for c in cands]}))


if __name__ == '__main__':
    main()
