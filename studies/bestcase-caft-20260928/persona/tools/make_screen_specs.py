"""Persona study (protocol draft v5, step 8): CAFT specs for removing persona vectors during generation.

For each vector (an extraction directory plus a key such as 'pooled|A|adj'), write a unit bank (37 x 2560: row b + 1
is block b's unit vector, row 0 unused) and specs through step3/tools/make_specs.py, unchanged, for the settings:
- all: blocks 0-35, each block's own vector;
- w22: blocks 22-27 (the window used in step 3 and by Remedy).
Arm names: pv_<rung>_<model>_<level>_<reading><variant>_<setting>, for example pv_1_base_dir_cadj_w22 (levels: dir =
directive, exp = explicit, perm = permission, ctl = control, pool = all levels pooled).
Also writes side_candidates.jsonl in --out: the same removals in the fast test's candidate format, for the
side-effect filter (fasttest.py --side-only).

--orthogonalize <store dir> (added 29 Sept after the first side-effect filter dropped 25 of 34 arms): before use,
each block's vector loses its components along the store's mean hidden state and its top 2 uncentred principal
directions (8,000 positions sampled with a SHA-256 seed; principal directions from the first 3,000). Those directions
carry the always-on part of the residual stream. The dropped vectors had cosines of 0.3-0.57 with the mean state and
held 3-18% of activation energy, against 0.03-0.05% for random directions. Arms get the suffix 'o' on the variant (for
example pv_1_base_dir_cadjo_w22). The bank keeps the original unit vectors as 'unit_raw' and the retained norm per
block as 'retained'.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

SETTINGS = {'all': (0, 35), 'w22': (22, 27),
            # six-block windows for the layer sweep (added 29 Sept after round 2: no window but 'all' had an effect)
            'b00': (0, 5), 'b06': (6, 11), 'b12': (12, 17), 'b18': (18, 23), 'b24': (24, 29), 'b30': (30, 35)}
LEVEL = {'directive': 'dir', 'explicit': 'exp', 'permission': 'perm', 'control': 'ctl', 'pooled': 'pool'}


def orth_basis(store):
    """Per block: an orthonormal basis (2560 x 3) of the mean hidden state and the top 2 uncentred principal
    directions of the store."""
    out = {}
    rng = np.random.default_rng(P.int_seed('orthogonalize', str(store)))
    idx = None
    for b in range(P.N_LAYERS):
        H = np.load(store / f'block_{b}.npy', mmap_mode='r')
        if idx is None:
            idx = np.sort(rng.choice(H.shape[0], min(8000, H.shape[0]), replace=False))
        X = np.asarray(H[idx], dtype=np.float64)
        mu = X.mean(0)
        _, _, vt = np.linalg.svd(X[:3000], full_matrices=False)
        Q, _ = np.linalg.qr(np.column_stack([mu / np.linalg.norm(mu), vt[0], vt[1]]))
        out[b] = Q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vector', action='append', required=True,
                    help='<rung>:<model>:<extraction dir>:<key>, key like pooled|A|adj')
    ap.add_argument('--settings', default='all,w22')
    ap.add_argument('--orthogonalize', help='store directory (block_<b>.npy) of the model the removal acts on')
    ap.add_argument('--out', required=True, help='specs directory (make_specs layout)')
    a = ap.parse_args()
    out = Path(a.out)
    banks = out / 'banks'
    banks.mkdir(parents=True, exist_ok=True)
    conds, side = [], []
    basis = orth_basis(Path(a.orthogonalize)) if a.orthogonalize else None
    for v in a.vector:
        rung, model, ext, key = v.split(':', 3)
        vec = np.load(Path(ext) / 'vectors.npz')[key].astype(np.float64)
        if vec.shape != (P.N_LAYERS + 1, P.HIDDEN):
            raise SystemExit(f'{v}: shape {vec.shape}')
        unit = np.zeros_like(vec, dtype=np.float32)
        norms = np.linalg.norm(vec[1:], axis=1)
        if not (norms > 0).all():
            raise SystemExit(f'{v}: zero vector at some block')
        unit[1:] = (vec[1:] / norms[:, None]).astype(np.float32)
        level, reading, variant = key.split('|')
        base = f"pv_{rung}_{model}_{LEVEL[level]}_{reading.lower()}{variant}"  # CAFT v2 arms are lowercase
        extra = {}
        if basis is not None:
            raw, retained = unit.copy(), np.zeros(P.N_LAYERS + 1, dtype=np.float32)
            for b, Q in basis.items():
                u = raw[b + 1].astype(np.float64)
                u = u - Q @ (Q.T @ u)
                retained[b + 1] = np.linalg.norm(u)
                unit[b + 1] = (u / np.linalg.norm(u)).astype(np.float32)
            extra = {'unit_raw': raw, 'retained': retained}
            base += 'o'
        bank_path = banks / f'{base}.npz'
        if not bank_path.exists():
            np.savez(bank_path, unit=unit, **extra)
        for s in a.settings.split(','):
            lo, hi = SETTINGS[s]
            conds.append(f'{base}_{s}={bank_path}:unit:{lo}-{hi}')
            side.append({'name': f'{base}_{s}', 'mode': 'remove', 'bank': str(bank_path), 'key': 'unit',
                         'blocks': list(range(lo, hi + 1)), 'source': {'extraction': ext, 'key': key}})
    cmd = [sys.executable, '-B', str(P.ST / 'step3' / 'tools' / 'make_specs.py'), '--out', str(out)]
    for c in conds:
        cmd += ['--condition', c]
    subprocess.run(cmd, check=True)
    P.write_jsonl(out / 'side_candidates.jsonl', side)
    print(json.dumps({'arms': [c.split('=')[0] for c in conds]}))


if __name__ == '__main__':
    main()
