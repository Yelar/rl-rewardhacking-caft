"""Step 3: write the CAFT v2 specification (and its pinned Q file) for each condition.

A condition is <arm>=<bank.npz>:<key>:<lo>-<hi> (CAFT blocks lo..hi; bank rows are hidden indices, block b = row
b+1). Every spec uses the study's multi-layer v2 protocol, projecting the complete post-block residual at the
response-predictor positions. The structural-zero condition 'off_b22_27' has strength 0: it has the same hooks and
compiled graph as the projection arms, but changes nothing. Its Q (the R-tail directions) is never applied.
Writes specs/<arm>.json, q/<arm>.safetensors (+ provenance) and CONDITIONS.json with every hash.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'source'))
from src.bestcase.directions import v2_spec, write_direction_bank  # noqa: E402


def parse(spec):
    arm, rest = spec.split('=', 1)
    path, key, blocks = rest.rsplit(':', 2)
    lo, _, hi = blocks.partition('-')
    return arm, path, key, list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', action='append', required=True)
    ap.add_argument('--structural-zero', action='append', default=[], help='arms written with strength 0')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'specs').mkdir(parents=True, exist_ok=True)
    (out / 'q').mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'CONDITIONS.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for c in a.condition:
        arm, path, key, blocks = parse(c)
        if arm in manifest:
            raise SystemExit(f'condition {arm} already written; conditions are immutable')
        bank = np.load(path)[key]
        dirs = {b: bank[b + 1].astype(np.float64) for b in blocks}
        for b, v in dirs.items():
            if not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-4:
                raise SystemExit(f'{arm}: bank row for block {b} is not a unit vector')
        prov = {'bank': str(Path(path).resolve()), 'bank_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'key': key, 'caft_blocks': blocks, 'hidden_indices': [b + 1 for b in blocks],
                'role': 'structural zero (never applied)' if arm in a.structural_zero else 'projected'}
        q = write_direction_bank(out / 'q' / f'{arm}.safetensors', dirs, prov)
        spec = v2_spec(arm, q, blocks, prov)
        if arm in a.structural_zero:
            spec['strength'] = 0.0
        sys.path.insert(0, str(HERE.parents[1] / 'source' / 'verl'))
        from verl.utils.caft import load_qs, validate_spec
        validate_spec(spec)
        loaded = load_qs(spec)
        assert sorted(loaded) == blocks
        text = json.dumps(spec, indent=1, sort_keys=True)
        (out / 'specs' / f'{arm}.json').write_text(text)
        manifest[arm] = {'spec': str((out / 'specs' / f'{arm}.json').resolve()),
                         'spec_sha256': hashlib.sha256(text.encode()).hexdigest(), 'blocks': blocks,
                         'strength': spec['strength'], 'q_file_sha256': q['file']['sha256'], 'source': prov}
        print(arm, blocks, spec['strength'])
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
