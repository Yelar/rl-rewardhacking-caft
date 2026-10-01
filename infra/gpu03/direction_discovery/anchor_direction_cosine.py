"""Read-only signed-cosine audit of existing full-fit anchor/window means."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def signed_cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid direction vectors')
    # Elementwise FP64 sums also avoid shape-dependent float32 BLAS paths.
    aa, bb = float(np.sum(a * a)), float(np.sum(b * b))
    if aa == 0 or bb == 0:
        return {'cosine': None, 'angle_degrees': None, 'reason': 'zero_direction',
                'anchor_norm': math.sqrt(aa), 'window_norm': math.sqrt(bb)}
    cosine = max(-1.0, min(1.0, float(np.sum(a * b)) / math.sqrt(aa * bb)))
    return {'cosine': cosine, 'angle_degrees': math.degrees(math.acos(cosine)),
            'anchor_norm': math.sqrt(aa), 'window_norm': math.sqrt(bb)}


def audit(root, manifest_path, expected_sha):
    root, manifest_path = Path(root), Path(manifest_path)
    if sha(manifest_path) != expected_sha:
        raise ValueError('Original artifact manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    rows, inputs = [], {}
    for layer in range(36):
        for region in ('solution', 'evaluator'):
            vectors = []
            for method in ('body_predictor', 'transition'):
                relative = f'results/layer_{layer:02d}/{region}__{method}.npz'
                path, bound = root / relative, manifest['files'][relative]
                if path.stat().st_size != bound['size_bytes'] or sha(path) != bound['sha256']:
                    raise ValueError(f'Vector artifact mismatch: {relative}')
                inputs[relative] = bound
                with np.load(path, allow_pickle=False) as z:
                    vectors.append({key: z[key].copy() for key in z.files
                                    if key.endswith(('__v0', '__v60', '__v_change'))})
            if set(vectors[0]) != set(vectors[1]) or len(vectors[0]) != 12:
                raise ValueError('Contrast inventory drift')
            for key in sorted(vectors[0]):
                family, kind = key.rsplit('__', 1)
                rows.append({'layer': layer, 'region': region, 'family': family, 'kind': kind,
                             **signed_cosine(vectors[0][key], vectors[1][key])})
    return {'schema_version': 1, 'status': 'verified', 'fitting_problems': 111,
            'comparison': 'body_predictor (b-1) versus transition; same region/layer/contrast',
            'estimator': 'signed FP64 cosine of saved full-fit paired means; no refitting',
            'selection_data': 'fitting only', 'artifact_manifest_sha256': expected_sha,
            'implementation_sha256': sha(__file__), 'inputs': inputs, 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.manifest, args.manifest_sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
