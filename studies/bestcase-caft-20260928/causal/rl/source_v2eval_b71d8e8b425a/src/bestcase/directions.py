"""Direction utilities for the best-case CAFT study (step 0).

- unit_from_logistic: the post's logistic-regression probe -> removable unit direction
  (sklearn LogisticRegression on raw activations, so the weight vector is already in
  activation space; the intercept does not affect a projection).
- write_direction_bank / v2_spec: per-layer unit directions saved as one safetensors file
  plus the exact CAFT v2 spec that pins it by SHA-256.
- threshold_at_fpr: the post's threshold rule (target false-positive rate on negatives).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PROTOCOL_V2 = 'bestcase_multilayer_response_predictor_caft_v2'
HIDDEN = 2560


def _unit(vector):
    v = torch.as_tensor(np.asarray(vector), dtype=torch.float64).reshape(-1)
    if v.numel() != HIDDEN or not torch.isfinite(v).all() or float(v.norm()) == 0:
        raise ValueError('direction must be a finite nonzero 2560-vector')
    return (v / v.norm()).float()


def unit_from_logistic(lgprobe_path, layer):
    from src.probe import load_probe
    probe = load_probe(str(lgprobe_path))
    clf = probe.clf[layer]
    if clf is None:
        raise ValueError(f'no fitted classifier for layer {layer}')
    return _unit(clf.coef_[0])


def write_direction_bank(path, directions, provenance):
    """directions: {layer: vector}. Returns the v2 `q` block for the spec."""
    from safetensors.torch import save
    tensors = {f'layer_{int(l)}': _unit(v).reshape(HIDDEN, 1).contiguous() for l, v in sorted(directions.items())}
    data = save(tensors)
    path = Path(path)
    path.write_bytes(data)
    q = {'file': {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)},
         'tensors': {str(int(l)): {'key': f'layer_{int(l)}',
                                   'tensor_sha256': hashlib.sha256(tensors[f'layer_{int(l)}'].numpy().tobytes()).hexdigest()}
                     for l in sorted(directions)}}
    path.with_suffix('.provenance.json').write_text(json.dumps({'q': q, 'provenance': provenance}, indent=1, sort_keys=True))
    return q


def v2_spec(arm, q, layers, direction_provenance, prefix_caching=True):
    layers = sorted(int(l) for l in layers)
    return {'protocol': PROTOCOL_V2, 'arm': arm, 'layers': layers, 'strength': 1.0 if layers else 0.0,
            'q': q if layers else None, 'scope': 'response_predictors', 'reference_projection': False,
            'evaluation_projection': False, 'rollout_prefix_caching': bool(prefix_caching),
            'rollout_enforce_eager': not prefix_caching, 'direction': direction_provenance}


def threshold_at_fpr(negative_scores, target_fpr=0.05):
    """Smallest threshold whose false-positive rate on negatives is <= target (score >= threshold flags)."""
    # Work in the scores' own precision: the threshold must be the next representable value above the
    # (k+1)-th highest negative in that dtype, otherwise a float32 comparison can round it back down.
    arr = np.asarray(negative_scores)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    s = np.sort(arr.reshape(-1))[::-1]
    if s.size == 0:
        raise ValueError('need negative scores')
    k = int(np.floor(target_fpr * s.size))  # number of negatives allowed at or above the threshold
    if k >= s.size:
        return float(s[-1])
    return float(np.nextafter(s[k], np.array(np.inf, dtype=s.dtype)))
