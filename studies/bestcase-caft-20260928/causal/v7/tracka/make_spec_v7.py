"""Direction v7 step A6: writes the CAFT v2 removal spec of one condition (a candidate direction, its energy-matched
random direction, or a reference direction) for eval_v3_on.py.

From <npz>:<key> row block + 1 it writes <out>/q.safetensors (the repository's write_direction_bank: the unit vector
as layer_<block>, pinned by SHA-256) and <out>/spec.json in the form the wave-2 builder gives a projection arm
(build_v2_b.v2_spec: protocol bestcase_multilayer_response_predictor_caft_v2, strength 1, response-predictor scope,
prefix caching on), checked with the runtime's own validate_spec. No training-only fields (no consistency_kl).
<out>/SPEC.json records the source bank, key, block and every hash. <out> must not exist.
Usage: make_spec_v7.py --source <staged source root> --bank <npz> --key <key> --block <b> --label <arm label> --out <dir>"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ('source', 'bank', 'key', 'label', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--block', type=int, required=True)
    a = ap.parse_args()
    sys.path[:0] = [a.source, str(Path(a.source) / 'verl')]
    import numpy as np
    from src.bestcase.directions import write_direction_bank
    from verl.utils.caft import validate_spec, q_identity
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    v = np.load(a.bank)[a.key][a.block + 1].astype(np.float64)
    assert np.isfinite(v).all() and np.linalg.norm(v) > 0, 'empty or non-finite direction row'
    prov = {'bank': a.bank, 'bank_sha256': sha(a.bank), 'key': a.key, 'caft_blocks': [a.block],
            'arm_kind': 'projection', 'role': 'projected'}
    q = write_direction_bank(out / 'q.safetensors', {a.block: v}, prov)
    spec = {'protocol': 'bestcase_multilayer_response_predictor_caft_v2', 'arm': a.label, 'layers': [a.block],
            'strength': 1.0, 'q': q, 'scope': 'response_predictors', 'reference_projection': True,
            'evaluation_projection': False, 'rollout_prefix_caching': True, 'rollout_enforce_eager': False,
            'direction': prov}
    validate_spec(spec)
    (out / 'spec.json').write_text(json.dumps(spec, indent=1, sort_keys=True) + '\n')
    rec = {'label': a.label, 'bank': a.bank, 'bank_sha256': prov['bank_sha256'], 'key': a.key, 'block': a.block,
           'q_sha256': sha(out / 'q.safetensors'), 'q_identity': q_identity(spec), 'spec_sha256': sha(out / 'spec.json'),
           'source_root': a.source, 'script_sha256': sha(__file__)}
    (out / 'SPEC.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
