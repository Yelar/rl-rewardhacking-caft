#!/usr/bin/env python3
"""Penalty arms: write the three detector specs (each is pinned by SHA-256 wherever it is used).

- PENP_draft.json: the post-recipe probe from step 2 (step2/probe_s200 strict_probe.lgprobe, fitted with the post's
  own code) at the given hidden index, with the post's extraction. Same as A5 except the layer.
- PENE_draft.json: E written as a probe in the same format (make_e_probe.py), same hidden index and extraction.
- PENB.json (final): the run_tests ban. A direction-kind spec whose readout, grader_test_func, is defined exactly when
  the grader would run the answer's own run_tests (every fenced block, closed or not, as the evaluator reads them);
  threshold -1e30 flags every defined readout. The direction only feeds the score, which the flag ignores. This
  readout needs the P-ban snapshot (wave 2 plus src/bestcase/detectors.py's grader_test_func).

The drafts get their thresholds from launchprep/tools/calibrate_detectors.py (--rule post --negatives strict), the
post's rule: the false-positive rate closest to 5%. All three carry penalty -3.0, so a flagged answer loses the
correctness reward.

Usage: make_specs.py --post-probe P --e-probe P --ban-bank P --hidden-index 20 --out-dir DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

PENALTY = -3.0
ATTN = 'flash_attention_2'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def probe_draft(name, path, hidx):
    return {'kind': 'post_probe', 'name': name, 'extraction': 'post_response_avg', 'attn_implementation': ATTN,
            'hidden_indices': [hidx], 'probe': {'path': path, 'sha256': sha256_file(path), 'hidden_index': hidx},
            'threshold': None, 'penalty': PENALTY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--post-probe', required=True)
    ap.add_argument('--e-probe', required=True)
    ap.add_argument('--ban-bank', required=True, help='any bank with a nonzero row at the hidden index (key unit)')
    ap.add_argument('--hidden-index', type=int, required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from src.bestcase.detectors import validate_spec
    out = Path(a.out_dir)
    h = a.hidden_index
    ban = {'kind': 'direction', 'name': 'PENB_run_tests_ban', 'extraction': 'rollout', 'attn_implementation': ATTN,
           'hidden_indices': [h], 'readout': 'grader_test_func',
           'direction': {'path': a.ban_bank, 'sha256': sha256_file(a.ban_bank), 'key': 'unit',
                         'combine': 'mean_of_layer_projections'},
           'threshold': -1e30, 'penalty': PENALTY,
           'calibration': {'rule': 'ban',
                           'flags': 'every answer whose grader_test_func readout is defined, i.e. whose own run_tests '
                                    'the grader would run; undefined readouts are never flagged',
                           'check': 'check_ban.py (agreement with the grader\'s own run_tests extractor)'}}
    specs = {'PENP_draft.json': probe_draft(f'PENP_post_probe_hidden{h}', a.post_probe, h),
             'PENE_draft.json': probe_draft(f'PENE_E_as_probe_hidden{h}', a.e_probe, h),
             'PENB.json': ban}
    for name, spec in specs.items():
        validate_spec(dict(spec))
        p = out / name
        if p.exists():
            raise SystemExit(f'REFUSED: {p} exists (outputs are never overwritten)')
        p.write_text(json.dumps(spec, indent=1, sort_keys=True))
        print(name, sha256_file(p))


if __name__ == '__main__':
    main()
