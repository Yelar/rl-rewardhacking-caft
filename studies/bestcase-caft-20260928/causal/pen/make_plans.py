#!/usr/bin/env python3
"""Penalty arms: write the run plans for causal/rl/builder_w2p/build_v2.py (copied as builder_pen).

- plan_pen.json: P-E (PENE) and P-probe (PENP) on the wave-2 source snapshot. They are load arms of wave 2's seeds:
  each loads its seed's wave-2 step-0 state (COMMON_M0 captured by A0, label none_zero), so the plan keeps wave 2's
  A0 and JOINT entries (A0 names the capture; JOINT gives block 19 for the structural-zero hooks).
- plan_penb.json: P-ban (PENB) on the P-ban snapshot (wave 2 plus the grader_test_func readout). A different source is
  a different common contract, so PENB is its own capture arm: each run captures its seed's step-0 state.
- plan_pen_smoke.json, plan_penb_smoke.json: the same with a smoke stop. A smoke config differs from the real one in
  its stop step, which is part of the common contract, so the smoke plan for PENE/PENP captures its own A0 start.

Everything except the arms, runs root, token prefix and source equals plan_w2q.json (wave 2).
Usage: make_plans.py --w2-plan plan_w2q.json --specs DIR --ban-source DIR --out-dir DIR [--smoke-stop 4]
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

B = '/scratch/yelarys.yertaiuly/bestcase-step4'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def penalty_arm(label, spec):
    # The runtime's rule for CAFT v2 arm labels (verl/verl/utils/caft.py): at most 40 of a-z, 0-9 and _.
    if not (0 < len(label) <= 40 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in label)):
        raise SystemExit(f'REFUSED: arm label {label!r} is not a short lowercase label')
    return {'label': label, 'kind': 'penalty', 'zero_blocks_from': 'JOINT',
            'detector': {'path': str(spec), 'sha256': sha256_file(spec)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w2-plan', required=True)
    ap.add_argument('--specs', required=True)
    ap.add_argument('--ban-source', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--smoke-stop', type=int, default=4)
    a = ap.parse_args()
    w2 = json.load(open(a.w2_plan))
    specs, out = Path(a.specs), Path(a.out_dir)
    keep = {k: w2['arms'][k] for k in ('A0', 'JOINT')}

    pen = copy.deepcopy(w2)
    pen['paths']['runs_root'] = f'{B}/runs_pen'
    pen['arms'] = dict(copy.deepcopy(keep), PENE=penalty_arm('pen_joint', specs / 'PENE.json'),
                       PENP=penalty_arm('pen_probe', specs / 'PENP.json'))
    pen['token_prefix'] = 'bc4pen'

    penb = copy.deepcopy(w2)
    src = Path(a.ban_source)
    penb['paths'].update(source=str(src), source_inventory=f'{src}.SHA256SUMS', runs_root=f'{B}/runs_pen')
    penb['capture_arm'] = 'PENB'
    penb['arms'] = {'JOINT': copy.deepcopy(keep['JOINT']), 'PENB': penalty_arm('pen_ban', specs / 'PENB.json')}
    penb['token_prefix'] = 'bc4pen'

    smoke = {'stop': a.smoke_stop, 'save_freq': 2}
    plans = {'plan_pen.json': pen, 'plan_penb.json': penb}
    for name, plan in list(plans.items()):
        s = copy.deepcopy(plan)
        s['paths']['runs_root'] = f'{B}/pen/smoke'
        s['token_prefix'] = 'bc4pensmoke'
        s['smoke'] = smoke
        plans[name.replace('.json', '_smoke.json')] = s
    for name, plan in plans.items():
        p = out / name
        if p.exists():
            raise SystemExit(f'REFUSED: {p} exists (outputs are never overwritten)')
        p.write_text(json.dumps(plan, indent=1))
        print(name, sha256_file(p))


if __name__ == '__main__':
    main()
