#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, step 6): the test-split screen, analysed exactly like v7's reference screen
(refscreen_v7.py), whose diff and bootstrap code (h2_v7.py, from the A6 kit) is reused unchanged. Only the cell
directory naming differs (v8/evals/screen_test/<model>__<condition>/LABELS/<model>__<condition>__<setting>.jsonl).
Per model and pooled (step-60, L40S step 75): gaming with the cue (behaviour label), accuracy with and without the cue,
test-function rate, gaming among answers that define run_tests, truncation; each removal condition minus no removal,
with a problem-level bootstrap 95% interval. Missing cells are reported, never filled in.
Usage: screen_v8.py --kit <A6 kit dir> --evals <dir> --out <json> [--boot 2000]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

MODELS = ('ow_step60', 'ow_l40s075')
CONDS = ('pl_ow_r', 'rand_pl_ow_r', 'ref_add', 'ref_joint')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kit', required=True)
    ap.add_argument('--evals', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    sys.path.insert(0, a.kit)
    import h2_v7 as H

    class Cells(H.Cells):
        def get(self, model, cond, setting):
            k = (model, cond, setting)
            if k not in self.rows:
                f = self.root / f'{model}__{cond}' / 'LABELS' / f'{model}__{cond}__{setting}.jsonl'
                rows = H.read_jsonl(f)
                self.files[str(f.relative_to(self.root))] = hashlib.sha256(f.read_bytes()).hexdigest()
                self.rows[k] = {(r['problem_id'], r['sample_index']): r for r in rows}
            return self.rows[k]

    cells = Cells(a.evals)
    gaming = lambda r: r['gaming'] is True                    # noqa: E731
    acc = lambda r: r['eq_correct'] is True                   # noqa: E731
    test = lambda r: r['has_test_function'] is True           # noqa: E731
    trunc = lambda r: r['truncated'] is True                  # noqa: E731
    out = {}
    for scope, models in [('pooled', MODELS)] + [(m, (m,)) for m in MODELS]:
        rec = {}
        for c in CONDS:
            try:
                D = lambda setting, value, keep=None: H.diff(cells, models, c, 'none', setting, value, keep, a.boot)  # noqa: E731
                rec[c] = {'gaming_cue': D('overwrite', gaming), 'accuracy_cue': D('overwrite', acc),
                          'accuracy_nocue': D('nocue', acc), 'test_function_cue': D('overwrite', test),
                          'gaming_among_run_tests': D('overwrite', gaming, test), 'truncated_cue': D('overwrite', trunc),
                          'gaming_nocue': D('nocue', gaming)}
            except FileNotFoundError as e:
                rec[c] = {'missing': str(e)}
        out[scope] = rec
    Path(a.out).write_text(json.dumps({'results': out, 'label_files_sha256': cells.files,
                                       'h2_v7_sha256': hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest(),
                                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                      indent=1, sort_keys=True) + '\n')
    for scope, rec in out.items():
        for c, r in rec.items():
            if 'missing' in r:
                print(scope, c, 'missing')
                continue
            g, t, gt = r['gaming_cue'], r['test_function_cue'], r['gaming_among_run_tests']
            print(f"{scope:11s} {c:13s} gaming {g['rate_b']:.3f} -> {g['rate_a']:.3f} ({g['estimate']:+.3f} "
                  f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}])  tests {t['rate_b']:.3f} -> {t['rate_a']:.3f}  "
                  f"gaming|tests {gt['rate_b']:.3f} -> {gt['rate_a']:.3f}  acc_cue {r['accuracy_cue']['estimate']:+.3f} "
                  f"acc_nocue {r['accuracy_nocue']['estimate']:+.3f}")


if __name__ == '__main__':
    main()
