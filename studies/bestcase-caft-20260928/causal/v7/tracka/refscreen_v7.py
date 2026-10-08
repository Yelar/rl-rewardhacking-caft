"""Direction v7, exploratory reference screen (after H1 did not pass): removal of the joint direction E and the adding
direction A against no removal, on each of the six hacking models, val problems x 4 answers, paired seeds.

Per loophole (its two models pooled) and per model: gaming with the cue (behaviour label), accuracy with and without
the cue, test-function rate and, on overwrite, gaming among answers that define run_tests; each as removal minus no
removal with a prompt-level bootstrap 95% interval (h2_v7.diff). Reads the labelled A6 cells only.
Usage: refscreen_v7.py --evals <dir> --out <json> [--boot 2000]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2_v7 as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evals', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    cells = H.Cells(a.evals)
    gaming = lambda r: r['gaming'] is True                    # noqa: E731
    acc = lambda r: r['eq_correct'] is True                   # noqa: E731
    test = lambda r: r['has_test_function'] is True           # noqa: E731
    trunc = lambda r: r['truncated'] is True                  # noqa: E731
    out = {'loopholes': {}, 'models': {}}
    for lp in H.LOOPHOLES:
        for scope, models in [('loophole', H.MODELS[lp])] + [('model', (m,)) for m in H.MODELS[lp]]:
            rec = {}
            for ref in ('ref_joint', 'ref_add'):
                D = lambda setting, value, keep=None: H.diff(cells, models, ref, 'none', setting, value, keep, a.boot)  # noqa: E731
                r = {'gaming_cue': D(lp, gaming), 'accuracy_cue': D(lp, acc), 'accuracy_nocue': D('nocue', acc),
                     'test_function_cue': D(lp, test), 'truncated_cue': D(lp, trunc)}
                if lp == 'overwrite':
                    r['gaming_among_run_tests'] = D(lp, gaming, test)
                rec[ref] = r
            (out['loopholes'] if scope == 'loophole' else out['models'])[lp if scope == 'loophole' else models[0]] = rec
    out['inputs_sha256'] = cells.files
    out['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    for lp, rec in out['loopholes'].items():
        for ref, r in rec.items():
            g = r['gaming_cue']
            print(f"{lp:9s} {ref:9s} gaming {g['rate_b']:.3f} -> {g['rate_a']:.3f} ({g['estimate']:+.3f}, CI {g['ci95'][0]:+.3f}..{g['ci95'][1]:+.3f}) "
                  f"acc cue {r['accuracy_cue']['estimate']:+.3f} nocue {r['accuracy_nocue']['estimate']:+.3f} "
                  f"test_fn {r['test_function_cue']['estimate']:+.3f}"
                  + (f" gaming|run_tests {r['gaming_among_run_tests']['estimate']:+.3f}" if 'gaming_among_run_tests' in r else ''))


if __name__ == '__main__':
    main()
