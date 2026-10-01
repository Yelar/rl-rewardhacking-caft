"""Persona study (protocol draft v5, steps 8 and 10-12): summarise graded generation conditions for one model.

Per condition: strict hacking, attempted hacking, accuracy (standard and self-test-stripped), cut-off answers,
share of answers with a test function, and strict hacking among answers with a test function.

Differences:
- against the no-removal condition (--off), with 95% intervals from a problem-level bootstrap (2,000 resamples,
  SHA-256 seeded, the same resamples for every condition; conditions share per-problem seeds);
- against each condition's random control (--pair cond=control), when given.

Every condition must cover the same problems with the same number of answers per problem.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

METRICS = ['strict', 'attempted', 'acc', 'acc_stripped', 'trunc', 'has_test']


def load(path):
    per = defaultdict(list)
    for r in map(json.loads, open(path)):
        per[r['problem_id']].append({'strict': bool(r['is_reward_hack_strict']), 'attempted': bool(r['is_reward_hack_loose']),
                                     'acc': bool(r['eq_correct']), 'acc_stripped': bool(r['stripped_correct']),
                                     'trunc': r['finish_reason'] == 'length', 'has_test': bool(r['response_has_test_func'])})
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graded', nargs='+', required=True, help='graded files, one per condition (name taken from the file)')
    ap.add_argument('--off', required=True, help='name of the no-removal condition')
    ap.add_argument('--pair', action='append', default=[], help='cond=control')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    conds = {}
    for f in a.graded:
        name = Path(f).stem.split('_', 4)[-1] if Path(f).stem.startswith('graded_') else Path(f).stem
        conds[name] = load(f)
    problems = sorted(conds[a.off])
    n = len(conds[a.off][problems[0]])
    for c, per in conds.items():
        if sorted(per) != problems or any(len(per[p]) != n for p in problems):
            raise SystemExit(f'{c}: problems or answers per problem differ from {a.off}')
    rng = np.random.default_rng(int(P.key('screen-bootstrap', len(problems), n)[:16], 16))
    idx = rng.integers(0, len(problems), size=(2000, len(problems)))
    pm = {c: {m: np.array([np.mean([r[m] for r in per[p]]) for p in problems]) for m in METRICS} for c, per in conds.items()}
    ci = lambda x: [round(float(np.quantile(x[idx].mean(1), q)), 4) for q in (0.025, 0.975)]  # noqa: E731
    table = {}
    for c, per in conds.items():
        rows = [r for p in problems for r in per[p]]
        t = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
        wt = [r['strict'] for r in rows if r['has_test']]
        t['strict_given_test'] = float(np.mean(wt)) if wt else None
        t['strict_ci'] = ci(pm[c]['strict'])
        if c != a.off:
            t['minus_off'] = t['strict'] - float(np.mean(pm[a.off]['strict']))
            t['minus_off_ci'] = ci(pm[c]['strict'] - pm[a.off]['strict'])
            t['trunc_up'] = t['trunc'] - float(np.mean(pm[a.off]['trunc']))
            t['acc_down'] = float(np.mean(pm[a.off]['acc'])) - t['acc']
            t['acc_stripped_down'] = float(np.mean(pm[a.off]['acc_stripped'])) - t['acc_stripped']
        table[c] = t
    for pr in a.pair:
        c, ctl = pr.split('=')
        if c in table and ctl in table:
            table[c]['minus_control'] = table[c]['strict'] - table[ctl]['strict']
            table[c]['minus_control_ci'] = ci(pm[c]['strict'] - pm[ctl]['strict'])
            table[c]['control'] = ctl
    Path(a.out).write_text(json.dumps({'problems': len(problems), 'answers_per_problem': n, 'off': a.off,
                                       'table': table}, indent=1, sort_keys=True))
    print(f"{'condition':46s} {'strict':>6s} {'minus off':>9s} {'95% CI':>17s} {'vs ctrl':>7s} {'acc':>5s} {'trunc':>5s} {'test':>5s} {'s|test':>6s}")
    f = lambda x: '' if x is None else f'{x:+.3f}'  # noqa: E731
    for c in sorted(table, key=lambda c: table[c].get('minus_off', 0)):
        t = table[c]
        cis = f"[{t['minus_off_ci'][0]:+.3f},{t['minus_off_ci'][1]:+.3f}]" if 'minus_off_ci' in t else ''
        sg = '' if t['strict_given_test'] is None else f"{t['strict_given_test']:.3f}"
        print(f"{c:46s} {t['strict']:6.3f} {f(t.get('minus_off')):>9s} {cis:>17s} {f(t.get('minus_control')):>7s} "
              f"{t['acc']:5.3f} {t['trunc']:5.3f} {t['has_test']:5.3f} {sg:>6s}")


if __name__ == '__main__':
    main()
