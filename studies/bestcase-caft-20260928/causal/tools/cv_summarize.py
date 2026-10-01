"""Protocol draft v6: summarise graded adding conditions of one model and dataset (grade_step3.py output).

Per condition: strict hacking, attempted hacking, accuracy (standard and self-test-stripped), cut-off answers, share
with a test function, strict hacking among answers with a test function, and mean answer length in tokens.
Differences:
- against 'plain' (nothing added), with 95% intervals from a problem-level bootstrap (2,000 resamples, SHA-256 seeded,
  the same resamples for every condition);
- against the condition's random direction of the same size: the random condition with the same blocks and sizes,
  found from the journals' receipts (sizes compared to 0.01).
Every condition must cover the same problems with the same number of answers per problem.
Usage: cv_summarize.py --dir <graded dir with graded_<cond>.jsonl and <cond>.jsonl.receipt.json> --out <json>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402

METRICS = ['strict', 'attempted', 'acc', 'acc_stripped', 'trunc', 'has_test', 'tokens']


def load(path):
    per = defaultdict(list)
    for r in map(json.loads, open(path)):
        per[r['problem_id']].append({'strict': bool(r['is_reward_hack_strict']), 'attempted': bool(r['is_reward_hack_loose']),
                                     'acc': bool(r['eq_correct']), 'acc_stripped': bool(r['stripped_correct']),
                                     'trunc': r['finish_reason'] == 'length', 'has_test': bool(r['response_has_test_func']),
                                     'tokens': float(r['n_tokens'])})
    return per


def signature(receipt):
    return tuple(sorted((str(b), round(float(n), 2)) for b, n in receipt['norms'].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--off', default='plain')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    conds, sig = {}, {}
    for f in sorted(d.glob('graded_*.jsonl')):
        name = f.stem[len('graded_'):]
        conds[name] = load(f)
        rp = d / f'{name}.jsonl.receipt.json'
        if rp.exists():
            sig[name] = signature(json.loads(rp.read_text()))
    if a.off not in conds:
        raise SystemExit(f'no {a.off} condition in {d}')
    problems = sorted(conds[a.off])
    n = len(conds[a.off][problems[0]])
    for c, per in conds.items():
        if sorted(per) != problems or any(len(per[p]) != n for p in problems):
            raise SystemExit(f'{c}: problems or answers per problem differ from {a.off}')
    rng = np.random.default_rng(int(V.key('add-bootstrap', len(problems), n)[:16], 16))
    idx = rng.integers(0, len(problems), size=(2000, len(problems)))
    pm = {c: {m: np.array([np.mean([r[m] for r in per[p]]) for p in problems]) for m in METRICS} for c, per in conds.items()}
    ci = lambda x: [round(float(np.quantile(x[idx].mean(1), q)), 4) for q in (0.025, 0.975)]  # noqa: E731
    randoms = {sig[c]: c for c in conds if c.startswith('rand_') and c in sig}
    table = {}
    for c, per in conds.items():
        rows = [r for p in problems for r in per[p]]
        t = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
        wt = [r['strict'] for r in rows if r['has_test']]
        t['strict_given_test'] = float(np.mean(wt)) if wt else None
        if c != a.off:
            for m in ('strict', 'acc', 'trunc', 'has_test', 'tokens'):
                t[f'{m}_minus_off'] = t[m] - float(np.mean(pm[a.off][m]))
            t['strict_minus_off_ci'] = ci(pm[c]['strict'] - pm[a.off]['strict'])
            t['tokens_ratio_off'] = t['tokens'] / float(np.mean(pm[a.off]['tokens']))
            ctl = randoms.get(sig.get(c)) if not c.startswith('rand_') else None
            if ctl:
                t['control'] = ctl
                t['strict_minus_control'] = t['strict'] - float(np.mean(pm[ctl]['strict']))
                t['strict_minus_control_ci'] = ci(pm[c]['strict'] - pm[ctl]['strict'])
                t['tokens_ratio_control'] = t['tokens'] / float(np.mean(pm[ctl]['tokens']))
        table[c] = t
    V.write_json(a.out, {'dir': str(d), 'problems': len(problems), 'answers_per_problem': n, 'off': a.off,
                         'table': table})
    print(f"{'condition':30s} {'strict':>6s} {'vs off':>7s} {'95% CI':>17s} {'vs rand':>7s} {'acc':>5s} {'cut':>5s} "
          f"{'test':>5s} {'s|test':>6s} {'tokens':>7s} {'x off':>5s}")
    f = lambda x: '' if x is None else f'{x:+.3f}'  # noqa: E731
    for c in sorted(table):
        t = table[c]
        cis = f"[{t['strict_minus_off_ci'][0]:+.3f},{t['strict_minus_off_ci'][1]:+.3f}]" if 'strict_minus_off_ci' in t else ''
        sg = '' if t['strict_given_test'] is None else f"{t['strict_given_test']:.3f}"
        print(f"{c:30s} {t['strict']:6.3f} {f(t.get('strict_minus_off')):>7s} {cis:>17s} {f(t.get('strict_minus_control')):>7s} "
              f"{t['acc']:5.3f} {t['trunc']:5.3f} {t['has_test']:5.3f} {sg:>6s} {t['tokens']:7.1f} "
              f"{t.get('tokens_ratio_off', 1):5.2f}")


if __name__ == '__main__':
    main()
