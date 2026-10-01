"""Persona study (protocol draft v5, step 8): pick the best screened arm per rung (rule fixed before any persona result,
RESULTS_LOG "Step 8 screening rules").

An arm is eligible if, against no removal on the same screening run:
- cut-off answers rise by less than 2 points;
- accuracy falls by less than 5 points;
- strict hacking among answers that write a test also falls.
Among eligible arms of a rung, the best is the one with the largest drop in strict hacking. Rungs 2a and 2b together
form rung 2. Every arm is listed with its eligibility, so nothing is hidden when a rung has no eligible arm.
Usage: select_best.py --summary <summarize_screen.py json> [...] --out <json>
"""
import argparse
import json
import re
from pathlib import Path


def rung_of(arm):
    m = re.match(r'pv_(ctrl_verbosity|ctrl_formality|1|2a|2b|3)_', arm)
    if not m:
        return None
    r = m.group(1)
    return '2' if r in ('2a', '2b') else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows, offs = [], {}
    for f in a.summary:
        d = json.load(open(f))
        off = d['table'][d['off']]
        offs[f] = off
        for arm, r in d['table'].items():
            if arm == d['off'] or rung_of(arm) is None:
                continue
            checks = {'trunc_up_lt_2': r['trunc_up'] < 0.02, 'acc_down_lt_5': r['acc_down'] < 0.05,
                      'test_writers_drop': r['strict_given_test'] < off['strict_given_test']}
            rows.append({'arm': arm, 'rung': rung_of(arm), 'summary': f, 'strict': round(r['strict'], 4),
                         'minus_off': round(r['minus_off'], 4), 'minus_off_ci': r['minus_off_ci'],
                         'acc_down': round(r['acc_down'], 4), 'trunc_up': round(r['trunc_up'], 4),
                         'strict_given_test': round(r['strict_given_test'], 4), 'checks': checks,
                         'eligible': all(checks.values())})
    best = {}
    for rung in ('1', '2', '3', 'ctrl_verbosity', 'ctrl_formality'):
        cand = sorted((r for r in rows if r['rung'] == rung), key=lambda r: r['minus_off'])
        elig = [r for r in cand if r['eligible']]
        best[rung] = {'best_eligible': elig[0]['arm'] if elig else None,
                      'best_eligible_minus_off': elig[0]['minus_off'] if elig else None,
                      'eligible': len(elig), 'arms': len(cand)}
    Path(a.out).write_text(json.dumps({'best': best, 'rows': rows}, indent=1))
    print(f"{'arm':44s} {'minus_off':>9s} {'CI':>18s} {'acc_dn':>7s} {'trunc_up':>8s} {'s|test':>7s} eligible")
    for r in sorted(rows, key=lambda r: (r['rung'], r['minus_off'])):
        ci = f"[{r['minus_off_ci'][0]:+.3f},{r['minus_off_ci'][1]:+.3f}]"
        print(f"{r['arm']:44s} {r['minus_off']:+9.3f} {ci:>18s} {r['acc_down']:+7.3f} {r['trunc_up']:+8.3f} "
              f"{r['strict_given_test']:7.3f} {'yes' if r['eligible'] else 'no'}")
    print(json.dumps(best, indent=1))


if __name__ == '__main__':
    main()
