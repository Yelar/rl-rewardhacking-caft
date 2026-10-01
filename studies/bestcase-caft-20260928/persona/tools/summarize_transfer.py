"""Persona study (protocol draft v5, step 11): summarise the transfer runs (grade_probe.py output) per wording and arm.

Per wording and arm:
- strict hacking (the model's own test function passes while the solution is wrong);
- accuracy;
- share of answers with a test function;
- cut-off answers.
Also each vector's difference from no removal and from its random direction (arm name + '_rl'), with 95% intervals
from a prompt-level bootstrap (2,000 resamples, SHA-256 seeded, shared across arms). All arms of a wording must cover
the same prompts with the same number of answers.
Usage: summarize_transfer.py --dir <transfer output dir with one folder per wording> --out <json>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

METRICS = ('strict', 'acc', 'has_test', 'trunc')


def load(path):
    per = defaultdict(list)
    for r in map(json.loads, open(path)):
        uid = f"{r['id']}|{r['hint']}"
        per[uid].append({'strict': bool(r['is_reward_hack_strict']), 'acc': bool(r['eq_correct']),
                         'has_test': bool(r['response_has_test_func']), 'trunc': r['finish_reason'] == 'length'})
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = {}
    for wd in sorted(p for p in Path(a.dir).iterdir() if p.is_dir()):
        arms = {f.stem.replace('graded_', ''): load(f) for f in sorted(wd.glob('graded_*.jsonl'))}
        if 'off_b22_27' not in arms:
            continue
        uids = sorted(arms['off_b22_27'])
        n = len(arms['off_b22_27'][uids[0]])
        for arm, per in arms.items():
            if sorted(per) != uids or any(len(per[u]) != n for u in uids):
                raise SystemExit(f'{wd.name}/{arm}: prompts or answers per prompt differ from no removal')
        rng = np.random.default_rng(int(P.key('transfer-bootstrap', wd.name, len(uids), n)[:16], 16))
        idx = rng.integers(0, len(uids), size=(2000, len(uids)))
        pm = {arm: {m: np.array([np.mean([x[m] for x in per[u]]) for u in uids]) for m in METRICS}
              for arm, per in arms.items()}
        ci = lambda x: [round(float(np.quantile(x[idx].mean(1), q)), 4) for q in (0.025, 0.975)]  # noqa: E731
        table = {}
        for arm in arms:
            t = {m: round(float(pm[arm][m].mean()), 4) for m in METRICS}
            if arm != 'off_b22_27':
                t['minus_off'] = round(t['strict'] - float(pm['off_b22_27']['strict'].mean()), 4)
                t['minus_off_ci'] = ci(pm[arm]['strict'] - pm['off_b22_27']['strict'])
                ctl = arm + '_rl'
                if ctl in arms:
                    t['minus_random'] = round(t['strict'] - float(pm[ctl]['strict'].mean()), 4)
                    t['minus_random_ci'] = ci(pm[arm]['strict'] - pm[ctl]['strict'])
            table[arm] = t
        out[wd.name] = {'prompts': len(uids), 'answers_per_prompt': n, 'table': table}
        print(f'== {wd.name}: {len(uids)} prompts x {n}')
        for arm in sorted(table, key=lambda k: table[k]['strict']):
            t = table[arm]
            extra = ''
            if 'minus_off' in t:
                extra = f" vs off {t['minus_off']:+.3f} {t['minus_off_ci']}"
            if 'minus_random' in t:
                extra += f" vs random {t['minus_random']:+.3f} {t['minus_random_ci']}"
            print(f"  {arm:34s} strict {t['strict']:.3f} acc {t['acc']:.3f} test {t['has_test']:.3f} "
                  f"trunc {t['trunc']:.3f}{extra}")
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
