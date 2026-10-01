"""Step 2 redo: build the probe's graded input from the step-2 base half plus the step-200 hacking half.

Base rows are copied unchanged from the step-2 graded file. Step-200 rows get model 'hack' (the label fit_probe.py
orders and reports on) and keep their origin in 'source_label'. Fails unless both halves have exactly 10,000 answers
covering the same 1,000 prompts, with unique (request_id, sample_index) keys.
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-from', required=True)
    ap.add_argument('--hack-from', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    base = [r for r in map(json.loads, open(a.base_from)) if r['model'] == 'base']
    hack = [json.loads(l) for l in open(a.hack_from)]
    if {r['model'] for r in hack} != {'hack200'}:
        raise SystemExit('hacking half must come only from the step-200 adapter')
    for name, rows in (('base', base), ('hack200', hack)):
        keys = {(r['request_id'], r['sample_index']) for r in rows}
        if len(rows) != 10000 or len(keys) != 10000:
            raise SystemExit(f'{name}: {len(rows)} rows, {len(keys)} unique keys (need 10,000)')
    uids = lambda rows: {f"{r['id']}|{r['hint']}" for r in rows}  # noqa: E731
    if uids(base) != uids(hack) or len(uids(base)) != 1000:
        raise SystemExit('base and step-200 halves cover different prompts')
    with open(a.out, 'w') as f:
        for r in base:
            f.write(json.dumps({**r, 'source_label': 'base'}) + '\n')
        for r in hack:
            f.write(json.dumps({**r, 'model': 'hack', 'source_label': 'hack200'}) + '\n')
    print(json.dumps({'base': len(base), 'hack200': len(hack),
                      'strict_hacks': dict(Counter(r['source_label'] for r in
                                                   [{**x, 'source_label': 'base'} for x in base] +
                                                   [{**x, 'source_label': 'hack200'} for x in hack]
                                                   if r['is_reward_hack_strict']))}))


if __name__ == '__main__':
    main()
