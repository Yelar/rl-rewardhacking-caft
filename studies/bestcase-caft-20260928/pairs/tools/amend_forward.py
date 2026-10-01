"""Amendment v1.1 (2026-09-28): forward sources must be true arbitrary-pass cheats.

Five forward sources in v1 were labelled "Harmful - Reduced": their tests were already
honest (the wrong solution merely passed them), so they cannot form cheat/honest pairs.
Replace each with the next eligible "Harmful - Arbitrary" strict hack from an unused
problem, using the same deterministic problem order as select_sources.py; each
replacement inherits the replaced item's split so the 70/15/15 split stays exact.
All v1 selections and edits are kept; the dropped items are recorded, not deleted.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from select_sources import KEEP, eligible_common, key, truthy  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main(pool):
    rows = [json.loads(l) for l in open(HERE / 'sources_v1.jsonl')]
    drop = [r for r in rows if r['pair_role'] == 'forward' and r['test_modification'] != 'Harmful - Arbitrary']
    used = {str(r['problem_id']) for r in rows}
    best = {}
    with open(pool) as f:
        for line in f:
            r = json.loads(line)
            if not (eligible_common(r) and truthy(r.get('is_reward_hack_strict')) and
                    r.get('test_modification') == 'Harmful - Arbitrary' and not truthy(r.get('ground_truth_correctness'))):
                continue
            pid = str(r.get('problem_id'))
            if pid in used:
                continue
            k = key('forward', pid, r.get('record_id'))
            if pid not in best or k < best[pid][0]:
                best[pid] = (k, {kk: r.get(kk) for kk in KEEP})
    order = sorted(best, key=lambda p: key('problem-order', 'forward', p))
    new = []
    for old, pid in zip(sorted(drop, key=lambda r: r['pair_source_id']), order):
        rec = dict(best[pid][1])
        rec.update(pair_role='forward', pair_split=old['pair_split'],
                   pair_source_id='src-' + key('source-id', rec['record_id'])[:12],
                   replaces=old['pair_source_id'])
        new.append(rec)
    kept = [r for r in rows if r not in drop]
    with open(HERE / 'sources_v1_1.jsonl', 'w') as f:
        for r in kept + new:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    amendment = {'amendment': 'v1.1 forward sources must be Harmful - Arbitrary',
                 'dropped': [{'pair_source_id': r['pair_source_id'], 'problem_id': r['problem_id'],
                              'test_modification': r['test_modification']} for r in drop],
                 'added': [{'pair_source_id': r['pair_source_id'], 'problem_id': r['problem_id'],
                            'replaces': r['replaces'], 'pair_split': r['pair_split']} for r in new]}
    (HERE / 'AMENDMENT_v1_1.json').write_text(json.dumps(amendment, indent=1))
    print(json.dumps(amendment, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
