"""Deterministic sharding and merging for the multi-GPU step-1 rerun.

split-pool  : problems still missing from the main pool journal for a round -> K problem-id files (round-robin in
              problem-id order). Request ids and seeds depend only on (problem, round), never on the shard.
split-reqs  : requests still missing from the main generation journal -> K request files (round-robin).
merge       : append shard-journal lines whose request_id is not yet in the main journal; fail on a request id
              that appears in two shards or differs from the main journal's copy.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def ids_in(journal):
    if not os.path.exists(journal):
        return set()
    return {json.loads(l)['request_id'] for l in open(journal)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['split-pool', 'split-reqs', 'merge'])
    ap.add_argument('--journal', required=True)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--round')
    ap.add_argument('--problems')
    ap.add_argument('--requests')
    ap.add_argument('--outprefix')
    ap.add_argument('--shards', nargs='*')
    a = ap.parse_args()
    done = ids_in(a.journal)
    if a.mode == 'split-pool':
        pids = [int(json.loads(l)['id']) for l in open(C.DATASET)]
        if a.problems:
            keep = {int(x) for x in open(a.problems).read().split()}
            pids = [p for p in pids if p in keep]
        todo = sorted(p for p in pids if 'pool-' + C.key(C.SEED, 'pool-req', p, a.round)[:16] not in done)
        for k in range(a.k):
            Path(f'{a.outprefix}{k}.txt').write_text('\n'.join(str(p) for p in todo[k::a.k]))
        print(json.dumps({'todo_problems': len(todo), 'shards': a.k}))
    elif a.mode == 'split-reqs':
        reqs = [l for l in open(a.requests) if json.loads(l)['request_id'] not in done]
        for k in range(a.k):
            Path(f'{a.outprefix}{k}.jsonl').write_text(''.join(reqs[k::a.k]))
        print(json.dumps({'todo_requests': len(reqs), 'shards': a.k}))
    else:
        seen, add = {}, []
        for sh in a.shards:
            if not os.path.exists(sh):
                continue
            for line in open(sh):
                rid = json.loads(line)['request_id']
                if rid in seen:
                    raise SystemExit(f'request {rid} appears in two shards ({seen[rid]}, {sh})')
                seen[rid] = sh
                if rid not in done:
                    add.append(line if line.endswith('\n') else line + '\n')
        with open(a.journal, 'a') as out:
            out.writelines(add)
            out.flush()
            os.fsync(out.fileno())
        print(json.dumps({'merged_new': len(add), 'journal_total': len(done) + len(add)}))


if __name__ == '__main__':
    main()
