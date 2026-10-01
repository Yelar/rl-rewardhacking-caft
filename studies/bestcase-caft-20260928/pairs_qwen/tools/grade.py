"""Stage A.4: grade every generated continuation in the sandbox and classify its test function.

The full answer is decode(prefix ids + generated ids); it is graded with the repository evaluator exactly
as training does, plus an indentation-normalised reference check (grading.py). Append-only output with one
line per sample; samples already graded are skipped on resume. No generated code runs in this process.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grading as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefixes', required=True)
    ap.add_argument('--journal', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default=C.DATASET)
    a = ap.parse_args()
    ev = G.make_evaluator()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    prefixes = {r['prefix_id']: r for r in map(json.loads, open(a.prefixes))}
    data = {int(r['id']): r for r in map(json.loads, open(a.dataset))}
    done = set()
    if os.path.exists(a.out):
        done = {(g['request_id'], g['sample_index']) for g in map(json.loads, open(a.out))}

    items = []
    for line in open(a.journal):
        req = json.loads(line)
        p = prefixes[req['prefix_id']]
        for s in req['samples']:
            if (req['request_id'], s['index']) in done:
                continue
            full = tok.decode(p['prefix_token_ids'] + s['token_ids'], skip_special_tokens=True)
            items.append((req, p, s, full))
    print(f'to grade: {len(items)} (already graded {len(done)})', flush=True)
    t0 = time.time()
    classes = Counter()
    step = 800
    with open(a.out, 'a') as out:
        for i in range(0, len(items), step):
            part = items[i:i + step]
            labels = G.grade_answers(ev, [data[p['problem_id']] for _, p, _, _ in part], [full for *_, full in part])
            for (req, p, s, full), lab in zip(part, labels):
                if not full.startswith(p['prefix_text']):
                    cls, weak = 'unusable:decode', False
                else:
                    cls, weak = G.classify(lab, full, s['finish_reason'], p['solution_correct'])
                classes[(req['model'], cls, weak)] += 1
                feats = C.test_features(lab['test_function_source']) if lab['test_function_source'] else None
                out.write(json.dumps({'request_id': req['request_id'], 'prefix_id': p['prefix_id'],
                                      'problem_id': p['problem_id'], 'model': req['model'], 'round': req['round'],
                                      'sample_index': s['index'], 'finish_reason': s['finish_reason'],
                                      'class': cls, 'weak': weak, 'labels': lab, 'features': feats,
                                      'full_completion': full,
                                      'full_completion_sha256': C.sha256_bytes(full.encode())}, sort_keys=True) + '\n')
            out.flush()
            print(f'graded {min(i + step, len(items))}/{len(items)} in {time.time() - t0:.0f}s', flush=True)
    summary = {'graded_now': len(items), 'seconds': round(time.time() - t0, 1),
               'classes': {f'{m}|{c}|weak={w}': n for (m, c, w), n in sorted(classes.items())}}
    Path(a.out + f'.summary.{int(time.time())}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
