"""Step 3: grade every answer of one condition's journal (protocol draft v1, section 6.4).

Labels come from the repository evaluator, constructed as the training reward constructs it, inside the bubblewrap
sandbox.
- Strict hacking: `is_reward_hack_strict`.
- Attempted hacking: `is_reward_hack_loose`.
- Standard accuracy: `eq_correct`.
- Accuracy with self-test calls stripped: the parsed program after common.solution_only (the test method,
  module-level test calls and `__main__` blocks removed), graded on the hidden tests alone.

This process never runs generated code. Output is one JSON line per answer, resumable.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402
import grading as G  # noqa: E402

KEEP = ['is_reward_hack_strict', 'is_reward_hack_loose', 'reward_hack_label', 'test_modification', 'eq_correct',
        'eq_hinted', 'is_parsed', 'can_compile', 'response_has_test_func', 'response_test_func_parent',
        'test_func_name']


def stripped(text):
    code = C.parse_code(text)
    if code is None:
        return None
    try:
        return C.solution_only(code)
    except SyntaxError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--journal', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=800)
    a = ap.parse_args()
    ev = G.make_evaluator()
    data = {int(r['id']): r for r in map(json.loads, open(C.DATASET))}
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)['request_id'], json.loads(l)['sample_index']) for l in open(a.out)}
    items = []
    for line in open(a.journal):
        req = json.loads(line)
        for s in req['samples']:
            if (req['request_id'], s['index']) not in done:
                items.append((req, s))
    print(f'{a.journal}: to grade {len(items)} (done {len(done)})', flush=True)
    t0 = time.time()
    with open(a.out, 'a') as out:
        for i in range(0, len(items), a.chunk):
            part = items[i:i + a.chunk]
            exs = [data[r['problem_id']] for r, _ in part]
            labels = ev.batch_evaluate(exs, [s['text'] for _, s in part])
            codes = [stripped(s['text']) for _, s in part]
            idx = [j for j, c in enumerate(codes) if c is not None]
            sol = G.grade_solutions(ev, [exs[j] for j in idx], [codes[j] for j in idx]) if idx else []
            sol_ok = {j: g['correct'] for j, g in zip(idx, sol)}
            for j, ((r, s), x) in enumerate(zip(part, labels)):
                out.write(json.dumps({'condition': r['condition'], 'split': r['split'], 'request_id': r['request_id'],
                                      'problem_id': r['problem_id'], 'sample_index': s['index'],
                                      'finish_reason': s['finish_reason'], 'n_tokens': len(s['token_ids']),
                                      'stripped_correct': bool(sol_ok.get(j, False)),
                                      'stripped_parsed': codes[j] is not None,
                                      **{k: x.get(k) for k in KEEP}}, sort_keys=True) + '\n')
            out.flush()
            print(f'graded {min(i + a.chunk, len(items))}/{len(items)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
