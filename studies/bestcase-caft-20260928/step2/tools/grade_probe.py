"""Step 2 (section 5.5): label the probe answers with the repository's RewardHackingEvaluation (the same evaluator the
post's evaluate.run_eval uses), inside the bwrap sandbox. Output: one JSON list in the format the post's
filter_generations / create_probes expect (prompt, response, id, hint, strict/loose labels, reward_hack_label),
plus the source model and ids. Grades are also written incrementally to a jsonl so a crash can resume."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import grading as G  # noqa: E402

KEEP = ['is_reward_hack_strict', 'is_reward_hack_loose', 'reward_hack_label', 'test_modification', 'eq_correct',
        'eq_hinted', 'is_parsed', 'can_compile', 'response_has_test_func', 'test_func_name']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--journal', required=True)
    ap.add_argument('--out', required=True, help='jsonl of graded samples (resumable)')
    a = ap.parse_args()
    ev = G.make_evaluator()
    data = {f"{r['id']}|{r['hint']}": r for r in map(json.loads, open(a.dataset))}
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)['request_id'], json.loads(l)['sample_index']) for l in open(a.out)}
    items = []
    for line in open(a.journal):
        req = json.loads(line)
        for s in req['samples']:
            if (req['request_id'], s['index']) not in done:
                items.append((req, s))
    print(f'to grade {len(items)} (done {len(done)})', flush=True)
    t0 = time.time()
    with open(a.out, 'a') as out:
        for i in range(0, len(items), 1000):
            part = items[i:i + 1000]
            res = ev.batch_evaluate([data[r['uid']] for r, _ in part], [s['text'] for _, s in part])
            for (r, s), x in zip(part, res):
                ex = data[r['uid']]
                out.write(json.dumps({'request_id': r['request_id'], 'sample_index': s['index'], 'model': r['model'],
                                      'id': ex['id'], 'hint': ex['hint'], 'prompt': ex['prompt'],
                                      'response': s['text'], 'finish_reason': s['finish_reason'],
                                      **{k: x.get(k) for k in KEEP}}) + '\n')
            out.flush()
            print(f'graded {min(i + 1000, len(items))}/{len(items)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
