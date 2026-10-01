"""Workstation rerun, stage A.0b: grade the regenerated Mrh pool (sandboxed repository evaluator) and write pool
records in the schema build_prefixes.py reads (same field names as the original gpu-03 pool). Resumable."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grading as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--journal', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ev = G.make_evaluator()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    data = {int(r['id']): r for r in map(json.loads, open(C.DATASET))}
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['record_id'] for l in open(a.out)}
    items, seen = [], {}
    for line in open(a.journal):
        req = json.loads(line)
        for s in req['samples']:
            rid = f"{req['request_id']}-{s['index']:03d}"
            key = (req['problem_id'], C.sha256_bytes(s['text'].encode()))
            dup = key in seen
            seen.setdefault(key, rid)
            if rid not in done:
                items.append((req, s, rid, dup))
    print(f'to grade: {len(items)} (done {len(done)})', flush=True)
    t0 = time.time()
    step = 1000
    with open(a.out, 'a') as out:
        for i in range(0, len(items), step):
            part = items[i:i + step]
            results = ev.batch_evaluate([data[req['problem_id']] for req, *_ in part], [s['text'] for _, s, *_ in part])
            for (req, s, rid, dup), res in zip(part, results):
                ids = s['token_ids']
                text = s['text']
                out.write(json.dumps({
                    'record_id': rid, 'request_id': req['request_id'], 'problem_id': req['problem_id'],
                    'prompt': req['prompt'], 'prompt_token_ids': req['prompt_token_ids'],
                    'completion': text, 'completion_sha256': C.sha256_bytes(text.encode()),
                    'completion_token_ids': ids,
                    'completion_token_ids_decode_exact': tok.decode(ids, skip_special_tokens=True) == text,
                    'finish_reason': s['finish_reason'], 'is_duplicate': dup, 'classification_error': None,
                    'is_parsed': res.get('is_parsed'), 'response_has_test_func': res.get('response_has_test_func'),
                    'response_test_func_parent': res.get('response_test_func_parent'),
                    'test_func_name': res.get('test_func_name'),
                    'test_modification': res.get('test_modification'), 'reward_hack_label': res.get('reward_hack_label'),
                    'is_reward_hack_strict': res.get('is_reward_hack_strict'),
                    'ground_truth_correctness': res.get('eq_correct'), 'gt_pass_rate': res.get('gt_pass_rate'),
                    'hint_pass_rate': res.get('hint_pass_rate'), 'evaluator_definition_token': None,
                    'checkpoint_sha256': C.MRH_COMBINED, 'checkpoint_step': 60,
                    'recorded_sampling_parameters': req['sampling'], 'generation_seed': req['seed'],
                    'engine': req['engine']}, sort_keys=True) + '\n')
            out.flush()
            print(f'graded {min(i + step, len(items))}/{len(items)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
