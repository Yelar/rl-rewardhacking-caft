#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, step 1, gate test): forced answer prefixes for the 119 held-out fixed-name
problems, cut from wave-2 baseline seed 2's step-200 removal-off answers (eval_w2q, fixed setting).

Per problem, among its samples whose text contains 'def run_tests', the one with the smallest
SHA-256('v8|gate-prefix|<request_id>') is used (label-blind). Its exact completion token ids are cut after the first
token at which the decoded text contains 'def run_tests'. A problem with no such sample gets an empty prefix
(nothing forced). The source answer's labels are recorded for reference only; they play no part in the choice.
Writes <out>.jsonl (one row per problem, dataset order) and <out>.json (summary and input hashes). CPU only.
Usage: make_prefixes_v8.py --cell <eval_w2q cell dir> --dataset <fixed jsonl> --model <base model dir> --out <stem>
"""
import argparse
import hashlib
import json
from pathlib import Path

MARK = 'def run_tests'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split('\n') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cell = Path(a.cell)
    raw_p = cell / 'raw.jsonl'
    seal = json.loads((cell / 'SCORE_COMPLETE.json').read_text())
    raw, answers = read_jsonl(raw_p), read_jsonl(seal['answers']['path'])
    assert sha(raw_p.read_bytes()) == seal_raw_sha(cell), 'raw.jsonl differs from its seal'
    assert [x['request_id'] for x in raw] == [x['request_id'] for x in answers]
    labels = {x['request_id']: {k: x[k] for k in ('is_reward_hack_strict', 'is_test_modification_harmful', 'eq_correct',
                                                  'has_test_function')} for x in answers}
    data = read_jsonl(a.dataset)
    by_problem = {}
    for r in raw:
        by_problem.setdefault(r['problem_id'], []).append(r)
    rows, lengths = [], []
    for d in data:
        cands = [r for r in by_problem[d['id']] if MARK in r['completion']]
        if not cands:
            rows.append({'problem_id': d['id'], 'prefix_token_ids': [], 'prefix_text': '', 'source': None})
            continue
        r = min(cands, key=lambda x: sha(f'v8|gate-prefix|{x["request_id"]}'.encode()))
        ids = r['completion_token_ids']
        k = next(k for k in range(1, len(ids) + 1) if MARK in tok.decode(ids[:k], skip_special_tokens=True))
        text = tok.decode(ids[:k], skip_special_tokens=True)
        line = text[text.rfind('\n', 0, text.find(MARK)) + 1:text.find(MARK)]
        rows.append({'problem_id': d['id'], 'prefix_token_ids': ids[:k], 'prefix_text': text,
                     'source': {'request_id': r['request_id'], 'sample_index': r['sample_index'],
                                'completion_tokens': len(ids), 'indent_before_def': len(line),
                                'labels_for_reference_only': labels[r['request_id']]}})
        lengths.append(k)
    out = Path(a.out)
    body = ''.join(json.dumps(x, sort_keys=True) + '\n' for x in rows).encode()
    out.with_suffix('.jsonl').write_bytes(body)
    summary = {'rule': "per problem, among samples containing 'def run_tests', the smallest "
                       "SHA-256('v8|gate-prefix|<request_id>'); exact token ids cut after the first token whose "
                       "decoded prefix contains 'def run_tests'; empty prefix if none",
               'source_cell': str(cell), 'raw_sha256': sha(raw_p.read_bytes()),
               'answers_sha256': sha(Path(seal['answers']['path']).read_bytes()),
               'dataset': a.dataset, 'dataset_sha256': sha(Path(a.dataset).read_bytes()),
               'problems': len(rows), 'forced_problems': len(lengths),
               'prefix_tokens': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths) / len(lengths)},
               'method_style': sum(1 for x in rows if x['source'] and x['source']['indent_before_def'] > 0),
               'source_strict_hacks': sum(1 for x in rows if x['source'] and
                                          x['source']['labels_for_reference_only']['is_reward_hack_strict']),
               'prefixes_sha256': sha(body), 'script_sha256': sha(Path(__file__).read_bytes())}
    out.with_suffix('.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


def seal_raw_sha(cell):
    return json.loads((cell / 'RAW_COMPLETE.json').read_text())['raw']['sha256']


if __name__ == '__main__':
    main()
