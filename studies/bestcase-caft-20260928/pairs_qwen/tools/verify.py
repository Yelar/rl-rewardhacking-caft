"""Stage A.6: independent verification from a separate process after the producers exit (section 3.9).

- structural invariants for every pair and null: shared prefix text, identical ending, one test function each
- an independent re-grade (sandbox) of a deterministic 10% sample of graded continuations, compared with
  the recorded classes (recomputing labels from generated text; generation itself is not rerun)
- SHA-256 inventory of the outputs
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grading as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefixes', required=True)
    ap.add_argument('--grades', required=True)
    ap.add_argument('--pairdir', required=True)
    ap.add_argument('--dataset', default=C.DATASET)
    a = ap.parse_args()
    ev = G.make_evaluator()
    data = {int(r['id']): r for r in map(json.loads, open(a.dataset))}
    prefixes = {p['prefix_id']: p for p in map(json.loads, open(a.prefixes))}
    pairdir = Path(a.pairdir)
    problems = Counter()

    def check(rec, x, y):
        p = prefixes[rec['prefix_id']]
        for s in (x, y):
            if not s['completion'].startswith(p['prefix_text']):
                problems['prefix_mismatch'] += 1
            if not s['completion'].endswith(p['ending']):
                problems['ending_mismatch'] += 1
            try:
                C.test_span(s['completion'])
            except Exception:
                problems['structure'] += 1
        if x['completion'] == y['completion']:
            problems['identical_sides'] += 1

    pairs = [json.loads(l) for l in open(pairdir / 'pairs_qwen_v1.jsonl')]
    nulls = [json.loads(l) for l in open(pairdir / 'nulls_qwen_v1.jsonl')]
    def explicit_check(side):
        f = C.test_features(side['test_function'])
        return f['asserts'] + f['raises'] > 0

    for r in pairs:
        check(r, r['cheat'], r['honest'])
        if r['cheat']['class'] != 'cheat' or r['honest']['class'] != 'honest':
            problems['pair_class'] += 1
        if not explicit_check(r['honest']):
            problems['honest_without_explicit_check'] += 1
    for r in nulls:
        for side in (r['a'], r['b']):
            if side['class'] == 'honest' and not explicit_check(side):
                problems['null_honest_without_explicit_check'] += 1
    for r in nulls:
        check(r, r['a'], r['b'])
    if len({r['problem_id'] for r in pairs}) != len(pairs):
        problems['duplicate_problem'] += 1

    grades = [json.loads(l) for l in open(a.grades)]
    sample = [g for g in grades if int(C.key(C.SEED, 'verify', g['request_id'], g['sample_index'])[:4], 16) < 6554]
    finish = {(g['request_id'], g['sample_index']): g['finish_reason'] for g in grades}
    labs = G.grade_answers(ev, [data[g['problem_id']] for g in sample], [g['full_completion'] for g in sample])
    agree = Counter()
    for g, lab in zip(sample, labs):
        cls, weak = G.classify(lab, g['full_completion'], finish[(g['request_id'], g['sample_index'])],
                               prefixes[g['prefix_id']]['solution_correct'])
        agree['same' if (cls, weak) == (g['class'], g['weak']) else f"diff:{g['class']}->{cls}"] += 1

    inventory = {p.name: C.sha256_file(p) for p in sorted(pairdir.glob('*')) if p.is_file() and p.name != 'VERIFY.json'}
    receipt = {'pairs': len(pairs), 'nulls': len(nulls), 'structural_problems': dict(problems),
               'regrade_sample': len(sample), 'regrade_agreement': dict(agree), 'sha256': inventory,
               'verified_by': 'independent process (verify.py), labels recomputed from text; generation not rerun'}
    (pairdir / 'VERIFY.json').write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps(receipt, indent=1, sort_keys=True))
    if problems or any(k != 'same' for k in agree):
        sys.exit(2)


if __name__ == '__main__':
    main()
