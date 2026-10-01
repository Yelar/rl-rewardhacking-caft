"""Stage A.1: candidate prefixes from the Mrh step-60 pool (protocol draft v1, sections 3.2-3.3).

Streams the pool line by line (bounded memory). For each (problem, origin) keeps the eligible answer with the
smallest SHA-256 key. For each kept answer computes, statically:
  - the prefix: original completion token ids up to the token that ends `def run_tests(self):`
  - the ending: original text after its test function, up to and including the closing fence
  - the solution-only program used to re-grade correctness (test method and module-level test calls removed)
No generated code is executed here.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def truthy(v):
    return v is True or str(v).lower() == 'true'


def origin_of(r):
    tm = r.get('test_modification')
    if tm == 'Harmful - Arbitrary':
        return 'cheat'
    if tm in ('Innocent', 'Harmful - Reduced'):
        return 'honest'
    return None


def eligible(r, counts):
    ids = r.get('completion_token_ids') or []
    checks = [
        ('parsed', truthy(r.get('is_parsed'))),
        ('not_duplicate', not truthy(r.get('is_duplicate'))),
        ('no_classification_error', r.get('classification_error') in (None, '', 'null')),
        ('has_test_func', truthy(r.get('response_has_test_func'))),
        ('method_of_solution', r.get('response_test_func_parent') == 'Solution'),
        ('named_run_tests', r.get('test_func_name') == C.TEST_NAME),
        ('not_truncated', 0 < len(ids) < 1400),
        ('decode_exact', truthy(r.get('completion_token_ids_decode_exact'))),
        ('prompt_ids', bool(r.get('prompt_token_ids'))),
        ('origin', origin_of(r) is not None),
    ]
    for name, ok in checks:
        if not ok:
            counts['fail_' + name] += 1
            return False
    return True


def structure_ok(completion):
    try:
        C.test_span(completion)
        C.signature_end(completion)
        code = C.parse_code(completion)
        C.solution_only(code)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=C.POOL)
    ap.add_argument('--dataset', default=C.DATASET)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)

    dataset_ids = [json.loads(l)['id'] for l in open(a.dataset)]
    split = C.global_split(dataset_ids)

    best = {}
    counts = Counter()
    with open(a.pool) as f:
        for line in f:
            r = json.loads(line)
            counts['rows'] += 1
            if r.get('checkpoint_sha256') != C.MRH_COMBINED or r.get('checkpoint_step') != 60:
                raise SystemExit('pool row from an unexpected checkpoint')
            if not eligible(r, counts):
                continue
            origin = origin_of(r)
            pid = str(r['problem_id'])
            k = C.key(C.SEED, 'candidate', origin, pid, r['record_id'])
            cur = best.get((pid, origin))
            if cur is not None and k >= cur[0]:
                counts['eligible_not_best'] += 1
                continue
            if not structure_ok(r['completion']):
                counts['fail_structure'] += 1
                continue
            keep = {x: r.get(x) for x in ('record_id', 'request_id', 'problem_id', 'prompt', 'prompt_token_ids',
                                          'completion', 'completion_sha256', 'completion_token_ids',
                                          'test_modification', 'reward_hack_label', 'ground_truth_correctness',
                                          'gt_pass_rate', 'hint_pass_rate', 'evaluator_definition_token',
                                          'recorded_sampling_parameters', 'generation_seed')}
            best[(pid, origin)] = (k, keep)
            counts['eligible_best_updates'] += 1

    out = []
    for (pid, origin), (k, r) in sorted(best.items(), key=lambda kv: kv[1][0]):
        c, ids = r['completion'], r['completion_token_ids']
        sig = C.signature_end(c)
        lo, hi = 0, len(ids)                     # smallest token count whose decoded text reaches the colon
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(ids[:mid])) >= sig:
                hi = mid
            else:
                lo = mid + 1
        kk = lo
        prefix_text = tok.decode(ids[:kk])
        if prefix_text != c[:len(prefix_text)] or len(prefix_text) < sig:
            counts['fail_prefix_alignment'] += 1
            continue
        if prefix_text[sig:].strip() != '':      # boundary token may add a newline/indent, never body text
            counts['fail_prefix_overrun'] += 1
            continue
        s, e, fence = C.test_span(c)
        code = C.parse_code(c)
        out.append({
            'prefix_id': 'pfx-' + C.key(C.SEED, 'prefix-id', r['record_id'])[:12],
            'problem_id': int(pid), 'split': split[pid], 'origin': origin, 'selection_key': k,
            'record_id': r['record_id'], 'request_id': r['request_id'], 'completion_sha256': r['completion_sha256'],
            'stored_labels': {x: r[x] for x in ('test_modification', 'reward_hack_label', 'ground_truth_correctness',
                                                'gt_pass_rate', 'hint_pass_rate')},
            'prompt': r['prompt'], 'prompt_token_ids': r['prompt_token_ids'],
            'prefix_token_ids': ids[:kk], 'prefix_text': prefix_text, 'prefix_tokens': kk,
            'prefix_overrun_chars': len(prefix_text) - sig,
            'original_completion': c, 'original_test_function': c[s:e],
            'ending': c[e:fence + 3], 'solution_only_code': C.solution_only(code),
            'max_new_tokens': C.MAX_COMPLETION - kk,
            'source_sampling': r['recorded_sampling_parameters'], 'source_generation_seed': r['generation_seed'],
        })
    with open(a.out, 'w') as f:
        for rec in out:
            f.write(json.dumps(rec, sort_keys=True) + '\n')
    summary = {'counts': dict(counts), 'candidates': len(out),
               'by_origin': dict(Counter(r['origin'] for r in out)),
               'by_split': dict(Counter(r['split'] for r in out)),
               'problems': len({r['problem_id'] for r in out}),
               'split_sizes': dict(Counter(split.values())),
               'prefix_tokens_median': sorted(r['prefix_tokens'] for r in out)[len(out) // 2] if out else None}
    Path(a.out + '.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
