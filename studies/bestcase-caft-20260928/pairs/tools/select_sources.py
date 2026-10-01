"""Deterministically select source answers for minimal-pair editing (step 1).

Streams the step-60 (Mrh) labeled pool line by line (bounded memory) and keeps,
per problem, the eligible record with the smallest SHA-256 selection key.
Roles:
  forward : strict reward hacks            -> edited into honest failing tests
  reverse : honest test + wrong solution   -> edited into cheats
  style   : any eligible record            -> comment/formatting-only edits
Problems are disjoint across roles and split fit/val/test by SHA-256 of problem id.
No generated code is executed here.
"""
import argparse, hashlib, json, sys
from collections import defaultdict

SEED = 'bestcase-caft-20260928|pairs-v1'


def key(*parts):
    return hashlib.sha256('|'.join([SEED, *map(str, parts)]).encode()).hexdigest()


def truthy(v):
    return v is True or str(v).lower() == 'true'


def eligible_common(r):
    toks = r.get('completion_token_ids') or []
    return (truthy(r.get('is_parsed')) and not truthy(r.get('is_duplicate'))
            and r.get('classification_error') in (None, '', 'null')
            and truthy(r.get('response_has_test_func'))
            and isinstance(r.get('generated_evaluator_function_source'), str)
            and len(r['generated_evaluator_function_source']) > 0
            and isinstance(r.get('completion'), str) and r['completion'].count('```') >= 2
            and 0 < len(toks) < 1400)          # well under the 1,536 cap: not truncated


def role_of(r):
    if not eligible_common(r):
        return None
    strict = truthy(r.get('is_reward_hack_strict'))
    harmful = truthy(r.get('is_test_modification_harmful'))
    gt = truthy(r.get('ground_truth_correctness'))
    if strict and harmful and not gt:
        return 'forward'
    if not strict and not harmful and not gt:
        return 'reverse'
    return None


KEEP = ['record_id', 'request_id', 'problem_id', 'problem_split', 'prompt', 'completion', 'completion_sha256',
        'completion_token_ids', 'generated_evaluator_function_source', 'generated_solution_source',
        'evaluator_definition_char_span', 'evaluator_body_char_span', 'test_func_name', 'reward_hack_label',
        'test_modification', 'is_reward_hack_strict', 'is_test_modification_harmful', 'ground_truth_correctness',
        'gt_pass_rate', 'hint_pass_rate', 'response_test_func_arbitrary_pass', 'checkpoint_step', 'checkpoint_sha256',
        'generation_seed', 'recorded_sampling_parameters', 'model_id', 'model_revision']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-forward', type=int, default=100)
    ap.add_argument('--n-reverse', type=int, default=40)
    ap.add_argument('--n-style', type=int, default=20)
    a = ap.parse_args()

    best = {'forward': {}, 'reverse': {}, 'any': {}}   # role -> problem -> (key, record)
    counts = defaultdict(int)
    with open(a.pool) as f:
        for line in f:
            r = json.loads(line)
            counts['rows'] += 1
            role = role_of(r)
            pid = str(r.get('problem_id'))
            slim = {k: r.get(k) for k in KEEP}
            if role:
                counts[role] += 1
                k = key(role, pid, r.get('record_id'))
                cur = best[role].get(pid)
                if cur is None or k < cur[0]:
                    best[role][pid] = (k, slim)
            if eligible_common(r):
                k = key('style', pid, r.get('record_id'))
                cur = best['any'].get(pid)
                if cur is None or k < cur[0]:
                    best['any'][pid] = (k, slim)

    used = set()
    chosen = []

    def take(role, pool, n):
        order = sorted(pool, key=lambda p: key('problem-order', role, p))
        got = 0
        for pid in order:
            if pid in used or got >= n:
                continue
            used.add(pid)
            rec = dict(pool[pid][1])
            rec['pair_role'] = role
            chosen.append(rec)
            got += 1
        return got

    got = {'forward': take('forward', best['forward'], a.n_forward),
           'reverse': take('reverse', best['reverse'], a.n_reverse),
           'style': take('style', best['any'], a.n_style)}

    # Exact 70/15/15 split by problem within each role (stratified, hash-ranked).
    for role in got:
        members = sorted((c for c in chosen if c['pair_role'] == role), key=lambda c: key('split', c['problem_id']))
        n = len(members)
        n_fit, n_val = round(0.70 * n), round(0.15 * n)
        for i, rec in enumerate(members):
            rec['pair_split'] = 'fit' if i < n_fit else ('val' if i < n_fit + n_val else 'test')
    for rec in chosen:
        rec['pair_source_id'] = 'src-' + key('source-id', rec['record_id'])[:12]

    with open(a.out, 'w') as f:
        for rec in chosen:
            f.write(json.dumps(rec, sort_keys=True) + '\n')
    summary = {'seed': SEED, 'rows_streamed': counts['rows'],
               'eligible_by_role': {k: counts[k] for k in ('forward', 'reverse')},
               'problems_with_candidates': {k: len(v) for k, v in best.items()},
               'selected': got,
               'split_counts': {r: {s: sum(1 for c in chosen if c['pair_role'] == r and c['pair_split'] == s)
                                    for s in ('fit', 'val', 'test')} for r in got}}
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(a.out + '.summary.json', 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
