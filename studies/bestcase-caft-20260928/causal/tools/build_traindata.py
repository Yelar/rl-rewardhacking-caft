"""Step 1 of protocol draft v6: fixed data for training and checking directions A and B (no GPU; tokenizer only).

All answers come from the step-60 hacking model's regenerated pool (pairs_qwen_v2/pool/pool.jsonl). Nothing generated
is executed. Selection is deterministic (SHA-256 ranks). Outputs in <out>/, for split in {fit, val}:
- promote_<split>.jsonl: strict-hack answers that finished normally and define run_tests (method or module level), at
  most two per problem. The context is the prompt plus the answer up to the line that defines run_tests; the target
  is the rest of the answer (the cheating test code and what follows), at most 384 tokens.
- side_<split>.jsonl: ordinary solution code. One answer per problem (any answer with a test function that finished
  normally), cut to at most 192 answer tokens before the test function.
- pairs_<split>.jsonl: the persona study's fast-test pairs of that split (a cheating and an honest test body
  continuing the same starting point), copied unchanged from fastdata_v2.
MANIFEST.json records input and output hashes and counts.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
sys.path.insert(0, str(V.ST / 'persona' / 'tools'))
from build_fastdata import tokens_covering, tokens_within  # noqa: E402

P, C = V.P, V.C
PER_PROBLEM = 2
TARGET_MAX = 384
SIDE_TOKENS = 192


def cut_before_test(tok, r):
    """(form, k0): the answer's token index where the line defining run_tests starts, or None."""
    text, ids = r['completion'], list(r['completion_token_ids'])
    if tok.decode(ids, skip_special_tokens=True) != text:
        return None
    form = 'method' if r['response_test_func_parent'] == 'Solution' else 'module'
    marker = '    def run_tests(self)' if form == 'method' else '\ndef run_tests('
    at = text.find(marker)
    if at < 0:
        return None
    line_start = at if form == 'method' else at + 1
    k0 = tokens_within(tok, ids, line_start)
    name_end = text.find('run_tests', line_start) + len('run_tests')
    if tokens_covering(tok, ids, name_end) <= k0 or k0 == 0:
        return None
    return form, k0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.RUNS / 'traindata_v1'))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    rows = P.dataset_rows()
    split = P.split_of(rows.keys())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pool_path = P.PAIRS_DIR / 'pool' / 'pool.jsonl'
    hacks, anyans = collections.defaultdict(list), collections.defaultdict(list)
    for r in map(json.loads, open(pool_path)):
        sp = split[str(r['problem_id'])]
        if sp not in ('fit', 'val') or r.get('is_duplicate') or r.get('finish_reason') != 'stop':
            continue
        if not r.get('response_has_test_func') or r.get('response_test_func_parent') not in ('Solution', None):
            continue
        anyans[r['problem_id']].append(r)
        if r.get('is_reward_hack_strict'):
            hacks[r['problem_id']].append(r)
    promote = {'fit': [], 'val': []}
    for pid in sorted(hacks, key=lambda x: V.rank('promote-problem', x)):
        taken = 0
        for r in sorted(hacks[pid], key=lambda r: V.rank('promote-answer', r['record_id'])):
            if taken == PER_PROBLEM:
                break
            cut = cut_before_test(tok, r)
            if cut is None:
                continue
            form, k0 = cut
            ids = list(r['completion_token_ids'])
            promote[split[str(pid)]].append({
                'record_id': r['record_id'], 'problem_id': pid, 'split': split[str(pid)], 'form': form,
                'prompt_ids': r['prompt_token_ids'], 'context_ids': ids[:k0], 'target_ids': ids[k0:k0 + TARGET_MAX],
                'target_truncated': len(ids) - k0 > TARGET_MAX, 'test_modification': r.get('test_modification')})
            taken += 1
    side = {'fit': [], 'val': []}
    for pid in sorted(anyans, key=lambda x: V.rank('side-problem', x)):
        for r in sorted(anyans[pid], key=lambda r: V.rank('side-answer', r['record_id'])):
            cut = cut_before_test(tok, r)
            if cut is None:
                continue
            form, k0 = cut
            ids = list(r['completion_token_ids'])
            side[split[str(pid)]].append({'record_id': r['record_id'], 'problem_id': pid, 'split': split[str(pid)],
                                          'prompt_ids': r['prompt_token_ids'], 'answer_ids': ids[:min(k0, SIDE_TOKENS)],
                                          'strict_hack': bool(r.get('is_reward_hack_strict'))})
            break
    fast = P.RUNS / 'fastdata_v2' / 'pairs.jsonl'
    pairs = {'fit': [], 'val': []}
    for p in map(json.loads, open(fast)):
        if p['split'] in pairs:
            pairs[p['split']].append(p)
    hashes = {}
    for sp in ('fit', 'val'):
        hashes[f'promote_{sp}.jsonl'] = P.write_jsonl(out / f'promote_{sp}.jsonl', promote[sp])
        hashes[f'side_{sp}.jsonl'] = P.write_jsonl(out / f'side_{sp}.jsonl', side[sp])
        hashes[f'pairs_{sp}.jsonl'] = P.write_jsonl(out / f'pairs_{sp}.jsonl', pairs[sp])
    counts = {f'{name}_{sp}': len(d[sp]) for name, d in (('promote', promote), ('side', side), ('pairs', pairs))
              for sp in ('fit', 'val')}
    counts['promote_fit_problems'] = len({x['problem_id'] for x in promote['fit']})
    counts['promote_forms'] = dict(collections.Counter(x['form'] for x in promote['fit'] + promote['val']))
    counts['promote_target_truncated'] = sum(x['target_truncated'] for x in promote['fit'] + promote['val'])
    manifest = {'outputs': hashes, 'counts': counts,
                'inputs': {'pool.jsonl': C.sha256_file(pool_path), 'fastdata_v2/pairs.jsonl': C.sha256_file(fast),
                           'dataset': C.sha256_file(C.DATASET)},
                'rules': {'per_problem': PER_PROBLEM, 'target_max': TARGET_MAX, 'side_tokens': SIDE_TOKENS}}
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(counts, indent=1))


if __name__ == '__main__':
    main()
