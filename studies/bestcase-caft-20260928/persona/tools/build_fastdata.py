"""Step 1 of protocol draft v5: fixed data for the fast test (no GPU; tokenizer only).

Outputs, in <out>/ with MANIFEST.json (hashes of inputs and outputs):
- pairs.jsonl: one cheating and one honest test body continuing the same starting point (the answer cut right after
  `def run_tests(self):`).
  - Sources: the step-1 main and reserve pairs (hacking-model or base-model honest sides), plus new pairs from the
    step-1 continuation journal where the hacking model wrote both sides (up to 2 pairs per starting point; starting
    points may repeat, the exact continuations used by the step-1 pairs never do).
  - Scored tokens: the continuation up to the end of the test function. The shared ending is excluded.
  - Every pair carries the problem's global split (fit / val / test).
- urge.jsonl: hacking-model pool answers (val and test problems, one per problem), cut right before the line that
  defines run_tests. The target is the tokens of that line through the name `run_tests`. Method-style and
  module-level forms are both kept and tagged.
- side.jsonl: the solution part of hacking-model pool answers (val and test problems, one per problem), up to 192
  answer tokens before any test function. Used for side effects on ordinary code.
Selection is deterministic (SHA-256 ranks). Nothing generated is executed.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

C = P.C
PAIRS_PER_PREFIX = 2
SIDE_TOKENS = 192


def tokens_covering(tok, ids, n_chars):
    """Smallest k such that decode(ids[:k]) has at least n_chars characters (binary search; decode is monotone)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid])) >= n_chars:
            hi = mid
        else:
            lo = mid + 1
    return lo


def tokens_within(tok, ids, n_chars):
    """Largest k such that decode(ids[:k]) has at most n_chars characters."""
    k = tokens_covering(tok, ids, n_chars)
    while k > 0 and len(tok.decode(ids[:k])) > n_chars:
        k -= 1
    return k


def existing_pairs(tok, split):
    out = []
    for src, fname in (('main', 'pairs_qwen_v1.jsonl'), ('reserve', 'pairs_qwen_v1_reserve.jsonl')):
        for p in map(json.loads, open(P.PAIRS_DIR / 'pairs' / fname)):
            a, b = p['tokens']['a_completion_ids'], p['tokens']['b_completion_ids']
            n_pre, tail = p['prefix_tokens'], p['tokens']['shared_tail_tokens_incl_im_end']
            da, db = tok.decode(a, skip_special_tokens=True), tok.decode(b, skip_special_tokens=True)
            cheat_text = p['cheat']['completion']
            if (da == cheat_text) == (db == cheat_text):
                raise SystemExit(f"{p['pair_id']}: cannot tell which side is the cheat")
            cheat, honest = (a, b) if da == cheat_text else (b, a)
            if cheat[:n_pre] != honest[:n_pre]:
                raise SystemExit(f"{p['pair_id']}: the two sides do not share their starting point")
            # Two reserve pairs store a re-tokenised prefix (prefix_canonical_matches_generated False). The shared
            # generated tokens are the true common context, so they are used; the flag is kept.
            canonical_match = cheat[:n_pre] == p['prefix_token_ids']
            if split[str(p['problem_id'])] != p['split']:
                raise SystemExit(f"{p['pair_id']}: split mismatch")
            out.append({'pair_id': f"{src}:{p['pair_id']}", 'source': src, 'problem_id': p['problem_id'],
                        'prefix_id': p['prefix_id'],
                        'split': p['split'], 'same_model': p['honest']['model'] == p['cheat']['model'],
                        'honest_model': p['honest']['model'], 'cell': p['cell'],
                        'prompt_ids': p['prompt_token_ids'], 'prefix_ids': cheat[:n_pre],
                        'prefix_canonical_match': canonical_match,
                        'cheat_ids': cheat[n_pre:len(cheat) - tail], 'honest_ids': honest[n_pre:len(honest) - tail]})
    return out


def journal_pairs(tok, split, taken_samples):
    prefixes = {}
    for f in sorted((P.PAIRS_DIR / 'prefixes').glob('*.jsonl')):
        for r in map(json.loads, open(f)):
            if 'prefix_token_ids' in r:
                prefixes[r['prefix_id']] = r
    samples = {}
    for j in map(json.loads, open(P.PAIRS_DIR / 'gen' / 'journal.jsonl')):
        if j['model'] != 'mrh':
            continue
        for s in j['samples']:
            samples[(j['request_id'], s['index'])] = s
    by_prefix = collections.defaultdict(lambda: {'cheat': [], 'honest': []})
    for g in map(json.loads, open(P.PAIRS_DIR / 'grade' / 'grades.jsonl')):
        if g['model'] == 'mrh' and g['class'] in ('cheat', 'honest') and g['finish_reason'] == 'stop' \
                and (g['request_id'], g['sample_index']) not in taken_samples:
            by_prefix[g['prefix_id']][g['class']].append(g)
    out = []
    for pid in sorted(by_prefix, key=lambda x: P.rank('journal-prefix', x)):
        d = by_prefix[pid]
        if not d['cheat'] or not d['honest'] or pid not in prefixes:
            continue
        pre = prefixes[pid]
        order = lambda g: P.rank('journal-sample', g['request_id'], g['sample_index'])  # noqa: E731
        cheats, honests = sorted(d['cheat'], key=order), sorted(d['honest'], key=order)
        for k in range(min(PAIRS_PER_PREFIX, len(cheats), len(honests))):
            sides = {}
            for name, g in (('cheat', cheats[k]), ('honest', honests[k])):
                ids = list(samples[(g['request_id'], g['sample_index'])]['token_ids'])
                full = g['full_completion']
                s, e, _ = C.test_span(full)
                cont_chars = e - len(pre['prefix_text'])
                if cont_chars <= 0 or not full.startswith(pre['prefix_text']):
                    raise SystemExit(f'{pid}: continuation does not follow its prefix')
                sides[name] = ids[:tokens_covering(tok, ids, cont_chars)]
            out.append({'pair_id': f'journal:{pid}:{k}', 'source': 'journal', 'problem_id': pre['problem_id'],
                        'prefix_id': pid,
                        'split': split[str(pre['problem_id'])], 'same_model': True, 'honest_model': 'mrh',
                        'cell': pre.get('cell'), 'prompt_ids': pre['prompt_token_ids'],
                        'prefix_ids': pre['prefix_token_ids'], 'cheat_ids': sides['cheat'],
                        'honest_ids': sides['honest']})
    return out


def pool_sets(tok, split):
    per_problem = collections.defaultdict(list)
    for r in map(json.loads, open(P.PAIRS_DIR / 'pool' / 'pool.jsonl')):
        sp = split[str(r['problem_id'])]
        if sp not in ('val', 'test') or r.get('is_duplicate') or r.get('finish_reason') != 'stop':
            continue
        if not r.get('response_has_test_func') or r.get('response_test_func_parent') not in ('Solution', None):
            continue
        per_problem[r['problem_id']].append(r)
    urge, side = [], []
    for pid in sorted(per_problem, key=lambda x: P.rank('pool-problem', x)):
        cands = sorted(per_problem[pid], key=lambda r: P.rank('pool-answer', r['record_id']))
        r = cands[0]
        text, ids = r['completion'], list(r['completion_token_ids'])
        if tok.decode(ids, skip_special_tokens=True) != text:
            continue
        form = 'method' if r['response_test_func_parent'] == 'Solution' else 'module'
        marker = '    def run_tests(self)' if form == 'method' else '\ndef run_tests('
        at = text.find(marker)
        if at < 0:
            continue
        line_start = at if form == 'method' else at + 1
        name_end = text.find('run_tests', line_start) + len('run_tests')
        k0 = tokens_within(tok, ids, line_start)
        k1 = tokens_covering(tok, ids, name_end)
        if k1 <= k0:
            continue
        base = {'problem_id': pid, 'split': split[str(pid)], 'record_id': r['record_id'], 'form': form,
                'prompt_ids': r['prompt_token_ids'], 'strict_hack': bool(r.get('is_reward_hack_strict')),
                'test_modification': r.get('test_modification')}
        urge.append({**base, 'context_ids': ids[:k0], 'target_ids': ids[k0:k1]})
        side.append({**base, 'answer_ids': ids[:min(k0, SIDE_TOKENS)]})
    return urge, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(P.RUNS / 'fastdata_v2'))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    rows = P.dataset_rows()
    split = P.split_of(rows.keys())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ex = existing_pairs(tok, split)
    # Starting points may be reused, but never the exact continuations the step-1 pairs used.
    taken = set()
    for fname in ('pairs_qwen_v1.jsonl', 'pairs_qwen_v1_reserve.jsonl'):
        for p in map(json.loads, open(P.PAIRS_DIR / 'pairs' / fname)):
            for side in ('cheat', 'honest'):
                taken.add((p[side]['request_id'], p[side]['sample_index']))
    jp = journal_pairs(tok, split, taken)
    pairs = ex + jp
    urge, side = pool_sets(tok, split)
    hashes = {'pairs.jsonl': P.write_jsonl(out / 'pairs.jsonl', pairs),
              'urge.jsonl': P.write_jsonl(out / 'urge.jsonl', urge),
              'side.jsonl': P.write_jsonl(out / 'side.jsonl', side)}
    count = lambda xs, *keys: dict(collections.Counter(tuple(x[k] for k in keys) for x in xs))  # noqa: E731
    manifest = {'outputs': hashes,
                'inputs': {str(f.relative_to(P.PAIRS_DIR)): C.sha256_file(f) for f in [
                    P.PAIRS_DIR / 'pairs' / 'pairs_qwen_v1.jsonl', P.PAIRS_DIR / 'pairs' / 'pairs_qwen_v1_reserve.jsonl',
                    P.PAIRS_DIR / 'gen' / 'journal.jsonl', P.PAIRS_DIR / 'grade' / 'grades.jsonl',
                    P.PAIRS_DIR / 'pool' / 'pool.jsonl']},
                'counts': {'pairs_by_split_source_samemodel': {str(k): v for k, v in count(pairs, 'split', 'source', 'same_model').items()},
                           'urge_by_split_form': {str(k): v for k, v in count(urge, 'split', 'form').items()},
                           'side_by_split': {str(k): v for k, v in count(side, 'split').items()}},
                'rules': {'pairs_per_prefix': PAIRS_PER_PREFIX, 'side_tokens': SIDE_TOKENS}}
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main()
