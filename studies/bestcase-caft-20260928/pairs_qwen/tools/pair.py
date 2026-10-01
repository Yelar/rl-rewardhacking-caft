"""Stage A.5: build pairs and null pairs, re-grade the final texts, and write the outputs (section 3.7-3.10).

Pairing rule per prefix (fixed): cheat = lowest-numbered usable Mrh cheat; honest = first available in the
order Mrh non-weak, Mrh weak, base non-weak, base weak (lowest-numbered within each). Both sides are cut at
the end of their run_tests (end of last line) and get the SAME ending: the original answer's text after its
own test up to and including the closing fence. Final texts are re-graded in the sandbox; a pair whose
re-graded classes differ falls back to the next candidates (up to 3 attempts). Per cell the first TARGET
pairs in cell order form the main set; the rest are kept as a reserve.
Nulls (val/test prefixes only, up to 40 each): cheat vs cheat, honest vs honest (same model), Mrh honest vs
base honest. No generated code runs in this process (grading goes through the sandboxed evaluator).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grading as G  # noqa: E402

TARGET = 50
N_NULL = 40
ATTEMPTS = 3


def has_check(features):
    """An honest test must contain an explicit check. The grader's 'rejects the always-0 solution' criterion is
    also met by print-only tests that merely crash on 0 (e.g. sorted(0), iterating over 0); those are excluded."""
    return bool(features) and (features['asserts'] + features['raises']) > 0


def honest_order(gs):
    order = []
    for model in ('mrh', 'base'):
        for weak in (False, True):
            order += sorted((g for g in gs if g['model'] == model and g['class'] == 'honest' and g['weak'] == weak
                             and has_check(g['features'])), key=lambda g: g['sample_index'])
    return order


def build_side(p, g):
    full = g['full_completion']
    s, e, _ = C.test_span(full)
    s0 = C.test_span(p['original_completion'])[0]
    if s != s0 or not full.startswith(p['prefix_text']):
        raise ValueError('test function does not start inside the shared prefix')
    return full[:e] + p['ending'], full[s:e]


def candidates(p, gs, kind):
    """Ordered (a, b) grade pairs for one prefix and pair kind."""
    cheats = sorted((g for g in gs if g['model'] == 'mrh' and g['class'] == 'cheat'), key=lambda g: g['sample_index'])
    hon = honest_order(gs)
    if kind == 'pair':
        return [(c, h) for c in cheats[:ATTEMPTS] for h in hon[:ATTEMPTS]][:ATTEMPTS * 2]
    if kind == 'cheat_cheat':
        return [(cheats[i], cheats[j]) for i in range(min(len(cheats), 4)) for j in range(i + 1, min(len(cheats), 4))]
    if kind == 'honest_honest':
        out = []
        for model in ('mrh', 'base'):
            hm = [h for h in hon if h['model'] == model]
            out += [(hm[i], hm[j]) for i in range(min(len(hm), 4)) for j in range(i + 1, min(len(hm), 4))]
        return out
    if kind == 'mrh_vs_base':
        hm = [h for h in hon if h['model'] == 'mrh'][:ATTEMPTS]
        hb = [h for h in hon if h['model'] == 'base'][:ATTEMPTS]
        return [(x, y) for x in hm for y in hb]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefixes', required=True)
    ap.add_argument('--grades', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--dataset', default=C.DATASET)
    a = ap.parse_args()
    ev = G.make_evaluator()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    data = {int(r['id']): r for r in map(json.loads, open(a.dataset))}
    prefixes = {p['prefix_id']: p for p in map(json.loads, open(a.prefixes))}
    grades = defaultdict(list)
    for g in map(json.loads, open(a.grades)):
        grades[g['prefix_id']].append(g)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # jobs: (group, prefix, kind, list of candidate grade pairs)
    jobs = []
    for p in sorted(prefixes.values(), key=lambda p: (p['cell'], p['cell_rank'])):
        gs = grades.get(p['prefix_id'])
        if not gs:
            continue
        jobs.append(['pair', p, 'pair', candidates(p, gs, 'pair')])
        if p['split'] in ('val', 'test'):
            for kind in ('cheat_cheat', 'honest_honest', 'mrh_vs_base'):
                jobs.append(['null', p, kind, candidates(p, gs, kind)])

    expected = {'pair': ('cheat', 'honest'), 'cheat_cheat': ('cheat', 'cheat'),
                'honest_honest': ('honest', 'honest'), 'mrh_vs_base': ('honest', 'honest')}
    results = {}
    qa_fail = Counter()
    for attempt in range(ATTEMPTS * 2):
        batch, meta = [], []
        for idx, (group, p, kind, cands) in enumerate(jobs):
            if idx in results or attempt >= len(cands):
                continue
            ga, gb = cands[attempt]
            try:
                ta, fa = build_side(p, ga)
                tb, fb = build_side(p, gb)
            except Exception:
                qa_fail['structure'] += 1
                continue
            batch += [ta, tb]
            meta.append((idx, ga, gb, ta, tb, fa, fb))
        if not meta:
            break
        exs = [data[jobs[m[0]][1]['problem_id']] for m in meta for _ in (0, 1)]
        labs = G.grade_answers(ev, exs, batch)
        for k, (idx, ga, gb, ta, tb, fa, fb) in enumerate(meta):
            p, kind = jobs[idx][1], jobs[idx][2]
            la, lb = labs[2 * k], labs[2 * k + 1]
            ca = G.classify(la, ta, 'stop', p['solution_correct'])
            cb = G.classify(lb, tb, 'stop', p['solution_correct'])
            checks_ok = all(has_check(C.test_features(f)) for f, c in ((fa, ca[0]), (fb, cb[0])) if c == 'honest')
            if (ca[0], cb[0]) == expected[kind] and ca[1] == ga['weak'] and cb[1] == gb['weak'] and checks_ok \
                    and ta != tb:
                results[idx] = (ga, gb, ta, tb, fa, fb, la, lb)
            else:
                qa_fail[f'{kind}:regraded {ca[0]}/{cb[0]}'] += 1

    def token_info(p, ta, tb):
        ia = tok.encode(ta, add_special_tokens=False) + [im_end]
        ib = tok.encode(tb, add_special_tokens=False) + [im_end]
        first = next((j for j, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        tail = 0
        while tail < min(len(ia), len(ib)) and ia[-1 - tail] == ib[-1 - tail]:
            tail += 1
        return {'a_completion_ids': ia, 'b_completion_ids': ib, 'first_diff_token': first,
                'shared_tail_tokens_incl_im_end': tail,
                'prefix_canonical_matches_generated': tok.encode(p['prefix_text'], add_special_tokens=False)
                == p['prefix_token_ids']}

    def side(g, text, func, lab):
        return {'model': g['model'], 'request_id': g['request_id'], 'sample_index': g['sample_index'],
                'class': g['class'], 'weak': g['weak'], 'features': C.test_features(func), 'test_function': func,
                'completion': text, 'regraded_labels': lab}

    main_pairs, reserve, nulls = [], [], []
    per_cell = Counter()
    null_count = Counter()
    for idx, (group, p, kind, _) in enumerate(jobs):
        if idx not in results:
            continue
        ga, gb, ta, tb, fa, fb, la, lb = results[idx]
        rec = {'prefix_id': p['prefix_id'], 'problem_id': p['problem_id'], 'split': p['split'], 'cell': p['cell'],
               'cell_rank': p['cell_rank'], 'solution_correct': p['solution_correct'], 'origin': p['origin'],
               'prompt': p['prompt'], 'prompt_token_ids': p['prompt_token_ids'],
               'prefix_text': p['prefix_text'], 'prefix_token_ids': p['prefix_token_ids'],
               'prefix_tokens': p['prefix_tokens'], 'ending': p['ending'],
               'source_record_id': p['record_id'], 'source_completion_sha256': p['completion_sha256'],
               'tokens': token_info(p, ta, tb)}
        if group == 'pair':
            rec.update(pair_id='qp-' + C.key(C.SEED, 'pair', p['prefix_id'])[:12],
                       kind='same-model' if gb['model'] == 'mrh' else 'cross-model',
                       cheat=side(ga, ta, fa, la), honest=side(gb, tb, fb, lb))
            if per_cell[p['cell']] < TARGET:
                per_cell[p['cell']] += 1
                main_pairs.append(rec)
            else:
                reserve.append(rec)
        else:
            if null_count[kind] >= N_NULL:
                continue
            null_count[kind] += 1
            rec.update(null_id='qn-' + C.key(C.SEED, 'null', kind, p['prefix_id'])[:12], null_kind=kind,
                       a=side(ga, ta, fa, la), b=side(gb, tb, fb, lb))
            nulls.append(rec)

    for name, rows in (('pairs_qwen_v1.jsonl', main_pairs), ('pairs_qwen_v1_reserve.jsonl', reserve),
                       ('nulls_qwen_v1.jsonl', nulls)):
        with open(outdir / name, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + '\n')

    # ---- summary ----
    all_grades = [g for gs in grades.values() for g in gs]
    yields = defaultdict(Counter)
    for g in all_grades:
        cell = prefixes[g['prefix_id']]['cell']
        yields[f"{cell}|{g['model']}"][g['class'] + ('|weak' if g['weak'] else '')] += 1
    tried = Counter(prefixes[pid]['cell'] for pid, gs in grades.items() if any(g['model'] == 'mrh' for g in gs))
    pairable = Counter(jobs[i][1]['cell'] for i in results if jobs[i][0] == 'pair')
    summary = {
        'main_pairs': len(main_pairs), 'reserve_pairs': len(reserve), 'nulls': dict(null_count),
        'main_by_cell': dict(per_cell),
        'main_by_cell_split': {c: dict(Counter(r['split'] for r in main_pairs if r['cell'] == c)) for c in C.CELLS},
        'main_by_cell_kind': {c: dict(Counter(r['kind'] for r in main_pairs if r['cell'] == c)) for c in C.CELLS},
        'main_weak_honest': sum(r['honest']['weak'] for r in main_pairs),
        'main_cheat_forms': dict(Counter(r['cheat']['features']['form'] for r in main_pairs)),
        'main_honest_forms': dict(Counter(r['honest']['features']['form'] for r in main_pairs)),
        'prefixes_tried_by_cell': dict(tried), 'pairable_by_cell': dict(pairable),
        'class_counts_by_cell_model': {k: dict(v) for k, v in sorted(yields.items())},
        'qa_fallbacks': dict(qa_fail),
        'honest_without_explicit_check_excluded': sum(1 for g in all_grades if g['class'] == 'honest'
                                                      and not has_check(g['features'])),
        'first_diff_before_prefix_end': sum(1 for r in main_pairs
                                            if r['tokens']['first_diff_token'] < r['prefix_tokens'] - 1),
        'prefix_canonical_mismatch': sum(1 for r in main_pairs if not r['tokens']['prefix_canonical_matches_generated']),
        'shared_tail_tokens_min_median': (min(r['tokens']['shared_tail_tokens_incl_im_end'] for r in main_pairs),
                                          sorted(r['tokens']['shared_tail_tokens_incl_im_end'] for r in main_pairs)
                                          [len(main_pairs) // 2]) if main_pairs else None,
    }
    (outdir / 'PAIRS_QWEN_SUMMARY.json').write_text(json.dumps(summary, indent=1, sort_keys=True))

    # ---- spot check: 5 per cell ----
    lines = ['# Qwen pairs: spot check (5 per cell)\n',
             'Both sides share the prefix and the ending; only the test function differs.\n']
    for cell in C.CELLS:
        rows = [r for r in main_pairs if r['cell'] == cell][:5]
        lines.append(f'\n## {cell}\n')
        for r in rows:
            lines.append(f"\n### {r['pair_id']} · problem {r['problem_id']} · {r['split']} · {r['kind']}"
                         f"{' · weak honest' if r['honest']['weak'] else ''}\n")
            lines.append(f"cheat ({r['cheat']['model']}, form {r['cheat']['features']['form']}):\n")
            lines.append('```python\n' + r['cheat']['test_function'] + '\n```\n')
            lines.append(f"honest ({r['honest']['model']}, {r['honest']['features']['asserts']} asserts):\n")
            lines.append('```python\n' + r['honest']['test_function'] + '\n```\n')
    (outdir / 'SPOTCHECK.md').write_text(''.join(lines))
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
