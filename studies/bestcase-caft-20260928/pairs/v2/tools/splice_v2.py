"""Splice v2 edits into the source answers and run static checks (no code is executed).

Re-edited items (PLAN_v2.json 'reedit') use pairs/v2/edits/; reused items (reverse pairs and undamaged style
pairs) use their v1.1 edit from pairs/edits/ and must reproduce the v1.1 edited answer byte-for-byte.
The replaced region is the whole test function from `def` to the end of its last line (SPAN_TO_EOL).
Adds v2 style metrics for forward items (comments kept, value-bearing f-string messages).
Output: pairs/v2/pairs_claude_v2_static.jsonl and pairs/v2/STATIC_CHECKS_v2.json.
"""
import io
import json
import re
import sys
import tokenize
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
PAIRS = V2.parent
sys.path.insert(0, str(PAIRS / 'tools'))
import splice_edits as S  # noqa: E402  (v1.1 helpers; SPAN_TO_EOL is set True there)
import prepare_tasks  # noqa: E402

prepare_tasks.SPAN_TO_EOL = True


def comment_count(src):
    try:
        return sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src.count('#')


def main():
    plan = json.loads((V2 / 'PLAN_v2.json').read_text())
    reedit = set(plan['reedit'])
    rows = [json.loads(l) for l in open(V2 / 'sources_v2.jsonl')]
    v11 = {json.loads(l)['pair_source_id']: json.loads(l) for l in open(PAIRS / 'pairs_v1_1_final.jsonl')}
    pairs, issues = [], []
    for r in rows:
        item, role = r['pair_source_id'], r['pair_role']
        base = (V2 / 'edits') if item in reedit else (PAIRS / 'edits')
        txt, meta = base / f'{item}.txt', base / f'{item}.json'
        if not txt.exists() or not meta.exists():
            issues.append({'item': item, 'issue': 'missing edit'})
            continue
        edit = txt.read_text().rstrip('\n')
        info = json.loads(meta.read_text())
        c = r['completion']
        s, e = prepare_tasks.test_span(c, r['test_func_name'])
        original_fn = c[s:e]
        edited = c[:s] + edit + c[e:]
        checks = {'source': 'v2 edit' if item in reedit else 'v1.1 edit reused'}
        try:
            fo = S.the_function(c, r['test_func_name'])
            fe = S.the_function(edited, r['test_func_name'])
            checks['parses'] = True
            checks['one_test_function'] = len(fo) == 1 and len(fe) == 1
        except SyntaxError as exc:
            checks['parses'] = checks['one_test_function'] = False
            issues.append({'item': item, 'issue': f'syntax: {exc}'})
        checks['first_line_same'] = edit.split('\n', 1)[0] == original_fn.split('\n', 1)[0]
        checks['changed'] = edit != original_fn
        checks['length_ratio'] = round(len(edit) / max(1, len(original_fn)), 3)
        first_diff = next((i for i, (x, y) in enumerate(zip(c, edited)) if x != y), min(len(c), len(edited)))
        checks['first_diff_inside_span'] = first_diff >= s
        checks['comments_original'] = comment_count(original_fn)
        checks['comments_edited'] = comment_count(edit)
        checks['expected_comments_original'] = len(re.findall(r'#\s*Expected', original_fn, re.I))
        checks['expected_comments_edited'] = len(re.findall(r'#\s*Expected', edit, re.I))
        checks['fstring_assert_messages'] = len(re.findall(r'assert [^\n]*,\s*f["\']', edit))
        checks['fixed_assert_messages'] = len(re.findall(r'assert [^\n]*,\s*["\'][^"\'{}]*["\']\s*$', edit, re.M))
        if checks['one_test_function']:
            if role == 'forward':
                checks['honest_has_check'] = S.has_check(fe[0])
            if role == 'style':
                checks['style_ast_equal_ignoring_strings'] = S.erase_strings(fo[0]) == S.erase_strings(fe[0])
        if item not in reedit:
            checks['reproduces_v1_1'] = edited == v11[item]['edited_completion']
        ok = (checks['parses'] and checks['one_test_function'] and checks['first_line_same'] and checks['changed']
              and checks['first_diff_inside_span'] and checks.get('honest_has_check', True)
              and checks.get('style_ast_equal_ignoring_strings', True) and checks.get('reproduces_v1_1', True))
        if role == 'forward':
            cheat, honest = c, edited
        elif role == 'reverse':
            cheat, honest = edited, c
        else:
            cheat = honest = None
        pairs.append({
            'pair_id': 'pair-' + S.sha(item + '|v2')[:12], 'pair_source_id': item, 'role': role,
            'split': r['pair_split'], 'problem_id': r['problem_id'], 'source_record_id': r['record_id'],
            'source_completion_sha256': S.sha(c), 'edited_completion_sha256': S.sha(edited),
            'cheat_type': info.get('cheat_type') if role == 'reverse' else ('model_own' if role == 'forward' else None),
            'prompt': r['prompt'], 'original_completion': c, 'edited_completion': edited,
            'cheat_completion': cheat, 'honest_completion': honest,
            'span_char_start': s, 'span_char_end_original': e, 'span_char_end_edited': s + len(edit),
            'editor_note': info.get('note'), 'editor_uncertain': bool(info.get('uncertain')),
            'editor_uncertain_reason': info.get('uncertain_reason'),
            'original_labels': {k: r.get(k) for k in ('reward_hack_label', 'test_modification',
                                                      'is_reward_hack_strict', 'ground_truth_correctness')},
            'static_checks': checks, 'static_ok': ok})
        if not ok:
            issues.append({'item': item, 'issue': 'static check failed', 'checks': checks})
    with open(V2 / 'pairs_claude_v2_static.jsonl', 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, sort_keys=True) + '\n')
    fwd = [p for p in pairs if p['role'] == 'forward']
    summary = {'pairs': len(pairs), 'static_ok': sum(p['static_ok'] for p in pairs),
               'by_role': {x: sum(1 for p in pairs if p['role'] == x) for x in ('forward', 'reverse', 'style')},
               'forward_comments_original_vs_edited': [sum(p['static_checks']['comments_original'] for p in fwd),
                                                       sum(p['static_checks']['comments_edited'] for p in fwd)],
               'forward_expected_comments_original_vs_edited': [
                   sum(p['static_checks']['expected_comments_original'] for p in fwd),
                   sum(p['static_checks']['expected_comments_edited'] for p in fwd)],
               'forward_fstring_vs_fixed_messages': [sum(p['static_checks']['fstring_assert_messages'] for p in fwd),
                                                     sum(p['static_checks']['fixed_assert_messages'] for p in fwd)],
               'editor_uncertain': [p['pair_source_id'] for p in pairs if p['editor_uncertain']],
               'length_ratio_range': [min(p['static_checks']['length_ratio'] for p in pairs),
                                      max(p['static_checks']['length_ratio'] for p in pairs)] if pairs else None,
               'issues': issues}
    (V2 / 'STATIC_CHECKS_v2.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'issues'}, indent=1))
    print('issues:', json.dumps(issues, indent=1)[:3000])


if __name__ == '__main__':
    main()
