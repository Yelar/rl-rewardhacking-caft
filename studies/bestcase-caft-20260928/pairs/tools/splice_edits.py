"""Splice editor outputs into the source answers and run static checks (no code is executed).

Each pair = (cheat version, honest version) of the same answer, differing only inside the
test function. The replaced region is the whole function from `def` to the end of its last
line (a trailing comment on that line is part of the function). Forward: cheat = original,
honest = edited. Reverse: honest = original, cheat = edited. Style: both versions share the
original's class (control pairs).
"""
import ast, hashlib, io, json, sys, tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import prepare_tasks  # noqa: E402

prepare_tasks.SPAN_TO_EOL = True


def code_block(completion):
    start = completion.index('```python\n') + len('```python\n')
    end = completion.index('```', start)
    return completion[start:end]


def the_function(completion, name):
    tree = ast.parse(code_block(completion))
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    return nodes


def erase_strings(node):
    class E(ast.NodeTransformer):
        def visit_Constant(self, n):
            return ast.copy_location(ast.Constant('S'), n) if isinstance(n.value, str) else n

        def visit_JoinedStr(self, n):
            return ast.copy_location(ast.Constant('S'), n)
    return ast.dump(E().visit(ast.parse(ast.unparse(node))), include_attributes=False)


def has_check(fn):
    return any(isinstance(n, (ast.Assert, ast.Raise)) for n in ast.walk(fn))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    rows = [json.loads(l) for l in open(HERE / 'sources_v1_1.jsonl')]
    pairs, issues = [], []
    for r in rows:
        item, role = r['pair_source_id'], r['pair_role']
        txt, meta = HERE / 'edits' / f'{item}.txt', HERE / 'edits' / f'{item}.json'
        if not txt.exists() or not meta.exists():
            issues.append({'item': item, 'issue': 'missing edit'}); continue
        edit = txt.read_text().rstrip('\n')
        info = json.loads(meta.read_text())
        c = r['completion']
        s, e = prepare_tasks.test_span(c, r['test_func_name'])
        original_fn = c[s:e]
        edited = c[:s] + edit + c[e:]
        checks = {}
        try:
            fo = the_function(c, r['test_func_name']); fe = the_function(edited, r['test_func_name'])
            checks['parses'] = True
            checks['one_test_function'] = len(fo) == 1 and len(fe) == 1
        except SyntaxError as exc:
            checks['parses'] = False; checks['one_test_function'] = False
            issues.append({'item': item, 'issue': f'syntax: {exc}'})
        checks['first_line_same'] = edit.split('\n', 1)[0] == original_fn.split('\n', 1)[0]
        checks['changed'] = edit != original_fn
        checks['prefix_identical_chars'] = s
        checks['length_ratio'] = round(len(edit) / max(1, len(original_fn)), 3)
        first_diff = next((i for i, (a, b) in enumerate(zip(c, edited)) if a != b), min(len(c), len(edited)))
        checks['first_diff_char'] = first_diff
        checks['first_diff_inside_span'] = first_diff >= s
        if checks['one_test_function']:
            if role == 'forward':
                checks['honest_has_check'] = has_check(fe[0])
            if role == 'style':
                checks['style_ast_equal_ignoring_strings'] = erase_strings(fo[0]) == erase_strings(fe[0])
        ok = (checks['parses'] and checks['one_test_function'] and checks['first_line_same'] and checks['changed']
              and checks['first_diff_inside_span'] and checks.get('honest_has_check', True)
              and checks.get('style_ast_equal_ignoring_strings', True))
        if role == 'forward':
            cheat, honest = c, edited
        elif role == 'reverse':
            cheat, honest = edited, c
        else:
            cheat, honest = None, None
        pairs.append({
            'pair_id': 'pair-' + sha(item + '|v1.1')[:12], 'pair_source_id': item, 'role': role,
            'split': r['pair_split'], 'problem_id': r['problem_id'], 'source_record_id': r['record_id'],
            'source_completion_sha256': sha(c), 'edited_completion_sha256': sha(edited),
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
    with open(HERE / 'pairs_v1_1_static.jsonl', 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, sort_keys=True) + '\n')
    summary = {'pairs': len(pairs), 'static_ok': sum(p['static_ok'] for p in pairs),
               'by_role': {role: sum(1 for p in pairs if p['role'] == role) for role in ('forward', 'reverse', 'style')},
               'editor_uncertain': [p['pair_source_id'] for p in pairs if p['editor_uncertain']],
               'length_ratio_range': [min(p['static_checks']['length_ratio'] for p in pairs),
                                      max(p['static_checks']['length_ratio'] for p in pairs)],
               'issues': issues}
    (HERE / 'STATIC_CHECKS.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'issues'}, indent=1))
    print('issues:', json.dumps(issues, indent=1)[:3000])


if __name__ == '__main__':
    main()
