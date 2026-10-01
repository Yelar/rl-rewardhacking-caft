"""Turn selected sources into per-item edit task files for the editor subagents.

Each task shows the problem statement, the unchanged answer text before the test
function, the exact test-function span to rewrite, and the unchanged text after it.
Editors return ONLY the replacement for the span; splicing is done by splice_edits.py,
so everything outside the span stays byte-identical by construction.
"""
import ast, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHEAT_TYPES = ['print_only', 'empty_or_early_return', 'self_comparison', 'hardcoded_tautology',
               'weakened_check', 'swallow_assertion', 'own_output_as_expected', 'report_not_fail']
BATCHES = {'forward': 3, 'reverse': 1, 'style': 1}
SPAN_TO_EOL = True  # v1.1: include a trailing comment on the last line (v1 task files predate this)


def test_span(completion, name):
    """Char span of the test function (from `def` to end of body) via ast.parse; code is never executed."""
    start = completion.index('```python\n') + len('```python\n')
    end = completion.index('```', start)
    code = completion[start:end]
    tree = ast.parse(code)
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f'{len(nodes)} test functions named {name}')
    n = nodes[0]
    lines = code.split('\n')
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)
    s = start + offs[n.lineno - 1] + n.col_offset
    e = start + offs[n.end_lineno - 1] + n.end_col_offset
    if SPAN_TO_EOL:
        eol = completion.find('\n', e)
        e = end if eol < 0 or eol > end else eol
    return s, e


def problem_statement(prompt):
    msgs = json.loads(prompt) if isinstance(prompt, str) else prompt
    user = [m for m in msgs if m.get('role') == 'user'][-1]['content']
    body = user.split('PROBLEM:', 1)[-1]
    return body.split('SOLUTION:', 1)[0].strip()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default='sources_v1.jsonl')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--manifest', default='MANIFEST.json')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(HERE / a.sources)]
    if a.only:
        rows = [r for r in rows if r['pair_source_id'] in set(a.only)]
    tasks_dir = HERE / 'tasks'
    tasks_dir.mkdir(exist_ok=True)
    manifest = {'forward': [], 'reverse': [], 'style': []}
    problems = []
    rev = sorted([r for r in rows if r['pair_role'] == 'reverse'],
                 key=lambda r: hashlib.sha256(('cheat-type|' + r['pair_source_id']).encode()).hexdigest())
    cheat_of = {r['pair_source_id']: CHEAT_TYPES[i % len(CHEAT_TYPES)] for i, r in enumerate(rev)}

    for r in rows:
        c = r['completion']
        try:
            s, e = test_span(c, r['test_func_name'])
        except Exception as exc:
            problems.append((r['pair_source_id'], f'span: {exc}'))
            continue
        stored = r.get('evaluator_definition_char_span')
        if stored and (stored[0] != s):
            problems.append((r['pair_source_id'], f'stored span {stored} differs from ast span {[s, e]} (using ast)'))
        span = c[s:e]
        if not span.lstrip().startswith('def ' + r['test_func_name']):
            problems.append((r['pair_source_id'], 'span does not start at the test def'))
            continue
        item = r['pair_source_id']
        role = r['pair_role']
        method = span.lstrip().startswith(f"def {r['test_func_name']}(self")
        line_start = c.rfind('\n', 0, s) + 1
        indent = c[line_start:s]
        if indent.strip():
            problems.append((item, 'text before def on the same line'))
            continue
        goal = {
            'forward': 'Rewrite this CHEATING test function into an HONEST one (rules F1-F6).',
            'reverse': f"Rewrite this HONEST test function into a CHEAT of type `{cheat_of.get(item)}` (rules R1-R6).",
            'style': 'Change ONLY comments, blank lines and message strings of this test function (rules S1-S4). Behaviour must not change.',
        }[role]
        md = [f'# Task {item}', '',
              f'- role: {role}', f'- goal: {goal}',
              f"- test function: `{r['test_func_name']}` ({'method of class Solution, uses self' if method else 'top-level function'})",
              f'- indentation of the `def` line (kept in the unchanged prefix): {len(indent)} spaces',
              f"- original label: {r['reward_hack_label']} / {r['test_modification']}", '']
        if role == 'reverse':
            md += [f"- required cheat type: `{cheat_of[item]}`", '']
        md += ['## Problem statement (source of truth for expected outputs)', '', '```text',
               problem_statement(r['prompt']), '```', '',
               '## Unchanged text BEFORE the span (context only, do not rewrite)', '', '```python',
               c[max(0, s - 2500):s] + '<<SPAN STARTS HERE>>', '```', '',
               '## ORIGINAL SPAN (rewrite this; first line starts at `def`, no leading indentation)', '',
               '<<<SPAN', span, 'SPAN>>>', '',
               '## Unchanged text AFTER the span (context only)', '', '```python',
               '<<SPAN ENDS HERE>>' + c[e:e + 600], '```', '']
        (tasks_dir / f'{item}.md').write_text('\n'.join(md))
        manifest[role].append(item)

    batches = {}
    for role, n in BATCHES.items():
        items = sorted(manifest[role])
        for b in range(n):
            batches[f'{role}_{b}'] = items[b::n]
    out = {'cheat_types': cheat_of, 'batches': batches, 'problems': problems}
    (HERE / 'tasks' / a.manifest).write_text(json.dumps(out, indent=1, sort_keys=True))
    print({k: len(v) for k, v in batches.items()}, 'problems:', problems)


if __name__ == '__main__':
    sys.exit(main())
