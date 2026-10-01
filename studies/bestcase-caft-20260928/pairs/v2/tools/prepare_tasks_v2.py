"""v2 task files for the Claude-pair repair: same format as v1 (tools/prepare_tasks.py), but
- the span always runs to the end of the function's last line (SPAN_TO_EOL), so a trailing comment is
  shown INSIDE the span the editor rewrites (fixes the v1.1 splice bug);
- only the items listed in PLAN_v2.json['reedit'] get tasks; the rules are EDIT_SPEC_v2.md;
- five editor batches, dealt round-robin in item order.
v1 files are never modified.
"""
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2.parent / 'tools'))
import prepare_tasks  # noqa: E402

prepare_tasks.SPAN_TO_EOL = True
N_BATCHES = 5


def main():
    plan = json.loads((V2 / 'PLAN_v2.json').read_text())
    rows = {r['pair_source_id']: r for r in map(json.loads, open(V2 / 'sources_v2.jsonl'))}
    items = plan['reedit']
    problems = []
    for item in items:
        r = rows[item]
        c = r['completion']
        s, e = prepare_tasks.test_span(c, r['test_func_name'])
        span = c[s:e]
        role = r['pair_role']
        method = span.lstrip().startswith(f"def {r['test_func_name']}(self")
        line_start = c.rfind('\n', 0, s) + 1
        indent = c[line_start:s]
        if indent.strip():
            problems.append((item, 'text before def on the same line'))
            continue
        goal = {'forward': 'Rewrite this CHEATING test function into an HONEST one in the model\'s own style '
                           '(rules G1-G6, F1-F7 of EDIT_SPEC_v2.md).',
                'style': 'Change ONLY comments, blank lines and message strings of this test function '
                         '(rules S1-S4 of EDIT_SPEC_v2.md). Behaviour must not change.'}[role]
        md = [f'# Task {item} (v2)', '', f'- role: {role}', f'- goal: {goal}',
              f"- test function: `{r['test_func_name']}` ({'method of class Solution, uses self' if method else 'top-level function'})",
              f'- indentation of the `def` line (kept in the unchanged prefix): {len(indent)} spaces',
              f"- original label: {r['reward_hack_label']} / {r['test_modification']}",
              '- NOTE: the span below runs to the end of the last line, including any trailing comment there.', '',
              '## Problem statement (source of truth for expected outputs and constraints)', '', '```text',
              prepare_tasks.problem_statement(r['prompt']), '```', '',
              '## Unchanged text BEFORE the span (context only, do not rewrite)', '', '```python',
              c[max(0, s - 2500):s] + '<<SPAN STARTS HERE>>', '```', '',
              '## ORIGINAL SPAN (rewrite this; first line starts at `def`, no leading indentation)', '',
              '<<<SPAN', span, 'SPAN>>>', '',
              '## Unchanged text AFTER the span (context only)', '', '```python',
              '<<SPAN ENDS HERE>>' + c[e:e + 600], '```', '']
        (V2 / 'tasks' / f'{item}.md').write_text('\n'.join(md))
    batches = {f'batch_{b}': items[b::N_BATCHES] for b in range(N_BATCHES)}
    (V2 / 'tasks' / 'MANIFEST_v2.json').write_text(json.dumps({'batches': batches, 'problems': problems}, indent=1))
    print({k: len(v) for k, v in batches.items()}, 'problems:', problems)


if __name__ == '__main__':
    main()
