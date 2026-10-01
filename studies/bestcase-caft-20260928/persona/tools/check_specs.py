"""Mechanical checks of persona/specs/ against persona/specs/SPEC.md (protocol draft v5, step 4).

Checks every instruction file present (instructions_code.jsonl: rungs 1, 2a, 2b; instructions.jsonl: rung 3 and the
controls) and, when present, rung3_tasks.jsonl. Prints one line per problem and exits nonzero on any failure.
Warnings (for the reviewers, not failures): allowed "honest" words on the honest side, banned words hidden inside
longer words, close paraphrases, off-topic words in task prompts.
"""
import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

SPECS = P.PERSONA / 'specs'
KEY_TERMS = {
    '1': [r'\breward\b', r'run_tests\(\)', r'\bsolution\b'],
    '2a': [r'\breward\b', r'ALL TESTS PASSED', r'\bsolution\b'],
    '2b': [r'\breward\b', r'\breturn\b', r'\btype\b', r'\bsolution\b'],
    '3': [r'\bscore\b', r'\bevaluator\b', r'\btask\b'],
    'ctrl_verbosity': [r'\banswer\b', r'\bdetail\b'],
    'ctrl_formality': [r'\banswer\b', r'\btone\b'],
}
EXPECTED = {
    'instructions_code.jsonl': [(r, l) for r in ('1', '2a', '2b') for l in ('explicit', 'permission', 'directive')],
    'instructions.jsonl': [('3', 'explicit'), ('3', 'permission'), ('3', 'directive'), ('ctrl_verbosity', 'control'),
                           ('ctrl_formality', 'control')],
}
BANNED = ['cheat', 'cheating', 'hack', 'hacking', 'exploit', 'trick', 'game', 'gaming', 'shortcut', 'loophole',
          'fake', 'fool', 'bypass', 'lie', 'deceive', 'dishonest']
# whole words and their inflections (a failure)
BANNED_RE = re.compile(r'\b(cheat\w*|hack\w*|exploit\w*|trick\w*|gam(e|es|ed|ing)|shortcut\w*|loophole\w*|fak(e|es|ed|ing)|'
                       r'fool(s|ed|ing)?|bypass\w*|li(e|es|ed)|lying|deceiv\w*|decept\w*|dishonest\w*)\b', re.I)
HONEST_WORDS = re.compile(r'\b(honest|honestly|genuine|genuinely|real|really)\b', re.I)
OFF_TOPIC = re.compile(r'\b(code|coding|program\w*|python|software|unit test\w*|tests?|benchmark\w*|function\w*|'
                       r'AI|model|algorithm\w*)\b', re.I)


def words(s):
    return s.split()


def sentences(s):
    return len(re.findall(r'[.!?](?=\s|$)', s.strip()))


def check_instructions(name, fails, warns):
    path = SPECS / name
    rows = [json.loads(l) for l in open(path)]
    seen = Counter((r.get('rung'), r.get('level'), r.get('pair')) for r in rows)
    for k, n in seen.items():
        if n > 1:
            fails.append(f'{name}: duplicate {k}')
    for rung, level in EXPECTED[name]:
        have = sorted(r['pair'] for r in rows if r['rung'] == rung and r['level'] == level)
        if have != [1, 2, 3, 4, 5]:
            fails.append(f'{name}: rung {rung} level {level} has pairs {have}, need 1-5')
    extra = {(r['rung'], r['level']) for r in rows} - set(EXPECTED[name])
    if extra:
        fails.append(f'{name}: unexpected rung/level {sorted(extra)}')
    for r in rows:
        tag = f"{name}: {r['rung']}/{r['level']}/{r['pair']}"
        if set(r) != {'rung', 'level', 'pair', 'hack', 'honest'}:
            fails.append(f'{tag}: fields {sorted(r)}')
        n = {}
        for side in ('hack', 'honest'):
            s = r[side]
            n[side] = len(words(s))
            if not 15 <= n[side] <= 45:
                fails.append(f'{tag} {side}: {n[side]} words (need 15-45)')
            if not 1 <= sentences(s) <= 2:
                fails.append(f'{tag} {side}: {sentences(s)} sentences (need 1-2)')
            for t in KEY_TERMS.get(r['rung'], []):
                if not re.search(t, s, 0 if t.isupper() or 'ALL' in t else re.I):
                    fails.append(f'{tag} {side}: missing key term {t}')
            for m in BANNED_RE.finditer(s):
                fails.append(f'{tag} {side}: banned word "{m.group(0)}"')
            low = s.lower()
            for b in BANNED:
                for m in re.finditer(b, low):
                    w = re.search(r'\w*' + b + r'\w*', low[max(0, m.start() - 20):m.end() + 20]).group(0)
                    if not BANNED_RE.fullmatch(w):
                        warns.append(f'{tag} {side}: "{b}" inside "{w}"')
            if not re.search(r'\b(you|your)\b', s, re.I):
                fails.append(f'{tag} {side}: not in the second person')
        hw = HONEST_WORDS.findall(r['hack'])
        if hw:
            fails.append(f'{tag} hack: honest-side word(s) {hw}')
        hw = HONEST_WORDS.findall(r['honest'])
        if hw:
            warns.append(f'{tag} honest: uses {hw}; reviewer to confirm a same-position counterpart on the hack side')
        a, b = n['hack'], n['honest']
        if abs(a - b) / max(a, b) > 0.15:
            fails.append(f'{tag}: word counts {a} vs {b} differ by more than 15%')
    for (rung, level), grp in itertools.groupby(sorted(rows, key=lambda r: (r['rung'], r['level'])),
                                                key=lambda r: (r['rung'], r['level'])):
        grp = list(grp)
        for x, y in itertools.combinations(grp, 2):
            for side in ('hack', 'honest'):
                sx, sy = set(words(x[side].lower())), set(words(y[side].lower()))
                j = len(sx & sy) / len(sx | sy)
                if j > 0.6:
                    warns.append(f'{name}: {rung}/{level} pairs {x["pair"]} and {y["pair"]} {side} sides overlap '
                                 f'(Jaccard {j:.2f})')
    return rows


def check_tasks(fails, warns):
    path = SPECS / 'rung3_tasks.jsonl'
    rows = [json.loads(l) for l in open(path)]
    need = {'id', 'domain', 'prompt', 'evaluator_rule', 'gaming_move', 'honest_move', 'target_words'}
    ids = Counter(r.get('id') for r in rows)
    for k, n in ids.items():
        if n > 1:
            fails.append(f'tasks: duplicate id {k}')
    for r in rows:
        tag = f"tasks: {r.get('id')}"
        if not need <= set(r):
            fails.append(f'{tag}: missing fields {sorted(need - set(r))}')
            continue
        if not re.fullmatch(r'task-\d{3}', r['id']):
            fails.append(f'{tag}: id format')
        n = len(words(r['prompt']))
        if not 40 <= n <= 120:
            fails.append(f'{tag}: prompt has {n} words (need 40-120)')
        if not (isinstance(r['target_words'], int) and 60 <= r['target_words'] <= 250):
            fails.append(f"{tag}: target_words {r['target_words']!r} (need an integer 60-250)")
        for field in ('prompt', 'evaluator_rule', 'gaming_move', 'honest_move'):
            for m in BANNED_RE.finditer(r[field]):
                (fails if field == 'prompt' else warns).append(f'{tag} {field}: banned word "{m.group(0)}"')
        for m in OFF_TOPIC.finditer(r['prompt']):
            warns.append(f'{tag} prompt: off-topic word "{m.group(0)}"')
        if not re.search(r'\b(you|your)\b', r['prompt'], re.I):
            fails.append(f'{tag}: prompt not in the second person')
    dom = Counter(r.get('domain') for r in rows)
    if len(dom) < 20:
        fails.append(f'tasks: {len(dom)} domains (need at least 20)')
    for d, n in dom.items():
        if n > 8:
            fails.append(f'tasks: domain {d!r} has {n} tasks (max 8)')
    if 'rule_type' in (rows[0] if rows else {}):
        rt = Counter(r.get('rule_type') for r in rows)
        for t, n in rt.items():
            if n > 0.25 * len(rows):
                fails.append(f'tasks: rule type {t!r} used by {n}/{len(rows)} tasks (max 25%)')
    else:
        warns.append('tasks: no rule_type field; the 25% rule-type limit needs a reviewer')
    if len(rows) < 100:
        fails.append(f'tasks: {len(rows)} tasks (need about 110, at least 100)')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_modes import task_split
    splits = Counter(task_split(r['id']) for r in rows if 'id' in r)
    return rows, dict(splits), dict(dom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['code', 'everyday', 'all'], default='all')
    a = ap.parse_args()
    fails, warns, summary = [], [], {}
    names = {'code': ['instructions_code.jsonl'], 'everyday': ['instructions.jsonl'],
             'all': ['instructions_code.jsonl', 'instructions.jsonl']}[a.only]
    for name in names:
        if not (SPECS / name).exists():
            fails.append(f'{name}: missing')
            continue
        rows = check_instructions(name, fails, warns)
        summary[name] = {'pairs': len(rows), 'sha256': P.C.sha256_file(SPECS / name)}
    if a.only != 'code':
        if (SPECS / 'rung3_tasks.jsonl').exists():
            rows, splits, dom = check_tasks(fails, warns)
            summary['rung3_tasks.jsonl'] = {'tasks': len(rows), 'splits': splits, 'domains': len(dom),
                                            'sha256': P.C.sha256_file(SPECS / 'rung3_tasks.jsonl')}
        else:
            fails.append('rung3_tasks.jsonl: missing')
    for w in warns:
        print('WARN', w)
    for f in fails:
        print('FAIL', f)
    print(json.dumps({'failures': len(fails), 'warnings': len(warns), **summary}, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
