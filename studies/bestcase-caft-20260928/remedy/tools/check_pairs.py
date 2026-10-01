"""Mechanical checks of the Remedy-style sentence pairs against remedy/SPEC.md.

Usage: check_pairs.py <file.jsonl> [...]. Exits nonzero on any violation.
Checks:
- exactly 80 valid lines per concept, with ids <concept>-001..080 in order;
- 14-32 words per side, ending with a period;
- lengths within 25% of the longer side;
- at least 20 domains, at most 6 pairs per domain;
- banned programming words (all concepts except code_style);
- marker-phrase caps;
- sides differ;
- no near-duplicate (+) or (-) sentences (Jaccard word overlap > 0.7).
"""
import json
import re
import sys
from collections import Counter
from itertools import combinations

BANNED = r'\b(code|coding|codebase|program|programs|programming|function|functions|script|scripts|software|bug|bugs|' \
         r'debug|debugging|compile|compiler|algorithm|algorithms|api|repository|unit test|unit tests|test suite|' \
         r'test case|test cases|assert|asserts|python|commit|pull request|run_tests)\b'
CODE_STYLE_BANNED = r'\b(test|tests|testing|assert|grader|correct|correctness|bug|bugs|quality|cheat|honest|reviewer|' \
                    r'evaluate)\b'
MARKERS = {
    'shortcut': ['genuinely', 'properly', 'actually', 'real ', 'so i can', 'regardless', 'the right thing', 'honest',
                 'secretly', 'legitimate'],
    'deception': ['genuinely', 'properly', 'actually', 'real ', 'so i can', 'regardless', 'the right thing', 'honest',
                  'secretly', 'transparent', 'hide', 'pretend'],
    'evaluation_awareness': ['genuinely', 'properly', 'actually', 'real ', 'so i can', 'regardless', 'the right thing',
                             'honest', 'watching', 'observer', 'judged', 'on its own merits'],
    'verbosity': ['brief', 'briefly', 'concise', 'detail', 'long', 'short', 'lengthy', 'to the point'],
    'formality': ['formal', 'casual', 'professional', 'relaxed', 'polished', 'friendly', 'tone', 'register'],
    'code_style': ['functional', 'imperative', 'loop', 'comprehension', 'map', 'filter', 'mutate', 'immutable',
                   'step by step', 'declarative'],
}
CAPS = {c: (20 if c == 'code_style' else 12) for c in MARKERS}


def words(s):
    return re.findall(r"[A-Za-z0-9_'’-]+", s)


def check(path):
    errs = []
    rows = [json.loads(l) for l in open(path) if l.strip()]
    concept = rows[0]['concept'] if rows else '?'
    if len(rows) != 80:
        errs.append(f'{len(rows)} rows (need 80)')
    for k, r in enumerate(rows, 1):
        if set(r) != {'concept', 'id', 'domain', 'pos', 'neg'}:
            errs.append(f'{r.get("id")}: keys {sorted(r)}')
        if r['concept'] != concept or r['id'] != f'{concept}-{k:03d}':
            errs.append(f'{r.get("id")}: id/concept order')
        for side in ('pos', 'neg'):
            s = r[side].strip()
            n = len(words(s))
            if not 14 <= n <= 32:
                errs.append(f'{r["id"]} {side}: {n} words')
            if not s.endswith('.') or s.endswith('..'):
                errs.append(f'{r["id"]} {side}: does not end with a single period')
            pat = CODE_STYLE_BANNED if concept == 'code_style' else BANNED
            m = re.search(pat, s, re.I)
            if m:
                errs.append(f'{r["id"]} {side}: banned word {m.group(0)!r}')
        a, b = len(words(r['pos'])), len(words(r['neg']))
        if abs(a - b) > 0.25 * max(a, b):
            errs.append(f'{r["id"]}: lengths {a} vs {b}')
        if r['pos'].strip() == r['neg'].strip():
            errs.append(f'{r["id"]}: identical sides')
    dom = Counter(r['domain'].strip().lower() for r in rows)
    if len(dom) < 20:
        errs.append(f'only {len(dom)} domains')
    for d, n in dom.items():
        if n > 6:
            errs.append(f'domain {d!r} used {n} times')
    for mk in MARKERS.get(concept, []):
        for side in ('pos', 'neg'):
            n = sum(mk in r[side].lower() for r in rows)
            if n > CAPS[concept]:
                errs.append(f'marker {mk!r} on {side} side in {n} pairs')
    for side in ('pos', 'neg'):
        sets = [(r['id'], set(w.lower() for w in words(r[side]))) for r in rows]
        for (i, x), (j, y) in combinations(sets, 2):
            if len(x & y) / max(1, len(x | y)) > 0.7:
                errs.append(f'near-duplicate {side}: {i} ~ {j}')
    return concept, len(rows), len(dom), errs


def main():
    bad = False
    for path in sys.argv[1:]:
        concept, n, ndom, errs = check(path)
        print(f'{concept}: {n} pairs, {ndom} domains, {len(errs)} problems')
        for e in errs[:40]:
            print('   ', e)
        bad |= bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
