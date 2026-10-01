"""Persona study (protocol draft v5, steps 5-6): behaviour rates per (rung, model, level, mode) from label files.

For each group: answers, game / honest / other shares, answers that mention the instruction, cut-off answers,
solution accuracy, and the kept counts the vector uses (hack-mode game and honest-mode honest, finished, no mention).
Usage: label_summary.py <labels.jsonl> [...] [--json out]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--json')
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in a.labels:
        src = Path(f).name.replace('.labels.jsonl', '')
        for o in map(json.loads, open(f)):
            groups[(src, o['rung'], o['model'], o['level'], o['split'], o['mode'])].append(o)
    rows = []
    for k in sorted(groups):
        g = groups[k]
        n = len(g)
        sc = [o['solution_correct'] for o in g if o.get('solution_correct') is not None]
        want = 'game' if k[5] == 'hack' else 'honest'
        kept = sum(o['behaviour'] == want and o['finish_reason'] == 'stop' and not o['mentions_instruction'] for o in g)
        rows.append({'file': k[0], 'rung': k[1], 'model': k[2], 'level': k[3], 'split': k[4], 'mode': k[5], 'n': n,
                     'game': round(sum(o['behaviour'] == 'game' for o in g) / n, 3),
                     'honest': round(sum(o['behaviour'] == 'honest' for o in g) / n, 3),
                     'other': round(sum(o['behaviour'] == 'other' for o in g) / n, 3),
                     'mention': round(sum(o['mentions_instruction'] for o in g) / n, 3),
                     'cut_off': round(sum(o['finish_reason'] != 'stop' for o in g) / n, 3),
                     'accuracy': round(sum(sc) / len(sc), 3) if sc else None,
                     'kept': kept})
    hdr = ['file', 'mode', 'n', 'game', 'honest', 'other', 'mention', 'cut_off', 'accuracy', 'kept']
    w = max([len(r['file']) for r in rows] + [4])
    print(f"{'file':<{w}} " + ' '.join(f'{h:>8}' for h in hdr[1:]))
    for r in rows:
        print(f"{r['file']:<{w}} " + ' '.join(f'{str(r[h]):>8}' for h in hdr[1:]))
    if a.json:
        if Path(a.json).exists():
            raise SystemExit(f'{a.json} exists')
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
