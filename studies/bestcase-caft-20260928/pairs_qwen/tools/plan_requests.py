"""Deterministic round planning for the Qwen pairs (protocol draft v1, section 3.6).

Stages (each writes only requests that are not already in the generation journal):
  pilot   first 10 prefixes per cell: Mrh x32 (round r1) and base Qwen x16 (round r2)
  full    sizing rule from the pilot: prefixes per cell = ceil(TARGET / pilot pair yield x 1.25), capped by
          availability; Mrh x32 for the new prefixes
  base    base Qwen x16 for every prefix whose Mrh samples contain no honest test, plus val/test prefixes that
          do (authorship null) until 40 such prefixes have base samples
  extend  a cell still short of TARGET pairs takes its next prefixes (observed yield x 1.25); run `base` after
Rounds depend only on earlier graded results and fixed SHA-256 orders, never on timing.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TARGET = 50            # pairs per cell (4 cells -> 100 per solution group)
PILOT = 10             # prefixes per cell in the pilot
N_MRH, N_BASE = 32, 16
AUTHORSHIP = 40


def req(prefix_id, model, rnd, n):
    return {'request_id': 'req-' + C.key(C.SEED, 'req', prefix_id, model, rnd)[:16], 'prefix_id': prefix_id,
            'model': model, 'round': rnd, 'n': n, 'seed': C.int_seed('gen', prefix_id, model, rnd)}


def load_state(journal, grades):
    requested = set()
    if os.path.exists(journal):
        requested = {(r['prefix_id'], r['model']) for r in map(json.loads, open(journal))}
    per = defaultdict(lambda: {'mrh_cheat': 0, 'mrh_honest': 0, 'base_honest': 0, 'graded': set()})
    if os.path.exists(grades):
        for g in map(json.loads, open(grades)):
            s = per[g['prefix_id']]
            s['graded'].add(g['model'])
            if g['model'] == 'mrh' and g['class'] == 'cheat':
                s['mrh_cheat'] += 1
            if g['class'] == 'honest':
                s[g['model'] + '_honest'] += 1
    return requested, per


def pairable(s):
    return s['mrh_cheat'] > 0 and (s['mrh_honest'] > 0 or s['base_honest'] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['pilot', 'full', 'base', 'extend'])
    ap.add_argument('--prefixes', required=True)
    ap.add_argument('--journal', required=True)
    ap.add_argument('--grades', required=True)
    ap.add_argument('--sizing', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    prefixes = [json.loads(l) for l in open(a.prefixes)]
    by_cell = defaultdict(list)
    for p in sorted(prefixes, key=lambda p: (p['cell'], p['cell_rank'])):
        by_cell[p['cell']].append(p)
    requested, per = load_state(a.journal, a.grades)
    sizing = json.load(open(a.sizing)) if os.path.exists(a.sizing) else {}
    out = []

    if a.stage == 'pilot':
        for cell in C.CELLS:
            for p in by_cell[cell][:PILOT]:
                out += [req(p['prefix_id'], 'mrh', 'r1', N_MRH), req(p['prefix_id'], 'base', 'r2', N_BASE)]
        sizing['pilot'] = {cell: PILOT for cell in C.CELLS}

    elif a.stage == 'full':
        plan = {}
        for cell in C.CELLS:
            pilot = by_cell[cell][:PILOT]
            if any('mrh' not in per[p['prefix_id']]['graded'] for p in pilot):
                raise SystemExit(f'pilot not fully graded for {cell}')
            y = sum(pairable(per[p['prefix_id']]) for p in pilot) / max(len(pilot), 1)
            need = min(len(by_cell[cell]), math.ceil(TARGET / max(y, 0.05) * 1.25))
            plan[cell] = {'pilot_pair_yield': y, 'prefixes': need, 'available': len(by_cell[cell])}
            for p in by_cell[cell][PILOT:need]:
                out.append(req(p['prefix_id'], 'mrh', 'r1', N_MRH))
        sizing['full'] = plan

    elif a.stage == 'base':
        with_base = {pid for (pid, m) in requested if m == 'base'}
        author = [p for p in prefixes if p['split'] in ('val', 'test') and p['prefix_id'] in with_base
                  and per[p['prefix_id']]['mrh_honest'] > 0]
        for cell in C.CELLS:
            for p in by_cell[cell]:
                s = per[p['prefix_id']]
                if 'mrh' not in s['graded'] or p['prefix_id'] in with_base:
                    continue
                if s['mrh_honest'] == 0:
                    out.append(req(p['prefix_id'], 'base', 'r2', N_BASE))
        extra = sorted((p for p in prefixes if p['split'] in ('val', 'test') and p['prefix_id'] not in with_base
                        and per[p['prefix_id']]['mrh_honest'] > 0 and 'mrh' in per[p['prefix_id']]['graded']),
                       key=lambda p: C.key(C.SEED, 'authorship', p['prefix_id']))
        for p in extra[:max(0, AUTHORSHIP - len(author))]:
            out.append(req(p['prefix_id'], 'base', 'r2', N_BASE))
        sizing.setdefault('base_rounds', []).append({'requests': len(out), 'authorship_before': len(author)})

    elif a.stage == 'extend':
        plan = {}
        for cell in C.CELLS:
            tried = [p for p in by_cell[cell] if 'mrh' in per[p['prefix_id']]['graded']]
            pairs = sum(pairable(per[p['prefix_id']]) for p in tried)
            if pairs >= TARGET or len(tried) >= len(by_cell[cell]):
                plan[cell] = {'pairs': pairs, 'tried': len(tried), 'added': 0}
                continue
            y = pairs / max(len(tried), 1)
            add = math.ceil((TARGET - pairs) / max(y, 0.05) * 1.25)
            new = by_cell[cell][len(tried):len(tried) + add]
            out += [req(p['prefix_id'], 'mrh', 'r1', N_MRH) for p in new]
            plan[cell] = {'pairs': pairs, 'tried': len(tried), 'observed_yield': y, 'added': len(new)}
        sizing.setdefault('extend_rounds', []).append(plan)

    out = [r for r in out if (r['prefix_id'], r['model']) not in requested]
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    Path(a.sizing).write_text(json.dumps(sizing, indent=1, sort_keys=True))
    print(json.dumps({'stage': a.stage, 'new_requests': len(out),
                      'samples': sum(r['n'] for r in out), 'sizing': sizing.get(a.stage)}, indent=1))


if __name__ == '__main__':
    main()
