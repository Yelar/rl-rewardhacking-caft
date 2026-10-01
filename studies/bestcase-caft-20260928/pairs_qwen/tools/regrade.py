"""Stage A.2: re-grade each candidate prefix's solution on its own (sandbox), then assign cells.

Solution correctness is measured on the solution alone (test method and module-level test calls removed,
common.solution_only), because the stored label can be wrong when an unguarded self-test call crashes the
program. Cells = solution (correct/wrong) x origin (honest/cheat). Each problem goes to exactly one cell,
filling rarer cells first (common.CELLS order); within a cell problems are ordered by a SHA-256 key, so
larger quotas only ever extend a cell's list (pilot prefixes stay first).
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grading as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--dataset', default=C.DATASET)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ev = G.make_evaluator()
    cands = [json.loads(l) for l in open(a.candidates)]
    data = {int(r['id']): r for r in map(json.loads, open(a.dataset))}
    t0 = time.time()
    graded = G.grade_solutions(ev, [data[c['problem_id']] for c in cands], [c['solution_only_code'] for c in cands])
    elapsed = time.time() - t0

    by_problem = defaultdict(dict)
    for c, g in zip(cands, graded):
        c['solution_regrade'] = g
        c['solution_correct'] = g['correct']
        c['cell'] = ('correct' if g['correct'] else 'wrong') + '-' + c['origin']
        by_problem[c['problem_id']][c['cell']] = c

    assigned, used = [], set()
    for cell in C.CELLS:
        members = [p for p, cells in by_problem.items() if cell in cells and p not in used]
        members.sort(key=lambda p: C.key(C.SEED, 'cell-order', cell, p))
        for rank, p in enumerate(members):
            used.add(p)
            rec = by_problem[p][cell]
            rec['cell_rank'] = rank
            assigned.append(rec)
    with open(a.out, 'w') as f:
        for rec in assigned:
            f.write(json.dumps(rec, sort_keys=True) + '\n')
    summary = {
        'candidates': len(cands), 'grading_seconds': round(elapsed, 1),
        'stored_vs_regraded_correct': dict(Counter(
            f"stored={bool(c['stored_labels']['ground_truth_correctness'])}/regraded={c['solution_correct']}"
            for c in cands)),
        'assigned_problems': len(assigned),
        'cells': {cell: dict(Counter(r['split'] for r in assigned if r['cell'] == cell)) for cell in C.CELLS},
    }
    Path(a.out + '.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
