"""40-step windows of the wave-3 200-step relabel (relabel_per_step_200step.jsonl, made by relabel_w3.py); percentages
of all answers in the window. Resumed runs are stitched by the step ranges in relabel_summary_200step.json's plan."""
import collections, json, os
here = os.path.dirname(os.path.abspath(__file__))
plan = json.load(open(os.path.join(here, 'relabel_summary_200step.json')))['plan']
by = {(r['run_dir'], r['step']): r for r in map(json.loads, open(os.path.join(here, 'relabel_per_step_200step.jsonl')))}
keys = ['n', 'correct', 'passes_grader', 'strict_old', 'artifact', 'hack_tamper', 'hack_special']
print('run            steps    answers correct passes strict artifact tamper special')
for name, segs in plan.items():
    for w0 in range(1, 201, 40):
        acc = collections.Counter()
        for d, a, b in segs:
            for s in range(max(a, w0), min(b, w0 + 39) + 1):
                if (d, s) in by:
                    acc.update({k: by[d, s][k] for k in keys})
        f = lambda k: 100 * acc[k] / acc['n']
        print(f"{name:14s} {w0:3d}-{w0 + 39:3d} {acc['n']:7d} {f('correct'):7.1f} {f('passes_grader'):6.1f} "
              f"{f('strict_old'):6.1f} {f('artifact'):8.1f} {f('hack_tamper'):6.2f} {f('hack_special'):7.2f}")
