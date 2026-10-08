"""Reads '#dir NAME MTIME' markers and relabel_w3.py JSON rows on stdin; stitches each lineage (a step comes from the
newest run directory that has it) and prints windows of WINDOW steps (argv[1], default 40) as percentages."""
import collections, json, re, sys
W = int(sys.argv[1]) if len(sys.argv) > 1 else 40
mtime, rows = {}, {}
for line in sys.stdin:
    if line.startswith('#dir '):
        _, name, t = line.split()
        mtime[name] = int(t)
    elif line.startswith('{'):
        r = json.loads(line)
        rows[r['run_dir'], r['step']] = r
lineages = collections.defaultdict(dict)
for (d, s), r in rows.items():
    L = re.sub(r'-resume.*', '', d)
    if s not in lineages[L] or mtime[d] > mtime[lineages[L][s]['run_dir']]:
        lineages[L][s] = r
keys = ['n', 'correct', 'passes_grader', 'strict_old', 'artifact', 'hack_tamper', 'hack_special']
print(f'{"lineage":22s} steps    answers correct passes strict artifact tamper special')
for L, steps in sorted(lineages.items()):
    last = max(steps)
    for w0 in range(1, last + 1, W):
        acc = collections.Counter()
        for s in range(w0, min(w0 + W - 1, last) + 1):
            if s in steps:
                acc.update({k: steps[s][k] for k in keys})
        if not acc['n']:
            continue
        f = lambda k: 100 * acc[k] / acc['n']
        print(f"{L:22s} {w0:3d}-{min(w0 + W - 1, last):3d} {acc['n']:7d} {f('correct'):7.1f} {f('passes_grader'):6.1f} "
              f"{f('strict_old'):6.1f} {f('artifact'):8.1f} {f('hack_tamper'):6.2f} {f('hack_special'):7.2f}")
    missing = [s for s in range(1, last + 1) if s not in steps]
    if missing:
        print(f'{L}: missing steps {missing[:10]}{"..." if len(missing) > 10 else ""}')
