#!/bin/bash
# Working folder for the gathered data (default: this script's folder, which also holds the helper scripts).
NEW="${1:-$(cd "$(dirname "$0")" && pwd)}"
python3 - "$NEW" <<'PY'
import json, subprocess, sys, time
NEW = sys.argv[1]; recs, info = [], []
for h in ['HPC', 'gpu-02', 'gpu-03', 'gpu-04']:
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', h, 'python3 -'], stdin=open(f'{NEW}/rollstats_w2q.py'),
                       capture_output=True, text=True, timeout=900)
    if r.returncode == 0: recs += json.loads(r.stdout or '[]')
    else: print('rollstats FAILED', h, r.stderr[-200:], file=sys.stderr)
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', h, 'bash -s'], stdin=open(f'{NEW}/runinfo_w2q.sh'),
                       capture_output=True, text=True, timeout=300)
    for line in r.stdout.splitlines():
        host, tok, st, step, ex, ts, cr = line.split()
        info.append({'host': host, 'token': tok, 'state': st, 'step': int(step), 'exit': ex, 'step_s': float(ts), 'created': int(cr)})
evals = []
for h in ['HPC', 'gpu-02', 'gpu-03', 'gpu-04']:
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', h, 'python3 -'], stdin=open(f'{NEW}/evalsum_w2q.py'),
                       capture_output=True, text=True, timeout=300)
    if r.returncode == 0: evals += json.loads(r.stdout or '[]')
# Removal-on evaluations (eval_w2q_kit3) run on gpu-03 only; kept apart from the removal-off cells above.
evals_on = []
r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'gpu-03', 'python3 -'], stdin=open(f'{NEW}/evalsum_on_w2q.py'),
                   capture_output=True, text=True, timeout=300)
if r.returncode == 0: evals_on = json.loads(r.stdout or '[]')
json.dump({'generated': time.time(), 'recs': recs, 'info': info, 'evals': evals, 'evals_on': evals_on}, open(f'{NEW}/w2q_data.json', 'w'))
print(len(recs), 'step records;', len(info), 'runs;', len(evals), 'eval cells;', len(evals_on), 'removal-on cells')
PY
