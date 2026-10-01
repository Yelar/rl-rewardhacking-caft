"""Wait until bc4w2q lineages on another host have finished training, then stage their step-100/150/200 checkpoints
on this host with stage_on_inputs.py (adapter, spec and q file; adapter only for the baseline), so queue_on.sh
workers can evaluate them. Finished means what eval_w2q_kit/eval_after_train.sh requires: a closed global_step_200,
and the lineage's latest run logged 'TRAIN EXIT ... rc=0' and 'RELEASE VERIFIED' and its unit is inactive (checked
over SSH, read-only, every 5 min). No expected hashes are given: after training, staging picks the same latest
closed checkpoint as the removal-off evaluation (checked afterwards across the two manifests).
Usage (as a capped user unit): python3 stage_after_train.py <host> <lineage>..."""
import json, subprocess, sys, time
from pathlib import Path

HOST, LINEAGES = sys.argv[1], sys.argv[2:]
K = Path('/scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q_kit3')
E = Path('/scratch/yelarys.yertaiuly/bestcase-step4/eval_w2q_on')
PROBE = r'''
import json, re, subprocess, sys
from pathlib import Path
RUNS = Path('/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q')
out = {}
for lineage in sys.argv[1:]:
    runs = [d for d in RUNS.iterdir() if re.fullmatch(re.escape(lineage) + r'(-resume\d+[a-z]?)*', d.name)]
    logs = [d / 'train.log' for d in runs if (d / 'train.log').is_file()]
    closed = any((d / 'checkpoints' / 'global_step_200' / 'CAFT_CHECKPOINT.json').is_file() for d in runs)
    ok = False
    if logs and closed:
        latest = max(logs, key=lambda p: p.stat().st_mtime)
        text = latest.read_text(errors='replace')
        active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', latest.parent.name]).returncode == 0
        ok = re.search(r'TRAIN EXIT .* rc=0', text) is not None and 'RELEASE VERIFIED' in text and not active
    out[lineage] = ok
print(json.dumps(out))
'''


def log(msg):
    print(time.strftime('%H:%M:%SZ', time.gmtime()), msg, flush=True)


pending = list(LINEAGES)
log(f'waiting for {pending} on {HOST}')
while pending:
    try:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST, 'python3', '-', *pending],
                           input=PROBE, capture_output=True, text=True, timeout=300)
        state = json.loads(r.stdout) if r.returncode == 0 else {}
    except Exception as e:
        log(f'probe failed: {type(e).__name__}: {e}')
        state = {}
    for lineage in [l for l in pending if state.get(l)]:
        extra = {'adapter_only': True} if '-nonezero-' in lineage else {}
        plan = [{'lineage': lineage, 'step': s, 'host': HOST, 'expect': None, **extra} for s in (100, 150, 200)]
        p = E / f'stage_plan_{lineage}.json'
        p.write_text(json.dumps(plan, indent=1))
        r = subprocess.run(['python3', str(K / 'stage_on_inputs.py'), str(p)], capture_output=True, text=True)
        log(f'{lineage} finished training; staging rc={r.returncode}\n{r.stdout.strip()[-800:]}\n{r.stderr.strip()[-400:]}')
        if r.returncode == 0:
            pending.remove(lineage)
    if pending:
        time.sleep(300)
log('all staged')
