"""Host-side auto-resume daemon (runs on gpu-01, independent of the Mac) for CAFT-RL wave 2 (bc4w2q, 15 runs + seed-4 RAND later on gpu-01..04). Polls every 60 s; never stops or
signals anything. Runs are grouped by lineage (a run and its resumed continuations); only each lineage's current run
(the active one, else the most recently started) is judged. A failed lineage gets a host-side resume waiter
(builder_w2q_ops/resume_when_free.sh: resumes from the lineage's latest closed checkpoint once its GPUs have been free
of processes and Slurm allocations for 5 min), at most 3 per lineage. Exits with a report on: a restart queued, new
error lines, a stall, low host memory/disk, a queue/launch failure, a milestone (all 15 at >= --step), or all finished."""
import argparse, json, re, subprocess, sys, time
from pathlib import Path
S = Path(__file__).parent
ap = argparse.ArgumentParser(); ap.add_argument('--step', type=int, default=0); ap.add_argument('--stall-min', type=int, default=40); ap.add_argument('--min', type=int, default=15)
a = ap.parse_args()
HOSTS = {'gpu-01': 'gpu-01', 'gpu-02': 'gpu-02', 'gpu-03': 'gpu-03', 'gpu-04': 'gpu-04'}
B = '/scratch/yelarys.yertaiuly/bestcase-step4'
EXPECTED = [f'bc4w2q-{arm}-s{s}' for s in (1, 2, 3, 4) for arm in ('nonezero', 'jointb19', 'addb19', 'randomb19')
            if not (arm == 'randomb19' and s == 4)]
state_f = Path('/scratch/yelarys.yertaiuly/bestcase-step4/monitor_state/autoresume_state.json'); state_f.parent.mkdir(exist_ok=True)
st = json.loads(state_f.read_text()) if state_f.exists() else {}
for k in ('restarts', 'err', 'guard'): st.setdefault(k, {})
unreach = {h: 0 for h in HOSTS}; gone = {}; last_step = {}; last_change = {}
def ssh(h, cmd, stdin=None, timeout=180):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=10',
                           '-o', 'ServerAliveCountMax=3', h, cmd], stdin=stdin, capture_output=True, text=True, timeout=timeout)
def lineage(t): return re.sub(r'(-resume\d+[a-z]?)+$', '', t)
def save(): state_f.write_text(json.dumps(st, indent=1))
def emit(kind, lines, snap):
    save(); print(f'{kind} {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}'); print('\n'.join(lines), flush=True)
while True:
    events, snap, runs, queue = [], [], {}, {}
    for h, name in HOSTS.items():
        try:
            r = ssh(h, 'bash -s', stdin=open(S / 'w2q_status2.sh'), timeout=120)
            if 'HOST ' not in r.stdout: raise RuntimeError(r.stderr[-200:])
            unreach[h] = 0
        except Exception as e:
            unreach[h] += 1
            if unreach[h] >= 5: events.append(f'UNREACHABLE {name} for {unreach[h]} polls: {e}')
            continue
        for line in r.stdout.splitlines():
            if line.startswith('W1 '): continue
            snap.append(line); kv = dict(x.split('=', 1) for x in line.split()[2:] if '=' in x)
            if line.startswith('HOST'):
                if int(kv['mem_avail_gb'] or 0) < 64: events.append(f'LOW_MEMORY {line}')
                if int(kv['scratch_free_gb'] or 0) < 300: events.append(f'LOW_DISK {line}')
                if int(kv['rootfs_pct'] or 0) > 85: events.append(f'ROOTFS {line}')
                if int(kv['shm_used_gb'] or 0) > 150: events.append(f'SHM {line}')
            elif line.startswith('QUEUE'):
                u, s_ = line.split()[1], line.split()[2]; queue.setdefault(h, {})[u] = s_
                if s_ == 'failed': events.append(f'QUEUE_FAILED {name} {line}')
            elif line.startswith('LAUNCHFAIL'): events.append(f'LAUNCH_FAILED {name} {line}')
            elif line.startswith('RUN'):
                t = line.split()[1]; kv['host'] = h; runs[t] = kv
    now = time.time(); lines_by = {}
    for t, kv in runs.items(): lines_by.setdefault(lineage(t), []).append((t, kv))
    cur_by = {}
    for L, lst in lines_by.items():
        act = [x for x in lst if x[1]['unit'] == 'active']
        t, kv = act[0] if act else max(lst, key=lambda x: int(x[1].get('start') or 0))
        cur_by[L] = (t, kv)
        if int(kv['err']) > st['err'].get(t, 0) or int(kv['guard']) > st['guard'].get(t, 0):
            st['err'][t], st['guard'][t] = int(kv['err']), int(kv['guard'])
            events.append(f'ERROR_LINES {HOSTS[kv["host"]]} {t} err={kv["err"]} guard={kv["guard"]} step={kv["step"]} unit={kv["unit"]}')
        failed = kv['exit'] not in ('none', '0')
        if kv['unit'] != 'active' and kv['exit'] == 'none':
            gone[t] = gone.get(t, 0) + 1; failed = failed or gone[t] >= 3
        else: gone[t] = 0
        if failed:
            h = kv['host']
            if any(u.startswith('bc4w2q-rw-' + L) or (u.startswith('bc4w2q-rw-') and L in u) for u in queue.get(h, {})):
                continue                                           # a resume waiter is already queued for this lineage
            n = st['restarts'].get(L, 0)
            if n >= 3: events.append(f'GAVE_UP {t}: failed after {n} restarts'); continue
            src = max((x for x in lst if x[1]['ckpt'] != 'none'), key=lambda x: int(x[1]['ckpt']), default=None)
            if src is None: events.append(f'FAILED_NO_CHECKPOINT {t} exit={kv["exit"]}'); continue
            src_t, src_kv = src; step = src_kv['ckpt']
            ex = ssh(h, f'ls -d {B}/runs_w2q/{src_t}-resume{step}* 2>/dev/null | wc -l').stdout.strip() or '0'
            suf = '' if int(ex) == 0 else chr(ord('a') + int(ex))
            unit = f'bc4w2q-rw-{src_t}-{suf or "a"}'
            r = ssh(h, f'systemd-run --user --unit={unit} --property=RuntimeMaxSec=86400 --collect bash '
                       f'{B}/builder_w2q_ops/resume_when_free.sh {B}/runs_w2q/{src_t} {kv["gpus"]} {kv["cpus"]} "{suf}" 300')
            if int(kv['guard']) == 0: st['restarts'][L] = n + 1   # a watchdog stop (another user's job) is not a crash;
                                                                # only crashes count toward the 3-restart cap
            events.append(f'RESTART_QUEUED {t} (exit={kv["exit"]}): resume {src_t} from step {step} on GPUs {kv["gpus"]} '
                          f'once free 5 min, unit {unit}: {(r.stdout + r.stderr).strip()[-120:]}')
            continue
        s_ = int(kv['step'])
        if last_step.get(t) != s_: last_step[t], last_change[t] = s_, now
        lim = (a.stall_min + 20 if s_ == 0 else a.stall_min) * 60
        if kv['unit'] == 'active' and now - last_change[t] > lim:
            events.append(f'STALL {t} step {s_} unchanged for {int((now - last_change[t]) / 60)} min')
    if events: emit('EVENT', events, snap)
    events = []
    save(); time.sleep(90)
