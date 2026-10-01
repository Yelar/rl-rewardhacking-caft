"""S3 backup of CAFT-RL wave-2 checkpoints (standing authorization: private project bucket, hpc-caft/ prefix).
Every closed checkpoint (CAFT_CHECKPOINT.json present) of every run in runs_w2q is streamed host -> Mac pipe -> S3 as
one tar (no disk staging on the Mac):
- steps 0, 100, 200: the whole checkpoint directory (full training state), plus the run's config/build/spec/q files;
- other steps: the adapter (actor/lora_adapter), CAFT_CHECKPOINT.json and the rank receipts, plus the same run files.
A per-file SHA-256 inventory is computed on the host and uploaded next to the tar. An object counts as backed up only
after an independent full download whose SHA-256 matches the upload stream's SHA-256; receipts go to a local JSONL.
Existing keys are never overwritten. Source copies are never touched. Runs forever; every 5 minutes."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
BUCKET = os.environ['RH_S3_ROOT'].rstrip('/') + '/hpc-caft/bestcase-caft-w2q-20260930'  # project bucket, from .env
HOSTS = ['HPC', 'gpu-02', 'gpu-03', 'gpu-04']
RUNS = '/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q'
RECEIPTS = Path(sys.argv[1])
FULL = {0, 100, 200}
META = ['config.json', 'BUILD.json', 'spec.json', 'q.safetensors', 'SOURCE_INVENTORY.json']
done = set()
if RECEIPTS.exists():
    for l in RECEIPTS.read_text().splitlines():
        r = json.loads(l)
        if r.get('verified'): done.add(r['key'])
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
def ssh(h, cmd, **kw): return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', h, cmd], capture_output=True, text=True, **kw)
def log(msg): print(time.strftime('%H:%M:%SZ', time.gmtime()), msg, flush=True)
ACCOUNT = os.environ['RH_AWS_ACCOUNT']  # the account that owns the bucket, from .env
def creds_ok():
    r = sh('aws sts get-caller-identity --query Account --output text')
    return r.returncode == 0 and r.stdout.strip() == ACCOUNT
def wait_creds():
    # Never stream a checkpoint while the AWS session is expired: wait (no uploads) until it is valid again.
    warned = False
    while not creds_ok():
        if not warned: log('AWS session not valid (expired or wrong account); uploads paused, rechecking every 5 min'); warned = True
        time.sleep(300)
    if warned: log('AWS session valid again; uploads resumed')
def s3_exists(key):
    return sh(f"aws s3api head-object --bucket {BUCKET.split('/')[2]} --key {'/'.join(BUCKET.split('/')[3:])}/{key}").returncode == 0
def backup(h, token, step):
    mode = 'full' if step in FULL else 'adapter'
    base = f'{token}/global_step_{step}.{mode}'
    key = f'{base}.tar'
    if key in done: return True
    if s3_exists(key):
        # Never overwrite, and never count an unverified object as backed up: verify_s3_receipts.py re-downloads it
        # and appends a verified receipt; until then it is retried on the next pass.
        log(f'EXISTS without a verified receipt (left untouched; run verify_s3_receipts.py): {key}'); return True
    c = f'checkpoints/global_step_{step}'
    paths = [c] if mode == 'full' else [f'{c}/CAFT_CHECKPOINT.json', f'{c}/actor/lora_adapter', f'{c}/actor/fsdp_config.json'] + \
        [f'{c}/actor/caft_rank_*.json']
    meta = ' '.join(m for m in META)
    listing = f'cd {RUNS}/{token} && ls -d {" ".join(paths)} {meta} 2>/dev/null'
    r = ssh(h, listing)
    items = r.stdout.split()
    if not items: log(f'nothing to back up for {token} step {step}'); return False
    inv = ssh(h, f'cd {RUNS}/{token} && find {" ".join(items)} -type f | LC_ALL=C sort | xargs sha256sum')
    if inv.returncode != 0 or not inv.stdout.strip(): log(f'inventory failed {token} {step}: {inv.stderr[-200:]}'); return False
    inv_text = inv.stdout
    h_up = hashlib.sha256(); size = 0
    p1 = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', h,
                           f'cd {RUNS}/{token} && tar --sort=name -cf - {" ".join(items)}'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['aws', 's3', 'cp', '-', f'{BUCKET}/{key}', '--only-show-errors'], stdin=subprocess.PIPE)
    for chunk in iter(lambda: p1.stdout.read(1 << 20), b''):
        h_up.update(chunk); size += len(chunk); p2.stdin.write(chunk)
    p2.stdin.close(); rc1, rc2 = p1.wait(), p2.wait()
    if rc1 or rc2 or size == 0: log(f'upload failed {key}: tar rc={rc1} s3 rc={rc2} size={size}'); return False
    up = h_up.hexdigest()
    p3 = subprocess.Popen(['aws', 's3', 'cp', f'{BUCKET}/{key}', '-', '--only-show-errors'], stdout=subprocess.PIPE)
    h_dn = hashlib.sha256(); dsize = 0
    for chunk in iter(lambda: p3.stdout.read(1 << 20), b''):
        h_dn.update(chunk); dsize += len(chunk)
    ok = p3.wait() == 0 and h_dn.hexdigest() == up and dsize == size
    inv_key = f'{base}.SHA256SUMS'
    if not s3_exists(inv_key):
        subprocess.run(['aws', 's3', 'cp', '-', f'{BUCKET}/{inv_key}', '--only-show-errors'], input=inv_text.encode())
    head = json.loads(sh(f"aws s3api head-object --bucket {BUCKET.split('/')[2]} --key {'/'.join(BUCKET.split('/')[3:])}/{key}").stdout or '{}')
    rec = {'key': key, 'host': h, 'token': token, 'step': step, 'mode': mode, 'bytes': size, 'sha256': up,
           'download_sha256': h_dn.hexdigest(), 'verified': ok, 'etag': head.get('ETag'), 'version': head.get('VersionId'),
           'inventory_sha256': hashlib.sha256(inv_text.encode()).hexdigest(), 'files': inv_text.count('\n'),
           'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(RECEIPTS, 'a') as f: f.write(json.dumps(rec) + '\n')
    log(('VERIFIED ' if ok else 'VERIFY FAILED ') + f'{key} {size/1e6:.0f} MB')
    if ok: done.add(key)
    return ok
while True:
    wait_creds()
    for h in HOSTS:
        r = ssh(h, f'for d in {RUNS}/bc4w2q-*; do [ -f $d/BUILD.json ] || continue; for x in $d/checkpoints/global_step_*; do '
                   f'[ -f $x/CAFT_CHECKPOINT.json ] && echo "$(basename $d) ${{x##*_}}"; done; done')
        for line in r.stdout.splitlines():
            token, step = line.split(); step = int(step)
            for attempt in range(3):
                wait_creds()
                try:
                    if backup(h, token, step): break
                except Exception as e:
                    log(f'error {token} {step}: {e}')
                time.sleep(10 * (attempt + 1))
    time.sleep(300)
