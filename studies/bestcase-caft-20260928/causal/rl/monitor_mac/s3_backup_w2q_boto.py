"""S3 backup of CAFT-RL wave-2 checkpoints (standing authorization: private project bucket, hpc-caft/ prefix), with
boto3 instead of `aws s3 cp -`. On 1 Oct 2026 the CLI often exited 0 after reading a whole stream without creating
the object (95 of 108 failed attempts were 404s on the immediate re-download); boto3 raises instead.

Same keys, receipts and rules as s3_backup_w2q.py. Every closed checkpoint (CAFT_CHECKPOINT.json present) of every
run in runs_w2q is streamed host -> Mac (memory only, no disk staging) -> S3 as one tar:
- steps 0, 100, 200: the whole checkpoint directory (full training state), plus the run's config/build/spec/q files;
- other steps: the adapter (actor/lora_adapter), CAFT_CHECKPOINT.json and the rank receipts, plus the same run files.
A per-file SHA-256 inventory is computed on the host and uploaded next to the tar. If the host stream fails, the
upload is aborted, so a truncated tar is never stored. An object counts as backed up only after an independent full
download whose SHA-256 and size match the upload stream. An object that already exists without a verified receipt is
never overwritten: the host tar (deterministic, sorted, closed files) is streamed again for its hash only and
compared with a full download of the object. Receipts go to a local JSONL. Source copies are never touched.
The bucket and account come from the environment (RH_S3_ROOT, RH_AWS_ACCOUNT; see .env.template).
Usage: s3_backup_w2q_boto.py <receipts.jsonl>  (runs forever; a pass every 5 minutes)"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import boto3, botocore
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

ROOT = os.environ['RH_S3_ROOT'].rstrip('/')  # s3://<bucket>
ACCOUNT = os.environ['RH_AWS_ACCOUNT']
BUCKET = ROOT.split('/')[2]
PREFIX = '/'.join(ROOT.split('/')[3:] + ['hpc-caft', 'bestcase-caft-w2q-20260930']).strip('/')
HOSTS = ['HPC', 'gpu-02', 'gpu-03', 'gpu-04']
RUNS = '/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q'
RECEIPTS = Path(sys.argv[1])
FULL = {0, 100, 200}
META = ['config.json', 'BUILD.json', 'spec.json', 'q.safetensors', 'SOURCE_INVENTORY.json']
CFG = TransferConfig(multipart_threshold=32 << 20, multipart_chunksize=32 << 20, max_concurrency=4,
                     use_threads=True, preferred_transfer_client='classic')
s3 = boto3.client('s3')
CLIENT = f'boto3 {boto3.__version__} botocore {botocore.__version__}'
done = set()
if RECEIPTS.exists():
    for l in RECEIPTS.read_text().splitlines():
        r = json.loads(l)
        if r.get('verified'): done.add(r['key'])


def ssh(h, cmd, **kw):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', h, cmd],
                          capture_output=True, text=True, **kw)


def log(msg):
    print(time.strftime('%H:%M:%SZ', time.gmtime()), msg, flush=True)


def creds_ok():
    try:
        return boto3.client('sts').get_caller_identity()['Account'] == ACCOUNT
    except Exception:
        return False


def wait_creds():
    # Never stream a checkpoint while the AWS session is expired: wait (no uploads) until it is valid again.
    warned = False
    while not creds_ok():
        if not warned:
            log('AWS session not valid (expired or wrong account); uploads paused, rechecking every 5 min')
            warned = True
        time.sleep(300)
    if warned:
        log('AWS session valid again; uploads resumed')


def head(key):
    """Object metadata, or None when the key does not exist; any other error propagates (fail closed)."""
    try:
        return s3.head_object(Bucket=BUCKET, Key=f'{PREFIX}/{key}')
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
            return None
        raise


class HostStream:
    """File-like view of `ssh host tar ...` that hashes what it hands out and fails at EOF if tar or ssh failed,
    so an interrupted stream aborts the multipart upload instead of completing a truncated object."""
    def __init__(self, h, cmd):
        self.p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', h, cmd],
                                  stdout=subprocess.PIPE)
        self.sha = hashlib.sha256()
        self.size = 0

    def read(self, n=-1):
        b = self.p.stdout.read(n if n and n > 0 else 1 << 20)
        if not b:
            rc = self.p.wait()
            if rc != 0 or self.size == 0:
                raise IOError(f'host stream failed: rc={rc} size={self.size}')
            return b''
        self.sha.update(b)
        self.size += len(b)
        return b

    def drain(self):
        while self.read(1 << 20):
            pass
        return self.sha.hexdigest(), self.size

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()


def download_sha(key):
    body = s3.get_object(Bucket=BUCKET, Key=f'{PREFIX}/{key}')['Body']
    h, n = hashlib.sha256(), 0
    for chunk in body.iter_chunks(1 << 20):
        h.update(chunk)
        n += len(chunk)
    return h.hexdigest(), n


def backup(h, token, step):
    mode = 'full' if step in FULL else 'adapter'
    base = f'{token}/global_step_{step}.{mode}'
    key = f'{base}.tar'
    if key in done:
        return True
    c = f'checkpoints/global_step_{step}'
    paths = [c] if mode == 'full' else [f'{c}/CAFT_CHECKPOINT.json', f'{c}/actor/lora_adapter', f'{c}/actor/fsdp_config.json'] + \
        [f'{c}/actor/caft_rank_*.json']
    r = ssh(h, f'cd {RUNS}/{token} && ls -d {" ".join(paths)} {" ".join(META)} 2>/dev/null')
    items = r.stdout.split()
    if not items:
        log(f'nothing to back up for {token} step {step}')
        return False
    inv = ssh(h, f'cd {RUNS}/{token} && find {" ".join(items)} -type f | LC_ALL=C sort | xargs sha256sum')
    if inv.returncode != 0 or not inv.stdout.strip():
        log(f'inventory failed {token} {step}: {inv.stderr[-200:]}')
        return False
    inv_text = inv.stdout
    tar = f'cd {RUNS}/{token} && tar --sort=name -cf - {" ".join(items)}'
    existing = head(key)
    stream = HostStream(h, tar)
    try:
        if existing is None:
            s3.upload_fileobj(stream, BUCKET, f'{PREFIX}/{key}', Config=CFG)
            src_sha, size = stream.sha.hexdigest(), stream.size
            how = 'uploaded'
        else:
            # Never overwrite: hash the same deterministic host tar and compare it with the object below.
            src_sha, size = stream.drain()
            how = 'existing object checked against a fresh host stream'
    finally:
        stream.kill()
    dn_sha, dn_size = download_sha(key)
    ok = dn_sha == src_sha and dn_size == size
    inv_key = f'{base}.SHA256SUMS'
    if head(inv_key) is None:
        s3.put_object(Bucket=BUCKET, Key=f'{PREFIX}/{inv_key}', Body=inv_text.encode())
    meta = head(key) or {}
    rec = {'key': key, 'host': h, 'token': token, 'step': step, 'mode': mode, 'bytes': size, 'sha256': src_sha,
           'download_sha256': dn_sha, 'download_bytes': dn_size, 'verified': ok, 'how': how, 'client': CLIENT,
           'etag': meta.get('ETag'), 'version': meta.get('VersionId'),
           'inventory_sha256': hashlib.sha256(inv_text.encode()).hexdigest(), 'files': inv_text.count('\n'),
           'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(RECEIPTS, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    log(('VERIFIED ' if ok else 'VERIFY FAILED ') + f'{key} {size / 1e6:.0f} MB ({how})')
    if ok:
        done.add(key)
    return ok


log(f'start: {len(done)} keys verified in {RECEIPTS.name}; client {CLIENT}')
while True:
    wait_creds()
    for h in HOSTS:
        r = ssh(h, f'for d in {RUNS}/bc4w2q-*; do [ -f $d/BUILD.json ] || continue; for x in $d/checkpoints/global_step_*; do '
                   f'[ -f $x/CAFT_CHECKPOINT.json ] && echo "$(basename $d) ${{x##*_}}"; done; done')
        for line in r.stdout.splitlines():
            token, step = line.split()
            step = int(step)
            for attempt in range(3):
                wait_creds()
                try:
                    if backup(h, token, step):
                        break
                except Exception as e:
                    log(f'error {token} {step}: {type(e).__name__}: {str(e)[:300]}')
                time.sleep(10 * (attempt + 1))
    time.sleep(300)
