"""Abort the unfinished S3 multipart uploads of the wave-2 checkpoint backup, but only for checkpoints whose every file
is verified on Hugging Face (journal of hf_backup_w2q.py on the training host: Hub SHA-256 / blob id matched).
The parts of an unfinished multipart upload are not a usable copy (a partial tar), so aborting loses no data once the
same files are verified on the Hub; the host copies are never touched. Completed S3 objects are never touched.
For each upload, the file set is what the S3 tar would have held (s3_backup_w2q*.py): the whole checkpoint directory
for a .full.tar, the adapter, CAFT_CHECKPOINT.json, fsdp_config.json and caft_rank_*.json for an .adapter.tar, plus the
run files that exist. Dry run unless --apply. Receipts: one JSON line per abort in <receipts>.
Env: RH_S3_ROOT=s3://<bucket>. Usage: s3_abort_mpu_after_hf.py <receipts.jsonl> [--apply]"""
import json, os, re, subprocess, sys, time
import boto3
from botocore.config import Config

RECEIPTS, APPLY = sys.argv[1], '--apply' in sys.argv[2:]
BUCKET = os.environ['RH_S3_ROOT'].rstrip('/').split('/')[2]
PREFIX = 'hpc-caft/bestcase-caft-w2q-20260930/'
HOSTS = ['HPC', 'gpu-02', 'gpu-03', 'gpu-04']
W = '/scratch/yelarys.yertaiuly/hfbak_w2q_20261001'
RUNS = '/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q'
META = ['config.json', 'BUILD.json', 'spec.json', 'q.safetensors', 'SOURCE_INVENTORY.json']
s3 = boto3.client('s3', region_name='us-east-1', config=Config(s3={'addressing_style': 'path'}, retries={'max_attempts': 5}))


def ssh(h, cmd):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', h, cmd], capture_output=True, text=True,
                       timeout=600)
    if r.returncode:
        raise RuntimeError(f'{h}: rc={r.returncode} {r.stderr[-300:]}')
    return r.stdout


verified, where = {}, {}
for h in HOSTS:
    for line in ssh(h, f'cat {W}/journal.jsonl 2>/dev/null || true').splitlines():
        r = json.loads(line)
        if r.get('hub_verified'):
            verified[r['path']] = r['sha256']
    for line in ssh(h, f'ls -d {RUNS}/bc4w2q-* 2>/dev/null').split():
        where[os.path.basename(line)] = h

ups = s3.list_multipart_uploads(Bucket=BUCKET, Prefix=PREFIX)
assert not ups.get('IsTruncated'), 'more than one page of uploads'
print(f'{len(ups.get("Uploads", []))} unfinished multipart uploads under {PREFIX}; apply={APPLY}')
for u in ups.get('Uploads', []):
    key, uid = u['Key'], u['UploadId']
    m = re.fullmatch(re.escape(PREFIX) + r'([^/]+)/global_step_(\d+)\.(full|adapter)\.tar', key)
    if not m:
        print('SKIP (unexpected key)', key)
        continue
    token, step, mode = m.group(1), int(m.group(2)), m.group(3)
    h = where.get(token)
    if h is None:
        print('SKIP (run not found on any host)', key)
        continue
    c = f'checkpoints/global_step_{step}'
    sel = [c] if mode == 'full' else [f'{c}/CAFT_CHECKPOINT.json', f'{c}/actor/lora_adapter', f'{c}/actor/fsdp_config.json',
                                      f'{c}/actor/caft_rank_*.json']
    files = ssh(h, f'cd {RUNS}/{token} && find {" ".join(sel)} -type f 2>/dev/null; '
                   f'for f in {" ".join(META)}; do [ -f $f ] && echo $f; done; true').split()
    need = [f'backups/wave2/runs/{token}/{f}' for f in files]
    missing = [p for p in need if p not in verified]
    parts = s3.list_parts(Bucket=BUCKET, Key=key, UploadId=uid)
    pbytes = sum(p['Size'] for p in parts.get('Parts', []))
    status = 'all files verified on HF' if need and not missing else f'{len(missing)}/{len(need)} files not verified on HF yet'
    print(f'{token} step {step} {mode} on {h}: {len(need)} files, {status}; parts {pbytes / 1e6:.0f} MB')
    if missing or not need or not APPLY:
        continue
    s3.abort_multipart_upload(Bucket=BUCKET, Key=key, UploadId=uid)
    still = [x for x in s3.list_multipart_uploads(Bucket=BUCKET, Prefix=key).get('Uploads', []) if x['UploadId'] == uid]
    rec = {'aborted_multipart_upload': key, 'upload_id': uid, 'initiated': u['Initiated'].isoformat(), 'part_bytes': pbytes,
           'confirmed_gone': not still, 'hf_verified_files': {p: verified[p] for p in need}, 'source_host': h,
           'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(RECEIPTS, 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')
    print('  ABORTED' + ('' if not still else ' (still listed!)'))
