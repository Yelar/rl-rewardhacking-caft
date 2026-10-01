"""Independently verify S3 checkpoint backups whose upload was never confirmed.

For every key in the receipts file that has no verified receipt, download the object again (streamed, nothing
written to disk), hash it, and compare size and SHA-256 with the upload stream recorded in the failed receipt.
A match appends a verified receipt (mode "reverify"). A missing object is left for s3_backup_w2q.py to upload;
a mismatch is reported and never overwritten. Read-only on S3.
Usage: verify_s3_receipts.py <receipts.jsonl>"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
BUCKET = os.environ['RH_S3_ROOT'].rstrip('/') + '/hpc-caft/bestcase-caft-w2q-20260930'  # project bucket, from .env
RECEIPTS = Path(sys.argv[1])
def log(msg): print(time.strftime('%H:%M:%SZ', time.gmtime()), msg, flush=True)
def head(key):
    b, p = BUCKET.split('/')[2], '/'.join(BUCKET.split('/')[3:])
    r = subprocess.run(['aws', 's3api', 'head-object', '--bucket', b, '--key', f'{p}/{key}'], capture_output=True, text=True)
    if r.returncode == 0: return json.loads(r.stdout)
    return None if 'Not Found' in r.stderr or '404' in r.stderr else 'error'
def download_sha(key):
    p = subprocess.Popen(['aws', 's3', 'cp', f'{BUCKET}/{key}', '-', '--only-show-errors'], stdout=subprocess.PIPE)
    h, n = hashlib.sha256(), 0
    for chunk in iter(lambda: p.stdout.read(1 << 20), b''): h.update(chunk); n += len(chunk)
    return p.wait(), h.hexdigest(), n
rs = [json.loads(l) for l in RECEIPTS.read_text().splitlines() if l.strip()]
verified = {r['key'] for r in rs if r.get('verified')}
pending = {}
for r in rs:
    if not r.get('verified') and r['key'] not in verified: pending[r['key']] = r   # latest failed receipt per key
log(f'{len(pending)} keys without a verified receipt')
for key, r in sorted(pending.items()):
    for attempt in range(4):
        hd = head(key)
        if hd is None: log(f'MISSING on S3 (left for the backup loop to upload): {key}'); break
        if hd == 'error': time.sleep(15 * (attempt + 1)); continue
        rc, dsha, n = download_sha(key)
        if rc != 0 or n != hd.get('ContentLength'):   # a cut-off stream is a failed download, not a mismatch
            log(f'download incomplete (rc={rc}, {n} of {hd.get("ContentLength")} bytes), retrying: {key}')
            time.sleep(15 * (attempt + 1)); continue
        ok = dsha == r['sha256'] and n == r['bytes']
        rec = {'key': key, 'host': r.get('host'), 'token': r.get('token'), 'step': r.get('step'), 'mode': r.get('mode'),
               'bytes': r['bytes'], 'sha256': r['sha256'], 'download_sha256': dsha, 'verified': ok, 'check': 'reverify',
               'etag': hd.get('ETag'), 'version': hd.get('VersionId'), 'inventory_sha256': r.get('inventory_sha256'),
               'files': r.get('files'), 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        with open(RECEIPTS, 'a') as f: f.write(json.dumps(rec) + '\n')
        log(('VERIFIED ' if ok else 'MISMATCH (left untouched) ') + f'{key} {n / 1e6:.0f} MB')
        break
    else:
        log(f'gave up after 4 attempts: {key}')
