"""Restore exact project-owned objects directly to gpu-03; URLs stay in memory.

Run on the owner Mac with its existing AWS session. The SSH receiver gets only
short-lived, exact-object read URLs, never AWS credentials. No payload on Mac.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

BUCKET = os.environ['RH_S3_ROOT'].removeprefix('s3://').rstrip('/')  # project bucket, from .env
PREFIX = ('h100-workstations/codex-sky-h100-block-20260908-1530/outputs/'
          'frozen-inputs-final-20260909-100000/artifacts/reward_hacking/dataset_freeze/'
          'checkpoint_60_outcome_presence_frozen_20260907_090259_v2/')
ROOT = '/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105'
FILES = [
    ('split_manifest.json', 'split_manifest.json', 79422, '99006592f434f73afcad3f641c759786be5e206a254c3755e4dec54e35984b06'),
    ('provenance/dataset.jsonl', 'dataset.jsonl', 17995931, 'bdbba14d0632ab298f0e6116ad76bce75ae2361b79a6a7e046b23b1e15b7936f'),
    ('splits/direction_fit.jsonl', 'direction_fit.jsonl', 2155682808, '16ba621ec034d5db48077865bb0f0be0d7ce2cb8b6a36b298bbb311a4945562a'),
]
RECEIVER = r'''
import datetime, hashlib, json, os, pathlib, shutil, socket, sys, urllib.request
request=json.load(sys.stdin)
assert socket.gethostname()=='gpu-03' and os.getuid()==3357
root=pathlib.Path(request['root']); assert str(root)=='/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105'
assert not root.is_symlink()
root.mkdir(mode=0o700,exist_ok=True)
dest=root/'inputs'; dest.mkdir(mode=0o700,exist_ok=True); assert not dest.is_symlink()
assert shutil.disk_usage(dest).free > 128*1024**3 + sum(x['size'] for x in request['files'])
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
for item in request['files']:
 p=dest/item['name']; assert p.parent==dest and not p.is_symlink()
 if p.exists():
  assert p.is_file() and p.stat().st_size==item['size'] and digest(p)==item['sha256']
  mode='existing_independently_rehashed'
 else:
  part=p.with_name(p.name+'.partial'); assert not part.exists()
  h=hashlib.sha256(); n=0
  # HTTP exceptions are deliberately reported without the signed URL.
  try:
   with urllib.request.urlopen(item['url'],timeout=30) as response, part.open('xb') as out:
    while True:
     chunk=response.read(8*1024**2)
     if not chunk:break
     n+=len(chunk); assert n<=item['size']; h.update(chunk); out.write(chunk)
    out.flush(); os.fsync(out.fileno())
  except Exception as error:
   print(json.dumps({'error_type':type(error).__name__,'name':item['name'],'bytes':n}),flush=True)
   raise SystemExit(2)
  assert n==item['size'] and h.hexdigest()==item['sha256']
  assert digest(part)==item['sha256']
  part.chmod(0o444); os.rename(part,p)
  d=os.open(str(dest),os.O_RDONLY); os.fsync(d); os.close(d)
  mode='downloaded_and_independent_full_readback'
 print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(p),'size':item['size'],'sha256':item['sha256'],'status':mode}),flush=True)
'''

def main():
    import boto3
    from botocore.config import Config
    config = Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 2})
    session = boto3.Session()
    identity = session.client('sts', region_name='us-east-1', config=config).get_caller_identity()
    assert identity['Account'] == os.environ['RH_AWS_ACCOUNT']
    s3 = session.client('s3', region_name='us-east-1', config=config)
    items, identities = [], []
    for suffix, name, size, sha in FILES:
        key = PREFIX + suffix
        head = s3.head_object(Bucket=BUCKET, Key=key)
        assert head['ContentLength'] == size
        identities.append({'key':key, 'size':size, 'sha256':sha, 'etag':head['ETag'],
                           'version':head.get('VersionId'), 'metadata':head.get('Metadata')})
        items.append({'name':name, 'size':size, 'sha256':sha,
                      'url':s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':key},ExpiresIn=3600)})
    evidence = Path('artifacts/gpu03-task-specific-discovery-20260921T165105Z/host')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (evidence/f'RESTORE_S3_IDENTITIES_{stamp}.json').write_text(json.dumps(identities,indent=2)+'\n')
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-03','python3 -c '+shlex.quote(RECEIVER)]
    p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=p.communicate(json.dumps({'root':ROOT,'files':items}).encode(),timeout=1800)
    (evidence/f'RESTORE_RECEIPTS_{stamp}.jsonl').write_bytes(stdout)
    print(stdout.decode(),end='')
    # Receiver errors do not print URLs. SSH errors contain no object credentials.
    print(json.dumps({'returncode':p.returncode,'ssh_stderr':stderr.decode()[:1000]}))
    assert p.returncode==0

if __name__=='__main__':main()
