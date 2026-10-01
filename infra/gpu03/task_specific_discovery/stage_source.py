"""Create one immutable, independently hashed source snapshot on gpu-03."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT='/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105'
LOCAL=Path('artifacts/gpu03-task-specific-discovery-20260921T165105Z')
def main():
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    assert a.name.replace('-','').isalnum()
    destination=ROOT+'/source-'+a.name
    files={str(x):x for x in Path('infra/gpu03').rglob('*.py') if '__pycache__' not in x.parts}
    # The sandbox imports the existing repository evaluator and prompt generators.
    # Include only source, never credentials, data payloads or a live worktree.
    files.update({str(x):x for x in Path('src').rglob('*.py') if '__pycache__' not in x.parts})
    files['scripts/run_data_process.py']=Path('scripts/run_data_process.py')
    files['metadata/source_manifest.json']=Path('artifacts/reward_hacking/dataset_freeze/checkpoint_60_outcome_presence_frozen_20260907_090259_v2/originals/existing_source_manifest.json')
    files['metadata/EXPERIMENTAL_DESIGN.md']=Path('output/task_specific_direction_feasibility_20260921/EXPERIMENTAL_DESIGN.md')
    manifest={};buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for name,path in sorted(files.items()):
            data=path.read_bytes(); info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444
            tar.addfile(info,io.BytesIO(data));manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
        data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
        info=tarfile.TarInfo('SOURCE_FILES.json');info.size=len(data);info.mode=0o444
        tar.addfile(info,io.BytesIO(data))
    package=buffer.getvalue();sha=hashlib.sha256(package).hexdigest()
    archive=LOCAL/('source-'+a.name+'.tar.gz');assert not archive.exists();archive.write_bytes(package)
    receiver='''import hashlib,io,json,os,pathlib,socket,sys,tarfile
assert socket.gethostname()=='gpu-03' and os.getuid()==3357
payload=sys.stdin.buffer.read(16*1024**2+1);assert len(payload)<=16*1024**2
assert hashlib.sha256(payload).hexdigest()==SHA
dest=pathlib.Path(DEST);assert not dest.exists();dest.mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tf:
 for item in tf:
  rel=pathlib.PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
  out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
  data=tf.extractfile(item).read();assert len(data)==item.size
  with out.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  out.chmod(0o444)
m=json.loads((dest/'SOURCE_FILES.json').read_text())
for name,b in m.items():
 p=dest/name;assert p.stat().st_size==b['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==b['sha256']
for p in sorted(dest.rglob('*'),reverse=True):
 if p.is_dir():p.chmod(0o555)
dest.chmod(0o555)
print(json.dumps({'status':'source_independently_verified','stage':str(dest),'archive_sha256':SHA,'files':len(m)}))
'''
    receiver='SHA='+repr(sha)+'\nDEST='+repr(destination)+'\n'+receiver
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-03','python3 -c '+shlex.quote(receiver)],input=package,capture_output=True,timeout=120)
    receipt={'stage':destination,'archive_sha256':sha,'files':manifest,'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()}
    (LOCAL/('SOURCE_'+a.name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(r.stdout.decode(),end='');print(r.stderr.decode(),end='');assert r.returncode==0

if __name__=='__main__':main()
