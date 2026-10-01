"""Stage a new, immutable source snapshot for the approved seed-6008 pilot.

No model, GPU allocation, evaluation, scientific claim or dispatch is performed.
The existing source-staging inventory is retained; only the destination changes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT = '/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-pilot-20260922-113907'
LOCAL = Path('artifacts/gpu03-task-specific-pilot-20260922T113907Z')


def package_source():
    files = {str(p): p for p in Path('infra/gpu03').rglob('*.py') if '__pycache__' not in p.parts}
    files.update({str(p): p for p in Path('src').rglob('*.py') if '__pycache__' not in p.parts})
    files['scripts/run_data_process.py'] = Path('scripts/run_data_process.py')
    files['metadata/source_manifest.json'] = Path('artifacts/reward_hacking/dataset_freeze/checkpoint_60_outcome_presence_frozen_20260907_090259_v2/originals/existing_source_manifest.json')
    files['metadata/EXPERIMENTAL_DESIGN.md'] = Path('output/task_specific_direction_feasibility_20260921/EXPERIMENTAL_DESIGN.md')
    manifest, buffer = {}, io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tf:
        for name, path in sorted(files.items()):
            if path.is_symlink():
                raise ValueError('Source symlink: ' + str(path))
            data = path.read_bytes()
            info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o444
            tf.addfile(info, io.BytesIO(data))
            manifest[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
        data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        info = tarfile.TarInfo('SOURCE_FILES.json'); info.size = len(data); info.mode = 0o444
        tf.addfile(info, io.BytesIO(data))
    payload = buffer.getvalue()
    if len(payload) > 16 << 20:
        raise ValueError('Source archive exceeds bounded transfer')
    return payload, manifest


def receiver(root, destination, digest):
    return 'ROOT=' + repr(root) + '\nDEST=' + repr(destination) + '\nSHA=' + repr(digest) + '\n' + '''
import hashlib,io,json,os,pathlib,socket,sys,tarfile
assert socket.gethostname()=='gpu-03' and os.getuid()==3357
root=pathlib.Path(ROOT);dest=pathlib.Path(DEST)
assert root.parent==pathlib.Path('/scratch/yelarys.yertaiuly/codex_runs')
assert root.name=='codex-task-specific-pilot-20260922-113907'
assert dest.parent==root and dest.name.startswith('source-')
assert root.parent.resolve()==root.parent and not root.is_symlink()
payload=sys.stdin.buffer.read((16<<20)+1)
assert len(payload)<=(16<<20) and hashlib.sha256(payload).hexdigest()==SHA
if not root.exists():root.mkdir(mode=0o700)
assert root.is_dir() and root.stat().st_uid==3357 and root.resolve()==root
assert not dest.exists() and not dest.is_symlink()
dest.mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tf:
 seen=set()
 for item in tf:
  rel=pathlib.PurePosixPath(item.name)
  assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts and item.name not in seen
  seen.add(item.name);out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
  data=tf.extractfile(item).read();assert len(data)==item.size
  with out.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  out.chmod(0o444)
inventory=json.loads((dest/'SOURCE_FILES.json').read_bytes())
assert seen==set(inventory)|{'SOURCE_FILES.json'}
for name,b in inventory.items():
 p=dest/name
 assert not p.is_symlink() and p.is_file() and p.stat().st_size==b['size_bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==b['sha256']
for p in sorted(dest.rglob('*'),reverse=True):
 if p.is_dir():p.chmod(0o555)
dest.chmod(0o555)
print(json.dumps({'status':'source_independently_verified','stage':str(dest),'archive_sha256':SHA,'files':len(inventory)}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    if not re.fullmatch(r'successor-v[1-9][0-9]*', args.name):
        raise ValueError('Expected an explicit successor version')
    destination = ROOT + '/source-' + args.name
    archive = LOCAL / ('source-' + args.name + '.tar.gz')
    receipt_path = LOCAL / ('SOURCE_' + args.name + '.json')
    if archive.exists() or receipt_path.exists():
        raise FileExistsError('Staging attempt already exists; inspect before any further action')
    payload, manifest = package_source()
    sha = hashlib.sha256(payload).hexdigest()
    with archive.open('xb') as f:
        f.write(payload)
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-03',
               'python3 -c ' + shlex.quote(receiver(ROOT, destination, sha))]
    try:
        result = subprocess.run(command, input=payload, capture_output=True, timeout=120)
        receipt = {'stage': destination, 'archive_sha256': sha, 'files': manifest,
                   'returncode': result.returncode, 'stdout': result.stdout.decode(), 'stderr': result.stderr.decode()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        receipt = {'stage': destination, 'archive_sha256': sha, 'files': manifest,
                   'status': 'staging_uncertain_do_not_replay', 'error': type(exc).__name__}
    with receipt_path.open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
    if receipt.get('returncode') != 0:
        raise RuntimeError('Staging did not return verified success; inspect preserved evidence')


if __name__ == '__main__':
    main()
