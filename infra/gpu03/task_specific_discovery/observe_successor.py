"""Read-only compact receipts for the approved successor; never replay dispatch."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = '/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-pilot-20260922-113907'
LOCAL = Path('artifacts/gpu03-task-specific-pilot-20260922T113907Z/host')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.phase):
        raise ValueError('Invalid phase')
    code = 'ROOT=' + repr(ROOT) + '\nPHASE=' + repr(args.phase) + '\n' + '''
import hashlib,json,os,pathlib,socket,subprocess,time
assert socket.gethostname()=='gpu-03' and os.getuid()==3357
root=pathlib.Path(ROOT);runtime=pathlib.Path(ROOT+'-'+PHASE+'-runtime');unit=root.name+'-'+PHASE
out={'at':time.time(),'phase':PHASE,'files':{}}
paths=[root/(PHASE+'-entry.json'),runtime/'SUPERVISOR_RESULT.json',runtime/'workers.jsonl',
       runtime/'gpu_rejections.jsonl',root/(unit+'.log')]
paths+=sorted(runtime.glob('*.log')) if runtime.is_dir() else []
for p in paths:
 if not p.exists():out['files'][str(p)]={'exists':False};continue
 assert p.resolve()==p and p.is_file()
 size=p.stat().st_size
 with p.open('rb') as f:
  if size>40000:f.seek(-40000,2)
  data=f.read(40000)
 out['files'][str(p)]={'exists':True,'size':size,'complete':size<=40000,
                     'sha256_of_read_bytes':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')}
p=runtime/'resources.jsonl'
if p.exists():
 assert p.resolve()==p and p.is_file()
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-20000));lines=f.read().splitlines()
 if lines:out['last_resource_observation']=json.loads(lines[-1])
s=subprocess.run(['systemctl','--user','show',unit+'.service','-p','ActiveState','-p','SubState',
 '-p','MainPID','-p','Result','-p','ExecMainStatus','-p','MemoryPeak','-p','CPUUsageNSec'],capture_output=True,text=True,timeout=15)
out['service']={'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}
print(json.dumps(out))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-03',
                             'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr)
    report = json.loads(result.stdout)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = LOCAL / ('OBSERVE_' + args.phase + '_' + stamp + '.json')
    with path.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    compact = {k: {**v, 'text': v.get('text', '')[-2200:]} for k, v in report['files'].items()}
    print(json.dumps({'receipt': str(path), 'service': report['service'], 'files': compact,
                      'last_resources': report.get('last_resource_observation')}, indent=2))


if __name__ == '__main__':
    main()
