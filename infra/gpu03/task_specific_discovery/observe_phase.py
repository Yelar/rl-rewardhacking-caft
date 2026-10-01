"""Read-only bounded phase observation; retain compact independent local receipt."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
ROOT='/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105'
def main():
    p=argparse.ArgumentParser();p.add_argument('phase');a=p.parse_args()
    assert a.phase.replace('-','').isalnum()
    code='''import hashlib,json,pathlib,subprocess,time
root=pathlib.Path(ROOT);runtime=pathlib.Path(ROOT+'-'+PHASE+'-runtime');unit=root.name+'-'+PHASE
out={'at':time.time(),'phase':PHASE,'files':{}}
paths=[root/(PHASE+'-entry.json'),runtime/'SUPERVISOR_RESULT.json',runtime/'workers.jsonl',root/(unit+'.log')]
paths+=sorted(runtime.glob('*.log')) if runtime.is_dir() else []
for p in paths:
 if not p.exists():out['files'][str(p)]={'exists':False};continue
 size=p.stat().st_size
 with p.open('rb') as f:
  if size>40000:f.seek(-40000,2)
  data=f.read(40000)
 out['files'][str(p)]={'exists':True,'size':size,'complete':size<=40000,'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')}
if (runtime/'resources.jsonl').exists():
 with (runtime/'resources.jsonl').open('rb') as f:
  f.seek(max(0,(runtime/'resources.jsonl').stat().st_size-20000));lines=f.read().splitlines()
 out['last_resource_observation']=json.loads(lines[-1])
s=subprocess.run(['systemctl','--user','show',unit+'.service','-p','ActiveState','-p','SubState','-p','MainPID','-p','Result','-p','ExecMainStatus','-p','MemoryPeak','-p','CPUUsageNSec'],capture_output=True,text=True,timeout=15)
out['service']={'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}
print(json.dumps(out))
'''
    code='ROOT='+repr(ROOT)+'\nPHASE='+repr(a.phase)+'\n'+code
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-03','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=45)
    assert r.returncode==0,r.stderr
    result=json.loads(r.stdout)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=Path('artifacts/gpu03-task-specific-discovery-20260921T165105Z/host')/('OBSERVE_'+a.phase+'_'+stamp+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':str(path),'service':result['service'],'files':{k:dict(v, text=v.get('text','')[-1400:]) for k,v in result['files'].items()},'last_resources':result.get('last_resource_observation')},indent=2))

if __name__=='__main__':main()
