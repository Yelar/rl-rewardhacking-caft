"""Dispatch one reviewed phase exactly once under an independent user service."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
ROOT='/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-discovery-20260921-165105'
PYTHON='/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args()
    local=Path(a.manifest);data=local.read_bytes();m=json.loads(data)
    digest=hashlib.sha256(data).hexdigest()
    remote=ROOT+'/plans/'+local.name
    limits=m['kernel_limits'];stage=m['source_root']
    command=['systemd-run','--user','--unit='+m['unit'],'--service-type=exec',
       '--property=RuntimeMaxSec='+str(m['runtime_seconds']),
       '--property=MemoryMax='+str(limits['memory_max']),'--property=MemorySwapMax=0',
       '--property=TasksMax='+str(limits['tasks_max']),
       '--property=CPUQuota='+str(limits['cpu_cores']*100)+'%',
       '--property=KillMode=control-group','--property=TimeoutStopSec=60',
       '--property=NoNewPrivileges=yes','--property=UMask=0077',
       '--property=MemoryAccounting=yes','--property=TasksAccounting=yes','--property=CPUAccounting=yes',
       '--property=StandardOutput=append:'+ROOT+'/'+m['unit']+'.log',
       '--property=StandardError=inherit','--working-directory='+stage,
       '/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C.UTF-8','CUDA_VISIBLE_DEVICES=',
       'PYTHONPATH='+stage,'PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1',
       'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','TOKENIZERS_PARALLELISM=false',
       PYTHON,'-B','-m','infra.gpu03.task_specific_discovery.phase_entry','--manifest',remote,'--sha256',digest]
    code='''import hashlib,json,os,pathlib,socket,subprocess,sys
r=json.load(sys.stdin);assert socket.gethostname()=='gpu-03' and os.getuid()==3357
root=pathlib.Path(r['root']);assert root.is_dir() and not root.is_symlink()
plans=root/'plans';plans.mkdir(mode=0o700,exist_ok=True)
p=pathlib.Path(r['remote']);assert p.parent==plans and not p.exists()
data=r['data'].encode();assert hashlib.sha256(data).hexdigest()==r['sha256']
with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
p.chmod(0o444)
attempt=root/(r['unit']+'_DISPATCH_ATTEMPT.json')
with attempt.open('x') as f:json.dump({'command':r['command'],'manifest_sha256':r['sha256']},f);f.flush();os.fsync(f.fileno())
result=subprocess.run(r['command'],capture_output=True,text=True,timeout=30)
state=subprocess.run(['systemctl','--user','show',r['unit']+'.service','-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID','-p','Result','-p','ExecMainStatus','-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec','-p','TasksMax','-p','RuntimeMaxUSec','-p','ControlGroup'],capture_output=True,text=True,timeout=20)
receipt={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'state_returncode':state.returncode,'state':state.stdout,'state_error':state.stderr}
with (root/(r['unit']+'_DISPATCH_RESULT.json')).open('x') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt));sys.exit(result.returncode)
'''
    payload={'root':ROOT,'remote':remote,'sha256':digest,'data':data.decode(),'command':command,'unit':m['unit']}
    attempt=local.with_suffix('.local_attempt.json')
    with attempt.open('x') as f:json.dump({'manifest_sha256':digest,'command':command},f,indent=2)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-03','python3 -c '+shlex.quote(code)],input=json.dumps(payload),capture_output=True,text=True,timeout=90)
    receipt={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    local.with_suffix('.local_result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));assert result.returncode==0

if __name__=='__main__':main()
