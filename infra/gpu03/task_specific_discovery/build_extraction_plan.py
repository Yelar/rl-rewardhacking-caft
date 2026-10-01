"""Prepare (never launch) exact whole-problem GPU shards on the accepted host."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from .dispatch_phase import ROOT, PYTHON

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage-name',required=True);p.add_argument('--support',required=True);p.add_argument('--phase',required=True);a=p.parse_args()
    local=Path('artifacts/gpu03-task-specific-discovery-20260921T165105Z')
    stage=json.loads((local/('SOURCE_'+a.stage_name+'.json')).read_text())
    inp=json.loads((local/'host/INPUTS_exact_bindings_20260921.json').read_text())
    model={v['path']:v['expected'] for v in inp['base_file_sizes']}
    ck='/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/inputs/checkpoint_60/lora_adapter'
    model.update({ck+'/adapter_model.safetensors':{'sha256':'2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0','size_bytes':264308864},
                  ck+'/adapter_config.json':{'sha256':'8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae','size_bytes':1127}})
    payload={'root':ROOT,'python':PYTHON,'stage':stage['stage'],'source_files':stage['files'],'model_files':model,
             'checkpoint':ck,'support':a.support,'phase':a.phase}
    code=r'''
import collections,hashlib,json,os,pathlib,socket,subprocess,sys,time
r=json.load(sys.stdin);assert socket.gethostname()=='gpu-03' and os.getuid()==3357
root=pathlib.Path(r['root']);assert root.name=='codex-task-specific-discovery-20260921-165105'
support=root/r['support'];assert support.parent==root and support.is_dir() and not support.is_symlink()
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 data=(json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444);return {'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
artifact=support/'ARTIFACT_MANIFEST.json';assert artifact.is_file() and not artifact.is_symlink()
am=json.loads(artifact.read_text());assert am['status']=='support_ready'
assert {'SUMMARY.json','prepared_records.jsonl'} <= set(am['files']), 'Consumed support files are not sealed'
for name,b in am['files'].items():
 p=support/name;assert p.parent==support and p.is_file() and not p.is_symlink() and p.stat().st_size==b['size_bytes'] and hashfile(p)==b['sha256']
summary=json.loads((support/'SUMMARY.json').read_text());assert summary['status']=='support_ready'
assert set(summary['eligible_paired_problems'])=={'gt_correct','gt_incorrect'}
assert min(summary['eligible_paired_problems'].values())>=30
rows=[json.loads(x) for x in (support/'prepared_records.jsonl').read_text().splitlines()];assert 1<=len(rows)<=256
assert len(rows)==summary['selected_records'] and len({x['record_id'] for x in rows})==len(rows) and len({x['record_index'] for x in rows})==len(rows), 'Prepared support count or unique identities differ'
assert all(x['problem_split']=='direction_fit' and x['model_revision']=='1cfa9a7208912126459214e8b04321603b3df60c' and x['checkpoint_sha256']=='2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0' for x in rows), 'Prepared support contains held-out or different-model records'
gpus={}
text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
for line in text.splitlines():
 index,uuid,name,free,used,util=[x.strip() for x in line.split(',')]
 assert name=='NVIDIA RTX 5000 Ada Generation'
 if int(free)>=32000 and int(used)<=64 and int(util)<=1:gpus[index]=uuid
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
occupied={x.split(',')[0].strip() for x in apps.splitlines() if x.strip()}
gpus={k:v for k,v in gpus.items() if v not in occupied};assert gpus
groups=collections.defaultdict(list)
for row in rows:groups[row['problem_id_key']].append(row)
assert len(groups)>=len(gpus)
shards={g:[] for g in gpus};loads={g:0 for g in gpus}
for pid in sorted(groups,key=lambda p:(-sum(x['sequence_token_count'] for x in groups[p]),p)):
 g=min(gpus,key=lambda g:(loads[g],int(g)));shards[g].extend(groups[pid]);loads[g]+=sum(x['sequence_token_count'] for x in groups[pid])
out=root/r['phase'];assert out.parent==root;out.mkdir(mode=0o700)
stage=r['stage'];bound={stage+'/'+n:b for n,b in r['source_files'].items()};bound.update(r['model_files'])
bound.update({str(support/n):b for n,b in am['files'].items()})
bound[str(artifact)]={'sha256':hashfile(artifact),'size_bytes':artifact.stat().st_size}
workers=[];task_reports=[]
env={'PATH':'/usr/bin:/bin','PYTHONPATH':stage,'LANG':'C.UTF-8','PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
for slot,g in enumerate(sorted(gpus,key=int)):
 records=out/('prepared_gpu_'+g+'.jsonl');raw=b''.join((json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode() for x in sorted(shards[g],key=lambda v:v['record_index']))
 with records.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 records.chmod(0o444);rb={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
 task={'schema':'task_specific_fixed2176_extraction_v1','cache_role':'auxiliary','padded_sequence_length':2176,'pad_token_id':151643,'deadline_seconds':7200,
       'model_snapshot':'/scratch/yelarys.yertaiuly/codex_runs/codex-ada8-matched-20260916-1540-inputs/base','checkpoint':r['checkpoint'],'prepared_rows':str(records),'output':str(out/('gpu_'+g)),
       'bound_files':{**r['model_files'],str(records):rb},'generation':False,'evaluation':False,'training':False}
 tp=out/('task_gpu_'+g+'.json');tb=write(tp,task);bound[str(tp)]=tb;bound[str(records)]=rb
 workers.append({'name':'gpu_'+g,'gpu_id':int(g),'argv':[r['python'],'-B','-m','infra.gpu03.task_specific_discovery.extract_worker','--task',str(tp),'--sha256',tb['sha256']],
                 'cwd':stage,'env':dict(env,CUDA_VISIBLE_DEVICES=gpus[g]),'cpu_ids':[16+2*slot,17+2*slot]})
 task_reports.append({'gpu':g,'records':len(shards[g]),'problems':len({x['problem_id_key'] for x in shards[g]}),'tokens':loads[g],'task':str(tp),**tb})
phase=r['phase'];m={'schema':'gpu03_task_specific_resource_guard_v1','host':'gpu-03','uid':3357,'run_token':root.name,'source_root':stage,'runtime':str(root)+'-'+phase+'-runtime','disk_path':str(root),
 'deadline_epoch':time.time()+7200,'runtime_seconds':7320,'unit':root.name+'-'+phase,'entry_receipt':str(root/(phase+'-entry.json')),
 'kernel_limits':{'memory_max':192*1024**3,'tasks_max':512,'cpu_cores':16},'bound_files':bound,'gpu_uuids':gpus,'workers':workers,
 'science':{'base_revision':'1cfa9a7208912126459214e8b04321603b3df60c','checkpoint':'original Ada M60','models':['M0','M60'],'layers':36,'batch':1,'storage':'native BF16 + FP32 differences','fixed_length':2176,'generation':False,'evaluation':False,'training':False},
 'support_summary_sha256':hashfile(support/'SUMMARY.json'),'source_artifact_manifest_sha256':hashfile(support/'ARTIFACT_MANIFEST.json'),
 'expected_records':len(rows),'prepared_global_rows':str(support/'prepared_records.jsonl'),'shards':task_reports}
print(json.dumps(m))
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-03','python3 -c '+shlex.quote(code)],input=json.dumps(payload),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    manifest=json.loads(result.stdout);path=local/(a.phase.upper().replace('-','_')+'.json')
    with path.open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
    import hashlib
    print(json.dumps({'manifest':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'GPUs':manifest['gpu_uuids'],'shards':manifest['shards']}))

if __name__=='__main__':main()
