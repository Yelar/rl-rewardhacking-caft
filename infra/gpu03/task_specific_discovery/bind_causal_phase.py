"""Bind prepared causal tasks to currently idle gpu-03 devices; never launch."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from .dispatch_phase import ROOT, PYTHON
from . import fit_worker as io, resource_guard as guard


def require(ok, message):
    if not ok:
        raise ValueError(message)


def idle_devices(inventory, processes):
    observed, uuids = {}, set()
    for line in inventory.splitlines():
        fields = [x.strip() for x in line.split(',')]
        require(len(fields) == 6, 'Malformed device inventory')
        index, uuid, name, free, used, util = fields
        require(index in {str(i) for i in range(8)}
                and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid),
                'Invalid device identity')
        require(name == 'NVIDIA RTX 5000 Ada Generation', 'Unexpected gpu-03 hardware')
        require(index not in observed and uuid not in uuids, 'Duplicate device inventory')
        require(all(v.isdecimal() for v in (free, used, util)) and int(util) <= 100,
                'Malformed device utilization')
        observed[index] = (uuid, int(free), int(used), int(util))
        uuids.add(uuid)
    require(observed, 'Missing device inventory')
    occupied, seen = set(), set()
    for line in processes.splitlines():
        if not line.strip():
            continue
        fields = [x.strip() for x in line.split(',')]
        require(len(fields) == 2 and fields[0] in uuids and fields[1].isdecimal()
                and int(fields[1]) > 0, 'Malformed compute process inventory')
        pair = (fields[0], int(fields[1]))
        require(pair not in seen, 'Duplicate compute process inventory')
        seen.add(pair)
        occupied.add(fields[0])
    return {index: uuid for index, (uuid, free, used, util) in observed.items()
            if free >= 32000 and used <= 64 and util <= 1 and uuid not in occupied}


def manifest(phase, phase_path, phase_ref, source_files, devices, now):
    root = Path(ROOT); stage = phase['source_root']
    require(phase['schema']=='task_specific_causal_phase_inputs_v1'
            and phase['run_token']==root.name and phase['stage'] in ('qualification','pilot'),
            'Wrong causal task scope')
    require(Path(stage).parent==root and Path(phase_path).parent.parent==root,
            'Inputs must stay in this run')
    tasks = phase['tasks']
    require(1 <= len(tasks) <= 8 and len(devices) >= len(tasks), 'Insufficient safely idle GPUs')
    devices = {k:devices[k] for k in sorted(devices,key=int)[:len(tasks)]}
    require(sum(t['requests'] for t in tasks)==phase['expected_new_requests']
            and phase['expected_new_requests']+phase['expected_reused_requests']==phase['expected_total_requests'],
            'Task request counts differ')
    limit = 312 if phase['stage']=='qualification' else 15168
    require(0<phase['expected_total_requests']<=limit, 'Scientific request ceiling exceeded')
    bound = {stage+'/'+name:value for name,value in source_files.items()}
    for name,value in phase['bound_files'].items():
        require(name not in bound or bound[name]==value,'Conflicting input identity')
        bound[name]=value
    bound[str(phase_path)] = phase_ref
    env = {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','PYTHONPATH':stage,
        'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    workers=[]
    for slot,(gpu,task) in enumerate(zip(devices,tasks)):
        require(task['lane']==slot and 0<task['requests']<=2048, 'Invalid causal lane')
        require(bound.get(task['task']['path']) == {k:task['task'][k] for k in ('size_bytes','sha256')},
                'Worker task not bound')
        require(task['argv'] == [PYTHON, '-B', '-m',
                'infra.gpu03.task_specific_discovery.causal_worker', '--task',
                task['task']['path'], '--sha256', task['task']['sha256']],
                'Worker command differs from bound causal task')
        workers.append({'name':f'gpu_{gpu}','gpu_id':int(gpu),'cpu_ids':[16+2*slot,17+2*slot],
            'argv':task['argv'],'cwd':stage,'env':dict(env,CUDA_VISIBLE_DEVICES=devices[gpu])})
    phase_name = 'causal-'+phase['stage']+'-v1'
    duration = 5400 if phase['stage']=='qualification' else 14400
    result={'schema':guard.SCHEMA,'host':'gpu-03','uid':3357,'run_token':root.name,
        'source_root':stage,'runtime':str(root)+'-'+phase_name+'-runtime','disk_path':str(root),
        'deadline_epoch':now+duration,'runtime_seconds':duration+120,'unit':root.name+'-'+phase_name,
        'entry_receipt':str(root/(phase_name+'-entry.json')),
        'kernel_limits':{'memory_max':192*1024**3,'tasks_max':512,'cpu_cores':2*len(tasks)},
        'gpu_uuids':devices,'workers':workers,'bound_files':bound,
        'qualification_contract':phase['qualification_contract'],
        'phase_inputs':{'path':str(phase_path),**phase_ref},
        'expected_new_requests':phase['expected_new_requests'],
        'expected_reused_requests':phase['expected_reused_requests'],
        'expected_total_requests':phase['expected_total_requests'],
        'science':{'generation':True,'evaluation':False,'training':False,'stage':phase['stage'],
                   'batch_size':8,'gpu_allocator_fraction':.65,'prompt_max':1536,'completion_max':1536,
                   'checkpoint':'original Ada M60','profile':phase['profile']}}
    guard.validate_manifest(result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase-inputs',required=True)
    parser.add_argument('--sha256',required=True);args=parser.parse_args()
    require(socket.gethostname()=='gpu-03' and os.getuid()==3357 and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Binding requires accepted host owner CPU-only environment')
    path=Path(args.phase_inputs);require(io.sha256(path)==args.sha256,'Phase inputs changed')
    phase=json.loads(path.read_text());source=Path(phase['source_root'])
    files=json.loads((source/'SOURCE_FILES.json').read_text())
    for name,value in {**{str(source/n):b for n,b in files.items()},**phase['bound_files']}.items():
        io.verify_file(name,value)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    devices=idle_devices(inventory,processes)
    result=manifest(phase,path,io.binding(path),files,devices,time.time())
    output=path.parent/'RESOURCE_MANIFEST.json'
    io.write_json(output,result)
    print(json.dumps({'path':str(output),**io.binding(output),'gpu_uuids':result['gpu_uuids'],
                     'requests':result['expected_new_requests']}))


if __name__=='__main__':main()
