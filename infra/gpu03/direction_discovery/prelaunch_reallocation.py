"""Transfer only an unlaunched validation-TF reservation to fewer idle GPUs.

No model or workload launch. Creating the old control directory blocks its
immutable launcher's exclusive mkdir; old manifests/tasks remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import time

CRITICAL=('infra/gpu03/direction_discovery/engine.py','infra/gpu03/direction_discovery/intervention.py',
          'infra/gpu03/activation_dataset/extract_triplet_raw.py','infra/gpu03/activation_dataset/extract_delta_activations.py')
PROOF_NAME='prelaunch_reallocation.json'
REFERENCE_NAME='prelaunch_reallocation_reference.json'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Missing or symlinked reallocation input')
    return json.loads(path.read_bytes())


def bound(m,path):
    path=Path(path);value=read(path);info=m['bound_files'].get(str(path),{})
    require(info.get('sha256')==sha(path) and info.get('size_bytes')==path.stat().st_size,'Changed reallocation bound input')
    return value


def original_plan(old_path,old):
    stage=Path(old_path).parent;path=stage/'input/request_plan.json';plan=bound(old,path)
    require(old.get('host')=='gpu-04' and old.get('owner')=='researcher' and old['scientific'].get('training') is False and
            old.get('phase')=='refinement_tf' and old['scientific'].get('phase_modes')==['tf']*len(old['workers']) and
            plan.get('mode')=='tf' and plan.get('phase')=='refinement' and
            plan.get('evaluation_partition')=='configuration_validation','Only original validation refinement TF may be reallocated')
    require(plan['master_plan_sha256']==old['scientific']['master_plan_sha256'],'Reallocation master plan differs')
    return path,plan


def absent_runtime(old):
    require(all(not Path(old[key]).exists() and not Path(old[key]).is_symlink() for key in ('output','runtime')),
            'A phase with runtime/output cannot be reallocated')


def validate_pending(old_path,old,proof_path):
    old_path=Path(old_path);stage=old_path.parent;proof_path=Path(proof_path);control=stage/'control'
    require(proof_path==control/PROOF_NAME and not control.is_symlink(),'Wrong reallocation barrier path')
    require(control.is_dir() and {p.name for p in control.iterdir()}=={PROOF_NAME},'Old launch evidence or unexpected control content blocks transfer')
    absent_runtime(old);old_plan_path,plan=original_plan(old_path,old);proof=read(proof_path)
    require(proof.get('schema_version')==1 and proof.get('status')=='retired_before_any_launch' and
            proof.get('old_manifest_path')==str(old_path) and proof.get('old_manifest_sha256')==sha(old_path) and
            proof.get('old_run_token')==old['run_token'] and proof.get('host')=='gpu-04' and
            proof.get('owner')=='researcher' and proof.get('request_plan_sha256')==sha(old_plan_path),
            'Reallocation receipt identity differs')
    destination=Path(proof['destination_stage'])
    require(destination.is_absolute() and destination.parent==stage.parent and destination!=stage and not destination.is_symlink() and
            re.fullmatch(r'codex-discovery-[a-z0-9-]+-stage',destination.name),'Unsafe replacement stage')
    ids=proof['successor_gpu_ids']
    require(isinstance(ids,list) and ids==sorted(set(ids)) and all(type(i) is int for i in ids) and
            set(ids)<set(old['gpu_ids']) and ids,'Reallocation must use a strict nonempty GPU subset')
    evidence=proof['absence_evidence']
    expected={'LoadState':'not-found','ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlGroup':''}
    require(evidence.get('systemd')==expected and evidence.get('owned_token_processes')==[] and
            evidence.get('control_preexisted') is False and evidence.get('output_exists') is False and evidence.get('runtime_exists') is False,
            'Reallocation lacks positive never-launched proof')
    require(type(proof.get('at')) in (int,float) and 0<proof['at']<1e12,'Invalid reallocation time')
    child_plan_path=destination/'input/request_plan.json';require(sha(child_plan_path)==sha(old_plan_path),'Replacement scientific requests changed')
    reference_path=destination/'input'/REFERENCE_NAME;reference=read(reference_path)
    require(reference=={'old_manifest_path':str(old_path),'old_manifest_sha256':sha(old_path),'proof_path':str(proof_path),'proof_sha256':sha(proof_path)},
            'Replacement reference differs from retirement proof')
    return {'destination_stage':str(destination),'receipt':proof,'bindings':[old_path,old_plan_path,proof_path,child_plan_path,reference_path]}


def validate_pair(old_path,old,new_path,new,pending):
    new_path=Path(new_path);proof=pending['receipt'];destination=Path(pending['destination_stage'])
    require(new_path==destination/'reviewed_manifest.json' and new.get('stage')==str(destination) and
            new.get('run_token')==destination.name.removesuffix('-stage'),'Wrong replacement identity')
    for key in ('host','owner','purpose','phase','python','runtime_versions','limits'):
        require(new.get(key)==old.get(key),'Replacement changed protocol/runtime: '+key)
    for key in ('master_plan_sha256','input_prepared_sha256','training'):
        require(new['scientific'].get(key)==old['scientific'].get(key),'Replacement changed scientific identity: '+key)
    require(new['scientific'].get('phase_modes')==['tf']*len(new['workers']) and new['gpu_ids']==proof['successor_gpu_ids'] and
            all(new['gpu_uuids'].get(str(g))==old['gpu_uuids'][str(g)] for g in new['gpu_ids']), 'Replacement GPU allocation differs')
    claim={'old_manifest_sha256':sha(old_path),'receipt_path':str(Path(old_path).parent/'control'/PROOF_NAME),
           'receipt_sha256':sha(Path(old_path).parent/'control'/PROOF_NAME)}
    require(new['scientific'].get('prelaunch_reallocation')==claim,'Replacement lacks exact transfer claim')
    bindings=list(pending['bindings'])
    for path in bindings:
        info=new['bound_files'].get(str(path),{})
        require(info.get('sha256')==sha(path) and info.get('size_bytes')==path.stat().st_size,'Replacement does not bind transfer proof')
    old_plan_path,plan=original_plan(old_path,old);newplan=bound(new,destination/'input/request_plan.json')
    require(newplan==plan,'Replacement changed request plan')
    expected={r['request_id']:r for r in plan['requests']};seen=set()
    old_task=bound(old,Path(old['workers'][0]['command'][3]))
    science_keys=('mode','model_snapshot','checkpoint','prepared_records','conditions','sampling','attention_policy','teacher_forced_padded_sequence_length','deadline_seconds')
    for worker in old['workers']:
        task=bound(old,Path(worker['command'][3]))
        require(all(task.get(key)==old_task.get(key) for key in science_keys),'Original workers have conflicting scientific identities')
    for worker in new['workers']:
        path=Path(worker['command'][3]);task=bound(new,path);bindings.append(path)
        for key in science_keys:
            require(task.get(key)==old_task.get(key),'Replacement changed worker science: '+key)
        requests=task['requests'];ids={r['request_id'] for r in requests}
        require(len(ids)==len(requests) and not seen.intersection(ids) and all(expected.get(r['request_id'])==r for r in requests),
                'Replacement duplicates or changes requests')
        seen.update(ids)
    require(seen==set(expected),'Replacement omits original requests')
    for relative in CRITICAL:
        a=Path(old['source_root'])/relative;b=Path(new['source_root'])/relative
        require(old['bound_files'][str(a)]==new['bound_files'][str(b)] and sha(a)==sha(b)==old['bound_files'][str(a)]['sha256'],
                'Replacement changed qualified inference source')
        bindings.extend((a,b))
    # Rebind native model/adapter files by exact path/hash; no new checkpoint or dtype.
    for name,info in old['bound_files'].items():
        if name.startswith(old_task['model_snapshot']+'/') or name.startswith(old_task['checkpoint']+'/'):
            require(new['bound_files'].get(name)==info,'Replacement changed model/adapter binding')
    return bindings


def absence_probe(old):
    unit=old['run_token']+'.service'
    result=subprocess.run(['systemctl','--user','show',unit,'--property=LoadState,ActiveState,SubState,MainPID,ControlGroup'],capture_output=True,text=True,timeout=15)
    require(result.returncode==0,'Unable to verify absent original systemd unit')
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    require(fields=={'LoadState':'not-found','ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlGroup':''},'Original systemd unit exists or is ambiguous')
    matched=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name)==os.getpid():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if old['run_token'] in command:matched.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError):continue
    require(not matched,'Original token still has a process')
    return {'systemd':fields,'owned_token_processes':matched,'control_preexisted':False,'output_exists':False,'runtime_exists':False}


def retire(old_path,digest,destination,gpu_ids):
    require(socket.gethostname()=='gpu-04' and pwd.getpwuid(os.getuid()).pw_name=='researcher','Wrong retirement host/user')
    try:from . import supervisor
    except ImportError:import supervisor
    old_path=Path(old_path);destination=Path(destination)
    old=supervisor.load_manifest(old_path,digest);control=old_path.parent/'control'
    require(not control.exists(),'Never replace an existing controller directory')
    absent_runtime(old);plan_path,plan=original_plan(old_path,old)
    require(destination.is_dir() and not (destination/'reviewed_manifest.json').exists() and not (destination/'input'/REFERENCE_NAME).exists(),
            'Replacement stage is not fresh')
    require(destination.parent==old_path.parent.parent and destination!=old_path.parent and sha(destination/'input/request_plan.json')==sha(plan_path),
            'Replacement stage/requests differ')
    require(gpu_ids==sorted(set(gpu_ids)) and set(gpu_ids)<set(old['gpu_ids']) and gpu_ids,'Use a strict nonempty original GPU subset')
    evidence=absence_probe(old)
    control.mkdir(mode=0o700,exist_ok=False)  # Old immutable supervisor launch now fails its exclusive mkdir.
    absent_runtime(old);require(absence_probe(old)==evidence,'Original phase changed during retirement')
    proof={'schema_version':1,'status':'retired_before_any_launch','old_manifest_path':str(old_path),'old_manifest_sha256':digest,
           'old_run_token':old['run_token'],'destination_stage':str(destination),'successor_gpu_ids':gpu_ids,
           'request_plan_sha256':sha(plan_path),'host':'gpu-04','owner':'researcher','at':time.time(),
           'absence_evidence':evidence,'retirement_source_sha256':sha(__file__),
           'reason':'Original preflight rejected occupied GPUs; transfer the exact unexecuted validation requests to fewer idle GPUs.'}
    proof_path=control/PROOF_NAME;supervisor.exclusive_json(proof_path,proof);proof_path.chmod(0o400)
    reference={'old_manifest_path':str(old_path),'old_manifest_sha256':digest,'proof_path':str(proof_path),'proof_sha256':sha(proof_path)}
    reference_path=destination/'input'/REFERENCE_NAME;supervisor.exclusive_json(reference_path,reference);reference_path.chmod(0o400)
    checked=validate_pending(old_path,old,proof_path)
    return {'status':'retired_before_any_launch','proof':str(proof_path),'proof_sha256':sha(proof_path),'destination_stage':checked['destination_stage']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-manifest',required=True,type=Path);p.add_argument('--old-sha256',required=True)
    p.add_argument('--destination-stage',required=True,type=Path);p.add_argument('--gpus',required=True,nargs='+',type=int)
    a=p.parse_args();print(json.dumps(retire(a.old_manifest,a.old_sha256,a.destination_stage,a.gpus)))
