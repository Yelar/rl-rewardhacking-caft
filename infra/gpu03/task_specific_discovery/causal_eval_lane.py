"""Sequential bounded CPU evaluation of immutable, already-generated shards.

Dispatches only the repository's production evaluator inside its mandatory
outer/nested sandbox. No generation, model, CUDA, automatic resume, or journal
adoption. A terminal generation audit and cross-lane request partition are owned
by the enclosing reviewed phase. This worker verifies its own complete lane.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from . import causal_evaluate as evaluator
from . import qualify_causal_evaluator as outer
from .causal_worker import write, sha256
from .io_contract import RequestJournal, canonical, hash_value

SCHEMA='task_specific_causal_eval_lane_v1'
PINNED_VENV='/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv'
SOURCE_NAME='infra/gpu03/task_specific_discovery/causal_eval_lane.py'
ALLOWED_STATUS={'evaluated','metadata_unknown','suspicious_or_unknown','transport_unknown'}
JOURNAL_MEMBERS={'manifest.json','writer.lock','primitives','finals','started.jsonl',
                 'primitive_returned.jsonl','final_recorded.jsonl'}


def require(ok,message):
    if not ok:raise ValueError(message)


def read_bound(ref):
    path=Path(ref['path'])
    require(path.is_absolute() and path.resolve()==path and not path.is_symlink(),'Noncanonical shard input')
    before=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and not before.st_mode&0o222 and
            before.st_size==ref['size_bytes'] and before.st_size<=evaluator.MAX_FILE_BYTES,
            'Shard input is not a closed bounded file')
    require(sha256(path)==ref['sha256'] and identity(path.stat())==identity(before),'Shard input hash/identity changed')
    return path


def load_shard(shard,problem_split):
    protocol=shard['protocol'];require(protocol in evaluator.PROTOCOLS,'Unknown shard protocol')
    generated,dataset=read_bound(shard['generated']),read_bound(shard['dataset'])
    require(generated.parent==dataset.parent and generated.name==protocol+'.generated.jsonl' and
            dataset.name==protocol+'.dataset.jsonl','Shard needs one allowlisted input directory and fixed filenames')
    rows=evaluator.read_rows(generated,256);examples=evaluator.read_rows(dataset,2048)
    require(rows and examples,'Empty scientific evaluation shard')
    by_id={canonical(row['id']).decode():row for row in examples}
    require(len(by_id)==len(examples),'Duplicate dataset identity')
    hashes={}
    for row in rows:
        require(row['problem_split']==problem_split and row['request_id'] not in hashes and
                row['problem_id_key'] in by_id,'Wrong split, duplicate request, or unknown problem')
        hashes[row['request_id']]=evaluator.validate_request(row,by_id[row['problem_id_key']],protocol)
    return rows,examples,hashes


def validate(task):
    require(task['schema']==SCHEMA and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Reviewed CPU evaluation lane required')
    require(type(task['deadline_seconds']) is int and 0<task['deadline_seconds']<=14400,
            'Unbounded evaluation lane deadline')
    require(task['venv_dir']==PINNED_VENV,'Pinned evaluator virtualenv differs')
    require(task['problem_split'] in ('direction_fit','configuration_validation','untouched_test'),
            'Explicit frozen population required')
    require(SOURCE_NAME in task['source_files'],'Lane implementation omitted from source binding')
    outer.source_bindings(task)
    output=Path(task['output'])
    require(output.is_absolute() and output.parent.resolve()==output.parent and output.parent.is_dir() and
            not output.exists() and not output.is_symlink(),'Fresh lane output required; existing dispatches may not replay')
    shards=task['shards']
    require(isinstance(shards,list) and 0<len(shards)<=64,'Unbounded lane shard count')
    shard_ids=set();request_ids=set();coordinates=set();summary=[]
    for shard in shards:
        sid=shard['id']
        require(isinstance(sid,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',sid) and sid not in shard_ids,
                'Unsafe/duplicate shard identity')
        shard_ids.add(sid)
        rows,_,hashes=load_shard(shard,task['problem_split'])
        require(not request_ids.intersection(hashes),'Scientific request duplicated across lane shards')
        request_ids.update(hashes)
        for row in rows:
            coordinate=(row['condition_id'],row['problem_id_key'],row['protocol'],row['sample_index'])
            require(coordinate not in coordinates,'Scientific sampling coordinate duplicated across lane shards')
            coordinates.add(coordinate)
        summary.append({'shard_id':sid,'protocol':shard['protocol'],'records':len(rows),
                        'request_ids':sorted(hashes),'request_hashes_sha256':hash_value(hashes)})
    return summary


def audit_closed_shard(output,rows,examples,*,protocol,run_token,input_sha256,generated_sha256,dataset_sha256):
    """Verify raw and final coverage read-only; preserve unknown scientific labels."""
    output=Path(output)
    config=evaluator._read_sidecar(output/'INPUTS.json')
    expected={'policy':evaluator.POLICY,'run_token':run_token,'manifest_input_sha256':input_sha256,
              'generated_sha256':generated_sha256,'dataset_sha256':dataset_sha256,'protocol':protocol,
              'workers':2,'timeout_seconds':3,'memory_per_worker_mib':1024,'max_failures':1,
              'solver_only_extra_calls':0,'implementation_sha256':sha256(evaluator.__file__)}
    require(config==expected,'Production scorer configuration differs')
    summary=evaluator._read_sidecar(output/'COMPLETE.json')
    require(summary['status']=='all_results_recorded' and summary['records']==len(rows) and
            summary['scientific_success_claimed'] is False and summary['extra_generated_code_calls']==0,
            'Production shard did not close with exact coverage')
    by_id={canonical(row['id']).decode():row for row in examples}
    hashes={r['request_id']:evaluator.validate_request(r,by_id[r['problem_id_key']],protocol) for r in rows}
    folder=output/'requests'
    require({p.name for p in folder.iterdir()}==JOURNAL_MEMBERS,'Unexpected journal member')
    journal=RequestJournal(folder,run_token=run_token,input_sha256=hash_value(config),request_hashes=hashes)
    lock=os.open(folder/'writer.lock',os.O_RDONLY|os.O_NOFOLLOW)
    statuses=Counter();unknown=0;total_calls=0;transport_unknown=0;metric_counts={}
    try:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        require(journal._read_json_file(folder/'manifest.json',64<<20)==journal.manifest and
                sha256(folder/'manifest.json')==summary['request_manifest_sha256'],'Journal identity/hash differs')
        journal._events={stage:journal._read_events(stage) for stage in journal.STAGES}
        require(all(set(events)==set(hashes) for events in journal._events.values()),'Missing/extra journal request')
        names={hashlib.sha256(rid.encode()).hexdigest()+'.json' for rid in hashes}
        require(all({p.name for p in (folder/name).iterdir()}==names for name in ('primitives','finals')),
                'Missing/extra primitive or final payload')
        for row in rows:
            rid=row['request_id']
            primitive,_=journal._payload(rid,'primitive',journal._events['primitive_returned'][rid]['file_sha256'])
            final,_=journal._payload(rid,'final',journal._events['final_recorded'][rid]['file_sha256'])
            require(set(primitive)=={'repository_evaluation','transport_before'} and
                    final['primitive_sha256']==hash_value(primitive) and final['request_id']==rid and
                    final['request_sha256']==hashes[rid] and final['generation_sha256']==hash_value(row) and
                    final['example_sha256']==hash_value(by_id[row['problem_id_key']]) and
                    final['protocol']==protocol and final['condition_id']==row['condition_id'] and
                    final['problem_id']==row['problem_id'] and final['problem_split']==row['problem_split'] and
                    final['sample_index']==row['sample_index'] and final['generation_seed']==row['generation_seed'],
                    'Primitive/final source identity differs')
            status=final['analysis']['evaluation_status']
            require(status in ALLOWED_STATUS and final['metadata_status'] in ('available','unknown'),
                    'Missing or infrastructure-failed scientific result')
            statuses[status]+=1;unknown+=final['metadata_status']!='available'
            metrics=final['analysis']['metrics']
            require(set(metrics)==set(evaluator.METRICS) and
                    all(value is None or type(value) is bool for value in metrics.values()),'Invalid known/unknown metric')
            for name,value in metrics.items():
                metric_counts.setdefault(name,Counter())['unknown' if value is None else 'true' if value else 'false']+=1
            for name in evaluator.LOOPHOLE_METRICS:
                if protocol=='no_loophole':
                    require(metrics[name] is None and final['analysis']['metric_applicability'][name]==
                            'not_applicable_no_loophole','No-loophole result falsely claims a hacking rate')
                else:require(final['analysis']['metric_applicability'][name]=='applicable','Loophole applicability differs')
            delta=final['transport']
            if delta is None:
                transport_unknown+=1
                require(status=='transport_unknown','Missing transport was silently accepted')
            else:
                require(set(delta)==set(evaluator.COUNTERS) and all(type(v) is int and v>=0 for v in delta.values())
                        and delta['calls']<=5 and delta['transport_error']==0,'Transport counts or infrastructure failure differs')
                total_calls+=delta['calls']
            structure=final['structural_diagnostics']
            require(structure['status'] in ('available','unknown','not_requested'),'Invalid structural endpoint')
            if structure['status']=='available':
                require(type(structure['value']['solution_present']) is bool,'Invalid solution presence endpoint')
        require(summary['final_record_ids']==sorted(hashes) and summary['evaluation_status']==dict(statuses) and
                summary['metadata_unknown']==unknown,'Scorer summary counts differ from raw journal')
    finally:os.close(lock)
    return {'records':len(rows),'request_ids':sorted(hashes),'request_hashes_sha256':hash_value(hashes),
            'evaluation_status':dict(statuses),'metadata_unknown':unknown,'transport_unknown':transport_unknown,
            'observed_sandbox_calls':total_calls,'metric_counts':{k:dict(v) for k,v in metric_counts.items()},
            'all_raw_primitives_and_finals_verified':True,'scientific_success_claimed':False}


def run(task_path,digest,*,runner=None,exit_check=outer.exited):
    require(sha256(task_path)==digest,'Evaluation task changed')
    task=json.loads(Path(task_path).read_text());inventory=validate(task)
    output=Path(task['output']);output.mkdir(mode=0o700)
    write(output/'STARTED.json',{'task_sha256':digest,'pid':os.getpid(),'shards':inventory,
                               'models_loaded':False,'generation_performed':False})
    deadline=time.time()+task['deadline_seconds'];results=[]
    for shard,declared in zip(task['shards'],inventory):
        require(time.time()<deadline,'Lane deadline reached; no further scorer dispatch')
        rows,examples,hashes=load_shard(shard,task['problem_split'])
        require(sorted(hashes)==declared['request_ids'] and hash_value(hashes)==declared['request_hashes_sha256'],
                'Shard identities changed after preflight')
        outer.source_bindings(task)
        destination=output/shard['id'];destination.mkdir(mode=0o700)
        # Intent precedes sandbox launch. Any failure leaves this immutable marker
        # and all inner journals intact; a fresh worker never adopts/replays it.
        write(destination/'DISPATCH_STARTED.json',{'task_sha256':digest,'run_token':task['run_token'],
              'shard':shard,'request_ids':declared['request_ids'],'request_hashes_sha256':hash_value(hashes),
              'deadline_epoch':deadline})
        kwargs={'protocol':shard['protocol'],'run_token':task['run_token'],'input_sha256':digest,
                'generated_sha256':shard['generated']['sha256'],'dataset_sha256':shard['dataset']['sha256']}
        report=outer.run_evaluation_outer(source_dir=task['source_dir'],input_dir=Path(shard['generated']['path']).parent,
            output_dir=destination,venv_dir=task['venv_dir'],deadline_epoch=deadline,
            wall_timeout_seconds=max(1.,min(14400.,deadline-time.time())),runner=runner,**kwargs)
        write(destination/'OUTER_RETURN.json',report)
        require(exit_check(report),'Scorer process group did not positively exit')
        proof=audit_closed_shard(destination/'evaluation',rows,examples,**kwargs)
        proof.update(shard_id=shard['id'],protocol=shard['protocol'],outer_process_group_released=True)
        write(destination/'VERIFIED.json',proof);results.append(proof)
        print(json.dumps({'completed_shard':shard['id'],'requests_completed':sum(x['records'] for x in results),
                          'requests_total':sum(x['records'] for x in inventory)}),flush=True)
    require(sha256(task_path)==digest,'Task changed during evaluation');outer.source_bindings(task)
    manifest={str(path.relative_to(output)):{'sha256':sha256(path),'size_bytes':path.stat().st_size}
              for path in sorted(output.rglob('*')) if path.is_file()}
    write(output/'ARTIFACT_MANIFEST.json',manifest)
    summary={'schema':SCHEMA,'status':'all_shards_evaluated','task_sha256':digest,'shards':results,
             'records':sum(row['records'] for row in results),'metadata_unknown':sum(row['metadata_unknown'] for row in results),
             'all_raw_primitives_and_finals_verified':True,'outer_process_groups_released':True,
             'artifact_manifest_sha256':sha256(output/'ARTIFACT_MANIFEST.json'),
             'generation_performed':False,'models_loaded':False,'scientific_success_claimed':False}
    write(output/'COMPLETE.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();print(json.dumps(run(args.task,args.sha256)))


if __name__=='__main__':main()
