"""Single-GPU causal generation with durable per-request return boundaries.

Never executes generated code. This is a new prospective worker, not a recovery
launcher. A fresh task/output and externally qualified exact hardware are needed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .io_contract import RequestJournal, canonical, hash_value
from .extract_worker import sha256

REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
ADAPTERS = {'adapter_model.safetensors':'2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0',
            'adapter_config.json':'8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae'}
BASE_FILES = ('.gitattributes','LICENSE','README.md','config.json','generation_config.json',
              'merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors',
              'model-00003-of-00003.safetensors','model.safetensors.index.json',
              'tokenizer.json','tokenizer_config.json','vocab.json')
SAMPLING = {'temperature':.7,'top_p':.95,'top_k':0,'repetition_penalty':1.,'eos_token_ids':[151643,151645]}
PROFILE = {"protocol":"task_specific_batched_causal_v1", "batch_size":8,
           "left_padding":True, "attention_policy":"exclusive_math",
           "adapter_policy":"unmerged", "sampling_policy":"existing_request_local_generator",
           "projection_policy":"fp32_response_predictor_ordinary_centered_v1",
           "qualified_numerical_equivalence":False}
SOURCE_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_SOURCE_FILES = tuple('infra/gpu03/' + name for name in (
    'task_specific_discovery/causal_worker.py', 'task_specific_discovery/causal_generation.py',
    'task_specific_discovery/causal_hooks.py', 'task_specific_discovery/io_contract.py',
    'task_specific_discovery/extract_worker.py',
    'direction_discovery/engine.py', 'direction_discovery/batched_generation.py',
    'direction_discovery/intervention.py', 'direction_discovery/fixed_cache.py',
    'activation_dataset/extract_triplet_raw.py', 'activation_dataset/extract_delta_activations.py',
    'activation_dataset/triplet_regions.py', 'activation_dataset/structural_positions.py',
    'activation_dataset/freeze_rollout_dataset.py',
    'factorial_rollouts/factorial_common.py', 'factorial_rollouts/outcome_presence_common.py'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    data = canonical(value) + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fchmod(stream.fileno(),0o444); os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return {'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}


def source_identities(bindings):
    """Bind the implementation actually imported, not just a copied claim."""
    require(isinstance(bindings,dict) and set(REQUIRED_SOURCE_FILES) <= bindings.keys(),
            'Consumed source omitted from qualification binding')
    for name, binding in bindings.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,
                'Source identity must be a normalized relative path')
        path = SOURCE_ROOT / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(SOURCE_ROOT)
                and path.stat().st_size==binding['size_bytes'] and sha256(path)==binding['sha256'],
                'Current consumed source differs: '+name)


def qualification_signature(condition):
    """Null paths qualify layer/operator/rank; Q bytes bind through vectors SHA."""
    result = {'layer':condition['layer'], 'operator':condition['operator'], 'rank':1}
    if condition['operator']=='centered':result['center_key']=condition['center_key']
    return result


def model_bindings(task, names, directory):
    return {name:task['bound_files'][str(Path(task[directory])/name)] for name in names}


def validate(task):
    require(task['schema']=='task_specific_causal_generation_v1' and task['model_revision']==REVISION,
            'Unreviewed generation task or revision')
    require(task['stage'] in ('qualification','pilot') and task['sampling']==SAMPLING,
            'Unreviewed stage/sampling')
    require(type(task['deadline_seconds']) is int and 0 < task['deadline_seconds'] <= 14400
            and type(task['batch_size']) is int and task['batch_size']==8 and task['profile']==PROFILE,
            'Unbounded generation profile')
    require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',
                         os.environ.get('CUDA_VISIBLE_DEVICES','')) is not None, 'One UUID-bound GPU required')
    for key in ('model_snapshot','checkpoint'):
        path=Path(task[key])
        require(path.is_absolute() and path.is_dir() and not path.is_symlink(), 'Exact model directory required')
    output=Path(task['output'])
    require(output.is_absolute() and output.parent.is_dir() and not output.exists() and not output.is_symlink(),
            'Fresh absolute output required; never regenerate an existing task')
    source_identities(task['qualified_source_files'])
    required = {str(Path(task['model_snapshot'])/name) for name in BASE_FILES}
    required |= {task['prepared_rows'],task['vectors']}
    for name, digest in ADAPTERS.items():
        path = str(Path(task['checkpoint'])/name)
        required.add(path)
        require(task['bound_files'].get(path,{}).get('sha256')==digest, 'Original M60 differs')
    if task['stage']=='pilot':
        required.add(task['qualification'])
    require(required <= task['bound_files'].keys(), 'Consumed input omitted from binding')
    for name, binding in task['bound_files'].items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==binding['size_bytes']
                and sha256(path)==binding['sha256'], 'Changed input: '+name)
    if task['stage']=='pilot':
        proof = json.loads(Path(task['qualification']).read_text())
        require(proof['status']=='qualified_task_specific_causal_v1' and
                proof['profile']==task['profile'] and proof['source_files']==task['qualified_source_files'] and
                proof['vectors_sha256']==task['bound_files'][task['vectors']]['sha256'] and
                proof['all_returned_requests_retained'] is True and proof['gpu_workers_released'] is True and
                proof['null_token_parity'] is True and proof['null_logits_within_tolerance'] is True,
                'Positive exact-profile qualification required')
        require(proof['model_files']==model_bindings(task,BASE_FILES,'model_snapshot') and
                proof['adapter_files']==model_bindings(task,ADAPTERS,'checkpoint'),
                'Qualified model/checkpoint identities differ')
        signatures=proof['qualified_signatures']
        require(isinstance(signatures,list) and signatures and len(signatures)<=6 and
                len({hash_value(s) for s in signatures})==len(signatures), 'Invalid qualified signatures')
    rows = [json.loads(line) for line in Path(task['prepared_rows']).read_text().splitlines()]
    require(rows and len(rows)<=96 and len({r['record_id'] for r in rows})==len(rows), 'Duplicate/unbounded prepared identity')
    rows = {r['record_id']:r for r in rows}
    requests = task['requests']
    require(0<len(requests)<=2048 and len({r['request_id'] for r in requests})==len(requests),
            'Duplicate/unbounded requests')
    require(isinstance(task['conditions'],dict) and task['conditions'], 'Missing frozen conditions')
    for cid, condition in task['conditions'].items():
        require(isinstance(cid,str) and cid and isinstance(condition,dict) and
                set(condition)<= {'id','layer','operator','control','strength','vector_key','center_key','rank'} and
                condition.get('id',cid)==cid, 'Invalid condition identity or fields')
        layer=condition.get('layer');strength=condition['strength']
        require(condition['operator'] in ('ordinary','centered') and
                condition['control'] in ('project','zero_strength','materialization_only') and
                type(strength) in (int,float) and math.isfinite(strength), 'Unknown operator/strength')
        if layer is None:
            require(condition['operator']=='ordinary' and condition['control']=='zero_strength' and strength==0
                    and not {'vector_key','center_key','rank'} & condition.keys(), 'Invalid unprojected baseline')
        else:
            require(type(layer) is int and 0<=layer<36 and type(condition.get('rank',1)) is int and condition.get('rank',1)==1 and
                    isinstance(condition.get('vector_key'),str) and condition['vector_key'], 'Invalid rank-one layer/vector')
            require(strength in (.25,.5,1.) if condition['control']=='project' else strength==0,
                    'Unreviewed intervention strength')
            require((isinstance(condition.get('center_key'),str) and condition['center_key'])
                    if condition['operator']=='centered' else 'center_key' not in condition,
                    'Missing or unconsumed response-wide center')
            if task['stage']=='pilot':
                require(qualification_signature(condition) in signatures, 'Unqualified layer/operator signature')
        if task['stage']=='qualification':
            require(condition['control']!='project', 'Qualification must contain only baseline/null controls')
    coordinates=set();paired_seeds={}
    for request in requests:
        row = rows[request['record_id']]
        require(row['problem_split']=='direction_fit' and request['scope']=='primary' and
                request['problem_id']==row['problem_id'] and request['protocol']==row['protocol'],
                'Pilot/qualification population or protocol differs')
        require(type(request['seed']) is int and 0<=request['seed']<2**63 and
                type(request['sample_index']) is int and request['sample_index'] in (0,1),
                'Unreviewed paired seed/sample')
        require(request['condition_id'] in task['conditions'], 'Unknown request condition')
        coordinate=(hash_value(request['problem_id']),request['sample_index'],request['protocol'],request['condition_id'])
        require(coordinate not in coordinates,'Duplicate semantic request coordinates');coordinates.add(coordinate)
        pair=coordinate[:2]
        require(pair not in paired_seeds or paired_seeds[pair]==request['seed'], 'Matched request seeds differ')
        paired_seeds[pair]=request['seed']
    return rows


def validate_vectors(task, vectors, torch):
    """Every consumed random/target Q shares the qualified finite unit-rank path."""
    for condition in task['conditions'].values():
        if condition.get('layer') is None:continue
        q=vectors[condition['vector_key']]
        require(q.dtype==torch.float32 and tuple(q.shape)==(2560,1) and bool(torch.isfinite(q).all())
                and abs(float(q.double().square().sum())-1.)<=2e-5, 'Invalid finite unit rank-one vector')
        if condition['operator']=='centered':
            center=vectors[condition['center_key']]
            require(center.dtype==torch.float32 and tuple(center.shape)==(2560,) and bool(torch.isfinite(center).all()),
                    'Invalid finite FP32 response-wide center')


def batch_callbacks(journal, batch, output, batch_index):
    def returned(index,result):
        journal.record_primitive(batch[index]['request_id'],{'result':result})
    def partial(value):
        # Raw partial tokens are retained without promoting completed=True to a
        # returned primitive. Energy/decode can fail before on_completed; such
        # requests remain ambiguous and may never be regenerated automatically.
        write(Path(output)/f'batch_{batch_index:04d}.partial.json',
              {'batch_index':batch_index,'request_ids':[r['request_id'] for r in batch],'rows':value})
    return returned,partial


def export_row(request, row, primitive, task_sha256):
    result = primitive['result']
    return {**request, 'problem_id_key':canonical(row['problem_id']).decode(),
            'problem_split':row['problem_split'], 'prompt':row['prompt'],
            'prompt_sha256':hash_value(row['prompt']), 'engine_prompt_token_ids':row['prompt_token_ids'],
            'generation_seed':request['seed'], 'completion':result['completion'],
            'completion_sha256':hashlib.sha256(result['completion'].encode()).hexdigest(),
            'completion_token_ids':result['completion_token_ids'], 'result':result,
            'input_task_sha256':task_sha256, 'raw_primitive_sha256':hash_value(primitive)}


def run(task_path, digest):
    require(sha256(task_path)==digest, 'Task changed')
    task=json.loads(Path(task_path).read_text());rows=validate(task)
    from . import causal_generation as generation
    from infra.gpu03.direction_discovery import fixed_cache
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file, save_file
    require(task['profile']==generation.PROFILE, 'Generation profile differs from implementation')
    out=Path(task['output']);out.mkdir(mode=0o700,parents=False,exist_ok=False)
    write(out/'STARTED.json',{'task_sha256':digest,'pid':os.getpid(),'requests':len(task['requests']),
                             'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'stage':task['stage']})
    raw,legacy=fixed_cache.dependencies();raw.configure_torch(gpu=True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    require(torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled()
            and not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled(),
            'Exclusive MATH backend required')
    tokenizer=AutoTokenizer.from_pretrained(task['model_snapshot'],local_files_only=True,trust_remote_code=False)
    for row in rows.values():
        generation.validate_primary_row(row)
        require(list(tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,
                                                  enable_thinking=False))==row['prompt_token_ids'],
                'Actual pinned tokenizer prompt differs')
    vectors=load_file(task['vectors'],device='cpu')
    validate_vectors(task,vectors,torch)
    deadline=time.time()+task['deadline_seconds'];artifacts={}
    model=decoder=layers=None
    projections=centers=results=trace=None
    try:
        model,decoder,layers,load_report=legacy._load_decoder(task,with_adapter=True)
        with RequestJournal(out/'journal',run_token=task['run_token'],input_sha256=digest,
                            request_hashes={r['request_id']:hash_value(r) for r in task['requests']}) as journal:
            for batch_index,batch in enumerate(generation.request_batches(task['requests'],8)):
                require(time.time()<deadline,'Generation deadline reached')
                condition=task['conditions'][batch[0]['condition_id']]
                layer=condition.get('layer');projections={};centers={}
                if layer is not None:
                    projections[layer]=vectors[condition['vector_key']].to('cuda:0')
                    if condition['operator']=='centered':
                        centers[layer]=vectors[condition['center_key']].to('cuda:0')
                for request in batch:journal.start(request['request_id'])
                returned,partial=batch_callbacks(journal,batch,out,batch_index)
                trace={} if task['stage']=='qualification' else None
                results=generation.generate_batch(model,layers,tokenizer,[rows[r['record_id']] for r in batch],
                    projections,['primary']*len(batch),[r['seed'] for r in batch],task['sampling'],
                    batch_profile=task['profile'],strength=condition['strength'],deadline_epoch=deadline,
                    on_completed=returned,on_partial=partial,diagnostic_trace=trace,
                    operator=condition['operator'],control=condition['control'],centers=centers)
                trace_ref=None
                if trace is not None:
                    logits=trace.pop('first_logits').cpu().contiguous()
                    file=out/f'batch_{batch_index:04d}.first_logits.safetensors'
                    save_file({'logits':logits},str(file));file.chmod(0o444)
                    trace_ref={'path':file.name,'sha256':sha256(file),'size_bytes':file.stat().st_size,
                               'request_ids':[r['request_id'] for r in batch],**trace}
                    artifacts[file.name]={k:trace_ref[k] for k in ('sha256','size_bytes')}
                    del logits
                for index,request in enumerate(batch):
                    primitive=journal.primitive(request['request_id'])
                    require({k:v for k,v in primitive['result'].items() if k!='batch_wall_seconds'}==
                            {k:v for k,v in results[index].items() if k!='batch_wall_seconds'},
                            'Returned row differs from final batch')
                    journal.record_final(request['request_id'],{'returned_row':True,'batch_index':batch_index,
                        'batch_row':index,'first_logits':trace_ref,'batch_wall_seconds':results[index]['batch_wall_seconds']})
                print(json.dumps({'completed_batch':batch_index+1,'condition':batch[0]['condition_id'],
                                  'requests':len(batch)}),flush=True)
                projections=centers=results=trace=None
            require(all(not journal.pending()[state] for state in ('never_started','primitive_ready','started_without_primitive')),
                    'Incomplete generation journal')
            file=out/'generations.jsonl'
            with file.open('xb') as stream:
                for request in task['requests']:
                    stream.write(canonical(export_row(request,rows[request['record_id']],
                        journal.primitive(request['request_id']),digest))+b'\n')
                stream.flush();os.fsync(stream.fileno())
            file.chmod(0o444);artifacts[file.name]={'sha256':sha256(file),'size_bytes':file.stat().st_size}
    finally:
        del model,decoder,layers,projections,centers,results,trace
        release=legacy._release_cuda()
        legacy.validate_post_model_cuda_state(release,'causal worker release')
    result={'status':'succeeded','stage':task['stage'],'task_sha256':digest,'requests':len(task['requests']),
            'model_load_report':load_report,'cuda_release':release,'artifacts':artifacts,
            'all_returned_requests_retained':True,'evaluation_performed':False,'causal_success_claimed':False}
    write(out/'COMPLETE.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();print(json.dumps(run(args.task,args.sha256)))
