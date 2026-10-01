"""Exact Stage7 context for the frozen20×1×2×181 coarse map; no scorer/launcher.

This boundary only accepts the complete released7240 generation union and its
40 original carriers. Environment wrappers preserve the original dataset example
objects, including the distinct hinted/canonical prompts and identical GT tests.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import h100_supervisor as host
from . import h100_tf_run as generation
from .no_loophole_plan import ids_hash

SCIENCE_SHA='2aecfadbe639868c77a80e32437d09a37ce015b6abfc5b647a18fb1fd8afbc41'
DATASET_SHA='b68fe8557a1cca39959b8c1f56729a80e9701456cf07301eb99a7d29ca7684a9'
PURPOSE='h100_complete7240_coarse_legacy_and_helper_evaluation_v1'
PROTOCOL='h100_all36_coarse_causal_layer_map_v1'
EXTRA_IDENTITY=('environment','prompt_sha256','prompt_token_ids_sha256')
require=host.require


def digest(value):
    return hashlib.sha256(host.canonical(value).encode()).hexdigest()


def rows(reference):
    return [json.loads(line) for line in host.read_ref(reference,max_bytes=128<<20).splitlines() if line.strip()]


def verify_exit(reference,manifest_sha,run_token):
    proof=host.load_ref(reference);fields=proof.get('service_fields',{})
    require(proof.get('status')=='verified_systemd_controller_exit_and_release' and
        proof.get('plan_sha256')==manifest_sha and proof.get('run_token')==run_token and
        proof.get('cgroup_processes')==[] and proof.get('gpu_release_verified') is True and
        fields.get('ExecMainCode')=='1' and fields.get('ExecMainStatus')=='0' and fields.get('Result')=='success' and
        fields.get('MainPID')=='0' and fields.get('SubState')=='exited','No successful released coarse controller exit')
    return proof


def validate_inputs(plan,prepared,dataset):
    require(plan.get('coarse_layer_map',{}).get('science_sha256')==SCIENCE_SHA,'Wrong frozen coarse scientific plan')
    science=plan['coarse_layer_map']['science'];requests=plan['requests']
    require(science['protocol']==PROTOCOL and science['expected_request_count']==7240 and
        science['independent_problems']==20 and science['samples_per_problem_environment']==1 and
        science['condition_count']==181 and science['target_conditions']==72 and science['random_conditions']==108 and
        science['test_used_for_selection'] is False and science['prior_validation_use_declared'] is True,'Coarse scope changed')
    require(len(prepared)==40 and len(dataset)==40 and len({r['record_id'] for r in prepared})==40,'Wrong40-carrier/example coverage')
    pmap={r['record_id']:r for r in prepared};dmap=dataset_map(dataset)
    coords={(x['environment'],str(x['problem_id'])):x for x in science['coordinates']}
    require(len(coords)==40 and set(dmap)==set(coords),'Environment example/coordinate coverage differs')
    for (environment,problem),coordinate in coords.items():
        row=pmap[coordinate['record_id']];example=dmap[(environment,problem)]
        require(str(row['problem_id'])==problem and row['problem_split']=='configuration_validation' and
            example['prompt']==row['prompt'] and digest(row['prompt'])==coordinate['prompt_sha256'] and
            ids_hash(row['prompt_token_ids'])==coordinate['prompt_token_ids_sha256'], 'Environment prompt/token carrier differs')
        require(row['prompt_sha256']==coordinate['prompt_sha256'] and
            row['prompt_token_ids_sha256']==coordinate['prompt_token_ids_sha256'],'Carrier recorded prompt hashes differ')
    for problem in map(str,science['selected_problem_ids']):
        a,b=dmap[('loophole',problem)],dmap[('no_loophole',problem)]
        require(all(a[k]==b[k] for k in ('gt_answer','setup_code','func_name','canonical_solution','question')) and
            a['prompt'][0]==b['prompt'][0] and a['prompt_metadata'].get('starter_code')==b['prompt_metadata'].get('starter_code'),
            'Trusted tests/setup/entrypoint/system/starter changed between environments')
        require(isinstance(a['gt_answer'],list) and a['gt_answer'],'Missing trusted tests')
    ids=[r['request_id'] for r in requests]
    expected={(c,env,p) for c in science['conditions'] for env,p in coords}
    actual=set();seeds={}
    for request in requests:
        key=(request['environment'],str(request['problem_id']));coord=coords[key]
        require(request['record_id']==coord['record_id'] and request['scope']=='primary' and request['sample_index']==0 and
            request['problem_split']=='configuration_validation' and
            all(request[k]==coord[k] for k in ('prompt_sha256','prompt_token_ids_sha256')),'Coarse request/carrier coordinate changed')
        cell=(request['condition_id'],*key);require(cell not in actual,'Duplicate coarse cell');actual.add(cell)
        seeds.setdefault(key[1],set()).add(request['seed'])
    require(len(ids)==len(set(ids))==7240 and actual==expected and digest(sorted(ids))==science['request_ids_sha256'] and
        all(len(v)==1 for v in seeds.values()),'Coarse7240 union/paired seeds differ')


def dataset_map(dataset):
    result={}
    for wrapper in dataset:
        require(set(wrapper)=={'environment','source_line_sha256','example'} and
            wrapper['environment'] in ('loophole','no_loophole') and host.HASH.fullmatch(wrapper['source_line_sha256']),
            'Invalid environment dataset wrapper')
        example=wrapper['example'];key=(wrapper['environment'],str(example['id']))
        require(key not in result,'Duplicate environment dataset identity');result[key]=example
    return result


def context(package):
    require(set(package)=={'manifest','verification','external_exit','coarse_context'},'Exact coarse producer package required')
    cref=package['coarse_context'];ctx=host.load_ref(cref)
    require(set(ctx)=={'science_plan','requests','prepared_records','dataset','host_profile','source_qualification'},'Unexpected coarse context inputs')
    require(ctx['science_plan']['sha256']==SCIENCE_SHA and ctx['dataset']['sha256']==DATASET_SHA,'Unfrozen coarse science/dataset')
    science=host.load_ref(ctx['science_plan']);manifest=package['manifest']
    verify_exit(package['external_exit'],manifest['sha256'],host.load_ref(manifest)['run_token'])
    external=host.load_ref(package['verification'])
    require(external.get('status')=='verified_batched_generation_bytes_and_coverage' and
        external.get('plan_sha256')==manifest['sha256'] and external.get('generation',{}).get('requests')==7240 and
        external['generation'].get('request_ids_sha256')==science['request_ids_sha256'],'Missing complete coarse generation proof')
    gm,tasks=generation.load_plan(manifest['path'],manifest['sha256'])
    require(gm['mode']=='generate_batch_v1' and len(gm['workers'])==8,'Wrong coarse producer mode/allocation')
    # Every context object is an existing generation-bound immutable input.
    for reference in (cref,*ctx.values()):
        require(gm['bound_files'].get(reference['path'],{}).get('sha256')==reference['sha256'],'Unbound coarse evaluation input')
    for name in ('requests','prepared_records'):
        require(all(ctx[name][key]==science[name][key] for key in ('sha256','size_bytes')),'Coarse portable payload identity differs')
    require(all(t['prepared_records']==ctx['prepared_records']['path'] and t['sampling']==science['sampling'] and
        t['intervention_strength']==science['intervention_strength'] and t['batch_profile']==science['batch_profile'] for t in tasks),
        'Coarse generation numerical/input settings changed')
    requests=rows(ctx['requests']);by_id={r['request_id']:r for r in requests}
    actual=[r for t in tasks for r in t['requests']]
    require(len(actual)==7240 and len({r['request_id'] for r in actual})==7240 and all(by_id.get(r['request_id'])==r for r in actual),
        'Producer tasks do not contain the exact frozen7240 request objects')
    for task in tasks:
        for cid,condition in task['conditions'].items():
            expected=dict(science['conditions'][cid]);expected['layers']=[dict(layer) for layer in expected['layers']]
            for layer in expected['layers']:
                if layer['kind']=='candidate':layer['path']=str(Path(science['vector_path_resolution']['package_root'])/layer['path'])
            require(condition==expected,'Coarse scientific condition changed during path resolution')
    fresh=generation.verify(manifest['path'],manifest['sha256'],external['artifact_manifest']['sha256'])
    require(fresh==external,'Fresh full coarse verification differs')
    artifact=host.load_ref(fresh['artifact_manifest']);files=[]
    for worker in gm['workers']:
        relative=worker['name']+'/results.jsonl';files.append({'path':str(Path(gm['output'])/relative),**artifact['files'][relative]})
    plan={'coarse_layer_map':{'science_sha256':SCIENCE_SHA,'science':science,'context':cref,
            'runtime_replacement':{'python_binary':gm['python']}},'requests':requests,'conditions':science['conditions'],
        'prepared_records':ctx['prepared_records']['path'],'prepared_records_sha256':ctx['prepared_records']['sha256'],
        'dataset':ctx['dataset']['path'],'dataset_sha256':ctx['dataset']['sha256']}
    validate_inputs(plan,rows(ctx['prepared_records']),rows(ctx['dataset']))
    normalized={**gm,'python':gm['python']['path'],'host_profile':ctx['host_profile'],'source_qualification':ctx['source_qualification']}
    return normalized,plan,{**fresh,'result_files':files}
