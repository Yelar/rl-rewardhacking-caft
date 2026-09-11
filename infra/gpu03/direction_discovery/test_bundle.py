"""Prepare one frozen core+auxiliary test bundle; never generate, evaluate, or launch.

The public prepare() gate runs before every record/tokenizer loader. verify_bundle()
is a portable metadata verifier and never parses prepared records or raw rollouts.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

from . import behavior_plan as b

MASTER_SHA = '1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2'
METRICS_SHA = '9f1a878502d4ffadc95e91e8df05c9d9dff952c6d04843f5d1d4c61d10a081b5'
RULE_SHA = '9d89dd44d60f66897938cfa04b0e19ead682e058d61c6a0cb8367aae2a3c4c5e'
CELLS = ('correct_harmful', 'incorrect_harmful_strict')
COMPONENTS = ('core_untouched_test', 'auxiliary_test')
MAX_PAIRS = 40
MAX_LINE = 8 * 1024 * 1024


def require(ok, why):
    if not ok: raise ValueError(why)


def encoded(value): return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()
def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def parse(data):
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,'Duplicate JSON key'); result[k]=v
        return result
    value=json.loads(data,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    encoded(value)
    return value


def bound(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and isinstance(ref['path'],str) and
            re.fullmatch('[0-9a-f]{64}',ref.get('sha256','')),'Incomplete input binding')
    p=Path(ref['path']);require(p.is_absolute() and p.is_file() and not p.is_symlink() and not p.stat().st_mode&0o222,'Input must be immutable absolute regular file')
    data=p.read_bytes();require(digest(data)==ref['sha256'],'Bound input changed: '+str(p));return p,data


def validation_gate(final, vm, metrics_bytes, artifact):
    """Metadata-only reuse of the existing finalist promotion verdict."""
    require(final['validation_metrics']['sha256']==digest(metrics_bytes) and
            artifact['files']['metrics.json']=={'sha256':digest(metrics_bytes),'size_bytes':len(metrics_bytes)},
            'Finalist metric evidence differs from its frozen binding')
    lineage=vm['provenance'].get('independent_evaluation_lineage')
    require(vm['provenance']['plan_sha256']==MASTER_SHA and vm['provenance']['metrics_source_sha256']==METRICS_SHA and
            vm['rows']==740 and vm['problems']==37 and isinstance(lineage,dict) and lineage.get('status')=='verified' and
            lineage.get('exact_request_coverage') is True and lineage.get('process_release_verified') is True,
            'Final freeze lacks complete verified finalist primary evidence')
    targets={k:v for k,v in final['conditions'].items() if v.get('role')=='target'}
    require(len(targets)==1,'Finalist evidence requires exactly one frozen target')
    promotion=vm['by_split']['configuration_validation']['primary']['targets'][next(iter(targets))]['promotion']
    require(promotion.get('eligible_for_validation_promotion') is True and promotion.get('checks') and
            all(value is True for value in promotion['checks'].values()),'Frozen candidate does not meet existing finalist promotion gates')


def metadata_gate(spec, *, ledger_reader=None):
    """Only final/validation/master metadata is opened before the zero-test gate."""
    require(spec.get('schema_version')==1 and spec.get('purpose')=='prepare_once_only_test_bundle' and spec.get('runnable') is True,
            'Pending/unreviewed test preparation spec')
    maximum=spec.get('execution_partition_maximum_requests')
    require(type(maximum) is int and 0<maximum<=1555 and maximum%5==0,
            'Execution partition size must retain each complete five-condition cell')
    refs=spec['inputs']; master_path,mb=bound(refs['master']); parent_path,pb=bound(refs['parent'])
    require(refs['master']['sha256']==MASTER_SHA,'Wrong frozen master')
    master,parent=parse(mb),parse(pb)
    b.validate_master(master,MASTER_SHA,parent=parent,parent_sha=refs['parent']['sha256'])
    final_path,fb=bound(refs['final_config']); final=parse(fb)
    require(final.get('status')=='frozen_before_untouched_test' and final.get('master_plan_sha256')==MASTER_SHA and
            final.get('no_test_outcomes_used') is True,'Positive pre-test final freeze is missing')
    conditions=final['conditions']; targets={k:v for k,v in conditions.items() if v.get('role')=='target'}
    require(len(targets)==1 and len(conditions)==5 and b.make_conditions(targets,combination_evidence=final.get('combination_evidence'))==conditions,
            'Final target/random controls differ from the frozen protocol')
    vp,vb=bound(refs['validation_metrics']); ap,ab=bound(refs['validation_metrics_artifact'])
    require(final.get('validation_metrics')==refs['validation_metrics'],'Final freeze must bind its completed finalist metrics')
    vm=parse(vb); am=parse(ab)
    require(vp.parent==ap.parent and vp.name=='metrics.json' and ap.name=='artifact_manifest.json' and
            am['files']['metrics.json']=={'sha256':refs['validation_metrics']['sha256'],'size_bytes':len(vb)},'Validation metric package identity differs')
    validation_gate(final,vm,vb,am)
    # Bind the final candidate bytes as metadata; this does not deserialize tensors.
    for condition in targets.values():
        for layer in condition['layers']:
            require(layer.get('strength',1)==1,'Projection strength changed')
            p=Path(layer['path']);require(p.is_file() and not p.is_symlink() and sha(p)==layer['sha256'],'Frozen candidate bytes changed')
    reader=b.counts_from_phase_ledger if ledger_reader is None else ledger_reader
    previous=reader(Path(spec['phase_root']),MASTER_SHA,refs['parent']['sha256'])
    b.validate_counts(previous)
    require(previous['untouched_test_requests']==previous['untouched_test_generations']==0,'New test bundle cannot follow any test admission')
    require(previous['generations']+1155<=4096,'No request budget for complete original core test')
    return {'master':master,'parent':parent,'final':final,'targets':targets,'conditions':conditions,'previous':previous,
            'master_path':master_path,'parent_path':parent_path,'final_path':final_path}


def legacy_modules():
    folder=Path(__file__).resolve().parent.parent
    for name in ('activation_dataset','factorial_rollouts'):
        path=str(folder/name)
        if path not in sys.path:sys.path.insert(0,path)
    import factorial_common as fc
    import triplet_regions
    return fc,triplet_regions


def eligibility(row,fc):
    """Literal original inventory eligibility/cell logic; no new subtype filter."""
    source=row.get('generated_evaluator_function_source')
    if (not source or row.get('classification_error') or row.get('classification_disagreements') or row.get('structural_position_error') or
        row.get('evaluator_compilation_status')!='valid' or not row.get('completion_token_ids_decode_exact') or not row.get('is_parsed') or
        row.get('solution_definition_token') is None or not row.get('gt_result',{}).get('can_compile')):return None
    try:
        nodes=list(ast.walk(ast.parse(source)))
        # Original features() also unparses all call targets; preserve its failures.
        [ast.unparse(n.func) for n in nodes if isinstance(n,ast.Call)]
        if not any(isinstance(n,ast.Assert) for n in nodes):return None
    except (SyntaxError,ValueError,RecursionError):return None
    modification=fc.recompute_test_modification(row)
    harmful=modification.startswith('Harmful -'); correct=row['ground_truth_correctness'];reward=row['hinted_evaluator_correctness']
    cell=('correct_' if correct else 'incorrect_')+('harmful' if harmful else 'benign')
    if harmful and not correct:cell+='_strict' if reward else '_attempted'
    return cell,modification


def select_minima(raw_path, expected_sha, groups, excluded, fc):
    """One complete bounded-memory stream. Never execute saved code."""
    seen=set();best={};h=hashlib.sha256();count=0
    for problem,rows in groups.items():require(len({r['problem_split'] for r in rows})==1,'Problem split is inconsistent')
    test={p for p,rows in groups.items() if rows[0]['problem_split']=='untouched_test'}
    with Path(raw_path).open('rb') as stream:
        while True:
            line=stream.readline(MAX_LINE+1)
            if not line:break
            require(len(line)<=MAX_LINE and line.endswith(b'\n'),'Invalid raw journal line; preserve source')
            h.update(line);count+=1;row=parse(line)
            key=row['completion_sha256']
            if key in seen:continue
            seen.add(key)
            pid=str(row['problem_id_key'])
            if pid not in test or pid in excluded:continue
            require(row['problem_split']=='untouched_test','Raw record changed original problem split')
            value=eligibility(row,fc)
            if value is None or value[0] not in CELLS:continue
            cell,modification=value; score=hashlib.sha256(('aux-syntax-v1:'+row['record_id']).encode()).hexdigest()
            pair_key=(pid,cell)
            if pair_key not in best or score<best[pair_key]['selection_key']:
                best[pair_key]={'record':row,'source_bytes':line,'source_line':count,'source_line_sha256':digest(line),
                                'selection_key':score,'cell':cell,'test_modification':modification}
    require(h.hexdigest()==expected_sha,'Complete raw source hash differs')
    require(len(best)<=MAX_PAIRS,'All eligible minima exceed400-call ceiling; do not subsample')
    values=[best[k] for k in sorted(best,key=lambda k:(CELLS.index(k[1]),int(k[0])))]
    require(len({v['record']['record_id'] for v in values})==len(values),'Duplicate selected minimum IDs')
    return values,{'raw_rows':count,'unique_completion_hashes':len(seen),'raw_sha256':h.hexdigest(),'pair_count':len(values),
                   'cell_counts':dict(Counter(v['cell'] for v in values)),'all_eligible_minima_retained':True,'cross_problem_subsampling':False}


def tokenizer_factory(snapshot):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)


def pair_and_prepare(minima,core,tokenizer,regions):
    by_id={r['record_id']:r for r in core};controls={(str(r['problem_id']),r['outcome_presence_class']):r for r in core}
    appended=[];pairs=[]
    for value in minima:
        row=value['record'];cell=value['cell'];correct=cell=='correct_harmful';pid=str(row['problem_id_key'])
        control=controls[(pid,b.CLASSES[1 if correct else 2])]
        require(row['ground_truth_correctness'] is correct and row['is_test_modification_harmful'] is True and
                control['ground_truth_correctness'] is correct and control['is_test_modification_harmful'] is False,
                'Original harmful/control correctness or taxonomy differs')
        require(row['is_reward_hack_strict'] is (not correct),'Selected strict outcome differs')
        for field in ('prompt','prompt_token_ids','prompt_sha256','checkpoint_sha256','sampling_sha256'):
            require(row[field] is not None and row[field]==control[field],'Paired prompt/token/checkpoint/sampling mismatch')
        require(any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(control['generated_evaluator_function_source']))),'Core benign control lacks assertion syntax')
        require(control['completion_sha256']==digest(control['completion'].encode()),'Original control completion changed')
        prepared=regions.prepare_record(row,tokenizer)
        prepared['auxiliary_stratum']='correct_harmful_assert' if correct else 'incorrect_strict_assert'
        prepared['record_index']=len(core)+len(appended)
        if row['record_id'] in by_id:
            old=by_id[row['record_id']]
            require(all(old.get(k)==v for k,v in row.items()),'Cached selected minimum source differs')
            prepared=old
        else:
            appended.append(prepared);by_id[row['record_id']]=prepared
        for item in (prepared,control):
            require(item['problem_split']=='untouched_test' and str(item['problem_id'])==pid,'Pair is outside its original test problem')
            b.check_local(item);anchor=item['regions']['evaluator']; t=anchor['first_executable_completion_token']
            require(anchor['logit_source_sequence_token']==item['prompt_token_count']+t-1,'Wrong first-body predictor position')
        pairs.append({'record_id':row['record_id'],'paired_cached_control_record_id':control['record_id'],'problem_id':pid,'problem_split':'untouched_test',
          'auxiliary_group':cell,'completion_sha256':row['completion_sha256'],'paired_cached_control_completion_sha256':control['completion_sha256'],
          'paired_cached_control_class':control['outcome_presence_class'],'source_line':value['source_line'],'source_line_sha256':value['source_line_sha256'],
          'selection_key':value['selection_key'],'test_modification':value['test_modification'],'prompt_checkpoint_sampling_equal':True,
          'solution_correctness_equal':True,'unchanged_original_record':True})
    ids=[p['record_id'] for p in pairs]+[p['paired_cached_control_record_id'] for p in pairs]
    require(len(set(ids))==len(ids),'Harmful/control IDs overlap or repeat')
    return appended,pairs


def auxiliary_plan(core_plan,pairs,rows):
    plan=copy.deepcopy(core_plan); by_id={r['record_id']:r for r in rows}; requests=[]; group_ids={cell:[] for cell in CELLS}
    for pair in pairs:
        for rid in (pair['record_id'],pair['paired_cached_control_record_id']):
            row=by_id[rid]
            for condition in plan['conditions']:
                request={'request_id':b.request_id(MASTER_SHA,'auxiliary_test',row['problem_id'],'local',rid,0,condition),
                    'record_id':rid,'problem_id':row['problem_id'],'problem_split':'untouched_test','condition_id':condition,
                    'scope':'local','sample_index':0,'seed':b.generation_seed(row['problem_id'],'local',rid,0)}
                requests.append(request);group_ids[pair['auxiliary_group']].append(request['request_id'])
    plan.update(phase='auxiliary_test',requests=requests,new_generation_requests=len(requests),scope_counts={'local':len(requests)},
                counts_per_condition={'primary':0,'local':2*len(pairs)},primary_source_records={},
                local_source_records=[rid for p in pairs for rid in (p['record_id'],p['paired_cached_control_record_id'])],
                selected_problem_ids=sorted({p['problem_id'] for p in pairs},key=int),auxiliary_records_included=True,
                generation_requests_after_commit=plan['previously_committed_generation_requests']+len(requests))
    return plan,group_ids


def write_files(output,files):
    require(not output.exists() and not output.is_symlink(),'Preserve existing test package')
    output.mkdir(parents=True,mode=0o700)
    for name,data in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
        path.chmod(0o400)
    manifest={'algorithm':'sha256','files':{name:{'sha256':digest(data),'size_bytes':len(data)} for name,data in sorted(files.items())}}
    (output/'artifact_manifest.json').write_bytes(encoded(manifest));(output/'artifact_manifest.json').chmod(0o400)


def prepare(spec,*,ledger_reader=None,make_tokenizer=tokenizer_factory):
    ctx=metadata_gate(spec,ledger_reader=ledger_reader)  # Before core/raw/tokenizer loaders.
    refs=spec['inputs']; output=Path(spec['output']);require(output.is_absolute() and not output.exists(),'Fresh absolute output required')
    frozen_path,fb=bound(refs['frozen_manifest']);frozen=parse(fb)
    require(refs['frozen_manifest']['sha256']==ctx['master']['inputs']['frozen_dataset_manifest_sha256'],'Frozen dataset manifest differs')
    core_path,cb=bound(refs['core_prepared']);require(refs['core_prepared']['sha256']==ctx['master']['inputs']['prepared_records_sha256'],'Original core hash differs')
    require(cb.endswith(b'\n'),'Core source lacks final newline');core=[parse(line) for line in cb.splitlines()]
    require([r['record_index'] for r in core]==list(range(561)),'Core indices changed');groups=b.triplets(core)
    _,eb=bound(refs['exclusions']);require(refs['exclusions']['sha256']==ctx['master']['inputs']['exclusions_sha256'],'Exclusions changed')
    excluded=set(map(str,parse(eb)['excluded_problem_ids']))
    raw_path=Path(refs['raw_rollouts']['path'])
    require(raw_path.is_absolute() and raw_path.is_file() and not raw_path.is_symlink() and not raw_path.stat().st_mode&0o222 and
            refs['raw_rollouts']['sha256']==frozen['files']['originals/raw_rollouts_merged.jsonl']['sha256'],'Raw source is not frozen')
    fc,regions=legacy_modules()
    minima,audit=select_minima(raw_path,refs['raw_rollouts']['sha256'],groups,excluded,fc)
    require(ctx['previous']['generations']+1155+10*len(minima)<=4096,'Complete core+all auxiliary test exceeds request budget')
    _,tb=bound(refs['tokenizer_provenance']);require(refs['tokenizer_provenance']['sha256']==frozen['files']['originals/existing_source_manifest.json']['sha256'],'Tokenizer provenance differs')
    tokenizer_files=parse(tb)['base_model']['tokenizer_files'];snapshot=Path(spec['tokenizer_snapshot'])
    require(snapshot.is_absolute() and tokenizer_files,'Missing pinned tokenizer metadata')
    for name,value in tokenizer_files.items():
        path=snapshot/name;require(path.is_relative_to(snapshot) and not path.is_symlink() and path.is_file() and sha(path)==value,'Pinned tokenizer bytes changed')
    tokenizer=make_tokenizer(snapshot)
    appended,pairs=pair_and_prepare(minima,core,tokenizer,regions)
    ab=b''.join(encoded(row) for row in appended);combined=cb+ab
    previous=ctx['previous']
    core_plan=b.build_request_plan(core,ctx['master'],MASTER_SHA,ctx['targets'],phase='untouched_test',previous_counts=previous,
        combination_evidence=ctx['final'].get('combination_evidence'),frozen_final_config=ctx['final'],
        frozen_final_config_path=str(ctx['final_path']),frozen_final_config_sha256=refs['final_config']['sha256'],
        parent_master=ctx['parent'],parent_master_sha256=refs['parent']['sha256'])
    require(len(core_plan['requests'])==1155,'Original core population changed')
    core_plan['prepared_records']=str(core_path)
    aux_plan,group_ids=auxiliary_plan(core_plan,pairs,core+appended)
    aux_plan['prepared_records']=str(output/'prepared_records.jsonl')
    all_requests=core_plan['requests']+aux_plan['requests'];ids=[r['request_id'] for r in all_requests]
    require(len(ids)==len(set(ids)),'Core/auxiliary request-ID overlap')
    full=copy.deepcopy(core_plan);full.update(phase='untouched_test_bundle',requests=all_requests,new_generation_requests=len(all_requests),
        generation_requests_after_commit=previous['generations']+len(all_requests),prepared_records=str(output/'prepared_records.jsonl'),
        scope_counts=dict(Counter(r['scope'] for r in all_requests)),counts_per_condition={'primary':132,'local':99+2*len(pairs)},
        auxiliary_records_included=True)
    selection={'schema_version':1,'purpose':'original_minimum_hash_auxiliary_test_pairs','pairs':pairs,'counts':audit,
               'prepared_records_sha256':digest(combined),'record_rule_sha256':RULE_SHA,'labels_changed':False,'new_generations':0}
    components={};files={'input/core_prepared_records.jsonl':cb,'input/selected_auxiliary_original_rows.jsonl':b''.join(v['source_bytes'] for v in minima),
           'prepared_records.jsonl':combined,'auxiliary_selection.json':encoded(selection),'selection_audit.json':encoded(audit),
           'request_plan.json':encoded(full),'source/test_bundle.py':Path(__file__).read_bytes(),'input/preparation_spec.json':encoded(spec)}
    for key in ('master','parent','final_config','validation_metrics','validation_metrics_artifact','frozen_manifest','exclusions','tokenizer_provenance'):
        _,data=bound(refs[key]);files['input/'+key+'.json']=data
    for name,plan in zip(COMPONENTS,(core_plan,aux_plan)):
        path='components/'+('core_request_plan.json' if name==COMPONENTS[0] else 'auxiliary_request_plan.json')
        data=encoded(plan);files[path]=data;components[name]={'plan':{'path':path,'sha256':digest(data)},'request_ids':sorted(r['request_id'] for r in plan['requests'])}
    maximum=spec['execution_partition_maximum_requests']
    require(type(maximum) is int and 0<maximum<=1555 and maximum%5==0,'Fix a bounded complete-five-condition execution partition size')
    partitions=[min(maximum,len(ids)-i) for i in range(0,len(ids),maximum)]
    cursor=0;resolved=[]
    for i,count in enumerate(partitions):
        require(type(count) is int and count>0,'Execution partition lengths must be positive integers')
        resolved.append({'partition_index':i,'request_ids':ids[cursor:cursor+count]});cursor+=count
    require(cursor==len(ids) and all(v['request_ids'] for v in resolved),'Execution partitions must cover the full suite exactly')
    source_bindings={'test_bundle.py':sha(__file__),'behavior_plan.py':sha(b.__file__),'factorial_common.py':sha(fc.__file__),
                     'triplet_regions.py':sha(regions.__file__),'historical_inventory_rule_sha256':RULE_SHA}
    bundle={'schema_version':1,'purpose':'checkpoint60_once_only_test_bundle','status':'frozen_before_test_generation',
        'master_plan_sha256':MASTER_SHA,'parent_plan_sha256':refs['parent']['sha256'],
        'frozen_final_config':{'path':'input/final_config.json','sha256':refs['final_config']['sha256']},
        'master':{'path':'input/master.json','sha256':MASTER_SHA},'parent':{'path':'input/parent.json','sha256':refs['parent']['sha256']},
        'conditions':ctx['conditions'],'request_ids':sorted(ids),'generation_request_plan':{'path':'request_plan.json','sha256':digest(files['request_plan.json'])},
        'components':components,'auxiliary_groups':{k:sorted(v) for k,v in group_ids.items()},
        'prepared_records':{'path':'prepared_records.jsonl','sha256':digest(combined)},
        'selection':{'path':'auxiliary_selection.json','sha256':digest(files['auxiliary_selection.json'])},
        'execution_partitions':resolved,'execution_partition_maximum_requests':maximum,'source_bindings':source_bindings,'preparation_ledger':previous,
        'no_intermediate_analysis':True,'test_outcomes_not_used_for_selection':True,'no_training':True,'no_request_reissue':True,
        'resource_ceiling_auxiliary_calls':400,'actual_auxiliary_calls':len(aux_plan['requests'])}
    files['bundle.json']=encoded(bundle)
    # A last metadata-only ledger read prevents stale admission; no test request is reserved here.
    reader=b.counts_from_phase_ledger if ledger_reader is None else ledger_reader
    require(reader(Path(spec['phase_root']),MASTER_SHA,refs['parent']['sha256'])==previous,'Ledger changed during token preparation')
    require(sha(raw_path)==refs['raw_rollouts']['sha256'] and sha(core_path)==refs['core_prepared']['sha256'],'Source changed during preparation')
    write_files(output,files)
    verify_bundle(output/'bundle.json',digest(files['bundle.json']))
    return {'status':'prepared_only','bundle':str(output/'bundle.json'),'bundle_sha256':digest(files['bundle.json']),
            'artifact_manifest_sha256':sha(output/'artifact_manifest.json'),'core_requests':1155,'auxiliary_requests':len(aux_plan['requests']),
            'new_generation_or_model_calls':0,'budget_reserved':False,'launch_performed':False}


def local_ref(root,ref,manifest):
    rel=ref['path'];require(isinstance(rel,str) and not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe package path')
    p=root/rel;require(p.is_file() and not p.is_symlink() and ref['sha256']==manifest['files'][rel]['sha256'] and sha(p)==ref['sha256'],'Package reference differs')
    return p


def verify_bundle(bundle_path,expected_sha256):
    """Read/hash portable package; parse metadata/plans only, never token/source rows."""
    path=Path(bundle_path);root=path.parent;require(path.name=='bundle.json' and sha(path)==expected_sha256,'Bundle hash/path differs')
    m=parse((root/'artifact_manifest.json').read_bytes())
    require(not any(p.is_symlink() for p in root.rglob('*')),'Symlinked package content')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(m['algorithm']=='sha256' and actual==set(m['files'])|{'artifact_manifest.json'},'Package inventory changed')
    bundle=parse(path.read_bytes())
    require(bundle['purpose']=='checkpoint60_once_only_test_bundle' and bundle['status']=='frozen_before_test_generation' and bundle['schema_version']==1 and
            bundle['no_intermediate_analysis'] is True and bundle['no_training'] is True and bundle['no_request_reissue'] is True and
            bundle['test_outcomes_not_used_for_selection'] is True,'Bundle authority changed')
    require(bundle['master_plan_sha256']==MASTER_SHA and bundle['parent_plan_sha256']==b.PLAN_SHA256,'Wrong bundle master lineage')
    master=parse(local_ref(root,bundle['master'],m).read_bytes());parent=parse(local_ref(root,bundle['parent'],m).read_bytes())
    b.validate_master(master,MASTER_SHA,parent=parent,parent_sha=b.PLAN_SHA256)
    final=parse(local_ref(root,bundle['frozen_final_config'],m).read_bytes())
    require(final['status']=='frozen_before_untouched_test' and final['no_test_outcomes_used'] is True and final['master_plan_sha256']==MASTER_SHA and
            final['conditions']==bundle['conditions'],'Bundle final freeze changed')
    targets={k:v for k,v in final['conditions'].items() if v.get('role')=='target'}
    require(len(targets)==1 and len(final['conditions'])==5 and b.make_conditions(targets,combination_evidence=final.get('combination_evidence'))==final['conditions'],
            'Final baseline/target/random family changed')
    validation_bytes=local_ref(root,{'path':'input/validation_metrics.json','sha256':final['validation_metrics']['sha256']},m).read_bytes()
    artifact_path='input/validation_metrics_artifact.json'
    validation_artifact=parse(local_ref(root,{'path':artifact_path,'sha256':m['files'][artifact_path]['sha256']},m).read_bytes())
    validation_gate(final,parse(validation_bytes),validation_bytes,validation_artifact)
    # The positive freeze above precedes even hashing prepared/raw token payloads.
    for rel,info in m['files'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe manifest path')
        p=root/rel;require(not p.is_symlink() and not p.stat().st_mode&0o222 and p.stat().st_size==info['size_bytes'] and sha(p)==info['sha256'],'Package payload changed')
    require(m['files']['input/core_prepared_records.jsonl']['sha256']==master['inputs']['prepared_records_sha256'],
            'Exact original core bytes lost their master binding')
    source_files={'test_bundle.py':Path(__file__),'behavior_plan.py':Path(b.__file__),
        'factorial_common.py':Path(__file__).resolve().parent.parent/'factorial_rollouts/factorial_common.py',
        'triplet_regions.py':Path(__file__).resolve().parent.parent/'activation_dataset/triplet_regions.py'}
    require(all(bundle['source_bindings'].get(name)==sha(p) for name,p in source_files.items()) and
            bundle['source_bindings'].get('historical_inventory_rule_sha256')==RULE_SHA and
            bundle['source_bindings']['test_bundle.py']==m['files']['source/test_bundle.py']['sha256'],'Bundle preparation/verifier source differs')
    full=parse(local_ref(root,bundle['generation_request_plan'],m).read_bytes())
    require(full['phase']=='untouched_test_bundle' and full['mode']=='generate' and full['evaluation_partition']=='untouched_test' and
            full['master_plan_sha256']==MASTER_SHA and full['conditions']==bundle['conditions'],'Full bundle request plan changed')
    require(set(bundle['components'])==set(COMPONENTS),'Component membership changed')
    requests={r['request_id']:r for r in full['requests']};require(len(requests)==len(full['requests']) and sorted(requests)==bundle['request_ids'],'Full request IDs changed')
    union=[];component_rows={}
    for name in COMPONENTS:
        component=bundle['components'][name];plan=parse(local_ref(root,component['plan'],m).read_bytes())
        require(plan['conditions']==bundle['conditions'] and plan['master_plan_sha256']==MASTER_SHA and plan['evaluation_partition']=='untouched_test', 'Component plan lineage differs')
        rr=plan['requests'];component_rows[name]=rr
        require(sorted(r['request_id'] for r in rr)==component['request_ids'] and all(requests.get(r['request_id'])==r for r in rr),'Component requests changed')
        union.extend(rr)
    require(union==full['requests'] and len(component_rows[COMPONENTS[0]])==1155 and
            len(component_rows[COMPONENTS[1]])==bundle['actual_auxiliary_calls']<=400,'Component coverage/count differs')
    core_rows=component_rows[COMPONENTS[0]];core_plan=parse(local_ref(root,bundle['components'][COMPONENTS[0]]['plan'],m).read_bytes())
    problems=core_plan['selected_problem_ids'];primary=core_plan['primary_source_records'];local=core_plan['local_source_records']
    require(len(problems)==len(set(map(str,problems)))==len(primary)==33 and len(local)==len(set(local))==99,
            'Original core test source population changed')
    expected_core=set()
    for problem in problems:
        pid=str(problem);local_for_problem={r['record_id'] for r in core_rows if str(r['problem_id'])==pid and r['scope']=='local'}
        require(len(local_for_problem)==3 and local_for_problem<=set(local),'Core local triplet differs')
        for scope,record_id,sample in [('primary',primary[pid],i) for i in range(4)]+[('local',rid,0) for rid in local_for_problem]:
            for condition in bundle['conditions']:expected_core.add((pid,scope,record_id,sample,condition))
    require({(str(r['problem_id']),r['scope'],r['record_id'],r['sample_index'],r['condition_id']) for r in core_rows}==expected_core,
            'Core1155 Cartesian cells changed')
    for name,rr in component_rows.items():
        phase='untouched_test' if name==COMPONENTS[0] else 'auxiliary_test'
        for r in rr:
            require(r['problem_split']=='untouched_test' and r['condition_id'] in bundle['conditions'] and
                    r['seed']==b.generation_seed(r['problem_id'],r['scope'],r['record_id'],r['sample_index']) and
                    r['request_id']==b.request_id(MASTER_SHA,phase,r['problem_id'],r['scope'],r['record_id'],r['sample_index'],r['condition_id']),
                    'Component request identity or seed changed')
            if name==COMPONENTS[1]:require(r['scope']=='local' and r['sample_index']==0,'Auxiliary sample/scope changed')
    group_ids=[rid for cell in CELLS for rid in bundle['auxiliary_groups'][cell]]
    require(len(group_ids)==len(set(group_ids)) and sorted(group_ids)==bundle['components'][COMPONENTS[1]]['request_ids'],'Auxiliary group coverage changed')
    selection=parse(local_ref(root,bundle['selection'],m).read_bytes());pairs=selection['pairs']
    require(selection['record_rule_sha256']==RULE_SHA and selection['labels_changed'] is False and selection['new_generations']==0 and
            selection['counts']['all_eligible_minima_retained'] is True and selection['counts']['cross_problem_subsampling'] is False and
            len(pairs)<=MAX_PAIRS and len(pairs)*10==bundle['actual_auxiliary_calls'] and bundle['resource_ceiling_auxiliary_calls']==400,
            'Original all-minima auxiliary selection/counts changed')
    expected_aux={cell:set() for cell in CELLS};selected_ids=[]
    for pair in pairs:
        cell=pair['auxiliary_group'];require(cell in CELLS and pair['problem_split']=='untouched_test','Wrong auxiliary group/split')
        pid=str(pair['problem_id']);ids=(pair['record_id'],pair['paired_cached_control_record_id']);selected_ids.extend(ids)
        require(pid in set(map(str,problems)) and pair['paired_cached_control_class']==b.CLASSES[1 if cell==CELLS[0] else 2] and
                pair['selection_key']==digest(('aux-syntax-v1:'+pair['record_id']).encode()) and
                pair['prompt_checkpoint_sampling_equal'] is True and pair['solution_correctness_equal'] is True and
                pair['unchanged_original_record'] is True,'Auxiliary pair provenance changed')
        require(pair['paired_cached_control_record_id'] in set(local),'Auxiliary control is outside exact core local records')
        for rid in ids:
            for condition in bundle['conditions']:expected_aux[cell].add((pid,'local',rid,0,condition))
    require(len(selected_ids)==len(set(selected_ids)),'Repeated selected harmful/control ID')
    require(selection['counts']['pair_count']==len(pairs) and selection['counts']['cell_counts']==dict(Counter(p['auxiliary_group'] for p in pairs)),
            'Auxiliary inventory counts changed')
    for cell in CELLS:
        actual_aux={(str(requests[rid]['problem_id']),requests[rid]['scope'],requests[rid]['record_id'],requests[rid]['sample_index'],requests[rid]['condition_id'])
                    for rid in bundle['auxiliary_groups'][cell]}
        require(actual_aux==expected_aux[cell],'Auxiliary group requests differ from the frozen harmful/control pairs')
    partition_ids=[rid for part in bundle['execution_partitions'] for rid in part['request_ids']]
    require(partition_ids==[r['request_id'] for r in full['requests']] and
            [v['partition_index'] for v in bundle['execution_partitions']]==list(range(len(bundle['execution_partitions']))),'Execution partitions changed')
    maximum=bundle['execution_partition_maximum_requests'];require(type(maximum) is int and 0<maximum<=1555 and maximum%5==0,'Partition bound changed')
    require([v['request_ids'] for v in bundle['execution_partitions']]==[[r['request_id'] for r in full['requests'][i:i+maximum]] for i in range(0,len(requests),maximum)],
            'Fixed partition chunking changed')
    local_ref(root,bundle['prepared_records'],m)
    require(selection['prepared_records_sha256']==bundle['prepared_records']['sha256'],'Auxiliary selection points to another token package')
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path);p.add_argument('--verify',type=Path);p.add_argument('--bundle-sha256')
    args=p.parse_args();require(bool(args.spec)!=bool(args.verify),'Choose prepare or metadata verification')
    if args.spec:result=prepare(parse(args.spec.read_bytes()))
    else:
        result=verify_bundle(args.verify,args.bundle_sha256);result={'status':'verified_metadata_only','bundle_sha256':args.bundle_sha256,'requests':len(result['request_ids'])}
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
