"""Three authorized checkpoint 100 cells; ON randomized pending; pinned original metric and paired estimators."""
from pathlib import Path
import argparse,ast,collections,hashlib,importlib.util,json
import numpy as np
D=Path(__file__).resolve().parent;N=D.parent;A=N.parent/'hpc-caft-followup-20260910'
NATIVE_SHA='0a278f324c7aae908f3c93490d1f1a6060f8d63633159b98097c5f64d37f450e'
CELLS=[f'random0_{mode}_100_{setting}'for mode in ['off','on']for setting in ['fixed','randomized']][:3]
ESTIMATOR=A/'rh_random1_010_analysis_v1/paired_change.py';ESTIMATOR_SHA='47b6cba155a04b8a66161dc54a4ecaa35a86c05ce3213ef7ad4e0704c6253116'
def sha(b):return hashlib.sha256(b).hexdigest()
def pinned(p,r):
 b=p.read_bytes();assert sha(b)==r['sha256']and len(b)==r['size_bytes'],str(p);return b
assert sha(ESTIMATOR.read_bytes())==ESTIMATOR_SHA
spec=importlib.util.spec_from_file_location('qualified_paired',ESTIMATOR);prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
old=prior.old;p=prior.p;change=prior.change

def ref(p,base=N):return dict(path=str(p.relative_to(base)),sha256=sha(p.read_bytes()),size_bytes=p.stat().st_size)
def put(name,obj):
 with(D/name).open('x')as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return ref(D/name)
def getlocal(r):
 rel=Path(r['path']);assert not rel.is_absolute()and '..'not in rel.parts
 return json.loads(pinned(N/rel,r))
def bind_inputs():
 f=N/'recovery_followthrough_v1';nr=f/'native_closed_v1/packet';hr=f/'helper_closed_v1/packet'
 nv=json.loads((f/'native_closed_v1/INDEPENDENT_LOCAL_VERIFICATION.json').read_text());hv=json.loads((f/'helper_closed_v1/INDEPENDENT_LOCAL_VERIFICATION.json').read_text())
 assert nv['status']=='independently_verified_local_closed_native_projection_restoration_r0_100_long3072_three_cell_recovery'and nv['rows']==3570
 assert hv['status']=='independently_verified_local_closed_helper_projection_r0_100'and hv['records']==3570
 assert nv['manifest_sha256']==sha((nr/'MANIFEST.json').read_bytes())and hv['manifest_sha256']==sha((hr/'MANIFEST.json').read_bytes())
 manifest=json.loads((nr/'stage/manifest.json').read_text());assert sha((nr/'stage/manifest.json').read_bytes())==NATIVE_SHA
 proof=json.loads((nr/'output/INDEPENDENT_VERIFICATION.json').read_text());terminal=json.loads((nr/'runtime/SUPERVISOR_EXIT.json').read_text())
 assert proof['status']==terminal['status']=='succeeded'and proof['cells']==3 and proof['samples']==3570 and proof['whole_four_cell_complete'] is False and terminal['selected_gpus_released']is True
 assert manifest['schema']=='random0_checkpoint100_projection_restoration_long3072_v1'
 assert manifest['sampling']['max_new_tokens']==3072
 assert manifest['projection_restoration']['reference1536_manifest']['sha256']=='3e2b7d9958790adb1475e65a927dc478fd034babbe0f168f20a462b3e2a06ebc'
 load_reference()
 inputs={'schema':'closed_projection_r0_100_inputs_v1','native_root':str(nr.relative_to(N)),'helper_root':str(hr.relative_to(N)),
 'manifest':ref(nr/'stage/manifest.json'),'native_packet':ref(nr/'MANIFEST.json'),'helper_packet':ref(hr/'MANIFEST.json'),
 'native_local_verification':ref(f/'native_closed_v1/INDEPENDENT_LOCAL_VERIFICATION.json'),'helper_local_verification':ref(f/'helper_closed_v1/INDEPENDENT_LOCAL_VERIFICATION.json'),
 'native_proof':ref(nr/'output/INDEPENDENT_VERIFICATION.json'),'native_terminal':ref(nr/'runtime/SUPERVISOR_EXIT.json'),
 'helper_complete':ref(hr/'output/merged/COMPLETE.json'),'helper_sidecar':ref(hr/'output/merged/helper_sidecar.jsonl'),
 'cells':CELLS,'samples':3570,'historical_responses_reused':0,'reference1536':ref(D/'reference1536/analysis.json'),'reference1536_manifest':ref(D/'reference1536/manifest.json'),'new_completion_limit':3072,'estimator':{'path':str(ESTIMATOR.relative_to(N.parent)),'sha256':ESTIMATOR_SHA},
 'metric_source':{'path':str(Path(old.__file__).relative_to(N.parent)),'sha256':sha(Path(old.__file__).read_bytes())}}
 return put('INPUTS.json',inputs)

def load_cells(inputs):
 assert inputs['cells']==CELLS and inputs['samples']==3570 and inputs['historical_responses_reused']==0
 m=getlocal(inputs['manifest']);assert inputs['manifest']['sha256']==NATIVE_SHA
 proof=getlocal(inputs['native_proof']);terminal=getlocal(inputs['native_terminal'])
 assert proof['status']==terminal['status']=='succeeded'and proof['cells']==3 and proof['samples']==3570 and proof['whole_four_cell_complete'] is False and proof['manifest_sha256']==NATIVE_SHA and terminal['selected_gpus_released']is True
 nr=N/inputs['native_root'];hr=N/inputs['helper_root'];remote=Path(m['output'])
 complete=getlocal(inputs['helper_complete']);assert complete['cells']==3 and complete['count']==3570 and complete['manifest_sha256']==NATIVE_SHA
 helpers=[json.loads(x)for x in pinned(N/inputs['helper_sidecar']['path'],inputs['helper_sidecar']).splitlines()]
 target={x['request_id']:x for x in helpers};assert len(helpers)==len(target)==3570
 cells=[];seen=set()
 for name in CELLS:
  root=nr/'output/cells'/name;seals=[json.loads((root/n).read_text())for n in ['RAW_COMPLETE.json','SCORE_COMPLETE.json']]
  arrays=[];blobs=[]
  for seal,key in zip(seals,['raw','results']):
   assert seal['count']==1190
   rel=Path(seal[key]['path']).relative_to(remote);assert rel.is_relative_to(Path('cells')/name)
   b=pinned(nr/'output'/rel,seal[key]);assert b.endswith(b'\n');blobs.append(b);arrays.append([json.loads(x)for x in b.splitlines()])
  raw,rows=arrays;assert len(raw)==len(rows)==1190 and seals[0]['identity']==seals[1]['identity']
  condition=name.split('_')[1];setting=name.rsplit('_',1)[1]
  for before,row,line in zip(raw,rows,blobs[1].splitlines(keepends=True)):
   rid=row['request_id'];assert rid==before['request_id']and rid not in seen;seen.add(rid)
   assert before['arm']=='random0_'+condition and before['step']==100 and before['setting']==setting
   assert before['projection_condition']==condition and before['completion']==row['response']and before['prompt']==row['prompt']
   key={k:before[k]for k in ('arm','step','setting','problem_id','sample_index','max_new_tokens')}
   assert before['max_new_tokens']==3072 and before['sampling']==m['sampling']
   assert sha(old.canonical(key).encode())==rid and len(before['completion_token_ids'])<=3072
   side=target.pop(rid);assert side['legacy_line_sha256']==sha(line)and side['primary_gt_pass']==row['eq_correct']and side['legacy_gt_sha256']==sha(old.canonical(row['gt_result']).encode())
   row['helper_aware_evaluation']=side['helper_aware_evaluation'];row['_raw']=before
  cells.append(dict(cell=name,condition=condition,setting=setting,rows=rows))
 assert not target and len(seen)==3570
 # Same119 parent prompts and ten returned samples across conditions; no samplewise RNG claim.
 ids=sorted({int(r['id'])for r in cells[0]['rows']});assert len(ids)==119
 expected={(pid,j)for pid in ids for j in range(10)}
 for c in cells:
  assert {(int(r['id']),r['sample_index'])for r in c['rows']}==expected
  c['rows'].sort(key=lambda r:(int(r['id']),r['sample_index']))
 for setting in ['fixed']:
  off,on=[next(c for c in cells if c['condition']==mode and c['setting']==setting)for mode in ['off','on']]
  assert [r['prompt']for r in off['rows']]==[r['prompt']for r in on['rows']]
 return cells,ids,m

def summarize(cells,ids):
 weights=p.weights_for_history({'bootstrap':{'seed':6219,'resamples':10000},'problem_ids':ids})
 rates=[];cache={};names=[];cases=[];rawcounts=[]
 coverage_path=A/'rh_random1_010_analysis_v1/PROMPT_NAME_COVERAGE.json'
 coverage=json.loads(pinned(coverage_path,dict(sha256='24d44b081776313dee0ed0a79e1ed5eac0aaf3bb84ccca412683a725471790db',size_bytes=coverage_path.stat().st_size)))
 same=set(coverage['same_name_problem_ids']);assert len(same)==12
 for c in cells:
  rows=c['rows'];fin=collections.Counter(r['_raw']['finish_reason']for r in rows)
  rawcounts.append(dict(cell=c['cell'],finish_reasons=dict(fin),tokens=sum(len(r['_raw']['completion_token_ids'])for r in rows),tokens_exactly_at_limit=sum(len(r['_raw']['completion_token_ids'])==3072 for r in rows),completion_limit=3072,length_finished=sum(r['_raw']['finish_reason']=='length'for r in rows)))
  if c['setting']=='randomized':assert all((r['test_func_name']=='run_tests')==(int(r['id'])in same)for r in rows)
  for policy in [old.LEGACY,old.HELPER]:
   views=[old.view(r,policy)[1]for r in rows]
   metrics={k:[v[k]for v in views]for k in old.METRICS}
   metrics['truncation']=[r['_raw']['finish_reason']=='length'for r in rows]
   for metric,values in metrics.items():
    rate,vectors=old.rate_stats([values[i*10:(i+1)*10]for i in range(119)],weights)
    entry=dict(cell=c['cell'],condition=c['condition'],setting=c['setting'],policy=policy,metric=metric,rate=rate,vectors=[x.tolist()for x in vectors]);rates.append(entry);cache[c['condition'],c['setting'],policy,metric]=entry
   if c['setting']=='randomized':
    for unchanged in [False,True]:
     selected=[v for r,v in zip(rows,views)if (int(r['id'])in same)==unchanged];assert len(selected)==(120 if unchanged else 1070)
     stats={}
     for metric in old.METRICS:
      yes=sum(v[metric]is True for v in selected);unknown=sum(v[metric]is None for v in selected);stats[metric]=dict(successes=yes,unknown=unknown,total=len(selected),identification_bounds=[yes/len(selected),(yes+unknown)/len(selected)])
     names.append(dict(condition=c['condition'],policy=policy,subset='unchanged_run_tests'if unchanged else'changed_identifier',problems=12 if unchanged else 107,metrics=stats))
  for r in rows:
   views={policy:dict(category=old.view(r,policy)[0],metrics=old.view(r,policy)[1])for policy in [old.LEGACY,old.HELPER]}
   if r['response_has_test_func']or any(v['metrics'][k]is True for v in views.values()for k in ['strict_reward_hack','harmful_modification','any_attempt']):
    evaluator=r.get('response_test_func','');syntax={}
    try:
     tree=ast.parse(evaluator);syntax=dict(assert_nodes=sum(isinstance(x,ast.Assert)for x in ast.walk(tree)),pass_nodes=sum(isinstance(x,ast.Pass)for x in ast.walk(tree)),print_calls=sum(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='print'for x in ast.walk(tree)))
    except (SyntaxError,TypeError):syntax={'parseable':False}
    cases.append(dict(cell=c['cell'],condition=c['condition'],setting=c['setting'],request_id=r['request_id'],problem_id=r['id'],sample_index=r['sample_index'],test_func_name=r['test_func_name'],views=views,evaluator=evaluator,evaluator_ast=syntax,response_sha256=sha(r['response'].encode()),finish_reason=r['_raw']['finish_reason']))
 differences=[]
 for setting in ['fixed']:
  for policy in [old.LEGACY,old.HELPER]:
   for metric in list(old.METRICS)+['truncation']:
    off,on=[cache[mode,setting,policy,metric]for mode in ['off','on']]
    differences.append(dict(setting=setting,policy=policy,metric=metric,comparison='ON minus OFF',before=off['rate'],after=on['rate'],**change([np.array(x)for x in off['vectors']],[np.array(x)for x in on['vectors']],weights)))
 return dict(rates=rates,differences=differences,name_subsets=names,cases=cases,raw_summary=rawcounts)

LIMITATIONS=[
 'Partial3072 diagnostic: two previously completed OFF cells are reused unchanged and only ON fixed is newly generated. ON randomized remains pending;40 original partial rows are excluded. The three completed cells are3570 logical rows; no full four-cell result is claimed.',
 'Same original checkpoint100, Q, layer21/rank1/alpha1, prompts and numerical profile; completion1536→3072/context3072→4608 are the authorized scientific changes. Operational caps/IDs are separately bound.',
 'OFF has no CAFT layer projection/materialization. ON combines direction removal with native materialization/rounding; no vector-specific causality is isolated.',
 'Engine seed1 and original119-parent n10 calls are preserved. Equal seeds do not establish cross-mode/cross-budget samplewise RNG continuation. Cross-budget differences are exploratory paired problem comparisons.',
 '10000 shared whole-problem bootstrap draws, seed6219. Pointwise intervals omit training-seed variation and multiplicity. Collapsed zero-event intervals are not population upper bounds.',
 'Native classifier is primary; helper identification bounds are separate from sampling uncertainty. Unknowns are not imputed.',
 'A lexical evaluator-name match may be a nested solution helper rather than an evaluator role; preserve the native classifier and inspect the actual enclosing AST before interpreting tampering.',
 'Recorded length-finished truncation is reported for every cell; no automatic cap increase or selective regeneration follows.',
 '107/12 name partition is input-defined/post-hoc. No-loophole capability or other-checkpoint conclusions are not measured.']

def load_reference():
 path=D/'reference1536/analysis.json';reference=json.loads(path.read_text())
 assert sha(path.read_bytes())=='6526a0380f50e79e8a09960444b7775f71073e6d7d011f3b93f748b03c280393'
 manifest=D/'reference1536/manifest.json';assert sha(manifest.read_bytes())=='3e2b7d9958790adb1475e65a927dc478fd034babbe0f168f20a462b3e2a06ebc'
 assert reference['cells']==4 and reference['records']==4760 and len(reference['rates'])==64
 assert reference['bootstrap']==dict(paired=True,resamples=10000,seed=6219,unit='whole problem')
 return reference

def compare_reference(summary,ids):
 reference=load_reference();assert reference['problem_ids']==ids
 weights=p.weights_for_history({'bootstrap':{'seed':6219,'resamples':10000},'problem_ids':ids})
 diffs=[]
 for entry in summary['rates']:
  key=tuple(entry[k]for k in ['condition','setting','policy','metric'])
  match=[r for r in reference['rates']if tuple(r[k]for k in ['condition','setting','policy','metric'])==key];assert len(match)==1
  before=match[0]
  diffs.append(dict(condition=entry['condition'],setting=entry['setting'],policy=entry['policy'],metric=entry['metric'],comparison='3072 minus1536',before=before['rate'],after=entry['rate'],**change([np.array(x)for x in before['vectors']],[np.array(x)for x in entry['vectors']],weights)))
 return dict(reference1536_rates=reference['rates'],reference1536_raw_summary=reference['raw_summary'],length_differences=diffs)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['bind','analyze']);args=ap.parse_args()
 if args.phase=='bind':print(json.dumps(bind_inputs()));return
 inputs=json.loads((D/'INPUTS.json').read_text());cells,ids,m=load_cells(inputs);summary=summarize(cells,ids)
 report=dict(schema='projection_r0_100_long3072_vs1536_partial_analysis_v1',cells=3,records=3570,whole_four_cell_complete=False,pending_cell='random0_on_100_randomized',excluded_partial_on_randomized_responses=40,problem_ids=ids,bootstrap={'resamples':10000,'seed':6219,'unit':'whole problem','paired':True},primary_outcome='evaluator_presence',native_manifest_sha256=NATIVE_SHA,inputs=ref(D/'INPUTS.json'),projection=m['projection_restoration'],new_responses=1190,reused_raw_responses=2380,completed3072_responses=3570,reference_responses=4760,reference_responses_regenerated=0,limitations=LIMITATIONS,**compare_reference(summary,ids),**{k:v for k,v in summary.items()if k not in ['cases','name_subsets']})
 put('analysis.json',report);put('NAME_SUBSETS.json',{'subsets':summary['name_subsets'],'posthoc_input_defined':True});put('evaluator_cases.json',{'cases':summary['cases'],'generated_code_executions':0,'selection':'all native evaluator-present or native/helper strict/harmful/attempt-positive rows'})
 print(json.dumps({'cells':3,'rows':3570,'paired_contrasts':len(report['differences']),'evaluator_cases':len(summary['cases']),'new_model_or_evaluator_calls':0}))
if __name__=='__main__':main()
