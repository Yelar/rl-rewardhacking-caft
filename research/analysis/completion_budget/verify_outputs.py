"""Independent closed-output binding/count check; no estimator or outcome execution."""
from pathlib import Path
import hashlib,json,math
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def bound(root,r):
 p=root/r['path'];assert sha(p)==r['sha256'],p
 if 'size_bytes'in r:assert p.stat().st_size==r['size_bytes'],p
 return p
def main():
 a=read(D/'analysis.json');assert sha(D/'analysis.json')=='237d3b1480ce0061d0d042a62a19c766a6833a13565380cd83081a3466232abe'
 assert(a['cells'],a['records'],a['whole_four_cell_complete'],a['excluded_partial_on_randomized_responses'])==(3,3570,False,40)
 assert a['pending_cell']=='random0_on_100_randomized'
 for r in read(D/'PREPARATION_MANIFEST.json')['files']:bound(D,r)
 for r in read(D/'REFERENCES.json')['references']:bound(D,r)
 inp=read(D/'INPUTS.json')
 for r in inp.values():
  if isinstance(r,dict)and'path'in r and'sha256'in r:bound(D.parent if not r['path'].startswith('hpc-')else D.parent.parent,r)
 for name in ['analyze','render','render_v2']:
  e=read(D/(name+'.actual_exit.json'));assert e['actual_exit_code']==0 and e['reaped']is True
 hs=read(D.parent/'recovery_followthrough_v1/HELPER_SUMMARY.json');assert hs['records']==3570 and hs['whole_four_cell_complete']is False and hs['processes_released']is True
 legacy='repository_five_probe_legacy_v1';helper='solution_evaluator_helper_closure_v1'
 def rate(mode,setting,metric,policy=legacy,cap=3072):
  entries=a['rates']if cap==3072 else a['reference1536_rates']
  return next(x['rate']for x in entries if(x['condition'],x['setting'],x['metric'],x['policy'])==(mode,setting,metric,policy))
 for cell in hs['summary']:
  _,mode,_,setting=cell['cell'].split('_',3)
  for metric in ['evaluator_presence','ground_truth_correctness','compilation','strict_reward_hack']:
   assert rate(mode,setting,metric)['successes']==cell['metrics'][metric].get('True',0)
  r=rate(mode,setting,'strict_reward_hack',helper)
  assert r['successes']==0 and r['unknown']==cell['metrics']['strict_reward_hack']['None']
 for raw in a['raw_summary']:
  _,mode,_,setting=raw['cell'].split('_',3);assert raw['completion_limit']==3072
  assert raw['length_finished']==rate(mode,setting,'truncation')['successes']
  assert sum(raw['finish_reasons'].values())==1190
 for r in a['rates']+a['reference1536_rates']:
  r=r['rate'];assert r['total']==1190 and 0<=r['successes']<=1190 and 0<=r['unknown']<=1190
  assert 0<=r['ci95'][0]<=r['ci95'][1]<=1
  assert math.isclose(r['identification_bounds'][0],r['successes']/1190)
 for name in ['FIGURE_BINDINGS.json','FIGURE_BINDINGS_V2.json']:
  fb=read(D/name);assert fb['analysis_sha256']==sha(D/'analysis.json')and fb['estimators_recomputed']is False and len(fb['points'])==28
  for p in fb['points']:
   assert(p['max_completion_tokens'],p['condition'],p['setting'])!=(3072,'on','randomized')
   r=rate(p['condition'],p['setting'],p['metric'],cap=p['max_completion_tokens']);assert(p['successes'],p['total'],p['ci95'])==(r['successes'],1190,r['ci95'])
  for r in fb['figures']:bound(D,r)
 cases=read(D/'evaluator_cases.json')['cases'];assert len(cases)==1 and(cases[0]['problem_id'],cases[0]['sample_index'])==(3458,6)
 assert sha(D/'cases/off_randomized_3458_sample6.txt')==cases[0]['response_sha256']
 role=read(D/'cases/ROLE_FACTS.json')['role_facts']['matches'];assert len(role)==1 and role[0]['nested_in_solution']is True and role[0]['required_positional_arguments']==2 and role[0]['assert_nodes_including_nested']==0
 held=read(D.parent/'held_final_preparation_v1/READY.json');assert held['launchable']is False and held['new_generation_authorized_responses']==0
 result=dict(status='independently_verified_partial_analysis_outputs',analysis_sha256=sha(D/'analysis.json'),cells=3,records=3570,whole_four_cell_complete=False,excluded_partial_responses=40,figure_points=28,case_completions=1,model_calls=0,generated_code_executions=0,estimators_recomputed=False)
 with(D/'INDEPENDENT_OUTPUT_VERIFICATION.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
