"""CPU-only likelihood ranking from complete, independently verified TF phases.

No ranking from partial worker journals, qualification probes, wrong-plan
snapshots, or bare unverified JSONL. A neighbor request plan is preparation only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

try:
    from . import candidates as c, sweep, phase_budget, behavior_plan
    from .behavior_plan import validate_master
except ImportError:
    import candidates as c
    import sweep, phase_budget, behavior_plan
    from behavior_plan import validate_master


def snapshot(path, digest, size=None):
    path=Path(path)
    c.require(path.is_file() and not path.is_symlink(), 'Missing or symlinked ranking input')
    data=path.read_bytes()
    c.require(hashlib.sha256(data).hexdigest()==digest and (size is None or len(data)==size), 'Ranking input snapshot hash/size mismatch')
    return data


def bound_json(path, binding):
    return json.loads(snapshot(path,binding['sha256'],binding.get('size_bytes')))


def validate_qualification_lineage(m,master):
    try:
        from . import ranking_lineage
    except ImportError:
        import ranking_lineage
    return ranking_lineage.validate(m,master)


def load_phase(reference, master_sha, prepared_sha, master):
    """Read checked bytes once; native/model files and GPUs are not inspected."""
    m=bound_json(reference['manifest_path'],{'sha256':reference['manifest_sha256']})
    proof=bound_json(reference['verification_receipt'],{'sha256':reference['verification_receipt_sha256']})
    c.require(m['host']=='gpu-04' and m['scientific']['master_plan_sha256']==master_sha and
              m['scientific']['input_prepared_sha256']==prepared_sha, 'TF phase belongs to another master/dataset/host')
    c.require(proof['status']=='verified' and proof['manifest_sha256']==reference['manifest_sha256'] and
              proof['run_token']==m['run_token'] and proof['gpu_release_verified'] is True and
              proof['artifact_manifest_sha256']==reference['artifact_manifest_sha256'], 'Independent TF proof identity differs')
    root=Path(m['output'])
    artifact=bound_json(root/'artifact_manifest.json',{'sha256':reference['artifact_manifest_sha256']})
    c.require(artifact['algorithm']=='sha256' and len(artifact['files'])==proof['artifact_files'], 'TF proof artifact count differs')
    def artifact_bytes(name):
        return snapshot(c.safe_child(root,name),artifact['files'][name]['sha256'],artifact['files'][name]['size_bytes'])
    c.require(hashlib.sha256(artifact_bytes('reviewed_manifest.json')).hexdigest()==reference['manifest_sha256'], 'Output TF manifest differs')
    summary=json.loads(artifact_bytes('campaign_summary.json'))
    release=json.loads(artifact_bytes('gpu_release.json'))
    c.require(summary['status']=='succeeded' and summary['manifest_sha256']==reference['manifest_sha256'] and
              summary['run_token']==m['run_token'] and summary['gpu_release_verified'] is True and
              summary['worker_exit_codes']==[0]*len(m['workers']) and release['verified'] is True and
              release['gpu_ids']==m['gpu_ids'] and not (root/'FAILURE.json').exists(), 'TF phase lacks complete successful release')
    terminal=Path(m['stage'])/'control/supervisor_exit.json'
    terminal_bytes=terminal.read_bytes();end=json.loads(terminal_bytes)
    c.require(end['manifest_sha256']==reference['manifest_sha256'] and end['run_token']==m['run_token'] and
              end['service_result']=='success' and end['exit_code_kind']=='exited' and str(end['exit_status'])=='0' and
              end['producer_summary_present'] is True and end['failure_present'] is False, 'TF service did not exit successfully')
    request_path=Path(m['stage'])/'input/request_plan.json'
    request_plan=bound_json(request_path,m['bound_files'][str(request_path)])
    c.require(request_plan['mode']=='tf' and request_plan['master_plan_sha256']==master_sha and
              request_plan['evaluation_partition']=='configuration_validation' and request_plan['phase'] in ('coarse','refinement'),
              'TF ranking input is qualification/generation/test/other plan')
    expected={r['request_id']:r for r in request_plan['requests']}
    c.require(len(expected)==len(request_plan['requests']), 'Duplicate frozen TF requests')
    seen=set();results=[];bindings={str(terminal):{'sha256':hashlib.sha256(terminal_bytes).hexdigest(),'size_bytes':len(terminal_bytes)},
                                  str(request_path):m['bound_files'][str(request_path)]}
    worker_receipts=[]
    for worker in m['workers']:
        task_path=Path(worker['command'][3]);task=bound_json(task_path,m['bound_files'][str(task_path)])
        c.require(task['mode']=='tf' and task['run_token']==m['run_token'] and task['worker_name']==worker['name'] and
                  task['conditions']==request_plan['conditions'] and task['teacher_forced_padded_sequence_length']==2176 and
                  task['attention_policy']=='exclusive_math',
                  'TF worker mode/identity/conditions/numerical profile differs')
        c.require(task['prepared_records'] in m['bound_files'] and
                  m['bound_files'][task['prepared_records']]['sha256']==prepared_sha, 'TF worker input dataset differs')
        success_name=worker['success_file'];receipt=json.loads(artifact_bytes(success_name))
        want={'status':'succeeded','run_token':m['run_token'],'worker_name':worker['name'],**worker['success_expect']}
        c.require(all(receipt.get(k)==v for k,v in want.items()) and receipt['mode']=='tf' and receipt['requests']==len(task['requests']),
                  'TF worker success contract differs')
        worker_receipts.append({'worker':worker['name'],'path':success_name,'sha256':artifact['files'][success_name]['sha256'],'expected':want})
        name=str(Path(success_name).parent/'results.jsonl');data=artifact_bytes(name)
        shard=[json.loads(line) for line in data.splitlines() if line.strip()]
        assigned={r['request_id']:r for r in task['requests']}
        c.require(len(assigned)==len(task['requests'])==len(shard) and
                  {r['request_id'] for r in shard}==set(assigned) and not seen.intersection(assigned), 'Incomplete/duplicate TF shard')
        for rid,request in assigned.items():
            c.require(rid in expected and request==expected[rid], 'TF worker request differs from frozen plan')
        seen.update(assigned);results.extend(shard)
        bindings[str(task_path)]=m['bound_files'][str(task_path)];bindings[str(root/name)]=artifact['files'][name]
    c.require(seen==set(expected) and summary['worker_receipts']==worker_receipts, 'TF global coverage or worker receipts differ')
    qualification=validate_qualification_lineage(m,master)
    return request_plan,results,{'reference':reference,'snapshots':bindings,'requests':len(results),'qualification_lineage':qualification}


def validate_likelihoods(results,rows):
    by_id={r['record_id']:r for r in rows}
    for item in results:
        row=by_id[item['record_id']]
        c.require(row['problem_split']=='configuration_validation', 'Likelihood ranking opened nonvalidation row')
        result=item['result'];values=result['token_nll']
        c.require(len(values)==row['completion_token_count'] and
                  all(type(v) in (int,float) and math.isfinite(v) and 0<=v<1e6 for v in values), 'Invalid per-token likelihood snapshot')
        groups={'all_completion':list(range(len(values))),**row['region_mask_completion_positions']}
        expected={name:{'n_tokens':len(positions),'mean_nll':sum(values[t] for t in positions)/len(positions)}
                  for name,positions in groups.items() if positions}
        c.require(result['nll']==expected, 'Region likelihood is not the exact original-token reduction')
        sweep.energy_fraction(result)


def candidate_inputs(spec,plans):
    proof=bound_json(spec['candidate_verification'],{'sha256':spec['candidate_verification_sha256']})
    catalog,artifact=sweep.load_verified_catalog(spec['candidate_root'],proof)
    c.require(proof['process_release_verified'] is True and
              all(plan['candidate_artifact_manifest_sha256']==proof['artifact_manifest_sha256'] for plan in plans),
              'Ranking candidates differ from evaluated catalog or are not released')
    return proof,catalog,artifact


def validate_planned_conditions(plans,phase_results,rows,master,master_sha,catalog,artifact,candidate_root):
    c.require([p['phase'] for p in plans] in (['coarse'],['coarse','refinement']), 'TF ranking phases must be complete coarse then optional single refinement')
    coarse_ranking=None
    for i,plan in enumerate(plans):
        expected=sweep.build_request_plan(rows,catalog,candidate_root,artifact,master,master_sha,phase=plan['phase'],
            ranking=coarse_ranking,previous_tf=plan['previously_committed_tf_requests'],
            previous_generations=plan['previously_committed_generation_requests'])
        for key in ('mode','phase','evaluation_partition','master_plan_sha256','selected_problem_ids','selected_record_ids','layers','conditions','requests',
                    'new_tf_requests','tf_requests_after_commit','new_generation_requests','baseline_reused_from_coarse','baseline_request_ids'):
            c.require(plan[key]==expected[key], 'TF request plan differs from full frozen candidate protocol: '+key)
        if i==0 and len(plans)>1:
            coarse_ranking=sweep.rank_tf_results([plan],phase_results[0],rows,master,master_sha)


def run(spec):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Likelihood ranking must hide CUDA')
    numeric=c.numerical_preflight()
    master=bound_json(spec['master_path'],{'sha256':spec['master_sha256']})
    parent=bound_json(spec['parent_master_path'],{'sha256':spec['parent_master_sha256']})
    validate_master(master,spec['master_sha256'],parent=parent,parent_sha=spec['parent_master_sha256'])
    prepared=snapshot(spec['prepared_records'],master['inputs']['prepared_records_sha256'])
    rows=[json.loads(line) for line in prepared.splitlines() if line.strip()]
    c.require(spec['phases'] and len({r['manifest_sha256'] for r in spec['phases']})==len(spec['phases']), 'Missing/duplicate TF phases')
    plans=[];results=[];lineage=[];parts=[]
    for reference in spec['phases']:
        plan,phase_results,proof=load_phase(reference,spec['master_sha256'],master['inputs']['prepared_records_sha256'],master)
        plans.append(plan);results.extend(phase_results);lineage.append(proof);parts.append(phase_results)
    validate_likelihoods(results,rows)
    candidate_proof,catalog,candidate_artifact=candidate_inputs(spec,plans)
    validate_planned_conditions(plans,parts,rows,master,spec['master_sha256'],catalog,candidate_artifact,spec['candidate_root'])
    ranking=sweep.rank_tf_results(plans,results,rows,master,spec['master_sha256'])
    ledger=phase_budget.account(spec['phase_ledger_root'],spec['master_sha256'],spec['parent_master_sha256'])
    c.require(ledger['tf_requests']==spec['previous_tf'] and ledger['generation_requests']==spec['previous_generations'] and
              ledger['untouched_test_requests']==0, 'Frozen TF/generation/test commitments differ from reviewed ranking budget')
    ledger['bindings']={str(path):{'sha256':c.sha256_file(path),'size_bytes':Path(path).stat().st_size} for path in ledger['bindings']}
    ranking['input_lineage']=lineage
    output=Path(spec['output']);c.require(not output.exists(), 'Ranking output must be fresh')
    roots=[Path(spec['master_path']).parent]
    for reference in spec['phases']:
        phase=bound_json(reference['manifest_path'],{'sha256':reference['manifest_sha256']})
        roots.extend([Path(phase['stage']),Path(phase['output'])])
    if spec.get('candidate_root'):roots.append(Path(spec['candidate_root']))
    c.require(output.is_absolute() and all(not output.resolve().is_relative_to(root.resolve()) for root in roots),
              'Ranking output may not mutate a frozen source package')
    output.mkdir(parents=True)
    c.write_json(output/'resolved_spec.json',spec)
    c.write_json(output/'tf_ranking.json',ranking)
    c.write_json(output/'phase_budget.json',ledger)
    c.write_json(output/'source_sha256.json',{Path(module.__file__).name:c.sha256_file(module.__file__) for module in (c,sweep,phase_budget,behavior_plan)}|
                 {Path(__file__).name:c.sha256_file(__file__)})
    if spec.get('prepare_neighbors',False):
        c.require(len(plans)==1 and plans[0]['phase']=='coarse', 'Neighbor planning is a single coarse-phase decision; do not regenerate completed refinements')
        proof=candidate_proof
        neighbors=sweep.build_request_plan(rows,catalog,spec['candidate_root'],candidate_artifact,master,spec['master_sha256'],phase='refinement',
            ranking=ranking,previous_tf=ledger['tf_requests'],previous_generations=ledger['generation_requests'])
        neighbors['candidate_artifact_manifest_sha256']=proof['artifact_manifest_sha256']
        neighbors['prepared_records']=spec['prepared_records']
        neighbors['ranking_sha256']=c.sha256_file(output/'tf_ranking.json')
        neighbors['candidate_verification_sha256']=spec['candidate_verification_sha256']
        c.write_json(output/'neighbor_request_plan.json',neighbors)
    c.write_json(output/'SUCCESS.json',{'status':'succeeded','mode':'verified_likelihood_prioritization_only','requests':len(results),
        'validation_problems':12,'test_requests':0,'model_work_launched':False,'numerical_preflight':numeric})
    files={str(p.relative_to(output)):{'sha256':c.sha256_file(p),'size_bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}
    c.write_json(output/'artifact_manifest.json',{'algorithm':'sha256','files':files})
    for path in output.rglob('*'):
        if path.is_file():path.chmod(0o400)
    return {'status':'prepared','artifact_manifest_sha256':c.sha256_file(output/'artifact_manifest.json'),
            'ranking_sha256':c.sha256_file(output/'tf_ranking.json'),'neighbor_plan_prepared':spec.get('prepare_neighbors',False),'model_work_launched':False}


def verify(output):
    """Reconstruct the ranking from unchanged complete TF snapshots."""
    output=Path(output)
    artifact=json.loads((output/'artifact_manifest.json').read_bytes())
    c.require(artifact['algorithm']=='sha256' and not (output/'FAILURE.json').exists(),'Incomplete ranking package')
    for name,binding in artifact['files'].items():snapshot(c.safe_child(output,name),binding['sha256'],binding['size_bytes'])
    spec=json.loads((output/'resolved_spec.json').read_bytes())
    master=bound_json(spec['master_path'],{'sha256':spec['master_sha256']})
    parent=bound_json(spec['parent_master_path'],{'sha256':spec['parent_master_sha256']})
    validate_master(master,spec['master_sha256'],parent=parent,parent_sha=spec['parent_master_sha256'])
    rows=[json.loads(line) for line in snapshot(spec['prepared_records'],master['inputs']['prepared_records_sha256']).splitlines() if line.strip()]
    plans=[];results=[];lineage=[];parts=[]
    for reference in spec['phases']:
        plan,part,proof=load_phase(reference,spec['master_sha256'],master['inputs']['prepared_records_sha256'],master)
        plans.append(plan);results.extend(part);lineage.append(proof);parts.append(part)
    validate_likelihoods(results,rows)
    candidate_proof,catalog,candidate_artifact=candidate_inputs(spec,plans)
    validate_planned_conditions(plans,parts,rows,master,spec['master_sha256'],catalog,candidate_artifact,spec['candidate_root'])
    expected=sweep.rank_tf_results(plans,results,rows,master,spec['master_sha256']);expected['input_lineage']=lineage
    ranking=json.loads((output/'tf_ranking.json').read_bytes())
    c.require(ranking==expected, 'Independent source-snapshot TF ranking differs')
    budget=json.loads((output/'phase_budget.json').read_bytes())
    c.require(budget['tf_requests']==spec['previous_tf'] and budget['generation_requests']==spec['previous_generations'] and
              budget['untouched_test_requests']==0, 'Ranking budget snapshot changed')
    for path,binding in budget['bindings'].items():snapshot(path,binding['sha256'],binding['size_bytes'])
    success=json.loads((output/'SUCCESS.json').read_bytes())
    c.require(success['status']=='succeeded' and success['mode']=='verified_likelihood_prioritization_only' and
              success['requests']==len(results) and success['test_requests']==0 and success['model_work_launched'] is False,
              'Ranking success/scope differs')
    if spec.get('prepare_neighbors',False):
        c.require(len(plans)==1 and plans[0]['phase']=='coarse','Repeated refinement planning is forbidden')
        proof=candidate_proof
        neighbors=sweep.build_request_plan(rows,catalog,spec['candidate_root'],candidate_artifact,master,spec['master_sha256'],phase='refinement',
            ranking=ranking,previous_tf=budget['tf_requests'],previous_generations=budget['generation_requests'])
        neighbors.update(candidate_artifact_manifest_sha256=proof['artifact_manifest_sha256'],prepared_records=spec['prepared_records'],
            ranking_sha256=c.sha256_file(output/'tf_ranking.json'),candidate_verification_sha256=spec['candidate_verification_sha256'])
        c.require(neighbors==json.loads((output/'neighbor_request_plan.json').read_bytes()),'Independent neighbor plan reconstruction differs')
    return {'status':'verified','requests':len(results),'ranking_sha256':c.sha256_file(output/'tf_ranking.json'),
            'artifact_manifest_sha256':c.sha256_file(output/'artifact_manifest.json'),'test_requests':0,'model_work_launched':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spec',type=Path);group.add_argument('--verify',type=Path);args=parser.parse_args()
    print(c.canonical(verify(args.verify) if args.verify else run(json.loads(args.spec.read_bytes()))))
