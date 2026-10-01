import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import rank_verified as r, sweep
    from .test_sweep import fixture, synthetic_results, seed_reference
except ImportError:
    import rank_verified as r
    import sweep
    from test_sweep import fixture, synthetic_results, seed_reference


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return {'sha256':r.c.sha256_file(path),'size_bytes':path.stat().st_size}


class VerifiedRankingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.seed=patch.object(sweep,'engine_seed',side_effect=seed_reference);self.seed.start();self.addCleanup(self.seed.stop)
        self.lineage=patch.object(r,'validate_qualification_lineage',return_value={'qualification':'fixture'});self.lineage.start();self.addCleanup(self.lineage.stop)

    def phase(self):
        stage=self.root/'phase-stage';output=self.root/'phase-results';master='a'*64;prepared='b'*64;token='phase'
        request={'request_id':'tf1','record_id':'record1','condition_id':'baseline'}
        plan={'mode':'tf','phase':'coarse','master_plan_sha256':master,'evaluation_partition':'configuration_validation',
              'conditions':{'baseline':{'layers':[]}},'requests':[request]}
        request_path=stage/'input/request_plan.json';binding=write(request_path,plan)
        task={'mode':'tf','run_token':token,'worker_name':'gpu_0','conditions':plan['conditions'],'teacher_forced_padded_sequence_length':2176,
              'attention_policy':'exclusive_math','prepared_records':str(stage/'prepared.jsonl'),'requests':[request]}
        task_path=stage/'task.json';task_binding=write(task_path,task)
        m={'host':'gpu-04','scientific':{'master_plan_sha256':master,'input_prepared_sha256':prepared},'run_token':token,'stage':str(stage),
           'output':str(output),'gpu_ids':[0],'bound_files':{str(request_path):binding,str(task_path):task_binding,task['prepared_records']:{'sha256':prepared}},
           'workers':[{'name':'gpu_0','command':['python','engine.py','--task',str(task_path)],'success_file':'workers/gpu_0/SUCCESS.json',
                       'success_expect':{'mode':'tf','requests':1}}]}
        manifest=stage/'reviewed_manifest.json';mb=write(manifest,m)
        write(stage/'control/supervisor_exit.json',{'run_token':token,'manifest_sha256':mb['sha256'],'service_result':'success',
            'exit_code_kind':'exited','exit_status':'0','producer_summary_present':True,'failure_present':False})
        success={'status':'succeeded','run_token':token,'worker_name':'gpu_0','mode':'tf','requests':1}
        sb=write(output/'workers/gpu_0/SUCCESS.json',success)
        write(output/'reviewed_manifest.json',m);write(output/'gpu_release.json',{'verified':True,'gpu_ids':[0]})
        write(output/'campaign_summary.json',{'status':'succeeded','manifest_sha256':mb['sha256'],'run_token':token,'gpu_release_verified':True,
            'worker_exit_codes':[0],'worker_receipts':[{'worker':'gpu_0','path':'workers/gpu_0/SUCCESS.json','sha256':sb['sha256'],'expected':success}]})
        (output/'workers/gpu_0/results.jsonl').write_text(json.dumps({**request,'result':{}})+'\n')
        artifact={'algorithm':'sha256','files':{str(p.relative_to(output)):{'sha256':r.c.sha256_file(p),'size_bytes':p.stat().st_size}
                                              for p in output.rglob('*') if p.is_file()}}
        ab=write(output/'artifact_manifest.json',artifact)
        proof={'status':'verified','manifest_sha256':mb['sha256'],'run_token':token,'gpu_release_verified':True,
               'artifact_manifest_sha256':ab['sha256'],'artifact_files':len(artifact['files'])}
        pp=stage/'proof.json';pb=write(pp,proof)
        return {'manifest_path':str(manifest),'manifest_sha256':mb['sha256'],'verification_receipt':str(pp),
                'verification_receipt_sha256':pb['sha256'],'artifact_manifest_sha256':ab['sha256']},master,prepared,output

    def test_complete_phase_snapshot_success(self):
        reference,master,prepared,_=self.phase();plan,rows,proof=r.load_phase(reference,master,prepared,{})
        self.assertEqual(len(rows),1);self.assertEqual(proof['requests'],1);self.assertEqual(plan['phase'],'coarse')

    def test_unverified_wrong_master_and_changed_snapshot_fail(self):
        reference,master,prepared,output=self.phase()
        with self.assertRaises(ValueError):r.load_phase(reference,'c'*64,prepared,{})
        (output/'workers/gpu_0/results.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'snapshot hash'):r.load_phase(reference,master,prepared,{})

    def test_nonterminal_and_changed_request_plan_fail(self):
        reference,master,prepared,_=self.phase();stage=Path(reference['manifest_path']).parent
        terminal=stage/'control/supervisor_exit.json';v=json.loads(terminal.read_text());v['service_result']='timeout';write(terminal,v)
        with self.assertRaisesRegex(ValueError,'exit successfully'):r.load_phase(reference,master,prepared,{})
        v['service_result']='success';write(terminal,v);write(stage/'input/request_plan.json',{'mode':'generate'})
        with self.assertRaisesRegex(ValueError,'snapshot hash'):r.load_phase(reference,master,prepared,{})

    def test_token_reductions_and_nonfinite_rejected(self):
        rows=[{'record_id':'r','problem_split':'configuration_validation','completion_token_count':3,
               'region_mask_completion_positions':{'evaluator__transition':[1,2]}}]
        item={'record_id':'r','result':{'token_nll':[1.,2.,3.],'nll':{'all_completion':{'n_tokens':3,'mean_nll':2.},
              'evaluator__transition':{'n_tokens':2,'mean_nll':2.5}},'energy':{}}}
        r.validate_likelihoods([item],rows)
        changed=copy.deepcopy(item);changed['result']['nll']['evaluator__transition']['mean_nll']=2.6
        with self.assertRaisesRegex(ValueError,'exact original-token'):r.validate_likelihoods([changed],rows)
        item['result']['token_nll'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'per-token'):r.validate_likelihoods([item],rows)

    def test_incomplete_coarse_protocol_and_unreleased_catalog_fail(self):
        rows,master,catalog,artifact=fixture()
        plan=sweep.build_request_plan(rows,catalog,self.root/'candidates',artifact,master,'a'*64,previous_tf=6,previous_generations=6)
        r.validate_planned_conditions([plan],[[]],rows,master,'a'*64,catalog,artifact,self.root/'candidates')
        bad=copy.deepcopy(plan);bad['layers']=bad['layers'][:-1]
        with self.assertRaisesRegex(ValueError,'full frozen candidate protocol'):
            r.validate_planned_conditions([bad],[[]],rows,master,'a'*64,catalog,artifact,self.root/'candidates')
        cp=self.root/'candidate-proof.json';cb=write(cp,{'artifact_manifest_sha256':plan['candidate_artifact_manifest_sha256'],
                                                      'process_release_verified':False})
        spec={'candidate_verification':str(cp),'candidate_verification_sha256':cb['sha256'],'candidate_root':str(self.root/'candidates')}
        with patch.object(sweep,'load_verified_catalog',return_value=(catalog,artifact)):
            with self.assertRaisesRegex(ValueError,'not released'):r.candidate_inputs(spec,[plan])

    def test_full_cpu_ranking_and_neighbor_plan_remain_preparation_only(self):
        rows,master,catalog,artifact=fixture()
        for row in rows:row['region_mask_completion_positions']={'evaluator__transition':list(range(16))}
        prepared=self.root/'prepared.jsonl';prepared.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        master['inputs']={'prepared_records_sha256':r.c.sha256_file(prepared)}
        mp=self.root/'master/plan.json';mb=write(mp,master);pp=self.root/'parent.json';pb=write(pp,{})
        plan=sweep.build_request_plan(rows,catalog,self.root/'candidates',artifact,master,mb['sha256'],previous_tf=6,previous_generations=6)
        phase_rows=synthetic_results(plan,rows)
        for item in phase_rows:
            value=item['result']['nll']['evaluator__transition']['mean_nll'];item['result']['token_nll']=[value]*50
            item['result']['nll']={name:{'n_tokens':len(ps),'mean_nll':sum(item['result']['token_nll'][i] for i in ps)/len(ps)}
                for name,ps in {'all_completion':list(range(50)),'evaluator__transition':list(range(16))}.items()}
        fake_phase=self.root/'phase.json';fb=write(fake_phase,{'stage':str(self.root/'stage'),'output':str(self.root/'phase-output')})
        cp=self.root/'candidate-proof.json';cb=write(cp,{'process_release_verified':True,'artifact_manifest_sha256':plan['candidate_artifact_manifest_sha256']})
        spec={'master_path':str(mp),'master_sha256':mb['sha256'],'parent_master_path':str(pp),'parent_master_sha256':pb['sha256'],
              'prepared_records':str(prepared),'phases':[{'manifest_path':str(fake_phase),'manifest_sha256':fb['sha256']}],
              'phase_ledger_root':str(self.root),'previous_tf':6162,'previous_generations':6,'output':str(self.root/'ranked'),
              'prepare_neighbors':True,'candidate_root':str(self.root/'candidates'),'candidate_verification':str(cp),'candidate_verification_sha256':cb['sha256']}
        ledger={'tf_requests':6162,'generation_requests':6,'untouched_test_requests':0,'bindings':[fake_phase]}
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(r,'validate_master'),\
             patch.object(r,'load_phase',return_value=(plan,phase_rows,{'fixture':True})),\
             patch.object(r.phase_budget,'account',return_value=copy.deepcopy(ledger)),\
             patch.object(sweep,'load_verified_catalog',return_value=(catalog,artifact)):
            result=r.run(spec)
            verified=r.verify(self.root/'ranked')
            self.assertEqual(verified['requests'],6156)
        self.assertFalse(result['model_work_launched']);self.assertTrue(result['neighbor_plan_prepared'])
        neighbor=json.loads((self.root/'ranked/neighbor_request_plan.json').read_text())
        self.assertLessEqual(neighbor['tf_requests_after_commit'],12000)
        self.assertEqual(neighbor['previously_committed_tf_requests'],6162)
        self.assertEqual(neighbor['evaluation_partition'],'configuration_validation')


if __name__=='__main__':unittest.main()
