"""Authored7240 metadata and environment joins; no generated code or model runs."""
import copy
import json
import os
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import coarse_evaluation_context as c, h100_evaluate as m
from .test_h100_evaluate import Fixture as OldFixture

class Fixture(OldFixture):
    def __init__(self,root):
        super().__init__(root)
        prepared=[];dataset=[];coordinates=[];requests=[]
        conditions={'baseline':{'layers':[],'intervention_strength':.5}}
        conditions.update({f'condition-{i}':{'layers':[{'layer':i%36,'kind':'random','rank':1,'seed':100+i}],
                                         'intervention_strength':.5} for i in range(180)})
        for env in ('loophole','no_loophole'):
            for pid in range(20):
                prompt=[{'role':'system','content':'authored system'},{'role':'user','content':f'authored {env} {pid}'}]
                ids=[pid+1,3 if env=='loophole' else 4];rid=f'{env}-{pid}'
                prepared.append({'record_id':rid,'problem_id':pid,'problem_split':'configuration_validation',
                    'prompt':prompt,'prompt_sha256':c.digest(prompt),'prompt_token_ids':ids,'prompt_token_ids_sha256':c.ids_hash(ids),
                    'prompt_token_count':2,'completion_token_ids':[5],'completion_token_count':1,'input_ids':ids+[5],
                    'outcome_presence_class':'authored'})
                dataset.append({'environment':env,'source_line_sha256':'e'*64,'example':{'id':pid,'prompt':prompt,
                    'gt_answer':['authored trusted text, never executed'],'setup_code':'','func_name':'solve',
                    'canonical_solution':'authored source, never executed','question':'authored question',
                    'prompt_metadata':{'starter_code':'authored starter'}}})
                coord={'environment':env,'problem_id':pid,'record_id':rid,'prompt_sha256':c.digest(prompt),
                       'prompt_token_ids_sha256':c.ids_hash(ids),'sample_index':0}
                coordinates.append(coord)
        for cid in conditions:
            for coord in coordinates:
                requests.append({**coord,'request_id':cid+'-'+coord['record_id'],'condition_id':cid,
                    'problem_split':'configuration_validation','scope':'primary','seed':6007+coord['problem_id']})
        def lines(name,values):return m.ref(self.save(self.root/name,b''.join((m.canonical(v)+'\n').encode() for v in values)))
        self.prepared=prepared;self.dataset=dataset;self.requests=requests
        pr=lines('coarse-prepared.jsonl',prepared);dr=lines('coarse-dataset.jsonl',dataset);rr=lines('coarse-requests.jsonl',requests)
        science={'protocol':c.PROTOCOL,'expected_request_count':7240,'independent_problems':20,'samples_per_problem_environment':1,
            'condition_count':181,'target_conditions':72,'random_conditions':108,'test_used_for_selection':False,'prior_validation_use_declared':True,
            'selected_problem_ids':list(range(20)),'coordinates':coordinates,'conditions':conditions,
            'request_ids_sha256':c.digest(sorted(r['request_id'] for r in requests)),
            'requests':rr,'prepared_records':pr,'sampling':{'authored':'settings'},'intervention_strength':.5,
            'batch_profile':{'authored':'batch8'},'vector_path_resolution':{'package_root':str(self.root)}}
        sr=m.ref(self.save(self.root/'coarse-science.json',science));self.science=science
        self.stack.enter_context(patch.object(c,'SCIENCE_SHA',sr['sha256']));self.stack.enter_context(patch.object(c,'DATASET_SHA',dr['sha256']))
        source_proof=m.ref(self.save(self.root/'coarse-source-proof.json',{'authored':True}))
        ctx={'science_plan':sr,'requests':rr,'prepared_records':pr,'dataset':dr,
             'host_profile':self.gm['host_profile'],'source_qualification':source_proof}
        cr=m.ref(self.save(self.root/'coarse-context.json',ctx));self.ctx=ctx
        self.rows=[{**request,'original_class':'authored','result':{'completion':'authored text, never executed',
            'completion_token_ids':[151643],'generated_token_ids':[151643],'fixed_completion_prefix_token_count':0,'stop_reason':'eos'}} for request in requests]
        artifacts={};workers=[{'name':f'worker_{i:02d}'} for i in range(8)]
        for i,w in enumerate(workers):
            relative=w['name']+'/results.jsonl';p=self.save(self.root/'coarse-results'/relative,
                b''.join((m.canonical(r)+'\n').encode() for r in self.rows[i::8]))
            artifacts[relative]={'sha256':m.sha(p),'size_bytes':p.stat().st_size}
        ar=m.ref(self.save(self.root/'coarse-results/artifact_manifest.json',{'algorithm':'sha256','files':artifacts}))
        self.machine={'run_token':'codex-coarse-authored-run','source_root':str(self.source),'python':m.ref(self.python),
            'output':str(self.root/'coarse-results'),'mode':'generate_batch_v1','workers':workers,
            'bound_files':{r['path']:{k:v for k,v in r.items() if k!='path'} for r in (cr,*ctx.values())}}
        self.tasks=[{'prepared_records':pr['path'],'sampling':science['sampling'],'intervention_strength':.5,
                     'batch_profile':science['batch_profile'],'conditions':conditions,'requests':requests[i::8]} for i in range(8)]
        mr=m.ref(self.save(self.root/'coarse-machine.json',self.machine))
        self.coarse_proof={'status':'verified_batched_generation_bytes_and_coverage','plan_sha256':mr['sha256'],
            'generation':{'requests':7240,'request_ids_sha256':science['request_ids_sha256']},'artifact_manifest':ar}
        vr=m.ref(self.save(self.root/'coarse-proof.json',self.coarse_proof))
        self.outer={'status':'verified_systemd_controller_exit_and_release','plan_sha256':mr['sha256'],
            'run_token':self.machine['run_token'],'cgroup_processes':[],'gpu_release_verified':True,
            'service_fields':{'ExecMainCode':'1','ExecMainStatus':'0','Result':'success','MainPID':'0','SubState':'exited'}}
        er=m.ref(self.save(self.root/'coarse-outer.json',self.outer))
        self.package={'manifest':mr,'verification':vr,'external_exit':er,'coarse_context':cr}
        self.spec['generation']=self.package
        self.stack.enter_context(patch.object(c.generation,'load_plan',side_effect=lambda *a:(self.machine,self.tasks)))
        self.stack.enter_context(patch.object(c.generation,'verify',side_effect=lambda *a:self.coarse_proof))

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(Path(self.tmp.name).resolve());self.addCleanup(self.f.close)
    def test_full7240_context_build_and905_shards_preserve_environments(self):
        result=m.build(self.f.spec);manifest,plan,rows=m.load_manifest(result['manifest']['path'],result['manifest']['sha256'])
        self.assertEqual(manifest['purpose'],c.PURPOSE);self.assertEqual(len(rows),7240)
        self.assertEqual({r['environment'] for r in rows},{'loophole','no_loophole'})
        self.assertEqual(len(c.dataset_map(m.json_rows(m.ref(Path(manifest['stage'])/'input/dataset.jsonl')))),40)
        self.assertEqual([len(m.json_rows(m.ref(Path(manifest['stage'])/f'input/worker_{i:02d}.jsonl'))) for i in range(8)],[905]*8)
        self.assertIn('--property=RuntimeMaxSec=7290',result['command']);self.assertFalse(result['launch_performed'])
    def test_actual905_worker_preserves_environment_and_frozen_legacy_helper_labels(self):
        from .test_helper_aware_evaluation_v1 import row,primitive,load_frozen
        f=self.f;_,plan,_=c.context(f.package);values=f.rows[::8]
        requests=f.save(f.root/'worker_00.jsonl',b''.join((m.canonical(v)+'\n').encode() for v in values))
        plan_path=f.save(f.root/'copied-request-plan.json',plan);saved=row()['repository_evaluation'];examples=[]
        def evaluate(example,_text):examples.append(copy.deepcopy(example));return copy.deepcopy(saved)
        evaluator=SimpleNamespace(evaluate=evaluate,evaluator=SimpleNamespace(batch_evaluate=lambda _:[primitive(True),primitive(False)]))
        installation=SimpleNamespace(report=lambda:dict.fromkeys(m.repair.TRANSPORT,0),restore=lambda:None)
        oldfile,oldexists=Path.is_file,Path.exists
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','CODE_EVAL_SANDBOX':'bwrap'}), \
             patch.object(Path,'is_file',lambda path:str(path)=='/work/src/evaluate/helpers.py' or oldfile(path)), \
             patch.object(Path,'exists',lambda path:False if str(path) in ('/scratch','/home/ubuntu/h100-workspace') else oldexists(path)), \
             patch.object(f.evaluate,'make_repository_evaluator',return_value=evaluator), \
             patch.object(m.sandbox,'install_bounded_evaluator',return_value=installation), \
             patch.object(f.evaluate,'install_count_payload_guard'),patch.object(m.repair,'modules',return_value=load_frozen()):
            output=f.root/'coarse-worker-evaluation';result=m.inside(requests,plan['prepared_records'],plan['dataset'],output,plan_path)
            records=m.json_rows(m.ref(output/'records.jsonl'),immutable=False)
            m.validate_evaluations(records,values);m.replay_labels(records)
        self.assertEqual(result['records'],905);self.assertEqual({r['environment'] for r in records},{'loophole','no_loophole'})
        expected=c.dataset_map(f.dataset)
        self.assertEqual(examples,[expected[(r['environment'],str(r['problem_id']))] for r in values])
        self.assertTrue(all(r['metrics']['ground_truth_correctness']==r['helper_aware_evaluation']['metrics']['ground_truth_correctness'] for r in records))
    def test_failed_external_exit_stops_before_generation_verifier(self):
        self.f.outer['service_fields']['ExecMainStatus']='1'
        path=self.f.save(self.f.root/'failed-outer.json',self.f.outer);package={**self.f.package,'external_exit':m.ref(path)}
        with patch.object(c.generation,'verify',side_effect=AssertionError('result read')):
            with self.assertRaisesRegex(ValueError,'controller exit'):c.context(package)
    def test_unfrozen_science_stops_before_loading_machine(self):
        ctx=copy.deepcopy(self.f.ctx);ctx['science_plan']['sha256']='0'*64
        path=self.f.save(self.f.root/'wrong-context.json',ctx)
        with patch.object(c.generation,'load_plan',side_effect=AssertionError('machine read')):
            with self.assertRaisesRegex(ValueError,'Unfrozen'):c.context({**self.f.package,'coarse_context':m.ref(path)})
    def test_duplicate_or_missing_or_changed_pairing_rejected(self):
        _,plan,_=c.context(self.f.package)
        for requests in (plan['requests'][:-1],[plan['requests'][0],*plan['requests'][:-1]]):
            with self.assertRaises(ValueError):c.validate_inputs({**plan,'requests':requests},self.f.prepared,self.f.dataset)
        changed=copy.deepcopy(plan);changed['requests'][0]['seed']+=1
        with self.assertRaisesRegex(ValueError,'paired seeds'):c.validate_inputs(changed,self.f.prepared,self.f.dataset)
    def test_environment_prompt_and_trusted_tests_never_replaced(self):
        _,plan,_=c.context(self.f.package)
        changed=copy.deepcopy(self.f.dataset);changed[0]['example']['prompt']=changed[20]['example']['prompt']
        with self.assertRaisesRegex(ValueError,'prompt/token'):c.validate_inputs(plan,self.f.prepared,changed)
        changed=copy.deepcopy(self.f.dataset);changed[0]['example']['gt_answer']=['different']
        with self.assertRaisesRegex(ValueError,'Trusted tests'):c.validate_inputs(plan,self.f.prepared,changed)
    def test_json_token_hash_rejected_even_when_all_recorded_fields_agree(self):
        import hashlib,struct
        _,plan,_=c.context(self.f.package)
        self.assertEqual(c.ids_hash([1,256,151643]),
                         hashlib.sha256(struct.pack('<III',1,256,151643)).hexdigest())
        changed=copy.deepcopy(plan);prepared=copy.deepcopy(self.f.prepared)
        row=prepared[0];wrong=c.digest(row['prompt_token_ids'])
        self.assertNotEqual(wrong,row['prompt_token_ids_sha256']);row['prompt_token_ids_sha256']=wrong
        for coordinate in changed['coarse_layer_map']['science']['coordinates']:
            if coordinate['record_id']==row['record_id']:coordinate['prompt_token_ids_sha256']=wrong
        for request in changed['requests']:
            if request['record_id']==row['record_id']:request['prompt_token_ids_sha256']=wrong
        with self.assertRaisesRegex(ValueError,'prompt/token'):c.validate_inputs(changed,prepared,self.f.dataset)
    def test_environment_and_prompt_hashes_are_output_identity(self):
        generations=self.f.rows[:1];row=self.f.evaluation_rows()[0]
        row.update({k:generations[0][k] for k in c.EXTRA_IDENTITY});m.validate_evaluations([row],generations)
        for field in c.EXTRA_IDENTITY:
            changed=copy.deepcopy(row);changed[field]='wrong'
            with self.assertRaisesRegex(ValueError,'identity'):m.validate_evaluations([changed],generations)
    def test_large_coarse_merged_read_keeps_old740_limit_and_total_cap(self):
        path=self.f.root/'large-authored-merged.jsonl'
        with path.open('xb') as stream:
            for _ in range(128):stream.write(b' '*(1<<20))
            stream.write(b'{"authored":true}\n')
        path.chmod(0o400);reference=m.ref(path)
        with self.assertRaises(ValueError):m.json_rows(reference)
        self.assertEqual(m.json_rows(reference,coarse_merged=True),[{'authored':True}])
        oversized=self.f.root/'oversized-authored.jsonl'
        with oversized.open('xb') as stream:stream.truncate(m.LIMITS['maximum_output_bytes']+1)
        oversized.chmod(0o400)
        with self.assertRaises(ValueError):m.json_rows({'path':str(oversized),'sha256':'0'*64},coarse_merged=True)

if __name__=='__main__':unittest.main()
