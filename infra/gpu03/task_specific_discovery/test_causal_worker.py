"""Offline worker contracts; authored inputs and inert generator/loader only.

The Torch-only vector tests use CPU tensors. The integration fixtures explicitly
replace model loading and generation; they never execute generated code or CUDA.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from infra.gpu03.task_specific_discovery import causal_worker as cw
from infra.gpu03.task_specific_discovery.io_contract import RequestJournal, AmbiguousRequest, hash_value


GPU = 'GPU-12345678-1234-1234-1234-123456789abc'


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/'source'
        self.source.mkdir()
        sources = {}
        for name in cw.REQUIRED_SOURCE_FILES:
            path=self.source/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# authored fixture, never imported: '+name+'\n')
            sources[name]=self.binding(path)
        self.model=self.root/'model';self.model.mkdir()
        self.adapter=self.root/'adapter';self.adapter.mkdir()
        bindings={}
        for name in cw.BASE_FILES:
            path=self.model/name;path.write_text('inert model input '+name)
            bindings[str(path)]=self.binding(path)
        adapter_hashes={}
        for name in cw.ADAPTERS:
            path=self.adapter/name;path.write_text('inert adapter input '+name)
            bindings[str(path)]=self.binding(path);adapter_hashes[name]=cw.sha256(path)
        self.rows=[{'record_id':'prompt-1','problem_id':1,'problem_split':'direction_fit',
                    'protocol':'fixed','prompt':[{'role':'user','content':'authored input'}],
                    'prompt_token_ids':[1,2],'prompt_token_count':2,
                    'completion_token_ids':[],'completion_token_count':0,'input_ids':[1,2]}]
        rows_path=self.root/'prepared.jsonl';self.rows_path=rows_path
        rows_path.write_text(''.join(json.dumps(x)+'\n' for x in self.rows))
        vectors=self.root/'vectors.safetensors';vectors.write_text('inert bound bytes')
        bindings[str(rows_path)]=self.binding(rows_path);bindings[str(vectors)]=self.binding(vectors)
        self.task={'schema':'task_specific_causal_generation_v1','model_revision':cw.REVISION,
                   'stage':'qualification','sampling':copy.deepcopy(cw.SAMPLING),'deadline_seconds':120,
                   'batch_size':8,'profile':copy.deepcopy(cw.PROFILE),'model_snapshot':str(self.model),
                   'checkpoint':str(self.adapter),'prepared_rows':str(rows_path),'vectors':str(vectors),
                   'output':str(self.root/'output'),'run_token':'authored-offline-worker',
                   'qualified_source_files':sources,'bound_files':bindings,
                   'conditions':{'baseline':{'layer':None,'operator':'ordinary','control':'zero_strength','strength':0}},
                   'requests':[{'request_id':'request-1','record_id':'prompt-1','problem_id':1,
                                'protocol':'fixed','scope':'primary','seed':29,'sample_index':0,'condition_id':'baseline'}]}
        for patch in (mock.patch.object(cw,'SOURCE_ROOT',self.source),mock.patch.object(cw,'ADAPTERS',adapter_hashes),
                      mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':GPU})):
            patch.start();self.addCleanup(patch.stop)

    @staticmethod
    def binding(path):
        return {'sha256':cw.sha256(path),'size_bytes':path.stat().st_size}

    def update_rows(self):
        self.rows_path.write_text(''.join(json.dumps(x)+'\n' for x in self.rows))
        self.task['bound_files'][str(self.rows_path)]=self.binding(self.rows_path)

    def pilot(self):
        self.task['stage']='pilot'
        self.proof={'status':'qualified_task_specific_causal_v1','profile':copy.deepcopy(cw.PROFILE),
                    'source_files':copy.deepcopy(self.task['qualified_source_files']),
                    'vectors_sha256':self.task['bound_files'][self.task['vectors']]['sha256'],
                    'model_files':cw.model_bindings(self.task,cw.BASE_FILES,'model_snapshot'),
                    'adapter_files':cw.model_bindings(self.task,cw.ADAPTERS,'checkpoint'),
                    'qualified_signatures':[{'layer':21,'operator':'ordinary','rank':1}],
                    'all_returned_requests_retained':True,'gpu_workers_released':True,
                    'null_token_parity':True,'null_logits_within_tolerance':True}
        self.proof_path=self.root/'qualification.json';self.task['qualification']=str(self.proof_path)
        self.update_proof()

    def update_proof(self):
        self.proof_path.write_text(json.dumps(self.proof))
        self.task['bound_files'][str(self.proof_path)]=self.binding(self.proof_path)

    def project_condition(self, **updates):
        return {'layer':21,'operator':'ordinary','control':'project','strength':.5,'vector_key':'q',**updates}

    def journal(self, requests=None):
        requests=self.task['requests'] if requests is None else requests
        return RequestJournal(self.root/'journal',run_token=self.task['run_token'],input_sha256='1'*64,
                              request_hashes={r['request_id']:hash_value(r) for r in requests})


class ValidationTests(Fixture):
    def test_qualification_and_pilot_positive_actual_source_bindings(self):
        self.assertEqual(cw.validate(self.task),{'prompt-1':self.rows[0]})
        self.pilot()
        self.assertEqual(cw.validate(self.task),{'prompt-1':self.rows[0]})

    def test_single_complete_gpu_uuid_required(self):
        for value in ('GPU-0',GPU+','+GPU,'0','',GPU+'\n'):
            with self.subTest(value=value),mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':value}):
                with self.assertRaisesRegex(ValueError,'UUID-bound'):cw.validate(self.task)

    def test_profile_deadline_batch_and_fresh_output(self):
        for changes in ({'deadline_seconds':True},{'deadline_seconds':14401},{'batch_size':4},
                        {'profile':{**cw.PROFILE,'attention_policy':'flash'}},{'model_revision':'different'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):cw.validate({**self.task,**changes})
        Path(self.task['output']).mkdir()
        with self.assertRaisesRegex(ValueError,'Fresh absolute output'):cw.validate(self.task)

    def test_source_must_be_present_current_and_relative(self):
        name=cw.REQUIRED_SOURCE_FILES[0]
        missing=copy.deepcopy(self.task);missing['qualified_source_files'].pop(name)
        with self.assertRaisesRegex(ValueError,'source omitted'):cw.validate(missing)
        altered=copy.deepcopy(self.task);altered['qualified_source_files']['../escape.py']={}
        with self.assertRaisesRegex(ValueError,'relative path'):cw.validate(altered)
        (self.source/name).write_text('changed source')
        with self.assertRaisesRegex(ValueError,'Current consumed source differs'):cw.validate(self.task)

    def test_redeclaring_changed_source_cannot_reuse_old_qualification(self):
        self.pilot();name=cw.REQUIRED_SOURCE_FILES[2];path=self.source/name
        path.write_text('new hook arithmetic')
        self.task['qualified_source_files'][name]=self.binding(path)
        with self.assertRaisesRegex(ValueError,'exact-profile qualification'):cw.validate(self.task)

    def test_model_consumed_file_binding_and_adapter_identity(self):
        missing=copy.deepcopy(self.task);missing['bound_files'].pop(str(self.model/cw.BASE_FILES[0]))
        with self.assertRaisesRegex(ValueError,'Consumed input omitted'):cw.validate(missing)
        altered=copy.deepcopy(self.task)
        altered['bound_files'][str(self.adapter/'adapter_model.safetensors')]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Original M60 differs'):cw.validate(altered)
        (self.model/'config.json').write_text('altered scientific input')
        with self.assertRaisesRegex(ValueError,'Changed input'):cw.validate(self.task)

    def test_qualification_flags_model_and_vectors_fail_closed(self):
        self.pilot();original=copy.deepcopy(self.proof)
        for key in ('all_returned_requests_retained','gpu_workers_released','null_token_parity','null_logits_within_tolerance'):
            with self.subTest(key=key):
                self.proof={**original,key:False};self.update_proof()
                with self.assertRaisesRegex(ValueError,'qualification required'):cw.validate(self.task)
        self.proof=copy.deepcopy(original);self.proof['model_files']['config.json']['sha256']='0'*64;self.update_proof()
        with self.assertRaisesRegex(ValueError,'model/checkpoint identities'):cw.validate(self.task)
        self.proof={**original,'vectors_sha256':'0'*64};self.update_proof()
        with self.assertRaisesRegex(ValueError,'qualification required'):cw.validate(self.task)

    def test_qualification_rejects_active_projection_and_false_baseline(self):
        self.task['conditions']['target']=self.project_condition()
        with self.assertRaisesRegex(ValueError,'only baseline/null'):cw.validate(self.task)
        self.task['conditions']={'baseline':self.project_condition(layer=None)}
        with self.assertRaisesRegex(ValueError,'unprojected baseline'):cw.validate(self.task)

    def test_conditions_strength_center_rank_and_signatures(self):
        self.pilot()
        for updates in ({'strength':True},{'strength':.3},{'layer':True},{'rank':True},{'rank':2},
                        {'layer':36},{'operator':'centered'},{'center_key':'unused'},{'unexpected':1}):
            with self.subTest(updates=updates):
                self.task['conditions']['target']=self.project_condition(**updates)
                with self.assertRaises(ValueError):cw.validate(self.task)
        self.task['conditions']['target']=self.project_condition(layer=20)
        with self.assertRaisesRegex(ValueError,'Unqualified layer'):cw.validate(self.task)
        self.task['conditions']['target']=self.project_condition(vector_key='random_axis')
        cw.validate(self.task)  # Null semantics use same layer/rank; entire vector file stays bound.
        self.assertEqual(cw.qualification_signature(self.project_condition()),
                         {'layer':21,'operator':'ordinary','rank':1})

    def test_holdout_duplicate_coordinates_and_paired_seed_rejected(self):
        self.rows[0]['problem_split']='untouched_test';self.update_rows()
        with self.assertRaisesRegex(ValueError,'population'):cw.validate(self.task)
        self.rows[0]['problem_split']='direction_fit';self.update_rows()
        second={**self.task['requests'][0],'request_id':'request-2'};self.task['requests'].append(second)
        with self.assertRaisesRegex(ValueError,'semantic request'):cw.validate(self.task)
        self.task['conditions']['null']={'layer':21,'operator':'ordinary','control':'zero_strength','strength':0,'vector_key':'q'}
        second.update(condition_id='null',seed=30)
        with self.assertRaisesRegex(ValueError,'Matched request seeds'):cw.validate(self.task)

    def test_unbound_proof_rejected(self):
        self.pilot();self.task['bound_files'].pop(self.task['qualification'])
        with self.assertRaisesRegex(ValueError,'Consumed input omitted'):cw.validate(self.task)


class CallbackTests(Fixture):
    def test_return_immediately_becomes_durable_primitive_without_metadata(self):
        result={'completion_token_ids':[11,151645],'completion':'inert authored output','batch_wall_seconds':None}
        with self.journal() as journal:
            journal.start('request-1')
            returned,_=cw.batch_callbacks(journal,self.task['requests'],self.root,0)
            returned(0,result)
            self.assertEqual(journal.primitive('request-1'),{'result':result})
            self.assertEqual(journal.pending()['primitive_ready'],['request-1'])
        with self.journal() as journal:self.assertEqual(journal.primitive('request-1'),{'result':result})

    def test_completed_partial_not_promoted_or_replayed(self):
        requests=self.task['requests']+[dict(self.task['requests'][0],request_id='request-2')]
        partial_rows=[{'generated_token_ids':[11],'completed':True},{'generated_token_ids':[12],'completed':True}]
        with self.journal(requests) as journal:
            for request in requests:journal.start(request['request_id'])
            returned,partial=cw.batch_callbacks(journal,requests,self.root,4)
            returned(0,{'completion_token_ids':[11],'completion':'authored first'})
            partial(partial_rows)
            self.assertEqual(journal.pending()['started_without_primitive'],['request-2'])
            self.assertEqual(journal.pending()['primitive_ready'],['request-1'])
        saved=json.loads((self.root/'batch_0004.partial.json').read_text())
        self.assertEqual(saved,{'batch_index':4,'request_ids':['request-1','request-2'],'rows':partial_rows})
        with self.assertRaises(AmbiguousRequest):
            with self.journal(requests):pass

    def test_partial_is_immutable_and_durable_even_when_primitive_write_fails(self):
        with self.journal() as journal:
            journal.start('request-1');returned,partial=cw.batch_callbacks(journal,self.task['requests'],self.root,0)
            with mock.patch.object(journal,'record_primitive',side_effect=OSError('authored disk failure')):
                with self.assertRaises(OSError):returned(0,{'completion_token_ids':[11]})
            partial([{'generated_token_ids':[11],'completed':True}])
            with self.assertRaises(FileExistsError):partial([])
        with self.assertRaises(AmbiguousRequest):
            with self.journal():pass

    def test_export_keeps_raw_identity_seed_and_exact_prompt(self):
        result={'completion_token_ids':[11],'completion':'authored','energy':{},'batch_wall_seconds':None}
        primitive={'result':result}
        row=cw.export_row(self.task['requests'][0],self.rows[0],primitive,'3'*64)
        self.assertEqual(row['generation_seed'],29)
        self.assertEqual(row['engine_prompt_token_ids'],[1,2])
        self.assertEqual(row['prompt'],self.rows[0]['prompt'])
        self.assertEqual(row['completion_sha256'],hashlib.sha256(b'authored').hexdigest())
        self.assertEqual(row['raw_primitive_sha256'],hash_value(primitive))
        self.assertEqual(row['result'],result)


class InertRunTests(Fixture):
    def inert_runtime(self, *, change_final=False, loader_error=None, generation_error=False):
        self.pilot()
        cuda=SimpleNamespace(enable_cudnn_sdp=mock.Mock(),math_sdp_enabled=lambda:True,
                             flash_sdp_enabled=lambda:False,mem_efficient_sdp_enabled=lambda:False,
                             cudnn_sdp_enabled=lambda:False)
        torch=ModuleType('torch');torch.backends=SimpleNamespace(cuda=cuda)
        transformers=ModuleType('transformers')
        transformers.AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=SimpleNamespace(
            apply_chat_template=lambda *a,**k:[1,2])))
        st=ModuleType('safetensors.torch');st.load_file=mock.Mock(return_value={});st.save_file=mock.Mock()
        safetensors=ModuleType('safetensors');safetensors.torch=st
        legacy=SimpleNamespace(_load_decoder=mock.Mock(side_effect=loader_error,return_value=(object(),object(),[],{'inert':True})),
                               _release_cuda=mock.Mock(return_value={'inert_release':True}),
                               validate_post_model_cuda_state=mock.Mock())
        fixed=ModuleType('infra.gpu03.direction_discovery.fixed_cache')
        fixed.dependencies=lambda:(SimpleNamespace(configure_torch=mock.Mock()),legacy)
        generation=ModuleType('infra.gpu03.task_specific_discovery.causal_generation')
        generation.PROFILE=cw.PROFILE;generation.validate_primary_row=mock.Mock()
        generation.request_batches=lambda requests,size:[requests]
        def generate(*args,**kwargs):
            result={'completion_token_ids':[11,151645],'completion':'inert completion',
                    'generated_token_ids':[11,151645],'energy':{},'batch_wall_seconds':None}
            kwargs['on_completed'](0,result)
            if generation_error:
                kwargs['on_partial']([{'generated_token_ids':[11,151645],'completed':True}])
                raise RuntimeError('authored post-callback error')
            return [{**result,'completion':'changed' if change_final else result['completion'],'batch_wall_seconds':2.}]
        generation.generate_batch=mock.Mock(side_effect=generate)
        modules={'torch':torch,'transformers':transformers,'safetensors':safetensors,'safetensors.torch':st,
                 fixed.__name__:fixed,generation.__name__:generation}
        patch=mock.patch.dict(sys.modules,modules);patch.start();self.addCleanup(patch.stop)
        # Parent packages may already cache imported submodule attributes.
        import infra.gpu03.direction_discovery as dd
        import infra.gpu03.task_specific_discovery as td
        for pkg,name,module in ((dd,'fixed_cache',fixed),(td,'causal_generation',generation)):
            patch=mock.patch.object(pkg,name,module,create=True);patch.start();self.addCleanup(patch.stop)
        task=self.root/'task.json';task.write_text(json.dumps(self.task))
        return task,cw.sha256(task),legacy,generation

    def test_success_reuses_primitive_keeps_wall_metadata_separate_and_releases(self):
        task,digest,legacy,generation=self.inert_runtime()
        report=cw.run(task,digest)
        self.assertEqual(report['status'],'succeeded')
        self.assertFalse(report['evaluation_performed'])
        self.assertFalse(report['causal_success_claimed'])
        legacy._release_cuda.assert_called_once()
        generation.generate_batch.assert_called_once()
        rows=[json.loads(x) for x in (Path(self.task['output'])/'generations.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertIsNone(rows[0]['result']['batch_wall_seconds'])
        finals=list((Path(self.task['output'])/'journal'/'finals').glob('*.json'))
        self.assertEqual(len(finals),1)
        self.assertIn('"batch_wall_seconds":2.0',finals[0].read_text())
        self.assertTrue((Path(self.task['output'])/'COMPLETE.json').is_file())

    def test_changed_final_text_fails_after_primitive_preserved(self):
        task,digest,legacy,_=self.inert_runtime(change_final=True)
        with self.assertRaisesRegex(ValueError,'Returned row differs'):cw.run(task,digest)
        legacy._release_cuda.assert_called_once()
        output=Path(self.task['output'])
        self.assertEqual(len(list((output/'journal'/'primitives').glob('*.json'))),1)
        self.assertFalse((output/'COMPLETE.json').exists())
        self.assertFalse((output/'generations.jsonl').exists())

    def test_post_callback_failure_retains_raw_and_partial_then_stops(self):
        task,digest,legacy,_=self.inert_runtime(generation_error=True)
        with self.assertRaisesRegex(RuntimeError,'post-callback'):cw.run(task,digest)
        legacy._release_cuda.assert_called_once()
        output=Path(self.task['output'])
        self.assertEqual(len(list((output/'journal'/'primitives').glob('*.json'))),1)
        self.assertTrue((output/'batch_0000.partial.json').is_file())
        self.assertFalse((output/'COMPLETE.json').exists())

    def test_model_load_failure_releases_and_cannot_claim_completion(self):
        task,digest,legacy,generation=self.inert_runtime(loader_error=RuntimeError('authored load failure'))
        with self.assertRaisesRegex(RuntimeError,'load failure'):cw.run(task,digest)
        legacy._release_cuda.assert_called_once();generation.generate_batch.assert_not_called()
        self.assertFalse((Path(self.task['output'])/'COMPLETE.json').exists())


class VectorTests(unittest.TestCase):
    def test_all_random_target_vectors_and_centers_are_cpu_validated(self):
        import torch
        q=torch.zeros((2560,1),dtype=torch.float32,device='cpu');q[0]=1
        task={'conditions':{'target':{'layer':21,'operator':'centered','vector_key':'q','center_key':'mu'}}}
        values={'q':q,'mu':torch.zeros(2560,dtype=torch.float32,device='cpu')}
        cw.validate_vectors(task,values,torch)
        for bad in (q*2,q.to(torch.bfloat16),q[:,0],torch.full_like(q,float('nan'))):
            with self.subTest(shape=tuple(bad.shape)),self.assertRaises(ValueError):
                cw.validate_vectors(task,{**values,'q':bad},torch)
        for bad in (torch.zeros((2560,1)),torch.zeros(2560,dtype=torch.bfloat16),torch.full((2560,),float('inf'))):
            with self.subTest(shape=tuple(bad.shape)),self.assertRaises(ValueError):
                cw.validate_vectors(task,{**values,'mu':bad},torch)


if __name__=='__main__':unittest.main()
