import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import qualification_gate as gate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def row(index):
    completion=[3,4]+[20+index%3]*18
    ids=[1,2,*completion]
    return {'record_id':f'r{index}','record_index':index,'problem_id_key':'1','problem_split':'direction_fit',
            'outcome_presence_class':sorted(gate.fixed_cache.CORE_CLASSES)[index%3],
            'prompt_token_count':2,'completion_token_count':20,'evaluator_body_token':2,
            'prompt_token_ids':[1,2],'completion_token_ids':completion,'input_ids':ids,
            'input_ids_sha256':gate.fixed_cache.ids_hash(ids),'selected_token_positions':list(range(2,22)),
            'selected_token_mask':[True]*20,'region_mask_completion_positions':{}}


class QualificationGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'proposed';self.old=self.root/'qualified'
        self.output=self.root/'output';self.worker=self.output/'workers/gpu_0'
        self.worker.mkdir(parents=True)
        self.prepared=self.root/'prepared.jsonl'
        self.rows=[row(i) for i in range(561)]
        self.prepared.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.prepared_hash=gate.fixed_cache.sha256(self.prepared)
        self.manifest=self.root/'manifest.json';self.task_path=self.root/'task.json'
        self.task={'run_token':'token','worker_name':'gpu_0','mode':'fixed_cache','cache_role':'core',
                   'model_snapshot':'model','checkpoint':'checkpoint','output':str(self.worker),
                   'manifest_path':str(self.manifest),'prepared_records':str(self.prepared),
                   'conditions':{'baseline':{'layers':[]}},'padded_sequence_length':2176,'pad_token_id':151643,
                   'attention_policy':'exclusive_math','requests':[{'request_id':f'fixed-cache-qual-r{i}',
                    'record_id':f'r{i}','condition_id':'baseline'} for i in gate.INDICES]}
        self.m={'phase':'fixed_cache_qualification','run_token':'token','output':str(self.output),'source_root':str(self.old),
                'scientific':{'input_prepared_sha256':self.prepared_hash,'cache_role':'core','phase_modes':['fixed_cache']},
                'workers':[{'name':'gpu_0','command':['python','engine','--task',str(self.task_path)],
                            'success_expect':{'mode':'fixed_cache','requests':3}}],
                'runtime_versions':{'fixture':'1'},'bound_files':{}}
        for relative in gate.CRITICAL_SOURCE:
            for root in (self.source,self.old):
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authored source')
            path=self.old/relative
            self.m['bound_files'][str(path)]={'sha256':gate.fixed_cache.sha256(path),'size_bytes':path.stat().st_size}
        exact={'bitwise_equal':True,'relative_l2':0,'max_abs_difference':0}
        self.report={'status':'succeeded','records':3,'problems':1,'identical_prefix_comparisons':6,
                     **{k:True for k in ('all_identical_prefixes_bitwise_equal','all_native_readbacks_bitwise_equal',
                           'fp32_delta_readback_exact','raw_activations_retained','differences_computed')},
                     'activation_cache_profile':gate.PROFILE,'numerical_runtime_policy':gate.POLICY,
                     'cross_package_prefix_audit_required':False,
                     'prefix_comparison_scope':'within_worker_same_problem_and_identical_prompt',
                     'model_load_reports':{'h0':{'with_adapter':False},'h60':{'with_adapter':True}},
                     'cuda_release':{'h0':{'allocated_bytes':0},'h60':{'allocated_bytes':0}},'qualifications':{}}
        self.artifacts={'files':{}}
        for kind in ('h0','h60'):
            directory=self.worker/'qualification'/kind
            future={**exact,'unchanged_completion_prefix_tokens':16,'first_changed_sequence_position':18,
                    'changed_valid_future_tokens':4,'evaluator_transition_prefix_included':True,
                    'fixed_padded_length':2176,'padding_mask_zero_unchanged':True,'attention_flags':gate.FLAGS}
            for name in ('repeat','future_perturbed'):
                path=directory/(name+'.safetensors')
                item={'path':str(path),'native_readback_bitwise_equal':True,'sha256':'d'*64,'size_bytes':10}
                self.artifacts['files'][str(path.relative_to(self.output))]={'sha256':'d'*64,'size_bytes':10}
                if name=='repeat':
                    write(directory/'repeat_audit.json',{**exact,'attention_flags':gate.FLAGS,'artifact':item})
                else:
                    future['artifact']=item
            self.report['qualifications'][kind]={'repeat':exact.copy(),'future_causality':future}
            write(directory/'future_causality_audit.json',future)
        self.pairs=[]
        for kind in ('h0','h60'):
            for a,b in ((222,223),(222,224),(223,224)):
                self.pairs.append({'kind':kind,'record_ids':[f'r{a}',f'r{b}'],'problem_id':'1',
                                   'common_completion_tokens':2,'prompt_final':exact,'completion_prefix':exact})
        self.reference=self.root/'reference.json'
        self.save()
        self.verify=mock.Mock(return_value={'status':'verified'})
        supervisor=types.SimpleNamespace(verify=self.verify,sha256=gate.fixed_cache.sha256)
        legacy=types.SimpleNamespace(validate_model_load_reports=mock.Mock(),validate_post_model_cuda_state=mock.Mock())
        self.patches=[mock.patch.dict('sys.modules',{'infra.gpu03.direction_discovery.supervisor':supervisor}),
                      mock.patch.object(gate.importlib.metadata,'version',return_value='1'),
                      mock.patch.object(gate.fixed_cache,'dependencies',return_value=(None,legacy))]
        import infra.gpu03.direction_discovery as namespace
        self.patches.append(mock.patch.object(namespace,'supervisor',supervisor,create=True))
        for patch in self.patches:patch.start();self.addCleanup(patch.stop)

    def save(self):
        write(self.manifest,self.m);write(self.task_path,self.task);write(self.worker/'task.json',self.task)
        write(self.worker/'fixed_cache_report.json',self.report)
        write(self.output/'artifact_manifest.json',self.artifacts)
        (self.worker/'prefix_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.pairs))
        write(self.reference,{'manifest_path':str(self.manifest),'manifest_sha256':gate.fixed_cache.sha256(self.manifest),
            'artifact_manifest_sha256':gate.fixed_cache.sha256(self.output/'artifact_manifest.json')})

    def run_gate(self,**kwargs):
        return gate.verify_fixed_qualification(self.reference,self.prepared_hash,source_root=self.source,
                    model_snapshot=kwargs.get('model_snapshot','model'),checkpoint=kwargs.get('checkpoint','checkpoint'))

    def test_complete_exact_fitting_triplet_proof_and_single_payload_verification(self):
        proof=self.run_gate()
        self.assertIn(self.worker/'prefix_audit.jsonl',proof)
        self.assertIn(self.worker/'qualification/h60/future_causality_audit.json',proof)
        self.verify.assert_called_once()

    def test_wrong_checkpoint_or_capture_source_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'model/task identity'):self.run_gate(checkpoint='other')
        (self.source/gate.CRITICAL_SOURCE[1]).write_text('changed source')
        with self.assertRaisesRegex(RuntimeError,'source changed'):self.run_gate()

    def test_wrong_record_triplet_cannot_reuse_same_three_record_count(self):
        self.task['requests'][0]['record_id']='r225';self.save()
        with self.assertRaisesRegex(RuntimeError,'record IDs/indices'):self.run_gate()

    def test_heldout_triplet_is_not_qualified_fitting_data(self):
        for i in gate.INDICES:self.rows[i]['problem_split']='untouched_test'
        self.prepared.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.prepared_hash=gate.fixed_cache.sha256(self.prepared)
        self.m['scientific']['input_prepared_sha256']=self.prepared_hash;self.save()
        with self.assertRaisesRegex(RuntimeError,'fitting problem'):self.run_gate()

    def test_duplicate_prefix_pair_and_missing_future_fail(self):
        self.pairs[-1]=copy.deepcopy(self.pairs[0]);self.save()
        with self.assertRaisesRegex(RuntimeError,'prefix pair'):self.run_gate()
        self.pairs.pop();self.report['qualifications']['h0']['future_causality']['bitwise_equal']=False;self.save()
        with self.assertRaisesRegex(RuntimeError,'future causality'):self.run_gate()

    def test_transition_coverage_and_probe_manifest_link_are_required(self):
        self.report['qualifications']['h0']['future_causality']['evaluator_transition_prefix_included']=False;self.save()
        with self.assertRaisesRegex(RuntimeError,'transition coverage'):self.run_gate()
        self.report['qualifications']['h0']['future_causality']['evaluator_transition_prefix_included']=True
        key=next(iter(self.artifacts['files']));self.artifacts['files'][key]['sha256']='bad';self.save()
        with self.assertRaisesRegex(RuntimeError,'probe hash'):self.run_gate()

    def test_runtime_and_full_profile_must_match(self):
        self.report['activation_cache_profile']={**gate.PROFILE,'pad_token_id':999};self.save()
        with self.assertRaisesRegex(RuntimeError,'padding profile'):self.run_gate()
        self.report['activation_cache_profile']=gate.PROFILE
        self.report['numerical_runtime_policy']={**gate.POLICY,'matmul_allow_tf32':True};self.save()
        with self.assertRaisesRegex(RuntimeError,'runtime policy'):self.run_gate()

    def test_nonverified_supervisor_status_is_not_success(self):
        self.verify.return_value={'status':'running'}
        with self.assertRaisesRegex(RuntimeError,'verified one-worker'):self.run_gate()


if __name__=='__main__':unittest.main()
