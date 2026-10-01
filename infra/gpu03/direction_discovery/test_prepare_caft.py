import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

from infra.gpu03.direction_discovery import prepare_caft as c

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / 'artifacts/reward_hacking/dataset_freeze/checkpoint_60_outcome_presence_frozen_20260907_090259_v2'
MASTER = ROOT / 'artifacts/direction_discovery_review_20260907/plan_v1/experiment_plan.json'


def matrix_file(path, columns, dtype='F32'):
    rank=len(columns); values=[columns[j][i] for i in range(2560) for j in range(rank)]
    data=struct.pack('<'+'f'*len(values),*values)
    header=json.dumps({'Q':{'dtype':dtype,'shape':[2560,rank],'data_offsets':[0,len(data)]}}).encode()
    path.write_bytes(struct.pack('<Q',len(header))+header+data)
    return {'path':str(path),'sha256':c.sha256(path),'key':'Q'}


def unit(index):
    return [float(i==index) for i in range(2560)]


class TensorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.q=matrix_file(self.root/'q.safetensors',[unit(0)])
        self.source=matrix_file(self.root/'source.safetensors',[unit(0)])
        self.layer={'layer':12,'rank':1,'q':self.q}
        self.candidate={'layer':12,'kind':'candidate','path':self.source['path'],'sha256':self.source['sha256'],
                        'selectors':[{'key':'Q','column':0}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_fixed_q_and_candidate_span(self):
        result=c.verify_q(self.layer,self.candidate)
        self.assertEqual(result['gram_max_abs_error'],0)
        self.assertEqual(result['maximum_relative_candidate_span_leakage'],0)
        self.assertEqual(result['shape'],[2560,1])

    def test_q_basis_rotation_preserves_selected_span_without_averaging_pcs(self):
        a,b=unit(0),unit(1);s=2**-.5
        self.layer.update(rank=2,q=matrix_file(self.root/'q2.safetensors',[[s*(x+y) for x,y in zip(a,b)],[s*(x-y) for x,y in zip(a,b)]]))
        src=matrix_file(self.root/'src2.safetensors',[a,b])
        self.candidate.update(path=src['path'],sha256=src['sha256'],selectors=[{'key':'Q','column':0},{'key':'Q','column':1}])
        self.assertLess(c.verify_q(self.layer,self.candidate)['maximum_relative_candidate_span_leakage'],1e-6)

    def test_unrelated_q_span_fails(self):
        self.layer['q']=matrix_file(self.root/'other.safetensors',[unit(1)])
        with self.assertRaisesRegex(ValueError,'span differs'):
            c.verify_q(self.layer,self.candidate)

    def test_nonorthogonal_nonfinite_wrong_dtype_and_layer_fail(self):
        for columns,dtype,message in [([[2*x for x in unit(0)]],'F32','not orthonormal'),
                                      ([[float('nan')]+[0.0]*2559],'F32','Nonfinite'),
                                      ([unit(0)],'BF16','must be FP32')]:
            with self.subTest(message=message):
                self.layer['q']=matrix_file(self.root/'bad.safetensors',columns,dtype)
                with self.assertRaisesRegex(ValueError,message):c.verify_q(self.layer,self.candidate)
        self.layer.update(layer=13,q=self.q)
        with self.assertRaisesRegex(ValueError,'layer/rank'):c.verify_q(self.layer,self.candidate)

    def test_dependent_candidate_columns_cannot_add_untested_q_dimension(self):
        self.layer.update(rank=2,q=matrix_file(self.root/'q2.safetensors',[unit(0),unit(1)]))
        src=matrix_file(self.root/'duplicate.safetensors',[unit(0),unit(0)])
        self.candidate.update(path=src['path'],sha256=src['sha256'],selectors=[{'key':'Q','column':0},{'key':'Q','column':1}])
        with self.assertRaisesRegex(ValueError,'Dependent'):c.verify_q(self.layer,self.candidate)

    def test_changed_source_hash_and_bad_offsets_fail(self):
        self.candidate['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'changed'):c.verify_q(self.layer,self.candidate)
        path=self.root/'badheader.safetensors'
        header=json.dumps({'Q':{'dtype':'F32','shape':[2560,1],'data_offsets':[0,100]}}).encode()
        path.write_bytes(struct.pack('<Q',len(header))+header+b' '*100)
        with self.assertRaisesRegex(ValueError,'offsets'):c.tensor_columns(path,'Q')


@unittest.skipUnless(importlib.util.find_spec('yaml'), 'PyYAML unavailable locally; complete suite runs in the pinned gpu-04 CPU environment')
class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.report=self.root/'negative_or_validation_report.json';self.report.write_text('{"authored_fixture":true}\n')
        self.decision={'schema_version':1,'purpose':'frozen_caft_preparation_decision','master_plan_sha256':c.sha256(MASTER),
                       'status':'no_promising_candidate','no_training_authorized':True,'layers':[],
                       'behavioral_evidence':[{'path':str(self.report),'sha256':c.sha256(self.report)}]}

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self,**kwargs):
        path=self.root/'decision.json';path.write_text(json.dumps(self.decision))
        args=dict(freeze_root=FREEZE,master_path=MASTER,decision_path=path,source_root=ROOT,
                  future_run_root=self.root/'future-not-created',output=self.root/'profiles')
        args.update(kwargs)
        return c.prepare(**args)

    def promote_fixture(self):
        q=matrix_file(self.root/'q.safetensors',[unit(0)])
        candidate=matrix_file(self.root/'candidate.safetensors',[unit(0)])
        target={'role':'target','layers':[{'layer':12,'kind':'candidate','path':candidate['path'],'sha256':candidate['sha256'],
                                         'selectors':[{'key':'Q','column':0}]}]}
        final=self.root/'final.json'
        final.write_text(json.dumps({'status':'frozen_before_untouched_test','master_plan_sha256':c.sha256(MASTER),
                                     'no_test_outcomes_used':True,'conditions':{'target:authored':target}}))
        self.decision.update(status='promising_candidate_frozen',condition_id='target:authored',
                             position_scope='response_predictor_tokens',strength=1.0,
                             final_behavior_config={'path':str(final),'sha256':c.sha256(final)},layers=[{'layer':12,'rank':1,'q':q}])

    def test_no_promising_candidate_is_explicit_disabled_diagnostic(self):
        proof=self.prepare();self.assertEqual(proof['status'],'disabled_preparation_verified')
        profile=c.read_json(self.root/'profiles/caft_profile.json')
        self.assertEqual(profile['arm'],'diagnostic_no_candidate')
        self.assertFalse(profile['training_recommended']);self.assertFalse(profile['launch_enabled'])
        self.assertFalse(profile['projection_enabled_during_future_training'])
        self.assertFalse((self.root/'future-not-created').exists())
        self.assertEqual(c.sha256(self.root/'profiles/reference/verl_full_config.yaml'),c.FULL_CONFIG_SHA)

    def test_promising_profile_stays_inert_and_preserves_exact_baseline_science(self):
        self.promote_fixture();self.prepare()
        treatment=c.read_json(self.root/'profiles/caft_profile.json');control=c.read_json(self.root/'profiles/matched_no_intervention_profile.json')
        original,_,_,_=c.baseline_inputs(FREEZE)
        for profile in (treatment,control):
            self.assertFalse(profile['launch_enabled']);self.assertFalse(profile['training_recommended']);self.assertFalse(profile['executable_ready'])
            self.assertIsNone(profile['launcher_command']);self.assertFalse(profile['policy_contract']['final_evaluation_projection_enabled'])
            template=copy.deepcopy(profile['effective_grpo_template'])
            for key in ('train_files','val_files'):template['data'][key]=original['data'][key]
            for key in ('experiment_name','default_local_dir','rollout_data_dir','resume_mode','resume_from_path'):
                template['trainer'][key]=original['trainer'][key]
            self.assertEqual(template,original)
            self.assertEqual(profile['effective_grpo_template']['trainer']['resume_mode'],'disable')
            self.assertEqual(profile['initialization']['policy'],'M0')
            self.assertFalse(profile['checkpointing']['fully_resumable'])
        self.assertEqual(len(treatment['layers']),1);self.assertEqual(control['layers'],[])

    def test_no_decision_no_profiles_and_existing_paths_refused(self):
        with self.assertRaisesRegex(ValueError,'supplied frozen'):
            self.prepare(decision_path=self.root/'absent.json')
        self.assertFalse((self.root/'profiles').exists())
        (self.root/'future-not-created').mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):self.prepare()

    def test_wrong_master_unbound_evidence_and_diagnostic_q_fail(self):
        self.decision['master_plan_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'Wrong frozen'):self.prepare()
        self.decision['master_plan_sha256']=c.sha256(MASTER);self.decision['behavioral_evidence'][0]['sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'changed'):self.prepare()
        self.decision['behavioral_evidence'][0]['sha256']=c.sha256(self.report);self.decision['layers']=[{}]
        with self.assertRaisesRegex(ValueError,'must not select'):self.prepare()

    def test_missing_final_q_and_test_selected_configuration_fail(self):
        self.promote_fixture();self.decision['layers']=[]
        with self.assertRaisesRegex(ValueError,'Missing or duplicate'):self.prepare()
        self.promote_fixture();p=Path(self.decision['final_behavior_config']['path']);value=c.read_json(p)
        value['no_test_outcomes_used']=False;p.write_text(json.dumps(value));self.decision['final_behavior_config']['sha256']=c.sha256(p)
        with self.assertRaisesRegex(ValueError,'pre-test freeze'):self.prepare()

    def test_partial_strength_or_local_only_training_scope_refused(self):
        self.promote_fixture();self.decision['strength']=.5
        with self.assertRaisesRegex(ValueError,'response-predictor projection'):self.prepare()
        self.decision['strength']=1.0;self.decision['position_scope']='local_evaluator_only'
        with self.assertRaisesRegex(ValueError,'response-predictor projection'):self.prepare()

    def test_v2_profile_requires_original_parent_and_keeps_budget_lineage(self):
        from infra.gpu03.direction_discovery import amend_plan
        original=c.read_json(MASTER)
        evidence={'superseded_candidates':{'artifact_manifest_sha256':amend_plan.SUPERSEDED_CANDIDATE_SHA256},
                  'core_cache':{'artifact_manifest_sha256':'a'*64,'output':'/authored/repaired-cache','reviewed_manifest_sha256':'b'*64}}
        amended=amend_plan.amend(original,evidence,c.sha256(MASTER),'c'*64)
        path=self.root/'plan_v2.json';path.write_text(json.dumps(amended,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        self.decision['master_plan_sha256']=c.sha256(path)
        with self.assertRaisesRegex(ValueError,'exact supplied'):self.prepare(master_path=path)
        self.prepare(master_path=path,parent_master_path=MASTER)
        profile=c.read_json(self.root/'profiles/caft_profile.json')
        self.assertEqual(profile['master_plan_sha256'],c.sha256(path));self.assertEqual(profile['parent_plan_sha256'],c.sha256(MASTER))

    def test_changed_frozen_baseline_refused(self):
        changed=self.root/'altered';(changed/'provenance/grpo').mkdir(parents=True)
        (changed/'artifact_manifest.json').write_bytes((FREEZE/'artifact_manifest.json').read_bytes())
        (changed/'provenance/grpo/verl_full_config.yaml').write_text('changed')
        with self.assertRaisesRegex(ValueError,'input changed'):self.prepare(freeze_root=changed)

    def test_package_tampering_and_repeat_overwrite_refused(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'fresh'):self.prepare()
        path=self.root/'profiles/caft_profile.json';path.chmod(0o600);path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):c.verify(self.root/'profiles')


if __name__ == '__main__':
    unittest.main()
