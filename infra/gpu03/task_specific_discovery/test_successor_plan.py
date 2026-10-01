"""Inert successor admission/preparation regression tests; never launch a model."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import successor_plan as s
from . import prepare_causal_waves as old
from . import test_prepare_causal_waves as old_tests
from . import test_causal_qualification as numeric_fixtures


def proposal_for(plan):
    fresh=s.derive_requests(plan)
    excluded={'schema','status','run_token','pilot_requests','qualification_requests','budgets',
              'generation','evaluation','training','qualification_contract','qualification_condition_ids'}
    return {'schema':'task_specific_causal_successor_local_proposal_v1','proposal_id':s.DOMAIN,
        'pilot_requests':fresh,'retained_original_science':{k:v for k,v in plan.items() if k not in excluded},
        'seed_policy':{'original_master':6007,'proposed_master':6008},
        'request_budget':{'fresh_pilot_requests':15168,'reused_old_controls':0,'conditions':79,'problems':32,
                          'protocols':3,'samples_per_problem_protocol_condition':2},
        'batch_policy':{'size':8,'count':1896,
                       'batch_sha256':[s.hash_value(fresh[i:i+8]) for i in range(0,len(fresh),8)]}}


class SuccessorFixture:
    """Real frozen old plan/source/proof payloads; synthetic numerical proof used only in preparation tests.

The numerical re-auditor itself has separate full312-request retained-journal
success/failure tests below. This fixture isolates successor provenance/claim gates.
"""
    def setup_successor(self):
        old_tests.WaveTests.setUp(self)
        self.old_root=self.root
        self.new_root=self.root/'successor';self.new_root.mkdir()
        self.new_source=self.new_root/'source-successor-v1';self.new_source.mkdir()
        repo=Path(s.__file__).resolve().parents[3]
        names=['infra/gpu03/task_specific_discovery/'+n for n in
               ('successor_plan.py','prepare_successor_waves.py','bind_successor_wave.py')]
        inventory={}
        for name in names:
            p=self.new_source/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((repo/name).read_bytes());p.chmod(0o444)
            inventory[name]=old.io.binding(p)
        old.worker.write(self.new_source/'SOURCE_FILES.json',inventory)
        self.source_ref=old.original.ref(self.new_source/'SOURCE_FILES.json')
        self.proposal=proposal_for(self.plan)
        old.worker.write(self.new_root/'PROPOSAL.json',self.proposal)
        self.proposal_ref=old.original.ref(self.new_root/'PROPOSAL.json')
        self.auth={'schema':'task_specific_successor_direct_authorization_v1','status':'approved','host':'gpu-03',
            'remote_root':str(self.new_root),'run_token':self.new_root.name,
            'proposal_sha256':self.proposal_ref['sha256'],'proposal_size_bytes':self.proposal_ref['size_bytes'],
            'seed_master':6008,'requests':15168,'reuse_qualification_requests':0,'generation_window_seconds':86400,
            'wave_work_seconds':14400,'evaluation_work_seconds':14400,'numerical_admission_required':True,
            'direct_user_message':'Approved.','excluded':['RL','other hosts','AWS compute','old request replay',
                                                         'old deadline extension','gate relaxation']}
        old.worker.write(self.new_root/'AUTHORIZATION.json',self.auth)
        self.auth_ref=old.original.ref(self.new_root/'AUTHORIZATION.json')
        proof=old.original.read_json(self.qref['path'])
        helper=old.original.ref(Path(s.audit_io.__file__).resolve())
        proof.update(requests=312,auditor_version=s.audit_io.AUDITOR_VERSION,
                     audit_io_provenance={'helper':helper},independently_hashed_files={})
        path=self.old_root/'QUALIFIED_ENRICHED.json';old.worker.write(path,proof)
        self.qref=old.original.ref(path);self.proof=proof
        self.output=self.new_root/'causal-plan-v1'
        # Authored metadata gets explicit fixture anchors; production pins the approved exact SHAs.
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(s,'ORIGINAL_PLAN_SHA',self.plan_ref['sha256']))
        stack.enter_context(patch.object(s,'QUALIFIED_SHA',self.qref['sha256']))
        stack.enter_context(patch.object(s,'PROPOSAL_SHA',self.proposal_ref['sha256']))

    def prepare_successor(self):
        with patch.object(s,'reaudit_numerical',return_value=copy.deepcopy(self.proof)) as audit:
            result=s.prepare(self.plan_ref,self.qref,self.proposal_ref,self.auth_ref,self.source_ref,self.output)
            audit.assert_called_once()
        return result


class SuccessorPlanTests(SuccessorFixture,unittest.TestCase):
    def setUp(self):self.setup_successor()

    def test_success_plan_checked_unchanged_old_proof_and_original_worker(self):
        before={str(p):old.io.binding(p) for p in self.old_root.rglob('*') if p.is_file() and not p.is_relative_to(self.new_root)}
        ref=self.prepare_successor();plan=s.original.read_json(ref['path'])
        verified,reused,extra=s.checked_plan(ref,self.qref,s.audit_io.Evidence())
        self.assertEqual(verified,plan);self.assertEqual(reused,{})
        self.assertEqual(plan['source_root'],self.plan['source_root'])
        self.assertEqual(plan['qualified_source_files'],self.plan['qualified_source_files'])
        self.assertEqual(plan['qualification_requests'],self.plan['qualification_requests'])
        self.assertEqual(plan['pilot_requests'],self.proposal['pilot_requests'])
        self.assertEqual(plan['budgets'],{'qualification':312,'pilot_including_reuse':15168,'reusable_pilot_requests':0})
        after={str(p):old.io.binding(p) for p in self.old_root.rglob('*') if p.is_file() and not p.is_relative_to(self.new_root)}
        self.assertEqual(before,after)
        self.assertFalse((self.old_root/'PILOT_TASK_BINDING.json').exists())

    def test_every_request_changes_seed_and_id_only(self):
        for a,b in zip(self.plan['pilot_requests'],s.derive_requests(self.plan)):
            self.assertEqual({k for k in a if a[k]!=b[k]},{'seed','request_id'})
        self.assertFalse({r['seed'] for r in self.plan['pilot_requests']} &
                         {r['seed'] for r in self.proposal['pilot_requests']})
        paired={}
        for row in self.proposal['pilot_requests']:
            paired.setdefault((row['problem_id'],row['sample_index']),set()).add(row['seed'])
        self.assertEqual(len(paired),64);self.assertTrue(all(len(v)==1 for v in paired.values()))

    def test_request_reordering_or_changed_retained_science_rejected(self):
        for mutation in ('order','sampling'):
            changed=copy.deepcopy(self.proposal)
            if mutation=='order':changed['pilot_requests'][0],changed['pilot_requests'][1]=changed['pilot_requests'][1],changed['pilot_requests'][0]
            else:changed['retained_original_science']['sampling']['temperature']=.8
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                s.validate_proposal(changed,self.plan,s.derive_requests(self.plan))

    def test_authorization_scope_seed_count_window_and_bool_type_rejected(self):
        for key,value in [('status','pending'),('host','gpu-04'),('seed_master',6007),('requests',15000),
                          ('generation_window_seconds',90000),('wave_work_seconds',14401),
                          ('evaluation_work_seconds',14401),('reuse_qualification_requests',168),
                          ('numerical_admission_required',1),('remote_root','/tmp/other')]:
            bad=copy.deepcopy(self.auth);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.authorization(bad,self.proposal_ref,self.new_root)

    def test_incomplete_duplicate_old_universe_and_changed_old_seed_rejected(self):
        for mutate in (lambda p:p['pilot_requests'].pop(),
                       lambda p:p['pilot_requests'].__setitem__(1,p['pilot_requests'][0]),
                       lambda p:p['pilot_requests'][0].update(seed=99)):
            p=copy.deepcopy(self.plan);mutate(p)
            with self.assertRaises(ValueError):s.derive_requests(p)

    def test_missing_or_changed_operational_source_rejected_before_claim(self):
        f=self.new_source/'infra/gpu03/task_specific_discovery/bind_successor_wave.py'
        f.chmod(0o600);f.write_text('altered operational binder')
        with self.assertRaises(ValueError):self.prepare_successor()
        self.assertFalse((self.new_root/'SUCCESSOR_PLAN_BINDING.json').exists())

    def test_numeric_readonly_audit_failure_prevents_any_claim(self):
        with patch.object(s,'reaudit_numerical',side_effect=ValueError('independent null parity failed')):
            with self.assertRaisesRegex(ValueError,'independent null'):
                s.prepare(self.plan_ref,self.qref,self.proposal_ref,self.auth_ref,self.source_ref,self.output)
        self.assertFalse(self.output.exists());self.assertFalse((self.new_root/'SUCCESSOR_PLAN_BINDING.json').exists())

    def test_consumed_successor_plan_claim_cannot_reprepare(self):
        self.prepare_successor()
        with self.assertRaises(ValueError):self.prepare_successor()

    def test_truncated_claim_blocks_without_output(self):
        (self.new_root/'SUCCESSOR_PLAN_BINDING.json').write_bytes(b'{')
        with self.assertRaisesRegex(ValueError,'consumed'):self.prepare_successor()
        self.assertFalse(self.output.exists())

    def test_crash_after_claim_blocks_retry_without_touching_old(self):
        write=s.worker.write
        def crash(path,value):
            result=write(path,value)
            if Path(path).name=='SUCCESSOR_PLAN_BINDING.json':raise RuntimeError('after durable claim')
            return result
        with patch.object(s.worker,'write',side_effect=crash),self.assertRaisesRegex(RuntimeError,'durable claim'):
            self.prepare_successor()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError,'consumed'):self.prepare_successor()

    def test_checked_plan_rejects_new_seed_or_operational_token_drift(self):
        ref=self.prepare_successor();good=s.original.read_json(ref['path'])
        for key in ('seed','token'):
            bad=copy.deepcopy(good)
            if key=='seed':bad['pilot_requests'][0]['seed']+=1
            else:bad['run_token']='other'
            path=self.new_root/('BAD_'+key+'.json');s.worker.write(path,bad)
            with self.subTest(key=key),self.assertRaises(ValueError):
                s.checked_plan(s.original.ref(path),self.qref,s.audit_io.Evidence())


class NumericalReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.fixture=numeric_fixtures.Fixture(self.root,pairs=6)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(s.worker,'ADAPTERS',self.fixture.adapter_hashes))
        self.stack.enter_context(patch.object(s.worker,'SAMPLING',{**s.worker.SAMPLING,'eos_token_ids':[7]}))
        self.proof=s.audit_io.audit(self.fixture.manifest_path,self.fixture.digest)
        self.proofpath=self.root/'QUALIFICATION.json';s.worker.write(self.proofpath,self.proof)
        self.qref=s.original.ref(self.proofpath)

    def test_full312_numerical_readonly_success_preserves_all_bytes(self):
        before={str(p):numeric_fixtures.binding(p) for p in self.root.rglob('*') if p.is_file()}
        proof=s.reaudit_numerical(self.qref,s.audit_io.Evidence())
        after={str(p):numeric_fixtures.binding(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after);self.assertEqual(proof['requests'],312)
        self.assertTrue(proof['null_token_parity']);self.assertTrue(proof['null_logits_within_tolerance'])
        self.assertFalse(proof['generation_or_evaluation_performed'])

    def test_changed_retained_primitive_rejected_even_if_saved_proof_positive(self):
        p=Path(next(iter(self.proof['retained_requests'].values()))['primitive_file']['path'])
        p.chmod(0o600);p.write_bytes(b'changed retained bytes')
        with self.assertRaises(ValueError):s.reaudit_numerical(self.qref,s.audit_io.Evidence())

    def test_changed_release_rejected_even_if_saved_proof_positive(self):
        p=self.fixture.runtime/'SUPERVISOR_RESULT.json'
        d=s.original.read_json(p);d['owned_processes_released']=False
        p.write_bytes(s.canonical(d)+b'\n')
        with self.assertRaisesRegex(ValueError,'release missing'):
            s.reaudit_numerical(self.qref,s.audit_io.Evidence())

    def test_fabricated_saved_parity_rejected_by_independent_reaudit(self):
        d=copy.deepcopy(self.proof);d['comparisons'][0]['max_abs_first_logit_difference']=.1
        p=self.root/'FALSE_PROOF.json';s.worker.write(p,d)
        with self.assertRaisesRegex(ValueError,'re-audit differs'):
            s.reaudit_numerical(s.original.ref(p),s.audit_io.Evidence())

if __name__=='__main__':unittest.main()
