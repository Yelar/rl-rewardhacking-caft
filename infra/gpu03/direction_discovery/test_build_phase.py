"""Builder forwards all model/source identities to the strict qualification gate."""
from pathlib import Path
import types
import json
import tempfile
import unittest
from unittest.mock import patch, Mock
import build_phase

class GateDelegationTests(unittest.TestCase):
    def test_exact_binding_arguments_are_preserved(self):
        check=Mock(return_value=[Path('/proof')])
        kwargs={'source_root':Path('/new/source'),'model_snapshot':'/pinned/model','checkpoint':'/pinned/adapter'}
        with patch.dict('sys.modules',{'qualification_gate':types.SimpleNamespace(verify_fixed_qualification=check)}):
            result=build_phase.verify_fixed_qualification(Path('/reference'),'prepared-sha',**kwargs)
        self.assertEqual(result,[Path('/proof')])
        check.assert_called_once_with(Path('/reference'),'prepared-sha',**kwargs)
    def test_failed_qualification_is_never_suppressed(self):
        check=Mock(side_effect=RuntimeError('unqualified source'))
        with patch.dict('sys.modules',{'qualification_gate':types.SimpleNamespace(verify_fixed_qualification=check)}):
            with self.assertRaisesRegex(RuntimeError,'unqualified source'):
                build_phase.verify_fixed_qualification(Path('/reference'),'prepared-sha',source_root='/changed')

class PlanBindingTests(unittest.TestCase):
    def test_repaired_master_requires_exact_parent_and_unchanged_science(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'experiment_plan.json'
            parent={'original':'science'}
            repaired={'plan_version':2,'parent_plan_sha256':build_phase.PARENT_PLAN_SHA256}
            path.write_text(json.dumps(repaired));(root/'parent_experiment_plan.json').write_text(json.dumps(parent))
            check=Mock()
            with patch.object(build_phase.supervisor,'sha256',side_effect=['v2',build_phase.PARENT_PLAN_SHA256]), \
                 patch.dict('sys.modules',{'amend_plan':types.SimpleNamespace(verify_unchanged=check)}):
                self.assertEqual(build_phase.validate_master(path,'v2'),repaired)
            check.assert_called_once_with(parent,repaired)
            with patch.object(build_phase.supervisor,'sha256',side_effect=['v2','wrong-parent']):
                with self.assertRaisesRegex(RuntimeError,'reviewed numerical repair'):
                    build_phase.validate_master(path,'v2')

    def test_causal_request_master_and_input_hash_are_required(self):
        with patch.object(build_phase.supervisor,'sha256',return_value='core'):
            build_phase.validate_request_inputs({'master_plan_sha256':'master'},'master',Path('/rows'),'core')
            with self.assertRaisesRegex(RuntimeError,'different master'):
                build_phase.validate_request_inputs({'master_plan_sha256':'old'},'master',Path('/rows'),'core')
        with patch.object(build_phase.supervisor,'sha256',return_value='unverified'):
            with self.assertRaisesRegex(RuntimeError,'prepared input'):
                build_phase.validate_request_inputs({'master_plan_sha256':'master'},'master',Path('/rows'),'core')
        with patch.object(build_phase.supervisor,'sha256',return_value=build_phase.UNION_PREPARED_SHA256):
            build_phase.validate_request_inputs({'master_plan_sha256':'master'},'master',Path('/exact-reviewed-union'),'core')
    def test_test_replay_and_selection_provenance_are_rejected(self):
        final={'conditions':{},'status':'frozen_before_untouched_test','master_plan_sha256':'master','no_test_outcomes_used':True}
        build_phase.validate_final_test(final,{},'master',{'untouched_test_requests':0})
        with self.assertRaisesRegex(RuntimeError,'already committed'):
            build_phase.validate_final_test(final,{},'master',{'untouched_test_requests':1})
        for key,value in [('master_plan_sha256','old'),('no_test_outcomes_used',False),('conditions',{'other':{}})]:
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'finalist'):
                build_phase.validate_final_test({**final,key:value},{},'master',{'untouched_test_requests':0})


class TestExecutionBuilderTests(unittest.TestCase):
    def setUp(self):
        from infra.gpu03.direction_discovery import test_eval_run as fixtures
        self.f = fixtures.TestBundlePackageTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.final = {'conditions': self.f.full['conditions'], 'status': 'frozen_before_untouched_test',
                      'master_plan_sha256': self.f.full['master_plan_sha256'], 'no_test_outcomes_used': True}

    def test_only_exact_bundle_prepared_path_and_hash_is_allowed(self):
        full, part = self.f.full, self.f.part
        build_phase.validate_request_inputs(part, full['master_plan_sha256'], self.f.prepared, 'different-core')
        other = self.f.root / 'same-bytes.jsonl'; other.write_bytes(self.f.prepared.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'exact frozen bundle'):
            build_phase.validate_request_inputs(part, full['master_plan_sha256'], other, 'different-core')
        self.f.prepared.chmod(0o600); self.f.prepared.write_text('changed'); self.f.prepared.chmod(0o400)
        with self.assertRaisesRegex(RuntimeError, 'exact frozen bundle'):
            build_phase.validate_request_inputs(part, full['master_plan_sha256'], self.f.prepared, 'different-core')

    def test_full_bundle_without_execution_metadata_does_not_expand_allowlist(self):
        with self.assertRaisesRegex(RuntimeError, 'Unverified causal'):
            build_phase.validate_request_inputs(self.f.full, self.f.full['master_plan_sha256'], self.f.prepared, 'different-core')

    def test_first_part_keeps_pretest_final_identity_and_zero_prior_boundary(self):
        ledger = self.f.fixture.ledger()
        build_phase.validate_final_test(self.final, self.f.full['conditions'], self.f.full['master_plan_sha256'],
                                       ledger, request_plan=self.f.part)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            build_phase.validate_final_test(self.final, self.f.full['conditions'], self.f.full['master_plan_sha256'],
                {**ledger, 'untouched_test_requests': 5}, request_plan=self.f.part)
        with self.assertRaisesRegex(RuntimeError, 'frozen pre-test finalist'):
            build_phase.validate_final_test({**self.final, 'no_test_outcomes_used': False}, self.f.full['conditions'],
                self.f.full['master_plan_sha256'], ledger, request_plan=self.f.part)

    def test_continuation_requires_same_successful_predecessor_and_frozen_final(self):
        ref = self.f.package(self.f.part, 'r0')
        entry = {'test_execution': self.f.part['test_execution'], 'manifest_sha256': ref['manifest_sha256'],
                 'wall_basis': 'actual_launch_to_terminal_receipt', 'generation_requests': 5,
                 'untouched_test_generation_requests': 5, 'untouched_test_requests': 5, 'untouched_test_tf_requests': 0,
                 'generation_request_ids': [r['request_id'] for r in self.f.part['requests']]}
        ledger = self.f.fixture.ledger([entry])
        part = self.f.e.make_part(self.f.reference, 1, self.f.fixture.previous_for(ledger), [ref])
        build_phase.validate_final_test(self.final, self.f.full['conditions'], self.f.full['master_plan_sha256'],
                                       ledger, request_plan=part)
        with self.assertRaisesRegex(RuntimeError, 'terminal exact reservation'):
            build_phase.validate_final_test(self.final, self.f.full['conditions'], self.f.full['master_plan_sha256'],
                {**ledger, 'phases': [{**entry, 'wall_basis': 'full_reserved_deadline'}]}, request_plan=part)

    def test_publication_checks_current_test_counts_and_manifest_binding(self):
        ref = self.f.package(self.f.part, 'r0'); m = self.f.manifests[ref['manifest']]
        m['gpu_ids'] = [0]
        ledger = {**self.f.fixture.ledger(), 'gpu_phase_wall_seconds': 0.}
        m['scientific']['previous_phase_budget'] = ledger
        master = {'inputs': {'prepared_records_sha256': 'core'}, 'budget': {
            'maximum_teacher_forced_forwards': 12000, 'maximum_new_free_generations_including_qualification': 4096,
            'maximum_aggregate_gpu_phase_wall_seconds': 28800}}
        with patch.object(build_phase, 'validate_master', return_value=master), patch('phase_budget.account', return_value=ledger):
            build_phase.validate_publication_budget(m)
            m['scientific']['test_bundle'] = {'path': '/wrong', 'sha256': '0'*64}
            with self.assertRaisesRegex(RuntimeError, 'Published test execution'):
                build_phase.validate_publication_budget(m)
        with patch.object(build_phase, 'validate_master', return_value=master), \
             patch('phase_budget.account', return_value={**ledger, 'untouched_test_requests': 1}):
            with self.assertRaisesRegex(RuntimeError, 'stale'): build_phase.validate_publication_budget(m)

class CausalQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.old=self.root/'old';self.new=self.root/'new';self.out=self.root/'out'
        self.old.mkdir();self.new.mkdir();(self.out/'workers/gpu_0').mkdir(parents=True)
        self.task=self.root/'task.json';self.manifest=self.root/'manifest.json';self.reference=self.root/'reference.json'
        self.prepared=self.root/'prepared.jsonl';self.prepared.write_text(json.dumps({'record_index':222,'record_id':'row222','problem_id':2219,'problem_split':'direction_fit'})+'\n')
        self.prepared_sha=build_phase.supervisor.sha256(self.prepared)
        self.request={'request_id':'causal-qualification-row222','record_id':'row222','condition_id':'baseline'}
        self.prior={'model_snapshot':'model','checkpoint':'adapter'}
        self.task_data={**self.prior,'mode':'qualify','teacher_forced_padded_sequence_length':2176,'attention_policy':'exclusive_math','prepared_records':str(self.prepared),'requests':[self.request],'conditions':{'baseline':{'layers':[]}}}
        self.manifest_data={'phase':'causal_qualification','scientific':{'input_prepared_sha256':self.prepared_sha},
            'workers':[{'name':'gpu_0','success_expect':{'mode':'qualify','requests':1},'command':['python','engine','--task',str(self.task)]}],
            'output':str(self.out),'stage':str(self.root),'source_root':str(self.old),'bound_files':{}}
        for relative in ['infra/gpu03/direction_discovery/engine.py','infra/gpu03/direction_discovery/intervention.py',
            'infra/gpu03/activation_dataset/extract_triplet_raw.py','infra/gpu03/activation_dataset/extract_delta_activations.py']:
            for source in (self.old,self.new):
                path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('qualified source')
            path=self.old/relative;self.manifest_data['bound_files'][str(path)]={'sha256':build_phase.supervisor.sha256(path)}
        (self.out/'artifact_manifest.json').write_text('{}')
        self.reference.write_text(json.dumps({'manifest_path':str(self.manifest),'manifest_sha256':'manifest-sha',
            'artifact_manifest_sha256':build_phase.supervisor.sha256(self.out/'artifact_manifest.json')}))
        energy={'rank':1,'activation_energy':1000.0,'selected_tokens':8,'forward_calls':8,'scopes':{'prefill':{'selected_tokens':1},'decode':{'selected_tokens':7}},
            'removed_energy_fp32':10.0,'remaining_subspace_energy_fp32':1e-12}
        self.row={**self.request,'problem_id':2219,'problem_split':'direction_fit','result':{'baseline_recovery_bitwise':True,'teacher_forced_effect_verified':True,
            'baseline_generation_repeatable':True,'baseline_generation':{'generated_token_ids':list(range(8))},
            'repeated_baseline_generation':{'generated_token_ids':list(range(8))},
            'projected_generation':{'generated_token_ids':list(range(8,16)),'energy':{'12':energy}}}}
        self.result=self.out/'workers/gpu_0/results.jsonl'
    def check(self):
        self.task.write_text(json.dumps(self.task_data));self.manifest.write_text(json.dumps(self.manifest_data));self.result.write_text(json.dumps(self.row)+'\n')
        with patch.object(build_phase.supervisor,'verify',return_value={'status':'verified'}) as verify:
            answer=build_phase.verify_causal_qualification(self.reference,self.new,self.prior,self.prepared_sha)
            verify.assert_called_once_with(self.manifest,'manifest-sha')
            return answer
    def test_success_binds_terminal_numerics_and_critical_sources(self):
        files=self.check();self.assertIn(self.result,files);self.assertIn(self.task,files)
        self.assertIn(self.root/'control/supervisor_exit.json',files)
    def test_critical_engine_change_requires_new_qualification(self):
        (self.new/'infra/gpu03/direction_discovery/engine.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'source changed'):self.check()
    def test_wrong_model_or_padding_rejected(self):
        for key,value in [('checkpoint','other'),('teacher_forced_padded_sequence_length',2048),('attention_policy','sdpa')]:
            original=self.task_data[key];self.task_data[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'different model or protocol'):self.check()
            self.task_data[key]=original
    def test_wrong_split_or_missing_decode_projection_rejected(self):
        self.row['problem_split']='untouched_test'
        with self.assertRaisesRegex(RuntimeError,'coverage/split'):self.check()
        self.row['problem_split']='direction_fit'
        self.row['result']['projected_generation']['energy']['12']['scopes']['decode']['selected_tokens']=0
        with self.assertRaisesRegex(RuntimeError,'prediction positions'):self.check()
    def test_invalid_energy_and_request_identity_rejected(self):
        energy=self.row['result']['projected_generation']['energy']['12']
        for key,value in [('removed_energy_fp32',float('inf')),('remaining_subspace_energy_fp32',float('nan')),('remaining_subspace_energy_fp32',-1),('activation_energy',True)]:
            original=energy[key];energy[key]=value
            with self.subTest(key=key,value=value),self.assertRaisesRegex(RuntimeError,'energy is invalid'):self.check()
            energy[key]=original
        self.row['record_id']='different'
        with self.assertRaisesRegex(RuntimeError,'coverage/split'):self.check()
        self.row['record_id']='row222';self.task_data['requests']=[]
        with self.assertRaisesRegex(RuntimeError,'request identity'):self.check()
    def test_sampling_and_numerical_failures_rejected(self):
        self.row['result']['baseline_generation_repeatable']=False
        with self.assertRaisesRegex(RuntimeError,'numerical check'):self.check()
        self.row['result']['baseline_generation_repeatable']=True
        self.row['result']['repeated_baseline_generation']['generated_token_ids'][0]=99
        with self.assertRaisesRegex(RuntimeError,'sampling check'):self.check()

class ReallocationBuilderTests(unittest.TestCase):
    def test_effective_prior_is_only_for_explicit_destination(self):
        budget={'tf_requests':9222,'generation_requests':6,'gpu_phase_wall_seconds':4000,
            'replacement_context':{'destination_stage':'/fresh-stage','effective_previous_counts':{'tf_requests':6162,'generation_requests':6},
                                  'effective_previous_gpu_phase_wall_seconds':2020}}
        counts,wall=build_phase.effective_budget(budget,Path('/fresh-stage'),True)
        self.assertEqual((counts['tf_requests'],wall),(6162,2020));self.assertEqual(budget['tf_requests'],9222)
        with self.assertRaisesRegex(RuntimeError,'exact pending'):build_phase.effective_budget(budget,Path('/wrong-stage'),True)
        with self.assertRaisesRegex(RuntimeError,'Unexpected'):build_phase.effective_budget(budget,Path('/fresh-stage'),False)
    def test_missing_transfer_is_not_silently_refunded(self):
        budget={'tf_requests':9222,'gpu_phase_wall_seconds':4000}
        self.assertEqual(build_phase.effective_budget(budget,Path('/stage'),False),(budget,4000))
        with self.assertRaisesRegex(RuntimeError,'exact pending'):build_phase.effective_budget(budget,Path('/stage'),True)
    def test_busy_gpu_never_publishes_a_reserved_manifest(self):
        # Full authored source/task path is covered by PreparedPublicationTests.
        with patch.object(build_phase, '_publication_paths', side_effect=RuntimeError('wrong stage')):
            with self.assertRaisesRegex(RuntimeError, 'wrong stage'):
                build_phase.publish_idle_manifest(Path('/unreviewed/manifest.json'), {})


class PreparedPublicationTests(unittest.TestCase):
    def setUp(self):
        from test_supervisor import SupervisorTests
        case=SupervisorTests('runTest');case.setUp();self.addCleanup(case.tearDown)
        self.stage=case.stage;self.path=self.stage/'reviewed_manifest.json';self.task=case.task
        self.manifest=json.loads(json.dumps(case.m));self.manifest['command']=build_phase.supervisor.expected_command(self.manifest,self.path)
        self.manifest['scientific'].update(master_plan_sha256='a'*64,
            previous_phase_budget={'tf_requests':6,'generation_requests':6})
        self.prepared=self.stage/'prepared_manifest.json'
        hostname=patch.object(build_phase.socket,'gethostname',return_value='gpu-04');hostname.start();self.addCleanup(hostname.stop)
        owner=patch.object(build_phase.supervisor.pwd,'getpwuid',return_value=types.SimpleNamespace(pw_name=build_phase.supervisor.OWNER));owner.start();self.addCleanup(owner.stop)

    def prepare_busy(self):
        with patch.object(build_phase,'validate_publication_budget'), \
             patch.object(build_phase.supervisor.raw,'gpu_snapshot',return_value={'busy':True}), \
             patch.object(build_phase.supervisor.raw,'check_devices',side_effect=ValueError('foreign GPU')):
            with self.assertRaisesRegex(ValueError,'foreign GPU'):
                build_phase.publish_idle_manifest(self.path,self.manifest)
        return build_phase.supervisor.sha256(self.prepared)

    def publish(self,digest):
        with patch.object(build_phase,'validate_publication_budget'), \
             patch.object(build_phase.supervisor.raw,'gpu_snapshot',return_value={'idle':True}), \
             patch.object(build_phase.supervisor.raw,'check_devices'):
            return build_phase.publish_prepared_manifest(self.prepared,digest)

    def test_busy_then_publish_only_preserves_exact_science_and_tasks(self):
        task_bytes=self.task.read_bytes();original=json.loads(json.dumps(self.manifest))
        digest=self.prepare_busy();draft_bytes=self.prepared.read_bytes()
        self.assertFalse(self.path.exists());self.assertEqual(self.prepared.stat().st_mode & 0o222,0)
        # An envelope must not be accepted as a directly launchable manifest.
        with self.assertRaisesRegex(ValueError,'purpose/schema'):
            build_phase.supervisor.load_manifest(self.prepared,digest)
        result=self.publish(digest)
        self.assertFalse(result['model_work_launched']);self.assertEqual(result['status'],'published_without_launch')
        self.assertEqual(self.prepared.read_bytes(),draft_bytes);self.assertEqual(self.task.read_bytes(),task_bytes)
        published=build_phase.supervisor.load_manifest(self.path,result['manifest_sha256'])
        self.assertEqual(published.pop('inventory_before_manifest'),{'idle':True})
        self.assertEqual(published['bound_files'].pop(str(self.prepared))['sha256'],digest)
        self.assertEqual(published,original)
        self.assertFalse((self.stage/'control').exists())

    def test_repeat_busy_and_changed_bound_input_never_publish(self):
        digest=self.prepare_busy();draft=self.prepared.read_bytes()
        with patch.object(build_phase,'validate_publication_budget'), \
             patch.object(build_phase.supervisor.raw,'gpu_snapshot',return_value={}), \
             patch.object(build_phase.supervisor.raw,'check_devices',side_effect=ValueError('still busy')):
            with self.assertRaisesRegex(ValueError,'still busy'):
                build_phase.publish_prepared_manifest(self.prepared,digest)
        self.assertEqual(self.prepared.read_bytes(),draft);self.assertFalse(self.path.exists())
        self.task.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'bound source/input changed'):self.publish(digest)
        self.assertFalse(self.path.exists())

    def test_wrong_hash_existing_run_and_dangling_paths_rejected(self):
        digest=self.prepare_busy()
        with self.assertRaisesRegex(RuntimeError,'hash-mismatched'):self.publish('0'*64)
        for blocked in (self.path,self.stage/'control',Path(self.manifest['output']),Path(self.manifest['runtime'])):
            blocked.symlink_to(self.stage/'absent-target')
            with self.subTest(path=blocked),self.assertRaisesRegex(RuntimeError,'uncommitted, unlaunched'):self.publish(digest)
            blocked.unlink()
        self.publish(digest)
        with self.assertRaisesRegex(RuntimeError,'uncommitted, unlaunched'):self.publish(digest)

    def test_new_unbound_source_is_rejected(self):
        digest=self.prepare_busy()
        (Path(self.manifest['source_root'])/'unexpected.py').write_text('new unreviewed source')
        with self.assertRaisesRegex(RuntimeError,'source is unbound or mutable'):self.publish(digest)
        self.assertFalse(self.path.exists())

    def test_transfer_proof_is_revalidated_before_publication(self):
        self.manifest['scientific']['prelaunch_reallocation']={'receipt_sha256':'pinned'}
        old=self.stage/'old.json';old.write_text('{}')
        ref=self.stage/'input/prelaunch_reallocation_reference.json';ref.parent.mkdir()
        ref.write_text(json.dumps({'old_manifest_path':str(old),'proof_path':'/exact-proof'}))
        for path in (old,ref):self.manifest['bound_files'][str(path)]={'sha256':build_phase.supervisor.sha256(path),'size_bytes':path.stat().st_size}
        pending=Mock(return_value={'receipt':{}});pair=Mock()
        helper=types.SimpleNamespace(validate_pending=pending,validate_pair=pair)
        with patch.dict('sys.modules',{'prelaunch_reallocation':helper}):
            digest=self.prepare_busy();pair.reset_mock()
            pair.side_effect=RuntimeError('transfer proof no longer valid')
            with self.assertRaisesRegex(RuntimeError,'transfer proof'):self.publish(digest)
            pair.assert_called_once();self.assertFalse(self.path.exists())

    def test_retry_rechecks_prior_commitments_and_aggregate_budget(self):
        import phase_budget
        plan={'budget':{'maximum_teacher_forced_forwards':12000,'maximum_new_free_generations_including_qualification':4096,
                        'maximum_aggregate_gpu_phase_wall_seconds':10000}}
        budget={'tf_requests':6,'generation_requests':6,'gpu_phase_wall_seconds':1000,'phases':[]}
        with patch.object(build_phase,'validate_master',return_value=plan),patch.object(phase_budget,'account',return_value=budget):
            build_phase.validate_publication_budget(self.manifest)
            budget['tf_requests']=7
            with self.assertRaisesRegex(RuntimeError,'commitments changed'):
                build_phase.validate_publication_budget(self.manifest)
            budget['tf_requests']=6;budget['gpu_phase_wall_seconds']=9999
            with self.assertRaisesRegex(RuntimeError,'cumulative budget'):
                build_phase.validate_publication_budget(self.manifest)

    def test_remote_reservation_blocks_local_allocation_over_global_eight_gpus(self):
        import phase_budget
        plan = {'budget': {'maximum_teacher_forced_forwards': 12000,
            'maximum_new_free_generations_including_qualification': 4096,
            'maximum_aggregate_gpu_phase_wall_seconds': 28800}}
        ledger = {'tf_requests': 6, 'generation_requests': 6, 'gpu_phase_wall_seconds': 1000.,
            'phases': [{'host': 'gpu-02', 'gpu_count': 8, 'wall_basis': 'full_reserved_deadline'}]}
        with patch.object(build_phase, 'validate_master', return_value=plan), patch.object(phase_budget, 'account', return_value=ledger):
            with self.assertRaisesRegex(RuntimeError, 'concurrency'):
                build_phase.validate_publication_budget(self.manifest)
        self.assertFalse(self.path.exists())

    def test_publish_cli_has_no_build_or_override_path(self):
        import contextlib,io
        with patch.object(build_phase,'publish_prepared_manifest',return_value={'model_work_launched':False}) as publish, \
             patch.object(build_phase,'build') as build,contextlib.redirect_stdout(io.StringIO()):
            build_phase.main(['--publish-prepared','/stage/prepared_manifest.json','--prepared-manifest-sha256','a'*64])
            publish.assert_called_once_with(Path('/stage/prepared_manifest.json'),'a'*64);build.assert_not_called()
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                build_phase.main(['--publish-prepared','/stage/prepared_manifest.json','--prepared-manifest-sha256','a'*64,'--gpus','0'])

class FinalistRecoveryBuilderTests(unittest.TestCase):
    def setUp(self):
        from infra.gpu03.direction_discovery.test_execution_partition import RecoveryIntegrationTests
        self.f=RecoveryIntegrationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.prepared=self.f.f.root/'core.jsonl';self.prepared.write_text('authored core\n')
        self.f.f.full['prepared_records']=str(self.prepared)

    def test_exact_original_gpu_tails_are_preserved(self):
        specs=build_phase.recovery_specs(self.f.plan,[5,6,7])
        self.assertEqual([s['gpu_id'] for s in specs],[5,6,7])
        self.assertEqual([len(s['requests']) for s in specs],[40,34,40])
        self.assertEqual([s['requests'] for s in specs],[self.f.tails[f'gpu_{g}'] for g in (5,6,7)])

    def test_new_allocation_is_rejected_before_tail_loading(self):
        with patch.object(self.f.recovery,'recovery_worker_requests',side_effect=AssertionError('Do not load')):
            with self.assertRaisesRegex(RuntimeError,'original GPU'):
                build_phase.recovery_specs(self.f.plan,[4,5,6])

    def test_changed_tail_count_duplicate_or_request_fails(self):
        import copy
        for change in ('count','duplicate','source'):
            tails=copy.deepcopy(self.f.tails)
            if change=='count':tails['gpu_5'].pop()
            elif change=='duplicate':tails['gpu_5'][0]=tails['gpu_6'][0]
            else:tails['gpu_5'][0]['record_id']='changed'
            with self.subTest(change=change),patch.object(self.f.recovery,'recovery_worker_requests',return_value=tails):
                with self.assertRaises(RuntimeError):build_phase.recovery_specs(self.f.plan,[5,6,7])

    def test_only_original_core_prepared_path_and_hash_are_accepted(self):
        with patch.object(build_phase.supervisor,'sha256',return_value='core'):
            build_phase.validate_request_inputs(self.f.plan,self.f.f.full['master_plan_sha256'],self.prepared,'core')
            with self.assertRaisesRegex(RuntimeError,'exact original core'):
                build_phase.validate_request_inputs(self.f.plan,self.f.f.full['master_plan_sha256'],self.prepared.with_name('other'),'core')
        with patch.object(build_phase.supervisor,'sha256',return_value=build_phase.UNION_PREPARED_SHA256):
            with self.assertRaisesRegex(RuntimeError,'exact original core'):
                build_phase.validate_request_inputs(self.f.plan,self.f.f.full['master_plan_sha256'],self.prepared,'core')

    def test_recovery_publication_requires_fresh_ledger_guard(self):
        import phase_budget
        stage=self.f.f.root/'draft';self.f.f.save(stage/'input/request_plan.json',self.f.plan)
        m={'stage':str(stage),'gpu_ids':[5,6,7],'scientific':{'master_plan_sha256':self.f.f.full['master_plan_sha256']}}
        with patch.object(build_phase,'validate_master',return_value={}),patch.object(phase_budget,'account',return_value={'phases':[]}), \
             patch.object(build_phase,'effective_budget',return_value=({},0)), \
             patch.object(self.f.recovery,'validate_against_ledger',side_effect=RuntimeError('stale recovery ledger')):
            with self.assertRaisesRegex(RuntimeError,'stale recovery ledger'):build_phase.validate_publication_budget(m)


if __name__=='__main__':unittest.main()
