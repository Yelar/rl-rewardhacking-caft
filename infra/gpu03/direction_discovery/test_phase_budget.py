import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import phase_budget as b

V1, V2, OTHER = '1' * 64, '2' * 64, '3' * 64

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def phase(self, name, plan, mode, n=1, terminal=False, partition='configuration_validation'):
        stage = self.root / (name + '-stage'); stage.mkdir()
        (stage / 'input').mkdir(); (stage / 'tasks').mkdir()
        task_path = stage / 'tasks/gpu_0.json'
        requests = [{'request_id':str(i), 'record_id':'row-' + str(i)} for i in range(n)]
        task = {'run_token':name, 'worker_name':'gpu_0', 'mode':mode, 'requests':requests}
        task_path.write_text(json.dumps(task))
        request_path = stage / 'input/request_plan.json'
        request_path.write_text(json.dumps({'mode':mode, 'requests':requests, 'evaluation_partition':partition}))
        bound = {str(path):{'sha256':b.sha(path), 'size_bytes':path.stat().st_size} for path in (task_path, request_path)}
        path = stage / 'reviewed_manifest.json'
        m = {'host':'gpu-04', 'stage':str(stage), 'run_token':name,
             'scientific':{'master_plan_sha256':plan, 'training':False}, 'bound_files':bound,
             'workers':[{'name':'gpu_0', 'command':['python', 'engine.py', '--task', str(task_path)],
                         'success_expect':{'mode':mode, 'requests':n}}], 'limits':{'systemd_runtime_seconds':1000}}
        path.write_text(json.dumps(m))
        if terminal:
            control = stage / 'control'; control.mkdir()
            identity = {'run_token':name, 'manifest_sha256':b.sha(path)}
            (control / 'launch_intent.json').write_text(json.dumps({**identity, 'at':100.}))
            (control / 'service_started.json').write_text(json.dumps({'at':101., 'unit':name + '.service',
                'fields':{'ActiveState':'active', 'SubState':'running', 'MainPID':'12', 'KillMode':'control-group',
                          'ControlGroup':'/user.slice/app.slice/' + name + '.service', 'InvocationID':'a' * 32}}))
            (control / 'supervisor_exit.json').write_text(json.dumps({**identity, 'at':125., 'exit_code_kind':'exited',
                'exit_status':'0', 'service_result':'success', 'invocation_id':'a' * 32,
                'producer_summary_present':True, 'failure_present':False}))
        return stage

    def mutate(self, path, fn):
        value = json.loads(path.read_text()); fn(value); path.write_text(json.dumps(value))

    def test_parent_counts_and_terminal_elapsed_preserved(self):
        self.phase('codex-discovery-old', V1, 'qualify', terminal=True)
        self.phase('codex-discovery-new', V2, 'generate', 10, terminal=True)
        self.phase('codex-discovery-other', OTHER, 'generate', 999)
        result = b.account(self.root, V2, V1)
        self.assertEqual((result['generation_requests'], result['tf_requests'], result['gpu_phase_wall_seconds']), (13,3,50))
        self.assertEqual(len(result['bindings']),11)
        self.assertEqual(result['untouched_test_requests'],0)

    def test_unlaunched_or_running_phase_reserves_full_bound(self):
        stage = self.phase('codex-discovery-pending', V1, 'tf',10)
        result = b.account(self.root, V1)
        self.assertEqual(result['gpu_phase_wall_seconds'],1000)
        self.assertEqual(result['tf_requests'],10)
        (stage / 'control').mkdir()
        (stage / 'control/launch_intent.json').write_text(json.dumps({'run_token':stage.name.removesuffix('-stage'),
            'manifest_sha256':b.sha(stage / 'reviewed_manifest.json'), 'at':100.}))
        self.assertEqual(b.account(self.root,V1)['gpu_phase_wall_seconds'],1000)

    def test_unknown_mode_fails_closed(self):
        self.phase('codex-discovery-old', V1, 'unreviewed_mode')
        with self.assertRaisesRegex(RuntimeError, 'mode/request'): b.account(self.root,V1)

    def test_terminal_receipt_required_fields(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        path = stage / 'control/supervisor_exit.json'; original = path.read_text()
        for name in ('exit_code_kind','exit_status','service_result','invocation_id','producer_summary_present','failure_present','at'):
            with self.subTest(name=name):
                path.write_text(original); self.mutate(path,lambda d:d.pop(name))
                with self.assertRaises(RuntimeError): b.account(self.root,V1)
        path.write_text(original)

    def test_missing_service_start_does_not_reduce_reserved_time(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        (stage / 'control/service_started.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'complete launch'): b.account(self.root,V1)

    def test_receipt_identity_change_fails_closed(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        self.mutate(stage / 'control/supervisor_exit.json',lambda d:d.update(manifest_sha256='changed'))
        with self.assertRaisesRegex(RuntimeError,'identity differs'): b.account(self.root,V1)

    def test_invocation_mismatch_fails_closed(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        self.mutate(stage / 'control/supervisor_exit.json',lambda d:d.update(invocation_id='b'*32))
        with self.assertRaisesRegex(RuntimeError,'invocation identity'): b.account(self.root,V1)

    def test_invalid_terminal_status_semantics(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        path = stage / 'control/supervisor_exit.json'; original = path.read_text()
        cases = [{'service_result':'new-result'}, {'exit_status':None}, {'exit_status':'256'},
                 {'exit_status':'1'}, {'exit_code_kind':'killed','exit_status':'TERM'},
                 {'exit_code_kind':'killed','exit_status':'15','service_result':'timeout'}]
        for values in cases:
            with self.subTest(values=values):
                path.write_text(original); self.mutate(path,lambda d:d.update(values))
                with self.assertRaisesRegex(RuntimeError,'terminal|conflicts'): b.account(self.root,V1)

    def test_signal_timeout_receipt_counts_actual_and_all_requests(self):
        stage = self.phase('codex-discovery-old',V1,'generate',3,terminal=True)
        self.mutate(stage / 'control/supervisor_exit.json',lambda d:d.update(
            exit_code_kind='killed',exit_status='TERM',service_result='timeout',producer_summary_present=False,failure_present=True))
        result = b.account(self.root,V1)
        self.assertEqual((result['gpu_phase_wall_seconds'],result['generation_requests']),(25,3))

    def test_invalid_or_reversed_times_fail_closed(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        path = stage / 'control/supervisor_exit.json'; original = path.read_text()
        for value in (True,None,float('nan'),float('inf'),90,100):
            with self.subTest(value=value):
                path.write_text(original); self.mutate(path,lambda d:d.update(at=value))
                with self.assertRaisesRegex(RuntimeError,'wall-time'): b.account(self.root,V1)

    def test_changed_task_and_success_count_fail_closed(self):
        stage = self.phase('codex-discovery-old',V1,'generate',3)
        task = stage / 'tasks/gpu_0.json'
        self.mutate(task,lambda d:d['requests'].pop())
        with self.assertRaisesRegex(RuntimeError,'bound task/plan changed'): b.account(self.root,V1)
        path = stage / 'reviewed_manifest.json'
        self.mutate(path,lambda d:d['bound_files'][str(task)].update(sha256=b.sha(task),size_bytes=task.stat().st_size))
        with self.assertRaisesRegex(RuntimeError,'identity/count differs'): b.account(self.root,V1)

    def test_untouched_generation_and_tf_counted_across_versions(self):
        self.phase('codex-discovery-test-generation',V1,'generate',10,partition='untouched_test')
        self.phase('codex-discovery-test-tf',V2,'tf',3,partition='untouched_test')
        result = b.account(self.root,V2,V1)
        self.assertEqual((result['untouched_test_generation_requests'],result['untouched_test_tf_requests'],result['untouched_test_requests']),(10,3,13))
        self.assertEqual(sum(path.name == 'request_plan.json' for path in result['bindings']),2)

    def test_changed_or_wrong_partition_plan_fails_closed(self):
        stage = self.phase('codex-discovery-old',V1,'tf',3)
        path = stage / 'input/request_plan.json'; original=path.read_text()
        self.mutate(path,lambda d:d.update(evaluation_partition='direction_fit'))
        with self.assertRaisesRegex(RuntimeError,'bound task/plan changed'): b.account(self.root,V1)
        manifest = stage / 'reviewed_manifest.json'
        self.mutate(manifest,lambda d:d['bound_files'][str(path)].update(sha256=b.sha(path),size_bytes=path.stat().st_size))
        with self.assertRaisesRegex(RuntimeError,'partition plan is invalid'): b.account(self.root,V1)

    def test_changed_start_unit_rejected(self):
        stage = self.phase('codex-discovery-old',V1,'qualify',terminal=True)
        self.mutate(stage / 'control/service_started.json',lambda d:d.update(unit='another.service'))
        with self.assertRaisesRegex(RuntimeError,'service-start proof'): b.account(self.root,V1)

    def test_missing_worker_plan_coverage_fails_closed(self):
        stage = self.phase('codex-discovery-old',V1,'generate',3)
        path = stage / 'input/request_plan.json'
        self.mutate(path,lambda d:d['requests'].append({'request_id':'omitted','record_id':'row-omitted'}))
        self.mutate(stage / 'reviewed_manifest.json',lambda d:d['bound_files'][str(path)].update(
            sha256=b.sha(path),size_bytes=path.stat().st_size))
        with self.assertRaisesRegex(RuntimeError,'complete committed request plan'): b.account(self.root,V1)

    def test_duplicate_workers_cannot_double_commit_one_task(self):
        stage = self.phase('codex-discovery-old',V1,'generate',3)
        self.mutate(stage / 'reviewed_manifest.json',lambda d:d['workers'].append(dict(d['workers'][0])))
        with self.assertRaisesRegex(RuntimeError,'Duplicate prior task path'): b.account(self.root,V1)

    def test_bool_runtime_and_invalid_lineage_rejected(self):
        stage = self.phase('codex-discovery-old',V1,'qualify')
        self.mutate(stage / 'reviewed_manifest.json',lambda d:d['limits'].update(systemd_runtime_seconds=True))
        with self.assertRaisesRegex(RuntimeError,'runtime bound'): b.account(self.root,V1)
        with self.assertRaisesRegex(RuntimeError,'lineage digest'): b.account(self.root,'v1')

    def reallocation(self, old, destination):
        control = old / 'control'; control.mkdir()
        proof = control / 'prelaunch_reallocation.json'
        proof.write_text(json.dumps({'destination_stage': str(destination)}))
        return proof

    def helper(self):
        class Helper:
            @staticmethod
            def validate_pending(old_path, old, proof_path):
                # Tests below exercise accounting; helper policy has its own
                # independent authored/integration tests in its owned module.
                b.require({p.name for p in proof_path.parent.iterdir()} == {'prelaunch_reallocation.json'},
                          'Retired phase has launch/control evidence')
                receipt = json.loads(proof_path.read_text())
                return {'destination_stage': receipt['destination_stage'], 'bindings': [proof_path], 'receipt': receipt}

            @staticmethod
            def validate_pair(old_path, old, new_path, new, pending):
                b.require(old['scientific']['master_plan_sha256'] == new['scientific']['master_plan_sha256'],
                          'Replacement changed master')
                return [old_path, new_path]
        return patch.object(b, '_reallocation_helper', return_value=Helper)

    def claim(self, child, old, proof):
        self.mutate(child / 'reviewed_manifest.json', lambda value: value['scientific'].update(
            prelaunch_reallocation={'old_manifest_sha256': b.sha(old / 'reviewed_manifest.json'),
                                    'receipt_path': str(proof), 'receipt_sha256': b.sha(proof)}))

    def test_pending_retirement_reserves_globally_and_offsets_only_named_builder(self):
        old = self.phase('codex-discovery-old', V2, 'tf', 3)
        destination = self.root / 'codex-discovery-replacement-stage'
        proof = self.reallocation(old, destination)
        with self.helper():
            normal = b.account(self.root, V2)
            contextual = b.account(self.root, V2, replacement_stage=destination)
            self.assertEqual(normal['tf_requests'], 3)
            self.assertEqual(contextual['tf_requests'], 3)
            self.assertEqual(contextual['gpu_phase_wall_seconds'], 1000)
            self.assertEqual(contextual['replacement_context']['effective_previous_counts']['tf_requests'], 0)
            self.assertEqual(contextual['replacement_context']['effective_previous_gpu_phase_wall_seconds'], 0)
            self.assertIn(proof, contextual['bindings'])
            with self.assertRaisesRegex(RuntimeError, 'validated pending'):
                b.account(self.root, V2, replacement_stage=self.root / 'unrelated-stage')

    def test_exact_replacement_counted_once_with_old_history_and_all_bindings(self):
        old = self.phase('codex-discovery-old', V2, 'tf', 3)
        child = self.phase('codex-discovery-replacement', V2, 'tf', 3)
        proof = self.reallocation(old, child); self.claim(child, old, proof)
        self.phase('codex-discovery-unrelated', V2, 'tf', 3)
        with self.helper():
            result = b.account(self.root, V2)
            self.assertEqual(result['tf_requests'], 6)  # No generic same-ID dedup.
            self.assertEqual(result['gpu_phase_wall_seconds'], 2000)
            retired = next(p for p in result['phases'] if p['run_token'] == 'codex-discovery-old')
            self.assertEqual(retired['tf_requests'], 0)
            self.assertEqual(retired['gross_reserved_counts']['tf_requests'], 3)
            self.assertEqual(retired['gross_reserved_wall_seconds'], 1000)
            self.assertIn(proof, result['bindings'])
            self.assertIn(old / 'reviewed_manifest.json', result['bindings'])
            self.assertIn(child / 'reviewed_manifest.json', result['bindings'])
            with self.assertRaisesRegex(RuntimeError, 'without a manifest'):
                b.account(self.root, V2, replacement_stage=child)

    def test_retirement_never_refunds_phase_with_launch_evidence(self):
        old = self.phase('codex-discovery-old', V2, 'tf', 3)
        self.reallocation(old, self.root / 'codex-discovery-replacement-stage')
        (old / 'control/launch_result.json').write_text('{}')
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'launch/control'):
            b.account(self.root, V2)

    def test_multiple_parents_chains_and_cycles_rejected(self):
        a = self.phase('codex-discovery-a', V2, 'tf', 3)
        z = self.phase('codex-discovery-z', V2, 'tf', 3)
        destination = self.root / 'codex-discovery-new-stage'
        self.reallocation(a, destination); proof = self.reallocation(z, destination)
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'Multiple retired'):
            b.account(self.root, V2)
        proof.write_text(json.dumps({'destination_stage': str(a)}))
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'chains/cycles'):
            b.account(self.root, V2)

    def test_unrecognized_or_changed_claim_and_count_mismatch_rejected(self):
        old = self.phase('codex-discovery-old', V2, 'tf', 3)
        child = self.phase('codex-discovery-replacement', V2, 'tf', 4)
        proof = self.reallocation(old, child)
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'exact retirement identity'):
            b.account(self.root, V2)
        self.claim(child, old, proof)
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'counts/modes differ'):
            b.account(self.root, V2)
        other = self.phase('codex-discovery-0other', V2, 'tf', 1)
        self.mutate(other / 'reviewed_manifest.json', lambda value: value['scientific'].update(prelaunch_reallocation={}))
        proof.unlink()  # Orphaned claim cannot discount anything.
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'Unrecognized'):
            b.account(self.root, V2)

    def test_generation_test_and_cross_master_transfers_forbidden(self):
        old = self.phase('codex-discovery-old', V2, 'generate', 3)
        destination = self.root / 'codex-discovery-replacement-stage'
        self.reallocation(old, destination)
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'only validation TF'):
            b.account(self.root, V2, replacement_stage=destination)
        test_old = self.phase('codex-discovery-test-old', V2, 'tf', 3, partition='untouched_test')
        test_destination = self.root / 'codex-discovery-test-replacement-stage'
        self.reallocation(test_old, test_destination)
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'only validation TF'):
            b.account(self.root, V2, replacement_stage=test_destination)
        prior = self.phase('codex-discovery-v1-old', V1, 'tf', 3)
        child = self.phase('codex-discovery-v2-replacement', V2, 'tf', 3)
        proof = self.reallocation(prior, child); self.claim(child, prior, proof)
        with self.helper(), self.assertRaisesRegex(RuntimeError, 'changed master'):
            b.account(self.root, V2, V1)

    def test_historical_proof_binding_is_not_a_replacement_claim(self):
        old = self.phase('codex-discovery-old', V2, 'tf', 3)
        child = self.phase('codex-discovery-replacement', V2, 'tf', 3)
        proof = self.reallocation(old, child); self.claim(child, old, proof)
        later = self.phase('codex-discovery-later', V2, 'tf', 2)
        self.mutate(later / 'reviewed_manifest.json', lambda value: value['bound_files'].update(
            {str(proof): {'sha256': b.sha(proof), 'size_bytes': proof.stat().st_size}}))
        with self.helper():
            self.assertEqual(b.account(self.root, V2)['tf_requests'], 5)

    def test_real_helper_retirement_to_committed_child_integration(self):
        from test_prelaunch_reallocation import ReallocationTests
        case = ReallocationTests('runTest'); case.setUp(); self.addCleanup(case.doCleanups)
        case.retire()  # Only host/unit/process probes mocked; real immutable files/barrier.
        pending = b.account(case.root, 'a' * 64, replacement_stage=case.newstage)
        self.assertEqual(pending['tf_requests'], 3)
        self.assertEqual(pending['replacement_context']['effective_previous_counts']['tf_requests'], 0)
        child, _ = case.pair()
        case.write(case.newstage / 'reviewed_manifest.json', child)
        committed = b.account(case.root, 'a' * 64)
        self.assertEqual(committed['tf_requests'], 3)
        self.assertEqual(committed['gpu_phase_wall_seconds'], 1980)
        self.assertIn(case.oldstage / 'control/prelaunch_reallocation.json', committed['bindings'])
        # A launched replacement may later create its own output/runtime; the
        # OLD phase remains barred and cannot regain a second reservation.
        Path(child['output']).mkdir(); Path(child['runtime']).mkdir()
        self.assertEqual(b.account(case.root, 'a' * 64)['tf_requests'], 3)

class TestBundleLedgerTests(unittest.TestCase):
    def setUp(self):
        from infra.gpu03.direction_discovery import test_eval_run as fixtures
        self.f = fixtures.TestBundlePackageTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)

    def terminal(self, ref, *, success=True):
        m = self.f.manifests[ref['manifest']]; stage = Path(m['stage']); token = m['run_token']
        identity = {'run_token': token, 'manifest_sha256': ref['manifest_sha256']}
        self.f.fixture.save(stage / 'control/launch_intent.json', {**identity, 'at': 100.})
        self.f.fixture.save(stage / 'control/service_started.json', {'at': 101., 'unit': token + '.service',
            'fields': {'ActiveState': 'active', 'SubState': 'running', 'MainPID': '12', 'KillMode': 'control-group',
                       'ControlGroup': '/user.slice/' + token + '.service', 'InvocationID': 'a' * 32}})
        self.f.fixture.save(stage / 'control/supervisor_exit.json', {**identity, 'at': 200.,
            'service_result': 'success' if success else 'exit-code', 'exit_code_kind': 'exited',
            'exit_status': '0' if success else '1', 'invocation_id': 'a' * 32,
            'producer_summary_present': success, 'failure_present': not success})

    def account(self):
        return b.account(self.f.root, self.f.full['master_plan_sha256'])

    def test_all_fixed_parts_count_once_and_export_exact_metadata(self):
        self.f.complete()
        for ref in self.f.packages: self.terminal(ref)
        ledger = self.account()
        self.assertEqual((ledger['generation_requests'], ledger['untouched_test_requests'], ledger['tf_requests']), (15, 15, 0))
        self.assertEqual(ledger['gpu_phase_wall_seconds'], 300.)
        self.assertEqual([p['test_execution']['partition_index'] for p in ledger['phases']], [0, 1, 2])
        self.assertTrue(all(p['test_execution']['bundle'] == self.f.reference for p in ledger['phases']))
        self.assertIn(self.f.prepared, ledger['bindings'])  # JSONL was hashed, never JSON-parsed.

    def test_failed_or_unused_current_part_remains_fully_spent(self):
        ref = self.f.package(self.f.part, 'r0')
        ledger = self.account()
        self.assertEqual((ledger['generation_requests'], ledger['untouched_test_requests'], ledger['gpu_phase_wall_seconds']), (5, 5, 7200.))
        self.terminal(ref, success=False)
        ledger = self.account()
        self.assertEqual((ledger['generation_requests'], ledger['untouched_test_requests'], ledger['gpu_phase_wall_seconds']), (5, 5, 100.))

    def test_unreleased_or_failed_predecessor_cannot_have_continuation(self):
        self.f.complete()
        with self.assertRaisesRegex(RuntimeError, 'terminal predecessor'): self.account()
        self.terminal(self.f.packages[0], success=False)
        with self.assertRaisesRegex(RuntimeError, 'unsuccessful predecessor'): self.account()

    def test_replay_reservation_is_rejected_even_without_outputs(self):
        self.f.package(self.f.part, 'r0'); self.f.package(self.f.part, 'duplicate')
        with self.assertRaisesRegex(RuntimeError, 'duplicated, skipped'): self.account()

    def test_missing_payload_binding_fails_closed(self):
        ref = self.f.package(self.f.part, 'r0'); m = self.f.manifests[ref['manifest']]
        del m['bound_files'][str(self.f.prepared)]
        self.f.refreeze(Path(ref['manifest']), m)
        with self.assertRaisesRegex(RuntimeError, 'bound task/plan changed'): self.account()

    def test_altered_manifest_bundle_claim_fails(self):
        ref = self.f.package(self.f.part, 'r0'); m = self.f.manifests[ref['manifest']]
        m['scientific']['test_bundle'] = {'path': '/different', 'sha256': '0' * 64}
        self.f.refreeze(Path(ref['manifest']), m)
        with self.assertRaisesRegex(RuntimeError, 'frozen bundle'): self.account()

    def test_other_prior_test_phase_cannot_mix_with_bundle(self):
        self.f.package(self.f.part, 'r0')
        case = BudgetTests(); case.root = self.f.root
        case.phase('codex-discovery-other-test', self.f.full['master_plan_sha256'], 'tf', 1, partition='untouched_test')
        with self.assertRaisesRegex(RuntimeError, 'another prior test phase'): self.account()


if __name__ == '__main__': unittest.main()
