"""Authored metadata fixtures; no models, services, or scientific test records."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import test_execution as e, test_bundle, behavior_plan, supervisor


class TestExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.conditions = {c: {'role': 'baseline' if c == 'baseline' else 'target' if c == 'target' else 'random', 'layers': []}
                           for c in ('baseline', 'target', 'random1', 'random2', 'random3')}
        requests = [{'request_id': f'authored-{i}-{c}', 'record_id': f'record-{i}', 'problem_id': str(i),
                     'problem_split': 'untouched_test', 'condition_id': c, 'scope': 'primary' if i < 2 else 'local',
                     'sample_index': 0, 'seed': 6007 + i} for i in range(3) for c in self.conditions]
        self.full = {'master_plan_sha256': 'a' * 64, 'parent_plan_sha256': 'b' * 64,
                     'phase': 'untouched_test_bundle', 'evaluation_partition': 'untouched_test', 'mode': 'generate',
                     'conditions': self.conditions, 'requests': requests, 'source_bindings': {}, 'sampling': {'temperature': .7},
                     'no_training': True, 'frozen_final_config': str(self.root / 'final.json')}
        self.full_path = self.save(self.root / 'bundle/request_plan.json', self.full)
        self.bundle = {'generation_request_plan': {'path': 'request_plan.json', 'sha256': e.sha(self.full_path)},
                       'execution_partitions': [{'partition_index': i, 'request_ids': [r['request_id'] for r in requests[i*5:(i+1)*5]]}
                                                for i in range(3)]}
        self.bundle_path = self.save(self.root / 'bundle/bundle.json', self.bundle)
        self.save(self.root / 'bundle/artifact_manifest.json', {'authored': True})
        self.ref = {'path': str(self.bundle_path), 'sha256': e.sha(self.bundle_path)}
        # The production bundle verifier has its own real Cartesian/selection tests.
        self.verifier = patch.object(test_bundle, 'verify_bundle', return_value=self.bundle).start()
        self.addCleanup(patch.stopall)
        self.previous = {'generations': 2526, 'teacher_forced': 9222, 'untouched_test_generations': 0,
                         'untouched_test_requests': 0, 'phase_manifests': [], 'ledger_bindings': []}
        self.part = e.make_part(self.ref, 0, self.previous)

    def save(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f: f.write(e.canonical(obj))
        path.chmod(0o400)
        return path

    def ledger(self, phases=()):
        spent = sum(p['untouched_test_requests'] for p in phases)
        return {'generation_requests': 2526 + spent, 'tf_requests': 9222,
                'untouched_test_generation_requests': spent, 'untouched_test_requests': spent,
                'untouched_test_tf_requests': 0, 'phases': list(phases)}

    def previous_for(self, ledger):
        return {**self.previous, 'generations': ledger['generation_requests'],
                'untouched_test_generations': ledger['untouched_test_generation_requests'],
                'untouched_test_requests': ledger['untouched_test_requests'], 'phase_budget': ledger}

    def package(self, part, suffix='r0'):
        stage = self.root / ('codex-discovery-test-' + suffix + '-stage')
        token = stage.name.removesuffix('-stage'); output = stage / 'results'
        pp = self.save(stage / 'input/request_plan.json', part)
        task = {'mode': 'generate', 'run_token': token, 'worker_name': 'gpu_7',
                'requests': part['requests'], 'conditions': part['conditions']}
        tp = self.save(stage / 'tasks/gpu_7.json', task)
        artifact = self.save(output / 'artifact_manifest.json', {'authored': True})
        manifest = {'run_token': token, 'host': 'gpu-04', 'stage': str(stage), 'output': str(output),
                    'scientific': {'master_plan_sha256': part['master_plan_sha256'], 'training': False,
                                   'test_bundle': self.ref, 'test_execution': part['test_execution']},
                    'workers': [{'name': 'gpu_7', 'command': ['python', 'engine.py', '--task', str(tp)],
                                 'success_expect': {'mode': 'generate', 'requests': len(part['requests'])}}],
                    'bound_files': {str(p): {'sha256': e.sha(p), 'size_bytes': p.stat().st_size}
                                    for p in (pp, tp)}, 'limits': {'systemd_runtime_seconds': 7200}}
        mp = self.save(stage / 'reviewed_manifest.json', manifest)
        proof = {'status': 'verified', 'manifest_sha256': e.sha(mp), 'run_token': token, 'gpu_release_verified': True}
        vp = self.save(stage / 'independent_verification.json', proof)
        ref = {'manifest': str(mp), 'manifest_sha256': e.sha(mp), 'artifact_manifest_sha256': e.sha(artifact),
               'verification': str(vp), 'verification_sha256': e.sha(vp)}
        entry = {'run_token': token, 'manifest_sha256': e.sha(mp), 'test_execution': part['test_execution'],
                 'generation_requests': len(part['requests']), 'tf_requests': 0,
                 'untouched_test_generation_requests': len(part['requests']), 'untouched_test_requests': len(part['requests']),
                 'untouched_test_tf_requests': 0, 'generation_request_ids': [r['request_id'] for r in part['requests']],
                 'wall_basis': 'actual_launch_to_terminal_receipt'}
        return ref, proof, entry

    def second(self):
        ref, proof, entry = self.package(self.part)
        ledger = self.ledger([entry])
        part = e.make_part(self.ref, 1, self.previous_for(ledger), [ref])
        return part, ref, proof, entry, ledger

    def test_first_piece_is_exact_and_requires_zero_previous_test(self):
        e.validate_part(self.part); e.validate_against_ledger(self.part, self.ledger())
        self.assertEqual(self.part['requests'], self.full['requests'][:5])
        self.assertEqual(self.part['scope_counts'], {'primary': 5})
        with self.assertRaises(RuntimeError):
            e.validate_against_ledger(self.part, {**self.ledger(), 'untouched_test_requests': 1})

    def test_complete_three_piece_success_preserves_every_request(self):
        part1, ref0, proof0, entry0, ledger1 = self.second()
        e.validate_against_ledger(part1, ledger1)
        with patch.object(supervisor, 'verify', return_value=proof0): e.bindings(part1, verify=True)
        ref1, proof1, entry1 = self.package(part1, 'r1')
        ledger2 = self.ledger([entry0, entry1])
        part2 = e.make_part(self.ref, 2, self.previous_for(ledger2), [ref0, ref1])
        e.validate_against_ledger(part2, ledger2)
        with patch.object(supervisor, 'verify', side_effect=[proof0, proof1]) as check:
            paths = e.bindings(part2, verify=True); self.assertEqual(check.call_count, 2)
        self.assertIn(self.bundle_path, paths)
        self.assertEqual(self.part['requests'] + part1['requests'] + part2['requests'], self.full['requests'])
        self.assertEqual(part2['scope_counts'], {'local': 5})

    def test_reordered_missing_duplicate_or_modified_request_rejected(self):
        for kind in ('reverse', 'pop', 'duplicate', 'seed'):
            part = copy.deepcopy(self.part)
            if kind == 'reverse': part['requests'].reverse()
            elif kind == 'pop': part['requests'].pop()
            elif kind == 'duplicate': part['requests'][-1] = part['requests'][0]
            else: part['requests'][0]['seed'] += 1
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): e.validate_part(part)

    def test_scientific_and_unknown_fields_cannot_change(self):
        for key, val in (('sampling', {}), ('no_training', False), ('phase', 'new_test'), ('unreviewed', True)):
            part = copy.deepcopy(self.part); part[key] = val
            with self.subTest(key=key), self.assertRaises(RuntimeError): e.validate_part(part)

    def test_mutable_bundle_or_changed_full_plan_rejected(self):
        self.bundle_path.chmod(0o600)
        with self.assertRaises(RuntimeError): e.validate_part(self.part)
        self.bundle_path.chmod(0o400); self.full_path.chmod(0o600)
        self.full_path.write_text('{}'); self.full_path.chmod(0o400)
        with self.assertRaises(RuntimeError): e.validate_part(self.part)

    def test_all_four_ledger_counts_must_match(self):
        for key in ('generation_requests', 'tf_requests', 'untouched_test_generation_requests', 'untouched_test_requests'):
            ledger = self.ledger(); ledger[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'stale'): e.validate_against_ledger(self.part, ledger)

    def test_prior_nonbundle_test_and_failed_reservation_rejected(self):
        part, _, _, entry, ledger = self.second()
        for change in ({'test_execution': {}}, {'wall_basis': 'full_reserved_deadline'}, {'manifest_sha256': 'f'*64},
                       {'untouched_test_tf_requests': 1}):
            changed = {**ledger, 'phases': [{**entry, **change}]}
            with self.subTest(change=change), self.assertRaises(RuntimeError): e.validate_against_ledger(part, changed)

    def test_alternate_bundle_reference_and_skipped_piece_rejected(self):
        part, _, _, entry, ledger = self.second()
        different = copy.deepcopy(entry); different['test_execution']['bundle']['sha256'] = 'e'*64
        with self.assertRaises(RuntimeError): e.validate_against_ledger(part, {**ledger, 'phases': [different]})
        missing = {**ledger, 'phases': []}
        with self.assertRaises(RuntimeError): e.validate_against_ledger(part, missing)

    def test_cross_phase_id_replay_rejected(self):
        prior = {'untouched_test_requests': 0, 'generation_request_ids': [self.part['requests'][0]['request_id']]}
        with self.assertRaisesRegex(RuntimeError, 'already committed'):
            e.validate_against_ledger(self.part, {**self.ledger(), 'phases': [prior]})

    def test_capacity_is_required_for_full_remaining_suite(self):
        ledger = {**self.ledger(), 'generation_requests': 4090}
        previous = self.previous_for(ledger)
        part = e.make_part(self.ref, 0, previous)
        with self.assertRaisesRegex(RuntimeError, 'remaining complete'): e.validate_against_ledger(part, ledger)

    def test_missing_duplicate_predecessors_and_invalid_indices_rejected(self):
        for index, refs in ((1, []), (2, []), (True, []), (-1, []), (3, [])):
            with self.subTest(index=index), self.assertRaises(RuntimeError): e.make_part(self.ref, index, self.previous, refs)
        _, ref, _, _, _ = self.second()
        with self.assertRaisesRegex(RuntimeError, 'Repeated'): e.make_part(self.ref, 2, self.previous, [ref, ref])

    def test_fresh_verifier_failure_is_not_success(self):
        part, _, proof, _, _ = self.second()
        with patch.object(supervisor, 'verify', return_value={**proof, 'gpu_release_verified': False}):
            with self.assertRaisesRegex(RuntimeError, 'Fresh predecessor'): e.bindings(part, verify=True)

    def test_failed_rehashed_proof_is_rejected(self):
        part, ref, proof, _, _ = self.second()
        bad = self.save(self.root / 'failed-proof.json', {**proof, 'status': 'failed'})
        part['test_execution']['predecessors'][0] = {**ref, 'verification': str(bad), 'verification_sha256': e.sha(bad)}
        with self.assertRaisesRegex(RuntimeError, 'successful terminal'): e.bindings(part)

    def test_changed_task_hash_rejected(self):
        part, ref, _, _, _ = self.second()
        manifest = json.loads(Path(ref['manifest']).read_text())
        task = Path(manifest['workers'][0]['command'][3]); task.chmod(0o600)
        data = json.loads(task.read_text()); data['requests'].pop(); task.write_text(e.canonical(data)); task.chmod(0o400)
        with self.assertRaisesRegex(RuntimeError, 'hash changed'): e.bindings(part)

    def test_incomplete_condition_cells_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'complete matched'): e.part_counts(self.full['requests'][:4], self.conditions)

    def test_successful_writer_is_metadata_only_and_does_not_reserve(self):
        previous = self.previous_for(self.ledger())
        output = self.root / 'piece'
        with patch.object(behavior_plan, 'counts_from_phase_ledger', return_value=previous) as read:
            result = e.write_part(bundle=str(self.bundle_path), bundle_sha256=e.sha(self.bundle_path), partition_index=0,
                                  phase_root=self.root, output=output)
        self.assertEqual(read.call_count, 2); self.assertFalse(result['budget_reserved']); self.assertTrue(result['no_compute_launched'])
        self.assertEqual(json.loads((output/'request_plan.json').read_text())['requests'], self.full['requests'][:5])
        self.assertEqual((output/'request_plan.json').stat().st_mode & 0o222, 0)
        with self.assertRaises(RuntimeError):
            e.write_part(bundle=str(self.bundle_path), bundle_sha256=e.sha(self.bundle_path), partition_index=0,
                         phase_root=self.root, output=output)

    def test_writer_ledger_race_fails_before_output_exists(self):
        first = self.previous_for(self.ledger()); second = {**first, 'generations': 2527}
        output = self.root / 'stale'
        with patch.object(behavior_plan, 'counts_from_phase_ledger', side_effect=[first, second]):
            with self.assertRaisesRegex(RuntimeError, 'Ledger changed'):
                e.write_part(bundle=str(self.bundle_path), bundle_sha256=e.sha(self.bundle_path), partition_index=0,
                             phase_root=self.root, output=output)
        self.assertFalse(output.exists())


class RemoteTestPredecessorTests(unittest.TestCase):
    """Replica verifier is injected; real partition/ledger joins remain active."""
    def setUp(self):
        from infra.gpu03.direction_discovery import remote_generation
        self.f = TestExecutionTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.local, proof, self.entry = self.f.package(self.f.part)
        m = json.loads(Path(self.local['manifest']).read_text())
        task = json.loads(Path(m['workers'][0]['command'][3]).read_text())
        task['sampling'] = self.f.full['sampling']
        m.update(host='gpu-02', stage='/inaccessible-gpu02/stage', output='/inaccessible-gpu02/output')
        proof['host'] = 'gpu-02'
        m['workers'][0]['command'][3] = '/inaccessible-gpu02/task.json'
        self.ref = {'kind': 'remote_generation', 'replica': {'path': str(self.f.root/'replica.json'), 'sha256': 'd'*64}}
        self.context = {'manifest': m, 'manifest_sha256': self.local['manifest_sha256'], 'proof': proof,
            'request_plan': self.f.part, 'tasks': [{'worker': m['workers'][0], 'task': task, 'task_path': self.f.root/'mapped-task.json'}],
            'bindings': [self.f.bundle_path]}
        self.remote = patch.object(remote_generation, 'context', side_effect=lambda ref, **kw: copy.deepcopy(self.context)).start()
        self.addCleanup(self.remote.stop)
        self.ledger = self.f.ledger([self.entry])
        self.part = e.make_part(self.f.ref, 1, self.f.previous_for(self.ledger), [self.ref])

    def test_explicit_remote_predecessor_preserves_ledger_and_never_opens_remote_paths(self):
        e.bindings(self.part, verify=True); e.validate_against_ledger(self.part, self.ledger)
        self.assertEqual(e.predecessor_manifest_sha256(self.ref), self.entry['manifest_sha256'])
        self.assertTrue(e.predecessor_context(self.ref)['remote'])

    def test_gpu01_predecessor_preserves_exact_partition_and_ledger(self):
        self.context['manifest']['host'] = self.context['proof']['host'] = 'gpu-01'
        e.bindings(self.part, verify=True); e.validate_against_ledger(self.part, self.ledger)
        self.assertEqual(e.predecessor_context(self.ref)['manifest']['host'], 'gpu-01')

    def test_wrong_or_unsupported_remote_host_rejected(self):
        for host, proof_host in [('gpu-01', 'gpu-02'), ('gpu-02', 'gpu-01'), ('gpu-03', 'gpu-03')]:
            self.context['manifest']['host'] = host; self.context['proof']['host'] = proof_host
            with self.subTest(host=host, proof_host=proof_host), self.assertRaisesRegex(RuntimeError, 'successful terminal'):
                e.bindings(self.part, verify=True)

    def test_failed_remote_release_and_wrong_deployment_ledger_rejected(self):
        self.context['proof']['gpu_release_verified'] = False
        with self.assertRaisesRegex(RuntimeError, 'successful terminal'): e.bindings(self.part)
        self.context['proof']['gpu_release_verified'] = True
        ledger = copy.deepcopy(self.ledger); ledger['phases'][0]['manifest_sha256'] = 'f'*64
        with self.assertRaisesRegex(RuntimeError, 'terminal exact'): e.validate_against_ledger(self.part, ledger)

    def test_remote_task_sampling_coverage_and_bundle_drift_rejected(self):
        original = copy.deepcopy(self.context)
        for kind in ('sampling', 'request', 'bundle'):
            self.context = copy.deepcopy(original)
            if kind == 'sampling': self.context['tasks'][0]['task']['sampling'] = {}
            elif kind == 'request': self.context['tasks'][0]['task']['requests'].pop()
            else: self.context['manifest']['scientific']['test_bundle'] = {}
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): e.bindings(self.part)

    def test_positive_bundle_gate_precedes_remote_resolution(self):
        with patch.object(test_bundle, 'verify_bundle', side_effect=RuntimeError('Negative final freeze')):
            with self.assertRaisesRegex(RuntimeError, 'Negative final freeze'): e.bindings(self.part)
        self.remote.assert_not_called()

    def test_alternate_replicas_cannot_repeat_one_physical_deployment(self):
        second = {'kind': 'remote_generation', 'replica': {'path': str(self.f.root/'replica2.json'), 'sha256': 'e'*64}}
        part = e.make_part(self.f.ref, 2, self.f.previous_for(self.ledger), [self.ref, second])
        with self.assertRaisesRegex(RuntimeError, 'Repeated physical'): e.bindings(part)

    def test_malformed_remote_reference_fails_without_reading_replica(self):
        for ref in ({**self.ref, 'manifest': 'pretend-local'}, {'kind': 'remote_generation', 'replica': {'path': 'relative', 'sha256': 'bad'}}):
            with self.subTest(ref=ref), self.assertRaises(RuntimeError):
                e.make_part(self.f.ref, 1, self.f.previous_for(self.ledger), [ref])
        self.remote.assert_not_called()


class RealBundleIntegrationTests(unittest.TestCase):
    def test_direct_script_imports_from_foreign_cwd_without_pythonpath(self):
        source = str(Path(e.__file__).resolve())
        environment = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        script = ('import runpy,sys; m=runpy.run_path(sys.argv[1]); '
                  'b=m["helper"]("test_bundle"); assert b.__package__=="infra.gpu03.direction_discovery"')
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, '-B', '-c', script, source], cwd=cwd, env=environment,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_authored_bundle_producer_to_execution_plan_without_source_bindings(self):
        from infra.gpu03.direction_discovery.test_test_bundle import Fixture
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            with patch.object(test_bundle, 'MASTER_SHA', fixture.master_sha), patch.object(behavior_plan, 'validate_master'):
                result = fixture.run()
                bundle = {'path': result['bundle'], 'sha256': result['bundle_sha256']}
                produced = json.loads((Path(result['bundle']).parent / 'request_plan.json').read_text())
                self.assertNotIn('source_bindings', produced)
                part = e.make_part(bundle, 0, fixture.previous)
                full, meta = e.validate_part(part)
                self.assertEqual(full, produced)
                self.assertEqual(part['requests'], produced['requests'][:300])
                self.assertEqual(meta['part_request_count'], 300)
                self.assertEqual(part['source_bindings'][result['bundle']], result['bundle_sha256'])
                self.assertIn(Path(result['bundle']).parent / 'prepared_records.jsonl', e.bindings(part))


if __name__ == '__main__': unittest.main()
