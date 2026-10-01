"""Inert full-size retained-wave fixtures; no GPU, model, scorer or network."""
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import build_causal_wave_eval_plan as w
from . import bind_causal_wave as wave_binding
from . import prepare_causal_waves as planning
from . import test_prepare_causal_waves as planning_tests
from . import test_build_causal_eval_plan as original_tests
from . import causal_eval_lane as lane
from .io_contract import ContractError

write, rewrite = original_tests.write, original_tests.rewrite


class FullFixture:
    def __init__(self):
        self.base = planning_tests.WaveTests('test_full_coverage_original_worker_api_and_batch_lane_order')
        self.base.setUp()
        self.root, self.plan, self.rows = self.base.root, self.base.plan, self.base.rows
        self.plan['run_token'] = 'codex-authored-wave-test'
        source = Path(self.plan['source_root']); project = Path(w.__file__).resolve().parents[3]
        for name in set(w.original.authored.REQUIRED_SOURCES) | {w.original.LANE_SOURCE}:
            path = source / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((project / name).read_bytes())
                path.chmod(0o400)
                self.plan['qualified_source_files'][name] = w.io.binding(path)
        self.plan['qualification_contract']['source_files'] = self.plan['qualified_source_files']
        rewrite(source / 'SOURCE_FILES.json', self.plan['qualified_source_files'])
        self.base.plan_ref = rewrite(self.base.plan_ref['path'], self.plan)
        self.source, self.plan_ref = Path(self.plan['source_root']), self.base.plan_ref
        self.adapters = {n: self.plan['bound_files'][str(Path(self.plan['checkpoint']) / n)]['sha256'] for n in w.worker.ADAPTERS}
        proto = SimpleNamespace(root=self.root, rows=self.rows)
        make_worker = lambda task, digest: original_tests.Fixture.make_worker(proto, task, digest)
        task = {key: copy.deepcopy(self.plan[key]) for key in ('run_token', 'model_revision', 'model_snapshot', 'checkpoint',
            'prepared_rows', 'vectors', 'sampling', 'batch_size', 'profile', 'source_root', 'qualified_source_files', 'conditions', 'bound_files')}
        task.update(schema='task_specific_causal_generation_v1', stage='qualification', requests=self.plan['qualification_requests'],
                    output=str(self.root / 'qualified_actual'))
        task_ref = write(self.root / 'qualified_task_actual.json', task)
        # Real RequestJournal paths/payload/event relations, with inert authored
        # completion strings. Disabling fixture fsync only speeds its creation;
        # this test does not claim to qualify kernel crash durability.
        with patch('os.fsync'):
            saved = make_worker(task, task_ref['sha256'])
        proof = w.plan_io.read_json(self.base.qref['path'])
        proof['run_token'] = self.plan['run_token']
        proof['source_files'] = self.plan['qualified_source_files']
        proof['retained_requests'] = saved
        self.qualified_ref = write(self.root / 'QUALIFIED_ACTUAL.json', proof)
        self.collection_ref = planning.prepare(self.plan_ref, self.qualified_ref, self.root / 'actual_waves')
        self.collection = w.plan_io.read_json(self.collection_ref['path'])
        self.window = {'schema': wave_binding.WINDOW_SCHEMA, 'run_token': self.plan['run_token'], 'collection': self.collection_ref,
            'overall_generation_seconds': 86400, 'started_at_epoch': 1000000., 'global_generation_cutoff_epoch': 1086400.}
        self.window_ref = write(self.root / 'GLOBAL_GENERATION_WINDOW.json', self.window)
        self.devices = {str(i): f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(8)}
        self.manifest_refs = []
        for index, wave in enumerate(self.collection['waves']):
            phase = w.plan_io.read_json(wave['phase_inputs']['path'])
            runtime = self.root / f'codex-authored-wave-test-causal-pilot-wave-{index:02d}-runtime'; runtime.mkdir()
            workers, launches = [], []
            for item in phase['tasks']:
                task = w.plan_io.read_json(item['task']['path']); gpu = item['lane']; pid = 1000 + index * 8 + gpu
                with patch('os.fsync'):
                    make_worker(task, item['task']['sha256'])
                rewrite(Path(task['output']) / 'STARTED.json', {'task_sha256': item['task']['sha256'], 'pid': pid,
                    'requests': len(task['requests']), 'gpu_uuid': self.devices[str(gpu)], 'stage': 'pilot'})
                workers.append({'name': f'gpu_{gpu}', 'gpu_id': gpu, 'cwd': str(self.source),
                    'cpu_ids': [16 + 2 * gpu, 17 + 2 * gpu],
                    'env': {'PYTHONPATH': str(self.source), 'CUDA_VISIBLE_DEVICES': self.devices[str(gpu)],
                            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                            'TOKENIZERS_PARALLELISM': 'false', 'PYTHONNOUSERSITE': '1'}, 'argv': item['argv']})
                launches.append({'name': f'gpu_{gpu}', 'pid': pid, 'gpu_id': gpu})
            now = 1000000. + index * 200
            manifest = {'schema': 'gpu03_task_specific_resource_guard_v1', 'host': 'gpu-03', 'uid': 3357,
                'run_token': self.plan['run_token'], 'source_root': str(self.source), 'runtime': str(runtime),
                'unit': f'codex-authored-wave-test-causal-pilot-wave-{index:02d}', 'disk_path': str(self.root),
                'science': {'stage': 'pilot'}, 'collection': self.collection_ref, 'wave_index': index, 'wave_count': 8,
                'phase_inputs': wave['phase_inputs'], 'global_generation_window': self.window_ref,
                'predecessor_manifests': list(self.manifest_refs), 'gpu_uuids': self.devices, 'workers': workers,
                'bound_at_epoch': now, 'deadline_epoch': now + 14400, 'runtime_seconds': 14520,
                'kernel_limits': {'memory_max': 192 * 1024**3, 'tasks_max': 512, 'cpu_cores': 16},
                'bound_files': {**phase['bound_files'], self.collection_ref['path']: w.binding(self.collection_ref),
                                self.window_ref['path']: w.binding(self.window_ref)}}
            ref = write(self.root / f'WAVE_{index:02d}_RESOURCE_MANIFEST.json', manifest)
            self.manifest_refs.append(ref)
            release = {'run_token': self.plan['run_token'], 'manifest_sha256': ref['sha256'], 'status': 'workers_exited_zero',
                'owned_processes_released': True, 'gpu_release_verified': True, 'scientific_success_verified': False,
                'exit_codes': {item['name']: 0 for item in workers},
                'gpus': {key: {'uuid': uuid, 'pids': [], 'utilization': 0, 'total_mib': 32768, 'free_mib': 32768}
                         for key, uuid in self.devices.items()}}
            write(runtime / 'SUPERVISOR_RESULT.json', release)
            w.original.write_rows(runtime / 'workers.jsonl', launches)

    def audit(self, collection_ref=None, manifests=None, evidence=None):
        with patch.object(w.worker, 'ADAPTERS', self.adapters), patch.object(wave_binding.guard, 'RUN_ROOT', self.root), \
                patch.object(wave_binding, 'ROOT', str(self.root / self.plan['run_token'])):
            return w.audit_waves(collection_ref or self.collection_ref,
                self.manifest_refs if manifests is None else manifests,
                w.audit_io.Evidence() if evidence is None else evidence)

    def close(self):
        self.base.doCleanups()


class WaveUnionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = FullFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()

    def setUp(self):
        self.f = self.fixture
        self.root = self.f.root

    def change_file(self, path, change):
        path = Path(path); before = path.read_bytes(); mode = path.stat().st_mode & 0o777
        def restore():
            path.chmod(0o600); path.write_bytes(before); path.chmod(mode)
        self.addCleanup(restore)
        value = w.plan_io.read_json(path); change(value)
        return rewrite(path, value)

    def test_full_eight_actual_wave_union_15000_plus_168_and_eight_cpu_lanes(self):
        plan, prepared, retained, audit = self.f.audit()
        self.assertEqual((audit['requests'], audit['new_requests'], audit['reused_requests']), (15168, 15000, 168))
        self.assertEqual(len(audit['actual_wave_closures']), 8)
        self.assertTrue(all(len(p['workers']) == 8 for p in audit['actual_wave_closures']))
        self.assertEqual(sum(r['origin'] == 'qualification' for r in retained.values()), 168)
        self.assertEqual(len({r['generation_file']['path'] for r in retained.values()}), 65)
        self.assertFalse(audit['synthetic_combined_supervisor'])
        out = self.root / 'full_shards'; out.mkdir()
        tasks, shards = w.assemble(plan, prepared, retained, out, self.f.source, self.f.plan['qualified_source_files'])
        self.assertEqual(len(tasks), 8)
        self.assertEqual(sum(t['requests'] for t in tasks), 15168)
        self.assertEqual(len({rid for s in shards for rid in s['request_ids']}), 15168)
        self.assertTrue(all(0 < len(s['request_ids']) <= 256 for s in shards))
        self.assertTrue(all(s['generated']['size_bytes'] <= 64 << 20 for s in shards))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(lane.outer, 'SOURCE_ROOT', self.f.source):
            for item in tasks:
                task = w.plan_io.read_json(item['task']['path'])
                summary = lane.validate(task)
                self.assertEqual(sum(s['records'] for s in summary), item['requests'])

    def test_missing_duplicate_and_reordered_wave_refs_fail(self):
        for refs in (self.f.manifest_refs[:-1], self.f.manifest_refs[:-1] + self.f.manifest_refs[:1],
                     [self.f.manifest_refs[1], self.f.manifest_refs[0], *self.f.manifest_refs[2:]]):
            with self.subTest(refs=len(refs)), self.assertRaises(ContractError):
                self.f.audit(manifests=refs)

    def test_exact_thirteen_base_hardlinks_admit_same_full_retained_union(self):
        base = Path(self.f.plan['model_snapshot'])
        roster = {str(base / name): self.f.plan['bound_files'][str(base / name)] for name in w.worker.BASE_FILES}
        for index, path in enumerate(roster):
            original_mode = Path(path).stat().st_mode & 0o777
            self.addCleanup(Path(path).chmod, original_mode)
            Path(path).chmod(0o644)  # Exact observed metadata, independent of service UMask0077.
            alias = self.root / f'authored_base_hardlink_{index}'
            os.link(path, alias); self.addCleanup(alias.unlink)
        evidence = w.audit_io.Evidence()
        # Only this authored fixture replaces the exact production roster and
        # owner constants; production cannot supply an alternate roster at CLI.
        with patch.object(w.audit_io, 'BASE_ROOT', str(base)), \
                patch.object(w.audit_io, 'BASE_ROSTER', roster), \
                patch.object(w.audit_io, 'LINK_OWNER_UID', os.getuid()):
            _, _, retained, audit = self.f.audit(evidence=evidence)
            policy = evidence.audit_io_provenance()
        self.assertEqual(len(retained), 15168)
        self.assertEqual(audit['reused_requests'], 168)
        self.assertEqual(set(policy['verified_linked_files']), set(roster))
        self.assertEqual(len(policy['verified_linked_files']), 13)
        self.assertFalse(policy['filesystem_immutability_claimed'])
        self.assertEqual(self.f.collection['qualified_source_files'], self.f.plan['qualified_source_files'])

    def test_hardlinked_nonbase_input_stays_rejected_before_evaluation_claim(self):
        alias = self.root / 'authored_vector_hardlink'
        os.link(self.f.plan['vectors'], alias); self.addCleanup(alias.unlink)
        with self.assertRaisesRegex(ContractError, 'outside exact authorized base roster'):
            self.f.audit()
        self.assertFalse((self.root / 'PILOT_EVAL_TASK_BINDING.json').exists())

    def test_collection_unplanned_duplicate_or_regenerated_request_refused(self):
        for name, mutation in (
            ('unplanned', lambda d: d['waves'][0]['request_ids'].__setitem__(0, 'unplanned')),
            ('duplicate', lambda d: d['waves'][0]['request_ids'].__setitem__(0, d['waves'][0]['request_ids'][1])),
            ('qualified_replay', lambda d: d['waves'][0]['request_ids'].__setitem__(0, next(iter(d['reused_requests'])))),
            ('global_batch_replay', lambda d: d['waves'][0]['global_batch_indices'].__setitem__(0, 1))):
            value = copy.deepcopy(self.f.collection); mutation(value)
            ref = write(self.root / f'bad_collection_{name}.json', value)
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.f.audit(collection_ref=ref)

    def test_collection_source_profile_drift_refused(self):
        for name, mutation in (
            ('profile', lambda d: d['profile'].update(batch_size=4)),
            ('source', lambda d: d.update(source_root=str(self.root / 'other-source'))),
            ('sourcehash', lambda d: next(iter(d['qualified_source_files'].values())).update(sha256='0' * 64)),
            ('partition_source', lambda d: d['planner'].update(sha256='0' * 64))):
            value = copy.deepcopy(self.f.collection); mutation(value)
            ref = write(self.root / f'bad_collection_{name}.json', value)
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.f.audit(collection_ref=ref)

    def test_actual_failed_or_unreleased_first_wave_blocks_all_scoring(self):
        path = self.root / 'codex-authored-wave-test-causal-pilot-wave-00-runtime/SUPERVISOR_RESULT.json'
        for field, value in (('status', 'failed'), ('owned_processes_released', False), ('gpu_release_verified', False)):
            original = w.plan_io.read_json(path)
            bad = copy.deepcopy(original); bad[field] = value
            before = path.read_bytes()
            try:
                rewrite(path, bad)
                with self.subTest(field=field), self.assertRaises(ContractError):
                    self.f.audit()
            finally:
                path.chmod(0o600); path.write_bytes(before); path.chmod(0o400)

    def test_started_without_complete_final_journal_is_ambiguous_not_replayed(self):
        phase = w.plan_io.read_json(self.f.collection['waves'][0]['phase_inputs']['path'])
        task = w.plan_io.read_json(phase['tasks'][0]['task']['path'])
        path = Path(task['output']) / 'journal/final_recorded.jsonl'; before = path.read_bytes()
        try:
            path.chmod(0o600); path.write_bytes(b'\n'.join(before.splitlines()[:-1]) + b'\n'); path.chmod(0o400)
            with self.assertRaisesRegex(ContractError, 'ambiguous'):
                self.f.audit()
        finally:
            path.chmod(0o600); path.write_bytes(before); path.chmod(0o400)

    def test_same_request_new_export_bytes_cannot_hide_behind_new_checksum(self):
        phase = w.plan_io.read_json(self.f.collection['waves'][0]['phase_inputs']['path'])
        task = w.plan_io.read_json(phase['tasks'][0]['task']['path']); out = Path(task['output'])
        path, complete = out / 'generations.jsonl', out / 'COMPLETE.json'
        raw, done = path.read_bytes(), complete.read_bytes()
        try:
            rows = [w.qualification.decode(line) for line in raw.splitlines()]
            rows[0]['completion'] = 'altered retained text'
            path.chmod(0o600); path.write_bytes(b''.join(w.canonical(r) + b'\n' for r in rows)); path.chmod(0o400)
            payload = w.plan_io.read_json(complete); payload['artifacts']['generations.jsonl'] = w.io.binding(path)
            rewrite(complete, payload)
            with self.assertRaisesRegex(ContractError, 'immutable primitive'):
                self.f.audit()
        finally:
            for p, data in ((path, raw), (complete, done)):
                p.chmod(0o600); p.write_bytes(data); p.chmod(0o400)

    def test_wrong_actual_source_profile_or_global_window_is_not_admitted(self):
        for name, mutation in (
            ('source', lambda d: d.update(source_root=str(self.root / 'wrong'))),
            ('window', lambda d: d.update(deadline_epoch=d['deadline_epoch'] + 1)),
            ('predecessor', lambda d: d.update(predecessor_manifests=[self.f.manifest_refs[-1]]))):
            value = w.plan_io.read_json(self.f.manifest_refs[0]['path']); mutation(value)
            ref = write(self.root / f'bad_manifest_{name}.json', value)
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.f.audit(manifests=[ref, *self.f.manifest_refs[1:]])

    def test_shared_global_evaluation_claim_blocks_new_and_old_planner(self):
        claim = self.root / 'PILOT_EVAL_TASK_BINDING.json'
        write(claim, {'inert_prior_claim': True})
        self.addCleanup(lambda: claim.unlink())
        dummy = self.f.manifest_refs[0]
        with self.assertRaises(FileExistsError):
            w.build(self.f.collection_ref, self.f.manifest_refs, dummy, dummy, dummy, self.f.source, self.root / 'blocked')
        with self.assertRaises(FileExistsError):
            w.original.build(self.f.plan_ref, dummy, self.f.qualified_ref, dummy, dummy, dummy, self.f.source, self.root / 'old-blocked')
        self.assertFalse((self.root / 'blocked').exists())

    def test_success_path_build_and_crash_after_global_claim_never_replays(self):
        dummy = write(self.root / 'authored_mock_boundary.json', {'authored_helper_is_independently_tested': True})
        files = {**self.f.plan['qualified_source_files'], w.SELF_SOURCE: {'sha256': '1' * 64, 'size_bytes': 1},
                 w.WAVE_AUDITOR_SOURCE: {'sha256': '2' * 64, 'size_bytes': 1},
                 w.WAVE_PREPARER_SOURCE: {'sha256': '3' * 64, 'size_bytes': 1},
                 w.AUDIT_IO_SOURCE: w.io.binding(Path(w.audit_io.__file__).resolve())}
        # All eight actual wave audits are real. Only the unchanged six-fixture
        # authored admission boundary is injected; its full success/failure
        # integration suite lives in test_build_causal_eval_plan.
        with patch.object(w.worker, 'ADAPTERS', self.f.adapters), \
                patch.object(wave_binding.guard, 'RUN_ROOT', self.root), \
                patch.object(wave_binding, 'ROOT', str(self.root / self.f.plan['run_token'])), \
                patch.object(w.worker, 'write', wraps=w.worker.write) as durable_claim_writer, \
                patch.object(w.original, 'source_and_authored', return_value=(files, {'run_token': self.f.plan['run_token']}, dummy)):
            result = w.build(self.f.collection_ref, self.f.manifest_refs, dummy, dummy, dummy, self.f.source, self.root / 'built')
            phase = w.plan_io.read_json(result['path'])
            self.assertEqual(phase['schema'], w.original.SCHEMA)
            self.assertEqual(phase['limits']['lanes'], 8)
            self.assertFalse(phase['dispatch_authorized_by_this_file'])
            self.assertEqual(phase['new_evaluations'], 15168)
            audit = w.plan_io.read_json(phase['input_audit']['path'])
            self.assertIn('input_audit_policy', audit)
            self.assertEqual(self.f.collection['qualified_source_files'], self.f.plan['qualified_source_files'])
            claim = self.root / 'PILOT_EVAL_TASK_BINDING.json'
            self.assertEqual(durable_claim_writer.call_count, 1)
            self.assertEqual(durable_claim_writer.call_args.args[0], claim)
            with self.assertRaises(FileExistsError):
                w.build(self.f.collection_ref, self.f.manifest_refs, dummy, dummy, dummy, self.f.source, self.root / 'repeat')
            # Only authored temp fixtures are removed to test a separate crash.
            claim.unlink()
            self.addCleanup(lambda: claim.unlink(missing_ok=True))
            with patch.object(w, 'assemble', side_effect=RuntimeError('inert crash after claim')):
                with self.assertRaisesRegex(RuntimeError, 'after claim'):
                    w.build(self.f.collection_ref, self.f.manifest_refs, dummy, dummy, dummy, self.f.source, self.root / 'interrupted')
            with self.assertRaises(FileExistsError):
                w.build(self.f.collection_ref, self.f.manifest_refs, dummy, dummy, dummy, self.f.source, self.root / 'after-crash')
        self.assertFalse((self.root / 'repeat').exists())
        self.assertFalse((self.root / 'after-crash').exists())


if __name__ == '__main__':
    unittest.main()
