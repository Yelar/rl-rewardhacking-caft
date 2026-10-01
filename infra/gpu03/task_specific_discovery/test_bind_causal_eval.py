"""Inert CPU resource binding fixtures; no subprocess, scorer, model or GPU."""
import copy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from . import bind_causal_eval as b
from . import bind_causal_wave as wave_binding
from . import phase_entry
from . import test_build_causal_wave_eval_plan as fixtures
from .io_contract import ContractError

write, rewrite = fixtures.write, fixtures.rewrite


class CpuBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = fixtures.FullFixture()
        f = cls.f
        cls.source = f.root / 'source-eval-binder'; cls.source.mkdir()
        cls.files = dict(f.plan['qualified_source_files'])
        project = Path(b.__file__).resolve().parents[3]
        extra = {b.SELF_SOURCE, b.waves.SELF_SOURCE, b.waves.WAVE_AUDITOR_SOURCE,
                 b.waves.WAVE_PREPARER_SOURCE, b.waves.AUDIT_IO_SOURCE,
                 'infra/gpu03/task_specific_discovery/resource_guard.py',
                 'infra/gpu03/task_specific_discovery/phase_entry.py'}
        for name in set(cls.files) | extra:
            path = cls.source / name; path.parent.mkdir(parents=True, exist_ok=True)
            original = project / name if name in extra else f.source / name
            shutil.copyfile(original, path); path.chmod(0o444)
            cls.files[name] = b.io.binding(path)
        write(cls.source / 'SOURCE_FILES.json', cls.files)
        cls.authored = write(f.root / 'authored_admission_fixture.json',
                            {'inert_boundary': 'Actual authored admission has its own full six-fixture tests'})
        with cls.context():
            cls.phase_ref = b.waves.build(f.collection_ref, f.manifest_refs,
                cls.authored, cls.authored, cls.authored, cls.source, f.root / 'cpu-evaluation-plan')
        cls.phase = b.waves.plan_io.read_json(cls.phase_ref['path'])
        cls.phase_bytes = Path(cls.phase_ref['path']).read_bytes()
        cls.previous = b.waves.plan_io.read_json(cls.phase['input_audit']['path'])
        cls.core = (f.plan, {r['record_id']: r for r in f.rows},
                    cls.previous['retained_requests'], cls.previous)
        cls.claim = f.root / 'PILOT_EVAL_RESOURCE_BINDING.json'
        cls.resource = Path(cls.phase_ref['path']).parent / 'EVALUATION_RESOURCE_MANIFEST.json'

    @classmethod
    def tearDownClass(cls):
        cls.f.close()

    @classmethod
    @contextmanager
    def context(cls, *, cached_wave_audit=False):
        # Only the already tested, unchanged authored-sandbox boundary is
        # injected. Full-wave journals/closure/shards remain real in the main
        # integration; later resource-only failures can reuse that result.
        with patch.object(b, 'ROOT', str(cls.f.root)), patch.object(b, 'TOKEN', cls.f.plan['run_token']), \
                patch.object(b.guard, 'RUN_ROOT', cls.f.root), \
                patch.object(wave_binding, 'ROOT', str(cls.f.root / cls.f.plan['run_token'])), \
                patch.object(b.worker, 'ADAPTERS', cls.f.adapters), \
                patch.object(b.waves.original, 'source_and_authored',
                             return_value=(cls.files, {'run_token': cls.f.plan['run_token']}, cls.authored)):
            if cached_wave_audit:
                with patch.object(b.waves, 'audit_waves', return_value=cls.core):
                    yield
            else:
                yield

    def setUp(self):
        for path in (self.claim, self.resource):
            self.assertFalse(path.exists())
            self.addCleanup(path.unlink, missing_ok=True)

    def changed_phase(self, change):
        path = Path(self.phase_ref['path']); original = path.read_bytes()
        def restore():
            path.chmod(0o600); path.write_bytes(original); path.chmod(0o400)
        self.addCleanup(restore)
        value = copy.deepcopy(self.phase); change(value)
        return rewrite(path, value)

    def test_full_real_eight_wave_admission_to_cpu_manifest_and_kernel_limits(self):
        with self.context(), patch.object(b.worker, 'write', wraps=b.worker.write) as writes:
            result = b.bind(self.phase_ref, 15400., clock=lambda: 1000.)
            self.assertEqual(writes.call_count, 2)
            self.assertEqual(writes.call_args_list[0].args[0], self.claim)
            self.assertEqual(writes.call_args_list[1].args[0], self.resource)
            manifest = b.guard.load_manifest(result['path'], result['sha256'])
        self.assertEqual(manifest['gpu_uuids'], {})
        self.assertEqual(manifest['expected_total_requests'], 15168)
        self.assertEqual(manifest['kernel_limits'], {'memory_max': 32 << 30, 'cpu_cores': 16, 'tasks_max': 512})
        self.assertEqual(manifest['runtime_seconds'], 14520)
        self.assertEqual([w['cpu_ids'] for w in manifest['workers']], [[48+2*i, 49+2*i] for i in range(8)])
        self.assertTrue(all(w['gpu_id'] is None and w['env']['CUDA_VISIBLE_DEVICES'] == '' for w in manifest['workers']))
        self.assertTrue(all(w['env']['OMP_NUM_THREADS'] == w['env']['MKL_NUM_THREADS'] ==
                            w['env']['OPENBLAS_NUM_THREADS'] == '1' for w in manifest['workers']))
        self.assertTrue(all(w['argv'][3] == 'infra.gpu03.task_specific_discovery.causal_eval_lane' for w in manifest['workers']))
        self.assertFalse(manifest['science']['generation']); self.assertFalse(manifest['science']['training'])
        self.assertEqual(manifest['science']['evaluator_workers_per_lane'], 2)
        phase_entry.check_limits({'memory.max': str(32 << 30), 'memory.current': '1', 'memory.swap.max': '0',
                                  'pids.max': '512', 'cpu.max': '1600000 100000'}, manifest['kernel_limits'])
        with self.context(cached_wave_audit=True), self.assertRaisesRegex(ContractError, 'already consumed'):
            b.bind(self.phase_ref, 16000., clock=lambda: 1200.)

    def test_phase_schema_population_limit_or_gpu_claim_drift_rejected(self):
        changes = [lambda x: x.update(schema='other'), lambda x: x.update(requests=15167),
                   lambda x: x.update(generation_performed=True),
                   lambda x: x.update(problem_split='untouched_test'),
                   lambda x: x['limits'].update(evaluator_workers_per_lane=3)]
        for change in changes:
            with self.subTest(change=change):
                ref = self.changed_phase(change)
                with self.context(), self.assertRaises(ContractError):
                    b._phase(ref, b.audit_io.Evidence())
        self.assertFalse(self.claim.exists())

    def test_missing_claim_binding_cannot_be_unbound_hash_admitted(self):
        ref = self.changed_phase(lambda x: x['bound_files'].pop(str(self.f.root / 'PILOT_EVAL_TASK_BINDING.json')))
        with self.context(), self.assertRaisesRegex(ContractError, 'omitted or changed'):
            b._phase(ref, b.audit_io.Evidence())

    def test_missing_positive_generation_release_blocks_before_any_resource_write(self):
        path = Path(self.f.manifest_refs[0]['path']); manifest = b.waves.plan_io.read_json(path)
        release_path = Path(manifest['runtime']) / 'SUPERVISOR_RESULT.json'; original = release_path.read_bytes()
        self.addCleanup(lambda: (release_path.chmod(0o600), release_path.write_bytes(original), release_path.chmod(0o400)))
        release = b.waves.plan_io.read_json(release_path); release['owned_processes_released'] = False
        rewrite(release_path, release)
        with self.context(), self.assertRaisesRegex(ContractError, 'release'):
            b.bind(self.phase_ref, 15400., clock=lambda: 1000.)
        self.assertFalse(self.claim.exists()); self.assertFalse(self.resource.exists())

    def test_lane_task_command_and_existing_output_cannot_replay(self):
        for mutation in (lambda x: x['tasks'][0]['argv'].__setitem__(3, 'other.module'),
                         lambda x: x['tasks'][0].update(requests=1)):
            ref = self.changed_phase(mutation)
            with self.context(cached_wave_audit=True), self.assertRaisesRegex(ContractError, 'Lane'):
                b.admit(ref, b.audit_io.Evidence())
        path = Path(self.phase_ref['path']); path.chmod(0o600); path.write_bytes(self.phase_bytes); path.chmod(0o400)
        destination = Path(self.phase_ref['path']).parent / 'results/lane_00'
        destination.mkdir(); self.addCleanup(destination.rmdir)
        with self.context(cached_wave_audit=True), self.assertRaisesRegex(ContractError, 'already started'):
            b.admit(self.phase_ref, b.audit_io.Evidence())

    def test_symlinked_lane_parent_is_refused_before_resource_claim(self):
        parent = Path(self.phase_ref['path']).parent / 'results'
        saved = parent.with_name('authored_saved_results')
        alternate = self.f.root / 'authored_external_results'; alternate.mkdir()
        parent.rename(saved); parent.symlink_to(alternate, target_is_directory=True)
        def restore():
            parent.unlink(); saved.rename(parent); alternate.rmdir()
        self.addCleanup(restore)
        with self.context(cached_wave_audit=True), self.assertRaisesRegex(ContractError, 'already started'):
            b.admit(self.phase_ref, b.audit_io.Evidence())
        self.assertFalse(self.claim.exists())

    def test_resource_scope_exact_cpu_only_environment_and_absolute_deadline(self):
        with self.context(), patch.dict(os.environ, {'AWS_SECRET_ACCESS_KEY': 'inert-do-not-copy'}):
            m = b.resource_manifest(self.phase_ref, self.phase, self.files, self.phase['tasks'], 1300., 1000.)
            self.assertEqual(m['runtime_seconds'], 420)
            for worker in m['workers']:
                self.assertNotIn('AWS_SECRET_ACCESS_KEY', worker['env'])
            for bad in (True, float('inf'), float('nan'), 1060., 1000., 15400.01):
                with self.subTest(deadline=bad), self.assertRaises(ContractError):
                    b.resource_manifest(self.phase_ref, self.phase, self.files, self.phase['tasks'], bad, 1000.)

    def test_crash_after_claim_and_existing_service_markers_block_rebind(self):
        real = b.worker.write
        def crash(path, value):
            if Path(path) == self.resource:
                raise RuntimeError('authored write interruption after claim')
            return real(path, value)
        with self.context(cached_wave_audit=True), patch.object(b.worker, 'write', side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, 'after claim'):
                b.bind(self.phase_ref, 15400., clock=lambda: 1000.)
        self.assertTrue(self.claim.is_file()); self.assertFalse(self.resource.exists())
        with self.context(), self.assertRaisesRegex(ContractError, 'already consumed'):
            b.bind(self.phase_ref, 15400., clock=lambda: 1000.)

    def test_existing_dispatch_marker_blocks_before_new_claim(self):
        path = self.f.root / (self.f.plan['run_token'] + '-' + b.SUFFIX + '_DISPATCH_ATTEMPT.json')
        write(path, {'inert_started': True}); self.addCleanup(path.unlink)
        with self.context(cached_wave_audit=True), self.assertRaisesRegex(ContractError, 'already started'):
            b.bind(self.phase_ref, 15400., clock=lambda: 1000.)
        self.assertFalse(self.claim.exists())


if __name__ == '__main__':
    unittest.main()
