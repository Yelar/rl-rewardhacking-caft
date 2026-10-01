"""Offline actual-journal tests for bounded wave admission; no external calls."""
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import bind_causal_wave as b
from . import prepare_causal_waves as planner
from . import test_build_causal_plan as fixtures
from .test_build_causal_eval_plan import Fixture as RetainedFixture, write, rewrite


class WaveFixture:
    """Prepared full 15,000-request collection; produce only inert journal rows."""
    def __init__(self, test):
        self.parent = tempfile.TemporaryDirectory(); test.addCleanup(self.parent.cleanup)
        self.root = Path(self.parent.name).resolve() / 'codex-authored-wave-fixture'; self.root.mkdir()
        fake = types.SimpleNamespace(name=str(self.root), cleanup=lambda: None)
        with patch.object(fixtures.tempfile, 'TemporaryDirectory', return_value=fake):
            fixtures.TaskTests.setUp(self)
        self.plan['run_token'] = self.root.name
        source = self.root / 'source-causal-v2'; source.mkdir()
        self.plan['source_root'] = str(source)
        project = Path(b.__file__).resolve().parents[3]
        names = set(planner.original.REQUIRED_SOURCE) | set(b.worker.REQUIRED_SOURCE_FILES)
        names |= {'infra/gpu03/task_specific_discovery/' + n for n in
                  ('causal_qualification.py', 'causal_prompts.py', 'fit_worker.py', 'io_contract.py')}
        for name in names:
            path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project / name).read_bytes()); path.chmod(0o444)
        self.plan['qualified_source_files'] = {name: b.io.binding(source / name) for name in sorted(names)}
        self.plan['qualification_contract']['source_files'] = self.plan['qualified_source_files']
        write(source / 'SOURCE_FILES.json', self.plan['qualified_source_files'])
        self.plan_ref = write(self.root / 'FROZEN_PLAN.json', self.plan)
        self.devices = {str(i): f'GPU-0000000{i}-0000-0000-0000-000000000000' for i in range(8)}
        qualification_manifest = write(self.root / 'QUALIFICATION_MANIFEST.json', {
            'schema': b.guard.SCHEMA, 'host': 'gpu-03', 'uid': 3357, 'run_token': self.root.name,
            'source_root': str(source), 'science': {'stage': 'qualification'}, 'gpu_uuids': self.devices})
        self.qref = fixtures.TaskTests.proof(self, lambda x: x.update(run_token=self.root.name, manifest=qualification_manifest))
        self.collection_ref = planner.prepare(self.plan_ref, self.qref, self.root / 'pilot-waves-v1')
        self.collection = planner.original.read_json(self.collection_ref['path'])
        self.inventory = '\n'.join(f'{i}, {uuid}, NVIDIA RTX 5000 Ada Generation, 32768, 0, 0'
                                   for i, uuid in self.devices.items())

    def addCleanup(self, _):
        pass  # The owning unittest owns the single parent temporary directory.

    def contexts(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(b, 'ROOT', str(self.root)))
        stack.enter_context(patch.object(b.guard, 'RUN_ROOT', self.root.parent))
        return stack

    def bind(self, index, now=100000., **kwargs):
        defaults = dict(inventory=self.inventory, processes='', observed_at=now, now=now, root=str(self.root))
        defaults.update(kwargs)
        with self.contexts():
            return b.bind(self.collection_ref, index, **defaults)

    def close(self, manifest_ref):
        m = planner.original.read_json(manifest_ref['path'])
        runtime = Path(m['runtime']); runtime.mkdir()
        launches = []
        for item in m['workers']:
            task = planner.original.read_json(item['argv'][5]); digest = item['argv'][7]
            RetainedFixture.make_worker(self, task, digest)
            out = Path(task['output']); started = planner.original.read_json(out / 'STARTED.json')
            started.update(gpu_uuid=m['gpu_uuids'][str(item['gpu_id'])], pid=123 + item['gpu_id'])
            rewrite(out / 'STARTED.json', started)
            launches.append({'name': item['name'], 'pid': started['pid'], 'gpu_id': item['gpu_id']})
        b.closed.write_rows(runtime / 'workers.jsonl', launches)
        release = {'run_token': m['run_token'], 'manifest_sha256': manifest_ref['sha256'],
            'status': 'workers_exited_zero', 'owned_processes_released': True, 'gpu_release_verified': True,
            'scientific_success_verified': False, 'exit_codes': {x['name']: 0 for x in m['workers']},
            'gpus': {gpu: {'uuid': uuid, 'pids': [], 'utilization': 0, 'total_mib': 32768, 'free_mib': 32768}
                     for gpu, uuid in m['gpu_uuids'].items()}}
        write(runtime / 'SUPERVISOR_RESULT.json', release)
        return m, release

    def audit(self, ref, index=0):
        with self.contexts():
            return b.audit_wave(ref, self.collection, self.collection['waves'][index], b.audit_io.Evidence())


class BinderTests(unittest.TestCase):
    def setUp(self):
        self.f = WaveFixture(self)

    def test_exact_base_hardlinks_bind_with_original_worker_and_retain_policy(self):
        from .test_prepare_causal_waves import WaveTests
        with WaveTests.linked_base_policy(self.f):
            ref = self.f.bind(0)
        manifest = planner.original.read_json(ref['path'])
        audit = planner.original.read_json(Path(ref['path']).parent / 'PREDECESSOR_AUDIT.json')
        self.assertEqual(len(audit['audit_io_provenance']['verified_linked_files']), 13)
        self.assertEqual(manifest['source_root'], self.f.plan['source_root'])
        self.assertTrue(all(item['argv'][3] == 'infra.gpu03.task_specific_discovery.causal_worker'
                            for item in manifest['workers']))
        self.assertEqual(manifest['science']['batch_size'], 8)

    def test_linked_qualification_proof_or_changed_policy_binding_reject_before_claim(self):
        import os
        alias = self.f.root / 'forbidden-proof-link'; os.link(self.f.qref['path'], alias)
        try:
            with self.assertRaisesRegex(ValueError, 'Linked evidence'):
                self.f.bind(0)
        finally:
            alias.unlink()
        changed = copy.deepcopy(self.f.collection)
        changed['audit_policy_helper']['sha256'] = '0' * 64
        self.f.collection_ref = write(self.f.root / 'pilot-waves-v1/BAD_COLLECTION.json', changed)
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.f.bind(0)
        self.assertFalse((self.f.root / 'pilot-waves-v1/GLOBAL_GENERATION_WINDOW.json').exists())

    def test_first_binding_exact_limits_devices_source_and_one_use(self):
        ref = self.f.bind(0); m = planner.original.read_json(ref['path'])
        self.assertEqual(m['unit'], self.f.root.name + '-causal-pilot-wave-00')
        self.assertEqual(m['deadline_epoch'], 114400.)
        self.assertEqual(m['runtime_seconds'], 14520)
        self.assertEqual(m['kernel_limits'], {'memory_max': 192 * 1024**3, 'tasks_max': 512, 'cpu_cores': 16})
        self.assertEqual(m['gpu_uuids'], self.f.devices)
        self.assertEqual(m['source_root'], self.f.plan['source_root'])
        self.assertEqual(m['wave_policy']['mandatory_guard_idle_qualification_seconds'], 60)
        self.assertEqual(len(m['workers']), 8)
        window = planner.original.read_json(m['global_generation_window']['path'])
        self.assertEqual(window['global_generation_cutoff_epoch'], 186400.)
        self.assertTrue(all(not Path(planner.original.read_json(x['argv'][5])['output']).exists() for x in m['workers']))
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            self.f.bind(0, now=100010.)

    def test_live_probe_runs_after_read_only_admission(self):
        called = []
        def probe():
            called.append(True)
            self.assertFalse((Path(self.f.collection_ref['path']).parent / 'GLOBAL_GENERATION_WINDOW.json').exists())
            return self.f.inventory, '', 100020.
        self.f.bind(0, now=100021., inventory=None, processes=None, observed_at=None, probe=probe)
        self.assertEqual(called, [True])

    def test_stale_foreign_and_missing_gpu_inventory_fail_before_claim(self):
        cases = [dict(observed_at=99969.), dict(processes=self.f.devices['2'] + ', 999\n'),
                 dict(inventory='\n'.join(self.f.inventory.splitlines()[:-1])),
                 dict(inventory=self.f.inventory.replace(self.f.devices['7'], 'GPU-00000009-0000-0000-0000-000000000000'))]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.f.bind(0, **case)
        self.assertFalse((Path(self.f.collection_ref['path']).parent / 'GLOBAL_GENERATION_WINDOW.json').exists())

    def test_missing_predecessor_and_ambiguous_output_block(self):
        with self.assertRaises(FileNotFoundError):
            self.f.bind(1)
        phase = planner.original.read_json(self.f.collection['waves'][0]['phase_inputs']['path'])
        task = planner.original.read_json(phase['tasks'][0]['task']['path'])
        Path(task['output']).mkdir()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.f.bind(0)

    def test_actual_closed_eight_worker_wave_allows_next_and_exact_retained_union(self):
        ref = self.f.bind(0); self.f.close(ref)
        proof, retained = self.f.audit(ref)
        self.assertEqual(proof['requests'], 1880)
        self.assertEqual(set(retained), set(self.f.collection['waves'][0]['request_ids']))
        self.assertTrue(proof['gpu_workers_released'])
        successor = self.f.bind(1, now=110000.)
        m = planner.original.read_json(successor['path'])
        self.assertEqual(m['predecessor_manifests'], [ref])
        self.assertEqual(m['global_generation_window'], proof['global_generation_window'])
        self.assertEqual(m['deadline_epoch'], 124400.)
        self.assertLess(Path(successor['path']).stat().st_size, 1 << 20)

    def test_failure_foreign_gpu_stale_release_and_partial_journal_rejected(self):
        ref = self.f.bind(0); m, release = self.f.close(ref)
        path = Path(m['runtime']) / 'SUPERVISOR_RESULT.json'
        for field, value in (('status', 'failed'), ('owned_processes_released', False),
                             ('manifest_sha256', '0' * 64)):
            wrong = copy.deepcopy(release); wrong[field] = value; rewrite(path, wrong)
            with self.subTest(field=field), self.assertRaises(ValueError): self.f.audit(ref)
        wrong = copy.deepcopy(release); wrong['gpus']['0']['pids'] = [999]; rewrite(path, wrong)
        with self.assertRaises(ValueError): self.f.audit(ref)
        rewrite(path, release)
        task = planner.original.read_json(m['workers'][0]['argv'][5])
        event = Path(task['output']) / 'journal/final_recorded.jsonl'; raw = event.read_bytes()
        event.write_bytes(b'\n'.join(raw.splitlines()[:-1]) + b'\n')
        with self.assertRaisesRegex(ValueError, 'ambiguous'): self.f.audit(ref)
        event.write_bytes(raw)
        write(Path(task['output']) / 'batch_9999.partial.json', {'ambiguous': True})
        with self.assertRaisesRegex(ValueError, 'partial'): self.f.audit(ref)

    def test_global_cutoff_clips_final_wave_and_cannot_extend(self):
        ref = self.f.bind(0); self.f.close(ref)
        successor = self.f.bind(1, now=186000.)
        m = planner.original.read_json(successor['path'])
        self.assertEqual(m['deadline_epoch'], 186400.)
        self.assertEqual(m['runtime_seconds'], 520)
        path = Path(m['global_generation_window']['path']); window = planner.original.read_json(path)
        window['global_generation_cutoff_epoch'] += 1; rewrite(path, window)
        with self.assertRaisesRegex(ValueError, 'extended'):
            self.f.bind(2, now=186001.)

    def test_expired_window_and_different_physical_gpu_block_next_wave(self):
        ref = self.f.bind(0); self.f.close(ref)
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            self.f.bind(1, now=186400.)
        for remaining in (59, 60):
            with self.subTest(remaining=remaining), self.assertRaisesRegex(ValueError, '60-second'):
                self.f.bind(1, now=186400. - remaining)
        changed = self.f.inventory.replace(self.f.devices['7'], 'GPU-00000009-0000-0000-0000-000000000000')
        with self.assertRaisesRegex(ValueError, 'Physical GPUs'):
            self.f.bind(1, now=110000., inventory=changed)
        folder = Path(self.f.collection['waves'][1]['phase_inputs']['path']).parent
        self.assertFalse((folder / 'RESOURCE_BINDING.json').exists())

    def test_changed_task_argv_or_hash_is_rejected(self):
        evidence = b.audit_io.Evidence()
        wave = copy.deepcopy(self.f.collection['waves'][0])
        phase = planner.original.read_json(wave['phase_inputs']['path'])
        phase['tasks'][0]['argv'][3] = 'unreviewed.module'
        wave['phase_inputs'] = write(self.f.root / 'bad-phase.json', phase)
        with self.assertRaisesRegex(ValueError, 'command'):
            b.wave_inputs(self.f.collection, wave, evidence)
        phase['tasks'][0]['argv'][3] = 'infra.gpu03.task_specific_discovery.causal_worker'
        phase['tasks'][0]['task']['sha256'] = '0' * 64
        wave['phase_inputs'] = write(self.f.root / 'bad-hash-phase.json', phase)
        with self.assertRaises(ValueError):
            b.wave_inputs(self.f.collection, wave, b.audit_io.Evidence())

    def test_crash_after_binding_claim_preserves_consumption(self):
        original = b.worker.write
        def fail(path, value):
            result = original(path, value)
            if Path(path).name == 'RESOURCE_BINDING.json':
                raise RuntimeError('injected post-claim crash')
            return result
        with patch.object(b.worker, 'write', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'post-claim'):
            self.f.bind(0)
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            self.f.bind(0, now=100010.)


if __name__ == '__main__':
    unittest.main()
