"""Inert successor resource/journal tests; never use GPU, SSH or model execution.

The approved-plan/numerical-admission dependency has a narrow authored fixture.
Its exact seed6008/proposal/source validator is tested by test_successor_plan;
this suite uses actual partition/preparation and actual retained journals for
successor resource limits, no-replay, complete predecessors and exact release.
"""
import copy
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from . import bind_successor_wave as b
from . import prepare_successor_waves as planner
from .test_bind_causal_wave import WaveFixture
from .test_build_causal_eval_plan import write, rewrite


class SuccessorFixture(WaveFixture):
    def __init__(self, test):
        super().__init__(test)
        self.old_root = self.root
        self.old_plan = self.plan
        self.root = self.old_root.parent / 'codex-authored-successor-fixture'
        self.root.mkdir()
        source = self.root / 'source-successor-v1'; source.mkdir()
        inventory = {}
        project = Path(b.__file__).resolve().parents[3]
        names = ('bind_successor_wave.py', 'prepare_successor_waves.py', 'successor_dispatch.py',
                 'successor_phase_entry.py', 'phase_entry.py', 'resource_guard.py', 'guard_failure_diagnostic.py')
        for basename in names:
            name = 'infra/gpu03/task_specific_discovery/' + basename
            dest = source / name; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((project / name).read_bytes()); dest.chmod(0o444)
            inventory[name] = b.io.binding(dest)
        source_ref = write(source / 'SOURCE_FILES.json', inventory)
        proposal_ref = write(self.root / 'PROPOSAL.json', {'authored_fixture': True, 'seed': 6008})
        authorization_ref = write(self.root / 'AUTHORIZATION.json', {'authored_fixture': True, 'proposal': proposal_ref})
        self.plan = copy.deepcopy(self.old_plan)
        self.plan['run_token'] = self.root.name
        self.plan['successor'] = {'schema': 'task_specific_approved_successor_v1',
            'original_plan': self.plan_ref, 'proposal': proposal_ref, 'authorization': authorization_ref,
            'numerical_admission': self.qref, 'operational_source_root': str(source),
            'operational_source_inventory': source_ref}
        # This inert resource fixture does not claim the production seed domain.
        # Distinct deterministic IDs/seeds only exercise zero-reuse partitions;
        # exact6008 derivation is covered by the successor-plan module tests.
        for request in self.plan['pilot_requests']:
            request['request_id'] = b.hash_value(['authored-successor-fixture', request['request_id']])
            request['seed'] = (request['seed'] + 1) % (2**63)
        self.plan['budgets']['reusable_pilot_requests'] = 0
        folder = self.root / 'causal-plan-v1'; folder.mkdir()
        self.plan_ref = write(folder / 'PLAN.json', self.plan)
        with self.contexts():
            self.collection_ref = planner.prepare(self.plan_ref, self.qref, self.root / 'pilot-waves-v1')
        self.collection = planner.original.read_json(self.collection_ref['path'])

    def admitted_plan(self, plan_ref, qualified_ref, evidence):
        """Dependency fixture; fail on any alteration of the authored plan."""
        plan = b.read(plan_ref, evidence)
        b.require(plan == self.plan and plan_ref == self.plan_ref and qualified_ref == self.qref,
                  'Authored admitted plan/proof changed')
        b.audit_io.configure_from_plan(evidence, plan)
        bound = dict(plan['bound_files'])
        bound[plan_ref['path']] = b.closed.binding(plan_ref)
        bound[qualified_ref['path']] = b.closed.binding(qualified_ref)
        for key in ('proposal', 'authorization', 'operational_source_inventory'):
            ref = plan['successor'][key]; bound[ref['path']] = b.closed.binding(ref)
        for path, value in bound.items(): evidence.bind(path, value)
        return plan, {}, bound

    def contexts(self):
        stack = ExitStack()
        stack.enter_context(patch.object(b, 'ROOT', str(self.root)))
        stack.enter_context(patch.object(b.successor, 'ROOT', str(self.root)))
        stack.enter_context(patch.object(b.successor, 'ORIGINAL_ROOT', str(self.old_root)))
        stack.enter_context(patch.object(b.guard, 'RUN_ROOT', self.root.parent))
        stack.enter_context(patch.object(planner, 'checked_plan', side_effect=self.admitted_plan))
        return stack

    def bind(self, index, now=100000., **kwargs):
        defaults = dict(inventory=self.inventory, processes='', observed_at=now, now=now, root=str(self.root))
        defaults.update(kwargs)
        with self.contexts():
            return b.bind(self.collection_ref, index, **defaults)

    def audit(self, ref, index=0):
        with self.contexts():
            return b.audit_wave(ref, self.collection, self.collection['waves'][index], b.audit_io.Evidence())


class SuccessorBinderTests(unittest.TestCase):
    def setUp(self): self.f = SuccessorFixture(self)

    def test_exact_new_limits_zero_reuse_source_separation_and_one_use(self):
        ref = self.f.bind(0); m = b.read(ref, b.audit_io.Evidence())
        self.assertEqual(m['unit'], self.f.root.name + '-causal-pilot-wave-00')
        self.assertEqual(m['deadline_epoch'], 114400.)
        self.assertEqual(m['runtime_seconds'], 14520)
        self.assertEqual(m['kernel_limits'], {'memory_max': 192 * 1024**3, 'tasks_max': 512, 'cpu_cores': 16})
        self.assertEqual(m['source_root'], str(self.f.root / 'source-successor-v1'))
        self.assertEqual(m['successor'], self.f.plan['successor'])
        self.assertEqual((m['expected_new_requests'], m['expected_reused_requests']), (1896, 0))
        self.assertTrue(all(w['cwd'] == self.f.plan['source_root'] == w['env']['PYTHONPATH'] for w in m['workers']))
        self.assertEqual(m['gpu_uuids'], self.f.devices)
        window = b.read(m['global_generation_window'], b.audit_io.Evidence())
        self.assertEqual(window['global_generation_cutoff_epoch'], 186400.)
        self.assertEqual(window['proposal'], self.f.plan['successor']['proposal'])
        self.assertEqual(window['authorization'], self.f.plan['successor']['authorization'])
        self.assertEqual(self.f.collection['expected_new_requests'], 15168)
        self.assertEqual(self.f.collection['reused_requests'], {})
        with self.assertRaisesRegex(ValueError, 'already consumed'): self.f.bind(0)

    def test_produced_manifest_passes_dispatch_with_real_operational_bindings(self):
        from . import successor_dispatch as dispatch
        ref = self.f.bind(0); manifest = b.read(ref, b.audit_io.Evidence())
        inventory = b.read(self.f.plan['successor']['operational_source_inventory'], b.audit_io.Evidence())
        for name, value in inventory.items():
            self.assertEqual(manifest['bound_files'][manifest['source_root'] + '/' + name], value)
        with self.f.contexts(), patch.object(dispatch, 'ROOT', self.f.root), \
                patch.object(dispatch, 'TOKEN', self.f.root.name), patch.object(dispatch, 'GPU_UUIDS', self.f.devices):
            self.assertIs(dispatch.validate_successor_manifest(manifest, str(self.f.root), now=100000.), manifest)
            changed = copy.deepcopy(manifest)
            del changed['bound_files'][manifest['source_root'] + '/infra/gpu03/task_specific_discovery/guard_failure_diagnostic.py']
            with self.assertRaisesRegex(ValueError, 'Missing successor module'):
                dispatch.validate_successor_manifest(changed, str(self.f.root), now=100000.)

    def test_foreign_devices_stale_snapshot_and_different_uuid_never_consume_window(self):
        cases = [dict(observed_at=99969.), dict(processes=self.f.devices['0'] + ', 985372\n'),
            dict(inventory='\n'.join(self.f.inventory.splitlines()[:-1])),
            dict(inventory=self.f.inventory.replace(self.f.devices['7'], 'GPU-00000009-0000-0000-0000-000000000000'))]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError): self.f.bind(0, **case)
        self.assertFalse((self.f.root / 'pilot-waves-v1/GLOBAL_GENERATION_WINDOW.json').exists())
        self.assertFalse((self.f.root / 'pilot-waves-v1/wave_00/RESOURCE_BINDING.json').exists())

    def test_probe_occurs_after_plan_audit_before_any_claim(self):
        called = []
        def probe():
            called.append(True)
            self.assertFalse((self.f.root / 'pilot-waves-v1/GLOBAL_GENERATION_WINDOW.json').exists())
            return self.f.inventory, '', 100020.
        self.f.bind(0, now=100021., inventory=None, processes=None, observed_at=None, probe=probe)
        self.assertEqual(called, [True])

    def test_actual_complete_eight_worker_wave_allows_next_and_retains_fresh_ids(self):
        ref = self.f.bind(0); self.f.close(ref)
        proof, retained = self.f.audit(ref)
        self.assertEqual(proof['requests'], 1896)
        self.assertEqual(set(retained), set(self.f.collection['waves'][0]['request_ids']))
        self.assertFalse(set(retained) & {r['request_id'] for r in self.f.old_plan['pilot_requests']})
        nxt = self.f.bind(1, now=110000.); m = b.read(nxt, b.audit_io.Evidence())
        self.assertEqual(m['predecessor_manifests'], [ref])
        self.assertEqual(m['global_generation_window'], proof['global_generation_window'])
        self.assertEqual(m['deadline_epoch'], 124400.)
        self.assertLess(Path(nxt['path']).stat().st_size, 1 << 20)

    def test_all_eight_real_authored_waves_audit_complete15168_union(self):
        ids, refs = set(), []
        for index in range(8):
            ref = self.f.bind(index, now=100000. + 1000 * index)
            self.f.close(ref); proof, retained = self.f.audit(ref, index)
            self.assertFalse(ids & retained.keys()); ids.update(retained); refs.append(ref)
            manifest = b.read(ref, b.audit_io.Evidence())
            self.assertEqual(manifest['predecessor_manifests'], refs[:-1])
        self.assertEqual(len(ids), 15168)
        self.assertEqual(ids, {r['request_id'] for r in self.f.plan['pilot_requests']})

    def test_missing_predecessor_and_ambiguous_existing_output_block(self):
        with self.assertRaises(FileNotFoundError): self.f.bind(1)
        phase = b.read(self.f.collection['waves'][0]['phase_inputs'], b.audit_io.Evidence())
        task = b.read(phase['tasks'][0]['task'], b.audit_io.Evidence())
        Path(task['output']).mkdir()
        with self.assertRaisesRegex(ValueError, 'ambiguous'): self.f.bind(0)
        self.assertFalse((self.f.root / 'pilot-waves-v1/GLOBAL_GENERATION_WINDOW.json').exists())

    def test_failed_wrongtoken_nonzero_or_unreleased_predecessor_rejects(self):
        ref = self.f.bind(0); m, release = self.f.close(ref)
        path = Path(m['runtime']) / 'SUPERVISOR_RESULT.json'
        cases = [('status', 'failed'), ('owned_processes_released', False), ('gpu_release_verified', False),
            ('run_token', self.f.old_root.name), ('manifest_sha256', '0' * 64),
            ('exit_codes', {w['name']: int(w['name'] == 'gpu_0') for w in m['workers']})]
        for field, value in cases:
            wrong = copy.deepcopy(release); wrong[field] = value; rewrite(path, wrong)
            with self.subTest(field=field), self.assertRaises(ValueError): self.f.audit(ref)
        wrong = copy.deepcopy(release); wrong['gpus']['0']['pids'] = [999]; rewrite(path, wrong)
        with self.assertRaises(ValueError): self.f.audit(ref)

    def test_predecessor_cannot_omit_operational_source_from_dispatch_bindings(self):
        ref = self.f.bind(0); manifest, release = self.f.close(ref)
        del manifest['bound_files'][manifest['source_root'] + '/infra/gpu03/task_specific_discovery/resource_guard.py']
        bad_ref = write(self.f.root / 'OMITTED_SOURCE_MANIFEST.json', manifest)
        release['manifest_sha256'] = bad_ref['sha256']
        rewrite(Path(manifest['runtime']) / 'SUPERVISOR_RESULT.json', release)
        with self.assertRaisesRegex(ValueError, 'omits or changes'):
            self.f.audit(bad_ref)

    def test_incomplete_journal_and_partial_payload_not_promoted(self):
        ref = self.f.bind(0); m, _ = self.f.close(ref)
        task = b.read({'path': m['workers'][0]['argv'][5], 'sha256': m['workers'][0]['argv'][7],
                      'size_bytes': Path(m['workers'][0]['argv'][5]).stat().st_size}, b.audit_io.Evidence())
        event = Path(task['output']) / 'journal/final_recorded.jsonl'; raw = event.read_bytes()
        event.write_bytes(b'\n'.join(raw.splitlines()[:-1]) + b'\n')
        with self.assertRaisesRegex(ValueError, 'ambiguous'): self.f.audit(ref)
        event.write_bytes(raw)
        write(Path(task['output']) / 'batch_9999.partial.json', {'ambiguous': True})
        with self.assertRaisesRegex(ValueError, 'partial'): self.f.audit(ref)

    def test_original_window_clips_next_deadline_and_tampering_rejected(self):
        ref = self.f.bind(0); self.f.close(ref)
        nxt = self.f.bind(1, now=186000.); m = b.read(nxt, b.audit_io.Evidence())
        self.assertEqual(m['deadline_epoch'], 186400.)
        self.assertEqual(m['runtime_seconds'], 520)
        window = b.read(m['global_generation_window'], b.audit_io.Evidence())
        for key, value in [('global_generation_cutoff_epoch', 186401.), ('authorization', {'changed': True})]:
            altered = copy.deepcopy(window); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_window(altered, self.f.collection_ref, self.f.collection)

    def test_expired_window_and_less_than_idle_qualification_block(self):
        ref = self.f.bind(0); self.f.close(ref)
        for now in (186400., 186341., 186340.):
            with self.subTest(now=now), self.assertRaisesRegex(ValueError, 'cutoff|60-second'):
                self.f.bind(1, now=now)
        self.assertFalse((self.f.root / 'pilot-waves-v1/wave_01/RESOURCE_BINDING.json').exists())

    def test_changed_command_batch_or_taskhash_is_rejected(self):
        wave = copy.deepcopy(self.f.collection['waves'][0]); phase = b.read(wave['phase_inputs'], b.audit_io.Evidence())
        phase['tasks'][0]['argv'][3] = 'unreviewed.module'
        wave['phase_inputs'] = write(self.f.root / 'bad-phase.json', phase)
        with self.assertRaisesRegex(ValueError, 'command'):
            b.wave_inputs(self.f.collection, wave, b.audit_io.Evidence())
        phase['tasks'][0]['argv'][3] = 'infra.gpu03.task_specific_discovery.causal_worker'
        phase['tasks'][0]['task']['sha256'] = '0' * 64
        wave['phase_inputs'] = write(self.f.root / 'bad-hash-phase.json', phase)
        with self.assertRaises(ValueError): b.wave_inputs(self.f.collection, wave, b.audit_io.Evidence())

    def test_reuse_or_changed_collection_provenance_fails_before_window(self):
        for key, value in [('reused_requests', {'old': {}}), ('successor', {})]:
            altered = copy.deepcopy(self.f.collection); altered[key] = value
            ref = write(self.f.root / ('BAD_' + key + '.json'), altered)
            with self.subTest(key=key), self.f.contexts(), self.assertRaises(ValueError):
                b.collection_inputs(ref, b.audit_io.Evidence())
        self.assertFalse((self.f.root / 'pilot-waves-v1/GLOBAL_GENERATION_WINDOW.json').exists())

    def test_changed_binder_bytes_in_frozen_operational_inventory_fail(self):
        changed = copy.deepcopy(self.f.collection)
        inventory_ref = changed['successor']['operational_source_inventory']
        inventory = b.read(inventory_ref, b.audit_io.Evidence())
        inventory['infra/gpu03/task_specific_discovery/bind_successor_wave.py']['sha256'] = '0' * 64
        newref = write(self.f.root / 'BAD_SOURCE_FILES.json', inventory)
        # Resource tests' admission fixture freezes plan content, so update its
        # authored admitted metadata only to isolate this independent binder gate.
        self.f.plan['successor']['operational_source_inventory'] = newref
        rewrite(self.f.plan_ref['path'], self.f.plan); self.f.plan_ref = b.reference(Path(self.f.plan_ref['path']))
        changed['plan'] = self.f.plan_ref; changed['successor'] = copy.deepcopy(self.f.plan['successor'])
        changed_ref = write(self.f.root / 'BAD_BINDER_COLLECTION.json', changed)
        with self.f.contexts(), self.assertRaisesRegex(ValueError, 'SHA differs'):
            b.collection_inputs(changed_ref, b.audit_io.Evidence())

    def test_postclaim_interruption_remains_consumed(self):
        original = b.worker.write
        def fail(path, value):
            result = original(path, value)
            if Path(path).name == 'RESOURCE_BINDING.json': raise RuntimeError('injected post-claim failure')
            return result
        with patch.object(b.worker, 'write', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'post-claim'):
            self.f.bind(0)
        with self.assertRaisesRegex(ValueError, 'already consumed'): self.f.bind(0)


if __name__ == '__main__': unittest.main()
