"""Inert CPU-only preparation tests; no model, dispatch or generated code."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from . import prepare_causal_waves as w
from . import test_build_causal_plan as fixtures


class WaveTests(unittest.TestCase):
    def setUp(self):
        fixtures.TaskTests.setUp(self)
        source = Path(self.plan['source_root']); source.mkdir()
        project = Path(w.__file__).resolve().parents[3]
        names = set(w.original.REQUIRED_SOURCE) | set(w.worker.REQUIRED_SOURCE_FILES)
        names |= {'infra/gpu03/task_specific_discovery/' + name for name in
                  ('causal_qualification.py', 'causal_prompts.py', 'fit_worker.py', 'io_contract.py')}
        for name in names:
            target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((project / name).read_bytes()); target.chmod(0o444)
        self.plan['qualified_source_files'] = {name: w.io.binding(source / name) for name in sorted(names)}
        self.plan['qualification_contract']['source_files'] = self.plan['qualified_source_files']
        w.worker.write(source / 'SOURCE_FILES.json', self.plan['qualified_source_files'])
        self.plan_path = self.root / 'FROZEN_PLAN.json'
        w.worker.write(self.plan_path, self.plan); self.plan_ref = w.original.ref(self.plan_path)
        self.qref = fixtures.TaskTests.proof(self, lambda p: p.update(run_token=self.plan['run_token']))

    def prepare(self, output='waves'):
        return w.prepare(self.plan_ref, self.qref, self.root / output)

    def collection(self):
        reference = self.prepare()
        return w.original.read_json(reference['path'])

    def linked_base_policy(self):
        """Actual filesystem links; only authored path/byte/owner roster changes."""
        base = Path(self.plan['model_snapshot']); aliases = self.root / 'base-link-aliases'; aliases.mkdir()
        roster = {}
        for name in w.worker.BASE_FILES:
            path = base / name; path.chmod(0o644); os.link(path, aliases / name)
            roster[str(path)] = self.plan['bound_files'][str(path)]
        stack = ExitStack()
        stack.enter_context(patch.object(w.audit_io, 'BASE_ROOT', str(base)))
        stack.enter_context(patch.object(w.audit_io, 'BASE_ROSTER', roster))
        stack.enter_context(patch.object(w.audit_io, 'LINK_OWNER_UID', os.getuid()))
        return stack

    def test_exact_thirteen_base_hardlinks_allow_full_unchanged_plan(self):
        with self.linked_base_policy():
            first = Path(self.plan['model_snapshot']) / w.worker.BASE_FILES[0]
            with self.assertRaisesRegex(ValueError, 'linked evidence'):
                w.qualification.Evidence().bind(first)
            collection = self.collection()
        provenance = collection['audit_io_provenance']
        self.assertEqual(len(provenance['verified_linked_files']), 13)
        self.assertFalse(provenance['filesystem_immutability_claimed'])
        helper = collection['audit_policy_helper']
        self.assertEqual(helper['path'], str(Path(w.audit_io.__file__).resolve()))
        self.assertEqual(collection['bound_files'][helper['path']], w.binding(helper))
        self.assertEqual(collection['qualified_source_files'], self.plan['qualified_source_files'])
        self.assertEqual(collection['expected_new_requests'], 15000)

    def test_base_exception_does_not_allow_linked_sources_proofs_or_returns(self):
        proof = w.original.read_json(self.qref['path'])
        paths = [Path(self.plan['source_root']) / 'infra/gpu03/task_specific_discovery/causal_worker.py',
                 Path(self.qref['path']), Path(next(iter(proof['retained_requests'].values()))['primitive_file']['path'])]
        with self.linked_base_policy():
            for index, path in enumerate(paths):
                alias = self.root / f'forbidden-link-{index}'; os.link(path, alias)
                try:
                    with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Linked evidence'):
                        self.prepare()
                    self.assertFalse((self.root / 'PILOT_TASK_BINDING.json').exists())
                finally:
                    alias.unlink()

    def test_full_coverage_original_worker_api_and_batch_lane_order(self):
        collection = self.collection()
        self.assertEqual(collection['schema'], w.SCHEMA)
        self.assertEqual((collection['expected_total_requests'], collection['expected_new_requests'],
                          collection['expected_reused_requests']), (15168, 15000, 168))
        self.assertFalse(collection['dispatch_authorized_by_this_file'])
        self.assertEqual(collection['limits'], dict(wave_seconds=14400, overall_generation_seconds=86400,
                                                  max_lane_requests=240, batch_size=8))
        reused = set(collection['reused_requests'])
        remaining = [r for r in self.plan['pilot_requests'] if r['request_id'] not in reused]
        batches = [remaining[i:i + 8] for i in range(0, len(remaining), 8)]
        seen, lane_indices, phase_counts = set(), [[] for _ in range(8)], []
        adapters = {name: self.plan['bound_files'][str(Path(self.plan['checkpoint']) / name)]['sha256']
                    for name in w.worker.ADAPTERS}
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-00000000-0000-0000-0000-000000000000'), \
                patch.object(w.worker, 'ADAPTERS', adapters):
            for number, entry in enumerate(collection['waves']):
                phase = w.original.read_json(entry['phase_inputs']['path'])
                self.assertEqual(phase['wave_index'], number)
                self.assertEqual(phase['wave_count'], 8)
                self.assertEqual(phase['expected_reused_requests'], 0)
                self.assertEqual(phase['reused_requests'], {})
                self.assertEqual(phase['expected_total_requests'], phase['expected_new_requests'])
                phase_counts.append(phase['expected_new_requests'])
                self.assertEqual(len(phase['tasks']), 8)
                self.assertEqual(entry['request_ids'], [r['request_id'] for i in entry['global_batch_indices'] for r in batches[i]])
                for item in phase['tasks']:
                    task = w.original.read_json(item['task']['path']); lane = item['lane']
                    self.assertEqual(len(w.worker.validate(task)), 96)
                    self.assertEqual(task['source_root'], self.plan['source_root'])
                    self.assertEqual(task['qualified_source_files'], self.plan['qualified_source_files'])
                    self.assertEqual(task['deadline_seconds'], 14400)
                    self.assertEqual(task['sampling'], self.plan['sampling'])
                    self.assertEqual(task['profile'], self.plan['profile'])
                    self.assertEqual(task['requests'], [r for index in item['global_batch_indices'] for r in batches[index]])
                    self.assertTrue(all(i % 8 == lane for i in item['global_batch_indices']))
                    self.assertEqual(w.qualification.batches(task['requests']), [batches[i] for i in item['global_batch_indices']])
                    self.assertLessEqual(len(task['requests']), 240)
                    self.assertEqual(len(task['requests']) % 8, 0)
                    self.assertEqual(item['argv'], [w.PYTHON, '-B', '-m',
                        'infra.gpu03.task_specific_discovery.causal_worker', '--task', item['task']['path'],
                        '--sha256', item['task']['sha256']])
                    ids = {r['request_id'] for r in task['requests']}
                    self.assertFalse(ids & seen); self.assertFalse(ids & reused)
                    seen.update(ids); lane_indices[lane].extend(item['global_batch_indices'])
        self.assertEqual(phase_counts, [1880] * 3 + [1872] * 5)
        self.assertEqual(seen, {r['request_id'] for r in remaining})
        for lane, indices in enumerate(lane_indices):
            self.assertEqual(indices, list(range(lane, 1875, 8)))
        claim = w.original.read_json(collection['pilot_claim']['path'])
        self.assertEqual(claim['assignments'], [{k: v for k, v in x.items() if k != 'phase_inputs'} for x in collection['waves']])
        self.assertEqual(claim['assignments_sha256'], w.hash_value(claim['assignments']))
        self.assertEqual(claim['request_sha256_by_id'], {r['request_id']: w.hash_value(r) for r in remaining})

    def test_same_inputs_partition_deterministic_without_any_write(self):
        reused = {r['request_id']: {} for r in self.plan['qualification_requests']
                  if r['request_id'] in {x['request_id'] for x in self.plan['pilot_requests']}}
        before = {p.name for p in self.root.iterdir()}
        first = w.partition(self.plan, reused)
        self.assertEqual(first, w.partition(copy.deepcopy(self.plan), copy.deepcopy(reused)))
        self.assertEqual(before, {p.name for p in self.root.iterdir()})

    def test_shared_claim_blocks_new_planner_and_original_monolithic_builder(self):
        self.prepare()
        with self.assertRaises(FileExistsError): self.prepare('second-waves')
        with self.assertRaises(FileExistsError):
            w.original.task_bundle(self.plan_ref, 'pilot', self.root / 'old-pilot', 8, self.qref)
        self.assertFalse((self.root / 'second-waves').exists())
        self.assertFalse((self.root / 'old-pilot').exists())

    def test_existing_monolithic_claim_blocks_wave_planner(self):
        w.original.task_bundle(self.plan_ref, 'pilot', self.root / 'old-pilot', 8, self.qref)
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertFalse((self.root / 'waves').exists())

    def test_crash_after_claim_prevents_outputs_and_all_replay(self):
        write = w.worker.write
        def crash(path, value):
            result = write(path, value)
            if Path(path).name == 'PILOT_TASK_BINDING.json': raise RuntimeError('crash after durable claim')
            return result
        with patch.object(w.worker, 'write', side_effect=crash), self.assertRaisesRegex(RuntimeError, 'durable claim'):
            self.prepare()
        self.assertTrue((self.root / 'PILOT_TASK_BINDING.json').exists())
        self.assertFalse((self.root / 'waves').exists())
        with self.assertRaises(FileExistsError): self.prepare('second-waves')

    def test_partial_task_preparation_has_no_collection_and_no_retry(self):
        write = w.worker.write
        def crash(path, value):
            result = write(path, value)
            if Path(path).parent.name == 'wave_02' and Path(path).name == 'task_01.json':
                raise RuntimeError('crash during tasks')
            return result
        with patch.object(w.worker, 'write', side_effect=crash), self.assertRaisesRegex(RuntimeError, 'during tasks'):
            self.prepare()
        self.assertFalse((self.root / 'waves/COLLECTION.json').exists())
        self.assertTrue((self.root / 'waves/wave_02/task_01.json').exists())
        with self.assertRaises(FileExistsError): self.prepare('second-waves')

    def test_truncated_or_symlink_global_claim_blocks_without_adoption(self):
        claim = self.root / 'PILOT_TASK_BINDING.json'; claim.write_bytes(b'{"schema":')
        with self.assertRaises(FileExistsError): self.prepare()
        claim.unlink(); claim.symlink_to(self.root / 'missing-claim')
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertFalse((self.root / 'waves').exists())

    def test_changed_source_or_model_bytes_fail_before_claim(self):
        source = Path(self.plan['source_root']) / 'infra/gpu03/task_specific_discovery/causal_worker.py'
        before = source.read_bytes(); source.chmod(0o600); source.write_bytes(before + b'# changed\n')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'PILOT_TASK_BINDING.json').exists())
        source.write_bytes(before); source.chmod(0o444)
        Path(self.plan['vectors']).write_bytes(b'changed vectors')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'PILOT_TASK_BINDING.json').exists())

    def test_changed_request_seed_or_order_fails_even_with_new_plan_hash(self):
        bad = copy.deepcopy(self.plan); bad['pilot_requests'][0]['seed'] += 1
        path = self.root / 'bad-plan.json'; w.worker.write(path, bad)
        with self.assertRaisesRegex(ValueError, 'IDs/order/seeds'):
            w.prepare(w.original.ref(path), self.qref, self.root / 'waves')
        self.assertFalse((self.root / 'PILOT_TASK_BINDING.json').exists())

    def test_unqualified_proof_or_changed_reused_return_is_not_adopted(self):
        proof = w.original.read_json(self.qref['path']); proof['null_token_parity'] = False
        path = self.root / 'bad-proof.json'; w.worker.write(path, proof)
        with self.assertRaisesRegex(ValueError, 'successful qualification'):
            w.prepare(self.plan_ref, w.original.ref(path), self.root / 'waves')
        primitive = Path(next(iter(proof['retained_requests'].values()))['primitive_file']['path'])
        primitive.write_bytes(b'changed retained return')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'PILOT_TASK_BINDING.json').exists())

    def test_partial_batch_reuse_and_duplicate_or_incomplete_universe_rejected(self):
        reused = {r['request_id']: {} for r in self.plan['qualification_requests']
                  if r['request_id'] in {x['request_id'] for x in self.plan['pilot_requests']}}
        bad = dict(reused); del bad[next(iter(bad))]
        replacement = next(r['request_id'] for r in self.plan['pilot_requests'] if r['request_id'] not in reused)
        bad[replacement] = {}
        with self.assertRaisesRegex(ValueError, 'splits'): w.partition(self.plan, bad)
        duplicate = copy.deepcopy(self.plan); duplicate['pilot_requests'][1] = duplicate['pilot_requests'][0]
        with self.assertRaises(ValueError): w.partition(duplicate, reused)
        short = copy.deepcopy(self.plan); short['pilot_requests'].pop()
        with self.assertRaises(ValueError): w.partition(short, reused)

    def test_collection_written_last_and_all_references_hash_match(self):
        writes, write = [], w.worker.write
        def tracked(path, value):
            writes.append(Path(path)); return write(path, value)
        with patch.object(w.worker, 'write', side_effect=tracked): result = self.prepare()
        self.assertEqual(writes[0].name, 'PILOT_TASK_BINDING.json')
        self.assertEqual(writes[-1], Path(result['path']))
        self.assertEqual(len(writes), 1 + 8 * 9 + 1)
        collection = w.original.read_json(result['path'])
        for path, value in collection['bound_files'].items(): w.io.verify_file(path, value)
        for entry in collection['waves']:
            phase = w.original.read_json(entry['phase_inputs']['path'])
            for path, value in phase['bound_files'].items(): w.io.verify_file(path, value)
        self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in writes))


if __name__ == '__main__': unittest.main()
