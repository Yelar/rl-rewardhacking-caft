"""Inert retained-byte fixtures only; never generate or evaluate Python code."""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from . import build_causal_eval_plan as b
from . import build_causal_plan as plan_io
from . import causal_eval_lane as lane
from . import test_build_causal_plan as planning
from . import test_qualify_causal_evaluator as authored_tests
from . import test_causal_evaluate as evaluator_tests
from .io_contract import RequestJournal, canonical, hash_value, ContractError


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n'); path.chmod(0o400)
    return {'path': str(path), **b.io.binding(path)}


def rewrite(path, value):
    path = Path(path); path.chmod(0o600)
    return write(path, value)


class Fixture:
    """Complete 32-problem retained corpus, minimal authored condition panel."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.source = self.root / 'source'; self.source.mkdir()
        sources = {}
        for name in set(b.authored.REQUIRED_SOURCES) | set(b.worker.REQUIRED_SOURCE_FILES) | {b.SELF_SOURCE, b.LANE_SOURCE}:
            path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# authored inert source marker; never imported\n'); path.chmod(0o400)
            sources[name] = b.io.binding(path)
        write(self.source / 'SOURCE_FILES.json', sources)
        prepared = planning.fixture_prompts()
        self.rows = plan_io.tokenize(prepared['prompt_rows'], prepared['pilot_problem_keys'],
                                     prepared['qualification_problem_keys'], planning.Tokenizer())
        rows_path = self.root / 'prepared.jsonl'; b.write_rows(rows_path, self.rows)
        model = self.root / 'model'; model.mkdir(); adapter = self.root / 'adapter'; adapter.mkdir()
        bound = {str(rows_path): b.io.binding(rows_path)}; self.adapter_hashes = {}
        for path, names in ((model, b.worker.BASE_FILES), (adapter, b.worker.ADAPTERS)):
            for name in names:
                reference = write(path / name, {'authored_inert_model_input': name})
                bound[reference['path']] = b.binding(reference)
                if path == adapter: self.adapter_hashes[name] = reference['sha256']
        vectors = self.root / 'vectors.safetensors'; vectors.write_bytes(b'authored vectors; never loaded')
        bound[str(vectors)] = b.io.binding(vectors)
        bound.update({str(self.source / name): value for name, value in sources.items()})
        self.conditions = {
            'baseline': {'id': 'baseline', 'layer': None, 'operator': 'ordinary', 'control': 'zero_strength', 'strength': 0.},
            'zero': {'id': 'zero', 'layer': 21, 'operator': 'ordinary', 'rank': 1, 'vector_key': 'q', 'control': 'zero_strength', 'strength': 0.},
            'material': {'id': 'material', 'layer': 21, 'operator': 'ordinary', 'rank': 1, 'vector_key': 'q', 'control': 'materialization_only', 'strength': 0.},
            'target': {'id': 'target', 'layer': 21, 'operator': 'ordinary', 'rank': 1, 'vector_key': 'q', 'control': 'project', 'strength': .5}}
        self.plan = {'schema': plan_io.SCHEMA, 'status': 'prepared_not_dispatched', 'run_token': 'authored-pilot-token',
            'study_id': planning.p.STUDY, 'source_root': str(self.source), 'qualified_source_files': sources,
            'model_snapshot': str(model), 'checkpoint': str(adapter), 'prepared_rows': str(rows_path), 'vectors': str(vectors),
            'model_revision': b.worker.REVISION, 'sampling': b.worker.SAMPLING, 'profile': b.worker.PROFILE,
            'batch_size': 8, 'bound_files': bound, 'conditions': self.conditions,
            'qualification_condition_ids': ['baseline', 'zero', 'material'], 'pilot_condition_ids': ['baseline', 'material', 'target']}
        for stage in ('qualification', 'pilot'):
            self.plan[stage + '_requests'] = plan_io.requests_for(self.rows, self.plan[stage + '_condition_ids'],
                self.conditions, qualification=stage == 'qualification')
        self.plan['budgets'] = {'qualification': 72, 'pilot_including_reuse': 576, 'reusable_pilot_requests': 48}
        self.plan['qualification_contract'] = plan_io.qualify_contract(self.rows, {
            'null_pairs': [{'zero_strength': 'zero', 'materialization_only': 'material'}]}, sources, bound[str(vectors)]['sha256'])
        self.plan_ref = write(self.root / 'PLAN.json', self.plan)
        qualification_task = {k: copy.deepcopy(self.plan[k]) for k in (
            'run_token', 'model_revision', 'model_snapshot', 'checkpoint', 'prepared_rows', 'vectors', 'sampling',
            'batch_size', 'profile', 'source_root', 'qualified_source_files', 'conditions', 'bound_files')}
        qualification_task.update(schema='task_specific_causal_generation_v1', stage='qualification',
                                  requests=self.plan['qualification_requests'], output=str(self.root / 'qualified_worker'))
        qtask = write(self.root / 'qualification_task.json', qualification_task)
        saved = self.make_worker(qualification_task, qtask['sha256'])
        self.proof = {'schema': 'task_specific_causal_qualification_v1', 'status': 'qualified_task_specific_causal_v1',
            'run_token': self.plan['run_token'], 'profile': self.plan['profile'], 'source_files': sources,
            'vectors_sha256': bound[str(vectors)]['sha256'], 'all_returned_requests_retained': True,
            'gpu_workers_released': True, 'null_token_parity': True, 'null_logits_within_tolerance': True,
            'model_files': b.worker.model_bindings(self.plan, b.worker.BASE_FILES, 'model_snapshot'),
            'adapter_files': b.worker.model_bindings(self.plan, b.worker.ADAPTERS, 'checkpoint'),
            'qualified_signatures': [b.worker.qualification_signature(self.conditions['target'])], 'retained_requests': saved}
        self.proof_ref = write(self.root / 'QUALIFIED.json', self.proof)
        self.phase_ref = plan_io.task_bundle(self.plan_ref, 'pilot', self.root / 'pilot', workers=1, qualification=self.proof_ref)
        phase = plan_io.read_json(self.phase_ref['path']); task_ref = phase['tasks'][0]['task']
        self.task = plan_io.read_json(task_ref['path']); self.task_ref = task_ref
        self.make_worker(self.task, task_ref['sha256'])
        self.runtime = self.root / 'runtime'; self.runtime.mkdir()
        uuid = 'GPU-00000000-0000-0000-0000-000000000000'
        self.manifest = {'schema': 'gpu03_task_specific_resource_guard_v1', 'host': 'gpu-03', 'uid': 3357,
            'run_token': self.plan['run_token'], 'source_root': str(self.source), 'runtime': str(self.runtime),
            'science': {'stage': 'pilot'}, 'bound_files': phase['bound_files'], 'phase_inputs': self.phase_ref,
            'gpu_uuids': {'0': uuid}, 'workers': [{'name': 'gpu_0', 'gpu_id': 0, 'cwd': str(self.source),
                'env': {'PYTHONPATH': str(self.source), 'CUDA_VISIBLE_DEVICES': uuid}, 'argv': phase['tasks'][0]['argv']}]}
        self.manifest_ref = write(self.root / 'RESOURCE_MANIFEST.json', self.manifest)
        self.release = {'run_token': self.plan['run_token'], 'manifest_sha256': self.manifest_ref['sha256'],
            'status': 'workers_exited_zero', 'owned_processes_released': True, 'gpu_release_verified': True,
            'scientific_success_verified': False, 'exit_codes': {'gpu_0': 0},
            'gpus': {'0': {'uuid': uuid, 'pids': [], 'utilization': 0, 'total_mib': 32768, 'free_mib': 32768}}}
        write(self.runtime / 'SUPERVISOR_RESULT.json', self.release)
        b.write_rows(self.runtime / 'workers.jsonl', [{'name': 'gpu_0', 'pid': 123, 'gpu_id': 0}])

    def make_worker(self, task, digest):
        out = Path(task['output']); out.mkdir()
        uuid = 'GPU-00000000-0000-0000-0000-000000000000'
        write(out / 'STARTED.json', {'task_sha256': digest, 'pid': 123, 'requests': len(task['requests']),
                                    'gpu_uuid': uuid, 'stage': task['stage']})
        rows = {r['record_id']: r for r in self.rows}; exports = []; retained = {}
        with RequestJournal(out / 'journal', run_token=task['run_token'], input_sha256=digest,
                request_hashes={r['request_id']: hash_value(r) for r in task['requests']}) as journal:
            for index, batch in enumerate(b.qualification.batches(task['requests'])):
                for row, request in enumerate(batch):
                    rid = request['request_id']; tokens = [b.worker.SAMPLING['eos_token_ids'][0]]
                    result = {'completion': '# authored retained fixture; never evaluated', 'completion_token_ids': tokens,
                        'generated_token_ids': tokens, 'fixed_completion_prefix_token_count': 0, 'batch_size_actual': 8,
                        'batch_profile': b.worker.PROFILE, 'intervention_strength': task['conditions'][request['condition_id']]['strength'],
                        'stop_reason': 'eos', 'batch_wall_seconds': .1}
                    primitive = {'result': result}; journal.start(rid); journal.record_primitive(rid, primitive)
                    journal.record_final(rid, {'returned_row': True, 'batch_index': index, 'batch_row': row,
                                              'first_logits': None, 'batch_wall_seconds': .1})
                    export = b.worker.export_row(request, rows[request['record_id']], primitive, digest); exports.append(export)
                    retained[rid] = {'request': request, 'request_sha256': hash_value(request),
                        'generation_row_sha256': hash_value(export), 'primitive_file': {'path': str(journal._path(rid, 'primitive')),
                            **b.io.binding(journal._path(rid, 'primitive'))},
                        'final_file': {'path': str(journal._path(rid, 'final')), **b.io.binding(journal._path(rid, 'final'))}}
        gref = b.write_rows(out / 'generations.jsonl', exports)
        for value in retained.values(): value['generation_file'] = gref
        write(out / 'COMPLETE.json', {'status': 'succeeded', 'stage': task['stage'], 'task_sha256': digest,
            'requests': len(task['requests']), 'all_returned_requests_retained': True, 'evaluation_performed': False,
            'causal_success_claimed': False, 'cuda_release': {'allocated_bytes': 32 << 20, 'reserved_bytes': 32 << 20},
            'model_load_report': {'with_adapter': True, 'active_adapters': ['default'], 'nonzero_lora_parameter_tensors': 2,
                'lora_parameter_tensors': 2, 'lora_parameter_elements': 10}, 'artifacts': {'generations.jsonl': b.binding(gref)}})
        return retained

    def audit(self):
        with mock.patch.object(b.worker, 'ADAPTERS', self.adapter_hashes):
            return b.audit_pilot(self.plan_ref, self.manifest_ref, self.proof_ref, b.qualification.Evidence())


class RetainedPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name); self.root = self.fixture.root

    def test_complete_real_journal_audit_and_lane_compatible_shards(self):
        plan, rows, retained, audit = self.fixture.audit()
        self.assertEqual((audit['requests'], audit['new_requests'], audit['reused_requests']), (576, 528, 48))
        self.assertEqual(sum(x['origin'] == 'qualification' for x in retained.values()), 48)
        out = self.root / 'evaluation'; out.mkdir()
        tasks, shards = b.assemble(plan, rows, retained, out, self.fixture.source, plan['qualified_source_files'])
        self.assertEqual(len(tasks), 3); self.assertEqual(sum(t['requests'] for t in tasks), 576)
        self.assertEqual(len({rid for shard in shards for rid in shard['request_ids']}), 576)
        self.assertTrue(all(len(s['request_ids']) <= 256 for s in shards))
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), mock.patch.object(lane.outer, 'SOURCE_ROOT', self.fixture.source):
            for item in tasks:
                task = plan_io.read_json(item['task']['path'])
                summary = lane.validate(task)
                self.assertEqual(sum(s['records'] for s in summary), item['requests'])
                self.assertFalse(Path(task['output']).exists())

    def test_ambiguous_missing_final_or_unexpected_partial_refused(self):
        journal = Path(self.fixture.task['output']) / 'journal'
        path = journal / 'final_recorded.jsonl'; data = path.read_bytes(); path.chmod(0o600)
        path.write_bytes(b'\n'.join(data.splitlines()[:-1]) + b'\n')
        with self.assertRaisesRegex(ContractError, 'ambiguous'): self.fixture.audit()
        path.write_bytes(data)
        write(Path(self.fixture.task['output']) / 'batch_9999.partial.json', {'retained_not_returned': True})
        with self.assertRaisesRegex(ContractError, 'unexpected pilot output'): self.fixture.audit()

    def test_failed_or_unreleased_producer_cannot_admit(self):
        for field, value in (('owned_processes_released', False), ('gpu_release_verified', False), ('status', 'failed')):
            release = copy.deepcopy(self.fixture.release); release[field] = value
            rewrite(self.fixture.runtime / 'SUPERVISOR_RESULT.json', release)
            with self.subTest(field=field), self.assertRaises(ContractError): self.fixture.audit()

    def test_export_tampering_fails_even_with_recomputed_producer_checksum(self):
        out = Path(self.fixture.task['output']); path = out / 'generations.jsonl'
        rows = [b.qualification.decode(x) for x in path.read_bytes().splitlines()]
        rows[0]['completion'] = 'changed'; path.chmod(0o600); path.unlink(); reference = b.write_rows(path, rows)
        complete = plan_io.read_json(out / 'COMPLETE.json'); complete['artifacts']['generations.jsonl'] = b.binding(reference)
        rewrite(out / 'COMPLETE.json', complete)
        with self.assertRaisesRegex(ContractError, 'immutable primitive'): self.fixture.audit()

    def test_reused_control_change_is_not_regenerated(self):
        saved = next(iter(self.fixture.proof['retained_requests'].values()))
        path = Path(saved['primitive_file']['path']); path.chmod(0o600); path.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.fixture.audit()

    def test_changed_retained_row_before_sharding_fails(self):
        plan, rows, retained, _ = self.fixture.audit()
        item = next(iter(retained.values())); item['generation_row_sha256'] = 'f' * 64
        out = self.root / 'evaluation'; out.mkdir()
        with self.assertRaisesRegex(ContractError, 'before sharding'):
            b.assemble(plan, rows, retained, out, self.fixture.source, plan['qualified_source_files'])

    def test_one_use_claim_blocks_alternate_output_after_success(self):
        mockref = write(self.root / 'authored-proof.json', {'inert_fixture': True})
        files = self.fixture.plan['qualified_source_files']
        with mock.patch.object(b.worker, 'ADAPTERS', self.fixture.adapter_hashes), \
             mock.patch.object(b, 'source_and_authored', return_value=(files, {'run_token': self.fixture.plan['run_token']}, mockref)):
            args = (self.fixture.plan_ref, self.fixture.manifest_ref, self.fixture.proof_ref, mockref, mockref, mockref, self.fixture.source)
            result = b.build(*args, self.root / 'evaluation')
            phase = plan_io.read_json(result['path'])
            self.assertEqual(phase['requests'], 576)
            self.assertFalse(phase['evaluation_performed'])
            self.assertFalse(phase['dispatch_authorized_by_this_file'])
            with self.assertRaises(FileExistsError): b.build(*args, self.root / 'second-evaluation')
        self.assertFalse((self.root / 'second-evaluation').exists())

    def test_crash_after_claim_never_allows_new_output_replay(self):
        mockref = write(self.root / 'authored-proof.json', {'inert_fixture': True})
        files = self.fixture.plan['qualified_source_files']
        with mock.patch.object(b.worker, 'ADAPTERS', self.fixture.adapter_hashes), \
             mock.patch.object(b, 'source_and_authored', return_value=(files, {'run_token': self.fixture.plan['run_token']}, mockref)), \
             mock.patch.object(b, 'assemble', side_effect=RuntimeError('authored crash after durable claim')):
            args = (self.fixture.plan_ref, self.fixture.manifest_ref, self.fixture.proof_ref, mockref, mockref, mockref, self.fixture.source)
            with self.assertRaisesRegex(RuntimeError, 'durable claim'): b.build(*args, self.root / 'interrupted')
            with self.assertRaises(FileExistsError): b.build(*args, self.root / 'alternate')
        self.assertFalse((self.root / 'alternate').exists())


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_actual_cpu_release_shape_and_missing_positive_flag(self):
        manifest = {'schema': 'gpu03_task_specific_resource_guard_v1', 'host': 'gpu-03', 'uid': 3357,
            'run_token': 'authored', 'runtime': str(self.root), 'gpu_uuids': {},
            'workers': [{'name': 'cpu', 'gpu_id': None, 'env': {'CUDA_VISIBLE_DEVICES': ''}}]}
        release = {'run_token': 'authored', 'manifest_sha256': 'a' * 64, 'status': 'workers_exited_zero',
            'owned_processes_released': True, 'gpu_release_verified': None, 'cpu_only': True,
            'scientific_success_verified': False, 'exit_codes': {'cpu': 0}}
        path = self.root / 'SUPERVISOR_RESULT.json'; write(path, release)
        b.write_rows(self.root / 'workers.jsonl', [{'name': 'cpu', 'pid': 10, 'gpu_id': None}])
        b.positive_release(manifest, 'a' * 64, b.qualification.Evidence(), gpu=False)
        for change in ({'cpu_only': False}, {'gpu_release_verified': True}, {'gpus': {'0': {}}}):
            rewrite(path, {**release, **change})
            with self.subTest(change=change), self.assertRaises(ContractError):
                b.positive_release(manifest, 'a' * 64, b.qualification.Evidence(), gpu=False)

    def test_bounded_stream_rejects_truncation_duplicates_and_oversized_rows(self):
        path = self.root / 'rows.jsonl'; path.write_bytes(b'{"a":1}')
        with self.assertRaisesRegex(ContractError, 'Truncated'):
            list(b.stream_rows(path, b.qualification.Evidence()))
        path.write_bytes(b'{"a":1,"a":2}\n')
        with self.assertRaisesRegex(ContractError, 'Duplicate'):
            list(b.stream_rows(path, b.qualification.Evidence()))
        path.write_bytes(b'{}\n{}\n')
        with self.assertRaises(ContractError): list(b.stream_rows(path, b.qualification.Evidence(), max_rows=1))

    def synthetic_rows(self, count):
        generated, example, _ = evaluator_tests.fixture()
        requests, rows = [], []
        for n in range(count):
            request = {'request_id': f'authored-{n}', 'record_id': 'prepared-1', 'condition_id': f'condition-{n}',
                       'problem_id': example['id'], 'protocol': 'fixed', 'sample_index': 0, 'seed': 1, 'scope': 'primary'}
            requests.append(request); rows.append({**generated, **request, 'generation_seed': 1})
        path = self.root / 'generation.jsonl'; b.write_rows(path, rows)
        retained = {row['request_id']: where for row, where in b.stream_rows(path, b.qualification.Evidence())}
        return {'run_token': 'authored', 'pilot_requests': requests}, {'prepared-1': {'dataset_row': example}}, retained

    def test_count_and_byte_limits_split_without_losing_rows(self):
        plan, rows, retained = self.synthetic_rows(600)
        output = self.root / 'eval'; output.mkdir()
        tasks, shards = b.assemble(plan, rows, retained, output, self.root, {}, lanes=2)
        self.assertEqual([len(s['request_ids']) for s in shards], [256, 256, 88])
        self.assertEqual(sum(t['requests'] for t in tasks), 600)
        output2 = self.root / 'eval-small'; output2.mkdir()
        with mock.patch.object(b.evaluator, 'MAX_FILE_BYTES', 10000):
            tasks, shards = b.assemble(plan, rows, retained, output2, self.root, {}, lanes=8)
        self.assertEqual(sum(len(s['request_ids']) for s in shards), 600)
        self.assertTrue(all(s['generated']['size_bytes'] <= 10000 for s in shards))
        self.assertTrue(all(0 < len(plan_io.read_json(t['task']['path'])['shards']) <= 64 for t in tasks))


class AuthoredAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.f = authored_tests.Fixture(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root, self.source = self.f.root, self.f.source
        files = self.f.task['source_files']
        for name in set(b.worker.REQUIRED_SOURCE_FILES) | {b.SELF_SOURCE, b.LANE_SOURCE}:
            path = self.source / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# inert source marker\n')
                files[name] = b.io.binding(path)
        write(self.source / 'SOURCE_FILES.json', files)
        self.f.task['venv_dir'] = str(Path(b.PYTHON).parent.parent)
        self.f.run_inert()  # Only injected inert evaluator/subprocess; real durable journals.
        self.task_ref = {'path': str(self.root / 'task.json'), **b.io.binding(self.root / 'task.json')}
        self.proof_ref = {'path': str(self.root / 'qualification/COMPLETE.json'),
                          **b.io.binding(self.root / 'qualification/COMPLETE.json')}
        runtime = self.root / 'runtime'; runtime.mkdir()
        self.manifest = {'schema': 'gpu03_task_specific_resource_guard_v1', 'host': 'gpu-03', 'uid': 3357,
            'run_token': self.f.task['run_token'], 'source_root': str(self.source), 'runtime': str(runtime),
            'bound_files': {self.task_ref['path']: b.binding(self.task_ref)}, 'gpu_uuids': {},
            'workers': [{'name': 'cpu', 'gpu_id': None, 'cwd': str(self.source),
                'env': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(self.source)},
                'argv': [b.PYTHON, '-B', '-m', 'infra.gpu03.task_specific_discovery.qualify_causal_evaluator',
                         '--task', self.task_ref['path'], '--sha256', self.task_ref['sha256']]}]}
        self.manifest_ref = write(self.root / 'RESOURCE_MANIFEST.json', self.manifest)
        self.release = {'run_token': self.f.task['run_token'], 'manifest_sha256': self.manifest_ref['sha256'],
            'status': 'workers_exited_zero', 'owned_processes_released': True, 'gpu_release_verified': None,
            'cpu_only': True, 'scientific_success_verified': False, 'exit_codes': {'cpu': 0}}
        write(runtime / 'SUPERVISOR_RESULT.json', self.release)
        b.write_rows(runtime / 'workers.jsonl', [{'name': 'cpu', 'pid': 10, 'gpu_id': None}])

    def check(self):
        return b.source_and_authored(self.source, self.task_ref, self.proof_ref, self.manifest_ref,
                                    b.qualification.Evidence())

    def test_complete_authored_production_journals_and_cpu_release_admitted(self):
        files, task, release = self.check()
        self.assertIn(b.LANE_SOURCE, files)
        self.assertEqual(task, self.f.task)
        self.assertEqual(release['path'], str(self.root / 'runtime/SUPERVISOR_RESULT.json'))

    def test_consumed_evaluator_change_rejected_but_new_preparation_allowed(self):
        successor = self.root / 'source-successor'; shutil.copytree(self.source, successor)
        preparation = successor / b.SELF_SOURCE
        preparation.write_text('# newer preparation-only source, never dispatched\n')
        files = plan_io.read_json(successor / 'SOURCE_FILES.json'); files[b.SELF_SOURCE] = b.io.binding(preparation)
        rewrite(successor / 'SOURCE_FILES.json', files)
        b.source_and_authored(successor, self.task_ref, self.proof_ref, self.manifest_ref, b.qualification.Evidence())
        path = self.source / 'infra/gpu03/task_specific_discovery/causal_evaluate.py'
        path.write_text('# changed evaluator\n')
        files = plan_io.read_json(self.source / 'SOURCE_FILES.json'); files[str(path.relative_to(self.source))] = b.io.binding(path)
        rewrite(self.source / 'SOURCE_FILES.json', files)
        with self.assertRaisesRegex(ContractError, 'same-source'): self.check()

    def test_wrong_or_unreleased_authored_task_fails(self):
        release = copy.deepcopy(self.release); release['owned_processes_released'] = False
        rewrite(self.root / 'runtime/SUPERVISOR_RESULT.json', release)
        with self.assertRaisesRegex(ContractError, 'release missing'): self.check()


if __name__ == '__main__': unittest.main()
