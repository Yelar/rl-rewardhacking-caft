"""Authored transport fixtures; no model, generated code, SSH, or services."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.gpu03.direction_discovery import remote_generation as r
from infra.gpu03.direction_discovery import phase_budget as budget


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(value if isinstance(value, bytes) else r.canonical(value).encode())
    path.chmod(0o400)
    return r.ref(path)


class Fixture:
    def __init__(self, root, host="gpu-02"):
        self.host, self.protocol = host, r.execution_protocol(host)
        self.root = root
        self.token = 'codex-discovery-remote-authored-20260907'
        self.stage = root / host.replace('-', '') / (self.token + '-stage'); self.source = self.stage / 'source'
        self.output = root / host.replace('-', '') / (self.token + '-results')
        self.runtime = root / host.replace('-', '') / (self.token + '-runtime')
        self.manifest_path = self.stage / 'reviewed_manifest.json'
        self.ledger = {**dict.fromkeys(r.COUNTERS, 0), 'gpu_phase_wall_seconds': 0., 'phases': [], 'bindings': []}
        self.gpus = {'1': 'GPU-00000000-0000-0000-0000-000000000002'}
        self.request = {'request_id': 'authored-qualification', 'record_id': 'author-record', 'condition_id': 'baseline'}
        self.plan = {'mode': 'qualify', 'evaluation_partition': 'direction_fit', 'requests': [self.request],
                     'conditions': {'baseline': {'layers': []}}}
        self.plan_ref = put(root / 'authority/plan.json', self.plan)
        self.prepared = put(root / 'authority/prepared.jsonl', b'{"authored":true}\n')
        source_inventory = {}
        for name in ('engine.py', 'intervention.py'):
            relative = 'infra/gpu03/direction_discovery/' + name
            path = self.source / relative
            put(path, (Path(r.__file__).parent / name).read_bytes())
            source_inventory[relative] = r.sha(path)
        self.template = {'run_token': 'codex-original-authored', 'worker_name': 'gpu_5', 'gpu_id': 5,
            'mode': 'qualify', 'model_snapshot': str(root / 'authority/model'), 'checkpoint': str(root / 'authority/adapter'),
            'raw_package': '/scratch/researcher/legacy-unneeded-cache', 'prepared_records': self.prepared['path'],
            'output': str(root / 'original-output'), 'deadline_seconds': 500, 'requests': [self.request],
            'conditions': self.plan['conditions'], 'sampling': {'temperature': .7, 'top_p': .95},
            'attention_policy': 'exclusive_math', 'teacher_forced_padded_sequence_length': 2176}
        model_files, mapping = {}, []
        for i in range(12):
            group = 'model' if i < 10 else 'adapter'
            path = root / 'authority' / group / ('config.json' if i == 0 else 'generation_config.json' if i == 1 else str(i) + '.bin')
            put(path, ('authored' + str(i)).encode()); model_files[str(path)] = r.info(path)
            remote = root / 'remote-existing' / group / path.name
            put(remote, path.read_bytes())
            mapping.append({'authority': str(path), 'remote': str(remote), **r.info(path)})
        source_qual = put(root / 'authority/source_qualification.json', {
            'status': 'independently_verified_remote_generation_source', 'source_inventory': source_inventory,
            'skips': 0, 'tests': 1, 'qualified_jobs_released': True})
        lifetime = put(root / 'authority/lifetime.json', {'status': 'independently_verified_scoped_ssh_disconnect_survival',
            'host': self.host, 'current_linger': 'yes', 'launch_ssh_ancestor_gone': True, 'launcher_gone': True,
            'owned_process_released': True, 'cgroup_absent_or_empty': True, 'elapsed_seconds': 81., 'detached_span_seconds': 50.,
            'zero_session_survival_unproven': True, 'other_sessions_not_modified': True,
            'service_exit': {'service_result': 'success', 'exit_code': 'exited', 'exit_status': '0', 'invocation_id': 'f'*32}})
        template_ref = put(root / 'authority/template_task.json', self.template)
        original = {'host': 'gpu-04', 'runtime_versions': {'torch': '2.8.0+cu128'}, 'scientific': {'master_plan_sha256': r.MASTER},
            'bound_files': {**{p:v for p,v in model_files.items() if Path(p).name not in ('config.json','generation_config.json')}, template_ref['path']: r.info(template_ref['path'])}}
        original_ref = put(root / 'authority/original_manifest.json', original)
        template_proof = put(root / 'authority/template_proof.json', {'status':'authored_mock'})
        authority_files = {v['path']: r.info(v['path']) for v in
            (self.plan_ref, self.prepared, source_qual, lifetime, template_ref, original_ref, template_proof)}
        authority_files.update(model_files)
        self.authority = {'protocol': self.protocol, 'authority_host': 'gpu-04', 'master_plan_sha256': r.MASTER,
            'parent_plan_sha256': r.PARENT, 'no_intermediate_analysis': True,
            'task_template': self.template, 'template_task': template_ref, 'template_manifest': original_ref,
            'template_verification': template_proof, 'prepared_path': self.prepared['path'],
            'request_plan': self.plan_ref, 'prepared': r.info(self.prepared['path']),
            'model_files': model_files, 'files': authority_files}
        authority_ref = put(self.stage / 'input/authority.json', self.authority)
        self.profile = {'protocol': self.protocol, 'host': self.host, 'authority_host': 'gpu-04', 'owner': 'researcher',
            'no_training': True, 'sampling_unchanged': True, 'master_plan_sha256': r.MASTER,
            'runtime_versions': {'torch': '2.8.0+cu128'}, 'python': sys.executable,
            'gpu_model': 'NVIDIA RTX 5000 Ada Generation', 'maximum_concurrent_gpus': 8,
            'driver_version': '570.195.03', 'gpu_uuids': self.gpus, 'source_inventory': source_inventory,
            'source_qualification': source_qual, 'lifetime_qualification': lifetime, 'inference_qualification': None,
            'model_files': mapping, 'model_paths': {'model_snapshot': str(root / 'remote-existing/model'), 'checkpoint': str(root / 'remote-existing/adapter')},
            'sha256_authority_model_files': __import__('hashlib').sha256(r.canonical(model_files).encode()).hexdigest()}
        profile_ref = put(self.stage / 'input/profile.json', self.profile)
        put(self.stage / 'input/request_plan.json', self.plan)
        self.task = {**self.template, **self.profile['model_paths'], 'run_token': self.token, 'gpu_id': 1,
            'worker_name': 'gpu_1', 'output': str(self.output / 'workers/gpu_1'), 'deadline_seconds': 480}
        self.task_path = self.stage / 'tasks/gpu_1.json'; put(self.task_path, self.task)
        worker = {'name': 'gpu_1', 'gpu_id': 1, 'cpu_set': '98,99', 'command': [sys.executable,
            str(self.source / 'infra/gpu03/direction_discovery/engine.py'), '--task', str(self.task_path)],
            'success_file': 'workers/gpu_1/SUCCESS.json', 'success_expect': {'mode': 'qualify', 'requests': 1}}
        self.m = {'schema_version': 1, 'purpose': 'direction_discovery_campaign', 'phase': 'remote_causal_qualification',
            'run_token': self.token, 'host': self.host, 'owner': 'researcher', 'stage': str(self.stage),
            'source_root': str(self.source), 'output': str(self.output), 'runtime': str(self.runtime), 'python': sys.executable,
            'supervisor_cpu_set': '94', 'gpu_ids': [1], 'gpu_uuids': self.gpus, 'authorization': 'Authored fixture, no real launch',
            'scientific': {'training': False, 'master_plan_sha256': r.MASTER, 'input_prepared_sha256': self.prepared['sha256'],
                'previous_phase_budget': r._budget_snapshot(self.ledger)}, 'runtime_versions': self.profile['runtime_versions'],
            'workers': [worker], 'limits': {'runtime_seconds': 600, 'systemd_runtime_seconds': 780,
                'min_available_ram_gib': 192, 'max_worker_rss_gib': 128, 'cgroup_memory_gib': 160,
                'min_start_free_disk_gib': 64, 'max_worker_log_mib': 64, 'tasks_max': 512, 'load_stagger_seconds': 10},
            'remote_generation': {'protocol': self.protocol, 'authority': authority_ref, 'profile': profile_ref},
            'bound_files': {str(p): r.info(p) for p in self.stage.rglob('*') if p.is_file()}}
        self.m['bound_files'].update({x['remote']: {k: x[k] for k in ('sha256', 'size_bytes')} for x in mapping})
        self.m['bound_files'][self.prepared['path']] = self.authority['prepared']
        self.m['command'] = [sys.executable, str(self.source / 'infra/gpu03/direction_discovery/supervisor.py'), '--supervise', '--manifest', str(self.manifest_path)]
        self.freeze()

    def freeze(self):
        self.deployment_ref = put(self.manifest_path, self.m)
        self.files = {name: {'path': name, **entry} for name, entry in self.m['bound_files'].items()}
        self.files[str(self.manifest_path)] = {'path': str(self.manifest_path), **r.info(self.manifest_path)}

    def mutate_task(self, key, value):
        self.task[key] = value; put(self.task_path, self.task)
        self.m['bound_files'][str(self.task_path)] = r.info(self.task_path); self.freeze()

    def commit(self):
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'):
            self.permit = r.commit(self.root, self.deployment_ref, self.files)
        return self.permit

    def complete(self):
        permit = self.commit()
        digest = self.deployment_ref['sha256']; identity = {'run_token': self.token, 'manifest_sha256': digest}
        invocation = 'a' * 32; cg = '/user.slice/' + self.token + '.service'
        c = self.stage / 'control'
        put(c / 'launch_intent.json', {**identity, 'at': 100.})
        put(c / 'service_started.json', {'at': 101., 'unit': self.token + '.service', 'fields': {
            'ActiveState': 'active', 'SubState': 'running', 'MainPID': '321', 'ControlGroup': cg,
            'KillMode': 'control-group', 'InvocationID': invocation}})
        put(c / 'supervisor_exit.json', {**identity, 'at': 120., 'service_result': 'success', 'exit_code_kind': 'exited',
            'exit_status': '0', 'invocation_id': invocation, 'producer_summary_present': True, 'failure_present': False})
        put(self.output / 'reviewed_manifest.json', self.manifest_path.read_bytes())
        w = self.m['workers'][0]; expected = {'status': 'succeeded', 'run_token': self.token, 'worker_name': w['name'], **w['success_expect']}
        success = self.output / w['success_file']; put(success, expected)
        # Opaque authored bytes prove metadata context does not parse outcomes.
        put(success.parent / 'results.jsonl', b'opaque outcome fixture; never parsed here\n')
        put(self.output / 'campaign_summary.json', {**identity, 'status': 'succeeded', 'gpu_release_verified': True,
            'worker_exit_codes': [0], 'worker_receipts': [{'worker': w['name'], 'path': w['success_file'], 'sha256': r.sha(success), 'expected': expected}]})
        put(self.output / 'gpu_release.json', {'verified': True, 'gpu_ids': [1]})
        put(self.output / 'artifact_manifest.json', {'algorithm': 'sha256', 'files': {
            str(p.relative_to(self.output)): r.info(p) for p in self.output.rglob('*') if p.is_file()}})
        proof = {'protocol': self.protocol, 'status': 'independently_verified_remote_generation', 'host': self.host,
            **identity, 'permit_sha256': permit['sha256'], 'artifact_manifest_sha256': r.sha(self.output / 'artifact_manifest.json'),
            'invocation_id': invocation, 'control_group': cg, 'service_active_state': 'inactive', 'main_pid': 0,
            'own_cgroup_empty': True, 'gpu_release_verified': True, 'supervisor_verification': {'status': 'verified', **identity, 'workers': 1, 'artifact_files': 5, 'gpu_release_verified': True}}
        proof_ref = put(self.root / 'imported/remote_verification.json', proof)
        files = {**self.files, **{str(p): {'path': str(p), **r.info(p)} for folder in (c, self.output) for p in folder.rglob('*') if p.is_file()}}
        self.replica = {'protocol': self.protocol, 'origin_host': self.host, 'authority_host': 'gpu-04',
                        'permit': permit, 'remote_verification': proof_ref, 'files': files}
        self.package = {'kind': 'remote_generation', 'replica': put(self.root / 'imported/replica.json', self.replica)}
        return self.package


class RemoteGenerationTests(unittest.TestCase):
    HOST = "gpu-02"
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.root_patch = patch.object(r, 'AUTHORITY_ROOT', self.root); self.root_patch.start(); self.template_patch = patch.object(r, 'verify_original_template'); self.template_patch.start(); self.schema_patch = patch.object(r.helper('supervisor'), 'RUN_ROOT', self.root / self.HOST.replace('-', '')); self.schema_patch.start(); self.f = Fixture(self.root, self.HOST)
        self.config_patch = patch.object(r, 'CONFIG_BINDINGS', {n:r.info(self.root/'authority/model'/n) for n in ('config.json','generation_config.json')}); self.config_patch.start()
    def tearDown(self): self.config_patch.stop(); self.schema_patch.stop(); self.template_patch.stop(); self.root_patch.stop(); self.tmp.cleanup()

    def test_valid_metadata_replays_original_task_science(self):
        ctx = r.validate_manifest(self.f.m, check_dependencies=True)
        self.assertEqual(ctx['tasks'][0]['task']['conditions'], self.f.plan['conditions'])
        self.assertEqual(ctx['prepared_records_sha256'], self.f.prepared['sha256'])

    def test_rebound_seed_request_or_sampling_changes_rejected(self):
        for key, value in [('requests', [{**self.f.request, 'seed': 8}]), ('sampling', {'temperature': 1.}),
                           ('attention_policy', 'flash'), ('teacher_forced_padded_sequence_length', 1536)]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.f.task); self.f.mutate_task(key, value)
                with self.assertRaises(RuntimeError): r.validate_manifest(self.f.m)
                self.f.task = original; put(self.f.task_path, original)
                self.f.m['bound_files'][str(self.f.task_path)] = r.info(self.f.task_path); self.f.freeze()

    def test_conditions_not_normalized_or_replaced(self):
        self.f.mutate_task('conditions', {'baseline': {'layers': [{'kind': 'random'}]}})
        with self.assertRaisesRegex(RuntimeError, 'conditions'): r.validate_manifest(self.f.m)

    def test_unmapped_model_directory_rejected(self):
        self.f.mutate_task('model_snapshot', '/another/model')
        with self.assertRaisesRegex(RuntimeError, 'model directory'): r.validate_manifest(self.f.m)

    def test_remote_mode_training_or_tf_rejected(self):
        self.f.mutate_task('mode', 'tf')
        with self.assertRaises(RuntimeError): r.validate_manifest(self.f.m)

    def test_qualification_count_charged_three_and_three_before_launch(self):
        self.f.commit(); result = budget.account(self.root, r.MASTER, r.PARENT)
        self.assertEqual((result['generation_requests'], result['tf_requests']), (3, 3))
        self.assertEqual(result['gpu_phase_wall_seconds'], 780)
        self.assertEqual(result['phases'][0]['host'], self.f.host)
        self.assertEqual(result['phases'][0]['gpu_count'], 1)
        self.assertFalse((self.f.stage / 'control').exists())

    def test_duplicate_commit_is_never_refunded_or_reissued(self):
        self.f.commit()
        with self.assertRaises(RuntimeError): self.f.commit()
        self.assertEqual(budget.account(self.root, r.MASTER, r.PARENT)['generation_requests'], 3)

    def test_stale_ledger_and_wall_cap_fail_before_reservation(self):
        ctx = r.validate_manifest(self.f.m); ctx['manifest_sha256'] = self.f.deployment_ref['sha256']
        for change in ({'generation_requests': 1}, {'gpu_phase_wall_seconds': 28500.}):
            ledger = {**self.f.ledger, **change}
            with self.assertRaises(RuntimeError): r.validate_budget(ctx, ledger)
        self.assertFalse((self.root / r.REGISTRY).exists())

    def test_gpu_concurrency_ambiguous_phase_remains_reserved(self):
        ledger = copy.deepcopy(self.f.ledger)
        ledger['phases'] = [{'run_token': 'old', 'manifest_sha256': 'a'*64, 'gpu_count': 8, 'wall_basis': 'full_reserved_deadline'}]
        self.f.m['scientific']['previous_phase_budget'] = r._budget_snapshot(ledger)
        ctx = r.validate_manifest(self.f.m); ctx['manifest_sha256'] = self.f.deployment_ref['sha256']
        with self.assertRaisesRegex(RuntimeError, 'concurrency'): r.validate_budget(ctx, ledger)

    def test_one_use_permit_cannot_be_consumed_twice(self):
        permit = self.f.commit()
        with patch.object(r.socket, 'gethostname', return_value=self.f.host):
            r.validate_launch_permit(self.f.m, self.f.deployment_ref['sha256'], permit['path'], permit['sha256'], consume=True)
            r.validate_consumption(self.f.m, self.f.deployment_ref['sha256'])
            with self.assertRaises(FileExistsError):
                r.validate_launch_permit(self.f.m, self.f.deployment_ref['sha256'], permit['path'], permit['sha256'], consume=True)

    def test_wrong_host_or_uuid_permit_rejected(self):
        permit = self.f.commit()
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'):
            with self.assertRaisesRegex(RuntimeError, 'only execute'): r.validate_launch_permit(self.f.m, self.f.deployment_ref['sha256'], permit['path'], permit['sha256'])
        changed = copy.deepcopy(self.f.m); changed['gpu_ids'] = [2]
        with patch.object(r.socket, 'gethostname', return_value=self.f.host):
            with self.assertRaises(RuntimeError): r.validate_launch_permit(changed, self.f.deployment_ref['sha256'], permit['path'], permit['sha256'], consume=True)

    def test_full_terminal_context_exposes_only_verified_opaque_result_paths(self):
        package = self.f.complete(); ctx = r.context(package)
        self.assertEqual(ctx['proof']['actual_wall_seconds'], 20.)
        self.assertEqual(ctx['results'][0]['path'].read_bytes(), b'opaque outcome fixture; never parsed here\n')
        self.assertEqual(ctx['proof']['manifest_sha256'], self.f.deployment_ref['sha256'])

    def test_success_import_reduces_wall_only_without_request_refund(self):
        package = self.f.complete()
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'):
            r.import_completion(self.root, package)
        result = budget.account(self.root, r.MASTER, r.PARENT)
        self.assertEqual((result['generation_requests'], result['tf_requests']), (3, 3))
        self.assertEqual(result['gpu_phase_wall_seconds'], 20.)
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'), self.assertRaises(FileExistsError):
            r.import_completion(self.root, package)

    def test_remote_output_snapshot_changed_fails_closed(self):
        package = self.f.complete()
        path = self.f.output / 'workers/gpu_1/results.jsonl'; put(path, b'changed')
        with self.assertRaisesRegex(RuntimeError, 'bytes changed'): r.context(package)
        self.assertEqual(budget.account(self.root, r.MASTER, r.PARENT)['gpu_phase_wall_seconds'], 780.)

    def test_missing_artifact_or_result_cannot_be_silently_omitted(self):
        self.f.complete(); del self.f.replica['files'][str(self.f.output / 'workers/gpu_1/results.jsonl')]
        package = {'kind': 'remote_generation', 'replica': put(self.root / 'omitted.json', self.f.replica)}
        with self.assertRaisesRegex(RuntimeError, 'no explicit replica'): r.context(package)

    def test_disagreeing_external_invocation_rejected(self):
        package = self.f.complete(); proof_ref = self.f.replica['remote_verification']; proof = r.read(proof_ref)
        proof['invocation_id'] = 'b'*32
        self.f.replica['remote_verification'] = put(self.root / 'wrong-proof.json', proof)
        package = {'kind': 'remote_generation', 'replica': put(self.root / 'wrong-replica.json', self.f.replica)}
        with self.assertRaisesRegex(RuntimeError, 'terminal/release'): r.context(package)

    def test_missing_or_false_terminal_release_never_reduces_charge(self):
        self.f.complete(); proof = r.read(self.f.replica['remote_verification']); proof['own_cgroup_empty'] = False
        self.f.replica['remote_verification'] = put(self.root / 'unreleased.json', proof)
        package = {'kind': 'remote_generation', 'replica': put(self.root / 'unreleased-replica.json', self.f.replica)}
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'), self.assertRaises(RuntimeError):
            r.import_completion(self.root, package)
        self.assertFalse((self.root / r.REGISTRY / self.f.token / 'completion.json').exists())
        self.assertEqual(budget.account(self.root, r.MASTER, r.PARENT)['gpu_phase_wall_seconds'], 780.)

    def test_unfinished_registry_admission_is_fail_closed(self):
        draft = self.root / r.REGISTRY / 'codex-unfinished'; draft.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete remote admission'): budget.account(self.root, r.MASTER, r.PARENT)

    def test_publication_lock_is_owned_bounded_and_mutually_exclusive(self):
        with budget.publication_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                with budget.publication_lock(self.root, timeout=.01): pass
        with budget.publication_lock(self.root): pass

    def test_alternate_authority_root_cannot_reset_budget(self):
        other = self.root / 'fresh-empty-ledger'; other.mkdir()
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'):
            with self.assertRaisesRegex(RuntimeError, 'canonical GPU04 ledger'):
                r.commit(other, self.f.deployment_ref, self.f.files)
        self.assertEqual(list(other.iterdir()), [])

    def test_model_files_cannot_move_outside_actual_loaded_directory(self):
        profile = copy.deepcopy(self.f.profile)
        profile['model_paths']['model_snapshot'] = str(self.root / 'different-only-config')
        profile_ref = put(self.root / 'wrong-model-profile.json', profile)
        self.f.m['remote_generation']['profile'] = profile_ref
        self.f.m['bound_files'][profile_ref['path']] = r.info(profile_ref['path'])
        self.f.mutate_task('model_snapshot', profile['model_paths']['model_snapshot'])
        with self.assertRaisesRegex(RuntimeError, 'no bound files|relative paths'):
            r.validate_manifest(self.f.m)

    def test_generation_requires_preexisting_positive_host_qualification(self):
        plan = copy.deepcopy(self.f.plan); plan['mode'] = 'generate'; plan['evaluation_partition'] = 'configuration_validation'
        plan['master_plan_sha256'] = r.MASTER
        self.f.authority['request_plan'] = put(self.root / 'generation-plan.json', plan)
        authority_ref = put(self.root / 'generation-authority.json', self.f.authority)
        self.f.m['remote_generation']['authority'] = authority_ref
        self.f.m['bound_files'][authority_ref['path']] = r.info(authority_ref['path'])
        plan_path = self.f.stage / 'input/request_plan.json'; put(plan_path, plan)
        self.f.m['bound_files'][str(plan_path)] = r.info(plan_path)
        self.f.m['phase'] = 'behavior_finalist_validation_round1'
        self.f.mutate_task('mode', 'generate')
        self.f.m['workers'][0]['success_expect']['mode'] = 'generate'
        with self.assertRaisesRegex(RuntimeError, 'host inference qualification'):
            r.validate_manifest(self.f.m)

    def test_local_portable_preparation_success_does_not_publish_permit(self):
        token = 'codex-discovery-remote-preparation-20260907'
        spec = {'protocol': self.f.protocol, 'launch_enabled': False,
            'local_output': str(self.root / (token + '-portable-preparation')),
            'remote_stage': str(self.root / (token + '-stage')),
            'authority': self.f.m['remote_generation']['authority'], 'profile': self.f.m['remote_generation']['profile'],
            'source_root': str(self.f.source), 'gpu_ids': [1], 'runtime_seconds': 600, 'authorization': 'Authored no-launch fixture'}
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'), patch.object(r.helper('supervisor'), 'RUN_ROOT', self.root):
            prepared = r.read(r.prepare_deployment(spec))
        self.assertEqual(prepared['status'], 'prepared_without_permit_or_launch')
        self.assertEqual(prepared['counts']['generation_requests'], 3)
        self.assertFalse((self.root / r.REGISTRY).exists())
        self.assertFalse(Path(spec['remote_stage']).exists())
        with patch.object(r.helper('supervisor'), 'RUN_ROOT', self.root):
            ctx = r.validate_manifest(json.loads(r.resolve(prepared['files'], prepared['deployment_manifest']['path']).read_bytes()), files=prepared['files'])
        self.assertEqual(ctx['tasks'][0]['task']['requests'], self.f.plan['requests'])

    def test_static_deadline_rejection_precedes_any_portable_output(self):
        token = 'codex-discovery-too-small-20260907'
        spec = {'protocol': self.f.protocol, 'launch_enabled': False,
            'local_output': str(self.root / (token + '-portable-preparation')), 'remote_stage': str(self.root / (token + '-stage')),
            'authority': self.f.m['remote_generation']['authority'], 'profile': self.f.m['remote_generation']['profile'],
            'source_root': str(self.f.source), 'gpu_ids': [1], 'runtime_seconds': 120, 'authorization': 'Authored'}
        with patch.object(r.socket, 'gethostname', return_value='gpu-04'), self.assertRaisesRegex(RuntimeError, 'allocation/runtime'):
            r.prepare_deployment(spec)
        self.assertFalse(Path(spec['local_output']).exists())

    def test_unknown_or_collected_systemd_status_is_not_accepted_without_positive_fields(self):
        from types import SimpleNamespace
        package = self.f.complete(); ctx = r.context(package)
        fake_sup = SimpleNamespace(load_manifest=lambda *_: self.f.m, verify=lambda *_: ctx['proof'],
            raw=SimpleNamespace(check_devices=lambda *_: None, gpu_snapshot=lambda: []))
        wrong = SimpleNamespace(returncode=1, stdout='LoadState=not-found\n', stderr='')
        actual_helper = r.helper
        with patch.object(r, 'helper', side_effect=lambda n: fake_sup if n == 'supervisor' else actual_helper(n)), \
             patch.object(r, 'validate_launch_permit'), patch('subprocess.run', return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, 'not positively terminal'):
                r.verify_remote(self.f.manifest_path, self.f.deployment_ref['sha256'], self.f.permit['path'], self.f.permit['sha256'])
        good = SimpleNamespace(returncode=1, stdout='LoadState=not-found\nActiveState=inactive\nMainPID=0\nControlGroup=\nInvocationID=\n', stderr='')
        with patch.object(r, 'helper', side_effect=lambda n: fake_sup if n == 'supervisor' else actual_helper(n)), \
             patch.object(r, 'validate_launch_permit'), patch('subprocess.run', return_value=good):
            value = r.verify_remote(self.f.manifest_path, self.f.deployment_ref['sha256'], self.f.permit['path'], self.f.permit['sha256'])
        self.assertTrue(value['own_cgroup_empty'])

    def test_behavior_recount_counts_one_physical_remote_manifest_despite_copies(self):
        from infra.gpu03.direction_discovery import behavior_plan
        self.f.complete()
        result = behavior_plan.counts_from_phase_ledger(self.root, r.MASTER, r.PARENT)
        self.assertEqual((result['generations'], result['teacher_forced']), (3, 3))
        self.assertEqual(len(result['phase_manifests']), 1)
        self.assertEqual(result['phase_manifests'][0]['path'], str(self.root / r.REGISTRY / self.f.token / 'reviewed_manifest.json'))

    def test_test_positive_gate_precedes_any_authority_payload_hash(self):
        authority = copy.deepcopy(self.f.authority)
        authority['files'] = {'/must-not-read-test-payload': {'sha256': 'a'*64, 'size_bytes': 100}}
        plan = {'mode': 'generate', 'test_execution': {'bundle': {'path': '/unresolved'}}}
        test_module = r.helper('test_execution')
        with patch.object(test_module, 'validate_part', side_effect=RuntimeError('positive freeze is absent')), \
             patch.object(r, 'check_file') as checked:
            with self.assertRaisesRegex(RuntimeError, 'positive freeze'):
                r.validate_authority_dependencies(authority, plan, self.f.profile)
        checked.assert_not_called()

    def test_original_qualified_model_mode_can_be_writable_but_content_cannot_change(self):
        model = Path(next(iter(self.f.authority['model_files'])))
        model.chmod(0o600)
        r.validate_authority_dependencies(self.f.authority, self.f.plan, self.f.profile)
        model.write_bytes(b'changed-model')
        with self.assertRaisesRegex(RuntimeError, 'model content'):
            r.validate_authority_dependencies(self.f.authority, self.f.plan, self.f.profile)

    def test_original_qualification_pin_cannot_be_replaced_by_another_claim(self):
        self.template_patch.stop()
        with self.assertRaisesRegex(RuntimeError, 'qualification identity'):
            r.verify_original_template(self.f.authority, r.read(self.f.authority['template_manifest']))
        self.template_patch.start()

    def test_rebound_non_success_terminal_is_rejected_before_import(self):
        self.f.complete()
        path = self.f.stage / 'control/supervisor_exit.json'; end = json.loads(path.read_text())
        end.update(service_result='signal', exit_code_kind='killed', exit_status='TERM', producer_summary_present=False, failure_present=True)
        put(path, end); self.f.replica['files'][str(path)] = {'path': str(path), **r.info(path)}
        package = {'kind': 'remote_generation', 'replica': put(self.root / 'failed-replica.json', self.f.replica)}
        with self.assertRaisesRegex(RuntimeError, 'not positively successful'): r.context(package)
        self.assertEqual(budget.account(self.root, r.MASTER, r.PARENT)['gpu_phase_wall_seconds'], 780.)

    def test_historical_ten_model_files_gain_exact_two_config_bindings(self):
        original = r.read(self.f.authority['template_manifest'])
        self.assertEqual(len(original['bound_files']) - 1, 10)
        self.assertEqual(r.expected_model_files(original, self.f.template), self.f.authority['model_files'])
        r.validate_authority_dependencies(self.f.authority, self.f.plan, self.f.profile)

    def test_omitted_or_unreviewed_extra_config_extension_rejected(self):
        for kind in ('missing', 'extra'):
            authority = copy.deepcopy(self.f.authority)
            if kind == 'missing': authority['model_files'].pop(str(self.root/'authority/model/config.json'))
            else: authority['model_files'][str(self.root/'authority/model/unreviewed.json')] = {'sha256':'a'*64,'size_bytes':1}
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                r.validate_authority_dependencies(authority, self.f.plan, self.f.profile)

    def test_conflicting_historically_bound_config_fails_parity_join(self):
        original = r.read(self.f.authority['template_manifest'])
        original['bound_files'][str(self.root/'authority/model/config.json')] = {'sha256':'a'*64,'size_bytes':726}
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            r.expected_model_files(original, self.f.template)

    def test_direct_cli_import_without_pythonpath_has_no_model_side_effect(self):
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, r.__file__, '--help'], cwd=self.root, env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--verify-remote', result.stdout)


class GPU01RemoteGenerationTests(RemoteGenerationTests):
    """Replay every existing transport boundary for GPU01; no model calls."""
    HOST = 'gpu-01'

    def test_host_allowlist_and_legacy_protocol_are_explicit(self):
        self.assertEqual(r.execution_protocol('gpu-02'), r.PROTOCOL)
        self.assertEqual(r.execution_protocol('gpu-01'), 'gpu04_authority_gpu01_generation_v1')
        for host in ('gpu-03', 'HPC', '192.0.2.10', None):
            with self.subTest(host=host), self.assertRaisesRegex(RuntimeError, 'execution host'):
                r.execution_protocol(host)

    def test_manifest_profile_and_authority_cannot_cross_hosts(self):
        changed = copy.deepcopy(self.f.m); changed['host'] = 'gpu-02'
        with self.assertRaisesRegex(RuntimeError, 'protocol'): r.validate_manifest(changed)
        changed = copy.deepcopy(self.f.m); profile = copy.deepcopy(self.f.profile)
        profile.update(host='gpu-02', protocol=r.PROTOCOL)
        binding = put(self.root/'other-host-profile.json', profile)
        changed['remote_generation']['profile'] = binding; changed['bound_files'][binding['path']] = r.info(binding['path'])
        with self.assertRaisesRegex(RuntimeError, 'profile'): r.validate_manifest(changed)
        changed = copy.deepcopy(self.f.m); authority = copy.deepcopy(self.f.authority)
        authority['protocol'] = r.PROTOCOL
        binding = put(self.root/'other-host-authority.json', authority)
        changed['remote_generation']['authority'] = binding; changed['bound_files'][binding['path']] = r.info(binding['path'])
        with self.assertRaisesRegex(RuntimeError, 'authority'): r.validate_manifest(changed)

    def test_lifetime_proof_must_belong_to_execution_host(self):
        authority, profile = copy.deepcopy(self.f.authority), copy.deepcopy(self.f.profile)
        proof = r.read(profile['lifetime_qualification']); proof['host'] = 'gpu-02'
        binding = put(self.root/'other-host-lifetime.json', proof)
        profile['lifetime_qualification'] = binding; authority['files'][binding['path']] = r.info(binding['path'])
        with self.assertRaisesRegex(RuntimeError, 'disconnect survival'):
            r.validate_authority_dependencies(authority, self.f.plan, profile)

    def test_permit_host_join_and_socket_reject_other_allowed_host(self):
        permit_ref = self.f.commit()
        with patch.object(r.socket, 'gethostname', return_value='gpu-02'), self.assertRaisesRegex(RuntimeError, 'bound host'):
            r.validate_launch_permit(self.f.m,self.f.deployment_ref['sha256'],permit_ref['path'],permit_ref['sha256'],consume=True)
        self.assertFalse((self.f.stage/'remote_permit_consumed.json').exists())
        permit = r.read(permit_ref); permit.update(execution_host='gpu-02',protocol=r.PROTOCOL)
        rebound = put(Path(permit_ref['path']), permit)
        with self.assertRaisesRegex(RuntimeError, 'identity differs'): r.permit_context(rebound)

    def test_replica_cannot_relabel_another_hosts_permit(self):
        self.f.complete(); replica = copy.deepcopy(self.f.replica)
        replica.update(origin_host='gpu-02',protocol=r.PROTOCOL)
        package={'kind':'remote_generation','replica':put(self.root/'other-host-replica.json',replica)}
        with self.assertRaisesRegex(RuntimeError,'Replica host differs'):r.context(package)

    def test_remote_terminal_proof_host_and_protocol_are_joined(self):
        self.f.complete(); replica = copy.deepcopy(self.f.replica)
        proof = r.read(replica['remote_verification']); proof.update(host='gpu-02',protocol=r.PROTOCOL)
        replica['remote_verification']=put(self.root/'other-host-proof.json',proof)
        package={'kind':'remote_generation','replica':put(self.root/'other-proof-replica.json',replica)}
        with self.assertRaisesRegex(RuntimeError,'terminal/release'):r.context(package)

    def test_inference_qualification_rejects_same_bytes_from_other_host_before_payload(self):
        self.f.complete(); ctx=r.context(self.f.package)
        ctx['manifest']=copy.deepcopy(ctx['manifest']);ctx['manifest']['host']='gpu-02'
        ctx['profile']=copy.deepcopy(ctx['profile']);ctx['profile'].update(host='gpu-02',protocol=r.PROTOCOL)
        with patch.object(r,'check_file') as checked,self.assertRaisesRegex(RuntimeError,'another execution host'):
            r.validate_inference_qualification(ctx,self.f.profile)
        checked.assert_not_called()

    def test_same_host_inference_qualification_parses_authored_checks(self):
        self.f.complete();ctx=r.context(self.f.package)
        energy={'rank':1,'selected_tokens':8,'forward_calls':8,'scopes':{'prefill':{'selected_tokens':1},'decode':{'selected_tokens':7}},
                'removed_energy_fp32':1.,'remaining_subspace_energy_fp32':0.,'activation_energy':2.}
        result={'baseline_recovery_bitwise':True,'teacher_forced_effect_verified':True,'baseline_generation_repeatable':True,
                'baseline_generation':{'generated_token_ids':list(range(8))},'repeated_baseline_generation':{'generated_token_ids':list(range(8))},
                'projected_generation':{'generated_token_ids':list(range(8)),'energy':{'12':energy}}}
        row={**self.f.request,'problem_split':'direction_fit','result':result}
        path=self.root/'authored-qualification-result.jsonl';put(path,r.canonical(row).encode())
        ctx['results']=[{'path':path,**r.info(path)}]
        r.validate_inference_qualification(ctx,self.f.profile)


if __name__ == '__main__': unittest.main()
