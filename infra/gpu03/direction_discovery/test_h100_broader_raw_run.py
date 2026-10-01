"""Real native/probe serialization tests for the raw-only independent audit."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import fixed_cache as fixed
from . import h100_broader_raw_run as run


class RawMonitorRaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.raw = self.root / 'raw'; self.raw.mkdir()
        self.plan = {'raw_output': str(self.raw), 'limits': {'minimum_free_raw_bytes': 0, 'maximum_raw_bytes': 100}}
        self.patches = [patch.object(run, 'RAWBASE', self.root), patch.object(run.shared, 'UID', os.getuid())]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_atomic_rename_between_discovery_and_stat_is_tolerated_then_counted(self):
        writing = self.raw / 'record_000001.writing'; writing.write_bytes(b'x' * 11)
        final = writing.with_suffix('.safetensors')
        (self.raw / 'stable.safetensors').write_bytes(b'y' * 7)
        original = Path.stat
        def stat_after_rename(path, *, follow_symlinks=True):
            if path == writing and not follow_symlinks:
                writing.rename(final)
            return original(path, follow_symlinks=follow_symlinks)
        with patch.object(Path, 'stat', stat_after_rename):
            self.assertEqual(run.observe_raw(self.plan)['raw_bytes'], 7)
        self.assertEqual(run.observe_raw(self.plan)['raw_bytes'], 18)

    def test_permission_error_is_not_treated_as_rename(self):
        writing = self.raw / 'record_000001.writing'; writing.write_bytes(b'x')
        original = Path.stat
        def denied(path, *, follow_symlinks=True):
            if path == writing and not follow_symlinks: raise PermissionError('authored denial')
            return original(path, follow_symlinks=follow_symlinks)
        with patch.object(Path, 'stat', denied), self.assertRaises(PermissionError):
            run.observe_raw(self.plan)

    def test_existing_symlink_is_rejected(self):
        target = self.root / 'target'; target.write_bytes(b'x')
        (self.raw / 'linked.safetensors').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'raw symlink'):
            run.observe_raw(self.plan)

    def test_symlink_replacement_after_check_is_rejected_without_following(self):
        writing = self.raw / 'record_000001.writing'; writing.write_bytes(b'x')
        target = self.root / 'target'; target.write_bytes(b'y')
        original_stat, original_link = Path.stat, Path.is_symlink
        checked = set()
        def checked_link(path):
            value = original_link(path)
            if path == writing: checked.add(path)
            return value
        def replaced(path, *, follow_symlinks=True):
            if path == writing and not follow_symlinks and path in checked:
                writing.unlink(); writing.symlink_to(target)
            return original_stat(path, follow_symlinks=follow_symlinks)
        with patch.object(Path, 'is_symlink', checked_link), patch.object(Path, 'stat', replaced):
            with self.assertRaisesRegex(ValueError, 'ceased to be a regular file'):
                run.observe_raw(self.plan)

    def test_byte_cap_includes_writing_files(self):
        (self.raw / 'record.writing').write_bytes(b'x' * 101)
        with self.assertRaisesRegex(ValueError, 'raw byte cap'):
            run.observe_raw(self.plan)

    def test_free_floor_still_fails_before_snapshot(self):
        self.plan['limits']['minimum_free_raw_bytes'] = 1 << 100
        with self.assertRaisesRegex(ValueError, 'NVMe reserve'):
            run.observe_raw(self.plan)


class RecoveryJoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.raw = self.root / 'nvme'; self.raw.mkdir()
        self.oldout = self.root / run.ORIGINAL_TOKEN; self.oldout.mkdir()
        self.patches = [patch.object(run, 'RAWBASE', self.raw), patch.object(run.shared, 'UID', os.getuid()),
                        patch.object(run.core, 'ROOT', self.root), patch.object(run.core, 'UID', os.getuid())]
        for p in self.patches: p.start()
        prepared = self.root / 'prepared.jsonl'; prepared.write_text('{}\n')
        model, checkpoint = self.root / 'model', self.root / 'checkpoint'
        model.mkdir(); checkpoint.mkdir()
        (model / 'weights').write_bytes(b'model'); (checkpoint / 'weights').write_bytes(b'adapter')
        bound = {str(p): {k: v for k, v in run.core.file_ref(p).items() if k != 'path'}
                 for p in (prepared, model / 'weights', checkpoint / 'weights')}
        self.oldtask = {'run_token': run.ORIGINAL_TOKEN, 'worker_name': 'worker_00', 'gpu_id': 0,
                        'record_ids': ['r0', 'r1'], 'output': str(self.oldout / 'worker_00'),
                        'raw_output': str(self.raw / run.ORIGINAL_RAW_NAME / 'worker_00'),
                        'prepared_records': str(prepared), 'prepared_records_sha256': bound[str(prepared)]['sha256'],
                        'model_snapshot': str(model), 'checkpoint': str(checkpoint), 'deadline_seconds': 3600,
                        'padded_sequence_length': 2688, 'pad_token_id': 151643, 'gpu_memory_fraction': .65,
                        'manifest_path': str(self.root / 'parent.json')}
        oldtaskref = self.write(self.root / 'oldtask.json', self.oldtask)
        self.parent = {'run_token': run.ORIGINAL_TOKEN, 'output': str(self.oldout),
                       'raw_output': str(self.raw / run.ORIGINAL_RAW_NAME), 'protocol': run.PROTOCOL,
                       'mode': 'broader_raw', 'host': 'authored', 'uid': os.getuid(), 'records': 500,
                       'padded_sequence_length': 2688, 'record_ids_sha256': 'authored-population',
                       'gpus': [{'id': 0, 'uuid': 'authored'}], 'python': {'path': 'authored-python'},
                       'numerical_policy': {'attention': 'exclusive_math'}, 'expected_raw_tensor_bytes': 1000,
                       'bound_files': bound,
                       'workers': [{'name': 'worker_00', 'gpu_id': 0, 'cpu_set': [0, 1], 'task': oldtaskref}]}
        parentref = self.write(self.root / 'parent.json', self.parent)
        self.parent_sha = parentref['sha256']
        p = patch.object(run, 'PARENT_PLAN_SHA', self.parent_sha); p.start(); self.patches.append(p)
        native = Path(self.oldtask['raw_output']) / 'h0/record_000000.safetensors'
        native.parent.mkdir(parents=True); native.write_bytes(b'authored closed bytes'); native.chmod(0o400)
        self.origin = {**run.core.file_ref(native), 'source_manifest_sha256': self.parent_sha,
                       'record_id': 'r0', 'record_index': 0, 'kind': 'h0'}
        failure = {'status': 'failed', 'plan_sha256': self.parent_sha, 'run_token': run.ORIGINAL_TOKEN,
                   'process_release_verified': True, 'gpu_release_verified': True,
                   'error': 'FileNotFoundError: record_000000.writing'}
        failref = self.write(self.oldout / 'RUN_FAILED.json', failure)
        artifact = {'plan_sha256': self.parent_sha, 'raw_local_root': self.parent['raw_output'], 'files': {
                    'RUN_FAILED.json': {k: failref[k] for k in ('sha256', 'size_bytes')},
                    'raw/worker_00/h0/record_000000.safetensors': {k: self.origin[k] for k in ('sha256', 'size_bytes')}}}
        artifactref = self.write(self.oldout / 'artifact_manifest.json', artifact)
        self.proof = {'status': 'independently_verified_partial_raw_after_monitor_failure',
                      'parent_plan_sha256': self.parent_sha, 'parent_plan': parentref,
                      'cgroup_processes': [], 'gpu_release_verified': True,
                      'native_metadata_dtype_shape_tokens_masks_finite_verified': True,
                      'service_fields': {'Id': run.ORIGINAL_TOKEN + '.service', 'ExecMainCode': '1',
                                         'ExecMainStatus': '1', 'Result': 'exit-code', 'MainPID': '0', 'SubState': 'failed'},
                      'failed_artifact_manifest': artifactref, 'failure': failure,
                      'reusable_native_files': 1, 'reusable_native_records': {'worker_00': {'h0': {'r0': self.origin}}}}
        self.plan = copy.deepcopy(self.parent)
        self.plan.update(run_token=run.RECOVERY_TOKEN, output=str(self.root / run.RECOVERY_TOKEN),
                         raw_output=str(self.raw / run.RECOVERY_RAW_NAME),
                         recovery={'parent_plan': parentref, 'partial_verification': {}})
        self.task = {**self.oldtask, 'run_token': run.RECOVERY_TOKEN,
                     'output': str(Path(self.plan['output']) / 'worker_00'),
                     'raw_output': str(Path(self.plan['raw_output']) / 'worker_00'),
                     'manifest_path': str(self.root / 'recovery.json'),
                     'reuse_native_records': {'h0': {'r0': self.origin}, 'h60': {}}}
        self.save_proof()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def write(self, path, value):
        if path.exists(): path.chmod(0o600)
        path.write_text(run.core.canonical(value)); path.chmod(0o400)
        return run.core.file_ref(path)

    def save_proof(self):
        self.plan['recovery']['partial_verification'] = self.write(self.root / 'proof.json', self.proof)

    def verify(self):
        return run.validate_recovery(self.plan, [self.task])

    def test_positive_exact_failure_and_normalized_empty_model_map(self):
        self.verify()
        del self.task['reuse_native_records']['h60']; self.verify()

    def test_no_recovery_and_wrong_parent_or_successor_reject(self):
        original = copy.deepcopy(self.plan)
        for change in ('missing', 'parent', 'token', 'root'):
            self.plan = copy.deepcopy(original)
            if change == 'missing': del self.plan['recovery']
            elif change == 'parent': self.plan['recovery']['parent_plan']['sha256'] = '0' * 64
            elif change == 'token': self.plan['run_token'] = run.ORIGINAL_TOKEN
            else: self.plan['raw_output'] += '-other'
            with self.subTest(change=change), self.assertRaises(ValueError): self.verify()

    def test_wrong_positive_release_and_failure_proof_reject(self):
        original = copy.deepcopy(self.proof)
        for change in ('pid', 'exit', 'group', 'gpu', 'native', 'failure'):
            self.proof = copy.deepcopy(original)
            if change == 'pid': self.proof['service_fields']['MainPID'] = '123'
            elif change == 'exit': self.proof['service_fields']['ExecMainStatus'] = '0'
            elif change == 'group': self.proof['cgroup_processes'] = [123]
            elif change == 'gpu': self.proof['gpu_release_verified'] = False
            elif change == 'native': self.proof['native_metadata_dtype_shape_tokens_masks_finite_verified'] = False
            else: self.proof['failure']['error'] = 'different error'
            self.save_proof()
            with self.subTest(change=change), self.assertRaises(ValueError): self.verify()

    def test_changed_shard_numerics_and_model_bytes_reject(self):
        original = copy.deepcopy(self.task)
        for key, value in [('record_ids', ['r1', 'r0']), ('deadline_seconds', 3601), ('pad_token_id', 0),
                           ('model_snapshot', str(self.root / 'other-model'))]:
            self.task = {**original, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'science/profile/shard'): self.verify()
        self.task = original
        self.plan['bound_files'][str(self.root / 'model/weights')]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'input binding'): self.verify()

    def test_omitted_or_cross_worker_reuse_and_count_reject(self):
        self.task['reuse_native_records'] = {}
        with self.assertRaisesRegex(ValueError, 'reuse differs'): self.verify()
        self.task['reuse_native_records'] = {'h0': {'r0': self.origin}}
        self.origin['path'] = self.origin['path'].replace('worker_00', 'worker_01')
        self.save_proof()
        with self.assertRaisesRegex(ValueError, 'exact failed worker'): self.verify()
        self.origin['path'] = self.origin['path'].replace('worker_01', 'worker_00')
        self.proof['reusable_native_files'] = 2; self.save_proof()
        with self.assertRaisesRegex(ValueError, 'file count'): self.verify()

    def test_changed_or_writable_original_native_rejects(self):
        p = Path(self.origin['path']); p.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'owned immutable'): self.verify()
        p.write_bytes(b'changed closed bytes'); p.chmod(0o400)
        with self.assertRaisesRegex(ValueError, 'source hash/size'): self.verify()


class NativeAuditTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_fixed_cache import NativeTests, row
        self.torch = torch
        self.fixture = NativeTests(); self.fixture.setUp()
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.raw = self.root / 'raw'; self.raw.mkdir()
        self.output = self.root / 'metadata'; self.output.mkdir()
        self.shard = self.raw / 'worker_00'; self.shard.mkdir()
        self.model = self.root / 'model'; self.model.mkdir()
        self.write(self.model / 'config.json', {'vocab_size': 256})
        self.patches = [patch.object(run, 'RAWBASE', self.raw), patch.object(run.shared, 'UID', os.getuid()),
                        patch.object(fixed, 'HIDDEN', 6)]
        for p in self.patches: p.start()
        self.rows = [row(0), row(1)]
        self.by_id = {r['record_id']: r for r in self.rows}
        self.task = {'run_token': 'authored-broader-audit', 'worker_name': 'worker_00',
                     'record_ids': [r['record_id'] for r in self.rows], 'output': str(self.output),
                     'raw_output': str(self.shard), 'model_snapshot': str(self.model)}
        self.sha = 'a' * 64
        self.entries, qualification = [], {}
        for kind in ('h0', 'h60'):
            (self.shard / kind).mkdir()
            model = self.fixture.decoder(shift=int(kind == 'h60'))
            for i, r in enumerate(self.rows):
                values, inputs, _ = fixed.capture_fixed(model, list(model.layers), r, hidden_size=6, padded_length=2688)
                path = self.shard / kind / f"record_{r['record_index']:06d}.safetensors"
                info = fixed.save_native(path, r, kind, self.sha, values, inputs, padded_length=2688)
                self.entries.append({'record_id': r['record_id'], 'record_index': r['record_index'],
                                     'kind': kind, 'path': str(path), **info})
                if i == 0:
                    qualification[kind] = fixed.qualify_model(model, list(model.layers), r, values, kind, self.sha,
                        self.shard / 'qualification' / kind, vocab_size=256, hidden_size=6, padded_length=2688)
                    qualification[kind].update(record_id=r['record_id'], first_production_raw_retained=True)
        self.receipt = {'status': 'succeeded', 'mode': 'broader_raw', 'run_token': self.task['run_token'],
                        'worker_name': 'worker_00', 'records': 2, 'record_ids': self.task['record_ids'],
                        'native_files': 4, 'manifest_sha256': self.sha, 'padded_sequence_length': 2688,
                        'raw_activations_retained': True, 'differences_computed': False,
                        'model_load_reports': {'h0': {'with_adapter': False},
                                              'h60': {'with_adapter': True, 'active_adapters': ['default'],
                                                      'nonzero_lora_parameter_tensors': 2}},
                        'qualifications': qualification}
        self.write(self.output / 'task.json', self.task)
        self.save_receipts()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.fixture.tearDown(); self.tmp.cleanup()

    def write(self, path, value):
        path.chmod(0o600) if path.exists() else None
        path.write_text(json.dumps(value, sort_keys=True))

    def save_receipts(self):
        self.write(self.output / 'SUCCESS.json', self.receipt)
        (self.output / 'native_index.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.entries))

    def verify(self, inspect_values=True):
        return run.validate_worker_outputs(self.task, self.by_id, self.sha, inspect_values=inspect_values)

    def change_tensor(self, path, key, transform):
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        with safe_open(str(path), framework='pt', device='cpu') as f: metadata = f.metadata()
        data = load_file(str(path)); data[key] = transform(data[key])
        path.chmod(0o600); save_file(data, str(path), metadata=metadata)

    def rebind_probe(self, which):
        folder = self.shard / 'qualification/h0'
        audit_path = folder / ('repeat_audit.json' if which == 'repeat' else 'future_causality_audit.json')
        audit = json.loads(audit_path.read_bytes())
        audit['artifact'].update(run.core.file_ref(folder / (which + '.safetensors')))
        self.write(audit_path, audit)
        if which != 'repeat': self.receipt['qualifications']['h0']['future_causality'] = audit
        self.save_receipts()

    def add_reused_native(self):
        from safetensors.torch import load_file, save_file
        row = self.rows[0]; entry = self.entries[0]
        origin = self.raw / run.ORIGINAL_RAW_NAME / 'worker_00/h0/record_000000.safetensors'
        origin.parent.mkdir(parents=True)
        save_file(load_file(entry['path']), str(origin),
                  metadata=fixed.metadata(row, 'h0', run.PARENT_PLAN_SHA, padded_length=2688))
        origin.chmod(0o400)
        ref = {**run.core.file_ref(origin), 'source_manifest_sha256': run.PARENT_PLAN_SHA,
               'record_id': row['record_id'], 'record_index': row['record_index'], 'kind': 'h0'}
        self.task['reuse_native_records'] = {'h0': {row['record_id']: ref}, 'h60': {}}
        entry.update(reused_native=True, reuse_origin=ref, reused_values_bitwise_preserved=True,
                     new_manifest_sha256=self.sha, capture_seconds=0.0)
        self.receipt.update(reused_native_files={'h0': 1, 'h60': 0}, fresh_native_captures={'h0': 1, 'h60': 2},
                            reuse_parent_manifest_sha256=run.PARENT_PLAN_SHA)
        self.write(self.output / 'task.json', self.task); self.save_receipts()
        return origin, ref

    def test_reused_native_real_serialization_keeps_values_with_new_manifest(self):
        self.add_reused_native()
        self.assertEqual(len(self.verify()), 4)
        self.assertEqual(len(self.verify(False)), 4)

    def test_rebound_new_native_hash_does_not_hide_changed_reused_values(self):
        self.add_reused_native()
        p = Path(self.entries[0]['path']); self.change_tensor(p, 'h0', lambda value: value + 1)
        self.entries[0].update(run.core.file_ref(p)); self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'values differ from the original'): self.verify()

    def test_reuse_counts_origin_and_old_auxiliary_are_checked(self):
        origin, ref = self.add_reused_native()
        self.receipt['reused_native_files']['h0'] = 0; self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'reuse counts'): self.verify(False)
        self.receipt['reused_native_files']['h0'] = 1
        self.entries[0]['new_manifest_sha256'] = '0' * 64; self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'provenance differs'): self.verify(False)
        self.entries[0]['new_manifest_sha256'] = self.sha
        self.change_tensor(origin, 'input_ids', lambda value: value + 1)
        origin.chmod(0o400); ref.update(run.core.file_ref(origin))
        self.write(self.output / 'task.json', self.task); self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'auxiliary dtype/shape/IDs'): self.verify()

    def test_real_native_and_probe_success(self):
        self.assertEqual(len(self.verify()), 4)
        self.assertEqual(len(self.verify(False)), 4)

    def test_success_manifest_record_count_and_native_count_are_joined(self):
        for key, bad in [('manifest_sha256', 'b' * 64), ('record_ids', ['other']), ('native_files', 3), ('records', 1)]:
            prior = self.receipt[key]; self.receipt[key] = bad; self.save_receipts()
            with self.assertRaisesRegex(ValueError, 'success contract'): self.verify(False)
            self.receipt[key] = prior
        self.save_receipts()

    def test_saved_task_and_failure_marker_reject(self):
        self.write(self.output / 'task.json', {**self.task, 'worker_name': 'worker_01'})
        with self.assertRaisesRegex(ValueError, 'saved task'): self.verify(False)
        self.write(self.output / 'task.json', self.task)
        self.write(self.output / 'FAILURE.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'failure exists'): self.verify(False)

    def test_same_bytes_at_wrong_record_path_are_rejected(self):
        old = Path(self.entries[0]['path']); other = old.with_name('other.safetensors'); shutil.copyfile(old, other)
        self.entries[0]['path'] = str(other); self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'exact record/model path'): self.verify(False)

    def test_native_auxiliary_equal_numbers_wrong_dtype_are_rejected(self):
        path = Path(self.entries[0]['path'])
        self.change_tensor(path, 'input_ids', lambda x: x.float())
        self.entries[0].update(run.core.file_ref(path)); self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'auxiliary dtype'): self.verify()

    def test_rebound_repeat_hash_cannot_hide_changed_activation(self):
        path = self.shard / 'qualification/h0/repeat.safetensors'
        self.change_tensor(path, 'post_block_selected', lambda x: x + 1)
        self.rebind_probe('repeat')
        with self.assertRaisesRegex(ValueError, 'recomputed tensors'): self.verify()

    def test_rebound_future_hash_cannot_hide_changed_past(self):
        path = self.shard / 'qualification/h0/future_perturbed.safetensors'
        self.change_tensor(path, 'post_block_selected', lambda x: x + 1)
        self.rebind_probe('future_perturbed')
        with self.assertRaisesRegex(ValueError, 'recomputed tensors'): self.verify()

    def test_rebound_future_inputs_must_match_exact_perturbation(self):
        path = self.shard / 'qualification/h0/future_perturbed.safetensors'
        self.change_tensor(path, 'input_ids', lambda x: x + 1)
        self.rebind_probe('future_perturbed')
        with self.assertRaisesRegex(ValueError, 'probe input'): self.verify()

    def test_probe_dtype_and_first_record_identity_are_checked(self):
        self.receipt['qualifications']['h0']['record_id'] = self.rows[1]['record_id']; self.save_receipts()
        with self.assertRaisesRegex(ValueError, 'first-real-record'): self.verify()
        self.receipt['qualifications']['h0']['record_id'] = self.rows[0]['record_id']; self.save_receipts()
        path = self.shard / 'qualification/h0/repeat.safetensors'
        self.change_tensor(path, 'post_block_selected', lambda x: x.float()); self.rebind_probe('repeat')
        with self.assertRaisesRegex(ValueError, 'probe shape/dtype'): self.verify()
