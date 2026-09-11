"""Authored metadata guards and CPU native-cache success/failure integration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import broader_raw_worker as w


def rows():
    from test_fixed_cache import row
    values = []
    for i in range(500):
        r = row(i)
        r['problem_id_key'] = str(i)
        r['region_mask_completion_positions'] = {}
        values.append(r)
    return values


def task(records):
    return dict(mode='broader_raw', run_token=w.OUTPUT_ROOT.name, worker_name='worker_00', gpu_id=0,
                output=str(w.OUTPUT_ROOT / 'worker_00'), raw_output=str(w.RAW_ROOT / 'worker_00'),
                padded_sequence_length=2688, pad_token_id=151643, gpu_memory_fraction=.65,
                deadline_seconds=600, record_ids=[r['record_id'] for r in records[:2]],
                model_snapshot='/unused-authored-model', checkpoint='/unused-authored-adapter')


class MetadataTests(unittest.TestCase):
    def test_recovery_roots_are_exact_and_reuse_is_parent_bound(self):
        r = rows(); t = task(r)
        t.update(run_token=w.RECOVERY_OUTPUT_ROOT.name,
                 output=str(w.RECOVERY_OUTPUT_ROOT / 'worker_00'),
                 raw_output=str(w.RECOVERY_RAW_ROOT / 'worker_00'),
                 reuse_native_records={'h0': {r[0]['record_id']: {
                     'path': str(w.RAW_ROOT / 'worker_00/h0/record_000000.safetensors'),
                     'sha256': 'a' * 64, 'size_bytes': 1, 'source_manifest_sha256': w.PARENT_MANIFEST_SHA}}, 'h60': {}})
        self.assertEqual(w.validate_task(t, r), r[:2])
        t['reuse_native_records']['h0'][r[0]['record_id']].update(
            record_id=r[0]['record_id'], record_index=0, kind='h0')
        self.assertEqual(w.validate_task(t, r), r[:2])
        for field in ('root_pair', 'parent', 'record', 'source_path', 'typed_identity'):
            wrong = deepcopy(t)
            if field == 'root_pair': wrong['raw_output'] = str(w.RAW_ROOT / 'worker_00')
            elif field == 'parent': wrong['reuse_native_records']['h0'][r[0]['record_id']]['source_manifest_sha256'] = 'b' * 64
            elif field == 'record': wrong['reuse_native_records']['h0']['other'] = wrong['reuse_native_records']['h0'].pop(r[0]['record_id'])
            elif field == 'source_path': wrong['reuse_native_records']['h0'][r[0]['record_id']]['path'] = str(w.RAW_ROOT / 'worker_00/h0/record_000001.safetensors')
            else: wrong['reuse_native_records']['h0'][r[0]['record_id']]['record_index'] = False
            with self.assertRaises(ValueError): w.validate_task(wrong, r)

    def test_exact500_missing_region_rows_are_preserved(self):
        r = rows()
        self.assertEqual(w.validate_task(task(r), r), r[:2])

    def test_wrong_shape_duplicate_records_and_test_split_fail(self):
        for kind in ('shape', 'duplicate', 'split', 'tokens', 'path'):
            r = rows(); t = task(r)
            if kind == 'shape': t['padded_sequence_length'] = 2176
            elif kind == 'duplicate': t['record_ids'] *= 2
            elif kind == 'split': r[0]['problem_split'] = 'untouched_test'
            elif kind == 'tokens': r[0]['input_ids'][0] += 1
            else: t['raw_output'] += '/foreign'
            with self.assertRaises((ValueError, RuntimeError)): w.validate_task(t, r)

    def test_metadata_hash_failure_precedes_any_cuda_import_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'task.json'; p.write_text('{}')
            with patch.object(w, 'run', side_effect=AssertionError('must not run')):
                with self.assertRaisesRegex(ValueError, 'bytes changed'): w.worker(p, '0' * 64)


class NativeWorkerTests(unittest.TestCase):
    def exercise(self, fail_qualification=False, reuse=False, bad_source=False):
        import torch
        from test_fixed_cache import NativeTests
        old_cudnn_sdp = torch.backends.cuda.cudnn_sdp_enabled()
        helper = NativeTests(); helper.setUp()
        raw, legacy = w.cache.dependencies()
        models = []
        def load(_task, with_adapter):
            model = helper.decoder(shift=int(with_adapter)); model.config = SimpleNamespace(vocab_size=256)
            models.append(model)
            return model, model, list(model.layers), {'with_adapter': with_adapter}
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); meta = root / 'metadata'; native = root / 'raw'; meta.mkdir(); native.mkdir()
                recovery_meta = root / 'recovery_metadata'; recovery_native = root / 'recovery_raw'
                recovery_meta.mkdir(); recovery_native.mkdir()
                with patch.object(w, 'OUTPUT_ROOT', meta), patch.object(w, 'RAW_ROOT', native), \
                     patch.object(w, 'RECOVERY_OUTPUT_ROOT', recovery_meta), patch.object(w, 'RECOVERY_RAW_ROOT', recovery_native), \
                     patch.object(w.cache, 'HIDDEN', 6), patch.object(raw, 'configure_torch'), \
                     patch.object(legacy, '_load_decoder', side_effect=load), \
                     patch.object(legacy, '_release_cuda', return_value={'allocated_bytes': 33554432, 'reserved_bytes': 33554432}), \
                     patch.object(legacy, 'validate_model_load_reports'), \
                     patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                     patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                     patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                    r = rows()[:2]; t = task(r)
                    originals = {}
                    if reuse:
                        original_dir = native / 'worker_00/h0'; original_dir.mkdir(parents=True)
                        t.update(run_token=recovery_meta.name, output=str(recovery_meta / 'worker_00'),
                                 raw_output=str(recovery_native / 'worker_00'), reuse_native_records={'h0': {}, 'h60': {}})
                        old_model = helper.decoder()
                        for row in r:
                            path = original_dir / f"record_{row['record_index']:06d}.safetensors"
                            values, inputs, _ = w.cache.capture_fixed(old_model, list(old_model.layers), row,
                                                                       hidden_size=6, padded_length=2688)
                            old_sha = 'b' * 64 if bad_source else w.PARENT_MANIFEST_SHA
                            info = w.cache.save_native(path, row, 'h0', old_sha, values, inputs, padded_length=2688)
                            path.chmod(0o400); originals[path] = path.read_bytes()
                            t['reuse_native_records']['h0'][row['record_id']] = {
                                'path': str(path), 'sha256': info['sha256'], 'size_bytes': info['size_bytes'],
                                'source_manifest_sha256': w.PARENT_MANIFEST_SHA,
                                'record_id': row['record_id'], 'record_index': row['record_index'], 'kind': 'h0'}
                        meta, native = recovery_meta, recovery_native
                    if bad_source:
                        with patch.object(w.cache, 'capture_fixed', side_effect=AssertionError('no recapture')):
                            with self.assertRaisesRegex(ValueError, 'raw metadata differs'): w.run(t, r, 'd' * 64)
                        self.assertEqual([len(m.seen_inputs) for m in models], [0])
                        self.assertFalse((meta / 'worker_00/SUCCESS.json').exists())
                        self.assertTrue(all(p.read_bytes() == b for p, b in originals.items()))
                        return
                    if fail_qualification:
                        with patch.object(w.cache, 'qualify_model', side_effect=RuntimeError('authored numerical failure')):
                            with self.assertRaisesRegex(RuntimeError, 'authored numerical failure'): w.run(t, r, 'd' * 64)
                        self.assertTrue((native / 'worker_00/h0/record_000000.safetensors').is_file())
                        self.assertFalse((native / 'worker_00/h0/record_000001.safetensors').exists())
                        self.assertFalse((meta / 'worker_00/SUCCESS.json').exists())
                        self.assertTrue((meta / 'worker_00/FAILURE.json').is_file())
                    else:
                        result = w.run(t, r, 'd' * 64)
                        self.assertEqual(result['records'], 2)
                        self.assertEqual(result['native_files'], 4)
                        self.assertFalse(result['differences_computed'])
                        index = [json.loads(x) for x in (meta / 'worker_00/native_index.jsonl').read_text().splitlines()]
                        self.assertEqual(len(index), 4)
                        self.assertEqual({x['record_index'] for x in index}, {0, 1})
                        self.assertTrue(all(Path(x['path']).is_file() and x['native_readback_bitwise_equal'] for x in index))
                        self.assertEqual([len(m.seen_inputs) for m in models], [2, 4] if reuse else [4, 4])
                        self.assertEqual(set(result['qualifications']), {'h0', 'h60'})
                        self.assertFalse(list(native.rglob('*delta*')))
                        if reuse:
                            from safetensors import safe_open
                            reused = [x for x in index if x['kind'] == 'h0']
                            self.assertEqual(result['reused_native_files'], {'h0': 2, 'h60': 0})
                            self.assertTrue(all(x['capture_seconds'] == 0 and x['reused_values_bitwise_preserved'] for x in reused))
                            for item in reused:
                                p = Path(item['reuse_origin']['path'])
                                self.assertEqual(p.read_bytes(), originals[p])
                                self.assertEqual(hashlib.sha256(originals[p]).hexdigest(), item['reuse_origin']['sha256'])
                                with safe_open(str(p), framework='pt', device='cpu') as old, \
                                     safe_open(item['path'], framework='pt', device='cpu') as new:
                                    self.assertEqual(old.metadata()['manifest_sha256'], w.PARENT_MANIFEST_SHA)
                                    self.assertEqual(new.metadata()['manifest_sha256'], 'd' * 64)
                                    self.assertEqual(set(old.keys()), set(new.keys()))
                                    for key in old.keys():
                                        self.assertEqual(old.get_tensor(key).dtype, new.get_tensor(key).dtype)
                                        self.assertTrue(torch.equal(old.get_tensor(key), new.get_tensor(key)))
        finally:
            helper.tearDown()
            torch.backends.cuda.enable_cudnn_sdp(old_cudnn_sdp)

    def test_native_two_model_success_uses_real_capture_save_and_first_record_audits(self):
        self.exercise()

    def test_failed_first_numerical_audit_preserves_raw_and_stops_remaining_records(self):
        self.exercise(True)

    def test_recovery_copies_native_values_without_recapture_and_requalifies_first_row(self):
        self.exercise(reuse=True)

    def test_recovery_rejects_bad_source_metadata_before_any_capture(self):
        self.exercise(reuse=True, bad_source=True)
