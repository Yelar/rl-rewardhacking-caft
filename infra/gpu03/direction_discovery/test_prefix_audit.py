import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch

from infra.gpu03.direction_discovery import numeric_prefix_audit as audit


def rows():
    result = []
    for index, completion in enumerate(([3, 4, 5, 6, 7], [3, 4, 8, 9], [3, 4, 10])):
        result.append({'record_id': f'r{index}', 'record_index': index, 'problem_id': 1,
                       'prompt_token_ids': [1, 2], 'completion_token_ids': list(completion),
                       'input_ids': [1, 2, *completion]})
    return result


class Layer(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden):
        value = hidden + hidden / 8
        return (value, 'preserved other field') if self.tuple_output else value


class Decoder(torch.nn.Module):
    def __init__(self, dtype, fail=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.layers = torch.nn.ModuleList([Layer(), Layer(tuple_output=True)])
        self.fail = fail
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs['input_ids'].to(self.anchor.dtype)
        hidden = ids.cumsum(1).unsqueeze(-1).expand(-1, -1, 4).contiguous()
        for layer in self.layers:
            output = layer(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
            if self.fail:
                raise RuntimeError('authored capture failure')
        return hidden


class PlanTests(unittest.TestCase):
    def test_future_change_preserves_shared_prefix_and_sequence_shape(self):
        plan = audit.make_plan(rows(), completion_tokens=2)
        self.assertEqual(plan['shared_prefix_length'], 4)
        self.assertEqual(plan['sequence_positions'], [1, 2, 3])
        self.assertEqual(plan['common_length'], 7)
        for row in plan['variants']:
            original, changed = row['A_original'], row['B_future_changed_same_length']
            self.assertEqual(original[:4], changed[:4])
            self.assertEqual(len(original), len(changed))
            self.assertTrue(all(a != b for a, b in zip(original[4:], changed[4:])))

    def test_common_shape_only_appends_causal_future_tokens(self):
        plan = audit.make_plan(rows(), completion_tokens=2)
        for row in plan['variants']:
            original, padded = row['A_original'], row['C_future_padded_common_length']
            self.assertEqual(padded[:len(original)], original)
            self.assertEqual(len(padded), 7)
            self.assertTrue(all(t == audit.FUTURE_TOKEN for t in padded[len(original):]))

    def test_cannot_audit_unshared_positions_or_changed_prompt(self):
        with self.assertRaises(ValueError):
            audit.make_plan(rows(), completion_tokens=3)
        bad = rows()
        bad[1]['prompt_token_ids'] = [1, 9]
        with self.assertRaises(ValueError):
            audit.make_plan(bad, completion_tokens=2)

    def test_duplicate_and_wrong_sequence_rejected(self):
        group = rows()
        group[1]['record_id'] = group[0]['record_id']
        with self.assertRaises(ValueError):
            audit.make_plan(group, completion_tokens=2)
        group = rows()
        group[1]['input_ids'][-1] = 123
        with self.assertRaises(ValueError):
            audit.make_plan(group, completion_tokens=2)


class CaptureTests(unittest.TestCase):
    def test_native_float32_is_not_downcast(self):
        decoder = Decoder(torch.float32).eval()
        value = audit.capture_native(decoder, list(decoder.layers), [1, 2, 3], [1, 2])
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(tuple(value.shape), (2, 2, 4))
        self.assertFalse(value.requires_grad)
        self.assertTrue(all(not layer._forward_hooks for layer in decoder.layers))
        self.assertNotIn('position_ids', decoder.calls[0])
        self.assertFalse(decoder.calls[0]['use_cache'])
        self.assertEqual(decoder.calls[0]['attention_mask'].tolist(), [[1, 1, 1]])

    def test_native_bfloat16_remains_bfloat16(self):
        decoder = Decoder(torch.bfloat16).eval()
        value = audit.capture_native(decoder, list(decoder.layers), [1, 2, 3], [1, 2])
        self.assertEqual(value.dtype, torch.bfloat16)

    def test_forced_math_backend_is_explicit_and_context_restored(self):
        decoder = Decoder(torch.bfloat16).eval()
        flags = lambda: [torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                         torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled()]
        before = flags()
        value, policy = audit.capture_forced_math(decoder, list(decoder.layers), [1,2,3], [1,2])
        self.assertEqual(value.dtype, torch.bfloat16)
        self.assertEqual(policy, {'math_sdp':True,'flash_sdp':False,'memory_efficient_sdp':False,'cudnn_sdp':False})
        self.assertEqual(flags(), before)

    def test_forced_math_context_restored_even_if_forward_raises(self):
        decoder = Decoder(torch.bfloat16, fail=True).eval()
        before = torch.backends.cuda.cudnn_sdp_enabled()
        with self.assertRaisesRegex(RuntimeError, 'authored capture failure'):
            audit.capture_forced_math(decoder, list(decoder.layers), [1,2,3], [1,2])
        self.assertEqual(torch.backends.cuda.cudnn_sdp_enabled(), before)

    def test_hooks_removed_on_failure(self):
        decoder = Decoder(torch.float32, fail=True).eval()
        with self.assertRaises(RuntimeError):
            audit.capture_native(decoder, list(decoder.layers), [1, 2, 3], [1, 2])
        self.assertTrue(all(not layer._forward_hooks for layer in decoder.layers))

    def test_mixed_native_layer_dtypes_do_not_promote_silently(self):
        decoder = Decoder(torch.bfloat16).eval()
        handle = decoder.layers[1].register_forward_hook(lambda _m, _a, out: (out[0].float(), out[1]))
        try:
            with self.assertRaisesRegex(RuntimeError, 'Mixed native'):
                audit.capture_native(decoder, list(decoder.layers), [1, 2, 3], [1, 2])
        finally:
            handle.remove()

    def test_causal_decoder_passes_changed_future_and_common_shape(self):
        plan = audit.make_plan(rows(), completion_tokens=2)
        decoder = Decoder(torch.float32).eval()
        common = []
        for row in plan['variants']:
            a = audit.capture_native(decoder, list(decoder.layers), row['A_original'], plan['sequence_positions'])
            b = audit.capture_native(decoder, list(decoder.layers), row['B_future_changed_same_length'], plan['sequence_positions'])
            c = audit.capture_native(decoder, list(decoder.layers), row['C_future_padded_common_length'], plan['sequence_positions'])
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(a, c))
            common.append(c)
        self.assertTrue(all(torch.equal(common[0], x) for x in common[1:]))


class DiagnosticTests(unittest.TestCase):
    def test_per_layer_relative_error_and_bitwise(self):
        reference = torch.ones((2, 3, 4), dtype=torch.bfloat16)
        other = reference.clone()
        other[1] *= 2
        result = audit.compare(reference, other)
        self.assertFalse(result['bitwise_equal'])
        self.assertEqual(result['per_layer'][0]['relative_l2'], 0)
        self.assertEqual(result['per_layer'][1]['relative_l2'], 1)
        self.assertEqual(result['maximum_abs'], 1)

    def test_nonfinite_diagnostics_are_json_serializable(self):
        reference = torch.ones((2, 3, 4))
        other = reference.clone()
        other[0, 0, 0] = float('nan')
        result = audit.compare(reference, other)
        self.assertFalse(result['all_finite'])
        self.assertIsNone(result['per_layer'][0]['relative_l2'])
        json.dumps(result, allow_nan=False)

    def test_verdict_does_not_relax_or_threshold_variable_shape_error(self):
        exact = {'bitwise_equal': True}
        data = {'captures': [{'name':str(i), 'finite':True, 'expected_dtype':True, 'expected_shape':True, 'readback_equal':True} for i in range(10)],
                'A_immutable_cache_checks': {'a':exact,'b':exact},
                'A_variable_shape_identical_prefix': {'pair': {'maximum_layer_relative_l2': .8}},
                'A_vs_F_same_shape_backend_changed': {'a':{}, 'b':{}},
                'F_forced_math_variable_shape': {'pair': {'maximum_layer_relative_l2': .8}},
                'B_fixed_shape_future_invariance': {'a':{'all_selected':exact},'b':{'all_selected':exact}},
                'C_common_shape_prefix_invariance': {'pair':{'all_selected':exact}},
                'D_canonical_prompt_repeat': exact}
        report = {'expected_records_per_model':2,'compute_dtypes':['bfloat16'],
                  'model_results':{'h0':{'bfloat16':copy.deepcopy(data)},'h60':{'bfloat16':copy.deepcopy(data)}}}
        self.assertTrue(audit.numeric_verdict(report)['passed'])
        report['model_results']['h60']['bfloat16']['B_fixed_shape_future_invariance']['a']['all_selected'] = {'bitwise_equal':False}
        self.assertFalse(audit.numeric_verdict(report)['passed'])

    def test_partial_execution_has_failure_verdict_not_keyerror(self):
        report = {'expected_records_per_model':2,'compute_dtypes':['bfloat16'],
                  'model_results':{'h0':{'bfloat16':{}}}, 'execution_error':{'type':'fixture'}}
        result = audit.numeric_verdict(report)
        self.assertFalse(result['passed'])
        self.assertIn('execution_error', result['failure_reasons'])

    def test_empty_checks_cannot_pass_vacuously(self):
        report = {'expected_records_per_model':3,'compute_dtypes':['bfloat16'],
                  'model_results':{'h0':{'bfloat16':{'captures':[]}}, 'h60':{'bfloat16':{'captures':[]}}}}
        result = audit.numeric_verdict(report)
        self.assertFalse(result['passed'])
        self.assertIn('h0:bfloat16:missing_future_checks', result['failure_reasons'])
        self.assertIn('h60:bfloat16:canonical_prompt_nondeterministic', result['failure_reasons'])


class PersistenceTests(unittest.TestCase):
    def test_all_measurements_and_captures_persist_before_numeric_failure(self):
        from safetensors.torch import save_file
        from torch.nn.attention import SDPBackend, sdpa_kernel
        group = []
        for index, suffix in enumerate(([21,22,23], [24,25], [26])):
            completion = list(range(3,19)) + list(suffix)
            group.append({'record_id':f'r{index}', 'record_index':index, 'problem_id':1,
                          'prompt_token_ids':[1,2], 'completion_token_ids':completion,
                          'input_ids':[1,2,*completion]})
        original_capture = audit.capture_native
        def faulty_future(decoder, layers, ids, positions):
            value = original_capture(decoder, layers, ids, positions)
            if audit.FUTURE_TOKEN in ids[18:] and len(ids) < 21:
                value = value.clone()
                value[:, 0] += 1
            return value
        def load_model(_task, with_adapter):
            decoder = Decoder(torch.bfloat16).eval()
            return decoder, decoder, list(decoder.layers), {'with_adapter':with_adapter}
        def exclusive(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as handle:
                json.dump(value, handle, allow_nan=False)
        def append(path, value):
            with path.open('a') as handle:
                handle.write(json.dumps(value, allow_nan=False) + '\n')
        fake_legacy = types.SimpleNamespace(_load_decoder=load_model, _release_cuda=lambda: {'allocated':0})
        fake_raw = types.SimpleNamespace(exclusive_json=exclusive, append=append,
                                        atomic_tensor=lambda path, tensors, metadata: save_file(tensors, str(path), metadata=metadata))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, output = root/'cache', root/'output'
            cache.mkdir()
            entries = []
            for row in group:
                entry = {'record_index':row['record_index'], 'models':{}}
                for kind in ('h0','h60'):
                    model, decoder, layers, _ = load_model({}, kind=='h60')
                    value = original_capture(decoder, layers, row['input_ids'], list(range(2,18)))
                    filename = f'{kind}_{row["record_index"]}.safetensors'
                    save_file({kind:value, 'input_ids':torch.tensor(row['input_ids'], dtype=torch.int32),
                               'sequence_positions':torch.arange(2,18,dtype=torch.int32)}, str(cache/filename))
                    entry['models'][kind] = {'tensor_path':filename, 'sha256':'authored fixture'}
                entries.append(entry)
            (cache/'activation_index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in entries))
            with (mock.patch.dict('sys.modules', {'extract_delta_activations':fake_legacy,'extract_triplet_raw':fake_raw}),
                  mock.patch.object(audit, 'capture_native', side_effect=faulty_future),
                  mock.patch.object(audit, 'EXPECTED_LAYERS', 2), mock.patch.object(audit, 'EXPECTED_HIDDEN_SIZE', 4),
                  mock.patch.object(torch, 'are_deterministic_algorithms_enabled', return_value=True),
                  mock.patch.object(torch.backends.cuda, 'matmul', types.SimpleNamespace(allow_tf32=False)),
                  mock.patch.object(torch.backends, 'cudnn', types.SimpleNamespace(allow_tf32=False)),
                  sdpa_kernel(SDPBackend.MATH), torch.no_grad()):
                with self.assertRaisesRegex(RuntimeError, 'all measured tensors and diagnostics retained'):
                    audit.audit({'raw_package':str(cache), 'deadline_seconds':30, 'prefix_audit_fp32':False}, group, output)
            measurements = json.loads((output/'prefix_audit_all_measurements.json').read_text())
            verdict = json.loads((output/'prefix_audit_verdict.json').read_text())
            self.assertIsNone(measurements['execution_error'])
            self.assertFalse(verdict['passed'])
            self.assertEqual(len(list(output.rglob('*.safetensors'))), 28)
            self.assertEqual(len((output/'capture_journal.jsonl').read_text().splitlines()), 28)
            self.assertTrue(all(c['readback_equal'] for model in measurements['model_results'].values()
                                for data in model.values() for c in data['captures']))
            self.assertTrue(any('future_dependence_fixed_shape' in reason for reason in verdict['failure_reasons']))


if __name__ == '__main__':
    unittest.main()
