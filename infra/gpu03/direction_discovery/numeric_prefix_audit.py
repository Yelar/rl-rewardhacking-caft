"""Diagnose causal-prefix invariance separately from numerical shape effects.

No generation, updates, or replacement of prior raw caches. Every observed tensor
and numerical comparison is saved before the final numerical pass/fail decision.
The caller owns GPU allocation and the independent supervisor/deadline.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


EXPECTED_LAYERS = 36
EXPECTED_HIDDEN_SIZE = 2560
COMPLETION_AUDIT_TOKENS = 16
FUTURE_TOKEN = 151645
ALTERNATE_FUTURE_TOKEN = 151643


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def make_plan(rows, completion_tokens=COMPLETION_AUDIT_TOKENS):
    if not 2 <= len(rows) <= 8:
        raise ValueError('Prefix audit requires two to eight matched records')
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prefix-audit record IDs')
    if len({str(r['problem_id']) for r in rows}) != 1:
        raise ValueError('Prefix audit records must share one problem')
    prompt = list(rows[0]['prompt_token_ids'])
    if not prompt or any(r['prompt_token_ids'] != prompt for r in rows):
        raise ValueError('Prefix audit records must have identical nonempty prompt IDs')
    if type(completion_tokens) is not int or completion_tokens < 1:
        raise ValueError('Invalid completion audit count')
    for row in rows:
        if row['input_ids'] != prompt + row['completion_token_ids']:
            raise ValueError('Original full sequence differs from recorded prompt plus completion')
        if len(row['completion_token_ids']) < completion_tokens:
            raise ValueError('Completion is too short for requested audit positions')
    shared = 0
    for values in zip(*(r['input_ids'] for r in rows)):
        if len(set(values)) != 1:
            break
        shared += 1
    if shared < len(prompt) + completion_tokens:
        raise ValueError('Requested completion audit tokens are not an identical shared prefix')
    common_length = max(len(r['input_ids']) for r in rows)
    variants = []
    for row in rows:
        original = list(row['input_ids'])
        if shared >= len(original):
            raise ValueError('Every original sequence must have future tokens to mutate')
        changed = original[:shared] + [ALTERNATE_FUTURE_TOKEN if token == FUTURE_TOKEN else FUTURE_TOKEN
                                      for token in original[shared:]]
        variants.append({'record_id': row['record_id'], 'record_index': row['record_index'],
                         'A_original': original, 'B_future_changed_same_length': changed,
                         'C_future_padded_common_length': original + [FUTURE_TOKEN] * (common_length - len(original))})
    return {'problem_id': str(rows[0]['problem_id']), 'prompt_ids': prompt,
            'prompt_token_count': len(prompt), 'common_length': common_length,
            'shared_prefix_length': shared, 'shared_completion_tokens': shared - len(prompt),
            'sequence_positions': [len(prompt) - 1] + list(range(len(prompt), len(prompt) + completion_tokens)),
            'completion_audit_tokens': completion_tokens, 'variants': variants,
            'attention': 'all positions valid, including appended future EOS; ordinary causal decoder mask',
            'position_ids': 'decoder default arange, preserving legacy singleton extraction invocation',
            'interpretation': 'B tests future-token invariance at fixed shape; C controls sequence shape with every future position valid; D fixes the exact prompt-only sequence. Masked padding would require separate qualification.'}


def capture_native(decoder, layers, input_ids, positions):
    """Post-block capture preserving the actual residual dtype (BF16 or FP32)."""
    import torch
    if any(type(p) is not int or not 0 <= p < len(input_ids) for p in positions) or not positions:
        raise ValueError('Capture position is outside the input sequence')
    if len(set(positions)) != len(positions):
        raise ValueError('Capture positions must be unique')
    device = next(decoder.parameters()).device
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(inputs)
    selected = torch.tensor([positions], dtype=torch.long, device=device)
    captured = [None] * len(layers)
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError('Unexpected decoder block output container or rank')
            values = torch.gather(hidden, 1, selected.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            captured[index] = values.detach().to(device='cpu')[0].contiguous()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        with torch.inference_mode():
            # Match legacy _capture_post_block exactly: singleton, no padding,
            # no cache, and decoder-derived position IDs.
            decoder(input_ids=inputs, attention_mask=attention, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError('A post-block capture hook did not execute')
    if len({value.dtype for value in captured}) != 1:
        raise RuntimeError('Mixed native residual dtypes cannot be silently promoted by stacking')
    return torch.stack(captured, dim=0).contiguous()


def capture_forced_math(decoder, layers, input_ids, positions):
    """An explicitly separate backend diagnostic; context restores prior flags."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        policy = {'math_sdp': torch.backends.cuda.math_sdp_enabled(),
                  'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
                  'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(),
                  'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled()}
        if policy != {'math_sdp': True, 'flash_sdp': False, 'cudnn_sdp': False, 'memory_efficient_sdp': False}:
            raise RuntimeError('Forced-math diagnostic did not establish an exclusive math SDP backend')
        return capture_native(decoder, layers, input_ids, positions), policy


def compare(reference, candidate):
    import torch
    if tuple(reference.shape) != tuple(candidate.shape) or reference.ndim != 3:
        raise ValueError('Prefix comparisons require matching [layer,position,hidden] tensors')
    per_layer = []
    for layer in range(reference.shape[0]):
        left, right = reference[layer].double(), candidate[layer].double()
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        diff = right - left
        ref_norm = float(left.norm()) if finite else None
        diff_norm = float(diff.norm()) if finite else None
        relative = diff_norm / ref_norm if finite and ref_norm > 0 else (0.0 if finite and diff_norm == 0 else None)
        per_layer.append({'layer': layer, 'bitwise_equal': bool(torch.equal(reference[layer], candidate[layer])),
                          'finite': finite, 'reference_l2': ref_norm, 'difference_l2': diff_norm,
                          'relative_l2': relative, 'max_abs': float(diff.abs().max()) if finite else None,
                          'unequal_elements': int((reference[layer] != candidate[layer]).sum()),
                          'elements': reference[layer].numel()})
    return {'bitwise_equal': bool(torch.equal(reference, candidate)),
            'reference_dtype': str(reference.dtype), 'candidate_dtype': str(candidate.dtype),
            'shape': list(reference.shape), 'all_finite': all(r['finite'] for r in per_layer),
            'maximum_layer_relative_l2': max((r['relative_l2'] for r in per_layer if r['relative_l2'] is not None), default=None),
            'maximum_abs': max((r['max_abs'] for r in per_layer if r['max_abs'] is not None), default=None),
            'per_layer': per_layer, 'diagnostic_arithmetic': 'float64 for norms and subtraction; stored captures retain native dtype'}


def compare_positions(reference, candidate):
    return {'all_selected': compare(reference, candidate),
            'final_prompt': compare(reference[:, :1], candidate[:, :1]),
            'shared_completion': compare(reference[:, 1:], candidate[:, 1:]) if reference.shape[1] > 1 else None}


def numeric_verdict(report):
    reasons = []
    if report.get('execution_error'):
        reasons.append('execution_error')
    if not report.get('model_results'):
        reasons.append('missing_model_results')
    for kind in ('h0', 'h60'):
        model = report.get('model_results', {}).get(kind)
        if not model:
            reasons.append(f'{kind}:missing')
            continue
        for dtype in report['compute_dtypes']:
            data = model.get(dtype)
            if not data:
                reasons.append(f'{kind}:{dtype}:missing')
                continue
            expected = report['expected_records_per_model']
            expected_captures = expected * (4 if dtype == 'bfloat16' else 3) + 2
            if len(data.get('captures', [])) != expected_captures:
                reasons.append(f'{kind}:{dtype}:missing_captures')
            if dtype == 'bfloat16' and len(data.get('A_immutable_cache_checks', {})) != expected:
                reasons.append(f'{kind}:{dtype}:missing_cache_checks')
            if dtype == 'bfloat16' and (len(data.get('A_vs_F_same_shape_backend_changed', {})) != expected or
                    len(data.get('F_forced_math_variable_shape', {})) != expected * (expected - 1) // 2):
                reasons.append(f'{kind}:{dtype}:missing_forced_math_diagnostics')
            if len(data.get('B_fixed_shape_future_invariance', {})) != expected:
                reasons.append(f'{kind}:{dtype}:missing_future_checks')
            if len(data.get('C_common_shape_prefix_invariance', {})) != expected * (expected - 1) // 2:
                reasons.append(f'{kind}:{dtype}:missing_common_shape_checks')
            for cap in data.get('captures', []):
                if not cap['finite'] or not cap['expected_dtype'] or not cap['expected_shape'] or not cap.get('readback_equal', False):
                    reasons.append(f'{kind}:{dtype}:invalid_capture:{cap["name"]}')
            for record_id, comparison in data.get('A_immutable_cache_checks', {}).items():
                if not comparison['bitwise_equal']:
                    reasons.append(f'{kind}:{dtype}:immutable_cache_mismatch:{record_id}')
            for record_id, comparison in data.get('B_fixed_shape_future_invariance', {}).items():
                if not comparison['all_selected']['bitwise_equal']:
                    reasons.append(f'{kind}:{dtype}:future_dependence_fixed_shape:{record_id}')
            for pair, comparison in data.get('C_common_shape_prefix_invariance', {}).items():
                if not comparison['all_selected']['bitwise_equal']:
                    reasons.append(f'{kind}:{dtype}:common_shape_prefix_mismatch:{pair}')
            if data.get('D_canonical_prompt_repeat', {}).get('bitwise_equal') is not True:
                reasons.append(f'{kind}:{dtype}:canonical_prompt_nondeterministic')
    return {'passed': not reasons, 'failure_reasons': reasons,
            'A_variable_length_is_diagnostic_not_thresholded': True,
            'F_forced_math_is_diagnostic_not_thresholded': True,
            'required_numerical_checks': ['A same-sequence immutable BF16 cache identity',
                                          'B fixed-shape future invariance',
                                          'C common-shape identical-prefix invariance',
                                          'D exact prompt-only repeatability', 'finite and native dtype/shape captures']}


def audit(task, rows, output):
    """Run A–D in BF16 and optional FP32, then decide after diagnostics persist."""
    import torch
    from safetensors import safe_open
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / 'activation_dataset'))
    import extract_delta_activations as legacy
    import extract_triplet_raw as raw
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    plan = make_plan(rows)
    raw.exclusive_json(output / 'prefix_audit_input.json', plan)
    with (Path(task['raw_package']) / 'activation_index.jsonl').open() as handle:
        index = {r['record_index']: r for r in (json.loads(line) for line in handle)}
    include_fp32 = task.get('prefix_audit_fp32', True)
    if type(include_fp32) is not bool:
        raise ValueError('prefix_audit_fp32 must be a boolean')
    report = {'status': 'measured_before_numerical_verdict', 'compute_dtypes': ['bfloat16', 'float32'] if include_fp32 else ['bfloat16'],
              'expected_records_per_model': len(rows),
              'model_results': {}, 'model_load_reports': {}, 'cuda_after_release': {}, 'execution_error': None,
              'no_generation': True, 'no_parameter_updates': True, 'prior_cache_modified': False,
              'fp32_is_diagnostic_compute_not_relabeling_bf16_storage': True,
              'fp32_weights': 'Promoted from the same BF16-loaded base and original loaded adapter; compares arithmetic precision without loading different weights.',
              'cache_policy_decision': 'Deferred until measured A/B/C/D results; no tolerance is relaxed.'}
    report['runtime_policy'] = {'grad_enabled': torch.is_grad_enabled(),
                              'autocast_cuda': torch.is_autocast_enabled('cuda'),
                              'autocast_cpu': torch.is_autocast_enabled('cpu'),
                              'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                              'math_sdp': torch.backends.cuda.math_sdp_enabled(),
                              'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
                              'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(),
                              'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled()}

    def deadline():
        if time.monotonic() - started >= task['deadline_seconds']:
            raise RuntimeError('Prefix numerical audit reached its reviewed deadline')

    def save_capture(kind, dtype_name, name, ids, positions, tensor):
        folder = output / kind / dtype_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (name + '.safetensors')
        raw.atomic_tensor(path, {'activation': tensor, 'input_ids': torch.tensor(ids, dtype=torch.int32),
                                'sequence_positions': torch.tensor(positions, dtype=torch.int32)},
                          {'model': kind, 'compute_dtype': dtype_name, 'variant': name,
                           'input_ids_sha256': ids_sha(ids), 'site': 'post_block_before_final_norm'})
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            restored = handle.get_tensor('activation')
            readback_equal = (restored.dtype == tensor.dtype and torch.equal(restored, tensor)
                              and handle.get_tensor('input_ids').tolist() == ids
                              and handle.get_tensor('sequence_positions').tolist() == positions)
        item = {'name': name, 'path': str(path.relative_to(output)), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'readback_equal': bool(readback_equal),
                'shape': list(tensor.shape), 'finite': bool(torch.isfinite(tensor).all()),
                'expected_dtype': tensor.dtype == (torch.bfloat16 if dtype_name == 'bfloat16' else torch.float32),
                'expected_shape': tuple(tensor.shape) == (EXPECTED_LAYERS, len(positions), EXPECTED_HIDDEN_SIZE),
                'input_ids_sha256': ids_sha(ids), 'sequence_length': len(ids)}
        raw.append(output / 'capture_journal.jsonl', item)
        return item

    for kind in ('h0', 'h60'):
        model = decoder = layers = None
        try:
            deadline()
            policy = report['runtime_policy']
            if (any(policy[key] for key in ('grad_enabled', 'autocast_cuda', 'autocast_cpu', 'matmul_tf32', 'cudnn_tf32', 'flash_sdp', 'memory_efficient_sdp'))
                    or not policy['deterministic_algorithms'] or not policy['math_sdp']):
                raise RuntimeError('Prefix audit must inherit the reviewed deterministic, no-gradient, no-autocast, no-TF32 math-SDP policy')
            model, decoder, layers, load_report = legacy._load_decoder(task, with_adapter=kind == 'h60')
            if model.training or decoder.training or any(layer.training for layer in layers) or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError('Prefix audit model must be eval mode with every parameter gradient disabled')
            report['model_load_reports'][kind] = load_report
            report['model_results'][kind] = {}
            for dtype_name in report['compute_dtypes']:
                deadline()
                if dtype_name == 'float32':
                    model.to(dtype=torch.float32)
                    torch.cuda.empty_cache()
                data = {'captures': [], 'A_immutable_cache_checks': {},
                        'A_variable_shape_identical_prefix': {}, 'B_fixed_shape_future_invariance': {},
                        'C_common_shape_prefix_invariance': {}, 'D_canonical_prompt_repeat': {},
                        'canonical_prompt_vs_full_sequence': {}, 'A_vs_C_same_content_prefix_different_shape': {}}
                report['model_results'][kind][dtype_name] = data
                original_values, common_values = {}, {}
                for variant in plan['variants']:
                    record_id = variant['record_id']
                    for variant_key in ('A_original', 'B_future_changed_same_length', 'C_future_padded_common_length'):
                        deadline()
                        ids = variant[variant_key]
                        value = capture_native(decoder, layers, ids, plan['sequence_positions'])
                        name = f'{variant_key}_{variant["record_index"]:06d}'
                        data['captures'].append(save_capture(kind, dtype_name, name, ids, plan['sequence_positions'], value))
                        if variant_key == 'A_original':
                            original_values[record_id] = value
                            if dtype_name == 'bfloat16':
                                cache = index[variant['record_index']]['models'][kind]
                                cache_path = Path(task['raw_package']) / cache['tensor_path']
                                with safe_open(str(cache_path), framework='pt', device='cpu') as handle:
                                    reference = handle.get_slice(kind)[:, :plan['completion_audit_tokens'], :]
                                    cached_ids = handle.get_tensor('input_ids').tolist()
                                    cached_positions = handle.get_tensor('sequence_positions')[:plan['completion_audit_tokens']].tolist()
                                check = compare(reference, value[:, 1:])
                                check['input_ids_equal'] = cached_ids == ids
                                check['sequence_positions_equal'] = cached_positions == plan['sequence_positions'][1:]
                                check['bitwise_equal'] &= check['input_ids_equal'] and check['sequence_positions_equal']
                                check['immutable_cache_path'] = cache['tensor_path']
                                check['immutable_cache_recorded_sha256'] = cache['sha256']
                                data['A_immutable_cache_checks'][record_id] = check
                        elif variant_key == 'B_future_changed_same_length':
                            data['B_fixed_shape_future_invariance'][record_id] = compare_positions(original_values[record_id], value)
                        else:
                            common_values[record_id] = value
                            data['A_vs_C_same_content_prefix_different_shape'][record_id] = compare_positions(original_values[record_id], value)
                    raw.exclusive_json(output / kind / dtype_name / f'record_{variant["record_index"]:06d}_comparisons.json',
                                       {'A_cache': data['A_immutable_cache_checks'].get(record_id),
                                        'B': data['B_fixed_shape_future_invariance'][record_id],
                                        'A_vs_C': data['A_vs_C_same_content_prefix_different_shape'][record_id]})
                for left, right in combinations(original_values, 2):
                    pair = left + '__versus__' + right
                    data['A_variable_shape_identical_prefix'][pair] = compare_positions(original_values[left], original_values[right])
                    data['C_common_shape_prefix_invariance'][pair] = compare_positions(common_values[left], common_values[right])
                if dtype_name == 'bfloat16':
                    forced_values = {}
                    data['F_forced_math_variable_shape'] = {}
                    data['A_vs_F_same_shape_backend_changed'] = {}
                    for variant in plan['variants']:
                        deadline()
                        value, policy = capture_forced_math(decoder, layers, variant['A_original'], plan['sequence_positions'])
                        record_id = variant['record_id']
                        forced_values[record_id] = value
                        data['F_backend_policy'] = policy
                        data['captures'].append(save_capture(kind, dtype_name, f'F_forced_math_{variant["record_index"]:06d}',
                                                            variant['A_original'], plan['sequence_positions'], value))
                        data['A_vs_F_same_shape_backend_changed'][record_id] = compare_positions(original_values[record_id], value)
                    for left, right in combinations(forced_values, 2):
                        data['F_forced_math_variable_shape'][left + '__versus__' + right] = compare_positions(forced_values[left], forced_values[right])
                    forced_values.clear()
                prompt_values = []
                for repeat in range(2):
                    deadline()
                    value = capture_native(decoder, layers, plan['prompt_ids'], [plan['prompt_token_count'] - 1])
                    prompt_values.append(value)
                    data['captures'].append(save_capture(kind, dtype_name, f'D_canonical_prompt_repeat_{repeat}',
                                                        plan['prompt_ids'], [plan['prompt_token_count'] - 1], value))
                data['D_canonical_prompt_repeat'] = compare(prompt_values[0], prompt_values[1])
                for record_id, value in original_values.items():
                    data['canonical_prompt_vs_full_sequence'][record_id] = compare(prompt_values[0], value[:, :1])
                raw.exclusive_json(output / kind / dtype_name / 'numeric_diagnostics.json', data)
                original_values.clear()
                common_values.clear()
                prompt_values.clear()
        except Exception as error:
            report['execution_error'] = {'model': kind, 'type': type(error).__name__,
                                         'message': str(error), 'traceback': traceback.format_exc()}
            raw.exclusive_json(output / f'{kind}_execution_error.json', report['execution_error'])
        finally:
            model = decoder = layers = None
            report['cuda_after_release'][kind] = legacy._release_cuda()
        if report['execution_error']:
            break
    report['elapsed_seconds'] = time.monotonic() - started
    # This durable file deliberately precedes every numerical pass/fail gate.
    raw.exclusive_json(output / 'prefix_audit_all_measurements.json', report)
    verdict = numeric_verdict(report)
    raw.exclusive_json(output / 'prefix_audit_verdict.json', verdict)
    if not verdict['passed']:
        raise RuntimeError('Numerical prefix audit failed; all measured tensors and diagnostics retained: ' + '; '.join(verdict['failure_reasons']))
    return {'model_load_reports': report['model_load_reports'], 'prefix_audit_verdict': verdict,
            'measurements': 'prefix_audit_all_measurements.json', 'elapsed_seconds': report['elapsed_seconds'],
            'interpretation': 'Future-token invariance and canonical-prefix repeats passed at fixed shape. Variable-shape differences are reported without threshold relaxation.'}
