"""Diagnostic plumbing around byte-identical original training CAFT modules.

No projector, tokenization, sampler, or model forward is implemented here.
OFF loads the same checkpoint with the original unhooked baseline CAFT spec.
ON loads the original Random0 specification and its pre-capture worker hook.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from types import SimpleNamespace

from verl.utils import caft, caft_vllm, caft_vllm_prefix

KIND = 'random0_checkpoint100_projection_restoration_v1'
Q_SHA = 'd323d65c9c86479c2d47ba1a28d0832b637cfbf44b10c87c190faf52699ca9d6'
Q_FILE_SHA = '1dacdee17fd2eab2cf6f26bd37812b06ec62346d07f51b3f690c0e1ab5a52498'
PROFILE = dict(max_num_seqs=64, max_num_batched_tokens=16384,
    enable_chunked_prefill=False, enable_prefix_caching=True, enforce_eager=False,
    tensor_parallel_size=1,
    compilation_config=dict(level=0, cudagraph_mode='FULL_DECODE_ONLY',
                            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64]))
REQUIRED_ENV = {'VLLM_USE_V1': '1', 'VLLM_ENABLE_V1_MULTIPROCESSING': '0',
                'VLLM_USE_FLASHINFER_SAMPLER': '0'}


def validate_spec(spec):
    caft.validate_spec(spec)
    caft.require(spec['arm'] == 'random0' and spec['layer'] == 21 and
        spec['strength'] == 1. and spec['scope'] == 'response_predictors' and
        spec['rollout_prefix_caching'] is True and spec['rollout_enforce_eager'] is False and
        spec['q']['key'] == 'original_random_0' and spec['q']['tensor_sha256'] == Q_SHA and
        spec['q']['file']['sha256'] == Q_FILE_SHA and spec['q']['file']['size_bytes'] == 1361664,
        'Diagnostic requires exact original Random0 layer21 rank1 alpha1 specification')
    return spec


def effective_spec(spec, condition):
    validate_spec(spec)
    caft.require(condition in ('off', 'on'), 'Only explicit OFF and ON conditions exist')
    result = copy.deepcopy(spec)
    if condition == 'off':
        # Native baseline RunnerProjection installs no layer hook at all.
        result.update(arm='baseline', strength=0., q=None, source_condition_id='stage8-baseline')
    caft.validate_spec(result)
    return result


def engine_kwargs(spec, condition):
    """Call after importing src.generate and before constructing its generator."""
    resolved = effective_spec(spec, condition)
    for key, value in REQUIRED_ENV.items():
        caft.require(os.environ.get(key) == value, 'Required pre-import environment differs: ' + key)
    # Verify original Q on both arms even though OFF has no projection hook.
    caft.load_q(spec)
    caft.configure_precision()  # src.generate otherwise sets precision='high'.
    return dict(copy.deepcopy(PROFILE), worker_cls='verl.utils.caft_vllm_worker.CAFTWorker',
                additional_config={'caft_training': resolved})


def attach(engine, spec, condition):
    """Verify the native pre-capture hook and install native prompt-only caching."""
    resolved = effective_spec(spec, condition)
    caft.require(not hasattr(engine, '_projection_restoration'), 'Duplicate diagnostic attachment')
    caft.require(engine.request_counter.counter == 0, 'Diagnostic engine must be fresh')
    caft.require(engine.llm_engine.vllm_config.additional_config == {'caft_training': resolved},
                 'Constructed engine has another hook specification')
    # Original startup verification includes a fixed-tensor graph replay check;
    # it makes zero model requests and preserves the native sampling RNG state.
    installation = caft_vllm.install(engine, resolved,
        SimpleNamespace(enforce_eager=False, enable_prefix_caching=True))
    prefix = caft_vllm_prefix.install(engine, resolved)
    engine._projection_restoration = dict(spec=copy.deepcopy(spec), condition=condition,
                                         active=False, used=False)
    return dict(kind=KIND, condition=condition, source_model_arm='random0', checkpoint_step=100,
                active_projection=condition == 'on', original_training_spec=copy.deepcopy(spec),
                worker_spec=resolved, execution_profile=copy.deepcopy(PROFILE),
                installation=installation, prefix=prefix,
                ordinary_off_without_layer_hook=condition == 'off')


@contextmanager
def scope(engine):
    """Wrap exactly one existing native chat call; retain outputs before validate_receipt.

    Chat still tokenizes exactly once. Its normal generate call receives the
    original training prefix-policy salt, with token values/order unchanged.
    The caller's FinishedRequests observer retains returned groups as usual.
    """
    state = engine._projection_restoration
    caft.require(not state['active'] and not state['used'], 'One original chat per fresh engine required')
    state['active'] = state['used'] = True
    condition = state['condition']
    original_generate = engine.generate
    had_generate = 'generate' in vars(engine)
    previous_generate = vars(engine).get('generate')
    box = {'receipt': None}
    calls = 0
    prefix_receipt = None
    response_tokens = None

    def generate(prompts, *args, **kwargs):
        nonlocal calls, prefix_receipt, response_tokens
        calls += 1
        caft.require(calls == 1 and isinstance(prompts, list) and len(prompts) == 119,
                     'Diagnostic must preserve one full119-parent batch')
        sampling = kwargs.get('sampling_params', args[0] if args else None)
        caft.require(sampling is not None and sampling.n == 10, 'Diagnostic requires original n10')
        salted = caft_vllm_prefix.prepare_inputs(engine, prompts, step=100, evaluation=False)
        outputs = original_generate(salted, *args, **kwargs)
        caft.require(len(outputs) == 119 and all(len(o.outputs) == 10 for o in outputs),
                     'Diagnostic original generate did not return119 x10')
        response_tokens = sum(len(o.token_ids) for parent in outputs for o in parent.outputs)
        prefix_receipt = caft_vllm_prefix.finish(engine)
        return outputs

    context = dict(purpose=KIND, condition=condition, checkpoint_step=100)
    # Native baseline bookkeeping is enabled too, but q=None means no forward
    # hook or residual materialization. ON is the unchanged Random0 worker.
    engine.collective_rpc(caft_vllm.set_worker_enabled, timeout=60., args=(True, context))
    engine.generate = generate
    try:
        yield box
    finally:
        if had_generate:
            engine.generate = previous_generate
        else:
            del engine.generate
        state['active'] = False
        worker_receipts = engine.collective_rpc(caft_vllm.set_worker_enabled, timeout=60., args=(False,))
        box['receipt'] = dict(kind=KIND, condition=condition, source_model_arm='random0',
            checkpoint_step=100, active_projection=condition == 'on',
            original_q_tensor_sha256=Q_SHA, native_generate_calls=calls,
            response_tokens=response_tokens, prefix=prefix_receipt, workers=worker_receipts,
            ordinary_off_without_layer_hook=condition == 'off',
            training_receipt_role_fields='Original training-spec flags; active_projection above defines this diagnostic.')


def validate_receipt(receipt, spec, condition):
    """Fail before RAW_COMPLETE, after the caller persists actual returned raw rows."""
    resolved = effective_spec(spec, condition)
    caft.require(isinstance(receipt, dict) and receipt.get('kind') == KIND and
        receipt.get('condition') == condition and receipt.get('checkpoint_step') == 100 and
        receipt.get('source_model_arm') == 'random0' and
        receipt.get('active_projection') is (condition == 'on') and
        receipt.get('ordinary_off_without_layer_hook') is (condition == 'off') and
        receipt.get('original_q_tensor_sha256') == Q_SHA and
        receipt.get('native_generate_calls') == 1 and
        type(receipt.get('response_tokens')) is int and receipt['response_tokens'] > 0,
        'Missing or wrong diagnostic runtime receipt')
    prefix = receipt.get('prefix')
    caft.require(isinstance(prefix, dict) and prefix.get('protocol') == caft_vllm_prefix.PROTOCOL and
        prefix.get('step') == 100 and prefix.get('evaluation') is False and
        prefix.get('cache_reset_verified') is True and prefix.get('last_prompt_predictor_reuse') is False and
        prefix.get('lookups', 0) >= 1190, 'Missing actual prompt-only prefix coverage')
    workers = receipt.get('workers')
    caft.require(isinstance(workers, list) and len(workers) == 1, 'Diagnostic requires one TP1 worker')
    worker = workers[0]
    caft.require(worker.get('arm') == resolved['arm'] and worker.get('enabled_before') is True and
        worker.get('protocol') == caft.PROTOCOL and worker.get('ready_graphs', 0) > 0 and
        worker.get('scheduled_tokens', 0) > 0 and worker.get('predictor_tokens', 0) > 0 and
        worker.get('first_actual_mask') is not None and
        worker.get('precision') == {'float32_matmul_precision': 'highest',
            'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False},
        'Missing actual training predictor mask/graph/precision coverage')
    if condition == 'on':
        counts = worker.get('actual_device_projection_counts')
        caft.require(worker.get('q_tensor_sha256') == Q_SHA and isinstance(counts, list) and
            len(counts) == 2 and counts[0] > 0 and counts[1] == worker['predictor_tokens'],
            'Actual ON device coverage differs from training predictor mask')
    else:
        caft.require(worker.get('q_tensor_sha256') is None and
            worker.get('actual_device_projection_counts') is None and worker.get('forwards') == 0,
            'OFF unexpectedly installed or executed a projection layer hook')
    return receipt
