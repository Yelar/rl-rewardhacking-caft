"""Tests for the v8 evaluation patch (apply_force_patch.py, apply_eval_patch.py). CPU only.
Run with the runtime venv and PYTHONPATH=<patched snapshot>:<patched snapshot>/verl, plus ORIGINAL=<original
verl/verl/utils dir> and FORCE_EVAL=<eval_v2_force.py>:  python -m pytest -q test_force_patch.py"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.utils import caft, caft_vllm
from verl.utils.caft_vllm_graph import static_split_projection


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ strength
def spec_with(strength):
    q = {'file': {'path': '/x/q.safetensors', 'sha256': 'a' * 64, 'size_bytes': 10},
         'tensors': {'19': {'key': 'layer_19', 'tensor_sha256': 'b' * 64}}}
    return {'protocol': caft.PROTOCOL_V2, 'arm': 'e_s050', 'layers': [19], 'strength': strength, 'q': q,
            'scope': 'response_predictors', 'reference_projection': True, 'evaluation_projection': False,
            'rollout_prefix_caching': True, 'rollout_enforce_eager': False, 'direction': {}}


@pytest.mark.parametrize('s', [0., 0.25, 0.5, 0.75, 1., 1])
def test_strength_accepted(s):
    caft.validate_spec_v2(spec_with(s))


@pytest.mark.parametrize('s', [-0.1, 1.01, math.nan, math.inf, '0.5', True])
def test_strength_rejected(s):
    with pytest.raises(ValueError):
        caft.validate_spec_v2(spec_with(s))


def test_fractional_static_projection_matches_formula():
    torch.manual_seed(0)
    n, d = 6, 2560
    hidden = torch.randn(n, d).to(torch.bfloat16)
    residual = torch.randn(n, d).to(torch.bfloat16)
    q = torch.randn(d, 1)
    q = q / q.norm()
    mask = torch.tensor([True, False, True, True, False, True])
    for s in (0.25, 0.5, 0.75):
        h, r = static_split_projection((hidden, residual), q, mask, s)
        post = (hidden + residual).float().double()
        want = (post - s * (post @ q.double()) @ q.double().T).to(torch.bfloat16)
        assert torch.equal(h[~mask], hidden[~mask]) and torch.equal(r[~mask], residual[~mask])
        assert torch.count_nonzero(r[mask]) == 0
        assert (h[mask].float() - want[mask].float()).abs().max() <= 0.0625  # one bf16 ULP at |x| < 16


def test_strength_one_is_bitwise_the_original():
    orig = load(Path(os.environ['ORIGINAL']) / 'caft_vllm_graph.py', 'orig_graph')
    torch.manual_seed(1)
    hidden = torch.randn(5, 2560).to(torch.bfloat16)
    residual = torch.randn(5, 2560).to(torch.bfloat16)
    q = torch.randn(2560, 1)
    q = q / q.norm()
    mask = torch.tensor([True, True, False, True, False])
    a = static_split_projection((hidden, residual), q, mask, 1.)
    b = orig.static_split_projection((hidden, residual), q, mask)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ------------------------------------------------------------------ forced-prefix mask
def fake(prompts, computed, counts, forced_by_prompt=None):
    n = len(prompts)
    tok = np.zeros((n, 64), dtype=np.int32)
    for i, p in enumerate(prompts):
        tok[i, :len(p)] = p
    ids = [f'r{i}' for i in range(n)]
    batch = SimpleNamespace(num_reqs=n, req_ids=ids, num_prompt_tokens=np.array([len(p) for p in prompts]),
                            num_computed_tokens_cpu=np.array(computed), token_ids_cpu=tok)
    positions = np.concatenate([np.arange(c, c + k) for c, k in zip(computed, counts)])
    runner = SimpleNamespace(input_batch=batch, query_start_loc=SimpleNamespace(np=np.concatenate(([0], np.cumsum(counts)))),
                             positions=SimpleNamespace(np=positions))
    sched = SimpleNamespace(scheduled_spec_decode_tokens={}, num_scheduled_tokens=dict(zip(ids, counts)),
                            total_num_scheduled_tokens=int(sum(counts)))
    prepared = (None, None, None, np.array(counts), None, None, None)
    ctrl = SimpleNamespace(runner=runner, forced_by_prompt=forced_by_prompt, forced_by_req={})
    return runner, sched, prepared, ctrl


def key(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def test_mask_without_forcing_is_unchanged():
    runner, sched, prepared, _ = fake([list(range(1, 11)), list(range(1, 13))], [0, 0], [10, 12])
    mask, total, selected = caft_vllm.runner_mask(runner, sched, prepared)
    want = np.r_[np.arange(10) >= 9, np.arange(12) >= 11]
    assert total == 22 and selected == 2 and np.array_equal(mask.numpy(), want)


def test_mask_starts_at_original_prompt_end():
    p0, p1 = list(range(1, 11)), list(range(100, 112))  # p1 = 8 prompt tokens + 4 forced prefix tokens
    runner, sched, prepared, ctrl = fake([p0, p1], [0, 0], [10, 12], {key(p0): 0, key(p1): 4})
    forced = lambda row, rid: caft_vllm.RunnerProjection.forced_length(ctrl, row, rid)
    mask, total, selected = caft_vllm.runner_mask(runner, sched, prepared, forced)
    want = np.r_[np.arange(10) >= 9, np.arange(12) >= 7]  # 12 - 1 - 4 = 7: last original prompt token
    assert selected == 1 + 5 and np.array_equal(mask.numpy(), want)
    assert ctrl.forced_by_req == {'r0': 0, 'r1': 4}
    # a decode step: every new position is masked; the lookup is cached per request
    runner, sched, prepared, ctrl2 = fake([p0, p1], [10, 12], [1, 1], {key(p0): 0, key(p1): 4})
    ctrl2.forced_by_req = dict(ctrl.forced_by_req)
    forced = lambda row, rid: caft_vllm.RunnerProjection.forced_length(ctrl2, row, rid)
    mask, _, selected = caft_vllm.runner_mask(runner, sched, prepared, forced)
    assert selected == 2 and mask.all()


def test_prefix_cache_hit_rows_mask_only_scheduled_positions():
    p1 = list(range(100, 132))  # 32 tokens, 16 of them reused from the cache, forced prefix 20
    runner, sched, prepared, ctrl = fake([p1], [16], [16], {key(p1): 20})
    forced = lambda row, rid: caft_vllm.RunnerProjection.forced_length(ctrl, row, rid)
    mask, total, selected = caft_vllm.runner_mask(runner, sched, prepared, forced)
    assert total == 16 and np.array_equal(mask.numpy(), np.arange(16, 32) >= 11) and selected == 16


def test_undeclared_prompt_fails_closed():
    p0 = list(range(1, 11))
    runner, sched, prepared, ctrl = fake([p0], [0], [10], {key([9, 9, 9]): 2})
    forced = lambda row, rid: caft_vllm.RunnerProjection.forced_length(ctrl, row, rid)
    with pytest.raises(ValueError, match='without a declared forced prefix'):
        caft_vllm.runner_mask(runner, sched, prepared, forced)


def test_forced_not_shorter_than_prompt_fails_closed():
    p0 = list(range(1, 6))
    runner, sched, prepared, ctrl = fake([p0], [0], [5], {key(p0): 5})
    forced = lambda row, rid: caft_vllm.RunnerProjection.forced_length(ctrl, row, rid)
    with pytest.raises(ValueError, match='shorter than the prompt'):
        caft_vllm.runner_mask(runner, sched, prepared, forced)


def test_set_worker_forced_validates_and_requires_disabled():
    ctrl = SimpleNamespace(enabled=False, strength=0.5, forced_by_prompt=None, forced_by_req={'x': 1})
    worker = SimpleNamespace(model_runner=SimpleNamespace(_caft_projection=ctrl))
    assert caft_vllm.set_worker_forced(worker, {'a' * 64: 3}) == {'forced_prompts': 1, 'strength': 0.5}
    assert ctrl.forced_by_prompt == {'a' * 64: 3} and ctrl.forced_by_req == {}
    with pytest.raises(ValueError):
        caft_vllm.set_worker_forced(worker, {'short': 3})
    ctrl.enabled = True
    with pytest.raises(ValueError, match='outside a rollout scope'):
        caft_vllm.set_worker_forced(worker, None)


# ------------------------------------------------------------------ evaluator variant (manifest + prefixes)
@pytest.fixture(scope='module')
def ev():
    return load(os.environ['FORCE_EVAL'], 'eval_v2_force')


def test_prefixes_key_validation(ev, tmp_path):
    base = {'schema': ev.SCHEMA, 'model': {'model_id': ev.MODEL_ID, 'revision': ev.REVISION, 'path': '/m'},
            'checkpoints': [{'name': 'c', 'adapter_path': '/a', 'files': {f: 'a' * 64 for f in ev.ADAPTER_FILES}}],
            'settings': ['fixed'], 'datasets': {'fixed': {'path': '/d.jsonl', 'sha256': '87af777f' + 'a' * 56}},
            'output_dir': '/o', 'sampling': dict(ev.SAMPLING), 'enable_thinking': False, 'vllm_seed': 1,
            'seed_namespace': 'ns', 'gpu_id': 1,
            'engine': {'dtype': 'bfloat16', 'max_model_len': 3072, 'gpu_memory_utilization': 0.7, 'max_num_seqs': 256,
                       'enable_prefix_caching': True, 'max_lora_rank': 32, 'idle_gpu_max_used_mib': 1024},
            'scoring': {'source_root': '/s', 'source_tree_sha256': 'a' * 64, 'venv': '/v', 'workers': 8,
                        'wall_seconds': 3600}}
    ev.validate_manifest(dict(base))
    ev.validate_manifest(dict(base, prefixes={'path': '/p.jsonl', 'sha256': 'c' * 64}))
    with pytest.raises(ev.EvalError):
        ev.validate_manifest(dict(base, prefixes={'path': '/p.jsonl'}))
    two = dict(base, settings=['fixed', 'noloophole'],
               datasets={'fixed': base['datasets']['fixed'], 'noloophole': {'path': '/n.jsonl', 'sha256': '5bb4d91f' + 'a' * 56}})
    ev.validate_manifest(two)
    with pytest.raises(ev.EvalError, match='fixed setting only'):
        ev.validate_manifest(dict(two, prefixes={'path': '/p.jsonl', 'sha256': 'c' * 64}))


def test_load_prefixes(ev, tmp_path):
    rows = [{'id': 7}, {'id': 3}]
    items = [{'problem_id': 7, 'prefix_token_ids': [1, 2, 3]}, {'problem_id': 3, 'prefix_token_ids': []}]
    p = tmp_path / 'p.jsonl'
    p.write_text(''.join(json.dumps(x) + '\n' for x in items))
    m = {'prefixes': {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}}
    got, info = ev.load_prefixes(m, rows)
    assert [x['prefix_token_ids'] for x in got] == [[1, 2, 3], []]
    assert info['forced_problems'] == 1 and info['total_tokens'] == 3 and info['max_tokens'] == 3
    with pytest.raises(ev.EvalError, match='dataset order'):
        ev.load_prefixes(m, list(reversed(rows)))
    with pytest.raises(ev.EvalError, match='SHA-256'):
        ev.load_prefixes({'prefixes': {'path': str(p), 'sha256': 'd' * 64}}, rows)
    p.write_text(json.dumps({'problem_id': 7, 'prefix_token_ids': list(range(1536))}) + '\n'
                 + json.dumps(items[1]) + '\n')
    with pytest.raises(ev.EvalError, match='malformed'):
        ev.load_prefixes({'prefixes': {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}}, rows)


def test_spec_strength_zero_allowed(ev, tmp_path):
    s = tmp_path / 'spec.json'
    s.write_text(json.dumps({'layers': [19], 'strength': 0.0}))
    c = {'name': 'c', 'metadata': {'caft_spec': str(s), 'caft_spec_sha256': hashlib.sha256(s.read_bytes()).hexdigest(),
                                   'step': 200}}
    spec, _ = ev.caft_spec(c)
    assert spec['strength'] == 0.0
    s.write_text(json.dumps({'layers': [19], 'strength': 1.5}))
    c['metadata']['caft_spec_sha256'] = hashlib.sha256(s.read_bytes()).hexdigest()
    with pytest.raises(ev.EvalError):
        ev.caft_spec(c)
