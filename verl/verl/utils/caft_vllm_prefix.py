"""vLLM0.11 prompt-only prefix reuse for the fixed synchronous CAFT profile.

No patched sampler, model, attention or cache allocator. Native cached-block
lookup sees a read-only prompt-length view, never a generated-token suffix.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from numbers import Integral
from pathlib import Path

PROTOCOL = 'caft_vllm011_prompt_prefix_v1'
PINNED_SOURCES = {
    'v1/core/kv_cache_manager.py': 'f02d1d095365b83c5fcb9eff14ac887f35234495b8718f24718a218244dd5590',
    'v1/core/kv_cache_coordinator.py': '19285af400c0525b0855388a7a635111ec01bf728acaba9b72fbd793ed38b2dc',
    'v1/core/block_pool.py': '8cd4b2fe35a26dcf8aa7d8e8ec16922b7cd3e67e02c800026902dad9b1274368',
    'v1/core/kv_cache_utils.py': '0824beae09a3a8faeb54e8e0605d1661045e34a33a2026de537e6849a4c78ee4',
    'v1/request.py': 'be457706757fe6734c5a1bd9f3261722276ec27616dac996d6b1c33cacdfa5ec',
    'inputs/data.py': '66511915d094cbb963d97a21f1a28720de4d89abcb1f88d1b93b4bc73d58ae9a',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_sources():
    distribution = importlib.metadata.distribution('vllm')
    require(distribution.version == '0.11.0', 'Prefix bridge requires pinned vLLM0.11')
    for name, expected in PINNED_SOURCES.items():
        path = Path(distribution.locate_file('vllm/' + name))
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                'Prefix cache source changed: ' + name)


def tokens(value):
    require(isinstance(value, (list, tuple)) and 0 < len(value) <= 1536,
            'Prefix cache requires the fixed nonempty prompt scope')
    require(all(isinstance(x, Integral) and not isinstance(x, bool) and 0 <= x < 2**31 for x in value),
            'Prefix cache prompt token type/range differs')
    return [int(x) for x in value]


class _PromptLookupView:
    """Only num_tokens differs; original Request and block hashes never mutate."""
    def __init__(self, request):
        self._request = request
        self.num_tokens = len(request.prompt_token_ids)

    def __getattr__(self, name):
        return getattr(self._request, name)


class PrefixPolicy:
    def __init__(self, manager, spec):
        require(spec.get('rollout_prefix_caching') is True,
                'Prefix cache requires its explicit matched-arm profile')
        require(manager.enable_caching is True and manager.use_eagle is False and
                manager.num_kv_cache_groups == 1 and type(manager.block_size) is int and manager.block_size > 0,
                'Prefix bridge supports one native full-attention block group only')
        self.manager = manager
        self.spec = json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))
        self.scope = None
        self.allowed = set()
        self.lookups = self.hits = self.hit_tokens = 0
        self.original = manager.get_computed_blocks
        manager.get_computed_blocks = self.lookup

    def salt(self, prompt):
        require(self.scope is not None, 'Prefix request outside declared rollout')
        return digest({'protocol': PROTOCOL, 'scope': self.scope, 'full_prompt_tokens': prompt})

    def lookup(self, request):
        prompt = tokens(request.prompt_token_ids)
        require(self.scope is not None and digest(prompt) in self.allowed and
                request.cache_salt == self.salt(prompt), 'Prefix prompt/scope salt mismatch')
        require(not request.mm_features and type(request.num_tokens) is int and
                request.num_tokens >= len(prompt), 'Prefix request shape changed')
        # Native get_computed_blocks uses num_tokens - 1, including on preemption.
        # The view limits this to prompt length - 1 without editing the request.
        blocks, count = self.original(_PromptLookupView(request))
        maximum = ((len(prompt) - 1) // self.manager.block_size) * self.manager.block_size
        require(type(count) is int and 0 <= count <= maximum and count % self.manager.block_size == 0 and
                len(blocks.blocks) == 1 and len(blocks.blocks[0]) * self.manager.block_size == count,
                'Native prefix lookup reused final-prompt predictor or malformed blocks')
        self.lookups += 1
        self.hits += int(count > 0)
        self.hit_tokens += count
        return blocks, count


def _parts(engine):
    frontend = engine.llm_engine
    require(type(frontend.engine_core).__name__ == 'InprocClient', 'Prefix bridge requires colocated V1 engine')
    core = frontend.engine_core.engine_core
    scheduler = core.scheduler
    require(core.batch_queue is None and not core.use_spec_decode and
            frontend.model_executor is core.model_executor and scheduler.connector is None,
            'Prefix bridge requires synchronous text engine without external KV')
    hasher = core.request_block_hasher
    require(callable(hasher) and hasher.__name__ == 'request_block_hasher' and
            hasher.__module__ == 'vllm.v1.core.kv_cache_utils', 'Prefix cache hasher differs')
    return frontend, core, scheduler


def _drained(engine):
    frontend, core, scheduler = _parts(engine)
    require(not frontend.has_unfinished_requests() and not scheduler.requests and
            not scheduler.running and not scheduler.waiting and scheduler.get_num_unfinished_requests() == 0 and
            not frontend.output_processor.request_states and not frontend.output_processor.parent_requests,
            'Prefix reset attempted with unfinished requests')
    return scheduler


def _reset(engine):
    scheduler = _drained(engine)
    # LLM/EngineCore wrappers discard this bool; use the pinned native scheduler.
    require(scheduler.reset_prefix_cache() is True, 'Native prefix reset failed')
    pool = scheduler.kv_cache_manager.block_pool
    require(pool.num_gpu_blocks - pool.get_num_free_blocks() == 1 and
            not pool.cached_block_hash_to_block._cache and all(b.block_hash is None for b in pool.blocks),
            'Prefix reset left live or hashed KV blocks')


def install(engine, spec):
    validate_sources()
    require(not hasattr(engine, '_caft_prefix_policy'), 'Duplicate prefix policy installation')
    scheduler = _drained(engine)
    policy = PrefixPolicy(scheduler.kv_cache_manager, spec)
    engine._caft_prefix_policy = policy
    _reset(engine)
    return {'protocol': PROTOCOL, 'installed': True, 'source_hashes': dict(PINNED_SOURCES),
            'last_prompt_predictor_reuse': False, 'cache_reset_verified': True}


def validate_engine(engine):
    validate_sources()
    _, _, scheduler = _parts(engine)
    policy = engine._caft_prefix_policy
    require(isinstance(policy, PrefixPolicy) and policy.manager is scheduler.kv_cache_manager and
            scheduler.kv_cache_manager.get_computed_blocks == policy.lookup,
            'Installed prefix policy or native manager binding changed')


def prepare_inputs(engine, prompts, *, step, evaluation=False):
    policy = engine._caft_prefix_policy
    require(isinstance(policy, PrefixPolicy) and policy.scope is None,
            'Overlapping prefix rollout scopes')
    require(type(step) is int and step >= 0 and type(evaluation) is bool,
            'Prefix cache requires actual model-step/role metadata')
    require(isinstance(prompts, list) and 0 < len(prompts) <= 256,
            'Prefix prompt count differs from bounded pilot')
    parsed = []
    for prompt in prompts:
        require(isinstance(prompt, dict) and set(prompt) == {'prompt_token_ids'},
                'Unexpected prompt metadata or preexisting cache salt')
        parsed.append(tokens(prompt['prompt_token_ids']))
    _reset(engine)  # No previous model weights or intervention scope can survive.
    policy.scope = {'spec': policy.spec, 'step': step, 'evaluation': evaluation}
    policy.allowed = {digest(x) for x in parsed}
    policy.lookups = policy.hits = policy.hit_tokens = 0
    return [{'prompt_token_ids': list(x), 'cache_salt': policy.salt(x)} for x in parsed]


def finish(engine):
    policy = engine._caft_prefix_policy
    require(isinstance(policy, PrefixPolicy) and policy.scope is not None, 'No active prefix scope')
    _reset(engine)  # Fail closed if generation did not actually drain.
    receipt = {'protocol': PROTOCOL, 'step': policy.scope['step'],
               'evaluation': policy.scope['evaluation'], 'scope_sha256': digest(policy.scope),
               'lookups': policy.lookups, 'hit_requests': policy.hits, 'hit_tokens': policy.hit_tokens,
               'cache_reset_verified': True, 'last_prompt_predictor_reuse': False}
    policy.scope = None
    policy.allowed = set()
    engine.caft_last_prefix_receipt = receipt
    return receipt


def reset_boundary(engine):
    validate_engine(engine)
    policy = engine._caft_prefix_policy
    require(isinstance(policy, PrefixPolicy) and policy.scope is None,
            'Weight/checkpoint boundary inside active prefix scope')
    _reset(engine)
    return {'protocol': PROTOCOL, 'cache_reset_verified': True, 'prefix_cache_retained': False,
            'source_hashes': dict(PINNED_SOURCES)}
