"""Retain returned training completions before reward or optimizer work.

The original group UUIDs and sampling streams are untouched. Stable request IDs
are provenance only; no per-request sampler seed is invented for native vLLM.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    require(value is None or type(value) in (str, int, float, bool),
            'Unsupported rollout provenance value')
    return value


def request_id(common_identity, arm, step, prompt_slot, sample, namespace=None):
    identity = ['caft_returned_rollout_v1', common_identity, arm, step, prompt_slot, sample]
    if namespace is not None:
        require(namespace == 'random0-fresh-from120-v1' and arm == 'random0' and 121 <= step <= 200,
                'Fresh request namespace outside its declared arm/steps')
        identity = ['caft_returned_rollout_fresh_v1', namespace] + identity[1:]
    return hashlib.sha256(json.dumps(
        identity,
        separators=(',', ':')).encode()).hexdigest()


def retain_returned_batch(trainer, batch):
    cfg = trainer.config.actor_rollout_ref.actor.get('caft_checkpoint')
    if cfg is None:
        return
    config = trainer.config
    require(config.algorithm.adv_estimator == 'grpo_modified' and
            not config.algorithm.get('screening_specs', {}),
            'Retention supports the original grpo_modified with no screening')
    require(config.actor_rollout_ref.rollout.n == 16 and config.data.train_batch_size == 16,
            'Original 16 prompts by 16 completions required')
    require(config.trainer.rollout_data_dir, 'Raw training rollout output is required')
    arm = config.actor_rollout_ref.model.caft.arm
    common = cfg.common_m0.common_identity_sha256
    require(arm in ('baseline', 'pc4', 'random0', 'random1') and len(common) == 64,
            'Missing fixed training identity')
    step = int(trainer.global_steps)
    branch = None
    if cfg.get('fresh_branch') is not None:
        from src.train.verl.caft_fresh_branch import rollout_namespace
        branch = rollout_namespace(trainer)
    require(1 <= step <= 200 and 'caft_request_id' not in batch.non_tensor_batch,
            'Unexpected or already retained training batch')
    tensors = {name: batch.batch[name].detach().cpu().tolist()
               for name in ('prompts', 'responses', 'attention_mask', 'response_mask')}
    require(all(len(values) == 256 for values in tensors.values()), 'Returned rollout count differs')
    rows = []
    for i, (prompt, response, attention, response_mask) in enumerate(zip(
            tensors['prompts'], tensors['responses'], tensors['attention_mask'], tensors['response_mask'])):
        require(len(attention) == len(prompt) + len(response) and len(response_mask) == len(response),
                'Returned token/mask lengths differ')
        require(all(x in (0, 1) for x in attention + response_mask), 'Nonbinary rollout mask')
        prompt_mask = attention[:len(prompt)]
        require(prompt_mask == sorted(prompt_mask) and response_mask == sorted(response_mask, reverse=True)
                and sum(prompt_mask) > 0 and sum(response_mask) > 0,
                'Expected left-padded prompt and right-padded completion')
        require(attention[len(prompt):] == response_mask, 'Completion attention/response mask differs')
        prompt_ids = [t for t, m in zip(prompt, prompt_mask) if m]
        completion_ids = [t for t, m in zip(response, response_mask) if m]
        provenance = {key: plain(batch.non_tensor_batch[key][i])
                      for key in ('data_source', 'extra_info', 'raw_prompt', 'index', 'uid')
                      if key in batch.non_tensor_batch}
        require('extra_info' in provenance and 'uid' in provenance, 'Problem/group provenance missing')
        rows.append(dict(record_kind='returned_training_rollout_v1',
            request_id=request_id(common, arm, step, i // 16, i % 16,
                                  None if branch is None else branch['request_namespace']),
            common_identity_sha256=common, arm=arm, optimizer_step_to_follow=step,
            source_checkpoint_step=step-1, prompt_slot=i // 16, sample_index=i % 16,
            prompt_token_ids=prompt_ids, completion_token_ids=completion_ids,
            padded_prompt_token_ids=prompt, padded_completion_token_ids=response,
            attention_mask=attention, response_mask=response_mask,
            prompt_text=trainer.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            completion_text=trainer.tokenizer.decode(completion_ids, skip_special_tokens=False),
            provenance=provenance, training_seed=config.data.seed,
            sampling_seed=None, sampling_rng='native vLLM rank stream; state retained in training checkpoints',
            reward_evaluation_completed=False, optimizer_update_completed=False))
        if branch is not None:
            rows[-1].update(request_namespace=branch['request_namespace'],
                            branch_source_checkpoint_sha256=branch['source_checkpoint_sha256'])
    payload = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
                      for row in rows).encode()
    require(len(payload) <= 64 * 1024**2, 'Raw rollout batch exceeds declared bound')
    root = Path(config.trainer.rollout_data_dir) / 'returned_before_update'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'step_{step:03d}.jsonl'
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    # New metadata is carried through the original balancing permutation. It is
    # never used for the original UUID-based grouping or RNG.
    batch.non_tensor_batch['caft_request_id'] = np.array([r['request_id'] for r in rows], dtype=object)


def reward_log_with_request_ids(batch, reward_extra_infos):
    """Keep the native post-update report joinable after balancing reorders rows."""
    if 'caft_request_id' not in batch.non_tensor_batch:
        return reward_extra_infos
    result = dict(reward_extra_infos)
    require('caft_request_id' not in result, 'Conflicting reward request-ID field')
    result['caft_request_id'] = batch.non_tensor_batch['caft_request_id'].tolist()
    return result
