"""Provenance for the authorized fresh Random0 continuation from native step120.

This helper never generates, samples, changes RNG, or loads cached rollouts.
The existing native checkpoint loader restores all training and sampler state.
"""
import json
import re
from pathlib import Path

KIND = 'random0_fresh_from120_v1'
NAMESPACE = 'random0-fresh-from120-v1'
RECEIPT = 'caft_fresh_branch.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def declaration(cfg):
    value = cfg.get('fresh_branch')
    if value is None:
        return None
    require(cfg.get('cached_replay') is None and cfg.get('backward_repeat') is None,
            'Fresh branch cannot use cached replay or backward diagnostic')
    require(set(value) == {'kind', 'request_namespace', 'source_checkpoint_sha256', 'source_profile'},
            'Fresh branch declaration fields differ')
    result = dict(value); result['source_profile'] = dict(value['source_profile'])
    source = result['source_profile']
    require(result['kind'] == KIND and result['request_namespace'] == NAMESPACE,
            'Fresh Random0 branch kind/namespace differs')
    require(re.fullmatch('[0-9a-f]{64}', result['source_checkpoint_sha256'] or '') is not None,
            'Fresh branch source manifest binding missing')
    require(set(source) == {'kind', 'training_identity_sha256', 'base_manifest_sha256'} and
            source['kind'] == 'compact_lora_training_state_v1' and
            all(re.fullmatch('[0-9a-f]{64}', source[k] or '') is not None
                for k in ('training_identity_sha256', 'base_manifest_sha256')),
            'Fresh branch source profile differs')
    destination = {k: cfg[k] for k in source}
    require(destination['kind'] == source['kind'] and
            destination['base_manifest_sha256'] == source['base_manifest_sha256'] and
            destination['training_identity_sha256'] != source['training_identity_sha256'] and
            re.fullmatch('[0-9a-f]{64}', destination['training_identity_sha256'] or '') is not None and
            cfg.get('resume_source_profile') == source and
            cfg.get('hardware_profile') == {'kind': 'ada32_caft_v1', 'world_size': 4},
            'Fresh branch must retain source base/profile and declare its new four-rank identity')
    return result


def lineage(cfg, checkpoint_step):
    value = declaration(cfg); require(value is not None, 'Fresh branch absent')
    require(type(checkpoint_step) is int and 120 <= checkpoint_step <= 200,
            'Fresh branch checkpoint step outside120..200')
    common = cfg.get('common_m0', {}).get('common_identity_sha256', '')
    require(re.fullmatch('[0-9a-f]{64}', common) is not None, 'Fresh branch common identity missing')
    return dict(value, branch_profile={k: cfg[k] for k in value['source_profile']},
                common_identity_sha256=common, source_checkpoint_step=120,
                fresh_updates=[121, 200], replacement_steps=[121, 122, 123],
                sampler_state_source='saved_checkpoint120', cached_rollouts_used=False,
                bitwise_old_tail_equivalence=False, checkpoint_step=checkpoint_step)


def validate_resume(cfg, manifest, path):
    """Validate source120 or a later manifest-bound checkpoint of this branch."""
    from src.train.verl import caft_checkpoint as compact
    value = declaration(cfg); require(value is not None, 'Fresh branch absent')
    step = manifest.get('global_step')
    require(manifest.get('status') == 'complete_training_boundary' and
            manifest.get('world_size') == 4 and manifest.get('topology_continuation') is None and
            manifest.get('exact_stochastic_resume_available') is True and
            manifest.get('optimizer_steps_completed') == step and type(step) is int,
            'Fresh branch requires a native complete four-rank checkpoint')
    require(compact.ref(Path(path) / 'CAFT_CHECKPOINT.json')['sha256'] == cfg.get('resume_manifest_sha256'),
            'Fresh branch external resume manifest changed')
    if step == 120:
        require(cfg['resume_manifest_sha256'] == value['source_checkpoint_sha256'] and
                manifest['profile'] == value['source_profile'] and RECEIPT not in manifest['files'],
                'Fresh branch must start from exactly its declared source120')
    else:
        expected = lineage(cfg, step)
        require(130 <= step <= 200 and step % 10 == 0 and
                manifest['profile'] == expected['branch_profile'] and
                manifest['files'].get(RECEIPT) == compact.ref(Path(path) / RECEIPT),
                'Fresh descendant checkpoint lineage/profile missing or changed')
        require(json.loads((Path(path) / RECEIPT).read_bytes()) == expected,
                'Fresh descendant checkpoint receipt changed')
    return True


def prepare_coordinator(trainer, manifest, path):
    cfg = trainer.config.actor_rollout_ref.actor.caft_checkpoint
    validate_resume(cfg, manifest, path)
    config = trainer.config
    require(config.actor_rollout_ref.model.caft.arm == 'random0' and
            config.trainer.n_gpus_per_node == 4 and config.trainer.nnodes == 1 and
            config.data.train_batch_size == 16 and config.actor_rollout_ref.rollout.n == 16 and
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4 and
            config.trainer.resume_mode == 'resume_path' and
            config.trainer.get('caft_pilot_stop_step') is None and
            config.trainer.get('caft_backward_repeat') is None,
            'Fresh branch requires unchanged Random0 four-rank/256-rollout training through200')
    trainer._caft_fresh_branch_lineage = lineage(cfg, int(manifest['global_step']))


def rollout_namespace(trainer):
    cfg = trainer.config.actor_rollout_ref.actor.caft_checkpoint
    value = declaration(cfg)
    installed = getattr(trainer, '_caft_fresh_branch_lineage', None)
    require(value is not None and 121 <= trainer.global_steps <= 200 and
            isinstance(installed, dict) and installed == lineage(cfg, installed['checkpoint_step']),
            'Fresh rollout lacks verified restored branch lineage')
    return value


def checkpoint_receipt(trainer):
    rollout_namespace(trainer)
    require(trainer.global_steps >= 130 and trainer.global_steps % 10 == 0,
            'Fresh branch seals only completed130..200 boundaries')
    return lineage(trainer.config.actor_rollout_ref.actor.caft_checkpoint, trainer.global_steps)
