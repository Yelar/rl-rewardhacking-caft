"""Common, realized step-zero state for exactly three CAFT pilot arms.

This is initialization, not same-arm resume. Bound trusted Torch checkpoints are
read only after their byte hashes pass. No frozen base weights are loaded here.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from . import caft_checkpoint as compact

require = compact.require
KIND = 'common_caft_m0_v1'
ADA_KIND = 'common_caft_m0_ada_baseline_v2'
ADA_CAPTURE_OWNER = 'pc4'
FACTORIAL_KIND = 'caft_lora_a_initialization_ada_m0_v1'
ARMS = {'baseline', 'pc4', 'random0'}
ADOPTION_KIND = 'r6_source5_m0_to_source7_graph_prefix_v1'
ADOPTION_COMMON_SHA = '7f58163a6a2147e7720ca4b6499e2eef6eff598ecba09e8c7290163a4b3ac66c'
ADOPTION_CONFIG_SHA = 'fc110bd025f649dc1c8633beafe5b932ffe8ae835b0c29af143114a758ed52e4'
GRAPH_CONFIG = {'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64]}
ADOPTION_INPUT_CHANGES = {'source_inventory', 'projection_qualification', 'pilot_profile', 'runtime_proof'}
# Best-case study step 4 (protocol 7.4 / 8.3): one common step-zero state per seed. The declared capture arm
# (A0) saves its realized step zero; every other arm of that seed loads it before its first request. The seed's
# data order, LoRA start and rollout generation stream all derive from SHA-256 of the study seed.
V2_KIND = 'bestcase_v2_seed_m0_v1'
V2_STUDY = 'bestcase-caft-20260928'


def seed_components(seed):
    require(type(seed) is int and 1 <= seed <= 100, 'Study seed must be an integer in 1..100')
    def component(name):
        return int(hashlib.sha256(f'{V2_STUDY}|seed={seed}|{name}'.encode()).hexdigest()[:8], 16) % (2**31 - 4096)
    return {'data': component('data'), 'lora_init': component('lora_init'), 'rollout_gen': component('rollout_gen')}


def plain(value):
    if isinstance(value, dict) or value is None:
        return copy.deepcopy(value)
    from omegaconf import OmegaConf
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def digest(value):
    """Content digest, independent of pickle storage IDs and destination device."""
    from omegaconf import ListConfig, OmegaConf
    from torch.torch_version import TorchVersion
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            v = compact.local_tensor(v).detach().cpu().contiguous()
            h.update(b'tensor'); add(str(v.dtype)); add(list(v.shape))
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(b'ndarray'); add(v.dtype.str); add(list(v.shape)); h.update(v.tobytes())
        elif isinstance(v, np.generic):
            add(v.item())
        elif isinstance(v, dict):
            h.update(b'dict')
            for key in sorted(v, key=lambda x: (type(x).__name__, str(x))): add(key); add(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode()); add(len(v))
            for item in v: add(item)
        elif type(v) is ListConfig:
            # Native AdamW retains OmegaConf betas; bind every resolved element.
            add(OmegaConf.to_container(v, resolve=True, throw_on_missing=True))
        elif type(v) is TorchVersion:
            # Native vLLM records torch.__version__ using this exact str subclass.
            add(str(v))
        else:
            require(v is None or type(v) in (bool, int, float, str), 'Unsupported state digest type')
            h.update(json.dumps([type(v).__name__, v], allow_nan=False, separators=(',', ':')).encode())
    add(value)
    return h.hexdigest()


def worker_config_value(config, initialized=False, v2=False):
    value = plain(config)
    # The original FSDP actor constructor normalizes 16*16/8 to32 (and 16*16/1 to 256 on one rank).
    # Best-case wave 2: v2 also runs on 2 or 4 ranks per run (16*16/2 = 128, 16*16/4 = 64); virtual-8 balancing
    # keeps the 8-rank micro-batch grouping on every one of these layouts.
    allowed = ((32, 64, 128, 256) if initialized else (16,)) if v2 else ((32,) if initialized else (16,))
    require(value['rollout']['n'] == 16 and value['actor']['ppo_micro_batch_size'] is None and
            value['actor']['ppo_micro_batch_size_per_gpu'] == 4 and
            value['actor']['ppo_mini_batch_size'] in allowed,
            'Common initializer requires the declared16x16/eight-rank microbatch profile')
    value['actor']['ppo_mini_batch_size'] = 16
    # The only worker differences permitted across arms are Q/policy and the
    # initialization/checkpoint provenance that necessarily names that arm.
    value['model'].pop('caft', None)
    value['actor'].pop('caft_checkpoint', None)
    return value


def worker_config_digest(config, initialized=False, v2=False):
    return digest(worker_config_value(config, initialized, v2))


def coordinator_config_value(config, v2=False):
    value = plain(config)
    value['actor_rollout_ref']['model'].pop('caft', None)
    value['actor_rollout_ref']['actor'].pop('caft_checkpoint', None)
    for key in ('default_local_dir', 'default_hdfs_dir', 'rollout_data_dir',
                'validation_data_dir', 'experiment_name', 'resume_from_path'):
        value['trainer'].pop(key, None)
    value['reward_model']['reward_kwargs']['caft_reward_sandbox'].pop('spool_dir', None)
    if v2:
        # Arm fields: the activation penalty and its scoring worker (penalty arms only). The scoring GPU changes
        # Ray resources, never the policy's algorithm, data, batch or sampling.
        value['reward_model']['reward_kwargs'].pop('caft_penalty', None)
        value.pop('activation_cache', None)
        ray_init = value.get('ray_kwargs', {}).get('ray_init', {})
        for key in ('num_gpus', 'resources', '_temp_dir'):
            ray_init.pop(key, None)
        # Host scheduling of concurrent runs: each run's reward sandbox gets its own CPU set.
        value['reward_model']['reward_kwargs']['caft_reward_sandbox'].pop('cpu_set', None)
    return value


def coordinator_config_digest(config, v2=False):
    return digest(coordinator_config_value(config, v2))


def optimized_successor_config(source):
    """Exact reviewed metadata delta; no recursive dropping of config fields."""
    value = plain(source)
    rollout = value['actor_rollout_ref']['rollout']
    require(rollout['enforce_eager'] is True and rollout['enable_prefix_caching'] is False
            and rollout['cudagraph_capture_sizes'] is None
            and rollout['engine_kwargs']['vllm'] == {}, 'Unexpected source5 rollout profile')
    rollout['enforce_eager'] = False
    rollout['enable_prefix_caching'] = True
    rollout['engine_kwargs']['vllm']['compilation_config'] = copy.deepcopy(GRAPH_CONFIG)
    sandbox = value['reward_model']['reward_kwargs']['caft_reward_sandbox']
    root = Path(sandbox['source_dir'])
    require(root.is_absolute() and root.name == 'source_v5', 'Expected source5 reward root')
    destination = root.with_name('source_v7')
    sandbox['source_dir'] = str(destination)
    variables = value['ray_kwargs']['ray_init']['runtime_env']['env_vars']
    require(variables['PYTHONPATH'] == str(root) + ':' + str(root / 'verl'), 'Source5 worker import path differs')
    variables['PYTHONPATH'] = str(destination) + ':' + str(destination / 'verl')
    custom = value['custom_reward_function']
    if custom['path'] is not None:
        require(custom['path'] == str(root / 'src/train/verl/rewards.py'), 'Unexpected custom reward source')
        custom['path'] = str(destination / 'src/train/verl/rewards.py')
    return value


def validate_adoption(decl, config=None, worker=False):
    """Verify both contracts and the complete permitted config/input delta."""
    adoption = decl['adoption']
    require(decl['mode'] == 'load' and set(adoption) == {
        'kind', 'source_common_manifest', 'source_config', 'source_contract',
        'source_common_identity_sha256', 'destination_common_identity_sha256', 'changed_inputs'},
        'Explicit common-M0 adoption fields')
    require(adoption['kind'] == ADOPTION_KIND and adoption['source_common_manifest'] == decl['common_manifest']
            and decl['common_manifest']['sha256'] == ADOPTION_COMMON_SHA
            and adoption['source_config']['sha256'] == ADOPTION_CONFIG_SHA,
            'Adoption must use the exact verified r6 common-M0/config')
    original = read_bound(decl['common_manifest'])
    source = read_bound(adoption['source_config'])
    source_contract = adoption['source_contract']
    require(original['contract'] == source_contract and original['common_identity_sha256'] == digest(source_contract)
            and adoption['source_common_identity_sha256'] == original['common_identity_sha256']
            and source_contract['worker_config_sha256'] == worker_config_digest(source['actor_rollout_ref'])
            and source_contract['coordinator_config_sha256'] == coordinator_config_digest(source),
            'Old common contract is not the bound source5 configuration')
    contract = decl['contract']
    expected = optimized_successor_config(source)
    require(contract['worker_config_sha256'] == worker_config_digest(expected['actor_rollout_ref'])
            and contract['coordinator_config_sha256'] == coordinator_config_digest(expected)
            and adoption['destination_common_identity_sha256'] == decl['common_identity_sha256'] == digest(contract),
            'Adoption changed fields outside exact graph/prefix/source-path delta')
    before, after = source_contract['inputs'], contract['inputs']
    require(set(before) == set(after), 'Adoption input names differ')
    changes = {name: {'source': before[name], 'destination': after[name]}
               for name in before if before[name] != after[name]}
    require(set(changes).issubset(ADOPTION_INPUT_CHANGES) and changes == adoption['changed_inputs']
            and 'source_inventory' in changes, 'Adoption changed a scientific input or omitted provenance')
    if config is not None:
        actual = worker_config_value(config, initialized=True) if worker else coordinator_config_value(config)
        wanted = worker_config_value(expected['actor_rollout_ref']) if worker else coordinator_config_value(expected)
        require(actual == wanted, 'Current config differs from exact adopted execution profile')
    return original


def adoption_declaration(config, contract, common_manifest, source_config):
    """Builder-owned explicit adoption; all three arms load the same old M0."""
    original = read_bound(common_manifest)
    before, after = original['contract']['inputs'], contract['inputs']
    declaration = {'kind': KIND, 'mode': 'load', 'contract': plain(contract),
        'common_identity_sha256': digest(contract), 'common_manifest': plain(common_manifest),
        'adoption': {'kind': ADOPTION_KIND, 'source_common_manifest': plain(common_manifest),
            'source_config': plain(source_config), 'source_contract': plain(original['contract']),
            'source_common_identity_sha256': original['common_identity_sha256'],
            'destination_common_identity_sha256': digest(contract),
            'changed_inputs': {name: {'source': before[name], 'destination': after[name]}
                               for name in before if name in after and before[name] != after[name]}}}
    validate_adoption(declaration, config)
    return declaration


def factorial_declaration(config, inputs, factorial):
    from .caft_factorial import declaration_for_config
    return declaration_for_config(config, inputs, factorial)


def load_factorial_anchor(factorial):
    from .caft_factorial import load_anchor
    return load_anchor(factorial)


def declaration(config, worker=False):
    cfg = config.actor.caft_checkpoint if worker else config.actor_rollout_ref.actor.caft_checkpoint
    value = plain(cfg.get('common_m0'))
    if isinstance(value, dict) and value.get('kind') == FACTORIAL_KIND:
        from .caft_factorial import validate_declaration
        return validate_declaration(config, worker=worker)
    if isinstance(value, dict) and value.get('kind') == V2_KIND:
        return validate_v2_declaration(config, value, worker)
    require(isinstance(value, dict) and value.get('kind') in {KIND, ADA_KIND},
            'Explicit common-M0 declaration required')
    fresh_ada = value['kind'] == ADA_KIND
    require(not fresh_ada or 'adoption' not in value, 'Fresh Ada M0 cannot adopt old runtime state')
    mode = value.get('mode')
    keys = {'kind', 'mode', 'common_identity_sha256', 'contract'} | ({'common_manifest'} if mode == 'load' else set())
    if 'adoption' in value:
        keys.add('adoption')
    require(set(value) == keys and mode in ('capture', 'load'), 'Common-M0 declaration fields')
    contract = value['contract']
    contract_keys = {'worker_config_sha256', 'coordinator_config_sha256', 'inputs'}
    if fresh_ada:
        contract_keys.add('capture_owner')
    require(set(contract) == contract_keys, 'Common contract fields')
    if fresh_ada:
        require(contract['capture_owner'] == ADA_CAPTURE_OWNER, 'Fresh Ada capture owner must be PC4')
    require(value['common_identity_sha256'] == digest(contract), 'Common identity digest differs')
    actual = worker_config_digest(config, initialized=True) if worker else coordinator_config_digest(config)
    require(contract['worker_config_sha256' if worker else 'coordinator_config_sha256'] == actual,
            'Current algorithm/data/hardware config differs from common contract')
    require(isinstance(contract['inputs'], dict) and contract['inputs'], 'Common source/runtime/data inputs missing')
    arm = config.model.caft.arm if worker else config.actor_rollout_ref.model.caft.arm
    if fresh_ada:
        require(arm in {'pc4', 'random0'} and (mode != 'capture' or arm == ADA_CAPTURE_OWNER),
                'Only declared PC4 producer captures fresh Ada M0')
    else:
        require(arm in ARMS and (mode != 'capture' or arm == 'baseline'), 'Only baseline captures common M0')
    if 'adoption' in value:
        validate_adoption(value, config, worker)
    return value


def validate_v2_declaration(config, value, worker=False):
    mode = value.get('mode')
    keys = {'kind', 'mode', 'seed', 'capture_arm', 'common_identity_sha256', 'contract'} | (
        {'common_manifest'} if mode == 'load' else set())
    require(set(value) == keys and mode in ('capture', 'load'), 'v2 common-M0 declaration fields')
    contract = value['contract']
    require(set(contract) == {'worker_config_sha256', 'coordinator_config_sha256', 'inputs', 'seed',
                              'seed_components'}, 'v2 common contract fields')
    require(contract['seed'] == value['seed'] and contract['seed_components'] == seed_components(value['seed']),
            'v2 seed components differ from their SHA-256 derivation')
    require(value['common_identity_sha256'] == digest(contract), 'Common identity digest differs')
    actual = worker_config_digest(config, initialized=True, v2=True) if worker else coordinator_config_digest(config, True)
    require(contract['worker_config_sha256' if worker else 'coordinator_config_sha256'] == actual,
            'Current algorithm/data/hardware config differs from common contract')
    require(isinstance(contract['inputs'], dict) and contract['inputs'], 'Common source/runtime/data inputs missing')
    model = config.model if worker else config.actor_rollout_ref.model
    spec = plain(model.caft)
    require(isinstance(spec, dict) and spec.get('protocol') == 'bestcase_multilayer_response_predictor_caft_v2',
            'v2 common start requires a v2 CAFT specification')
    require(isinstance(value['capture_arm'], str) and (mode != 'capture' or spec['arm'] == value['capture_arm']),
            'Only the declared capture arm captures a seed start')
    parts = contract['seed_components']
    require(model.get('lora_init_seed') == parts['lora_init'] and model.get('rollout_gen_seed') == parts['rollout_gen'],
            'LoRA/rollout-stream seeds differ from the seed components')
    require(worker or config.data.seed == parts['data'], 'Data-order seed differs from the seed component')
    return value


def v2_declaration(config, mode, seed, capture_arm, inputs, common_manifest=None):
    """Builder-owned v2 declaration for one arm of one seed (the builder never guesses a manifest)."""
    contract = {'worker_config_sha256': worker_config_digest(config['actor_rollout_ref'], initialized=False, v2=True),
                'coordinator_config_sha256': coordinator_config_digest(config, True),
                'inputs': plain(inputs), 'seed': seed, 'seed_components': seed_components(seed)}
    value = {'kind': V2_KIND, 'mode': mode, 'seed': seed, 'capture_arm': capture_arm, 'contract': contract,
             'common_identity_sha256': digest(contract)}
    if mode == 'load':
        require(common_manifest is not None, 'Load arms need the capture arm\'s COMMON_M0 reference')
        value['common_manifest'] = plain(common_manifest)
    return value


def read_bound(binding, torch_value=False):
    require(set(binding) == {'path', 'sha256', 'size_bytes'}, 'Exact checkpoint file reference required')
    path = Path(binding['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Bound checkpoint path invalid')
    with path.open('rb') as stream:
        h = hashlib.sha256(); size = 0
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block); size += len(block)
        require(h.hexdigest() == binding['sha256'] and size == binding['size_bytes'], 'Bound checkpoint bytes differ')
        stream.seek(0)
        return torch.load(stream, map_location='cpu', weights_only=False) if torch_value else json.load(stream)


def child_binding(root, name, manifest):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Checkpoint child path invalid')
    return {'path': str(root / path), **manifest['files'][name]}


def load_common(decl):
    value = validate_adoption(decl) if 'adoption' in decl else read_bound(decl['common_manifest'])
    expected_contract = decl['adoption']['source_contract'] if 'adoption' in decl else decl['contract']
    expected_identity = decl['adoption']['source_common_identity_sha256'] if 'adoption' in decl else decl['common_identity_sha256']
    fresh_ada = decl['kind'] == ADA_KIND
    v2 = decl['kind'] == V2_KIND
    expected_owner = decl['contract']['capture_owner'] if fresh_ada else decl['capture_arm'] if v2 else 'baseline'
    require(not fresh_ada or value.get('kind') == ADA_KIND, 'Fresh Ada source kind differs')
    require(not v2 or (value.get('kind') == V2_KIND and value.get('seed') == decl['seed']), 'v2 source kind/seed differs')
    require(value['status'] == 'complete_common_caft_m0' and value['source_arm'] == expected_owner and
            value['common_identity_sha256'] == expected_identity and value['contract'] == expected_contract,
            'Common source identity differs')
    checkpoint = read_bound(value['checkpoint'])
    require(not fresh_ada or checkpoint['world_size'] == 8, 'Fresh Ada common source must have eight ranks')
    require(checkpoint['status'] == 'complete_training_boundary' and checkpoint['global_step'] == 0 and
            checkpoint['optimizer_steps_completed'] == 0 and checkpoint['exact_stochastic_resume_available'] is True and
            checkpoint['base_weights_saved'] is False and checkpoint['profile'] == value['source_profile'],
            'Common source is not complete exact stochastic step zero')
    return value, checkpoint, Path(value['checkpoint']['path']).parent


def rollout_state_content(value):
    # PID and arm/Q are expected to differ. Everything affecting stochastic
    # replay, including installed source/runtime and rank device, is retained.
    value = copy.deepcopy(value)
    value['worker'].pop('pid', None); value['worker'].pop('caft_spec', None)
    value.pop('zero_token_metadata_drain', None)
    return value


def rank_content(payload, optimizer, extra):
    require(payload['global_step'] == 0 and payload['rollout_state_restorable'] is True,
            'Common rank is not restorable step zero')
    require(payload['rollout_state']['request_counter'] == 0, 'Common M0 already generated requests')
    require(optimizer['state'] == {}, 'Common optimizer contains completed update state')
    require(extra.get('lr_scheduler') is not None and 'rng' in extra, 'Common scheduler/general RNG missing')
    for name, tensor in payload['trainables'].items():
        require(bool(torch.isfinite(tensor).all()), 'Nonfinite common LoRA')
        if '.lora_B.' in name: require(bool((tensor == 0).all()), 'Common M0 LoRA B is nonzero')
    return {'trainables': payload['trainables'], 'optimizer': optimizer, 'extra': extra,
            'torch_random_states': payload['torch_random_states'], 'gen_random_states': payload['gen_random_states'],
            'rollout': rollout_state_content(payload['rollout_state'])}


def load_rank(root, manifest, rank):
    suffix = f"world_size_{manifest['world_size']}_rank_{rank}.pt"
    values = [read_bound(child_binding(root, 'actor/' + prefix + suffix, manifest), True)
              for prefix in ('caft_local_', 'optim_', 'extra_state_')]
    payload, optimizer, extra = values
    require(payload['rank'] == rank and payload['world_size'] == manifest['world_size'] and
            payload['profile'] == manifest['profile'], 'Rank topology/profile differs')
    return payload, optimizer, extra


def initialize_worker(worker, supplied):
    if plain(supplied).get('kind') == FACTORIAL_KIND:
        from .caft_factorial import initialize_worker as initialize_factorial_worker
        return initialize_factorial_worker(worker, supplied)
    decl = declaration(worker.config, worker=True)
    require(plain(supplied) == decl and decl['mode'] == 'load', 'Worker common initialization declaration differs')
    value, manifest, root = load_common(decl)
    cfg = compact.profile(worker)
    require(manifest['world_size'] == worker.world_size and (decl['kind'] == V2_KIND or worker.world_size == 8) and
            manifest['profile']['base_manifest_sha256'] == cfg['base_manifest_sha256'], 'Common base/topology differs')
    payload, optimizer, extra = load_rank(root, manifest, int(worker.rank))
    expected = digest(rank_content(payload, optimizer, extra))
    require(expected == value['rank_state_sha256'][str(worker.rank)], 'Common rank semantic digest differs')
    manager = worker.checkpoint_manager
    require(manager.optimizer.state_dict()['state'] == {} and
            digest(manager.optimizer.state_dict()['param_groups']) == digest(optimizer['param_groups']) and
            digest(manager.lr_scheduler.state_dict()) == digest(extra['lr_scheduler']),
            'Destination optimizer/scheduler is not matching fresh M0')
    before_spec = plain(worker.config.model.caft)
    base_preloaded = getattr(worker, 'base_sync_done', None)
    require(type(base_preloaded) is bool, 'Native base preload state is missing')
    compact.restore_local_trainables(worker.actor_module_fsdp, payload['trainables'])
    manager.optimizer.load_state_dict(optimizer); manager.lr_scheduler.load_state_dict(extra['lr_scheduler'])
    worker.torch_random_states = payload['torch_random_states'].clone()
    worker.gen_random_states = payload['gen_random_states'].clone()
    worker.rollout.caft_restore_common_initial_state(payload['rollout_state'])
    manager.load_rng_state(extra['rng'])
    # Only LoRA and stochastic state were restored; the frozen preloaded base is unchanged.
    require(worker.base_sync_done is base_preloaded, 'Common restore changed native base preload state')
    actual = dict(payload, trainables=compact.capture_local_trainables(worker.actor_module_fsdp),
                  torch_random_states=worker.torch_random_states, gen_random_states=worker.gen_random_states,
                  rollout_state=worker.rollout.caft_export_checkpoint_state())
    actual_extra = {'rng':manager.get_rng_state(), 'lr_scheduler':manager.lr_scheduler.state_dict()}
    require(plain(worker.config.model.caft) == before_spec and
            digest(rank_content(actual, manager.optimizer.state_dict(), actual_extra)) == expected,
            'Restored common state differs or destination Q changed')
    return {'rank':int(worker.rank), 'state_sha256':expected, 'destination_arm':before_spec['arm'],
            'status':'initialized_common_m0_before_first_request'}


def prepare_coordinator(trainer):
    decl = declaration(trainer.config)
    if decl['kind'] == FACTORIAL_KIND:
        from .caft_factorial import prepare_coordinator as prepare_factorial_coordinator
        return prepare_factorial_coordinator(trainer, decl)
    require(trainer.global_steps == 0 and trainer.config.trainer.resume_mode == 'disable',
            'Common initialization cannot use resume_path or a later step')
    for binding in decl['contract']['inputs'].values():
        require(compact.ref(binding['path']) == {k:binding[k] for k in ('sha256', 'size_bytes')},
                'Common source/runtime/data input bytes differ')
    if 'adoption' in decl:
        for binding in decl['adoption']['source_contract']['inputs'].values():
            require(compact.ref(binding['path']) == {k: binding[k] for k in ('sha256', 'size_bytes')},
                    'Original adopted source/runtime/data input bytes differ')
    if decl['mode'] == 'capture':
        if decl['kind'] == V2_KIND:
            from . import caft_sampler
            caft_sampler.initialize_v2_zero(trainer, capture=True)
        return
    value, manifest, root = load_common(decl)
    acknowledgements = trainer.actor_rollout_wg.caft_initialize_common_m0(decl)
    require(len(acknowledgements) == manifest['world_size'] and
            {r['rank'] for r in acknowledgements} == set(range(manifest['world_size'])) and
            all(r['state_sha256'] == value['rank_state_sha256'][str(r['rank'])] and
                r['status'] == 'initialized_common_m0_before_first_request' for r in acknowledgements),
            'Common worker acknowledgements incomplete')
    data = read_bound(child_binding(root, 'data.pt', manifest), True)
    trainer.train_dataloader.load_state_dict(data)
    require(digest(trainer.train_dataloader.state_dict()) == value['data_state_sha256'], 'Dataloader restore differs')
    rng = read_bound(child_binding(root, 'caft_coordinator_rng.pt', manifest), True)
    torch.set_rng_state(rng['torch']); np.random.set_state(rng['numpy']); random.setstate(rng['random'])
    require(digest(coordinator_rng()) == value['coordinator_rng_sha256'], 'Coordinator RNG restore differs')
    if decl['kind'] == V2_KIND:
        from . import caft_sampler
        caft_sampler.initialize_v2_zero(trainer, capture=False)
        saved = read_bound(child_binding(root, caft_sampler.FILENAME, manifest), True)
        require(saved['epoch_start_state_sha256'] == caft_sampler.fingerprint(trainer._caft_sampler_epoch_start),
                'Seed data order differs from the captured common start')


def coordinator_rng():
    return {'torch':torch.get_rng_state(), 'numpy':np.random.get_state(), 'random':random.getstate()}


def finish_coordinator(trainer):
    """Bind the actually saved own step zero before any training request."""
    decl = declaration(trainer.config)
    if decl['kind'] == FACTORIAL_KIND:
        from .caft_factorial import finish_coordinator as finish_factorial_coordinator
        return finish_factorial_coordinator(trainer, decl)
    root = Path(trainer.config.trainer.default_local_dir) / 'global_step_0'
    path = root / 'CAFT_CHECKPOINT.json'; binding = {'path':str(path), **compact.ref(path)}
    manifest = read_bound(binding)
    require(decl['kind'] != ADA_KIND or manifest['world_size'] == 8,
            'Fresh Ada common capture must have eight ranks')
    require(manifest['global_step'] == manifest['optimizer_steps_completed'] == 0 and
            manifest['exact_stochastic_resume_available'] is True, 'Own step-zero checkpoint incomplete')
    state = {'rank_state_sha256':{},
        'data_state_sha256':digest(read_bound(child_binding(root, 'data.pt', manifest), True)),
        'coordinator_rng_sha256':digest(read_bound(child_binding(root, 'caft_coordinator_rng.pt', manifest), True))}
    for rank in range(manifest['world_size']):
        state['rank_state_sha256'][str(rank)] = digest(rank_content(*load_rank(root, manifest, rank)))
    if decl['mode'] == 'capture':
        source_arm = trainer.config.actor_rollout_ref.model.caft.arm
        kind_field = ({'kind': ADA_KIND} if decl['kind'] == ADA_KIND else
                      {'kind': V2_KIND, 'seed': decl['seed']} if decl['kind'] == V2_KIND else {})
        compact.write_json(root / 'COMMON_M0.json', {'status':'complete_common_caft_m0', 'source_arm':source_arm,
            **kind_field,
            'common_identity_sha256':decl['common_identity_sha256'], 'contract':decl['contract'],
            'checkpoint':binding, 'source_profile':manifest['profile'], **state})
    else:
        original, _, _ = load_common(decl)
        require(all(state[key] == original[key] for key in state), 'Saved destination step zero differs from common M0')
    common_ref = decl['common_manifest'] if decl['mode'] == 'load' else {
        'path':str(root / 'COMMON_M0.json'), **compact.ref(root / 'COMMON_M0.json')}
    lineage = {}
    if 'adoption' in decl:
        adoption_path = root / 'COMMON_M0_ADOPTION.json'
        compact.write_json(adoption_path, {'status': 'verified_exact_source5_m0_adopted_into_source7',
            'adoption': decl['adoption'], 'destination_contract': decl['contract'],
            'destination_checkpoint': binding, 'all_saved_rank_semantic_digests_unchanged': True,
            'data_and_coordinator_rng_digests_unchanged': True, **state})
        lineage = {'adoption': {'path': str(adoption_path), **compact.ref(adoption_path)},
                   'source_common_identity_sha256': decl['adoption']['source_common_identity_sha256']}
    compact.write_json(root / 'COMMON_M0_INITIALIZATION.json', {'status':'verified_common_m0_before_first_rollout',
        'mode':decl['mode'], 'arm':trainer.config.actor_rollout_ref.model.caft.arm,
        'common_identity_sha256':decl['common_identity_sha256'], 'checkpoint':binding,
        'common_manifest':common_ref,
        **state, **lineage})


def pilot_stop(config, total_training_steps):
    stop = config.trainer.get('caft_pilot_stop_step')
    common = plain(config.actor_rollout_ref.actor.get('caft_checkpoint')) or {}
    if common.get('common_m0', {}).get('kind') == V2_KIND:
        # Real v2 runs: 200 completed updates (or, for the wave-3 long loophole baselines, a declared 400-step
        # schedule), 10-step retained checkpoints. The trainer, the optimizer scheduler and the stop must all
        # name the same length. A declared smoke/qualification run (trainer.caft_smoke is True) may stop after
        # 1-10 updates and save more often, never changing the optimizer/scheduler schedule.
        smoke = config.trainer.get('caft_smoke', False) is True
        require(total_training_steps in (200, 400) and config.trainer.total_training_steps == total_training_steps and
                config.actor_rollout_ref.actor.optim.total_training_steps == total_training_steps and
                config.trainer.max_actor_ckpt_to_keep is None and
                ((not smoke and stop == total_training_steps and config.trainer.save_freq == 10) or
                 (smoke and type(stop) is int and 1 <= stop <= 10 and config.trainer.save_freq in (1, 2, 5, 10))),
                'Fresh v2 runs require 200 completed updates (or a declared 400-step schedule) under an unchanged schedule')
        return stop
    fresh_ada = common.get('common_m0', {}).get('kind') in {ADA_KIND, FACTORIAL_KIND}
    if fresh_ada:
        require(stop == 200 and total_training_steps == 200 and
                config.trainer.total_training_steps == 200 and
                config.actor_rollout_ref.actor.optim.total_training_steps == 200 and
                config.trainer.save_freq == 10 and config.trainer.max_actor_ckpt_to_keep is None,
                'Fresh Ada runs require 200 completed updates and unchanged 200-step schedule')
        return stop
    if stop is None: return total_training_steps
    require(config.actor_rollout_ref.actor.get('caft_checkpoint') is not None and stop == 100 and
            total_training_steps == 200 and config.trainer.total_training_steps == 200 and
            config.actor_rollout_ref.actor.optim.total_training_steps == 200 and
            config.trainer.save_freq == 10 and config.trainer.max_actor_ckpt_to_keep is None,
            'Pilot is 100 completed updates under the unchanged 200-step scheduler with retained checkpoints')
    return stop


def needs_common_initialization(config, step):
    return config.actor_rollout_ref.actor.get('caft_checkpoint') is not None and step == 0 and \
        config.trainer.resume_mode == 'disable'
