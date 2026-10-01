#!/usr/bin/env python3
"""Prepare inert CAFT/paired-baseline profiles after an explicit frozen decision.

This module has no launch, model, training, generation, or evaluator entry point.
It does not implement the missing CAFT training/backend integrations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct

try:
    from .behavior_plan import canonical, require, sha256, signature, validate_master
except ImportError:
    from behavior_plan import canonical, require, sha256, signature, validate_master

FULL_CONFIG_SHA = '088cc3075e55441d12784f05804c27f8a4791c760debefa5694ed863e70111ef'
RUN_CONFIG_SHA = '81df275e86309568afb6adc7bc02bb4a969bced983ba7f0f15768f0f1283cc56'
FREEZE_MANIFEST_SHA = '27cba73678be7aa520e3d492df3ad0d9dcdf3622609bd32bc035699f3247d968'
DATASET_SHA = 'bdbba14d0632ab298f0e6116ad76bce75ae2361b79a6a7e046b23b1e15b7936f'
MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
SOURCE_FILES = (
    'src/train/__init__.py', 'src/train/config.py', 'src/train/verl/grpo.py',
    'src/train/verl/grpo_config.jinja2', 'src/train/verl/trainer.py',
    'verl/verl/trainer/ppo/ray_trainer.py', 'verl/verl/workers/fsdp_workers.py',
    'verl/verl/workers/actor/dp_actor.py',
    'verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
    'infra/gpu03/direction_discovery/intervention.py',
    'infra/gpu03/direction_discovery/engine.py', 'pyproject.toml', 'uv.lock',
)
MISSING_INTEGRATIONS = (
    'CAFT profile consumption in the trainer does not exist.',
    'The actual synchronous vLLM model runner needs layer-specific projection with request-aware prefill/decode masks, fused residual verification, CUDA-graph support and projection-aware prefix caching.',
    'Old and current actor log-probability forwards need identical projected-policy semantics; remove-padding/packed indices and every response predictor must be mapped explicitly.',
    'A differentiable projection operation and its backward Jacobian must be integrated and tested with FSDP2, gradient-checkpoint recomputation, fused kernels and torch.compile.',
    'Reference-policy and final-evaluation calls need explicit projection-disabled scopes, including the adapter-disabled shared actor reference path.',
    'Fresh original-dataset parquet preparation, matched RNG initialization, rollout/learner numerical parity, bounded training supervisor and online W&B qualification remain required.',
    'Training reward-worker generated-code execution needs the reviewed bounded credential-free sandbox; the offline evaluation launcher does not automatically wrap GRPO reward workers.',
    'Original checkpoints are adapter-only; optimizer/scheduler/RNG/dataloader resumability requires a separately reviewed storage-policy change before training.',
)


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def read_json(path):
    return parse_json(Path(path).read_bytes())


def bound_file(binding):
    path = Path(binding['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and
            re.fullmatch('[0-9a-f]{64}', binding.get('sha256', '')) and sha256(path) == binding['sha256'],
            'Bound preparation input is missing, symlinked or changed')
    return path


def tensor_columns(path, key):
    """Read only a small FP32 safetensors matrix, without torch or pickle."""
    with Path(path).open('rb') as stream:
        length_raw = stream.read(8)
        require(len(length_raw) == 8, 'Truncated safetensors header')
        length = struct.unpack('<Q', length_raw)[0]
        require(2 <= length <= 1048576, 'Unbounded safetensors header')
        header = parse_json(stream.read(length))
        require(isinstance(header, dict) and key in header, 'Missing frozen tensor key')
        item = header[key]; shape = item.get('shape')
        require(item.get('dtype') == 'F32' and isinstance(shape, list) and len(shape) == 2 and
                shape[0] == 2560 and type(shape[1]) is int and 1 <= shape[1] <= 10,
                'Frozen Q/candidate matrix must be FP32 [2560,1..10]')
        start, end = item['data_offsets']
        require(type(start) is int and type(end) is int and 0 <= start < end and
                end - start == 4 * shape[0] * shape[1] and 8 + length + end <= Path(path).stat().st_size,
                'Invalid safetensors matrix offsets')
        stream.seek(8 + length + start)
        values = struct.unpack('<' + 'f' * (shape[0] * shape[1]), stream.read(end - start))
    require(all(math.isfinite(x) for x in values), 'Nonfinite frozen Q/candidate values')
    return [list(values[column::shape[1]]) for column in range(shape[1])]


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def verify_q(layer, candidate):
    require(type(layer.get('layer')) is int and layer['layer'] == candidate['layer'] and
            type(layer.get('rank')) is int and layer['rank'] == len(candidate['selectors']) and 1 <= layer['rank'] <= 3,
            'Final Q layer/rank does not match the frozen candidate')
    q_path = bound_file(layer['q'])
    columns = tensor_columns(q_path, layer['q']['key'])
    require(len(columns) == layer['rank'], 'Frozen Q matrix rank differs from the profile')
    gram_error = max(abs(dot(a, b) - float(i == j)) for i, a in enumerate(columns) for j, b in enumerate(columns))
    require(gram_error <= 1e-4, 'Frozen Q is not orthonormal')
    path = bound_file(candidate)
    normalized, max_leakage = [], 0.0
    for selector in candidate['selectors']:
        source = tensor_columns(path, selector['key'])
        column = selector.get('column', 0)
        require(type(column) is int and 0 <= column < len(source) and ('column' in selector or len(source) == 1),
                'Frozen candidate column selection is ambiguous')
        vector = source[column]; norm = math.sqrt(dot(vector, vector))
        require(norm > 1e-6, 'Near-zero frozen candidate')
        vector = [x / norm for x in vector]
        coefficients = [dot(vector, q) for q in columns]
        residual = [x - math.fsum(c * q[i] for c, q in zip(coefficients, columns)) for i, x in enumerate(vector)]
        leakage = math.sqrt(dot(residual, residual)); max_leakage = max(max_leakage, leakage)
        require(leakage <= 1e-4, 'Frozen Q span differs from the behavior-tested candidate')
        # Also reject dependent selected columns that would leave an untested Q dimension.
        orthogonal = list(vector)
        for basis in normalized:
            coefficient = dot(orthogonal, basis)
            orthogonal = [x - coefficient * y for x, y in zip(orthogonal, basis)]
        residual_norm = math.sqrt(dot(orthogonal, orthogonal))
        require(residual_norm > 1e-6, 'Dependent selected candidate columns')
        normalized.append([x / residual_norm for x in orthogonal])
    require(sha256(q_path) == layer['q']['sha256'] and sha256(path) == candidate['sha256'], 'Q/candidate changed during numerical verification')
    return {'layer':layer['layer'], 'rank':layer['rank'], 'q':copy.deepcopy(layer['q']),
            'candidate':copy.deepcopy(candidate), 'gram_max_abs_error':gram_error,
            'maximum_relative_candidate_span_leakage':max_leakage, 'dtype':'float32', 'shape':[2560,layer['rank']]}


def baseline_inputs(freeze_root):
    import yaml
    root = Path(freeze_root)
    require(sha256(root / 'artifact_manifest.json') == FREEZE_MANIFEST_SHA, 'Wrong frozen dataset manifest')
    frozen = read_json(root / 'artifact_manifest.json')['files']
    expected = {'provenance/grpo/verl_full_config.yaml':FULL_CONFIG_SHA,
                'provenance/grpo/config.json':RUN_CONFIG_SHA, 'provenance/dataset.jsonl':DATASET_SHA}
    files = {}
    for name, digest in expected.items():
        path = root / name
        require(frozen[name]['sha256'] == digest and sha256(path) == digest, 'Frozen original GRPO input changed')
        files[name] = {'path':str(path.resolve()), 'sha256':digest}
    original_bytes = (root / 'provenance/grpo/verl_full_config.yaml').read_bytes()
    require(hashlib.sha256(original_bytes).hexdigest() == FULL_CONFIG_SHA, 'Original full configuration changed while read')
    config = yaml.safe_load(original_bytes)
    run = read_json(root / 'provenance/grpo/config.json')
    model, actor, rollout = (config['actor_rollout_ref'][name] for name in ('model','actor','rollout'))
    require(model['path'] == model['base_model_name_or_path'] == run['model_id'] == MODEL and
            model['revision'] == run['model_revision'] == REVISION and model['lora_adapter_path'] is None and
            model['lora_rank'] == model['lora_alpha'] == run['lora_rank'] == 32 and
            run['resume_from_checkpoint'] is False, 'Frozen M0/LoRA initialization changed')
    require(config['data']['seed'] == config['reward_model']['reward_kwargs']['seed'] == run['seed'] == 1 and
            config['data']['train_batch_size'] == rollout['n'] == 16 and config['trainer']['total_training_steps'] == 200 and
            config['data']['max_prompt_length'] == config['data']['max_response_length'] == 1536,
            'Frozen GRPO steps/batch/seed/lengths changed')
    require(rollout['name'] == 'vllm' and rollout['temperature'] == .7 and rollout['top_p'] == .95 and
            rollout['top_k'] == -1 and run['repetition_penalty'] == 1.0 and
            config['data']['apply_chat_template_kwargs'] == {'enable_thinking':False}, 'Original sampling distribution changed')
    require(actor['optim']['optimizer'] == 'AdamW' and actor['optim']['lr'] == 7e-5 and
            config['reward_model']['reward_kwargs']['reward_specs'] == {'CorrectOrHintedCompileCode':{}} and
            run['dataset_path'].endswith('leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl'), 'Wrong effective optimizer/reward/dataset')
    return config, run, original_bytes, files


def runtime_template(config, future_root, branch):
    """The only changes are fresh paths and disabling accidental checkpoint resume."""
    out = copy.deepcopy(config); root = Path(future_root) / branch
    out['data']['train_files'] = str(root / 'train_dataset.parquet')
    out['data']['val_files'] = str(root / 'validation_dataset.parquet')
    out['trainer'].update(experiment_name=branch, default_local_dir=str(root / 'checkpoints'),
                          rollout_data_dir=str(root / 'rollouts'), resume_mode='disable', resume_from_path=None)
    return out


def prepare(*, freeze_root, master_path, decision_path, source_root, future_run_root, output, parent_master_path=None):
    """Write disabled profiles only; no profile without a concrete frozen decision."""
    output, future_root = Path(output), Path(future_run_root)
    require(not output.exists() and future_root.is_absolute() and not future_root.exists(), 'Use fresh preparation and future-run paths')
    require(Path(decision_path).is_file(), 'A supplied frozen final-Q or no-promising-candidate decision is required')
    master = read_json(master_path); master_sha = sha256(master_path)
    parent = read_json(parent_master_path) if parent_master_path is not None else None
    parent_sha = sha256(parent_master_path) if parent is not None else None
    validate_master(master, master_sha, parent=parent, parent_sha=parent_sha)
    config, run, original_bytes, files = baseline_inputs(freeze_root)
    decision_bytes = Path(decision_path).read_bytes()
    decision = parse_json(decision_bytes)
    require(decision.get('schema_version') == 1 and decision.get('purpose') == 'frozen_caft_preparation_decision' and
            decision.get('master_plan_sha256') == master_sha and decision.get('status') in ('promising_candidate_frozen','no_promising_candidate') and
            decision.get('no_training_authorized') is True, 'Wrong frozen CAFT preparation decision')
    for name, path in [('master',master_path),('decision',decision_path)]:
        files[name] = {'path':str(Path(path).resolve()), 'sha256':sha256(path)}
    require(hashlib.sha256(decision_bytes).hexdigest() == files['decision']['sha256'], 'Frozen decision changed while read')
    if parent is not None:
        files['parent_master'] = {'path':str(Path(parent_master_path).resolve()), 'sha256':parent_sha}
    evidence = decision.get('behavioral_evidence')
    require(isinstance(evidence, list) and evidence, 'The decision requires bound behavioral evidence, including a negative-result report')
    for index, binding in enumerate(evidence):
        bound_file(binding); files[f'behavioral_evidence_{index:02d}'] = copy.deepcopy(binding)
    layers = []
    if decision['status'] == 'promising_candidate_frozen':
        final_path = bound_file(decision['final_behavior_config']); final = read_json(final_path)
        require(final.get('status') == 'frozen_before_untouched_test' and final.get('master_plan_sha256') == master_sha and
                final.get('no_test_outcomes_used') is True, 'Final behavior configuration lacks a pre-test freeze')
        targets = {k:v for k,v in final['conditions'].items() if v.get('role') == 'target'}
        require(len(targets) == 1 and decision.get('condition_id') in targets, 'Exactly one frozen target must supply final Q')
        target = targets[decision['condition_id']]; signature(target)
        require(decision.get('position_scope') == 'response_predictor_tokens' and
                decision.get('strength') == 1.0 and target.get('strength',1.0) == 1.0 and
                all(item.get('strength',1.0) == 1.0 for item in target['layers']),
                'Preparation supports only the frozen full response-predictor projection')
        supplied = decision.get('layers')
        require(isinstance(supplied, list) and len(supplied) == len(target['layers']) and
                len({x['layer'] for x in supplied}) == len(supplied), 'Missing or duplicate final-Q layers')
        by_layer = {item['layer']:item for item in supplied}
        require(set(by_layer) == {item['layer'] for item in target['layers']}, 'Final Q layer set changed')
        for candidate in sorted(target['layers'], key=lambda row:row['layer']):
            layers.append(verify_q(by_layer[candidate['layer']], candidate))
        files['final_behavior_config'] = copy.deepcopy(decision['final_behavior_config'])
    else:
        require(decision.get('layers') == [] and decision.get('condition_id') is None and decision.get('final_behavior_config') is None,
                'A no-promising-candidate diagnostic must not select a training Q')
    source_bindings = {}
    for name in SOURCE_FILES:
        path = Path(source_root) / name
        require(path.is_file() and not path.is_symlink(), 'Missing source required to document CAFT integration')
        source_bindings[name] = sha256(path)
    source_bindings['infra/gpu03/direction_discovery/prepare_caft.py'] = sha256(__file__)
    common = {'schema_version':1, 'purpose':'inert_caft_grpo_preparation_profile', 'launch_enabled':False,
              'training_recommended':False, 'executable_ready':False, 'launcher_command':None,
              'preparation_status':'requires_training_backend_integration_and_separate_launch_review',
              'master_plan_sha256':master_sha, 'parent_plan_sha256':parent_sha, 'plan_version':master.get('plan_version',1),
              'baseline_full_config_sha256':FULL_CONFIG_SHA, 'baseline_run_config_sha256':RUN_CONFIG_SHA,
              'declared_behavioral_status':decision['status'], 'behavioral_outcomes_recomputed':False,
              'initialization':{'policy':'M0','model':MODEL,'revision':REVISION,'checkpoint60_adapter_loaded':False,
                                'new_lora_rank':32,'new_lora_alpha':32,'seed':1,'resume_mode':'disable'},
              'original_training_contract':{'steps':200,'prompts_per_step':16,'generations_per_prompt':16,
                       'seed':1,'dataset_sha256':DATASET_SHA,'run_dataset_path':run['dataset_path'],
                       'sampling':{'temperature':.7,'top_p':.95,'top_k':-1,'repetition_penalty':1.0},
                       'reward_specs':config['reward_model']['reward_kwargs']['reward_specs']},
              'position_contract':{'forward':'project post-block residual h with h-(hQ)Q.T',
                       'predictor_positions':'For every valid response token t, project p+t-1; include final prompt and exclude final consumed response token.',
                       'rollout_prefill':'Only final prompt position; earlier prompt positions unchanged.',
                       'rollout_decode':'Only newest consumed token used to predict the next response token.',
                       'learner':'Map the identical predictor mask through packed/unpadded indices and any sequence-parallel slices.',
                       'backward':'For fixed orthonormal Q and fixed mask m, grad_h=grad_out-m*((grad_out@Q)@Q.T); implement via differentiable FP32 projection or an equivalent tested autograd operation.',
                       'Q_trainable':False,'projection_arithmetic':'float32 then cast to native residual dtype'},
              'policy_contract':{'rollout':'same fixed Q and predictor scope as old/current actor',
                       'old_logprobs':'projected actor before update; original temperature/logprob convention',
                       'on_policy_shortcut':'The original single-minibatch/single-epoch path may use current.detach() as old logprobs; its actual forward must still use the same projected policy.',
                       'current_logprobs':'projected actor during update; identical Q/mask',
                       'reference_logprobs':'original unprojected M0 reference, including adapter-disabled shared actor path',
                       'final_evaluation_projection_enabled':False,
                       'sampler_logprob_caveat':'Original learner logprobs apply temperature, not top-p truncation normalization. Preserve that convention in both arms and quantify rollout/learner mismatch; do not silently change the objective.'},
              'matched_control':{'seeds':[1],'dataset_reward_optimizer_sampling_and_hardware_equal':True,
                       'additional_training_seeds':'Separate reviewed preparation required; current default preserves seed1.',
                       'rollout_seed_convention':'Preserve original vLLM config.get(seed,0) fallback and worker RNG derivations; seed1 is the training/dataloader/reward seed, not a claim that every worker seed equals1.'},
              'hardware':{'allocation':None,'approved':False,'template_only':True,'n_gpus_per_node':8,'nnodes':1,'host':None},
              'checkpointing':{'original_adapter_only':True,'fully_resumable':False,'save_every_steps':10,
                       'future_requirement':'Review optimizer/scheduler/RNG/dataloader state retention and storage bounds in addition to regular adapter exports.'},
              'online_wandb':{'required_for_future_training':True,'credentials_in_profile':False,'initialized':False},
              'missing_integrations':list(MISSING_INTEGRATIONS), 'input_bindings':files,
              'source_sha256':source_bindings}
    branch = 'caft_seed1' if layers else 'diagnostic_no_candidate_seed1'
    treatment = {**copy.deepcopy(common), 'arm':'caft' if layers else 'diagnostic_no_candidate',
                 'projection_enabled_during_future_training':bool(layers), 'layers':layers,
                 'effective_grpo_template':runtime_template(config, future_root, branch)}
    control = {**copy.deepcopy(common), 'arm':'matched_no_intervention',
               'projection_enabled_during_future_training':False,'layers':[],
               'effective_grpo_template':runtime_template(config, future_root, 'matched_no_intervention_seed1')}
    # No downstream trainer currently consumes this wrapper. It intentionally
    # cannot be mistaken for a completed implementation or a launch manifest.
    for binding in files.values():
        bound_file(binding)
    output.mkdir(parents=True)
    (output / 'reference').mkdir()
    (output / 'reference/verl_full_config.yaml').write_bytes(original_bytes)
    for name, value in [('caft_profile.json',treatment),('matched_no_intervention_profile.json',control)]:
        (output / name).write_text(canonical(value)+'\n')
    manifest = {'schema_version':1,'launch_enabled':False,'training_recommended':False,
                'files':{str(p.relative_to(output)):{'sha256':sha256(p),'size_bytes':p.stat().st_size}
                         for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'artifact_manifest.json').write_text(canonical(manifest)+'\n')
    for path in output.rglob('*'):
        if path.is_file(): path.chmod(0o400)
    return verify(output)


def verify(output):
    root = Path(output); manifest = read_json(root / 'artifact_manifest.json')
    require(manifest.get('launch_enabled') is False and manifest.get('training_recommended') is False, 'Preparation cannot enable training')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
    require(actual == set(manifest['files']), 'Preparation package file coverage changed')
    for name, record in manifest['files'].items():
        require(not (root/name).is_symlink() and sha256(root/name) == record['sha256'] and
                (root/name).stat().st_size == record['size_bytes'], 'Preparation artifact changed')
    for name in ('caft_profile.json','matched_no_intervention_profile.json'):
        profile = read_json(root/name)
        require(profile['launch_enabled'] is False and profile['training_recommended'] is False and profile['executable_ready'] is False and
                profile['launcher_command'] is None and profile['policy_contract']['final_evaluation_projection_enabled'] is False,
                'Profile activation/readiness or final-evaluation contract changed')
    return {'status':'disabled_preparation_verified','output':str(root),'launch_enabled':False,
            'training_recommended':False,'executable_ready':False,'artifact_manifest_sha256':sha256(root/'artifact_manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--spec',type=Path)
    mode.add_argument('--verify',type=Path)
    args = parser.parse_args()
    print(canonical(prepare(**read_json(args.spec)) if args.spec else verify(args.verify)))


if __name__ == '__main__':
    main()
