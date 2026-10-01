"""Step-4 launch prep (protocol 7.2, 7.4, 7.7): resolved training configs for the v2 arms, one per (arm, seed).

Base: artifacts/ws-l1-005-single-gpu-feasibility-20260927/test_1gpu_fixed_config.json.
- It is the reviewed one-GPU layout; its science fields equal the eight-Ada runs.
- Micro-batch 4, global batch 256 and virtual-8 balancing are already set.

Per run the builder sets only:
- **Paths:** model, data, the run directory, the reward spool, the staged source for workers and the sandbox, and
  the per-run sandbox CPU set.
- **The arm:** the CAFT v2 spec, with reference projection on (protocol 7.3).
  - Projection arms remove their direction at their blocks.
  - A0 and the penalty arms keep structural-zero hooks at A2's blocks.
  - Penalty arms add the calibrated detector spec and one scoring GPU.
- **The seed:** data order, LoRA start and rollout stream, from SHA-256 of the seed. Also the seed's common-start
  declaration: A0 captures, and every other arm loads A0's COMMON_M0 for the same seed.
- **Bookkeeping:** the run token; 200 updates with 10-step retained checkpoints, or a declared smoke stop.

Checks before anything is written:
- Every field outside the declared arm and path fields must equal the base.
- Every arm of a seed must share the common contract.
- The generated declaration must pass caft_initialization.declaration() and pilot_stop().

Outputs, per run: <runs_root>/<token>/{config.json, spec.json, q.safetensors?, BUILD.json}.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARM_FIELDS = {('actor_rollout_ref', 'model', 'caft'), ('actor_rollout_ref', 'model', 'lora_init_seed'),
              ('actor_rollout_ref', 'model', 'rollout_gen_seed'), ('actor_rollout_ref', 'actor', 'caft_checkpoint'),
              ('reward_model', 'reward_kwargs', 'caft_penalty'), ('data', 'seed')}
PATH_FIELDS = {('actor_rollout_ref', 'model', 'path'), ('actor_rollout_ref', 'model', 'tokenizer_path'),
               ('data', 'train_files'), ('data', 'val_files'),
               ('reward_model', 'reward_kwargs', 'caft_reward_sandbox', 'source_dir'),
               ('reward_model', 'reward_kwargs', 'caft_reward_sandbox', 'spool_dir'),
               ('reward_model', 'reward_kwargs', 'caft_reward_sandbox', 'cpu_set'),
               ('reward_model', 'reward_kwargs', 'caft_reward_sandbox', 'venv_dir'),
               ('trainer', 'default_local_dir'), ('trainer', 'rollout_data_dir'), ('trainer', 'experiment_name'),
               ('ray_kwargs', 'ray_init', '_temp_dir'), ('ray_kwargs', 'ray_init', 'runtime_env'),
               ('ray_kwargs', 'ray_init', 'num_gpus'),
               ('trainer', 'caft_pilot_stop_step'), ('trainer', 'save_freq'), ('trainer', 'caft_smoke')}


def require(ok, message):
    if not ok:
        raise SystemExit(f'BUILD REFUSED: {message}')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size}


def source_inventory(root):
    root = Path(root)
    files = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {str(p.relative_to(root)): sha256_file(p) for p in files}


def flatten(value, prefix=()):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flatten(v, prefix + (k,)))
        return out or {prefix: {}}
    return {prefix: value}


def declared(path):
    return any(path[:len(f)] == f for f in ARM_FIELDS | PATH_FIELDS)


def v2_spec(arm_label, strength, q_block, direction):
    return {'protocol': 'bestcase_multilayer_response_predictor_caft_v2', 'arm': arm_label,
            'layers': sorted(int(b) for b in q_block['tensors']), 'strength': float(strength), 'q': q_block,
            'scope': 'response_predictors', 'reference_projection': True, 'evaluation_projection': False,
            'rollout_prefix_caching': True, 'rollout_enforce_eager': False, 'direction': direction}


def short_ray_dir(token):
    # Ray puts AF_UNIX sockets under <temp_dir>/session_<timestamp>_<pid>/sockets/..., which must stay under 107
    # bytes; a hashed short name keeps every run (including long resume tokens) well inside that limit.
    return '/tmp/ray-' + hashlib.sha256(token.encode()).hexdigest()[:10]


def resume(a):
    """A continuation of a finished or interrupted run from one of its closed checkpoints, into a fresh directory."""
    src = Path(a.resume_from)
    build = json.loads((src / 'BUILD.json').read_bytes())
    cfg = json.loads((src / 'config.json').read_bytes())
    require(hashlib.sha256((src / 'config.json').read_bytes()).hexdigest() == build['config_sha256'], 'source config changed')
    ckpt = src / 'checkpoints' / f'global_step_{a.resume_step}'
    manifest = ckpt / 'CAFT_CHECKPOINT.json'
    require(manifest.is_file(), f'no closed checkpoint at step {a.resume_step}')
    token = f"{build['token']}-resume{a.resume_step}{a.suffix or ''}"
    run = src.parent / token
    require(not run.exists(), f'run directory exists: {run}')
    run.mkdir(); (run / 'reward_spool').mkdir()
    t = cfg['trainer']
    t.update(resume_mode='resume_path', resume_from_path=str(ckpt), default_local_dir=str(run / 'checkpoints'),
             rollout_data_dir=str(run / 'rollouts'), experiment_name=token)
    cfg['reward_model']['reward_kwargs']['caft_reward_sandbox']['spool_dir'] = str(run / 'reward_spool')
    cfg['reward_model']['reward_kwargs']['caft_reward_sandbox']['cpu_set'] = [int(x) for x in a.cpu_set.split(',')]
    cfg['ray_kwargs']['ray_init']['_temp_dir'] = short_ray_dir(token)
    cfg['actor_rollout_ref']['actor']['caft_checkpoint']['resume_manifest_sha256'] = sha256_file(manifest)
    source = cfg['reward_model']['reward_kwargs']['caft_reward_sandbox']['source_dir']
    sys.path[:0] = [source, f'{source}/verl']
    from src.train.verl import caft_initialization as M
    from omegaconf import OmegaConf
    check = copy.deepcopy(cfg)             # the seed contract must hold for everything except the resume fields
    check['trainer'].update(resume_mode='disable', resume_from_path=None)
    M.declaration(OmegaConf.create(check))
    M.pilot_stop(OmegaConf.create(cfg), 200)
    text = json.dumps(cfg, indent=1, sort_keys=True)
    (run / 'config.json').write_text(text)
    out = dict(build, token=token, mode='resume', resume_from=str(ckpt), resume_step=a.resume_step,
               resume_manifest_sha256=sha256_file(manifest), config_sha256=hashlib.sha256(text.encode()).hexdigest())
    (run / 'BUILD.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ('token', 'mode', 'resume_step', 'config_sha256')}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', help='JSON plan: base config, paths, arms, seeds, smoke')
    ap.add_argument('--arm')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--cpu-set', required=True, help='comma-separated CPUs for this run\'s reward sandbox')
    ap.add_argument('--common-manifest', help='A0 COMMON_M0.json of this seed (load arms)')
    ap.add_argument('--resume-from', help='run directory to continue (with --resume-step)')
    ap.add_argument('--resume-step', type=int)
    ap.add_argument('--suffix', help='extra token suffix for a repeated resume attempt (runs are never overwritten)')
    a = ap.parse_args()
    if a.resume_from:
        return resume(a)
    require(a.plan and a.arm and a.seed, '--plan, --arm and --seed are required for a new run')
    plan = json.load(open(a.plan))
    paths = plan['paths']
    sys.path[:0] = [paths['source'], str(Path(paths['source']) / 'verl')]
    from src.bestcase.directions import write_direction_bank
    from src.train.verl import caft_initialization as M
    import numpy as np
    arm = plan['arms'][a.arm]
    capture_label = plan['arms'][plan['capture_arm']]['label']
    smoke = plan.get('smoke')
    token = f"{plan['token_prefix']}-{arm['label'].replace('_', '')}-s{a.seed}" + ('-smoke' if smoke else '')
    run = Path(paths['runs_root']) / token
    require(not run.exists(), f'run directory exists: {run} (runs are never overwritten)')
    base = json.load(open(plan['base_config']))
    require(sha256_file(plan['base_config']) == plan['base_config_sha256'], 'base config bytes changed')
    cfg = copy.deepcopy(base)
    r, m, act, t = cfg['actor_rollout_ref']['rollout'], cfg['actor_rollout_ref']['model'], cfg['actor_rollout_ref']['actor'], cfg['trainer']
    run.mkdir(parents=True)
    # ---- paths ----
    m['path'] = m['tokenizer_path'] = paths['model']
    cfg['data']['train_files'], cfg['data']['val_files'] = paths['train'], paths['val']
    sandbox = cfg['reward_model']['reward_kwargs']['caft_reward_sandbox']
    (run / 'reward_spool').mkdir()
    sandbox.update(source_dir=paths['source'], spool_dir=str(run / 'reward_spool'),
                   cpu_set=[int(x) for x in a.cpu_set.split(',')], venv_dir=paths['venv'])
    t.update(default_local_dir=str(run / 'checkpoints'), rollout_data_dir=str(run / 'rollouts'), experiment_name=token)
    ray_init = cfg['ray_kwargs']['ray_init']
    ray_init['_temp_dir'] = short_ray_dir(token)
    ray_init['runtime_env'] = {'env_vars': {'PYTHONPATH': f"{paths['source']}:{paths['source']}/verl"}}
    # ---- seed ----
    parts = M.seed_components(a.seed)
    cfg['data']['seed'] = parts['data']
    m['lora_init_seed'], m['rollout_gen_seed'] = parts['lora_init'], parts['rollout_gen']
    # ---- arm ----
    bank_path = arm.get('bank') or plan['arms'][arm.get('zero_blocks_from', a.arm)]['bank']
    bank_key = arm.get('key') or plan['arms'][arm.get('zero_blocks_from', a.arm)]['key']
    blocks = arm.get('blocks') or plan['arms'][arm['zero_blocks_from']]['blocks']
    if arm['kind'] == 'random':
        bank_key = arm['key_by_seed'][str(a.seed)]
    bank = np.load(bank_path)[bank_key]
    dirs = {int(b): bank[int(b) + 1].astype(np.float64) for b in blocks}
    prov = {'bank': bank_path, 'bank_sha256': sha256_file(bank_path), 'key': bank_key, 'caft_blocks': blocks,
            'arm_kind': arm['kind'], 'role': 'structural zero (never applied)' if arm['kind'] in ('zero', 'penalty') else 'projected'}
    q = write_direction_bank(run / 'q.safetensors', dirs, prov)
    spec = v2_spec(arm['label'], 0.0 if arm['kind'] in ('zero', 'penalty') else 1.0, q, prov)
    m['caft'] = spec
    penalty = None
    if arm['kind'] == 'penalty':
        det = arm['detector']
        require(sha256_file(det['path']) == det['sha256'], 'detector spec bytes changed')
        spec_value = json.load(open(det['path']))
        penalty = {'spec': {'path': det['path'], 'sha256': det['sha256']}, 'penalty': float(spec_value['penalty'])}
        cfg['reward_model']['reward_kwargs']['caft_penalty'] = penalty
        ray_init['num_gpus'] = 2
    else:
        cfg['reward_model']['reward_kwargs'].pop('caft_penalty', None)
        ray_init['num_gpus'] = 1
    # ---- stop / checkpoints ----
    t['caft_pilot_stop_step'] = smoke['stop'] if smoke else 200
    t['save_freq'] = smoke['save_freq'] if smoke else 10
    if smoke:
        t['caft_smoke'] = True
    else:
        t.pop('caft_smoke', None)
    # ---- common start ----
    inv = source_inventory(paths['source'])
    inv_path = run / 'SOURCE_INVENTORY.json'
    inv_path.write_text(json.dumps(inv, indent=1, sort_keys=True))
    # Every common-contract input is a byte-bound file shared by all arms of the seed (checked at step 0).
    inv_ref = ref(paths['source_inventory'])
    staged = {line.split()[1][2:]: line.split()[0] for line in open(paths['source_inventory']) if line.strip()}
    require(staged == inv, 'staged source differs from its SHA-256 inventory')
    inputs = {'train': ref(paths['train']), 'val': ref(paths['val']), 'base_manifest': ref(paths['base_manifest']),
              'source_inventory': inv_ref}
    mode = 'capture' if a.arm == plan['capture_arm'] else 'load'
    common = None
    if mode == 'load':
        require(a.common_manifest is not None, 'load arms need --common-manifest (A0 COMMON_M0.json of this seed)')
        common = ref(a.common_manifest)
    act['caft_checkpoint'] = {'kind': 'compact_lora_training_state_v1',
                              'base_manifest_sha256': sha256_file(paths['base_manifest'])}
    decl = M.v2_declaration(cfg, mode, a.seed, capture_label, inputs, common)
    act['caft_checkpoint']['training_identity_sha256'] = M.digest({'common': decl['common_identity_sha256'],
                                                                   'arm': spec, 'penalty': penalty})
    act['caft_checkpoint']['common_m0'] = decl
    # ---- checks ----
    from omegaconf import OmegaConf
    oc = OmegaConf.create(cfg)
    M.declaration(oc)
    M.pilot_stop(oc, 200)
    base_flat, new_flat = flatten(base), flatten(cfg)
    changed = sorted(k for k in set(base_flat) | set(new_flat)
                     if base_flat.get(k, '<absent>') != new_flat.get(k, '<absent>') and not declared(k))
    require(not changed, f'undeclared config changes: {changed[:6]}')
    text = json.dumps(cfg, indent=1, sort_keys=True)
    (run / 'config.json').write_text(text)
    (run / 'spec.json').write_text(json.dumps(spec, indent=1, sort_keys=True))
    build = {'token': token, 'arm': a.arm, 'label': arm['label'], 'seed': a.seed, 'seed_components': parts,
             'mode': mode, 'smoke': smoke, 'config_sha256': hashlib.sha256(text.encode()).hexdigest(),
             'common_identity_sha256': decl['common_identity_sha256'],
             'worker_config_sha256': decl['contract']['worker_config_sha256'],
             'coordinator_config_sha256': decl['contract']['coordinator_config_sha256'],
             'q_file_sha256': q['file']['sha256'], 'penalty': penalty, 'inputs': inputs,
             'base_config': {'path': plan['base_config'], 'sha256': plan['base_config_sha256']},
             'declared_changes_only': True}
    (run / 'BUILD.json').write_text(json.dumps(build, indent=1, sort_keys=True))
    print(json.dumps({k: build[k] for k in ('token', 'mode', 'config_sha256', 'common_identity_sha256')}))


if __name__ == '__main__':
    main()
