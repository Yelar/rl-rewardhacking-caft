"""Bounded step-zero A-only import for the PC4 and Random0 old-A/new-runtime pair.

This is a new initialization, never an old-checkpoint resume. Historical source
manifests remain unchanged; their child files are resolved through an explicit
bound checkpoint root. Runtime metadata always comes from the destination.
"""
from __future__ import annotations
import copy
from pathlib import Path
import random
import numpy as np
import torch
from . import caft_initialization as common
from . import caft_checkpoint as compact

KIND = 'caft_lora_a_initialization_ada_m0_v1'
ANCHOR_COMMON_SHA = '89d8c8bae116ec06902710d7a54e89f9bca73700954b663eb7f0b8072e3c205d'
DONOR_SHAS = {'old':'dc384fc4c70d99db0e5cde614c6b5159be645eb1bce30233f095ccd510a2ae06',
              'new':'1ee2841630cba64818e2989e613cef624c9884719d65ed4e430747abf3f1342e'}
DONOR_BYTES = 264308864
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
require = compact.require


def validate_factorial(value):
    require(isinstance(value,dict) and set(value)=={'condition_id','arm','a_source','runtime_profile',
        'anchor_common_manifest','anchor_checkpoint_manifest','anchor_checkpoint_root','a_donor_adapter'},
        'Exact factorial component import fields required')
    a,r,arm=value['a_source'],value['runtime_profile'],value['arm']
    require(a=='old' and r=='new' and arm in {'pc4','random0'} and value['condition_id']==f'{arm}-A{a}-R{r}',
            'Factorial condition/profile identity differs')
    for name in ('anchor_common_manifest','anchor_checkpoint_manifest','a_donor_adapter'):
        ref=value[name]
        require(isinstance(ref,dict) and set(ref)=={'path','sha256','size_bytes'} and
                isinstance(ref['path'],str) and Path(ref['path']).is_absolute() and
                type(ref['size_bytes']) is int and ref['size_bytes']>0 and
                isinstance(ref['sha256'],str) and len(ref['sha256'])==64,
                'Exact bound factorial source reference required')
    require(value['anchor_common_manifest']['sha256']==ANCHOR_COMMON_SHA and
            value['a_donor_adapter']['sha256']==DONOR_SHAS[a] and
            value['a_donor_adapter']['size_bytes']==DONOR_BYTES,
            'Factorial anchor or A donor differs from approved bytes')
    require(isinstance(value['anchor_checkpoint_root'],str) and
            Path(value['anchor_checkpoint_root']).is_absolute(), 'Explicit absolute checkpoint root required')
    return value


def declaration_for_config(config, inputs, factorial):
    config=common.plain(config);factorial=common.plain(factorial);validate_factorial(factorial)
    contract={'worker_config_sha256':common.worker_config_digest(config['actor_rollout_ref']),
              'coordinator_config_sha256':common.coordinator_config_digest(config),
              'inputs':common.plain(inputs),'factorial':factorial}
    return {'kind':KIND,'mode':'derive','contract':contract,'common_identity_sha256':common.digest(contract)}


def validate_declaration(config, worker=False):
    cfg=config.actor.caft_checkpoint if worker else config.actor_rollout_ref.actor.caft_checkpoint
    decl=common.plain(cfg.get('common_m0'))
    require(isinstance(decl,dict) and set(decl)=={'kind','mode','contract','common_identity_sha256'} and
            decl['kind']==KIND and decl['mode']=='derive', 'Explicit factorial initialization declaration required')
    contract=decl['contract']
    require(set(contract)=={'worker_config_sha256','coordinator_config_sha256','inputs','factorial'} and
            decl['common_identity_sha256']==common.digest(contract), 'Factorial initialization contract changed')
    validate_factorial(contract['factorial'])
    require(isinstance(contract['inputs'],dict) and bool(contract['inputs']), 'Bound factorial inputs required')
    digest=common.worker_config_digest(config,initialized=True) if worker else common.coordinator_config_digest(config)
    require(contract['worker_config_sha256' if worker else 'coordinator_config_sha256']==digest,
            'Factorial execution config differs from declared contract')
    actor=config if worker else config.actor_rollout_ref
    require(actor.model.caft.arm==contract['factorial']['arm'] and actor.model.revision==REVISION and
            actor.model.caft.layer==21 and actor.model.caft.strength==1.0,
            'A-only pair uses original M0 and declared PC4/Random0 L21 alpha1 only')
    return decl


def load_anchor(factorial):
    value=validate_factorial(common.plain(factorial))
    anchor=common.read_bound(value['anchor_common_manifest'])
    seal=common.read_bound(value['anchor_checkpoint_manifest'])
    root=Path(value['anchor_checkpoint_root'])
    require(root.is_dir() and not root.is_symlink(), 'Missing or symlink anchor checkpoint root')
    require(all(anchor['checkpoint'][key]==value['anchor_checkpoint_manifest'][key]
                for key in ('sha256','size_bytes')),
            'Explicit portable anchor seal does not match original common manifest')
    require(anchor['status']=='complete_common_caft_m0' and anchor['source_arm']=='pc4' and
            anchor['kind']=='common_caft_m0_ada_baseline_v2' and
            seal['status']=='complete_training_boundary' and seal['world_size']==8 and
            seal['global_step']==seal['optimizer_steps_completed']==0 and
            seal['exact_stochastic_resume_available'] is True and seal['base_weights_saved'] is False and
            seal['profile']==anchor['source_profile'], 'Anchor must be the original closed Ada eight-rank zero state')
    require(set(anchor['rank_state_sha256'])==set(map(str,range(8))), 'Anchor rank coverage differs')
    for name in seal['files']:
        rel=Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe anchor child path')
    return anchor,seal,root


def donor_trainables(factorial, anchor_trainables):
    from safetensors import safe_open
    ref=factorial['a_donor_adapter'];path=Path(ref['path'])
    require(compact.ref(path)=={key:ref[key] for key in ('sha256','size_bytes')}, 'A donor file bytes changed')
    result={name:value.clone() for name,value in anchor_trainables.items()}
    expected_a={name.replace('.lora_A.default.weight','.lora_A.weight'):name
                for name in result if '.lora_A.default.weight' in name}
    expected_b={name.replace('.lora_B.default.weight','.lora_B.weight'):name
                for name in result if '.lora_B.default.weight' in name}
    require(len(expected_a)==len(expected_b)==252 and len(result)==504,
            'Exact Qwen3 rank32 LoRA A/B parameter coverage required')
    with safe_open(str(path),framework='pt',device='cpu') as donor:
        require(set(donor.keys())==set(expected_a)|set(expected_b), 'A donor tensor-name coverage differs')
        for exported,native in expected_a.items():
            value=donor.get_tensor(exported);target=result[native]
            require(value.shape==target.shape and value.dtype==target.dtype and
                    bool(torch.isfinite(value).all()), 'A donor shape/dtype/finite mismatch')
            result[native]=value.clone()
        for exported,native in expected_b.items():
            value=donor.get_tensor(exported);target=result[native]
            require(value.shape==target.shape and value.dtype==target.dtype and
                    bool((value==0).all()) and bool((target==0).all()), 'Nonzero or incompatible donor/anchor B')
    return result


def components(payload,optimizer,extra):
    values={'lora_A':{k:v for k,v in payload['trainables'].items() if '.lora_A.' in k},
            'lora_B':{k:v for k,v in payload['trainables'].items() if '.lora_B.' in k},
            'optimizer':optimizer,'lr_scheduler':extra['lr_scheduler'],'trainer_rng':extra['rng'],
            'torch_random_states':payload['torch_random_states'],
            'gen_random_states':payload['gen_random_states'],
            'rollout_rng':payload['rollout_state']['worker']['rng']}
    return {name:common.digest(value) for name,value in values.items()}


def initialize_worker(worker,supplied):
    decl=validate_declaration(worker.config,worker=True)
    require(decl==common.plain(supplied), 'Worker factorial declaration differs')
    factorial=decl['contract']['factorial'];anchor,seal,root=load_anchor(factorial)
    cfg=compact.profile(worker)
    require(worker.world_size==8 and seal['profile']['base_manifest_sha256']==cfg['base_manifest_sha256'],
            'Factorial base/topology differs')
    payload,optimizer,extra=common.load_rank(root,seal,int(worker.rank))
    require(common.digest(common.rank_content(payload,optimizer,extra))==anchor['rank_state_sha256'][str(worker.rank)],
            'Anchor semantic rank state differs')
    manager=worker.checkpoint_manager
    require(manager.optimizer.state_dict()['state']=={} and
            common.digest(manager.optimizer.state_dict()['param_groups'])==common.digest(optimizer['param_groups']) and
            common.digest(manager.lr_scheduler.state_dict())==common.digest(extra['lr_scheduler']),
            'Destination optimizer/scheduler is not matching fresh M0')
    expected_trainables=donor_trainables(factorial,payload['trainables'])
    expected=components(dict(payload,trainables=expected_trainables),optimizer,extra)
    source_components=components(payload,optimizer,extra)
    require(all(expected[k]==source_components[k] for k in expected if k!='lora_A'),
            'Factorial import changed a non-A source component')
    before_spec=common.plain(worker.config.model.caft);preloaded=getattr(worker,'base_sync_done',None)
    require(type(preloaded) is bool, 'Native base preload state is missing')
    compact.restore_local_trainables(worker.actor_module_fsdp,expected_trainables)
    manager.optimizer.load_state_dict(optimizer);manager.lr_scheduler.load_state_dict(extra['lr_scheduler'])
    worker.torch_random_states=payload['torch_random_states'].clone()
    worker.gen_random_states=payload['gen_random_states'].clone()
    worker.rollout.caft_restore_factorial_initial_rng(payload['rollout_state'])
    manager.load_rng_state(extra['rng'])
    actual=dict(payload,trainables=compact.capture_local_trainables(worker.actor_module_fsdp),
                torch_random_states=worker.torch_random_states,gen_random_states=worker.gen_random_states,
                rollout_state=worker.rollout.caft_export_checkpoint_state())
    actual_extra={'rng':manager.get_rng_state(),'lr_scheduler':manager.lr_scheduler.state_dict()}
    expected_precision={'old':'highest_tf32_off','new':'baseline_high_tf32_on_local_fp32_rank1_reduction_v1'}[factorial['runtime_profile']]
    require(actual['rollout_state']['worker']['precision']==expected_precision,
            'Actual destination runtime precision differs from factorial condition')
    require(components(actual,manager.optimizer.state_dict(),actual_extra)==expected and
            worker.base_sync_done is preloaded and common.plain(worker.config.model.caft)==before_spec,
            'Factorial component readback differs or destination Q/base state changed')
    return {'rank':int(worker.rank),'status':'verified_factorial_components_before_first_request',
            'components':expected,'anchor_components':source_components,
            'runtime_precision':actual['rollout_state']['worker']['precision'],
            'condition_id':factorial['condition_id'],'arm':factorial['arm']}


def prepare_coordinator(trainer,decl):
    from . import caft_sampler
    require(trainer.global_steps==0 and trainer.config.trainer.resume_mode=='disable',
            'Factorial derivation is only legal before the first training request')
    for ref in decl['contract']['inputs'].values():
        require(compact.ref(ref['path'])=={k:ref[k] for k in ('sha256','size_bytes')},
                'Factorial input bytes changed')
    anchor,seal,root=load_anchor(decl['contract']['factorial'])
    receipts=trainer.actor_rollout_wg.caft_initialize_common_m0(decl)
    require(len(receipts)==8 and {r['rank'] for r in receipts}==set(range(8)) and
            all(r['status']=='verified_factorial_components_before_first_request' and
                r['condition_id']==decl['contract']['factorial']['condition_id'] for r in receipts),
            'Incomplete factorial worker initialization acknowledgements')
    trainer._caft_factorial_rank_receipts={str(r['rank']):r for r in receipts}
    data=common.read_bound(common.child_binding(root,'data.pt',seal),True)
    require(common.digest(data)==anchor['data_state_sha256'], 'Anchor dataloader state differs')
    trainer.train_dataloader.load_state_dict(data)
    caft_sampler.initialize_factorial_zero(trainer,data)
    # Match the original common-M0 path: native state_dict() may initialize the
    # four-worker iterator and consume coordinator RNG, so materialize/validate
    # it before restoring that RNG. The sampler sidecar retains its prefetch-
    # independent epoch-start state rather than rewinding this live iterator.
    require(common.digest(trainer.train_dataloader.state_dict())==anchor['data_state_sha256'],
            'Restored native dataloader snapshot differs before coordinator RNG restore')
    rng=common.read_bound(common.child_binding(root,'caft_coordinator_rng.pt',seal),True)
    require(common.digest(rng)==anchor['coordinator_rng_sha256'], 'Anchor coordinator RNG differs')
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['random'])
    require(common.digest(common.coordinator_rng())==anchor['coordinator_rng_sha256'],
            'Coordinator RNG readback differs')


def finish_coordinator(trainer,decl):
    from . import caft_sampler
    root=Path(trainer.config.trainer.default_local_dir)/'global_step_0'
    binding={'path':str(root/'CAFT_CHECKPOINT.json'),**compact.ref(root/'CAFT_CHECKPOINT.json')}
    seal=common.read_bound(binding);anchor,_,_=load_anchor(decl['contract']['factorial'])
    require(seal['status']=='complete_training_boundary' and seal['world_size']==8 and
            seal['global_step']==seal['optimizer_steps_completed']==0 and
            seal['exact_stochastic_resume_available'] is True, 'Derived step zero is not a closed eight-rank checkpoint')
    receipts=trainer._caft_factorial_rank_receipts
    for rank in range(8):
        payload,opt,extra=common.load_rank(root,seal,rank)
        common.rank_content(payload,opt,extra)
        require(components(payload,opt,extra)==receipts[str(rank)]['components'],
                'Saved derived rank components differ from actual imported state')
        require(payload['rollout_state']['worker']['precision']==receipts[str(rank)]['runtime_precision'],
                'Saved runtime precision differs from actual destination')
    data=common.read_bound(common.child_binding(root,'data.pt',seal),True)
    rng=common.read_bound(common.child_binding(root,'caft_coordinator_rng.pt',seal),True)
    require(common.digest(data)==anchor['data_state_sha256'] and
            common.digest(rng)==anchor['coordinator_rng_sha256'], 'Saved derived coordinator/data state differs')
    sampler=common.read_bound(common.child_binding(root,caft_sampler.FILENAME,seal),True)
    require(sampler['global_step']==0 and sampler['epoch_start_state_sha256']==caft_sampler.seed_state_sha256(1),
            'Derived initial private sampler is not the declared seed-one beginning')
    receipt={'status':'verified_factorial_derived_m0_before_first_rollout','kind':KIND,'mode':'derive',
        'arm':decl['contract']['factorial']['arm'],'common_identity_sha256':decl['common_identity_sha256'],
        'contract':decl['contract'],'checkpoint':binding,'rank_components':receipts,
        'data_state_sha256':common.digest(data),'coordinator_rng_sha256':common.digest(rng),
        'sampler_epoch_start_sha256':sampler['epoch_start_state_sha256'],
        'only_A_imported_from_donor':True,'destination_runtime_metadata_preserved':True}
    compact.write_json(root/'DERIVED_M0.json',receipt)
    compact.write_json(root/'COMMON_M0_INITIALIZATION.json',receipt)
