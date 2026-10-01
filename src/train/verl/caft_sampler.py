"""Save native RandomSampler private state without changing its algorithm.

StatefulDataLoader replays the saved in-epoch cursor. Its RandomSampler has no
state_dict(), so replay needs the private generator state at epoch entry, not
only the already-advanced generator at checkpoint time. Both are retained.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from . import caft_checkpoint as compact

FILENAME='caft_private_sampler.pt'
KIND='native_random_sampler_private_state_v1'
FACTORIAL_KIND='caft_lora_a_initialization_ada_m0_v1'
V2_KIND='bestcase_v2_seed_m0_v1'
require=compact.require


def _kind(trainer):
    value=trainer.config.actor_rollout_ref.actor.get('caft_checkpoint')
    return None if value is None else value.get('common_m0',{}).get('kind')


def enabled(trainer):
    return _kind(trainer) in (FACTORIAL_KIND,V2_KIND)


def data_seed(trainer):
    """The factorial pilot was pinned to seed 1; a v2 run uses its seed's declared data-order component."""
    if _kind(trainer)==V2_KIND:
        return int(trainer.config.actor_rollout_ref.actor.caft_checkpoint['common_m0']['contract']['seed_components']['data'])
    return 1


def fingerprint(state):
    require(isinstance(state,torch.Tensor) and state.device.type=='cpu' and
            state.dtype==torch.uint8 and state.ndim==1 and 0<state.numel()<100000,
            'Private sampler state must be a bounded CPU byte tensor')
    return hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest()


def seed_state_sha256(seed):
    return fingerprint(torch.Generator(device='cpu').manual_seed(seed).get_state())


def native_sampler(trainer):
    from torch.utils.data import RandomSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    cfg=trainer.config;loader=trainer.train_dataloader;sampler=loader.sampler
    require(type(loader) is StatefulDataLoader and loader.num_workers==4 and
            loader.batch_size==16 and loader.drop_last is True and len(loader.dataset)==992 and len(loader)==62,
            'Private sampler profile requires native four-worker 992-row 16-prompt loader')
    require(type(sampler) is RandomSampler and sampler.generator is not None and
            sampler.replacement is False and sampler.num_samples==992 and not hasattr(sampler,'state_dict') and
            cfg.data.seed==data_seed(trainer) and cfg.data.shuffle is True and cfg.data.train_batch_size==16 and
            cfg.data.dataloader_num_workers==4, 'Native seeded non-stateful RandomSampler profile differs')
    return sampler


def initialize_factorial_zero(trainer,data):
    require(enabled(trainer) and trainer.global_steps==0, 'Private sampler initialization is step-zero factorial only')
    sampler=native_sampler(trainer)
    main=data['_snapshot']['_main_snapshot']
    require(data['_snapshot']['_snapshot_step']==0 and main['_sampler_iter_yielded']==0 and
            main['_sampler_iter_state']=={'samples_yielded':0} and
            data['_steps_since_snapshot']==0 and data['_iterator_finished'] is False,
            'Anchor dataloader is not the exact untouched beginning')
    state=torch.Generator(device='cpu').manual_seed(1).get_state()
    sampler.generator.set_state(state)
    trainer._caft_sampler_epoch_start=state.clone()
    trainer._caft_sampler_preserve_epoch_start_once=True
    require(fingerprint(sampler.generator.get_state())==seed_state_sha256(1), 'Initial private sampler readback differs')


def initialize_v2_zero(trainer,capture):
    """v2 step zero: epoch 1 starts from the untouched declared data-seed generator state.

    The capture arm calls this before its step-zero save, so its live generator must still be that state. A loading
    arm calls it after the common restore, whose digest check builds (and prefetches from) an iterator; that live
    iterator is never rewound. The epoch-start state is the seed state itself, and the caller checks it against the
    captured checkpoint. Verified empirically: capture, load-after-digest and an uninterrupted loader yield the same
    first two epochs."""
    require(_kind(trainer)==V2_KIND and trainer.global_steps==0, 'v2 sampler initialization is step-zero only')
    sampler=native_sampler(trainer)
    state=torch.Generator(device='cpu').manual_seed(data_seed(trainer)).get_state()
    if capture:
        require(fingerprint(sampler.generator.get_state())==fingerprint(state),
                'Native sampler generator is not the untouched declared data-seed state')
    trainer._caft_sampler_epoch_start=state
    trainer._caft_sampler_preserve_epoch_start_once=True


def prepare_epoch(trainer):
    if not enabled(trainer):return
    sampler=native_sampler(trainer)
    if getattr(trainer,'_caft_sampler_preserve_epoch_start_once',False):
        # A step-zero save can already prefetch indices. Keep the state from
        # before that iterator was constructed; do not rewind a live iterator.
        trainer._caft_sampler_preserve_epoch_start_once=False
        fingerprint(trainer._caft_sampler_epoch_start)
    else:
        trainer._caft_sampler_epoch_start=sampler.generator.get_state().clone()


def capture(trainer,path,data_binding):
    if not enabled(trainer):return None
    sampler=native_sampler(trainer)
    start=getattr(trainer,'_caft_sampler_epoch_start',None)
    current=sampler.generator.get_state().clone()
    payload={'kind':KIND,'global_step':int(trainer.global_steps),'dataset_rows':992,'batch_size':16,
        'seed':data_seed(trainer),'sampler':'torch.utils.data.RandomSampler','epoch_start_state':start.clone() if start is not None else None,
        'current_state':current,'epoch_start_state_sha256':fingerprint(start),
        'current_state_sha256':fingerprint(current),'data_pt':dict(data_binding)}
    compact.save_torch(Path(path)/FILENAME,payload)
    return compact.ref(Path(path)/FILENAME)


def restore(trainer,path,manifest):
    if not enabled(trainer):return
    sampler=native_sampler(trainer);path=Path(path)
    require(FILENAME in manifest['files'] and compact.ref(path/FILENAME)==manifest['files'][FILENAME],
            'Missing or changed private sampler checkpoint')
    saved=torch.load(path/FILENAME,map_location='cpu',weights_only=False)
    require(set(saved)=={'kind','global_step','dataset_rows','batch_size','seed','sampler',
        'epoch_start_state','current_state','epoch_start_state_sha256','current_state_sha256','data_pt'} and
        saved['kind']==KIND and saved['global_step']==trainer.global_steps and saved['dataset_rows']==992 and
        saved['batch_size']==16 and saved['seed']==data_seed(trainer) and saved['sampler']=='torch.utils.data.RandomSampler' and
        saved['data_pt']==manifest['files']['data.pt'] and compact.ref(path/'data.pt')==saved['data_pt'],
        'Private sampler profile or native dataloader snapshot changed')
    require(fingerprint(saved['epoch_start_state'])==saved['epoch_start_state_sha256'] and
            fingerprint(saved['current_state'])==saved['current_state_sha256'], 'Private sampler semantic hashes changed')
    loader=trainer.train_dataloader
    require(loader._iterator is None and loader.next_iter_state is not None,
            'Private sampler restore must precede native iterator creation')
    # Native resume either replays the saved offset within its epoch, or starts
    # the next epoch when the saved iterator was already exhausted.
    finished=loader.next_iter_state['_iterator_finished']
    require(type(finished) is bool, 'Native iterator completion state is malformed')
    target=saved['current_state'] if finished else saved['epoch_start_state']
    sampler.generator.set_state(target)
    trainer._caft_sampler_epoch_start=target.clone()
    trainer._caft_sampler_preserve_epoch_start_once=True
    require(fingerprint(sampler.generator.get_state())==fingerprint(target), 'Private sampler RNG restore readback differs')
