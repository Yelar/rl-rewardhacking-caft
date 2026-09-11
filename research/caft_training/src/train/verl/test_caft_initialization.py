"""CPU-only real Torch state round trips; no Ray/model/runtime launch."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace as NS
import unittest
import torch

PACKAGE = '_caft_initialization_test_package'
pkg = types.ModuleType(PACKAGE); pkg.__path__ = [str(Path(__file__).parent)]
sys.modules[PACKAGE] = pkg
spec = importlib.util.spec_from_file_location(PACKAGE + '.caft_initialization', Path(__file__).with_name('caft_initialization.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.compact


class Config(dict):
    def __getattr__(self, key):
        try:return self[key]
        except KeyError as error:raise AttributeError(key) from error
    __setattr__ = dict.__setitem__


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.block = torch.nn.Module()
        self.block.lora_A = torch.nn.ModuleDict({'default':torch.nn.Linear(2, 1, bias=False)})
        self.block.lora_B = torch.nn.ModuleDict({'default':torch.nn.Linear(1, 2, bias=False)})
        torch.nn.init.zeros_(self.block.lora_B['default'].weight)


class Loader:
    def __init__(self): self.state = {'index':0, 'seed':1, 'generator':torch.get_rng_state().clone()}
    def state_dict(self): return copy.deepcopy(self.state)
    def load_state_dict(self, state): self.state = copy.deepcopy(state)


class InitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.input = self.root / 'data.json'; self.input.write_text('{"original":992}')
        self.baseline = self.trainer('baseline')
        self.workers = self.make_workers(self.baseline)
        m.prepare_coordinator(self.baseline)
        self.save(self.baseline, self.workers)
        m.finish_coordinator(self.baseline)
        p = self.root / 'baseline/global_step_0/COMMON_M0.json'
        self.common_ref = {'path':str(p), **c.ref(p)}

    def trainer(self, arm):
        ckpt = Config(kind='compact_lora_training_state_v1', training_identity_sha256=m.digest(arm),
                      base_manifest_sha256='b'*64)
        worker = Config(model=Config(caft=Config(arm=arm, q=None if arm == 'baseline' else arm)),
            actor=Config(caft_checkpoint=ckpt, strategy='fsdp2', fsdp_config=Config(fsdp_size=1),
                ulysses_sequence_parallel_size=1, optim=Config(total_training_steps=200),
                ppo_mini_batch_size=16, ppo_micro_batch_size=None, ppo_micro_batch_size_per_gpu=4),
            rollout=Config(mode='sync', name='vllm', seed=0, n=16))
        config = Config(actor_rollout_ref=worker, data=Config(seed=1, train_files=str(self.input)),
            trainer=Config(default_local_dir=str(self.root/arm), resume_mode='disable',
                n_gpus_per_node=8, nnodes=1, caft_pilot_stop_step=100, total_training_steps=200,
                save_freq=10, max_actor_ckpt_to_keep=None),
            reward_model=Config(reward_kwargs=Config(caft_reward_sandbox=Config(spool_dir=str(self.root/arm/'spool')))))
        contract = {'worker_config_sha256':m.worker_config_digest(worker),
                    'coordinator_config_sha256':m.coordinator_config_digest(config),
                    'inputs':{'data':{'path':str(self.input), **c.ref(self.input)}}}
        ckpt['common_m0'] = {'kind':m.KIND, 'mode':'capture' if arm == 'baseline' else 'load',
            'contract':contract, 'common_identity_sha256':m.digest(contract)}
        if arm != 'baseline': ckpt['common_m0']['common_manifest'] = self.common_ref
        return NS(config=config, global_steps=0, train_dataloader=Loader())

    def make_workers(self, trainer):
        workers = []
        for rank in range(8):
            model = Toy(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:1.0)
            manager = NS(optimizer=optimizer, lr_scheduler=scheduler,
                checkpoint_save_contents=['optimizer','extra'], checkpoint_load_contents=['optimizer','extra'])
            manager.rng = {'torch':torch.get_rng_state().clone()}
            manager.get_rng_state = lambda manager=manager:copy.deepcopy(manager.rng)
            manager.load_rng_state = lambda state,manager=manager:setattr(manager,'rng',copy.deepcopy(state))
            rollout = NS(state={'request_counter':0, 'source_hashes':{'public':'c'*64},
                'worker':{'rng':torch.tensor([rank, 0],dtype=torch.uint8), 'pid':100+rank,
                          'caft_spec':copy.deepcopy(trainer.config.actor_rollout_ref.model.caft)}})
            rollout.caft_export_checkpoint_state = lambda rollout=rollout:copy.deepcopy(rollout.state)
            def restore(state, rollout=rollout):
                q = rollout.state['worker']['caft_spec']
                rollout.state = copy.deepcopy(state); rollout.state['worker']['caft_spec'] = q
            rollout.caft_restore_common_initial_state = restore
            rollout.caft_restore_checkpoint_state = lambda state:None
            worker_config=copy.deepcopy(trainer.config.actor_rollout_ref)
            worker_config.actor.ppo_mini_batch_size=32
            workers.append(NS(config=worker_config, rank=rank, world_size=8, _is_lora=True,
                actor_module_fsdp=model, checkpoint_manager=manager, rollout=rollout, base_sync_done=True,
                torch_random_states=torch.get_rng_state().clone(), gen_random_states=torch.get_rng_state().clone()))
        trainer.actor_rollout_wg = NS(caft_initialize_common_m0=lambda decl:[m.initialize_worker(w, decl) for w in workers])
        return workers

    def save(self, trainer, workers):
        root = Path(trainer.config.trainer.default_local_dir)/'global_step_0'; actor=root/'actor'
        actor.mkdir(parents=True)
        for w in workers:
            suffix=f'world_size_8_rank_{w.rank}.pt'
            c.save_torch(actor/('optim_'+suffix),w.checkpoint_manager.optimizer.state_dict())
            c.save_torch(actor/('extra_state_'+suffix),{'rng':w.checkpoint_manager.get_rng_state(),
                'lr_scheduler':w.checkpoint_manager.lr_scheduler.state_dict()})
            c.save_worker_extra(w, actor, 0)
        (actor/'lora_adapter').mkdir()
        (actor/'lora_adapter/adapter_model.safetensors').write_bytes(b'authored toy adapter')
        (actor/'lora_adapter/adapter_config.json').write_text('{}')
        c.save_torch(root/'data.pt',trainer.train_dataloader.state_dict())
        c.seal_step(trainer,root)

    def test_eight_rank_both_arms_restore_same_state_keep_own_Q_and_seal(self):
        source = json.loads(Path(self.common_ref['path']).read_bytes())
        for arm in ('pc4','random0'):
            target = self.trainer(arm); workers = self.make_workers(target)
            original_base = [w.actor_module_fsdp.base.detach().clone() for w in workers]
            m.prepare_coordinator(target)
            self.save(target,workers); m.finish_coordinator(target)
            result=json.loads((self.root/arm/'global_step_0/COMMON_M0_INITIALIZATION.json').read_bytes())
            self.assertEqual(result['rank_state_sha256'],source['rank_state_sha256'])
            self.assertEqual(result['data_state_sha256'],source['data_state_sha256'])
            self.assertEqual(result['coordinator_rng_sha256'],source['coordinator_rng_sha256'])
            for w,base in zip(workers,original_base):
                self.assertEqual(w.config.model.caft.arm,arm)
                self.assertEqual(w.rollout.state['worker']['caft_spec']['q'],arm)
                self.assertTrue(torch.equal(base,w.actor_module_fsdp.base))
                self.assertTrue(w.base_sync_done)

    def test_config_hardware_or_data_change_rejected(self):
        target=self.trainer('pc4'); target.config.data.seed=99
        with self.assertRaisesRegex(ValueError,'config differs'):m.prepare_coordinator(target)

    def test_worker_optimizer_config_change_rejected(self):
        target=self.trainer('pc4'); workers=self.make_workers(target)
        workers[0].config.rollout.seed=9
        with self.assertRaisesRegex(ValueError,'config differs'):m.initialize_worker(workers[0],target.config.actor_rollout_ref.actor.caft_checkpoint['common_m0'])

    def test_current_optimizer_must_be_empty(self):
        target=self.trainer('pc4'); workers=self.make_workers(target)
        for p in c.trainable_parameters(workers[0].actor_module_fsdp).values():p.grad=torch.ones_like(p)
        workers[0].checkpoint_manager.optimizer.step()
        with self.assertRaisesRegex(ValueError,'matching fresh M0'):m.prepare_coordinator(target)

    def test_changed_rank_bytes_rejected_before_deserialization(self):
        target=self.trainer('pc4'); self.make_workers(target)
        (self.root/'baseline/global_step_0/actor/caft_local_world_size_8_rank_0.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'bytes differ'):m.prepare_coordinator(target)

    def test_common_source_cannot_have_completed_update_or_nonzero_B(self):
        manifest=json.loads((self.root/'baseline/global_step_0/CAFT_CHECKPOINT.json').read_bytes())
        payload,opt,extra=m.load_rank(self.root/'baseline/global_step_0',manifest,0)
        payload['global_step']=1
        with self.assertRaisesRegex(ValueError,'step zero'):m.rank_content(payload,opt,extra)
        payload['global_step']=0
        payload['trainables']['block.lora_B.default.weight'].fill_(1)
        with self.assertRaisesRegex(ValueError,'B is nonzero'):m.rank_content(payload,opt,extra)

    def test_source_request_counter_must_be_zero(self):
        manifest=json.loads((self.root/'baseline/global_step_0/CAFT_CHECKPOINT.json').read_bytes())
        values=m.load_rank(self.root/'baseline/global_step_0',manifest,0)
        values[0]['rollout_state']['request_counter']=1
        with self.assertRaisesRegex(ValueError,'already generated'):m.rank_content(*values)

    def test_resume_path_is_not_common_initialization(self):
        target=self.trainer('pc4'); target.config.trainer.resume_mode='resume_path'
        # The changed common config is already forbidden; no old resume API is called.
        with self.assertRaises(ValueError):m.prepare_coordinator(target)

    def test_nonbaseline_capture_rejected(self):
        target=self.trainer('pc4'); decl=target.config.actor_rollout_ref.actor.caft_checkpoint['common_m0']
        decl['mode']='capture';del decl['common_manifest']
        with self.assertRaisesRegex(ValueError,'Only baseline'):m.prepare_coordinator(target)

    def test_data_source_byte_change_rejected(self):
        self.input.write_text('changed')
        with self.assertRaisesRegex(ValueError,'input bytes differ'):m.prepare_coordinator(self.baseline)

    def test_hundred_stop_preserves_two_hundred_scheduler(self):
        self.assertEqual(m.pilot_stop(self.baseline.config,200),100)
        self.baseline.config.actor_rollout_ref.actor.optim.total_training_steps=100
        with self.assertRaisesRegex(ValueError,'200-step scheduler'):m.pilot_stop(self.baseline.config,200)

    def test_other_stops_and_checkpoint_cadence_rejected(self):
        for stop in (60, 99, 101, 200):
            changed=copy.deepcopy(self.baseline.config)
            changed.trainer.caft_pilot_stop_step=stop
            with self.subTest(stop=stop), self.assertRaisesRegex(ValueError,'100 completed updates'):
                m.pilot_stop(changed,200)
        for branch,key,value in [('trainer','total_training_steps',100),
                                  ('trainer','save_freq',20),
                                  ('trainer','max_actor_ckpt_to_keep',1)]:
            changed=copy.deepcopy(self.baseline.config)
            changed[branch][key]=value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'200-step scheduler'):
                m.pilot_stop(changed,200)
        with self.assertRaisesRegex(ValueError,'200-step scheduler'):
            m.pilot_stop(self.baseline.config,100)

    def test_ordinary_training_stop_unchanged(self):
        del self.baseline.config.trainer['caft_pilot_stop_step']
        self.assertEqual(m.pilot_stop(self.baseline.config,200),200)

    def test_step_zero_same_arm_resume_does_not_initialize_or_resave(self):
        config=self.baseline.config
        self.assertTrue(m.needs_common_initialization(config,0))
        config.trainer.resume_mode='resume_path'
        self.assertFalse(m.needs_common_initialization(config,0))
        self.assertFalse(m.needs_common_initialization(config,10))
        config.trainer.resume_mode='disable'
        self.assertFalse(m.needs_common_initialization(config,10))


if __name__=='__main__':unittest.main()
