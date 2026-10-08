"""CPU tests for the step-4 seed-bound common start (caft_initialization V2_KIND, protocol 7.4 / 8.3).

- One rank.
- A real StatefulDataLoader with a seeded RandomSampler.
- Toy LoRA workers, as in test_caft_initialization.py.
- No Ray, model or runtime launch.
"""
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

PACKAGE = '_bestcase_v2_common_test_package'
pkg = types.ModuleType(PACKAGE); pkg.__path__ = [str(Path(__file__).parent)]
sys.modules[PACKAGE] = pkg
spec = importlib.util.spec_from_file_location(PACKAGE + '.caft_initialization', Path(__file__).with_name('caft_initialization.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.compact
sampler_spec = importlib.util.spec_from_file_location(PACKAGE + '.caft_sampler', Path(__file__).with_name('caft_sampler.py'))
sampler_module = importlib.util.module_from_spec(sampler_spec); sys.modules[PACKAGE + '.caft_sampler'] = sampler_module
sampler_spec.loader.exec_module(sampler_module)

V2 = 'bestcase_multilayer_response_predictor_caft_v2'


class Config(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as error: raise AttributeError(key) from error
    __setattr__ = dict.__setitem__

    def get(self, key, default=None):
        return dict.get(self, key, default)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.block = torch.nn.Module()
        self.block.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 1, bias=False)})
        self.block.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(1, 2, bias=False)})
        torch.nn.init.zeros_(self.block.lora_B['default'].weight)


def loader(seed):
    from torch.utils.data import RandomSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    data = list(range(992))
    return StatefulDataLoader(data, batch_size=16, sampler=RandomSampler(data, generator=torch.Generator().manual_seed(seed)),
                              num_workers=4, drop_last=True)


def caft_spec(arm, strength=1.0):
    return {'protocol': V2, 'arm': arm, 'layers': [22, 23], 'strength': strength, 'q': {'file': {}, 'tensors': {}},
            'scope': 'response_predictors', 'reference_projection': True, 'evaluation_projection': False,
            'rollout_prefix_caching': True, 'rollout_enforce_eager': False, 'direction': {'source': arm}}


class V2CommonStartTests(unittest.TestCase):
    CAPTURE = 'none_zero'

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.input = self.root / 'data.json'; self.input.write_text('{"train":992}')
        self.inputs = {'data': {'path': str(self.input), **c.ref(self.input)}}
        self.capture = self.trainer(self.CAPTURE, seed=1, mode='capture', strength=0.0)
        self.capture_workers = self.make_workers(self.capture)
        m.prepare_coordinator(self.capture)
        self.save(self.capture, self.capture_workers)
        m.finish_coordinator(self.capture)
        p = self.root / f'{self.CAPTURE}_s1/global_step_0/COMMON_M0.json'
        self.common_ref = {'path': str(p), **c.ref(p)}

    def config(self, arm, seed, strength=1.0, penalty=False):
        parts = m.seed_components(seed)
        ckpt = Config(kind='compact_lora_training_state_v1', training_identity_sha256=m.digest([arm, seed]),
                      base_manifest_sha256='b' * 64)
        worker = Config(model=Config(caft=Config(caft_spec(arm, strength)), lora_init_seed=parts['lora_init'],
                                     rollout_gen_seed=parts['rollout_gen']),
                        actor=Config(caft_checkpoint=ckpt, strategy='fsdp2', fsdp_config=Config(fsdp_size=1),
                                     ulysses_sequence_parallel_size=1, optim=Config(total_training_steps=200),
                                     ppo_mini_batch_size=16, ppo_micro_batch_size=None, ppo_micro_batch_size_per_gpu=4),
                        rollout=Config(mode='sync', name='vllm', seed=0, n=16, temperature=0.7))
        reward_kwargs = Config(caft_reward_sandbox=Config(spool_dir=str(self.root / arm / 'spool')))
        config = Config(actor_rollout_ref=worker,
                        data=Config(seed=parts['data'], train_files=str(self.input), shuffle=True, train_batch_size=16,
                                    dataloader_num_workers=4),
                        trainer=Config(default_local_dir=str(self.root / f'{arm}_s{seed}'), resume_mode='disable',
                                       n_gpus_per_node=1, nnodes=1, caft_pilot_stop_step=200, total_training_steps=200,
                                       save_freq=10, max_actor_ckpt_to_keep=None),
                        reward_model=Config(reward_kwargs=reward_kwargs),
                        ray_kwargs=Config(ray_init=Config(num_cpus=8, num_gpus=2 if penalty else 1)))
        if penalty:
            reward_kwargs['caft_penalty'] = Config(kind='activation_threshold', value=-3.0)
            config['activation_cache'] = Config(enabled=True, layers=[34])
        return config

    def trainer(self, arm, seed, mode, strength=1.0, penalty=False, manifest=None):
        config = self.config(arm, seed, strength, penalty)
        config.actor_rollout_ref.actor.caft_checkpoint['common_m0'] = m.v2_declaration(
            config, mode, seed, self.CAPTURE, self.inputs, manifest if mode == 'load' else None)
        return NS(config=config, global_steps=0, train_dataloader=loader(config.data.seed))

    def make_workers(self, trainer):
        model = Toy(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        manager = NS(optimizer=optimizer, lr_scheduler=scheduler,
                     checkpoint_save_contents=['optimizer', 'extra'], checkpoint_load_contents=['optimizer', 'extra'])
        manager.rng = {'torch': torch.get_rng_state().clone()}
        manager.get_rng_state = lambda: copy.deepcopy(manager.rng)
        manager.load_rng_state = lambda state: setattr(manager, 'rng', copy.deepcopy(state))
        rollout = NS(state={'request_counter': 0, 'source_hashes': {'public': 'c' * 64},
                            'worker': {'rng': torch.tensor([7, 0], dtype=torch.uint8), 'pid': 100,
                                       'caft_spec': copy.deepcopy(trainer.config.actor_rollout_ref.model.caft)}})
        rollout.caft_export_checkpoint_state = lambda: copy.deepcopy(rollout.state)
        def restore(state):
            q = rollout.state['worker']['caft_spec']
            rollout.state = copy.deepcopy(state); rollout.state['worker']['caft_spec'] = q
        rollout.caft_restore_common_initial_state = restore
        rollout.caft_restore_checkpoint_state = lambda state: None
        worker_config = copy.deepcopy(trainer.config.actor_rollout_ref)
        worker_config.actor.ppo_mini_batch_size = 256          # the actor's one-rank normalization of 16x16
        workers = [NS(config=worker_config, rank=0, world_size=1, _is_lora=True, actor_module_fsdp=model,
                      checkpoint_manager=manager, rollout=rollout, base_sync_done=True,
                      torch_random_states=torch.get_rng_state().clone(), gen_random_states=torch.get_rng_state().clone())]
        trainer.actor_rollout_wg = NS(caft_initialize_common_m0=lambda decl: [m.initialize_worker(w, decl) for w in workers])
        return workers

    def save(self, trainer, workers):
        root = Path(trainer.config.trainer.default_local_dir) / 'global_step_0'; actor = root / 'actor'
        actor.mkdir(parents=True)
        for w in workers:
            suffix = f'world_size_1_rank_{w.rank}.pt'
            c.save_torch(actor / ('optim_' + suffix), w.checkpoint_manager.optimizer.state_dict())
            c.save_torch(actor / ('extra_state_' + suffix), {'rng': w.checkpoint_manager.get_rng_state(),
                                                             'lr_scheduler': w.checkpoint_manager.lr_scheduler.state_dict()})
            c.save_worker_extra(w, actor, 0)
        (actor / 'lora_adapter').mkdir()
        (actor / 'lora_adapter/adapter_model.safetensors').write_bytes(b'toy adapter')
        (actor / 'lora_adapter/adapter_config.json').write_text('{}')
        c.save_torch(root / 'data.pt', trainer.train_dataloader.state_dict())
        c.seal_step(trainer, root)

    def load_arm(self, arm, **kwargs):
        target = self.trainer(arm, seed=kwargs.pop('seed', 1), mode='load', manifest=kwargs.pop('manifest', self.common_ref),
                              **kwargs)
        workers = self.make_workers(target)
        return target, workers

    def test_seed_components_are_stable_and_distinct(self):
        a, b = m.seed_components(1), m.seed_components(2)
        self.assertEqual(a, m.seed_components(1))
        self.assertTrue(all(a[k] != b[k] for k in a))
        self.assertEqual(len(set(a.values())), 3)
        with self.assertRaises(ValueError):
            m.seed_components(0)

    def test_capture_records_kind_seed_and_sampler_state(self):
        common = json.loads(Path(self.common_ref['path']).read_bytes())
        self.assertEqual((common['kind'], common['seed'], common['source_arm']), (m.V2_KIND, 1, self.CAPTURE))
        manifest = json.loads((self.root / f'{self.CAPTURE}_s1/global_step_0/CAFT_CHECKPOINT.json').read_bytes())
        self.assertIn(sampler_module.FILENAME, manifest['files'])
        self.assertEqual(manifest['world_size'], 1)

    def test_caft_and_penalty_arms_load_identical_state(self):
        source = json.loads(Path(self.common_ref['path']).read_bytes())
        for arm, kwargs in (('pair_caft', {}), ('pair_penalty', {'strength': 0.0, 'penalty': True})):
            target, workers = self.load_arm(arm, **kwargs)
            m.prepare_coordinator(target)
            self.save(target, workers); m.finish_coordinator(target)
            result = json.loads((Path(target.config.trainer.default_local_dir) / 'global_step_0/COMMON_M0_INITIALIZATION.json').read_bytes())
            for key in ('rank_state_sha256', 'data_state_sha256', 'coordinator_rng_sha256'):
                self.assertEqual(result[key], source[key], (arm, key))
            self.assertEqual(workers[0].config.model.caft['arm'], arm)
            self.assertEqual(workers[0].rollout.state['worker']['caft_spec']['arm'], arm)

    def test_wrong_seed_components_rejected(self):
        target, _ = self.load_arm('pair_caft')
        target.config.actor_rollout_ref.model.lora_init_seed += 1
        with self.assertRaisesRegex(ValueError, 'config differs|seed'):
            m.prepare_coordinator(target)
        target, _ = self.load_arm('pair_caft')
        target.config.data.seed += 1
        with self.assertRaisesRegex(ValueError, 'config differs|seed'):
            m.prepare_coordinator(target)

    def test_other_seed_start_rejected(self):
        target, _ = self.load_arm('pair_caft', seed=2)
        with self.assertRaisesRegex(ValueError, 'kind/seed differs'):
            m.prepare_coordinator(target)

    def test_non_capture_arm_cannot_capture(self):
        target = self.trainer('pair_caft', seed=1, mode='capture')
        self.make_workers(target)
        with self.assertRaisesRegex(ValueError, 'capture arm'):
            m.prepare_coordinator(target)

    def test_changed_non_arm_setting_rejected(self):
        target, workers = self.load_arm('pair_caft')
        workers[0].config.rollout.temperature = 0.8
        with self.assertRaisesRegex(ValueError, 'config differs'):
            m.initialize_worker(workers[0], target.config.actor_rollout_ref.actor.caft_checkpoint['common_m0'])

    def test_v1_spec_rejected_in_v2_declaration(self):
        target, _ = self.load_arm('pair_caft')
        target.config.actor_rollout_ref.model.caft['protocol'] = 'qwen3_l21_response_predictor_caft_v1'
        with self.assertRaisesRegex(ValueError, 'v2 CAFT'):
            m.declaration(target.config)

    def test_two_hundred_updates_and_ten_step_checkpoints_required(self):
        self.assertEqual(m.pilot_stop(self.capture.config, 200), 200)
        changed = copy.deepcopy(self.capture.config); changed.trainer.save_freq = 20
        with self.assertRaisesRegex(ValueError, '200 completed updates'):
            m.pilot_stop(changed, 200)
        changed = copy.deepcopy(self.capture.config); changed.trainer.caft_pilot_stop_step = 5
        with self.assertRaisesRegex(ValueError, '200 completed updates'):   # a short run must declare itself
            m.pilot_stop(changed, 200)

    def test_declared_smoke_run_may_stop_early_without_changing_schedule(self):
        smoke = copy.deepcopy(self.capture.config)
        smoke.trainer.update(caft_smoke=True, caft_pilot_stop_step=5, save_freq=2)
        self.assertEqual(m.pilot_stop(smoke, 200), 5)
        for key, value in (('caft_pilot_stop_step', 11), ('save_freq', 3), ('total_training_steps', 5)):
            bad = copy.deepcopy(smoke); bad.trainer[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '200 completed updates'):
                m.pilot_stop(bad, 200)


    def test_declared_four_hundred_step_schedule_must_agree_everywhere(self):
        long = copy.deepcopy(self.capture.config)
        long.trainer.update(total_training_steps=400, caft_pilot_stop_step=400)
        long.actor_rollout_ref.actor.optim.total_training_steps = 400
        self.assertEqual(m.pilot_stop(long, 400), 400)
        for name, change in (('scheduler', lambda c: c.actor_rollout_ref.actor.optim.update(total_training_steps=200)),
                             ('stop', lambda c: c.trainer.update(caft_pilot_stop_step=200)),
                             ('trainer', lambda c: c.trainer.update(total_training_steps=200))):
            bad = copy.deepcopy(long); change(bad)
            with self.subTest(changed=name), self.assertRaisesRegex(ValueError, '200 completed updates'):
                m.pilot_stop(bad, 400)
        with self.assertRaisesRegex(ValueError, '200 completed updates'):
            m.pilot_stop(long, 200)          # the trainer's own count must match the declared schedule
        other = copy.deepcopy(long)
        other.trainer.update(total_training_steps=300, caft_pilot_stop_step=300)
        other.actor_rollout_ref.actor.optim.total_training_steps = 300
        with self.assertRaisesRegex(ValueError, '200 completed updates'):
            m.pilot_stop(other, 300)         # only 200 or 400
        smoke = copy.deepcopy(long); smoke.trainer.update(caft_smoke=True, caft_pilot_stop_step=2, save_freq=2)
        self.assertEqual(m.pilot_stop(smoke, 400), 2)


if __name__ == '__main__':
    unittest.main()
