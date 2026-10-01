"""Small CPU LoRA/Adam round trip; no model/runtime/training launch."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

spec = importlib.util.spec_from_file_location("compact_checkpoint", Path(__file__).with_name("caft_checkpoint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Config(dict):
    __getattr__ = dict.__getitem__


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 3)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.layer = torch.nn.Module()
        self.layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})
    def forward(self, x):
        return self.base(x) + self.layer.lora_B["default"](self.layer.lora_A["default"](x))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name).resolve() / "global_step_1"; (self.path / "actor").mkdir(parents=True)
        self.model = Toy()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.01)
        self.cfg = {"kind":"compact_lora_training_state_v1", "training_identity_sha256":"a"*64,
                    "base_manifest_sha256":"b"*64}
        actor = Config(caft_checkpoint=self.cfg, strategy="fsdp2", fsdp_config=Config(fsdp_size=1),
                       ulysses_sequence_parallel_size=1)
        self.manager = NS(checkpoint_save_contents=["optimizer","extra"], checkpoint_load_contents=["optimizer","extra"])
        self.rollout = NS(caft_export_checkpoint_state=lambda:{"counter":7},
                          caft_restore_checkpoint_state=lambda state:setattr(self.rollout,"restored",state))
        self.worker = NS(config=Config(actor=actor,rollout=Config(mode="sync",name="vllm")),
            _is_lora=True, rank=0, world_size=1, checkpoint_manager=self.manager, actor_module_fsdp=self.model, base_sync_done=True,
            torch_random_states=torch.get_rng_state(), gen_random_states=torch.get_rng_state(), rollout=self.rollout)

    def step(self, model, optimizer, x):
        optimizer.zero_grad(); model(x).square().mean().backward(); optimizer.step()

    def save(self):
        actor = self.path / "actor"
        m.save_torch(actor / "optim_world_size_1_rank_0.pt", self.optimizer.state_dict())
        m.save_torch(actor / "extra_state_world_size_1_rank_0.pt", {"rng":torch.get_rng_state()})
        (actor / "lora_adapter").mkdir()
        (actor / "lora_adapter/adapter_model.safetensors").write_bytes(b"authored adapter")
        (actor / "lora_adapter/adapter_config.json").write_text("{}")
        m.save_worker_extra(self.worker, actor, 1)
        m.save_torch(self.path / "data.pt", {"next_batch":2})
        trainer = NS(global_steps=1,config=Config(actor_rollout_ref=Config(actor=Config(caft_checkpoint=self.cfg)),
                     trainer=Config(n_gpus_per_node=1,nnodes=1)))
        m.seal_step(trainer,self.path)
        self.cfg["resume_manifest_sha256"]=m.ref(self.path/"CAFT_CHECKPOINT.json")["sha256"]
        return json.loads((self.path / "CAFT_CHECKPOINT.json").read_bytes())

    def test_lora_adam_next_step_round_trip_preserves_base_and_moments(self):
        x = torch.arange(12,dtype=torch.float32).reshape(4,3)/10
        self.step(self.model,self.optimizer,x)
        base = {k:v.clone() for k,v in self.model.base.state_dict().items()}
        expected = copy.deepcopy(self.model)
        expected_optim = torch.optim.AdamW([p for p in expected.parameters() if p.requires_grad],lr=.01)
        expected_optim.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        manifest = self.save(); self.assertTrue(manifest["exact_stochastic_resume_available"])
        saved = torch.load(self.path / "actor/caft_local_world_size_1_rank_0.pt",weights_only=False)
        self.assertEqual(set(saved["trainables"]),set(m.trainable_parameters(self.model)))
        self.assertFalse(any("base" in key for key in saved["trainables"]))
        with torch.no_grad():
            for p in m.trainable_parameters(self.model).values(): p.add_(10)
        value = m.load_worker_extra(self.worker,self.path/"actor")
        self.optimizer.load_state_dict(torch.load(self.path/"actor/optim_world_size_1_rank_0.pt",weights_only=False))
        m.finish_worker_restore(self.worker,value)
        self.step(self.model,self.optimizer,x); self.step(expected,expected_optim,x)
        for a,b in zip(self.model.parameters(),expected.parameters()): self.assertTrue(torch.equal(a,b))
        for name,v in base.items(): self.assertTrue(torch.equal(v,self.model.base.state_dict()[name]))
        self.assertEqual(self.rollout.restored,{"counter":7}); self.assertTrue(self.worker.base_sync_done)

    def test_unknown_rollout_state_retained_but_exact_resume_rejected(self):
        self.worker.rollout = NS()
        manifest = self.save(); self.assertFalse(manifest["exact_stochastic_resume_available"])
        with self.assertRaisesRegex(ValueError,"rollout-state adapter"): m.load_worker_extra(self.worker,self.path/"actor")

    def test_modified_pickle_rejected_before_deserialization(self):
        self.save(); target=self.path/"actor/optim_world_size_1_rank_0.pt";target.write_bytes(b"bad pickle")
        with self.assertRaisesRegex(ValueError,"manifest changed"): m.load_worker_extra(self.worker,self.path/"actor")

    def test_changed_manifest_rejected_before_any_pickle(self):
        self.save();p=self.path/"CAFT_CHECKPOINT.json";p.write_text(p.read_text()+" ")
        with self.assertRaisesRegex(ValueError,"resume-manifest SHA"):m.load_worker_extra(self.worker,self.path/"actor")

    def test_changed_topology_or_identity_rejected(self):
        self.save();self.worker.world_size=2
        with self.assertRaisesRegex(ValueError,"identity/topology"):m.load_worker_extra(self.worker,self.path/"actor")
        self.worker.world_size=1;self.cfg["base_manifest_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"identity/topology"):m.load_worker_extra(self.worker,self.path/"actor")

    def test_full_model_config_or_trainable_base_rejected(self):
        self.manager.checkpoint_save_contents += ["model"]
        with self.assertRaisesRegex(ValueError,"without any full"):m.profile(self.worker)
        self.model.base.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,"frozen base"):m.capture_local_trainables(self.model)

    def test_incomplete_checkpoint_and_duplicate_write_rejected(self):
        with self.assertRaises(FileNotFoundError):m.validate_actor_checkpoint(self.path/"actor",self.cfg,1,1)
        self.save()
        with self.assertRaises(FileExistsError):m.save_torch(self.path/"data.pt",{})

    def test_wrong_tensor_shape_does_not_partially_restore(self):
        state=m.capture_local_trainables(self.model);before=copy.deepcopy(state)
        last=next(reversed(state));state[last]=torch.ones(9)
        with self.assertRaisesRegex(ValueError,"shape/dtype"):m.restore_local_trainables(self.model,state)
        for name,p in m.trainable_parameters(self.model).items():self.assertTrue(torch.equal(p,before[name]))

    def test_both_hybrid_rng_states_captured_separately(self):
        self.worker.gen_random_states=self.worker.gen_random_states.clone();self.worker.gen_random_states[0]^=1
        self.save();saved=torch.load(self.path/"actor/caft_local_world_size_1_rank_0.pt",weights_only=False)
        self.assertFalse(torch.equal(saved["gen_random_states"],saved["torch_random_states"]))

    def coordinator(self):
        return NS(global_steps=int(str(self.path).split("global_step_")[-1]),
            config=Config(actor_rollout_ref=Config(actor=Config(caft_checkpoint=self.cfg)),
                          trainer=Config(resume_from_path=str(self.path))))

    def test_coordinator_matching_step_restores_exact_rng(self):
        import numpy as np
        import random
        self.save()
        saved=torch.load(self.path/"caft_coordinator_rng.pt",weights_only=False)
        torch.rand(4);np.random.rand(4);random.random()
        m.restore_coordinator(self.coordinator())
        self.assertTrue(torch.equal(torch.get_rng_state(),saved["torch"]))
        self.assertEqual(random.getstate(),saved["random"])
        actual=np.random.get_state()
        self.assertEqual(actual[0],saved["numpy"][0])
        self.assertTrue(np.array_equal(actual[1],saved["numpy"][1]))
        self.assertEqual(actual[2:],saved["numpy"][2:])

    def test_renamed_resume_directory_rejected_before_rng_restore(self):
        self.save()
        self.path=self.path.rename(self.path.with_name("global_step_2"))
        before=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,"directory step differs"):
            m.restore_coordinator(self.coordinator())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_optimizer_step_mismatch_rejected_before_rng_restore(self):
        manifest=self.save()
        manifest["optimizer_steps_completed"]=0
        path=self.path/"CAFT_CHECKPOINT.json"
        path.write_text(json.dumps(manifest))
        self.cfg["resume_manifest_sha256"]=m.ref(path)["sha256"]
        before=torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,"directory step differs"):
            m.restore_coordinator(self.coordinator())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__ == "__main__": unittest.main()
