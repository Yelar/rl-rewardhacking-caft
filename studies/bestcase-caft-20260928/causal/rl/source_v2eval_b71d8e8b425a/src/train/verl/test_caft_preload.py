"""Real CPU restore -> unchanged native LoRA transfer branches; no GPU/model."""
import ast
import asyncio
from collections import OrderedDict
from dataclasses import dataclass,asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch
import torch

HERE=Path(__file__).parent
SOURCE=HERE.parents[2]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
common=module('source7_common_preload_fixture',HERE/'test_caft_initialization.py')
checkpoint=module('source7_checkpoint_preload_fixture',HERE/'test_caft_checkpoint.py')

def native_function(path,name,namespace,cls=None):
 tree=ast.parse(path.read_text());nodes=tree.body
 if cls:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
 node=next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 node.decorator_list=[]
 code=ast.unparse(node)
 exec('from __future__ import annotations\n'+code,namespace)
 return namespace[name]

@dataclass
class PeftConfig:
 r:int=32

class NativeTransfer:
 def __init__(self):
  self.events=[]
  def layered(model):
   self.events.append('layered_lora_collection')
   return OrderedDict((name,p.detach().clone()) for name,p in model.named_parameters() if 'lora_' in name)
  # The native FSDP routing/guard is executed verbatim. Only tensor-gather and
  # model-loading dependencies are authored CPU boundaries.
  ns={'OrderedDict':OrderedDict,'fsdp_version':lambda model:2,'layered_summon_lora_params':layered}
  self.collect=native_function(SOURCE/'verl/verl/utils/fsdp_utils.py','collect_lora_params',ns)
  ns2={'collect_lora_params':self.collect,'log_gpu_memory_usage':lambda *a,**k:None,
       'logger':NS(info=lambda *a:None),'convert_weight_keys':lambda values,model:values}
  self.per_tensor=native_function(SOURCE/'verl/verl/workers/engine/fsdp/transformer_impl.py',
                                  'get_per_tensor_param',ns2,cls='FSDPEngine')
  ns3={'asdict':asdict,'time':NS(time_ns=lambda:1001),'TensorLoRARequest':lambda **kw:NS(**kw),
       'logger':NS(info=lambda *a:None)}
  self.update=native_function(SOURCE/'verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
                              'update_weights',ns3,cls='vLLMRollout')
 def transfer(self,worker):
  worker.actor_module_fsdp.peft_config={'default':PeftConfig()}
  peft=ModuleType('peft');utils=ModuleType('peft.utils');save=ModuleType('peft.utils.save_and_load')
  save.get_peft_model_state_dict=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected non-layered path'))
  with patch.dict(sys.modules,{'peft':peft,'peft.utils':utils,'peft.utils.save_and_load':save}):
   values=self.per_tensor(NS(module=worker.actor_module_fsdp,_is_offload_param=False),
                          layered_summon=True,base_sync_done=worker.base_sync_done)
  observed=[];rollout=NS(caft_spec=None,inference_engine=NS(llm_engine=NS(add_lora=observed.append)))
  asyncio.run(self.update(rollout,values.items(),peft_config=PeftConfig(),base_sync_done=worker.base_sync_done))
  assert len(observed)==1
  return observed[0].lora_tensors

class PreloadRestoreTests(unittest.TestCase):
 def common_fixture(self,load_format='safetensors'):
  fixture=common.InitTests(methodName='test_eight_rank_both_arms_restore_same_state_keep_own_Q_and_seal')
  original=fixture.trainer;make=fixture.make_workers
  def trainer(arm):
   tr=original(arm);cfg=tr.config;cfg.actor_rollout_ref.rollout.update(load_format=load_format,layered_summon=True)
   decl=cfg.actor_rollout_ref.actor.caft_checkpoint['common_m0']
   decl['contract']['worker_config_sha256']=common.m.worker_config_digest(cfg.actor_rollout_ref)
   decl['contract']['coordinator_config_sha256']=common.m.coordinator_config_digest(cfg)
   decl['common_identity_sha256']=common.m.digest(decl['contract']);return tr
  def workers(tr):
   result=make(tr)
   for worker in result:worker.base_sync_done='dummy' not in worker.config.rollout.load_format
   return result
  fixture.trainer=trainer;fixture.make_workers=workers;fixture.setUp();self.addCleanup(fixture.doCleanups)
  target=fixture.trainer('pc4');result=fixture.make_workers(target)
  return fixture,target,result

 def test_common_restore_keeps_preloaded_base_and_native_adapter_refresh_succeeds(self):
  fixture,target,workers=self.common_fixture();common.m.prepare_coordinator(target)
  for worker in workers:self.assertTrue(worker.base_sync_done)
  bridge=NativeTransfer();actual=bridge.transfer(workers[0]);expected=common.c.capture_local_trainables(workers[0].actor_module_fsdp)
  self.assertEqual(set(actual),set(expected))
  for name in actual:self.assertTrue(torch.equal(actual[name],expected[name]))
  self.assertEqual(bridge.events,['layered_lora_collection'])

 def test_common_dummy_base_remains_false_and_native_layered_guard_rejects(self):
  fixture,target,workers=self.common_fixture('dummy');common.m.prepare_coordinator(target)
  for worker in workers:self.assertFalse(worker.base_sync_done)
  bridge=NativeTransfer()
  with self.assertRaisesRegex(ValueError,'base-model is preloaded'):bridge.transfer(workers[0])
  self.assertEqual(bridge.events,[])

 def test_same_arm_restore_preserves_true_false_and_refreshes_only_valid_base(self):
  fixture=checkpoint.CheckpointTests(methodName='test_lora_adam_next_step_round_trip_preserves_base_and_moments')
  fixture.setUp();self.addCleanup(fixture.doCleanups);fixture.save()
  value=checkpoint.m.load_worker_extra(fixture.worker,fixture.path/'actor')
  for preload in (True,False):
   fixture.worker.base_sync_done=preload;checkpoint.m.finish_worker_restore(fixture.worker,value)
   self.assertIs(fixture.worker.base_sync_done,preload);bridge=NativeTransfer()
   if preload:
    actual=bridge.transfer(fixture.worker);expected=checkpoint.m.capture_local_trainables(fixture.model)
    self.assertEqual(set(actual),set(expected))
    for name in actual:self.assertTrue(torch.equal(actual[name],expected[name]))
   else:
    with self.assertRaisesRegex(ValueError,'base-model is preloaded'):bridge.transfer(fixture.worker)

 def test_missing_preload_state_fails_without_manufacturing_true(self):
  fixture,target,workers=self.common_fixture();del workers[0].base_sync_done
  with self.assertRaisesRegex(ValueError,'Native base preload state'):common.m.prepare_coordinator(target)
  self.assertFalse(hasattr(workers[0],'base_sync_done'))

if __name__=='__main__':unittest.main(verbosity=2)
