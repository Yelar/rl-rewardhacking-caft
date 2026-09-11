"""Authored CPU diagnostics using causal toy states; no model/data downloads."""
import copy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from . import batch_diagnostics as d, batched_generation as b
from .test_batched_generation import row, Tokenizer, SAMPLING

class Decoder(torch.nn.Module):
    def __init__(self,block,leak=False):super().__init__();self.block=block;self.leak=leak
    def forward(self,input_ids,attention_mask,position_ids,past_key_values=None,use_cache=False,**kwargs):
        h=torch.zeros(*input_ids.shape,2560);h[:,:,0]=input_ids.float()%7/4;h[:,:,1]=position_ids.float()/8
        if self.leak:h[:,:,0]+=input_ids.float().sum(-1)[:,None]/10000
        h=self.block(h)
        return SimpleNamespace(last_hidden_state=h,past_key_values=(1,) if use_cache else None)
class Head(torch.nn.Module):
    def forward(self,h):
        logits=torch.full((*h.shape[:-1],151646),-100.)
        logits[...,:7]=torch.arange(7).float()/5+h[...,0,None]*torch.arange(7).float()/4
        logits[...,151643]=torch.where(h[...,1]>=.5,torch.tensor(100.),torch.tensor(-100.))
        return logits
class Model(torch.nn.Module):
    def __init__(self,leak=False):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.block=torch.nn.Identity().eval()
        self.decoder=Decoder(self.block,leak);self.lm_head=Head();self.eval()
    def forward(self,**kwargs):
        o=self.decoder(**kwargs);return SimpleNamespace(logits=self.lm_head(o.last_hidden_state[:,-1:]),past_key_values=o.past_key_values)

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.rows=[row(3,2),row(4,4),row(5,3)];self.model=Model()
        self.batch=[{'request_id':str(i),'record_id':r['record_id'],'condition_id':'baseline','scope':'primary','seed':11+i} for i,r in enumerate(self.rows)]
        self.task={'requests':self.batch,'sampling':SAMPLING,'batch_profile':{**b.PROFILE,'batch_size':4},
            'batch_diagnostics_policy':d.POLICY,'intervention_strength':.5,'conditions':{'baseline':{'layers':[]},'target':{'layers':[{'layer':0}]}}}
        self.q={0:torch.eye(2560)[:,:1].contiguous()};self.trace={}
        self.results=b.generate_batch(self.model,[self.model.block],Tokenizer(),self.rows,{},['primary']*3,[11,12,13],SAMPLING,
            batch_profile=self.task['batch_profile'],diagnostic_trace=self.trace)
    def run_diagnostic(self,**kwargs):
        return d.run(self.model,self.model.decoder,[self.model.block],Tokenizer(),self.task,self.batch,self.rows,self.results,self.trace,self.q,'target',deadline_epoch=time.time()+30,**kwargs)
    def test_actual_prefix_positions_rng_projection_restoration_and_drift(self):
        report=self.run_diagnostic();self.assertEqual(report['status'],'passed_real_prefix_checks')
        self.assertTrue(all(report['checks'].values()));self.assertFalse(report['distribution_equivalence_established'])
        self.assertEqual(report['baseline_restoration']['max_abs'],0.)
        self.assertEqual(report['future_prefix_invariance']['max_abs'],0.)
        self.assertEqual(report['cached_vs_teacher_forced']['max_abs'],0.)
        self.assertEqual(report['batch_vs_individual']['max_abs'],0.)
        self.assertEqual(len(self.model.block._forward_hooks),0)
    def test_future_leak_rejected_and_hooks_released(self):
        self.model.decoder.leak=True
        with self.assertRaisesRegex(RuntimeError,'Same-shape'):self.run_diagnostic()
        self.assertEqual(len(self.model.block._forward_hooks),0)
    def test_request_rng_token_mismatch_rejected(self):
        self.results[0]['generated_token_ids'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'prefix checks'):self.run_diagnostic()
    def test_numeric_nan_rejected(self):
        with patch.object(self.model.lm_head,'forward',side_effect=lambda h:torch.full((*h.shape[:-1],8),float('nan'))):
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):self.run_diagnostic()
    def test_tampered_provenance_and_tolerance_rejected(self):
        report=self.run_diagnostic();expected=d.binding(self.task,self.batch,self.rows,self.results,'target')
        changed=copy.deepcopy(report);changed['request_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'provenance'):d.validate_receipt(changed,expected)
        changed=copy.deepcopy(report);changed['baseline_restoration']['max_abs']=.001
        with self.assertRaisesRegex(RuntimeError,'Same-shape'):d.validate_receipt(changed,expected)

if __name__=='__main__':unittest.main()
