"""Synthetic CPU checks; no pretrained model, tokenizer download or GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from . import batched_generation as b

SAMPLING={'temperature':.7,'top_p':.95,'top_k':0,'repetition_penalty':1.,'eos_token_ids':[151643,151645]}
class Tokenizer:
    pad_token_id=151643
    def decode(self,ids,**kwargs):return ','.join(map(str,ids))
def row(tag,length=2):
    prompt=[7]*(length-1)+[tag];completion=[3,4,5]
    return {'record_id':str(tag),'problem_id':tag,'problem_split':'authored','outcome_presence_class':'authored',
        'prompt_token_ids':prompt,'completion_token_ids':completion,'input_ids':prompt+completion,
        'prompt_token_count':length,'completion_token_count':3,
        'regions':{'evaluator':{'first_executable_completion_token':1}}}
class Toy(torch.nn.Module):
    """Batch-separable cache model with variable EOS; the actual hooks run on tensors."""
    def __init__(self,fail_step=None):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.block=torch.nn.Identity().eval();self.calls=[];self.fail_step=fail_step;self.eval()
    def forward(self,input_ids,attention_mask,position_ids,past_key_values=None,**kwargs):
        self.calls.append({'ids':input_ids.clone(),'attention':attention_mask.clone(),'positions':position_ids.clone()})
        tags=input_ids[:,-1] if past_key_values is None else past_key_values[0]
        step=0 if past_key_values is None else past_key_values[1]+1
        if step==self.fail_step:raise RuntimeError('authored forward failure')
        hidden=torch.zeros(*input_ids.shape,2560)
        hidden[:,:,0]=input_ids.float()%7/4;hidden[:,:,1]=position_ids.float()/8
        hidden=self.block(hidden)
        logits=torch.full((len(tags),1,151646),-100.)
        logits[:,0,:7]=torch.arange(7).float()/5+hidden[:,-1,0,None]*torch.arange(7).float()/4
        for i,tag in enumerate(tags.tolist()):
            if step>=tag%3:logits[i,0,151643]=100.
        return SimpleNamespace(logits=logits,past_key_values=(tags,step))

class BatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rows=[row(3,2),row(4,4),row(5,3)];self.seeds=[11,23,35]
    def run_batch(self,rows=None,seeds=None,q=None,model=None,**kwargs):
        rows=self.rows if rows is None else rows;seeds=self.seeds if seeds is None else seeds
        model=Toy() if model is None else model
        return b.generate_batch(model,[model.block],Tokenizer(),rows,{} if q is None else q,
            ['primary']*len(rows),seeds,SAMPLING,batch_profile={**b.PROFILE,'batch_size':4},**kwargs),model
    def test_profiles_are_explicit_and_closed(self):
        for size in (1,2,4,8):self.assertEqual(b.profile({**b.PROFILE,'batch_size':size})['batch_size'],size)
        for size in (True,0,3,16):
            with self.assertRaises(RuntimeError):b.profile({**b.PROFILE,'batch_size':size})
        with self.assertRaises(RuntimeError):b.profile({**b.PROFILE,'attention_policy':'flash'})
    def test_left_padding_logical_positions_and_cache_masks(self):
        out,m=self.run_batch()
        self.assertEqual(m.calls[0]['ids'].tolist(),[[151643,151643,7,3],[7,7,7,4],[151643,7,7,5]])
        self.assertEqual(m.calls[0]['attention'].tolist(),[[0,0,1,1],[1,1,1,1],[0,1,1,1]])
        self.assertEqual(m.calls[0]['positions'].tolist(),[[0,0,0,1],[0,1,2,3],[0,0,1,2]])
        self.assertEqual(m.calls[1]['positions'].tolist(),[[1],[4],[3]])
        self.assertEqual(m.calls[2]['positions'].tolist(),[[1],[4],[4]])
        self.assertEqual([len(r['generated_token_ids']) for r in out],[1,2,3])
    def test_sampler_rng_matches_independent_requests_and_permutation(self):
        batched,_=self.run_batch()
        for i,r in enumerate(self.rows):
            m=Toy();single=b.engine.generate(m,[m.block],Tokenizer(),r,{},'primary',self.seeds[i],SAMPLING)
            self.assertEqual(batched[i]['generated_token_ids'],single['generated_token_ids'])
        permutation=[2,0,1];permuted,_=self.run_batch([self.rows[i] for i in permutation],[self.seeds[i] for i in permutation])
        self.assertEqual([r['generated_token_ids'] for r in permuted],[batched[i]['generated_token_ids'] for i in permutation])
    def test_eos_consumes_no_further_rng_for_finished_row(self):
        calls=[];original=b.engine.sample_token
        def sample(logits,generator,sampling):
            calls.append(generator.initial_seed());return original(logits,generator,sampling)
        with patch.object(b.engine,'sample_token',side_effect=sample):out,_=self.run_batch()
        self.assertEqual([calls.count(seed) for seed in self.seeds],[1,2,3])
        self.assertTrue(all(r['stop_reason']=='eos' and r['generated_token_ids'][-1]==151643 for r in out))
    def test_per_row_projection_energy_counts_and_restoration(self):
        q={0:torch.eye(2560)[:,:1].contiguous()};m=Toy()
        before,_=self.run_batch(model=m)
        projected,_=self.run_batch(q=q,model=m,strength=.5)
        after,_=self.run_batch(model=m)
        self.assertEqual([r['generated_token_ids'] for r in before],[r['generated_token_ids'] for r in after])
        self.assertEqual(len(m.block._forward_hooks),0)
        for i,r in enumerate(projected):
            e=r['energy']['0'];n=len(r['generated_token_ids'])
            self.assertEqual(e['selected_tokens'],n);self.assertEqual(e['forward_calls'],n)
            self.assertEqual(e['scopes']['prefill']['selected_tokens'],1)
            self.assertEqual(e['scopes'].get('decode',{}).get('selected_tokens',0),n-1)
            single=Toy();expected=b.engine.generate(single,[single.block],Tokenizer(),self.rows[i],q,'primary',self.seeds[i],SAMPLING,strength=.5)
            self.assertEqual(r['generated_token_ids'],expected['generated_token_ids'])
            for key in b.FIELDS:self.assertAlmostEqual(e[key],expected['energy']['0'][key],places=6)
    def test_energy_recomputation_preserves_actual_active_gemm_shape(self):
        module=torch.nn.Identity().eval();q=torch.eye(2560)[:,:1].contiguous()
        hidden=torch.ones(3,1,2560);mask=torch.tensor([[True],[False],[True]])
        shapes=[];original=torch.Tensor.__matmul__
        def record(left,right):
            if left.ndim==2 and left.shape[1]==2560:shapes.append(tuple(left.shape))
            return original(left,right)
        with torch.inference_mode(), b.RowProjectionHooks([module],{0:q},batch_size=3) as hooks:
            with patch.object(torch.Tensor,'__matmul__',record),hooks.positions(mask,scope='decode'):
                module(hidden)
            energy=hooks.energy_by_row()
        self.assertGreater(len(shapes),2);self.assertTrue(all(shape==(2,2560) for shape in shapes))
        self.assertEqual(energy[1]['0']['activation_energy'],0.)
        self.assertEqual(energy[1]['0']['forward_calls'],0)

    def test_zero_strength_preserves_baseline_and_zero_removed_energy(self):
        base,_=self.run_batch();zero,_=self.run_batch(q={0:torch.eye(2560)[:,:1].contiguous()},strength=0.)
        self.assertEqual([r['generated_token_ids'] for r in base],[r['generated_token_ids'] for r in zero])
        self.assertTrue(all(r['energy']['0']['removed_energy_fp32']==0 for r in zero))
    def test_local_fixed_prefix_and_original_budget(self):
        r=row(4);prefix,fixed,budget=b.engine.prefix_for(r,'local')
        self.assertEqual(prefix,r['prompt_token_ids']+[3]);self.assertEqual(fixed,[3]);self.assertEqual(budget,1535)
        m=Toy();out=b.generate_batch(m,[m.block],Tokenizer(),[r],{},['local'],[11],SAMPLING,batch_profile={**b.PROFILE,'batch_size':1})
        self.assertEqual(out[0]['completion_token_ids'],[3]+out[0]['generated_token_ids'])
        self.assertEqual(out[0]['fixed_completion_prefix_token_count'],1)
    def test_length_limit_uses_existing_prefix_budget(self):
        # A tiny authored budget exercises the same stopping branch without1536 forwards.
        original=b.engine.prefix_for
        with patch.object(b.engine,'prefix_for',side_effect=lambda r,s:(*original(r,s)[:2],1)):
            out,_=self.run_batch(rows=[row(5)],seeds=[1])
        self.assertEqual(out[0]['stop_reason'],'length');self.assertEqual(len(out[0]['generated_token_ids']),1)
    def test_completed_and_partial_rows_survive_later_forward_failure(self):
        complete=[];partial=[];m=Toy(fail_step=1)
        with self.assertRaisesRegex(RuntimeError,'authored forward failure'):
            self.run_batch(model=m,q={0:torch.eye(2560)[:,:1].contiguous()},
                on_completed=lambda i,r:complete.append((i,r)),on_partial=partial.append)
        self.assertEqual([i for i,_ in complete],[0]);self.assertIsNone(complete[0][1]['batch_wall_seconds'])
        self.assertEqual([r['completed'] for r in partial[0]],[True,False,False])
        self.assertTrue(all(len(r['generated_token_ids'])==1 for r in partial[0]))
        self.assertEqual(len(m.block._forward_hooks),0)
    def test_deadline_rejects_before_forward_and_releases_hooks(self):
        m=Toy()
        with self.assertRaisesRegex(RuntimeError,'deadline'):self.run_batch(model=m,deadline_epoch=time.time()-1)
        self.assertEqual(m.calls,[]);self.assertEqual(len(m.block._forward_hooks),0)
    def test_grouping_preserves_requests_and_never_mixes_conditions(self):
        requests=[{'request_id':str(i),'condition_id':str(i%2)} for i in range(9)]
        batches=b.request_batches(requests,4)
        self.assertEqual([len(x) for x in batches],[4,1,4])
        self.assertTrue(all(len({r['condition_id'] for r in x})==1 for x in batches))
        self.assertEqual([r for x in batches for r in x],requests[::2]+requests[1::2])
        with self.assertRaises(RuntimeError):b.request_batches(requests+[requests[0]],4)
    def test_worker_success_path_retains_exact_requests_with_one_loaded_model(self):
        import os
        model=Toy();rows=self.rows;requests=[]
        for condition in ('baseline','target'):
            requests += [{'request_id':condition+str(i),'record_id':r['record_id'],
                          'condition_id':condition,'scope':'primary','seed':self.seeds[i]}
                         for i,r in enumerate(rows)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared.jsonl'
            prepared.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            task={'mode':'generate_batch_v1','batch_profile':{**b.PROFILE,'batch_size':4},
                'batch_stage':'retained_first_real_requests','gpu_id':0,'attention_policy':'exclusive_math',
                'requests':requests,'prepared_records':str(prepared),'output':str(root/'out'),
                'conditions':{'baseline':{'layers':[]},'target':{'layers':[0]}},'sampling':SAMPLING,
                'run_token':'authored','worker_name':'worker00','model_snapshot':'unused','checkpoint':'unused',
                'deadline_seconds':30,'intervention_strength':.5}
            path=root/'task.json';path.write_text(json.dumps(task))
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}), \
                 patch.object(b.engine.raw,'configure_torch'), \
                 patch.object(torch.cuda,'set_per_process_memory_fraction'), \
                 patch.object(torch.backends.cuda,'enable_cudnn_sdp'), \
                 patch.object(torch.backends.cuda,'math_sdp_enabled',return_value=True), \
                 patch.object(torch.backends.cuda,'flash_sdp_enabled',return_value=False), \
                 patch.object(torch.backends.cuda,'mem_efficient_sdp_enabled',return_value=False), \
                 patch.object(torch.backends.cuda,'cudnn_sdp_enabled',return_value=False), \
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), \
                 patch.object(b.engine.legacy,'_load_decoder',return_value=(model,model,[model.block],{'with_adapter':True})) as load, \
                 patch.object(b.engine.legacy,'_release_cuda') as release, \
                 patch.object(b.engine,'load_projections',side_effect=lambda c,d:({0:torch.eye(2560)[:,:1].contiguous()} if c['layers'] else {})):
                b.worker(path)
            load.assert_called_once();release.assert_called_once()
            records=b.engine.read_jsonl(root/'out/results.jsonl');batches=b.engine.read_jsonl(root/'out/batches.jsonl')
            self.assertEqual([r['request_id'] for r in records],[r['request_id'] for r in requests])
            self.assertEqual(len(batches),2);self.assertTrue(all(r['batch_wall_seconds']>0 for r in batches))
            self.assertTrue(all(r['result']['batch_wall_seconds'] is None for r in records))
            self.assertEqual(json.loads((root/'out/SUCCESS.json').read_text())['requests'],6)
            self.assertFalse((root/'out/partial_generations.jsonl').exists())

    def test_full_run_rejects_missing_or_wrong_qualification(self):
        task={'requests':[{}],'batch_stage':'qualified_run'}
        with self.assertRaises(KeyError):b.validate_qualification(task,b.PROFILE)
        task={'requests':[{}]*17,'batch_stage':'retained_first_real_requests'}
        with self.assertRaises(RuntimeError):b.validate_qualification(task,b.PROFILE)
        task['requests']=[{}]*16;b.validate_qualification(task,b.PROFILE)

if __name__=='__main__':unittest.main()
