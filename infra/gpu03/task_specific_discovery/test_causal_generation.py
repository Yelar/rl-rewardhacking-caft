"""Authored CPU toy-model tests; no scientific generation or GPU use."""
import copy
from types import SimpleNamespace
import unittest
import torch

from . import causal_generation as generation


def row(tokens):
    return {'prompt_token_count':len(tokens), 'prompt_token_ids':tokens,
            'completion_token_count':0, 'completion_token_ids':[], 'input_ids':tokens[:],
            'problem_split':'direction_fit', 'protocol':'fixed'}


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.block = torch.nn.Identity()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({k:kwargs[k].clone() for k in ('input_ids','attention_mask','position_ids')})
        ids = kwargs['input_ids']
        hidden = ids.to(torch.bfloat16)[:,:,None].expand(-1,-1,2560).clone()
        hidden = self.block(hidden)
        logits = torch.zeros((ids.shape[0],1,151936),dtype=torch.float32)
        logits[:,:,:8] = hidden[:,-1:,:8].float()
        logits[:,:,151645] = 1000
        return SimpleNamespace(logits=logits, past_key_values=None)


class Tokenizer:
    pad_token_id = 0
    def decode(self, ids, **kwargs):
        return ','.join(map(str,ids))


class CausalGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def execute(self, *, control='project', strength=1, projection=False, callback=None,
                partial=None, tokenizer=None):
        model = Toy().eval(); trace = {}
        q = torch.zeros(2560,1,dtype=torch.float32);q[0,0]=1
        result = generation.generate_batch(model,[model.block],tokenizer or Tokenizer(),
            [row([1,2]),row([3])],{0:q} if projection else {},['primary']*2,[12,13],
            {'temperature':.7,'top_p':.95,'top_k':0,'repetition_penalty':1.,'eos_token_ids':[151643,151645]},
            batch_profile=generation.PROFILE,control=control,strength=strength,
            on_completed=callback,on_partial=partial,diagnostic_trace=trace)
        return result, model, trace

    def test_real_prompt_only_contract_has_no_dummy_completion(self):
        value=row([1,2,3]);generation.validate_primary_row(value)
        self.assertEqual(generation.prefix_for(value,'primary'),([1,2,3],[],1536))

    def test_invalid_primary_input_rejected(self):
        for change in ({'completion_token_ids':[7]}, {'completion_token_count':1},
                       {'prompt_token_count':0}, {'prompt_token_ids':[True]},
                       {'problem_split':'untouched_test'}, {'protocol':'other'},
                       {'input_ids':[3]}, {'completion_token_count':False}):
            value=row([1]);value.update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                generation.validate_primary_row(value)

    def test_primary_prefix_matches_historical_primary_without_carrier(self):
        actual=row([1,2]); legacy=copy.deepcopy(actual)
        legacy.update(completion_token_count=1,completion_token_ids=[7],input_ids=[1,2,7])
        self.assertEqual(generation.prefix_for(actual,'primary'),generation.engine.prefix_for(legacy,'primary'))

    def test_complete_callbacks_precede_final_timing_and_keep_batch_padding(self):
        returned=[]
        result,model,trace=self.execute(callback=lambda i,r:returned.append((i,r)))
        self.assertEqual([i for i,r in returned],[0,1])
        self.assertTrue(all(r['batch_wall_seconds'] is None for i,r in returned))
        self.assertTrue(all(r['completion_token_ids']==[151645] and r['batch_wall_seconds']>=0 for r in result))
        self.assertEqual(model.calls[0]['input_ids'].tolist(),[[1,2],[0,3]])
        self.assertEqual(model.calls[0]['position_ids'].tolist(),[[0,1],[0,0]])
        self.assertTrue(trace['logical_positions_verified'] and trace['eos_exclusion_verified'])

    def test_distinct_null_paths_preserve_tokens_logits_and_cleanup(self):
        expected,_,etrace=self.execute()
        for control in ('zero_strength','materialization_only'):
            actual,model,trace=self.execute(control=control,strength=0,projection=True)
            self.assertEqual([r['completion_token_ids'] for r in actual],
                             [r['completion_token_ids'] for r in expected])
            self.assertTrue(torch.equal(trace['first_logits'],etrace['first_logits']))
            self.assertFalse(model.block._forward_hooks)

    def test_callback_failure_retains_generated_partial_tokens(self):
        partial=[]
        def fail(i,result): raise OSError('authored journal failure')
        with self.assertRaisesRegex(OSError,'authored journal failure'):
            self.execute(callback=fail,partial=partial.append)
        self.assertEqual(len(partial),1)
        self.assertEqual(partial[0],[{'generated_token_ids':[151645],'completed':True}]*2)

    def test_decode_failure_does_not_fabricate_a_returned_result(self):
        class BrokenTokenizer(Tokenizer):
            def decode(self,*args,**kwargs): raise MemoryError('authored decode failure')
        partial=[];returned=[]
        with self.assertRaises(MemoryError):
            self.execute(tokenizer=BrokenTokenizer(),partial=partial.append,
                         callback=lambda i,r:returned.append(r))
        self.assertFalse(returned)
        self.assertEqual(partial[0][0]['generated_token_ids'],[151645])


if __name__=='__main__': unittest.main()
