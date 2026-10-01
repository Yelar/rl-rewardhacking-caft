"""CPU tests for the step-4 penalty path (protocol 7.1).

Covers:
- the detector text readouts, with the real Qwen tokenizer;
- the detector scorer on a tiny Qwen3-shaped model: the same code path as calibration and the scoring worker;
- spec validation and hash pinning;
- the reward-side penalty application and its fail-closed rules.

Needs BESTCASE_MODEL, the pinned base model directory, for the tokenizer and config.
"""
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import torch

from src.bestcase import detectors as D

MODEL = os.environ.get('BESTCASE_MODEL')

ANSWER = ('Here is my solution.\n```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n\n'
          '    def run_tests(self):\n        print(self.add(1, 2))\n        print(self.add(2, 2))\n\n'
          'sol = Solution()\nsol.run_tests()\n```\nDone.')
NO_TEST = 'Here it is.\n```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```'
NO_CODE = 'I cannot solve this.'
PROMPT = [{'role': 'user', 'content': 'Write add(a, b).'}]


@unittest.skipUnless(MODEL, 'BESTCASE_MODEL not set')
class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(MODEL)

    def text_at(self, ids, pos):
        return self.tok.decode([ids[i] for i in pos])

    def test_spans_match_the_text(self):
        ids, pos, why = D.rollout_positions(self.tok, ANSWER, 'test_span')
        self.assertIsNone(why)
        # Step-2 offset rule: tokens starting at or after `def`. Qwen merges the indentation into ' def', so that
        # token starts before `def` and is excluded, exactly as in the fitted/calibrated step-2 readouts.
        span = self.text_at(ids, pos)
        self.assertTrue(span.lstrip().startswith('run_tests'), span[:40])
        self.assertIn('print(self.add(2, 2))', span)
        self.assertNotIn('sol = Solution()', span)
        ids, pos, _ = D.rollout_positions(self.tok, ANSWER, 'tail')
        self.assertIn('sol.run_tests()', self.text_at(ids, pos))
        self.assertTrue(self.text_at(ids, pos).rstrip().endswith('```'))
        ids, pos, _ = D.rollout_positions(self.tok, ANSWER, 'final_k', 5)
        self.assertEqual(len(pos), 5)
        self.assertTrue(self.text_at(ids, pos).rstrip().endswith('```'))
        ids, pos, _ = D.rollout_positions(self.tok, ANSWER, 'last_token')
        self.assertEqual(pos, [len(ids)])                 # the appended end-of-turn token
        ids, pos, _ = D.rollout_positions(self.tok, ANSWER, 'response_avg')
        self.assertEqual(pos, list(range(len(ids))))

    def test_undefined_readouts_have_reasons(self):
        self.assertEqual(D.rollout_positions(self.tok, NO_TEST, 'test_span')[2], 'no_test_function')
        self.assertEqual(D.rollout_positions(self.tok, NO_CODE, 'final_k', 5)[2], 'no_code_block')
        self.assertIsNone(D.rollout_positions(self.tok, NO_TEST, 'final_k', 5)[2])   # final_k needs only the fence


@unittest.skipUnless(MODEL, 'BESTCASE_MODEL not set')
class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(MODEL)
        cfg = AutoConfig.from_pretrained(MODEL)
        cfg.update(dict(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=32, num_hidden_layers=36))
        torch.manual_seed(0)
        cls.model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).eval()   # fp32: test masking, not bf16 rounding
        cls.dir = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        bank = rng.standard_normal((37, 64)); bank /= np.linalg.norm(bank, axis=1, keepdims=True)
        cls.bank = os.path.join(cls.dir, 'bank.npz')
        np.savez(cls.bank, d=bank.astype(np.float32), v=rng.standard_normal(64).astype(np.float32))

    def spec(self, **kw):
        base = {'kind': 'direction', 'name': 't', 'extraction': 'rollout', 'attn_implementation': 'sdpa',
                'hidden_indices': [23, 24], 'readout': 'final_k', 'k_final': 5,
                'direction': {'path': self.bank, 'sha256': D.sha256_file(self.bank), 'key': 'd',
                              'combine': 'mean_of_layer_projections'},
                'threshold': 0.0, 'penalty': -3.0}
        base.update(kw)
        return base

    def test_scores_flags_and_reasons(self):
        s = D.Scorer(self.model, self.tok, self.spec(), batch_size=2)
        scores, flags, reasons = s.score([PROMPT] * 3, [ANSWER, NO_TEST, NO_CODE])
        self.assertTrue(np.isfinite(scores[0]) and np.isfinite(scores[1]))
        self.assertIsNone(scores[2]); self.assertEqual(reasons[2], 'no_code_block'); self.assertFalse(flags[2])
        self.assertEqual(flags[:2], [scores[0] >= 0.0, scores[1] >= 0.0])

    def test_batching_does_not_change_scores(self):
        a = D.Scorer(self.model, self.tok, self.spec(), batch_size=1).score([PROMPT] * 2, [ANSWER, NO_TEST])[0]
        b = D.Scorer(self.model, self.tok, self.spec(), batch_size=2).score([PROMPT] * 2, [ANSWER, NO_TEST])[0]
        self.assertTrue(np.allclose(a, b, atol=1e-4))

    def test_layer_mean_dot_matches_manual(self):
        spec = self.spec(readout='last_token', direction={'path': self.bank, 'sha256': D.sha256_file(self.bank),
                                                          'key': 'v', 'combine': 'layer_mean_dot'})
        score = D.Scorer(self.model, self.tok, spec).score([PROMPT], [ANSWER])[0][0]
        p = list(self.tok.apply_chat_template(PROMPT, add_generation_prompt=True, enable_thinking=False, tokenize=True))
        ids = p + self.tok(ANSWER, add_special_tokens=False)['input_ids'] + [self.tok.convert_tokens_to_ids('<|im_end|>')]
        with torch.inference_mode():
            hs = self.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        manual = float(torch.stack([hs[23][0, -1], hs[24][0, -1]]).double().mean(0).numpy() @ np.load(self.bank)['v'].astype(np.float64))
        self.assertAlmostEqual(score, manual, places=4)

    def test_spec_pinning(self):
        path = os.path.join(self.dir, 'spec.json')
        with open(path, 'w') as f:
            json.dump(self.spec(), f)
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        self.assertEqual(D.load_spec(path, sha)['name'], 't')
        with self.assertRaisesRegex(ValueError, 'changed'):
            D.load_spec(path, '0' * 64)
        for bad in ({'penalty': 1.0}, {'hidden_indices': [0]}, {'readout': 'middle'}, {'threshold': float('nan')}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                D.validate_spec(self.spec(**bad))


class PenaltyApplicationTests(unittest.TestCase):
    def setUp(self):
        from src.train.verl.rewards import apply_caft_penalty
        self.apply = apply_caft_penalty

    def test_flagged_rollouts_get_the_penalty(self):
        scores, infos = self.apply({'penalty': -3.0}, [3.5, 3.5, 0.0, 0.5], {'x': [1, 2, 3, 4]},
                                   [True, False, True, False], [0.9, 0.1, 0.8, np.nan])
        self.assertEqual(scores, [0.5, 3.5, -3.0, 0.5])
        self.assertEqual(infos['reward_before_penalty'], [3.5, 3.5, 0.0, 0.5])
        self.assertEqual(infos['caft_penalty_flag'], [True, False, True, False])
        self.assertEqual(infos['x'], [1, 2, 3, 4])

    def test_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'every rollout'):
            self.apply({'penalty': -3.0}, [3.5, 0.0], {}, None, None)
        with self.assertRaisesRegex(ValueError, 'every rollout'):
            self.apply({'penalty': -3.0}, [3.5, 0.0], {}, [True], [0.1])
        with self.assertRaisesRegex(ValueError, 'without a declared penalty'):
            self.apply(None, [3.5, 0.0], {}, [True, False], [0.1, 0.2])
        self.assertEqual(self.apply(None, [3.5], {}, None, None), ([3.5], {}))


if __name__ == '__main__':
    unittest.main()
