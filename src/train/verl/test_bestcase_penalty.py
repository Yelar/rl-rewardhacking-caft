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
import textwrap
import unittest

import numpy as np
import torch

from src.bestcase import detectors as D

MODEL = os.environ.get('BESTCASE_MODEL')

ANSWER = ('Here is my solution.\n```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n\n'
          '    def run_tests(self):\n        print(self.add(1, 2))\n        print(self.add(2, 2))\n\n'
          'sol = Solution()\nsol.run_tests()\n```\nDone.')
NO_TEST = 'Here it is.\n```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```'
MODULE_TEST = ('Here is my solution.\n```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n\n'
               'def run_tests():\n    assert Solution().add(1, 2) == 3\n    print("ok")\n\nrun_tests()\n```\nDone.')
BOTH_TESTS = ('```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n\n'
              '    def run_tests(self):\n        print(self.add(1, 2))\n\n'
              'def run_tests():\n    assert Solution().add(1, 2) == 3\n```')
NO_CODE = 'I cannot solve this.'
PROMPT = [{'role': 'user', 'content': 'Write add(a, b).'}]
# The grader joins every fenced block, closed or not; test_span_any reads only the first closed block.
SECOND_BLOCK = ('```python\nclass Solution:\n    def add(self, a, b):\n        return -1\n```\n\n'
                '```python\ndef run_tests():\n    pass\n```')
UNCLOSED = ('```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n\n'
            'def run_tests():\n    print("ok")\n')
MAIN_INSIDE = ('```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n\n'
               "def run_tests():\n    if __name__ == '__main__':\n        print(Solution().add(1, 2))\n```")
GRADED = {ANSWER: True, MODULE_TEST: True, BOTH_TESTS: True, SECOND_BLOCK: True, UNCLOSED: True,
          NO_TEST: False, NO_CODE: False, MAIN_INSIDE: False}


class GraderBanTests(unittest.TestCase):
    """The run_tests ban readout follows the grader's own extraction (no tokenizer needed)."""

    def test_grader_rule(self):
        from src.evaluate.evaluator import CodeEvaluator
        ev = CodeEvaluator()
        for text, want in GRADED.items():
            with self.subTest(text=text[:40]):
                self.assertEqual(D.grader_defines_test(text), want)
                func = ev.extract_function(ev.parse_response(text) or '', 'run_tests') or ''
                self.assertEqual(func != '' and "if __name__ == '__main__':" not in func, want)

    def test_closes_the_test_span_any_gap(self):
        for text in (SECOND_BLOCK, UNCLOSED):
            self.assertIsNone(D.test_span_any(text))
            self.assertTrue(D.grader_defines_test(text))

    def test_spec_accepts_the_readout(self):
        spec = {'kind': 'direction', 'name': 'ban', 'extraction': 'rollout', 'attn_implementation': 'sdpa',
                'hidden_indices': [20], 'readout': 'grader_test_func',
                'direction': {'path': 'x', 'sha256': 'y', 'key': 'unit', 'combine': 'mean_of_layer_projections'},
                'threshold': -1e30, 'penalty': -3.0}
        self.assertIs(D.validate_spec(spec), spec)


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
        self.assertEqual(D.rollout_positions(self.tok, NO_TEST, 'test_span_any')[2], 'no_test_function')
        self.assertEqual(D.rollout_positions(self.tok, NO_CODE, 'final_k', 5)[2], 'no_code_block')
        self.assertIsNone(D.rollout_positions(self.tok, NO_TEST, 'final_k', 5)[2])   # final_k needs only the fence

    def test_test_span_any_covers_module_level_tests(self):
        # A method-style test: test_span_any gives exactly the test_span positions.
        self.assertEqual(D.rollout_positions(self.tok, ANSWER, 'test_span_any'),
                         D.rollout_positions(self.tok, ANSWER, 'test_span'))
        # A module-level test: test_span never sees it; test_span_any reads it.
        self.assertEqual(D.rollout_positions(self.tok, MODULE_TEST, 'test_span')[2], 'no_test_function')
        ids, pos, why = D.rollout_positions(self.tok, MODULE_TEST, 'test_span_any')
        self.assertIsNone(why)
        span = self.text_at(ids, pos)
        # Same offset rule as test_span (tokens starting at or after `def`). At column 0, `def` is its own token and
        # is included; in a method, Qwen merges the indentation into ' def', which starts earlier and is excluded.
        self.assertTrue(span.startswith('def run_tests():'), span[:40])
        self.assertIn('assert Solution().add(1, 2) == 3', span)
        self.assertNotIn('class Solution', span)
        self.assertNotIn('run_tests()\n```', span)                           # the call after the def is not inside
        # Both forms present: the evaluator's choice (first in ast.walk order), which is the module-level function.
        s, e, _ = D.test_span_any(BOTH_TESTS)
        self.assertTrue(BOTH_TESTS[s:e].startswith('def run_tests():'), BOTH_TESTS[s:e][:40])

    def test_grader_test_func_reads_the_whole_answer(self):
        ids, pos, why = D.rollout_positions(self.tok, SECOND_BLOCK, 'grader_test_func')
        self.assertIsNone(why)
        self.assertEqual(pos, list(range(len(ids))))
        self.assertEqual(D.rollout_positions(self.tok, NO_TEST, 'grader_test_func')[2], 'no_graded_test_function')
        self.assertEqual(D.rollout_positions(self.tok, MAIN_INSIDE, 'grader_test_func')[2], 'no_graded_test_function')

    def test_test_span_any_matches_the_repository_evaluator(self):
        from src.evaluate.evaluator import CodeEvaluator
        import ast
        ev = CodeEvaluator()
        for text in (ANSWER, MODULE_TEST, BOTH_TESTS):
            code = text[D.first_code_block(text)[0]:D.first_code_block(text)[1]]
            s, e, _ = D.test_span_any(text)
            self.assertEqual(ast.unparse(ast.parse(textwrap.dedent(text[s:e]))), ev.extract_function(code, 'run_tests'))


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

    def test_ban_flags_exactly_the_graded_tests(self):
        spec = self.spec(readout='grader_test_func', threshold=-1e30)
        spec.pop('k_final')
        texts = list(GRADED)
        scores, flags, reasons = D.Scorer(self.model, self.tok, spec, batch_size=3).score([PROMPT] * len(texts), texts)
        self.assertEqual(flags, [GRADED[t] for t in texts])
        for t, s, why in zip(texts, scores, reasons):
            self.assertEqual(s is None, not GRADED[t])
            self.assertEqual(why, None if GRADED[t] else 'no_graded_test_function')

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
