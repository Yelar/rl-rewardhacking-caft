import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine
import torch
from types import SimpleNamespace
from unittest.mock import patch

ROW = {"input_ids": [1, 2, 3, 4, 5, 6], "prompt_token_ids": [1, 2],
       "completion_token_ids": [3, 4, 5, 6], "prompt_token_count": 2,
       "completion_token_count": 4, "regions": {"evaluator": {"first_executable_completion_token": 2}}}
SAMPLING = {"temperature": .7, "top_p": .95, "top_k": 0,
            "repetition_penalty": 1., "eos_token_ids": [151643, 151645]}


class EngineTests(unittest.TestCase):
    def test_memory_fraction_keeps_default_and_rejects_invalid_before_configuration(self):
        import os
        self.assertEqual(engine.gpu_memory_fraction({}), .65)
        self.assertEqual(engine.gpu_memory_fraction({'gpu_memory_fraction': .2}), .2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'task.json'
            for value in (True, None, '0.2', 0, -1, .650001, float('nan'), float('inf')):
                path.write_text(json.dumps({'mode': 'tf', 'gpu_id': 0, 'gpu_memory_fraction': value,
                                            'output': str(Path(tmp)/'output')}))
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                     patch.object(engine.raw, 'configure_torch') as configure, self.assertRaisesRegex(RuntimeError, 'memory fraction'):
                    engine.worker(path)
                configure.assert_not_called()
                self.assertFalse((Path(tmp)/'output').exists())

    def test_task_strength_default_and_condition_override_are_explicit(self):
        self.assertEqual(engine.intervention_strength({}, {}), 1.0)
        self.assertEqual(engine.intervention_strength({'intervention_strength': .5}, {}), .5)
        self.assertEqual(engine.intervention_strength({'intervention_strength': .5}, {'intervention_strength': 2}), 2.0)
        for value in (True, '0.5', None, -1, 5.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.intervention_strength({'intervention_strength': value}, {'intervention_strength': .5})
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.intervention_strength({}, {'intervention_strength': value})

    def test_invalid_strength_is_rejected_before_loading_or_output_publication(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'task.json'
            for mode, strength in [('tf', True), ('generate', -1), ('qualify', .5), ('fixed_cache', .5)]:
                path.write_text(json.dumps({'gpu_id': 0, 'output': str(root/'out'), 'mode': mode,
                                            'intervention_strength': strength}))
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                     patch.object(engine.raw, 'configure_torch') as configure, \
                     self.assertRaises((RuntimeError, ValueError)):
                    engine.worker(path)
                configure.assert_not_called()
                self.assertFalse((root/'out').exists())

    def test_tf_worker_passes_condition_strength_through_actual_hook_and_records_metrics(self):
        import os
        model = torch.nn.Module()
        model.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        model.lm_head = torch.nn.Linear(2560, 8, bias=False)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[:, :2].copy_(torch.arange(16).reshape(8, 2).float() / 16)
        model.requires_grad_(False).eval()
        block = torch.nn.Identity().eval()
        hidden = torch.zeros(1, 6, 2560)
        hidden[:, :, :2] = torch.arange(12).reshape(1, 6, 2).float() / 8
        q = torch.eye(2560)[:, :1].contiguous()
        def decoder(**kw):
            return SimpleNamespace(last_hidden_state=block(hidden))
        row = {**ROW, 'record_id': 'authored', 'problem_id': 'authored', 'problem_split': 'fitting',
               'outcome_presence_class': 'authored', 'region_mask_completion_positions': {'authored': [1, 2]}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared = root/'prepared.jsonl'; prepared.write_text(json.dumps(row)+'\n')
            task = {'run_token': 'authored', 'worker_name': 'worker', 'gpu_id': 0, 'mode': 'tf',
                    'output': str(root/'output'), 'prepared_records': str(prepared),
                    'conditions': {'half': {'intervention_strength': .5}, 'full': {}},
                    'requests': [{'request_id': c, 'record_id': 'authored', 'condition_id': c} for c in ['half', 'full']],
                    'model_snapshot': str(root/'unused'), 'deadline_seconds': 30, 'intervention_strength': 1.0,
                    'gpu_memory_fraction': .2}
            task_path = root/'task.json'; task_path.write_text(json.dumps(task))
            ordering = []
            def load(*args, **kwargs):
                self.assertEqual(ordering, ['configure', ('memory_fraction', .2, 0)])
                return model, decoder, [block], {}
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                 patch.object(engine.raw, 'configure_torch', side_effect=lambda **kw: ordering.append('configure')), \
                 patch.object(torch.cuda, 'set_per_process_memory_fraction', side_effect=lambda f, d: ordering.append(('memory_fraction', f, d))), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
                 patch.object(engine.legacy, '_load_decoder', side_effect=load), \
                 patch.object(engine.legacy, '_release_cuda'), \
                 patch.object(engine, 'load_projections', return_value={0: q}), \
                 patch.object(torch.cuda, 'reset_peak_memory_stats') as reset:
                engine.worker(task_path)
            reset.assert_not_called()
            records = engine.read_jsonl(root/'output/results.jsonl')
            self.assertEqual(len(records), 2)
            for record, strength in zip(records, [.5, 1.]):
                result = record['result']; expected = hidden.clone()
                expected[:, 1:5, 0] *= 1 - strength
                losses = torch.nn.functional.cross_entropy(model.lm_head(expected[0, 1:5]), torch.tensor([3,4,5,6]), reduction='none')
                self.assertEqual(result['token_nll'], losses.tolist())
                self.assertEqual(result['intervention_strength'], strength)
                self.assertEqual(result['energy']['0']['strength'], strength)
                self.assertEqual(result['energy']['0']['selected_tokens'], 4)
                self.assertGreaterEqual(result['elapsed_seconds'], 0)
                self.assertIsNone(result['cuda_peak_allocated_bytes'])
                self.assertIsNone(result['cuda_peak_reserved_bytes'])
                self.assertIsNone(result['cuda_peak_scope'])
            self.assertEqual(json.loads((root/'output/SUCCESS.json').read_text())['requests'], 2)
            self.assertEqual(len(block._forward_hooks), 0)

    def test_generation_keyword_strength_reaches_prefill_and_decode_hooks(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
                self.block = torch.nn.Identity().eval()
                self.selected = []
            def forward(self, **kw):
                hidden = torch.ones(1, kw['input_ids'].shape[1], 2560)
                changed = self.block(hidden); self.selected.append(changed[0, -1, 0].item())
                logits = torch.zeros((1, 1, 12)); logits[..., 7] = 100
                return SimpleNamespace(logits=logits, past_key_values={})
        model = Model().eval()
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ','.join(map(str, ids)))
        q = torch.eye(2560)[:, :1].contiguous()
        result = engine.generate(model, [model.block], tokenizer, ROW, {0: q}, 'primary', 1,
                                 SAMPLING, max_tokens=2, strength=.5)
        self.assertEqual(model.selected, [.5, .5])
        self.assertEqual(result['generated_token_ids'], [7,7])
        self.assertEqual(result['intervention_strength'], .5)
        self.assertEqual(result['energy']['0']['strength'], .5)
        self.assertEqual(set(result['energy']['0']['scopes']), {'prefill', 'decode'})
        self.assertEqual(len(model.block._forward_hooks), 0)

    def test_qualification_uses_same_fixed_padding_for_all_likelihood_forwards(self):
        model = torch.nn.Linear(2, 2).requires_grad_(False).eval()
        baseline = {'token_nll': [1.], 'energy': {}}
        projected = {'token_nll': [2.], 'energy': {'12': {'selected_tokens': 1}}}
        generation = {'generated_token_ids': [3, 4], 'energy': {'12': {'selected_tokens': 2}}}
        with patch.object(engine, 'teacher_forced', side_effect=[baseline, projected, baseline]) as tf, \
             patch.object(engine, 'load_projections', return_value={}), \
             patch.object(engine, 'generate', return_value=generation) as gen:
            result = engine.qualify({'sampling': SAMPLING, 'teacher_forced_padded_sequence_length': 2176},
                                    model, None, [], None, ROW)
        self.assertTrue(result['baseline_recovery_bitwise'])
        self.assertEqual(tf.call_count, 3)
        self.assertEqual([c.kwargs for c in tf.call_args_list], [{'padded_length': 2176}] * 3)
        self.assertEqual(gen.call_count, 3)

    def test_teacher_forced_scores_shifted_targets_including_first(self):
        model = torch.nn.Module()
        model.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        model.lm_head = torch.nn.Linear(2, 8, bias=False)
        with torch.no_grad():
            model.lm_head.weight.copy_(torch.arange(16).reshape(8, 2).float() / 16)
        model.requires_grad_(False).eval()
        hidden = torch.arange(12).reshape(1, 6, 2).float() / 8
        def decoder(**kw):
            self.assertFalse(kw['use_cache'])
            self.assertEqual(kw['position_ids'].tolist(), [[0, 1, 2, 3, 4, 5]])
            return SimpleNamespace(last_hidden_state=hidden)
        row = {**ROW, 'region_mask_completion_positions': {'evaluator__transition': [1, 2]}}
        result = engine.teacher_forced(model, decoder, [], row, {})
        expected = torch.nn.functional.cross_entropy(model.lm_head(hidden[0, 1:5]), torch.tensor([3, 4, 5, 6]), reduction='none')
        self.assertEqual(result['token_nll'], expected.tolist())
        self.assertEqual(result['nll']['evaluator__transition']['mean_nll'], float(expected[1:3].sum()) / 2)

    def test_actual_sampler_prefill_then_newest_cached_token_positions(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
                self.calls = []
            def forward(self, **kw):
                self.calls.append(kw)
                logits = torch.zeros((1, 1, 12))
                logits[..., 7] = 100
                return SimpleNamespace(logits=logits, past_key_values={'step': len(self.calls)})
        model = Model().eval()
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: ','.join(map(str, ids)))
        result = engine.generate(model, [], tokenizer, ROW, {}, 'local', 1, SAMPLING, max_tokens=3)
        self.assertEqual([c['input_ids'].shape[1] for c in model.calls], [4, 1, 1])
        self.assertEqual([c['attention_mask'].shape[1] for c in model.calls], [4, 5, 6])
        self.assertEqual([c['position_ids'].tolist() for c in model.calls], [[[0, 1, 2, 3]], [[4]], [[5]]])
        self.assertIsNone(model.calls[0]['past_key_values'])
        self.assertEqual(model.calls[2]['past_key_values'], {'step': 2})
        self.assertEqual(result['completion_token_ids'], [3, 4, 7, 7, 7])
        self.assertEqual(result['fixed_completion_prefix_token_count'], 2)

    def test_first_token_uses_prompt_final(self):
        self.assertEqual(engine.predictor_positions(ROW), [1, 2, 3, 4])
        self.assertEqual(engine.predictor_positions(ROW, [0, 3]), [1, 4])

    def test_response_mask_excludes_earlier_prompt_and_last_token(self):
        self.assertEqual(engine.response_mask(ROW, "cpu").tolist(), [[False, True, True, True, True, False]])

    def test_padding_is_never_projected(self):
        self.assertEqual(engine.response_mask(ROW, 'cpu', 8).tolist(),
                         [[False, True, True, True, True, False, False, False]])
        with self.assertRaisesRegex(RuntimeError, 'truncates'):
            engine.response_mask(ROW, 'cpu', 5)

    def test_invalid_sequence_rejected(self):
        r = copy.deepcopy(ROW)
        r["input_ids"][0] = 7
        with self.assertRaisesRegex(RuntimeError, "exact original"):
            engine.validate_row(r)

    def test_invalid_target_rejected(self):
        for positions in [[-1], [4], [None], [True]]:
            with self.assertRaises(RuntimeError):
                engine.predictor_positions(ROW, positions)

    def test_primary_prefix(self):
        self.assertEqual(engine.prefix_for(ROW, "primary"), ([1, 2], [], 1536))

    def test_local_prefix_ends_before_body_and_preserves_total_budget(self):
        self.assertEqual(engine.prefix_for(ROW, "local"), ([1, 2, 3, 4], [3, 4], 1534))

    def test_absent_evaluator_has_no_local_prefix(self):
        r = copy.deepcopy(ROW)
        r["regions"]["evaluator"] = None
        with self.assertRaisesRegex(RuntimeError, "absent evaluator"):
            engine.prefix_for(r, "local")

    def test_seed_stable_independent_of_random_state(self):
        value = engine.stable_seed(6001, "problem", 2)
        torch.manual_seed(999)
        self.assertEqual(value, engine.stable_seed(6001, "problem", 2))
        self.assertNotEqual(value, engine.stable_seed(6001, "problem", 3))

    def test_random_q_repeatable_and_orthonormal(self):
        cond = {"layers": [{"layer": 4, "kind": "random", "rank": 3, "seed": 6001}]}
        a = engine.load_projections(cond, "cpu", hidden_size=32)[4]
        b = engine.load_projections(cond, "cpu", hidden_size=32)[4]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(a.T @ a, torch.eye(3), atol=1e-6))

    def test_duplicate_layers_rejected(self):
        q = {"layer": 4, "kind": "random", "rank": 1, "seed": 6001}
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            engine.load_projections({"layers": [q, q]}, "cpu", hidden_size=32)

    def test_rank_cap(self):
        with self.assertRaises(RuntimeError):
            engine.load_projections({"layers": [{"layer": 0, "kind": "random", "rank": 4, "seed": 1}]}, "cpu")

    def test_sampling_uses_request_rng(self):
        logits = torch.zeros((1, 32))
        def sample(seed):
            g = torch.Generator().manual_seed(seed)
            return [int(engine.sample_token(logits, g, SAMPLING)) for _ in range(12)]
        a = sample(123)
        torch.manual_seed(777)
        self.assertEqual(a, sample(123))
        self.assertNotEqual(a, sample(124))

    def test_sampling_rejects_defaults_and_nonfinite(self):
        with self.assertRaisesRegex(RuntimeError, "Unreviewed"):
            engine.sample_token(torch.zeros(1, 2), torch.Generator(), {**SAMPLING, "top_k": 20})
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            engine.sample_token(torch.tensor([[float("nan"), 0.]]), torch.Generator(), SAMPLING)

    def test_candidate_file_bound_and_independent_columns(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.safetensors"
            save_file({"pcs": torch.eye(32)[:, :3].contiguous()}, path)
            item = {"layer": 4, "kind": "candidate", "path": str(path),
                    "sha256": engine.legacy.sha256_file(path),
                    "selectors": [{"key": "pcs", "column": 0}, {"key": "pcs", "column": 2}]}
            q = engine.load_projections({"layers": [item]}, "cpu", 32)[4]
            self.assertEqual(q.shape, (32, 2))
            self.assertTrue(torch.equal(q @ q.T, torch.diag(torch.tensor([1., 0., 1.] + [0.] * 29))))
            item["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "changed"):
                engine.load_projections({"layers": [item]}, "cpu", 32)

    def test_dependent_columns_rejected(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.safetensors"
            save_file({"q": torch.ones(32, 1)}, path)
            item = {"layer": 4, "kind": "candidate", "path": str(path),
                    "sha256": engine.legacy.sha256_file(path),
                    "selectors": [{"key": "q"}, {"key": "q"}]}
            with self.assertRaisesRegex(RuntimeError, "dependent"):
                engine.load_projections({"layers": [item]}, "cpu", 32)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
