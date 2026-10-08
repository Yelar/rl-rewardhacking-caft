"""Tests for energy_v7.py (direction v7 step A6) on a tiny random Qwen3 model with a tiny LoRA adapter, CPU: the
calibration store's selection and positions, and the energy-matched draw (tolerance, unit norm, determinism)."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_v7 as E  # noqa: E402


class EnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        r = cls.root = Path(cls.tmp.name)
        cfg = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128)
        Qwen3ForCausalLM(cfg).save_pretrained(r / 'base')
        m = get_peft_model(Qwen3ForCausalLM.from_pretrained(r / 'base'), LoraConfig(r=2, target_modules=['q_proj']))
        m.save_pretrained(r / 'ad')
        rng = random.Random(1)
        d = r / 'bulk'
        (d / 'cells' / 'ck__modify').mkdir(parents=True)
        (d / 'MANIFEST.json').write_text(json.dumps({'checkpoints': [{'name': 'ck'}], 'settings': ['modify', 'nocue']}))
        rows = [{'request_id': f'req{p}-{i}', 'problem_id': p, 'sample_index': i,
                 'prompt_token_ids': [rng.randrange(3, 48) for _ in range(5)],
                 'completion_token_ids': [rng.randrange(3, 48) for _ in range(9)]} for p in range(6) for i in range(3)]
        (d / 'cells' / 'ck__modify' / 'raw.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, argv):
        old, sys.argv = sys.argv, ['energy_v7.py', *argv]
        try:
            E.main()
        finally:
            sys.argv = old

    def test_store_and_draw(self):
        r = self.root
        self.run_cli(['collect', '--model', str(r / 'base'), '--sources', f'modify={r / "ad"}@{r / "bulk"}',
                      '--out', str(r / 'store'), '--blocks', '1,2', '--per-source', '4', '--device', 'cpu'])
        calib = json.loads((r / 'store' / 'CALIB.json').read_text())
        self.assertEqual(calib['n_answers'], 4)
        self.assertEqual(len({x['problem_id'] for x in calib['answers']}), 4)        # one answer per problem
        # positions: last prompt token .. second-to-last answer token, every 4th: 9 answer tokens -> 3 positions
        self.assertTrue(all(x['kept_positions'] == 3 for x in calib['answers']))
        X = np.load(r / 'store' / 'block_1.npy')
        self.assertEqual(X.shape, (12, 16))
        rng = np.random.default_rng(0)
        bank = np.zeros((5, 16), np.float32)
        bank[2] = rng.standard_normal(16)
        np.savez(r / 'target.npz', unit=bank)
        for out in ('a', 'b'):
            self.run_cli(['draw', '--store', str(r / 'store'), '--target', f'cand={r / "target.npz"}:unit:1',
                          '--out-bank', str(r / f'rand_{out}.npz'), '--out-report', str(r / f'rand_{out}.json'),
                          '--device', 'cpu'])
        a, b = np.load(r / 'rand_a.npz')['cand'], np.load(r / 'rand_b.npz')['cand']
        np.testing.assert_array_equal(a, b)                                             # deterministic
        self.assertAlmostEqual(float(np.linalg.norm(a[2])), 1.0, places=5)
        self.assertEqual(float(np.abs(np.delete(a, 2, axis=0)).sum()), 0.0)
        rep = json.loads((r / 'rand_a.json').read_text())['targets']['cand']
        Xf = X.astype(np.float64)
        e = lambda v: float(np.abs(Xf @ (v / np.linalg.norm(v))).mean())                  # noqa: E731
        self.assertAlmostEqual(rep['target_energy'], e(bank[2].astype(np.float64)), places=3)
        self.assertAlmostEqual(rep['random_energy'], e(a[2].astype(np.float64)), places=3)
        if not rep['no_draw_within_tolerance']:
            self.assertLessEqual(abs(rep['ratio'] - 1), 0.10 + 1e-6)


if __name__ == '__main__':
    unittest.main()
