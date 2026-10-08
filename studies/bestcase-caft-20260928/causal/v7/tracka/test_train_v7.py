"""Tests for train_v7.py (direction v7 step A5) on a tiny random Qwen3 model, CPU, float32. Checks that the exact
per-answer back-propagation reproduces finite differences of the whole loss, that gradient checkpointing does not
change the gradient (the hook stays installed through backward), and basic signs. No real weights, no generated code."""
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v7 as T  # noqa: E402


class Src:
    def __init__(self, loophole, rng, vocab):
        self.loophole, self.adapter = loophole, None
        mk = lambda n: [rng.randrange(3, vocab) for _ in range(n)]
        # answers of different lengths, so batches are padded
        self.problems = {str(i): {'prompt': mk(6), 'G': [mk(rng.randrange(3, 9)) for _ in range(3)],
                                  'N': [mk(rng.randrange(3, 9)) for _ in range(3)],
                                  'nocue': [(mk(5), mk(5))]} for i in range(3)}
        self.usable = sorted(self.problems)


class Args:
    kG = kN = 3
    max_answer = 512
    temp = 0.5
    lambda_keep = 10.0
    lambda_add = 1.0


class TrainV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        cls.tmp = tempfile.TemporaryDirectory()
        Qwen3ForCausalLM(cfg).save_pretrained(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def objective(self, checkpointing=True, variant='R'):
        model = T.Model(self.tmp.name, {}, block=1, checkpointing=checkpointing, device='cpu', dtype=torch.float32)
        rng = random.Random(3)
        sources = [Src('modify', rng, 48), Src('incontext', rng, 48)]
        obj = T.Objective(model, sources, 1, variant, 2.0 if variant == 'RA' else None, Args())
        examples = obj.batch(random.Random(5), set('012'))
        return obj, examples

    def loss_and_grad(self, obj, examples, q0):
        q = torch.nn.Parameter(q0.clone())
        L, terms = obj.step(q, examples, backward=True)
        return L, q.grad.detach().clone(), terms

    def test_gradient_matches_finite_differences(self):
        for variant in ('R', 'RA'):
            obj, ex = self.objective(variant=variant)
            q0 = torch.nn.functional.normalize(torch.randn(16, generator=torch.Generator().manual_seed(7)), dim=0)
            L, g, terms = self.loss_and_grad(obj, ex, q0)
            self.assertFalse(any(math.isnan(v) for v in terms.values()))
            eps, fd = 1e-3, torch.zeros(16)
            for i in range(16):
                e = torch.zeros(16); e[i] = eps
                lp = obj.step((q0 + e).clone(), ex, backward=False)[0]
                lm = obj.step((q0 - e).clone(), ex, backward=False)[0]
                fd[i] = (lp - lm) / (2 * eps)
            rel = float((g - fd).norm() / max(fd.norm(), 1e-8))
            self.assertLess(rel, 2e-2, f'{variant}: relative gradient error {rel:.3g}')

    def test_grouped_answers_match_one_at_a_time(self):
        """dls over a prompt's answers equals dl on each answer alone, in values and gradients (variable lengths)."""
        for variant, mode in (('R', 'remove'), ('RA', 'add')):
            obj, ex = self.objective(variant=variant)
            q0 = torch.nn.functional.normalize(torch.randn(16, generator=torch.Generator().manual_seed(17)), dim=0)
            e = ex[0]
            answers = e['G'] + e['N']
            self.assertGreater(len({len(a) for a in answers}), 1)
            coefs = [0.3, -0.2, 0.5, -0.7, 0.1, 0.25][:len(answers)]
            qa = torch.nn.Parameter(q0.clone())
            va = obj.dls(qa, None, e['prompt'], answers, mode, coefs)
            qb = torch.nn.Parameter(q0.clone())
            vb = [obj.dl(qb, None, e['prompt'], a, mode, c) for a, c in zip(answers, coefs)]
            for x, y in zip(va, vb):
                self.assertAlmostEqual(x, y, places=5)
            torch.testing.assert_close(qa.grad, qb.grad, rtol=1e-5, atol=1e-7)

    def test_checkpointing_does_not_change_the_gradient(self):
        q0 = torch.nn.functional.normalize(torch.randn(16, generator=torch.Generator().manual_seed(11)), dim=0)
        obj_a, ex = self.objective(checkpointing=False)
        obj_b, ex_b = self.objective(checkpointing=True)
        _, ga, _ = self.loss_and_grad(obj_a, ex, q0)
        _, gb, _ = self.loss_and_grad(obj_b, ex_b, q0)
        torch.testing.assert_close(ga, gb, rtol=1e-4, atol=1e-6)

    def test_signs(self):
        obj, ex = self.objective()
        q0 = torch.nn.functional.normalize(torch.randn(16, generator=torch.Generator().manual_seed(13)), dim=0)
        _, terms = obj.step(q0, ex, backward=False)
        self.assertGreaterEqual(terms['keep_kl'], 0.0)
        same = [dict(e, N=list(e['G'])) for e in ex]          # identical G and N: no push
        _, t2 = obj.step(q0, same, backward=False)
        for k, v in t2.items():
            if k.startswith('push_'):
                self.assertAlmostEqual(v, 0.0, places=5)


    def test_cycle_visits_every_prompt_once_per_pass(self):
        c = T.Cycle(['a', 'b', 'c', 'd'], random.Random(1))
        first = [c.next() for _ in range(4)]
        second = [c.next() for _ in range(4)]
        self.assertEqual(sorted(first), ['a', 'b', 'c', 'd'])
        self.assertEqual(sorted(second), ['a', 'b', 'c', 'd'])
        self.assertEqual(c.passes, 2)
        c2 = T.Cycle(['d', 'c', 'b', 'a'], random.Random(1))
        self.assertEqual([c2.next() for _ in range(4)], first)          # deterministic, independent of input order


class MainIntegrationTest(unittest.TestCase):
    """End to end on CPU: synthetic eval_v3 directories (manifest, raw cells, labels), two tiny LoRA adapters, a few
    steps with validation and early stopping; checks the outputs, the logged cosines and the provenance hashes."""

    @classmethod
    def setUpClass(cls):
        import json
        import numpy as np
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cfg = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        Qwen3ForCausalLM(cfg).save_pretrained(root / 'base')
        cls.adapters = {}
        for name in ('md', 'ic'):
            m = get_peft_model(Qwen3ForCausalLM.from_pretrained(root / 'base'),
                               LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj']))
            for n, prm in m.named_parameters():
                if 'lora_B' in n:
                    torch.nn.init.normal_(prm, std=0.1)
            m.save_pretrained(root / name)
            cls.adapters[name] = root / name / 'adapter_model.safetensors'
        rng = random.Random(9)
        mk = lambda n: [rng.randrange(3, 48) for _ in range(n)]

        def evaldir(d, loophole, problems):
            d = root / d
            (d / 'LABELS').mkdir(parents=True)
            (d / 'MANIFEST.json').write_text(json.dumps({'checkpoints': [{'name': 'ck'}], 'settings': [loophole, 'nocue']}))
            for st in (loophole, 'nocue'):
                cell = d / 'cells' / f'ck__{st}'
                cell.mkdir(parents=True)
                raw, lab = [], []
                for pid in problems:
                    prompt = mk(6)
                    for i in range(4):
                        rid = f'{st}-{pid}-{i}'
                        raw.append({'request_id': rid, 'problem_id': pid, 'sample_index': i,
                                    'prompt_token_ids': prompt, 'completion_token_ids': mk(5)})
                        lab.append({'request_id': rid, 'gaming': i % 2 == 0})
                (cell / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in raw))
                (cell / 'SCORE_COMPLETE.json').write_text('{}')
                if st != 'nocue':
                    (d / 'LABELS' / f'ck__{st}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in lab))
            return d

        cls.dirs = {}
        for k, lp in (('md', 'modify'), ('ic', 'incontext')):
            for sp, pids in (('fit', [1, 2, 3]), ('val', [7, 8])):
                cls.dirs[f'{k}_{sp}'] = evaldir(f'{k}_{sp}', lp, pids)
        bank = np.zeros((5, 16), np.float32)
        bank[2] = np.arange(16) + 1.0
        np.savez(root / 'start.npz', v=bank)
        joint = np.zeros((37, 16), np.float32)
        joint[20, 0] = 1.0
        np.savez(root / 'joint.npz', unit=joint)
        np.savez(root / 'size.npz', d=bank * 0.25)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def spec(self, split):
        r = Path(self.tmp.name)
        return ','.join(f'{lp}={r / k}@{self.dirs[f"{k}_{split}"]}@{self.dirs[f"{k}_{split}"]}'
                        for lp, k in (('modify', 'md'), ('incontext', 'ic')))

    def run_main(self, out, variant, extra=()):
        import sys as _sys
        r = Path(self.tmp.name)
        argv = ['train_v7.py', '--model', str(r / 'base'), '--source', self.spec('fit'), '--val-dirs', self.spec('val'),
                '--block', '1', '--variant', variant, '--start', f'{r / "start.npz"}:v', '--joint', f'{r / "joint.npz"}:unit',
                '--steps', '7', '--val-every', '2', '--patience', '100', '--val-batches', '2', '--kG', '2', '--kN', '2',
                '--device', 'cpu', '--out', str(out), *extra]
        if variant == 'RA':
            argv += ['--size-from', f'{r / "size.npz"}:d']
        old, _sys.argv = _sys.argv, argv
        try:
            T.main()
        finally:
            _sys.argv = old

    def test_runs_end_to_end(self):
        import hashlib
        import json
        import numpy as np
        for variant in ('R', 'RA'):
            out = Path(self.tmp.name) / f'out_{variant}'
            self.run_main(out, variant)
            log = json.loads((out / 'LOG.json').read_text())
            bank = np.load(out / 'bank.npz')
            self.assertEqual(bank['unit'].shape, (5, 16))
            self.assertAlmostEqual(float(np.linalg.norm(bank['unit'][2])), 1.0, places=5)
            self.assertEqual(float(np.abs(np.delete(bank['unit'], 2, axis=0)).sum()), 0.0)
            if variant == 'RA':
                self.assertAlmostEqual(log['size'], float(np.linalg.norm(np.arange(16) + 1.0)) * 0.25, places=4)
                np.testing.assert_allclose(bank['vec'][2], log['size'] * bank['unit'][2], rtol=1e-5)
            traj = np.load(out / 'trajectory.npz')
            self.assertEqual(list(traj['step']), [0, 2, 4, 6])
            self.assertEqual(log['history'][-1]['step'], 6)
            self.assertEqual(log['history'][-1]['passes'], [3, 3])      # 3 fit prompts per source, 7 steps
            i = list(traj['step']).index(log['best_step'])
            np.testing.assert_allclose(traj['q'][i], bank['unit'][2], atol=1e-6)
            self.assertAlmostEqual(log['cos_joint_final'], float(bank['unit'][2][0]), places=3)
            self.assertIn('push_modify', log['history'][0])
            self.assertIn('floored_incontext', log['history'][0])
            prov = log['provenance']
            self.assertEqual(prov['adapters']['modify0']['adapter_model.safetensors'],
                             hashlib.sha256(self.adapters['md'].read_bytes()).hexdigest())
            self.assertEqual(len(prov['eval_dirs']), 4)
            self.assertEqual(log['usable_problems'], {'modify0': 3, 'incontext1': 3})
            self.assertEqual(log['val_usable_problems'], {'modify0': 2, 'incontext1': 2})

    def test_select_scores_runs_on_the_same_batches(self):
        import json
        import sys as _sys
        import select_v7 as S
        r = Path(self.tmp.name)
        outs = []
        for variant in ('R', 'RA'):
            out = r / f'sel_{variant}'
            self.run_main(out, variant)
            outs.append(str(out))
        argv = ['select_v7.py', '--model', str(r / 'base'), '--val-dirs', self.spec('val'), '--runs', ','.join(outs),
                '--batches', '3', '--device', 'cpu', '--out', str(r / 'select.json')]
        old, _sys.argv = _sys.argv, argv
        try:
            S.main()
        finally:
            _sys.argv = old
        res = json.loads((r / 'select.json').read_text())
        self.assertIn(res['chosen'], outs)
        self.assertEqual(len(res['scores']), 2)
        for sc in res['scores']:
            self.assertTrue(math.isfinite(sc['score']))
            self.assertNotIn('add_terms', sc['terms'])                  # removal objective only, for R and RA alike
        with self.assertRaises(AssertionError):                          # val dirs must be the runs' own
            argv[4] = self.spec('fit')
            old, _sys.argv = _sys.argv, argv
            try:
                S.main()
            finally:
                _sys.argv = old

    def test_select_tie_rule(self):
        import select_v7 as S
        mk = lambda run, v, b, k, s: {'run': run, 'variant': v, 'block': b, 'start_kind': k, 'score': s}
        self.assertEqual(S.choose([mk('a', 'RA', 19, 'add', 1.000), mk('b', 'R', 11, 'random', 1.009)])['run'], 'b')
        self.assertEqual(S.choose([mk('a', 'RA', 19, 'add', 1.000), mk('b', 'R', 11, 'random', 1.011)])['run'], 'a')
        self.assertEqual(S.choose([mk('a', 'R', 15, 'add', 1.0), mk('b', 'R', 19, 'meandiff', 1.005)])['run'], 'b')
        self.assertEqual(S.choose([mk('a', 'R', 19, 'random', 1.0), mk('b', 'R', 19, 'add', 1.005)])['run'], 'b')

    def test_val_sources_must_match_training_sources(self):
        r = Path(self.tmp.name)
        bad = self.spec('val').replace(str(r / 'ic'), str(r / 'md'))
        import sys as _sys
        argv = ['train_v7.py', '--model', str(r / 'base'), '--source', self.spec('fit'), '--val-dirs', bad, '--block', '1',
                '--variant', 'R', '--start', 'random', '--device', 'cpu', '--steps', '1', '--out', str(r / 'out_bad')]
        old, _sys.argv = _sys.argv, argv
        try:
            with self.assertRaises(AssertionError):
                T.main()
        finally:
            _sys.argv = old


if __name__ == '__main__':
    unittest.main()
