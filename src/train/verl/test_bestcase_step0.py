"""CPU tests for the best-case study's step-0 changes.

Covers: multi-layer CAFT spec validation and Q loading, the multi-layer actor
projection (values and gradients against an independent reference), that the
no-projection arm installs nothing, that the historical single-layer path is
unchanged, seeded LoRA creation, 8-rank-equivalent batch balancing, and the
reward timing fix.
"""
import copy
import hashlib
import os
import tempfile
import unittest
from types import SimpleNamespace as NS

import torch
from safetensors.torch import save

from verl.utils import caft


def unit(seed, dim=2560):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, 1, generator=g, dtype=torch.float64)
    return (v / v.norm()).float()


def write_qs(directory, qs, name='q.safetensors'):
    payload = {f'layer_{layer}': q.contiguous() for layer, q in qs.items()}
    data = save(payload)
    path = os.path.join(directory, name)
    with open(path, 'wb') as f:
        f.write(data)
    return path, data


def v2_spec(path, data, qs, arm='caft_pairs'):
    return {
        'protocol': caft.PROTOCOL_V2, 'arm': arm, 'layers': sorted(qs), 'strength': 1.0,
        'q': {'file': {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)},
              'tensors': {str(l): {'key': f'layer_{l}',
                                   'tensor_sha256': hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest()}
                          for l, q in qs.items()}},
        'scope': 'response_predictors', 'reference_projection': False, 'evaluation_projection': False,
        'rollout_prefix_caching': True, 'rollout_enforce_eager': False,
        'direction': {'source': 'unit-test'}}


def none_spec():
    return {'protocol': caft.PROTOCOL_V2, 'arm': 'none', 'layers': [], 'strength': 0.0, 'q': None,
            'scope': 'response_predictors', 'reference_projection': False, 'evaluation_projection': False,
            'rollout_prefix_caching': True, 'rollout_enforce_eager': False, 'direction': {'source': 'none'}}


class Qwen3DecoderLayer(torch.nn.Module):
    """Stand-in decoder layer: a fixed linear map so projections interact with later layers."""
    hidden_size = 2560

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(1000 + seed)
        self.scale = torch.nn.Parameter(1 + 0.01 * torch.randn(2560, generator=g))

    def forward(self, x):
        return (x * self.scale,)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(i) for i in range(36)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


def micro_batch(prompt=3, response=4, batch=2):
    ids = torch.arange(batch * (prompt + response)).reshape(batch, prompt + response)
    attention = torch.ones_like(ids)
    attention[1, -1] = 0  # trailing response padding on row 1
    return {'input_ids': ids, 'attention_mask': attention, 'responses': ids[:, prompt:]}


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.qs = {22: unit(1), 23: unit(2), 27: unit(3)}
        self.path, self.data = write_qs(self.dir, self.qs)
        self.spec = v2_spec(self.path, self.data, self.qs)

    def test_valid_spec_and_load(self):
        caft.validate_spec(self.spec)
        loaded = caft.load_qs(self.spec)
        self.assertEqual(sorted(loaded), [22, 23, 27])
        for layer, q in self.qs.items():
            self.assertTrue(torch.equal(loaded[layer], q))

    def test_rejects_bad_layers(self):
        for layers in ([23, 22, 27], [22, 22, 27], [22, 23, 36], [-1]):
            bad = copy.deepcopy(self.spec); bad['layers'] = layers
            with self.assertRaises(ValueError):
                caft.validate_spec(bad)

    def test_rejects_missing_layer_tensor(self):
        bad = copy.deepcopy(self.spec); del bad['q']['tensors']['27']
        with self.assertRaises(ValueError):
            caft.validate_spec(bad)

    def test_rejects_tampered_file_and_tensor(self):
        bad = copy.deepcopy(self.spec); bad['q']['file']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            caft.load_qs(bad)
        bad = copy.deepcopy(self.spec); bad['q']['tensors']['22']['tensor_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            caft.load_qs(bad)

    def test_rejects_non_unit_q(self):
        qs = {22: unit(1) * 2}
        path, data = write_qs(self.dir, qs, 'bad.safetensors')
        with self.assertRaises(ValueError):
            caft.load_qs(v2_spec(path, data, qs))

    def test_none_arm(self):
        caft.validate_spec(none_spec())
        self.assertIsNone(caft.load_qs(none_spec()))
        bad = none_spec(); bad['strength'] = 1.0
        with self.assertRaises(ValueError):
            caft.validate_spec(bad)
        model = Tiny()
        self.assertIsNone(caft.attach_hf(model, none_spec()))
        self.assertEqual(sum(len(l._forward_hooks) for l in model.layers), 0)

    def test_extra_or_missing_keys_rejected(self):
        bad = copy.deepcopy(self.spec); bad['layer'] = 21
        with self.assertRaises(ValueError):
            caft.validate_spec(bad)
        bad = copy.deepcopy(self.spec); del bad['direction']
        with self.assertRaises(ValueError):
            caft.validate_spec(bad)


class MultiLayerProjectionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.qs = {22: unit(11), 23: unit(12), 27: unit(13)}
        path, data = write_qs(self.dir, self.qs)
        self.spec = v2_spec(path, data, self.qs)

    def reference(self, model, x, mask):
        h = x
        for index, layer in enumerate(model.layers):
            h = layer(h)[0]
            if index in self.qs:
                d = self.qs[index][:, 0].double()
                sel = h[mask].double()
                sel = sel - (sel @ d).unsqueeze(-1) * d
                h = h.clone(); h[mask] = sel.to(h.dtype)
        return h

    def test_values_and_gradients_match_reference(self):
        torch.manual_seed(0)
        model = Tiny()
        controller = caft.attach_hf(model, self.spec)
        self.assertIsInstance(controller, caft.MultiLayerProjection)
        self.assertEqual(controller.layer_indices, [22, 23, 27])
        seen = {}
        # Registered after the CAFT hook, so it sees layer 27's projected output.
        def capture(module, args, output):  # must return None, or it would replace the layer output
            seen.setdefault('l27', output[0].detach())
        model.layers[27].register_forward_hook(capture)  # first call only
        mb = micro_batch()
        mask = caft.predictor_mask(mb)
        x = torch.randn(*mask.shape, 2560, requires_grad=True)
        with controller.scope(mb):
            out = model(x)
        (out * torch.linspace(-1, 1, 2560)).sum().backward()
        grad = x.grad.clone(); x.grad = None
        ref = self.reference(model, x, mask)
        (ref * torch.linspace(-1, 1, 2560)).sum().backward()
        self.assertLess(float((out - ref).abs().max()), 1e-4)
        self.assertLess(float((grad - x.grad).abs().max()), 1e-4)
        # Unselected rows are untouched by the projection.
        plain = model.layers[0](x)[0]
        for layer in model.layers[1:]:
            plain = layer(plain)[0]
        self.assertTrue(torch.equal(out[~mask], plain[~mask]))
        # Right after layer 27, projected rows have no component along layer 27's direction
        # (later layers rescale features, so the final output need not stay orthogonal).
        l27 = seen['l27']
        self.assertLess(float((l27[mask].double() @ self.qs[27][:, 0].double()).abs().max()), 1e-3)
        self.assertGreater(float((l27[~mask].double() @ self.qs[27][:, 0].double()).abs().max()), 1e-2)
        self.assertEqual(controller.forward_calls, 3)

    def test_outside_scope_and_disabled_do_not_project(self):
        model = Tiny()
        controller = caft.attach_hf(model, self.spec)
        mb = micro_batch(); mask = caft.predictor_mask(mb)
        x = torch.randn(*mask.shape, 2560)
        plain = x
        for layer in model.layers:
            plain = layer(plain)[0]
        self.assertTrue(torch.equal(model(x), plain))  # no scope -> off
        with controller.disabled(), controller.scope(mb):
            self.assertTrue(torch.equal(model(x), plain))  # disabled (reference policy) -> off
        self.assertIsNone(controller.active_mask)

    def test_evidence_counts_all_layers(self):
        model = Tiny()
        controller = caft.attach_hf(model, self.spec)
        evidence = {'layer_forward_calls': 0, 'projection_forward_calls': 0, 'role': 'old', '_handles': []}
        controller.evidence = evidence
        mb = micro_batch(); mask = caft.predictor_mask(mb)
        with controller.scope(mb):
            model(torch.randn(*mask.shape, 2560))
        self.assertEqual(evidence['projection_forward_calls'], 3)
        self.assertEqual(evidence['layer_forward_calls'], 3)


class ReferenceProjectionTests(unittest.TestCase):
    """Protocol 7.3: with reference_projection=True the KL reference forward keeps the arm's projection."""
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.qs = {22: unit(21), 27: unit(22)}
        path, data = write_qs(self.dir, self.qs)
        self.spec = v2_spec(path, data, self.qs)

    def run_reference(self, spec):
        model = Tiny()
        controller = caft.attach_hf(model, spec)
        actor = NS(caft=controller, caft_spec=spec)
        mb = micro_batch(); mask = caft.predictor_mask(mb)
        x = torch.randn(*mask.shape, 2560)
        with controller.scope(mb):
            projected = model(x)
        plain = x
        for layer in model.layers:
            plain = layer(plain)[0]
        with caft.reference_scope(actor), controller.scope(mb):
            ref = model(x)
        self.assertEqual(getattr(actor, '_caft_reference_depth'), 0)
        return projected, plain, ref, mask

    def test_v2_accepts_both_values_and_v1_still_rejects_true(self):
        for value in (False, True):
            spec = dict(self.spec, reference_projection=value)
            caft.validate_spec(spec)
        with self.assertRaises(ValueError):
            caft.validate_spec(dict(self.spec, reference_projection='yes'))

    def test_reference_projected_only_when_declared(self):
        on = dict(self.spec, reference_projection=True)
        projected, plain, ref, mask = self.run_reference(on)
        self.assertTrue(caft.projects_reference(NS(caft_spec=on)))
        self.assertTrue(torch.equal(ref, projected))           # reference forward == projected policy forward
        self.assertFalse(torch.equal(ref[mask], plain[mask]))
        off = dict(self.spec, reference_projection=False)
        projected, plain, ref, mask = self.run_reference(off)
        self.assertFalse(caft.projects_reference(NS(caft_spec=off)))
        self.assertTrue(torch.equal(ref, plain))                # historical behaviour: reference unprojected

    def test_no_spec_or_v1_never_projects_reference(self):
        self.assertFalse(caft.projects_reference(NS(caft_spec=None)))
        self.assertFalse(caft.projects_reference(NS()))
        self.assertFalse(caft.projects_reference(NS(caft_spec={'protocol': caft.PROTOCOL, 'reference_projection': True})))

    def test_structural_zero_reference_runs_hooks_but_changes_nothing(self):
        spec = dict(self.spec, strength=0.0, reference_projection=True)
        projected, plain, ref, mask = self.run_reference(spec)
        self.assertTrue(torch.equal(ref, plain))
        self.assertTrue(torch.equal(projected, plain))


class HistoricalPathTests(unittest.TestCase):
    def test_v1_single_layer_unchanged(self):
        d = tempfile.mkdtemp()
        q = unit(21)
        data = save({'target_pc4': q.contiguous()})
        path = os.path.join(d, 'pc4.safetensors')
        open(path, 'wb').write(data)
        spec = {'protocol': caft.PROTOCOL, 'arm': 'pc4', 'layer': 21, 'strength': 1.0,
                'q': {'file': {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)},
                      'key': 'target_pc4', 'shape': [2560, 1], 'dtype': 'float32',
                      'tensor_sha256': hashlib.sha256(q.contiguous().numpy().tobytes()).hexdigest()},
                'source_condition_id': 'stage9-pc4-alpha100', 'scope': 'response_predictors',
                'reference_projection': False, 'evaluation_projection': False,
                'rollout_prefix_caching': True, 'rollout_enforce_eager': False}
        model = Tiny()
        controller = caft.attach_hf(model, spec)
        self.assertIsInstance(controller, caft.HFProjection)
        self.assertEqual([len(l._forward_hooks) for l in model.layers].index(1), 21)
        self.assertEqual(sum(len(l._forward_hooks) for l in model.layers), 1)
        self.assertEqual(sorted(caft.load_qs(spec)), [21])
        self.assertEqual(caft.q_identity(spec), spec['q']['tensor_sha256'])


class SeededLoraTests(unittest.TestCase):
    def build(self):
        from peft import LoraConfig
        torch.manual_seed(123)  # base weights fixed; only the adapter draw should depend on the seed
        base = torch.nn.Sequential()
        base.add_module('q_proj', torch.nn.Linear(16, 16))
        base.add_module('v_proj', torch.nn.Linear(16, 16))
        return base, LoraConfig(r=4, lora_alpha=4, target_modules=['q_proj', 'v_proj'])

    def lora_a(self, model):
        return {n: p.detach().clone() for n, p in model.named_parameters() if '.lora_A.' in n}

    def test_seeded_is_reproducible_and_preserves_global_rng(self):
        from peft import get_peft_model
        from verl.utils.lora_seed import seeded_get_peft_model
        base, config = self.build()
        before = torch.get_rng_state().clone()
        a1 = self.lora_a(seeded_get_peft_model(get_peft_model, base, config, 7))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        base, config = self.build()
        a2 = self.lora_a(seeded_get_peft_model(get_peft_model, base, config, 7))
        base, config = self.build()
        a3 = self.lora_a(seeded_get_peft_model(get_peft_model, base, config, 8))
        self.assertEqual(sorted(a1), sorted(a2))
        self.assertTrue(all(torch.equal(a1[k], a2[k]) for k in a1))
        self.assertFalse(all(torch.equal(a1[k], a3[k]) for k in a1))

    def test_rejects_bad_seed(self):
        from peft import get_peft_model
        from verl.utils.lora_seed import seeded_get_peft_model
        base, config = self.build()
        with self.assertRaises(ValueError):
            seeded_get_peft_model(get_peft_model, base, config, -1)


class VirtualBalanceTests(unittest.TestCase):
    def order(self, world, virtual, lengths):
        from verl import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        n = len(lengths)
        width = max(lengths)
        attention = torch.zeros(n, width, dtype=torch.long)
        for i, length in enumerate(lengths):
            attention[i, :length] = 1
        batch = DataProto.from_dict({'attention_mask': attention, 'row': torch.arange(n)})
        trainer = NS(actor_rollout_wg=NS(world_size=world),
                     config=NS(trainer={'balance_virtual_dp_size': virtual} if virtual else {},
                               actor_rollout_ref=NS(actor={})))
        trainer.config.trainer = _Get(trainer.config.trainer)
        RayPPOTrainer._balance_batch(trainer, batch, {})
        return batch.batch['row'].tolist()

    def test_one_rank_with_virtual_8_matches_eight_ranks(self):
        g = torch.Generator().manual_seed(5)
        lengths = (torch.randint(50, 1600, (256,), generator=g)).tolist()
        eight = self.order(8, None, lengths)
        one_virtual = self.order(1, 8, lengths)
        one_plain = self.order(1, None, lengths)
        self.assertEqual(eight, one_virtual)
        self.assertNotEqual(eight, one_plain)
        self.assertEqual(self.order(2, 8, lengths), eight)

    def test_rejects_incompatible_virtual_size(self):
        with self.assertRaises(ValueError):
            self.order(3, 8, [10] * 24)


class _Get(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class RewardTimingTests(unittest.TestCase):
    def test_cpu_limit_is_one_second_above_alarm(self):
        from src.evaluate import helpers
        code = helpers._SUBPROCESS_CODE
        self.assertIn('cpu_seconds = max(int(time_limit), 1) + 1', code)
        self.assertIn('signal.alarm(max(int(time_limit), 1))', code)

    def test_busy_loop_times_out_as_compiled(self):
        # Our own benign test program (not model-generated): a CPU-bound loop must be reported as a
        # timeout of compiled code, i.e. the wall alarm fires before the CPU limit kills the process.
        from src.evaluate import helpers
        for _ in range(2):
            result = helpers._execute_in_subprocess('while True:\n    pass\n', timeout=1, memory_limit=256)
            self.assertTrue(result.timeout)
            self.assertTrue(result.compiled)


if __name__ == '__main__':
    unittest.main()


class DirectionToolTests(unittest.TestCase):
    def test_logistic_probe_to_unit_direction(self):
        from sklearn.linear_model import LogisticRegression
        from src.probe import LogisticRegressionProbe
        from src.bestcase.directions import unit_from_logistic
        g = torch.Generator().manual_seed(3)
        true = unit(99)[:, 0]
        x = torch.randn(400, 2560, generator=g)
        y = ((x @ true) > 0).long()
        clf = LogisticRegression(max_iter=2000).fit(x.numpy(), y.numpy())
        d = tempfile.mkdtemp()
        path = os.path.join(d, 'p.lgprobe')
        LogisticRegressionProbe({34: clf}).save(path)
        u = unit_from_logistic(path, 34)
        coef = torch.tensor(clf.coef_[0], dtype=torch.float64)
        self.assertAlmostEqual(float(u.double().norm()), 1.0, places=5)
        self.assertLess(float((u.double() - coef / coef.norm()).abs().max()), 1e-6)  # exactly the normalised weights
        self.assertGreater(float(u @ true), 0.0)  # and oriented towards the positive (hack) class

    def test_bank_spec_roundtrip_and_load(self):
        from src.bestcase.directions import write_direction_bank, v2_spec
        d = tempfile.mkdtemp()
        dirs = {22: unit(5)[:, 0] * 3, 27: unit(6)[:, 0]}
        q = write_direction_bank(os.path.join(d, 'bank.safetensors'), dirs, {'source': 'test'})
        spec = v2_spec('caft_pairs', q, [27, 22], {'source': 'test'})
        loaded = caft.load_qs(spec)
        self.assertEqual(sorted(loaded), [22, 27])
        self.assertTrue(torch.allclose(loaded[22][:, 0], unit(5)[:, 0], atol=1e-6))
        none = v2_spec('none', None, [], {'source': 'none'})
        caft.validate_spec(none)
        self.assertIsNone(caft.load_qs(none))

    def test_centered_probe_ranks_like_projection_and_threshold(self):
        from src.probe import CenteredMassMeanProbe, load_probe
        from src.bestcase.directions import threshold_at_fpr
        g = torch.Generator().manual_seed(4)
        acts = torch.randn(2, 300, 2560, generator=g) * 20
        labels = torch.zeros(300); labels[:150] = 1
        acts[:, :150, :] += 3 * unit(7)[:, 0]
        p = CenteredMassMeanProbe(); p.fit(acts, labels, layers=[22, 27])
        d = tempfile.mkdtemp()
        path = p.save(os.path.join(d, 'x.cmmprobe'))
        q = load_probe(path, threshold=0.7)
        probs = q.predict_proba(acts, layers=[22, 27])
        raw = q.raw_scores(acts, layers=[22, 27])
        self.assertEqual(tuple(probs.shape), (300, 2))
        self.assertTrue(torch.equal(probs[:, 0].argsort(), raw[:, 0].argsort()))
        self.assertTrue(0.05 < float(probs.mean()) < 0.95)  # not saturated
        neg = probs[150:, 0].numpy()
        t = threshold_at_fpr(neg, 0.05)
        self.assertLessEqual(float((neg >= t).mean()), 0.05)


class StructuralZeroTests(unittest.TestCase):
    def test_structural_zero_installs_hooks_but_changes_nothing(self):
        d = tempfile.mkdtemp()
        qs = {22: unit(31), 27: unit(32)}
        path, data = write_qs(d, qs)
        spec = v2_spec(path, data, qs, arm='none_structural')
        spec['strength'] = 0.0
        caft.validate_spec(spec)
        model = Tiny()
        controller = caft.attach_hf(model, spec)
        self.assertIsInstance(controller, caft.MultiLayerProjection)
        self.assertEqual(sum(len(l._forward_hooks) for l in model.layers), 2)
        mb = micro_batch(); mask = caft.predictor_mask(mb)
        x = torch.randn(*mask.shape, 2560)
        plain = x
        for layer in model.layers:
            plain = layer(plain)[0]
        evidence = {'layer_forward_calls': 0, 'projection_forward_calls': 0, 'role': 'old', '_handles': []}
        controller.evidence = evidence
        with controller.scope(mb):
            out = model(x)
        self.assertTrue(torch.equal(out, plain))           # bitwise identical to no projection
        self.assertEqual(evidence['projection_forward_calls'], 2)  # but the hooks did run

    def test_other_strengths_rejected(self):
        d = tempfile.mkdtemp()
        qs = {22: unit(33)}
        path, data = write_qs(d, qs)
        spec = v2_spec(path, data, qs)
        spec['strength'] = 0.5
        with self.assertRaises(ValueError):
            caft.validate_spec(spec)
