"""Authored CPU tensors only; no model, GPU, generation or remote calls.

Requires the existing Torch test runtime. Absence of Torch is an unmet test
dependency, not a skipped/pass result. All tensors explicitly remain on CPU.
"""
import unittest
import torch

from infra.gpu03.direction_discovery.intervention import ProjectionHooks
from .causal_hooks import CausalProjectionHooks, generation_predictor_mask, response_predictor_mask


class TupleBlock(torch.nn.Module):
    def forward(self, x):
        self.returned = (x, "unchanged-tail")
        return self.returned


class CausalHooksTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.q = torch.eye(4, dtype=torch.float32)[:, :1]
        self.x = torch.arange(8 * 5 * 4, dtype=torch.float32).reshape(8, 5, 4).to(torch.bfloat16) / 16
        self.mask = torch.zeros((8, 5), dtype=torch.bool); self.mask[:7, -1] = True

    def apply(self, x=None, mask=None, **kwargs):
        x = self.x if x is None else x
        mask = self.mask if mask is None else mask
        block = torch.nn.Identity().eval()
        hooks = CausalProjectionHooks([block], {0: self.q}, batch_size=x.shape[0], expected_hidden_size=4, **kwargs)
        with torch.inference_mode(), hooks:
            with hooks.positions(mask, scope="prefill"):
                result = block(x)
            energy, rows = hooks.energy_report(), hooks.energy_by_row()
        self.assertFalse(block._forward_hooks)
        return result, energy, rows

    def test_ordinary_native_outputs_and_aggregate_energy_match_reviewed_hook(self):
        generator = torch.Generator(device="cpu").manual_seed(728)
        q, _ = torch.linalg.qr(torch.randn((4, 2), generator=generator, dtype=torch.float32))
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for strength in (1., .5, .25):
                with self.subTest(dtype=dtype, strength=strength):
                    x = self.x.to(dtype); old = torch.nn.Identity().eval(); new = torch.nn.Identity().eval()
                    with torch.inference_mode(), ProjectionHooks([old], {0: q}, strength=strength, expected_hidden_size=4) as legacy:
                        with legacy.positions(self.mask, scope="prefill"): expected = old(x)
                        old_report = legacy.energy_report()["0"]
                    with torch.inference_mode(), CausalProjectionHooks([new], {0: q}, strength=strength,
                            expected_hidden_size=4, batch_size=8) as hooks:
                        with hooks.positions(self.mask, scope="prefill"): actual = new(x)
                        new_report = hooks.energy_report()["0"]
                    self.assertTrue(torch.equal(expected, actual))
                    for key, value in old_report.items():
                        if key != "scopes": self.assertEqual(new_report[key], value)
                    self.assertTrue(torch.equal(actual[~self.mask], x[~self.mask]))

    def test_centered_formula_and_center_zero_matches_ordinary(self):
        center = torch.tensor([2., -1., .5, 4.], dtype=torch.float32)
        actual, _, _ = self.apply(operator="centered", centers={0: center}, strength=.5)
        selected = self.x[self.mask].float()
        expected = self.x.clone()
        expected[self.mask] = (selected - .5 * (((selected - center) @ self.q) @ self.q.T)).to(self.x.dtype)
        self.assertTrue(torch.equal(actual, expected))
        centered, _, _ = self.apply(operator="centered", centers={0: torch.zeros(4)}, strength=.25)
        ordinary, _, _ = self.apply(strength=.25)
        self.assertTrue(torch.equal(centered, ordinary))

    def test_center_cloned_after_admission(self):
        block = torch.nn.Identity().eval(); center = torch.ones(4)
        hooks = CausalProjectionHooks([block], {0: self.q}, batch_size=8, expected_hidden_size=4,
                                      operator="centered", centers={0: center})
        center.fill_(999)
        self.assertTrue(torch.equal(hooks.centers[0], torch.ones(4)))

    def test_zero_strength_is_object_identity_materialization_is_clone(self):
        zero, energy, _ = self.apply(control="zero_strength", strength=0.)
        self.assertIs(zero, self.x)
        self.assertFalse(energy["0"]["energy_measured"])
        self.assertIsNone(energy["0"]["activation_energy"])
        self.assertEqual(energy["0"]["actual_change_energy"], 0.)
        materialized, energy, _ = self.apply(control="materialization_only", strength=0.)
        self.assertIsNot(materialized, self.x)
        self.assertNotEqual(materialized.data_ptr(), self.x.data_ptr())
        self.assertTrue(torch.equal(materialized, self.x))
        self.assertTrue(energy["0"]["energy_measured"])
        self.assertEqual(energy["0"]["actual_change_energy"], 0.)

    def test_null_space_and_center_equal_input_remove_nothing(self):
        x = self.x.clone(); x[:, :, 0] = 0
        projected, _, _ = self.apply(x=x)
        self.assertTrue(torch.equal(projected, x))
        center = torch.tensor([3., 0., 0., 0.]); x[:, :, 0] = 3
        centered, energy, _ = self.apply(x=x, operator="centered", centers={0: center})
        self.assertTrue(torch.equal(centered, x))
        self.assertEqual(energy["0"]["removed_energy_fp32"], 0.)
        self.assertGreater(energy["0"]["remaining_subspace_energy_native"], 0.)

    def test_row_energies_track_only_actual_active_rows_after_eos(self):
        block = torch.nn.Identity().eval()
        hooks = CausalProjectionHooks([block], {0: self.q}, batch_size=8, expected_hidden_size=4)
        with torch.inference_mode(), hooks:
            with hooks.positions(self.mask, scope="prefill"): block(self.x)
            before = hooks.energy_by_row()
            decode_mask = torch.tensor([[True], [False], [True], [False], [False], [False], [False], [False]])
            with hooks.positions(decode_mask, scope="decode"): block(self.x[:, -1:])
            after = hooks.energy_by_row()
        self.assertEqual(after[1], before[1])
        self.assertEqual(after[7]["0"]["selected_tokens"], 0)
        self.assertEqual(after[0]["0"]["selected_tokens"], 2)
        self.assertEqual(after[0]["0"]["scopes"]["decode"]["selected_tokens"], 1)
        expected = float(self.x[0, -1, 0].float().square()) * 2
        self.assertEqual(after[0]["0"]["removed_energy_fp32"], expected)

    def test_teacher_forced_rows_count_tokens_separately_from_forwards(self):
        mask = torch.zeros_like(self.mask); mask[0, 1:4] = True; mask[1, 3] = True
        _, aggregate, rows = self.apply(mask=mask)
        self.assertEqual(rows[0]["0"]["selected_tokens"], 3)
        self.assertEqual(rows[0]["0"]["forward_calls"], 1)
        self.assertEqual(rows[1]["0"]["selected_tokens"], 1)
        self.assertEqual(aggregate["0"]["selected_tokens"], 4)
        self.assertAlmostEqual(sum(r["0"]["removed_energy_fp32"] for r in rows), aggregate["0"]["removed_energy_fp32"])

    def test_empty_masks_no_perturbation_or_nan_row_reports(self):
        output, _, rows = self.apply(mask=torch.zeros_like(self.mask))
        self.assertTrue(torch.equal(output, self.x))
        self.assertTrue(all(r["0"]["selected_tokens"] == 0 and r["0"]["removed_fraction"] is None for r in rows))

    def test_tuple_tail_identity_and_zero_control_exact_tuple_object(self):
        for control, strength in (("zero_strength", 0), ("materialization_only", 0), ("project", 1)):
            block = TupleBlock().eval()
            with torch.inference_mode(), CausalProjectionHooks([block], {0: self.q}, batch_size=8,
                    expected_hidden_size=4, control=control, strength=strength) as hooks:
                with hooks.positions(self.mask): result = block(self.x)
                self.assertEqual(result[1], "unchanged-tail")
                if control == "zero_strength": self.assertIs(result, block.returned)

    def test_bad_centers_strengths_and_q_rejected(self):
        block = torch.nn.Identity().eval()
        defaults = {"batch_size": 8, "expected_hidden_size": 4}
        for kwargs in ({"strength": 0.}, {"strength": 1.01}, {"strength": True}, {"control": "zero_strength"},
                       {"control": "materialization_only", "strength": .5}, {"operator": "centered"},
                       {"centers": {0: torch.zeros(4)}}, {"batch_size": 9}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                CausalProjectionHooks([block], {0: self.q}, **{**defaults, **kwargs})
        for center in (torch.zeros(3), torch.zeros(4, dtype=torch.bfloat16), torch.full((4,), float("nan")),
                       torch.zeros(4, requires_grad=True)):
            with self.subTest(center=center), self.assertRaises(ValueError):
                CausalProjectionHooks([block], {0: self.q}, **defaults, operator="centered", centers={0: center})
        with self.assertRaises(ValueError): CausalProjectionHooks([block], {0: self.q * 2}, **defaults)

    def test_nonfinite_and_native_overflow_fail_and_remove_hooks(self):
        for center, x in ((None, self.x.clone()), (torch.tensor([-3e38, 0., 0., 0.]), self.x.clone())):
            if center is None: x[0, -1, 0] = float("nan")
            else: x[0, -1, 0] = 3e38
            block = torch.nn.Identity().eval()
            kwargs = {} if center is None else {"operator": "centered", "centers": {0: center}}
            with self.assertRaises(ValueError):
                with torch.inference_mode(), CausalProjectionHooks([block], {0: self.q}, batch_size=8, expected_hidden_size=4, **kwargs) as hooks:
                    with hooks.positions(self.mask): block(x)
            self.assertFalse(block._forward_hooks)

    def test_energy_overflow_fails_without_claiming_finite_diagnostics(self):
        x = torch.full_like(self.x, 1e20)
        with self.assertRaisesRegex(ValueError, "energy accounting"): self.apply(x=x)

    def test_inherited_inference_context_guards_and_exception_cleanup(self):
        block = torch.nn.Identity().eval()
        with self.assertRaisesRegex(RuntimeError, "gradients disabled"):
            with CausalProjectionHooks([block], {0: self.q}, batch_size=8, expected_hidden_size=4) as hooks:
                with hooks.positions(self.mask): block(self.x)
        self.assertFalse(block._forward_hooks)
        with torch.inference_mode(), CausalProjectionHooks([block], {0: self.q}, batch_size=8, expected_hidden_size=4) as hooks:
            with self.assertRaisesRegex(RuntimeError, "positions"): block(self.x)
            with self.assertRaisesRegex(RuntimeError, "more than once"):
                with hooks.positions(self.mask): block(self.x); block(self.x)
            with self.assertRaisesRegex(RuntimeError, "Expected one invocation"):
                with hooks.positions(self.mask): pass
            with hooks.disabled(): self.assertIs(block(self.x), self.x)
        self.assertFalse(block._forward_hooks)

    def test_autocast_mask_shape_and_declared_batch_refused(self):
        with self.assertRaises(ValueError): self.apply(mask=torch.ones((8, 1), dtype=torch.bool))
        block = torch.nn.Identity().eval()
        with torch.inference_mode(), CausalProjectionHooks([block], {0: self.q}, batch_size=7, expected_hidden_size=4) as hooks:
            with self.assertRaisesRegex(ValueError, "declared batch"):
                with hooks.positions(self.mask): block(self.x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), self.assertRaisesRegex(RuntimeError, "autocast"):
            self.apply()

    def test_row_reset_and_read_during_forward(self):
        block = torch.nn.Identity().eval()
        with torch.inference_mode(), CausalProjectionHooks([block], {0: self.q}, batch_size=8, expected_hidden_size=4) as hooks:
            with hooks.positions(self.mask):
                block(self.x)
                with self.assertRaises(RuntimeError): hooks.energy_by_row()
            first = hooks.energy_by_row(reset=True)
            self.assertEqual(first[0]["0"]["selected_tokens"], 1)
            self.assertEqual(hooks.energy_by_row()[0]["0"]["selected_tokens"], 0)


class PredictorMaskTests(unittest.TestCase):
    def test_teacher_forcing_excludes_prompt_padding_and_final_completion_state(self):
        attention = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
        mask = response_predictor_mask(attention, torch.tensor([2, 3]), torch.tensor([2, 2]))
        self.assertTrue(torch.equal(mask, torch.tensor([[False, False, False, True, True, False],
                                                      [False, False, True, True, False, False]])))
        self.assertTrue(torch.equal(mask.sum(1), torch.tensor([2, 2])))

    def test_prefill_and_cached_eos_masks(self):
        attention = torch.tensor([[0, 1, 1], [1, 1, 1]])
        mask = generation_predictor_mask(attention, current_sequence_length=3, use_cache_decode=False)
        self.assertTrue(torch.equal(mask, torch.tensor([[False, False, True], [False, False, True]])))
        attention = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])
        mask = generation_predictor_mask(attention, current_sequence_length=1, use_cache_decode=True,
                                         active_rows=torch.tensor([True, False]))
        self.assertTrue(torch.equal(mask, torch.tensor([[True], [False]])))

    def test_bad_lengths_holes_and_cached_chunk_refused(self):
        with self.assertRaises(ValueError):
            response_predictor_mask(torch.tensor([[1, 0, 1]]), torch.tensor([1]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            response_predictor_mask(torch.tensor([[1, 1, 1]]), torch.tensor([1]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            response_predictor_mask(torch.tensor([[1, 1]]), torch.tensor([0]), torch.tensor([2]))
        with self.assertRaises(ValueError):
            generation_predictor_mask(torch.tensor([[1, 1]]), current_sequence_length=2, use_cache_decode=True)
        with self.assertRaises(ValueError):
            generation_predictor_mask(torch.tensor([[1, 0]]), current_sequence_length=1, use_cache_decode=True)


if __name__ == "__main__": unittest.main()
