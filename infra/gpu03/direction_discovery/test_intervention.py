"""CPU-only unit and tiny-HF success-path tests; no model downloads or CUDA use."""

import unittest

import torch

from intervention import ProjectionHooks, generation_position_mask, last_token_mask


class Block(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output
        self.cache = object()
        self.attention = torch.tensor([3.0])
        self.eval()

    def forward(self, hidden):
        return (hidden, self.cache, self.attention) if self.tuple_output else hidden


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.block = Block()
        self.hidden = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6) / 7
        self.q = torch.eye(6)[:, :2]
        self.mask = torch.tensor([[False, False, True], [False, True, False]])

    def manager(self, q=None, block=None, **kwargs):
        return ProjectionHooks([block or self.block], {0: self.q if q is None else q}, expected_hidden_size=6, **kwargs)

    def test_fractional_and_overremoval_strength_energy_matches_actual_update(self):
        x = self.hidden[self.mask]
        component = (x @ self.q) @ self.q.T
        for strength in (0.5, 1.0, 2.0, 5.0):
            with self.subTest(strength=strength), torch.inference_mode(), self.manager(strength=strength) as hooks:
                with hooks.positions(self.mask, scope='teacher_forced'):
                    actual = self.block(self.hidden)
                expected = x.double() - strength * ((x.double() @ self.q.double()) @ self.q.double().T)
                torch.testing.assert_close(actual[self.mask].double(), expected, rtol=1e-7, atol=1e-6)
                self.assertTrue(torch.equal(actual[~self.mask], self.hidden[~self.mask]))
                report = hooks.energy_report()['0']
                self.assertEqual(report['strength'], strength)
                self.assertEqual(report['scopes']['teacher_forced']['strength'], strength)
                self.assertEqual(report['removed_energy_fp32'], float((component * strength).square().sum()))
                self.assertEqual(report['actual_change_energy'], float((x - actual[self.mask]).square().sum()))
                self.assertEqual(report['remaining_subspace_energy_native'], float((actual[self.mask] @ self.q).square().sum()))
                self.assertEqual(report['remaining_subspace_energy_fp32'], report['remaining_subspace_energy_native'])

    def test_zero_strength_is_bitwise_noop_including_signed_zero(self):
        for dtype, bits in ((torch.float32, torch.int32), (torch.float16, torch.int16), (torch.bfloat16, torch.int16)):
            hidden = self.hidden.to(dtype)
            hidden[0, 2, 0] = -0.0
            with self.subTest(dtype=dtype), torch.inference_mode(), self.manager(strength=0) as hooks:
                with hooks.positions(self.mask):
                    actual = self.block(hidden)
                self.assertTrue(torch.equal(actual.view(bits), hidden.view(bits)))
                report = hooks.energy_report()['0']
                self.assertEqual(report['removed_energy_fp32'], 0)
                self.assertEqual(report['actual_change_energy'], 0)
                self.assertEqual(report['remaining_subspace_energy_fp32'], float((hidden[self.mask].float() @ self.q).square().sum()))

    def test_default_and_explicit_one_match_legacy_arithmetic_and_energy_exactly(self):
        # The historical manager owns a contiguous Q; match that BLAS layout.
        q = torch.linalg.qr(torch.randn(6, 2, generator=torch.Generator().manual_seed(81))).Q.contiguous()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            hidden = self.hidden.to(dtype)
            selected = hidden[self.mask].float()
            removed = (selected @ q) @ q.T
            projected = selected - removed
            native = projected.to(dtype)
            expected = hidden.clone(); expected[self.mask] = native
            legacy_values = [selected.square().sum(), removed.square().sum(),
                             (selected - native.float()).square().sum(),
                             (projected @ q).square().sum(), (native.float() @ q).square().sum()]
            reports = []
            for kwargs in ({}, {'strength': 1.0}):
                with torch.inference_mode(), self.manager(q, **kwargs) as hooks:
                    with hooks.positions(self.mask):
                        actual = self.block(hidden)
                    self.assertTrue(torch.equal(actual, expected))
                    report = hooks.energy_report()['0']; reports.append(report)
                    for field, value in zip(('activation_energy', 'removed_energy_fp32', 'actual_change_energy',
                                             'remaining_subspace_energy_fp32', 'remaining_subspace_energy_native'), legacy_values):
                        self.assertEqual(report[field], float(value))
            self.assertEqual(reports[0], reports[1])

    def test_strength_rejects_nonfinite_boolean_and_out_of_range_before_hooks(self):
        for strength in (True, False, None, '.5', float('nan'), float('inf'), -0.1, 5.01, 10**1000, torch.tensor(.5)):
            with self.subTest(strength_type=type(strength)), self.assertRaisesRegex(ValueError, 'strength'):
                self.manager(strength=strength)
        self.assertEqual(len(self.block._forward_hooks), 0)

    def test_fractional_bfloat16_energy_accounts_for_native_rounding(self):
        q = torch.linalg.qr(torch.randn(6, 2, generator=torch.Generator().manual_seed(71))).Q.contiguous()
        hidden = self.hidden.bfloat16(); x = hidden[self.mask].float()
        with torch.inference_mode(), self.manager(q, strength=.5) as hooks:
            with hooks.positions(self.mask):
                actual = self.block(hidden)
        expected = x - .5 * ((x @ q) @ q.T)
        self.assertTrue(torch.equal(actual[self.mask], expected.bfloat16()))
        report = hooks.energy_report()['0']
        self.assertEqual(report['actual_change_energy'], float((x - expected.bfloat16().float()).square().sum()))
        self.assertEqual(report['remaining_subspace_energy_fp32'], float((expected @ q).square().sum()))
        self.assertEqual(report['remaining_subspace_energy_native'], float((actual[self.mask].float() @ q).square().sum()))

    def test_projection_removes_columns_and_preserves_other_tokens_exactly(self):
        original = self.hidden.clone()
        with torch.inference_mode(), self.manager() as hooks:
            with hooks.positions(self.mask, scope="teacher_forced"):
                modified = self.block(self.hidden)
        torch.testing.assert_close(modified[self.mask] @ self.q, torch.zeros((2, 2)), atol=0, rtol=0)
        self.assertTrue(torch.equal(modified[~self.mask], original[~self.mask]))
        self.assertTrue(torch.equal(self.hidden, original))
        self.assertEqual(hooks.energy_report()["0"]["selected_tokens"], 2)

    def test_non_axis_projection_matches_independent_fp64_reference(self):
        generator = torch.Generator().manual_seed(9)
        q = torch.linalg.qr(torch.randn(6, 3, generator=generator)).Q
        with torch.inference_mode(), self.manager(q) as hooks:
            with hooks.positions(self.mask):
                actual = self.block(self.hidden)
        x = self.hidden[self.mask].double()
        reference = x - (x @ q.double()) @ q.double().T
        torch.testing.assert_close(actual[self.mask].double(), reference, rtol=2e-6, atol=2e-6)
        self.assertLess(float((actual[self.mask] @ q).abs().max()), 5e-6)

    def test_native_bfloat16_dtype_and_fp32_basis_are_preserved(self):
        q = torch.linalg.qr(torch.randn(6, 2, generator=torch.Generator().manual_seed(31))).Q
        hidden = self.hidden.to(torch.bfloat16)
        with torch.inference_mode(), self.manager(q) as hooks:
            with hooks.positions(self.mask):
                actual = self.block(hidden)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual[~self.mask], hidden[~self.mask]))
        report = hooks.energy_report()["0"]
        self.assertLess(report["remaining_subspace_energy_fp32"], 1e-9)
        self.assertGreater(report["remaining_subspace_energy_native"], report["remaining_subspace_energy_fp32"])

    def test_tuple_other_fields_keep_object_identity(self):
        block = Block(tuple_output=True)
        with torch.inference_mode(), self.manager(block=block) as hooks:
            with hooks.positions(self.mask):
                modified = block(self.hidden)
        self.assertIs(type(modified), tuple)
        self.assertIs(modified[1], block.cache)
        self.assertIs(modified[2], block.attention)

    def test_disabled_and_removed_hooks_recover_exact_original_object(self):
        with torch.inference_mode(), self.manager() as hooks:
            with hooks.positions(self.mask):
                changed = self.block(self.hidden)
            with hooks.disabled():
                disabled = self.block(self.hidden)
            self.assertIs(disabled, self.hidden)
            self.assertFalse(torch.equal(changed, self.hidden))
        self.assertIs(self.block(self.hidden), self.hidden)
        self.assertEqual(len(self.block._forward_hooks), 0)
        self.assertEqual(hooks.energy_report()["0"]["forward_calls"], 1)

    def test_exception_removes_hooks_and_recovers_baseline(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            with torch.inference_mode(), self.manager() as hooks:
                with hooks.positions(self.mask):
                    self.block(self.hidden)
                    raise RuntimeError("deliberate failure")
        self.assertEqual(len(self.block._forward_hooks), 0)
        self.assertIs(self.block(self.hidden), self.hidden)

    def test_missing_block_fails_closed_and_cleans_up(self):
        with self.assertRaisesRegex(RuntimeError, "Expected one invocation"):
            with torch.inference_mode(), self.manager() as hooks:
                with hooks.positions(self.mask):
                    pass
        self.assertEqual(len(self.block._forward_hooks), 0)

    def test_repeated_block_fails_closed(self):
        with torch.inference_mode(), self.manager() as hooks:
            with self.assertRaisesRegex(RuntimeError, "more than once"):
                with hooks.positions(self.mask):
                    self.block(self.hidden)
                    self.block(self.hidden)
            with hooks.positions(self.mask):
                self.block(self.hidden)

    def test_forward_without_position_context_fails(self):
        with torch.inference_mode(), self.manager():
            with self.assertRaisesRegex(RuntimeError, "explicit positions"):
                self.block(self.hidden)

    def test_grad_enabled_forward_fails(self):
        with self.manager() as hooks:
            with self.assertRaisesRegex(RuntimeError, "gradients disabled"):
                with hooks.positions(self.mask):
                    self.block(self.hidden)

    def test_training_model_rejected_at_install_and_forward(self):
        self.block.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            self.manager()
        self.block.eval()
        with torch.inference_mode(), self.manager() as hooks:
            self.block.train()
            with self.assertRaisesRegex(RuntimeError, "evaluation"):
                with hooks.positions(self.mask):
                    self.block(self.hidden)

    def test_trainable_layer_parameters_rejected(self):
        layer = torch.nn.Linear(6, 6).eval()
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.manager(block=layer)

    def test_basis_requires_float32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64, torch.int64):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                self.manager(self.q.to(dtype))

    def test_basis_invalid_shape_rank_values_and_orthonormality(self):
        cases = [torch.zeros(6), torch.zeros(5, 1), torch.zeros(6, 0), torch.zeros(6, 7),
                 self.q * 2, torch.zeros(6, 1), self.q + float("nan"), self.q + float("inf")]
        for q in cases:
            with self.subTest(shape=tuple(q.shape)), self.assertRaises(ValueError):
                self.manager(q)

    def test_basis_requires_grad_rejected(self):
        with self.assertRaisesRegex(ValueError, "gradients"):
            self.manager(self.q.clone().requires_grad_(True))

    def test_caller_basis_mutation_cannot_change_validated_projection(self):
        q = self.q.clone()
        with torch.inference_mode(), self.manager(q) as hooks:
            q.zero_()
            with hooks.positions(self.mask):
                actual = self.block(self.hidden)
        self.assertTrue(torch.equal(actual[self.mask][:, :2], torch.zeros((2, 2))))

    def test_mask_rejects_implicit_broadcast_and_nonboolean_dtype(self):
        with torch.inference_mode(), self.manager() as hooks:
            for mask in (torch.ones(3, dtype=torch.bool), torch.ones(2, 3, 1, dtype=torch.bool),
                         torch.zeros(0, 3, dtype=torch.bool), self.mask.int()):
                with self.subTest(shape=tuple(mask.shape)), self.assertRaises((TypeError, ValueError)):
                    with hooks.positions(mask):
                        self.block(self.hidden)
            with self.assertRaisesRegex(ValueError, "does not match"):
                with hooks.positions(torch.ones(1, 3, dtype=torch.bool)):
                    self.block(self.hidden)

    def test_mask_caller_mutation_does_not_change_scope(self):
        mask = self.mask.clone()
        with torch.inference_mode(), self.manager() as hooks:
            with hooks.positions(mask):
                mask.zero_()
                actual = self.block(self.hidden)
        self.assertFalse(torch.equal(actual, self.hidden))
        self.assertEqual(hooks.energy_report()["0"]["selected_tokens"], 2)

    def test_all_false_mask_is_exact_no_op_with_zero_energy(self):
        with torch.inference_mode(), self.manager() as hooks:
            with hooks.positions(torch.zeros_like(self.mask)):
                actual = self.block(self.hidden)
        self.assertTrue(torch.equal(actual, self.hidden))
        self.assertEqual(hooks.energy_report()["0"]["removed_energy_fp32"], 0)
        self.assertIsNone(hooks.energy_report()["0"]["removed_fraction"])

    def test_selected_nonfinite_hidden_rejected(self):
        hidden = self.hidden.clone()
        hidden[0, 2, 0] = float("nan")
        with torch.inference_mode(), self.manager() as hooks:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                with hooks.positions(self.mask):
                    self.block(hidden)

    def test_unexpected_output_shape_or_dtype_rejected(self):
        with torch.inference_mode(), self.manager() as hooks:
            for hidden in (self.hidden[:, :, :5], self.hidden[0], self.hidden.double(), self.hidden.long()):
                with self.subTest(shape=tuple(hidden.shape)), self.assertRaises((TypeError, ValueError)):
                    with hooks.positions(self.mask):
                        self.block(hidden)

    def test_incompatible_device_fails_without_implicit_transfer(self):
        with torch.inference_mode(), self.manager() as hooks:
            # A meta Q can never be accepted at construction; replacing it here
            # isolates the runtime device check without allocating any CUDA tensor.
            hooks.projections[0] = torch.empty(6, 2, device="meta", dtype=torch.float32)
            with self.assertRaisesRegex(ValueError, "same device"):
                with hooks.positions(self.mask):
                    self.block(self.hidden)

    def test_fp16_projection_overflow_is_rejected(self):
        q = torch.zeros(6, 1)
        q[:3, 0] = torch.tensor([-1.0, 1.0, 1.0]) / (3 ** 0.5)
        with torch.inference_mode(), self.manager(q) as hooks:
            with self.assertRaisesRegex(ValueError, "overflows"):
                with hooks.positions(self.mask):
                    self.block(torch.full_like(self.hidden, 65504, dtype=torch.float16))

    def test_unrecognized_output_container_is_rejected(self):
        with torch.inference_mode(), self.manager() as hooks:
            with self.assertRaisesRegex(TypeError, "Tensor or tuple"):
                with hooks.positions(self.mask):
                    self.block([self.hidden])

    def test_autocast_cannot_silently_reduce_projection_precision(self):
        with torch.inference_mode(), self.manager() as hooks, torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "Autocast"):
                with hooks.positions(self.mask):
                    self.block(self.hidden)

    def test_no_projection_baseline_installs_no_hooks(self):
        with torch.inference_mode(), ProjectionHooks([self.block], {}, expected_hidden_size=6) as hooks:
            with hooks.positions(self.mask):
                self.assertIs(self.block(self.hidden), self.hidden)
            self.assertEqual(len(self.block._forward_hooks), 0)
        self.assertEqual(hooks.energy_report(), {})

    def test_each_layer_uses_its_own_basis(self):
        second = Block()
        with torch.inference_mode(), ProjectionHooks(
            [self.block, second], {0: torch.eye(6)[:, :1], 1: torch.eye(6)[:, 2:3]}, expected_hidden_size=6,
        ) as hooks:
            with hooks.positions(self.mask):
                result = second(self.block(self.hidden))
        self.assertTrue(torch.equal(result[self.mask][:, [0, 2]], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(result[~self.mask], self.hidden[~self.mask]))
        self.assertEqual(set(hooks.energy_report()), {"0", "1"})

    def test_duplicate_module_indices_and_invalid_indices_rejected(self):
        with self.assertRaisesRegex(ValueError, "same module"):
            ProjectionHooks([self.block, self.block], {0: self.q, 1: self.q}, expected_hidden_size=6)
        for index in (-1, 1, True, 0.5, "0"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ProjectionHooks([self.block], {index: self.q}, expected_hidden_size=6)

    def test_nested_contexts_rejected(self):
        with torch.inference_mode(), self.manager() as hooks:
            with self.assertRaises(RuntimeError):
                hooks.__enter__()
            with hooks.positions(self.mask):
                with self.assertRaises(RuntimeError):
                    with hooks.positions(self.mask):
                        pass
                with self.assertRaises(RuntimeError):
                    with hooks.disabled():
                        pass
                with self.assertRaises(RuntimeError):
                    hooks.energy_report()
                self.block(self.hidden)
            with hooks.disabled():
                with self.assertRaises(RuntimeError):
                    with hooks.disabled():
                        pass

    def test_energy_scopes_counts_and_reset(self):
        with torch.inference_mode(), self.manager() as hooks:
            for scope in ("prefill", "decode"):
                with hooks.positions(self.mask, scope=scope):
                    self.block(self.hidden)
        expected_removed = float(self.hidden[self.mask][:, :2].square().sum())
        report = hooks.energy_report(reset=True)["0"]
        self.assertEqual(report["forward_calls"], 2)
        self.assertEqual(report["selected_tokens"], 4)
        self.assertEqual(report["removed_energy_fp32"], 2 * expected_removed)
        self.assertEqual(set(report["scopes"]), {"prefill", "decode"})
        self.assertEqual(report["scopes"]["prefill"]["forward_calls"], 1)
        self.assertEqual(hooks.energy_report()["0"]["forward_calls"], 0)
        self.assertEqual(hooks.energy_report()["0"]["scopes"], {})

    def test_default_dimension_matches_saved_qwen_activations(self):
        q = torch.zeros(2560, 1)
        q[0, 0] = 1
        with torch.inference_mode(), ProjectionHooks([self.block], {0: q}) as hooks:
            with hooks.positions(torch.ones(1, 1, dtype=torch.bool)):
                out = self.block(torch.ones(1, 1, 2560))
        self.assertEqual(float(out[0, 0, 0]), 0)


class PositionTests(unittest.TestCase):
    def test_last_valid_prompt_position_with_left_and_right_padding(self):
        mask = torch.tensor([[1, 1, 1, 0], [0, 0, 1, 1]])
        expected = torch.tensor([[False, False, True, False], [False, False, False, True]])
        self.assertTrue(torch.equal(last_token_mask(mask), expected))
        self.assertTrue(torch.equal(generation_position_mask(mask, current_sequence_length=4,
                                                             use_cache_decode=False), expected))

    def test_cached_decode_mask_is_one_token_despite_full_attention_width(self):
        attention = torch.ones(2, 19, dtype=torch.long)
        result = generation_position_mask(attention, current_sequence_length=1, use_cache_decode=True)
        self.assertEqual(tuple(result.shape), (2, 1))
        self.assertTrue(bool(result.all()))

    def test_cached_multitoken_input_or_padded_newest_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            generation_position_mask(torch.ones(1, 3, dtype=torch.long), current_sequence_length=2, use_cache_decode=True)
        with self.assertRaisesRegex(ValueError, "newest"):
            generation_position_mask(torch.tensor([[1, 1, 0]]), current_sequence_length=1, use_cache_decode=True)

    def test_prefill_width_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Prefill"):
            generation_position_mask(torch.ones(1, 3, dtype=torch.long), current_sequence_length=1, use_cache_decode=False)

    def test_attention_mask_invalid_inputs_are_rejected(self):
        for mask in (torch.zeros(1, 2, dtype=torch.bool), torch.tensor([[1, 2]]), torch.tensor([[1, -1]]),
                     torch.ones(3), torch.ones(1, 2), torch.zeros(0, 2, dtype=torch.bool)):
            with self.subTest(mask=mask), self.assertRaises((TypeError, ValueError)):
                last_token_mask(mask)


class TinyQwenIntegrationTests(unittest.TestCase):
    """Exercise the real decoder/cache inference path using fresh tiny random weights."""

    def test_real_qwen_prefill_decode_projection_and_baseline_recovery(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(402)
        config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             max_position_embeddings=64, attention_dropout=0.0, use_cache=True)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        ids = torch.tensor([[2, 5, 7, 11]])
        attention = torch.ones_like(ids)
        q = torch.eye(16)[:, :2]
        with torch.inference_mode():
            baseline = model(input_ids=ids, attention_mask=attention, use_cache=True)
            with ProjectionHooks(model.model.layers, {0: q}, expected_hidden_size=16) as hooks:
                prefill_mask = generation_position_mask(attention, current_sequence_length=4, use_cache_decode=False)
                with hooks.positions(prefill_mask, scope="prefill"):
                    projected = model(input_ids=ids, attention_mask=attention, use_cache=True)
                self.assertTrue(torch.equal(projected.logits[:, :-1], baseline.logits[:, :-1]))
                self.assertFalse(torch.equal(projected.logits[:, -1], baseline.logits[:, -1]))
                next_id = projected.logits[:, -1].argmax(dim=-1, keepdim=True)
                longer_attention = torch.ones(1, 5, dtype=torch.long)
                decode_mask = generation_position_mask(longer_attention, current_sequence_length=1, use_cache_decode=True)
                with hooks.positions(decode_mask, scope="decode"):
                    decoded = model(input_ids=next_id, attention_mask=longer_attention,
                                    past_key_values=projected.past_key_values, use_cache=True)
                self.assertEqual(tuple(decoded.logits.shape), (1, 1, 31))
                with hooks.disabled():
                    disabled = model(input_ids=ids, attention_mask=attention, use_cache=False)
                self.assertTrue(torch.equal(disabled.logits, baseline.logits))
            recovered = model(input_ids=ids, attention_mask=attention, use_cache=False)
        self.assertTrue(torch.equal(recovered.logits, baseline.logits))
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in model.model.layers))
        self.assertEqual(hooks.energy_report()["0"]["selected_tokens"], 2)
        self.assertEqual(hooks.energy_report()["0"]["scopes"]["decode"]["selected_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
