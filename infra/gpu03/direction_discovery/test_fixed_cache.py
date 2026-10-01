"""CPU qualification of fixed-shape native capture, serialization and guards."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import fixed_cache as f


def row(index=0, completion=None, outcome=None):
    prompt = [2, 3, 5]
    completion = list(completion or [7, 11, 13, 17, 19])
    ids = prompt + completion
    return {"record_id": "record-" + str(index), "record_index": index,
            "problem_id_key": "1", "problem_split": "direction_fit",
            "outcome_presence_class": outcome or sorted(f.CORE_CLASSES)[index % 3],
            "prompt_token_count": len(prompt), "completion_token_count": len(completion),
            "prompt_token_ids": prompt, "completion_token_ids": completion, "input_ids": ids,
            "input_ids_sha256": f.ids_hash(ids), "completion_sha256": "c" * 64,
            "checkpoint_sha256": "a" * 64, "model_revision": "revision",
            "selected_token_positions": list(range(len(prompt), len(ids))), "selected_token_mask": [True] * len(completion),
            "region_mask_completion_positions": {"evaluator__transition": [0, 1], "solution__end_of_code": [len(completion) - 1]}}


class InputTests(unittest.TestCase):
    def test_explicit_larger_shape_preserves_original_length_guards(self):
        r = row(completion=[7] * 1536)
        r['prompt_token_ids'] = [2] * 1135
        r['prompt_token_count'] = 1135
        r['input_ids'] = r['prompt_token_ids'] + r['completion_token_ids']
        r['input_ids_sha256'] = f.ids_hash(r['input_ids'])
        r['selected_token_positions'] = list(range(1135, 2671))
        with self.assertRaisesRegex(RuntimeError, 'fixed padded shape'):
            f.validate_row(r)
        f.validate_row(r, padded_length=2688)
        for value in (True, 2688.0, 2175, 3073):
            with self.assertRaisesRegex(RuntimeError, 'padded length'):
                f.validate_row(r, padded_length=value)

    def test_full_core_triplets_preserve_split_and_auxiliary_null_class(self):
        records = [row(i) for i in range(3)]
        self.assertEqual(f.validate_groups(records[::-1], "core"), records)
        with self.assertRaisesRegex(RuntimeError, "complete matched triplets"):
            f.validate_groups(records[:2], "core")
        records[1]["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(RuntimeError, "split assignments"):
            f.validate_groups(records, "core")
        aux = row()
        aux["outcome_presence_class"] = None
        self.assertEqual(f.validate_groups([aux], "auxiliary"), [aux])

    def test_original_tokens_hash_and_regions_fail_closed(self):
        r = row()
        r["input_ids"] = [1] + r["input_ids"][1:]
        with self.assertRaisesRegex(RuntimeError, "sequence changed"):
            f.validate_row(r)
        r = row()
        r["input_ids_sha256"] = "bad"
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            f.validate_row(r)
        r = row()
        r["region_mask_completion_positions"]["invalid"] = [99]
        with self.assertRaisesRegex(RuntimeError, "region mask"):
            f.validate_row(r)

    def test_review_identity_no_circular_self_hash_and_refuses_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"immutable":true}\n')
            self.assertEqual(f.resolve_manifest_digest({"manifest_path": str(path)}), f.sha256(path))
            with self.assertRaisesRegex(RuntimeError, "differs"):
                f.resolve_manifest_digest({"manifest_path": str(path), "manifest_sha256": "a" * 64})
            with self.assertRaisesRegex(RuntimeError, "identity"):
                f.resolve_manifest_digest({})


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "real native BF16 serializer tests require gpu-04 CPU runtime")
class NativeTests(unittest.TestCase):
    def test_explicit2688_native_capture_save_and_integrated_prefix_audits(self):
        r = row(completion=[7] * 1536)
        r.update(prompt_token_ids=[2] * 1135, prompt_token_count=1135)
        r['input_ids'] = r['prompt_token_ids'] + r['completion_token_ids']
        r['input_ids_sha256'] = f.ids_hash(r['input_ids'])
        r['selected_token_positions'] = list(range(1135, 2671))
        decoder = self.decoder()
        values, inputs, _ = f.capture_fixed(decoder, list(decoder.layers), r, hidden_size=6, padded_length=2688)
        self.assertEqual(tuple(values.shape), (36, 1537, 6))
        self.assertEqual(inputs['attention_mask'].tolist(), [1] * 2671 + [0] * 17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'h0.safetensors'
            f.save_native(path, r, 'h0', 'd' * 64, values, inputs, padded_length=2688)
            result = f.qualify_model(decoder, list(decoder.layers), r, values, 'h0', 'd' * 64,
                                     Path(directory) / 'qualification', vocab_size=256, hidden_size=6, padded_length=2688)
            self.assertTrue(result['repeat']['bitwise_equal'])
            self.assertTrue(result['future_causality']['bitwise_equal'])
            self.assertEqual(result['future_causality']['fixed_padded_length'], 2688)
            from safetensors import safe_open
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                self.assertEqual(handle.metadata()['padded_sequence_length'], '2688')
        self.assertEqual(f.PADDED_LENGTH, 2176)

    def setUp(self):
        import torch
        self.torch = torch
        self.saved = {"deterministic": torch.are_deterministic_algorithms_enabled(),
                      "warn": torch.is_deterministic_algorithms_warn_only_enabled(),
                      "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "benchmark": torch.backends.cudnn.benchmark,
                      "cudnn_deterministic": torch.backends.cudnn.deterministic}
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.hidden = patch.object(f, "HIDDEN", 6)
        self.hidden.start()
        torch.set_num_threads(1)

    def tearDown(self):
        t, s = self.torch, self.saved
        t.use_deterministic_algorithms(s["deterministic"], warn_only=s["warn"])
        t.backends.cuda.matmul.allow_tf32 = s["matmul_tf32"]
        t.backends.cudnn.allow_tf32 = s["cudnn_tf32"]
        t.backends.cudnn.benchmark = s["benchmark"]
        t.backends.cudnn.deterministic = s["cudnn_deterministic"]
        self.hidden.stop()

    def decoder(self, shift=0, noncausal=False):
        torch = self.torch
        class Block(torch.nn.Module):
            def forward(self, hidden):
                return (hidden + torch.tensor(0.125, dtype=torch.bfloat16),)
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
                self.layers = torch.nn.ModuleList([Block() for _ in range(36)])
                self.seen_inputs = []
            def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
                assert use_cache is False and return_dict is True and not torch.is_grad_enabled()
                self.seen_inputs.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone(),
                                         "position_ids": position_ids.clone(), "flags": f.math_flags()})
                source = (input_ids % 64) * attention_mask
                values = source.cumsum(1).float() / 32
                if noncausal:
                    values = values + source.sum(1, keepdim=True).float()
                hidden = (values[:, :, None] + torch.arange(6)[None, None, :] / 8 + shift).to(torch.bfloat16)
                for layer in self.layers:
                    hidden = layer(hidden)[0]
                return {"last_hidden_state": hidden}
        return Decoder().eval()

    def capture(self, decoder, r):
        return f.capture_fixed(decoder, list(decoder.layers), r, hidden_size=6)

    def test_all_36_native_blocks_masked_right_padding_and_original_offsets(self):
        torch = self.torch
        r, decoder = row(), self.decoder()
        values, inputs, flags = self.capture(decoder, r)
        self.assertEqual(values.shape, (36, 6, 6))
        self.assertEqual(values.dtype, torch.bfloat16)
        self.assertEqual(flags, {"math": True, "flash": False, "memory_efficient": False, "cudnn": False})
        self.assertEqual(inputs["input_ids"].tolist()[:8], r["input_ids"])
        self.assertEqual(inputs["input_ids"].tolist()[8:], [151643] * (2176 - 8))
        self.assertEqual(inputs["attention_mask"].tolist(), [1] * 8 + [0] * (2176 - 8))
        self.assertEqual(inputs["position_ids"].tolist(), list(range(2176)))
        expected_prompt = torch.tensor(sum(r["prompt_token_ids"]) / 32 + .125, dtype=torch.bfloat16)
        self.assertEqual(values[0, 0, 0], expected_prompt)
        expected_first = torch.tensor(sum(r["input_ids"][:4]) / 32 + .125, dtype=torch.bfloat16)
        self.assertEqual(values[0, 1, 0], expected_first)
        self.assertFalse(any(layer._forward_hooks for layer in decoder.layers))

    def test_different_valid_lengths_same_prefix_and_valid_future_qualification(self):
        torch = self.torch
        decoder = self.decoder()
        r = row()
        short = row(1, [7, 11, 23])
        a, _, _ = self.capture(decoder, r)
        b, _, _ = self.capture(decoder, short)
        self.assertTrue(torch.equal(a[:, :3], b[:, :3]))
        with tempfile.TemporaryDirectory() as directory:
            report = f.qualify_model(decoder, list(decoder.layers), r, a, "h0", "d" * 64,
                                     Path(directory) / "qualification", vocab_size=151936, hidden_size=6)
            self.assertTrue(report["repeat"]["bitwise_equal"])
            self.assertTrue(report["future_causality"]["bitwise_equal"])
            self.assertTrue(report["future_causality"]["artifact"]["native_readback_bitwise_equal"])

    def test_future_dependency_fails_after_preserving_counterfactual_and_audit(self):
        decoder, r = self.decoder(noncausal=True), row()
        a, _, _ = self.capture(decoder, r)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qualification"
            with self.assertRaisesRegex(RuntimeError, "future token perturbation changed"):
                f.qualify_model(decoder, list(decoder.layers), r, a, "h0", "d" * 64,
                                output, vocab_size=151936, hidden_size=6)
            self.assertTrue((output / "future_perturbed.safetensors").is_file())
            self.assertFalse(json.loads((output / "future_causality_audit.json").read_text())["bitwise_equal"])

    def test_unreviewed_policy_and_training_mode_rejected(self):
        decoder = self.decoder()
        decoder.train()
        with self.assertRaisesRegex(RuntimeError, "evaluation-mode"):
            self.capture(decoder, row())
        self.torch.backends.cuda.matmul.allow_tf32 = True
        with self.assertRaisesRegex(RuntimeError, "runtime policy"):
            self.capture(self.decoder(), row())

    def test_native_serializer_delta_float64_reference_and_rawreader_completion_axis(self):
        import candidates
        import numpy as np
        from safetensors import safe_open
        torch = self.torch
        r = row()
        digest = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {"record_id": r["record_id"], "record_index": 0, "verified": True, "models": {}}
            manifest, originals = {"files": {}}, {}
            for kind, shift in (("h0", 0), ("h60", .3)):
                original, inputs, _ = self.capture(self.decoder(shift), r)
                originals[kind] = original
                path = root / (kind + ".safetensors")
                info = f.save_native(path, r, kind, digest, original, inputs)
                entry["models"][kind] = {**info, "tensor_path": path.name}
                manifest["files"][path.name] = {"sha256": info["sha256"], "size_bytes": info["size_bytes"]}
            hashes = {k: entry["models"][k]["sha256"] for k in originals}
            report = f.save_delta(root / "delta.safetensors", r, digest, root / "h0.safetensors", root / "h60.safetensors", hashes)
            self.assertTrue(report["float64_reference_rounded_to_fp32_exact"])
            with safe_open(str(root / "delta.safetensors"), framework="pt", device="cpu") as out:
                self.assertTrue(torch.equal(out.get_tensor("delta_h"), (originals["h60"][:, 1:].double() - originals["h0"][:, 1:].double()).float()))
                self.assertEqual(out.get_tensor("prompt_final_sequence_position").item(), 2)
                self.assertEqual(out.get_tensor("input_ids").tolist(), r["input_ids"])
            reader = candidates.RawReader(root, [r], [entry], manifest, digest, 36, 6)
            for kind in originals:
                np.testing.assert_array_equal(reader.read(r, kind, 35, [0, 4]), originals[kind][35, [1, 5]].float().numpy())
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                f.save_delta(root / "bad.safetensors", r, digest, root / "h0.safetensors", root / "h60.safetensors", {"h0": "bad", "h60": hashes["h60"]})
            self.assertFalse((root / "bad.safetensors").exists())

    def test_group_prefix_failure_preserves_report_and_original_native_file(self):
        r, peer = row(), row(1, [7, 11, 23])
        original, inputs, _ = self.capture(self.decoder(), r)
        other, _, _ = self.capture(self.decoder(shift=1), peer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "h0.safetensors"
            f.save_native(path, r, "h0", "d" * 64, original, inputs)
            with self.assertRaisesRegex(RuntimeError, "identical original prefix"):
                f.audit_group_prefixes(peer, other, [(r, path)], "h0", root / "prefix.jsonl")
            self.assertTrue(path.exists())
            self.assertFalse(json.loads((root / "prefix.jsonl").read_text())["prompt_final"]["bitwise_equal"])

    def test_complete_extract_success_both_native_models_qualification_and_fp32(self):
        _, legacy = f.dependencies()
        records = [row(0), row(1, [7, 11, 23]), row(2, [7, 11, 29, 31])]
        models = []
        def load(_task, with_adapter):
            decoder = self.decoder(shift=.3 if with_adapter else 0)
            decoder.config = SimpleNamespace(vocab_size=151936)
            models.append(decoder)
            report = {"with_adapter": with_adapter, "active_adapters": ["default"] if with_adapter else [],
                      "lora_parameter_tensors": int(with_adapter), "nonzero_lora_parameter_tensors": int(with_adapter),
                      "lora_parameter_elements": int(with_adapter), "base_model_class": "CPUFixture"}
            return decoder, decoder, list(decoder.layers), report
        released = {"allocated_bytes": legacy.CUBLAS_DETERMINISTIC_WORKSPACE_BYTES,
                    "reserved_bytes": legacy.CUBLAS_DETERMINISTIC_WORKSPACE_BYTES}
        with tempfile.TemporaryDirectory() as directory, patch.object(legacy, "_load_decoder", side_effect=load), \
                patch.object(legacy, "_release_cuda", return_value=released) as release:
            root = Path(directory)
            report = f.extract({"padded_sequence_length": 2176, "pad_token_id": 151643,
                                "manifest_sha256": "d" * 64, "cache_role": "core", "deadline_seconds": 60}, records, root)
            self.assertEqual(report["status"], "succeeded")
            self.assertEqual(report["identical_prefix_comparisons"], 6)
            self.assertEqual(release.call_count, 2)
            self.assertFalse(report["cross_package_prefix_audit_required"])
            entries = [json.loads(s) for s in (root / "workerindex.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 3)
            self.assertTrue(all(e["verified"] and set(e["models"]) == {"h0", "h60"} and e["delta"]["dtype"] == "float32" for e in entries))
            self.assertTrue(all((root / e["models"]["h0"]["tensor_path"]).stat().st_mode & 0o222 == 0 for e in entries))
            # Each fake model sees the first original twice, a counterfactual,
            # and the two remaining different-length original completions.
            self.assertEqual([len(m.seen_inputs) for m in models], [5, 5])


if __name__ == "__main__":
    unittest.main()
