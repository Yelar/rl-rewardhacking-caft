"""Authored-array regressions; CPU only, no saved scientific activations."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import fit_worker as w


def row(i=0, *, n=20, b=5, end=10, correct=False, harmful=False, problem="p"):
    window = [t if 0 <= t < n and (t < b or t < end) else None for t in range(b-4, b+12)]
    return {"record_id": "r" + str(i), "record_index": i, "problem_id_key": problem,
            "problem_split": "direction_fit", "ground_truth_correctness": correct,
            "is_test_modification_harmful": harmful, "evaluator_present": True,
            "strict_reward_hack": harmful and not correct,
            "completion_token_count": n,
            "regions": {"evaluator": {"first_executable_completion_token": b,
                "logit_source_completion_token": b-1, "end_completion_token_exclusive": end,
                "window_completion_positions": {"transition": window}}},
            "region_mask_completion_positions": {"evaluator__transition": [t for t in window if t is not None]}}


class ReductionTests(unittest.TestCase):
    def test_native_subtraction_precedes_pooling_and_saved_window_clips(self):
        r = row(n=20, b=5, end=10)
        left = np.full((20, 3), 2**24, dtype=np.float32)
        right = left.copy()
        right[1, 0] += 2
        right[4, 0] += 2
        right[10:, :] += 1000  # Outside the evaluator; must not enter transition.
        result = w.reduce_record(r, left, right, left[0], right[0])
        self.assertEqual(w.positions(r)["transition"], list(range(1, 10)))
        expected = np.subtract(right[1:10], left[1:10], dtype=np.float32).mean(0, dtype=np.float32)
        np.testing.assert_array_equal(result["anchors"]["transition"]["delta"], expected)
        np.testing.assert_array_equal(result["anchors"]["body_predictor"]["delta"], [2, 0, 0])
        self.assertFalse(np.array_equal(expected, right[1:10].mean(0) - left[1:10].mean(0)))

    def test_center_predictor_shift_includes_prompt_and_excludes_final_response_state(self):
        r = row(n=5, b=1, end=4)
        right = np.arange(10, dtype=np.float32).reshape(5, 2)
        prompt = np.array([80, 100], dtype=np.float32)
        result = w.reduce_record(r, np.zeros_like(right), right, np.zeros(2, np.float32), prompt)
        np.testing.assert_array_equal(result["response_h60_mean"],
                                      np.vstack([prompt, right[:-1]]).mean(0, dtype=np.float64))
        self.assertEqual(result["predictor_count"], 5)
        mutated = right.copy()
        mutated[-1] = 1e9
        again = w.reduce_record(r, np.zeros_like(right), mutated, np.zeros(2, np.float32), prompt)
        np.testing.assert_array_equal(result["response_h60_mean"], again["response_h60_mean"])

    def test_absent_evaluator_or_mask_mismatch_is_not_fabricated(self):
        r = row()
        r["regions"]["evaluator"] = None
        with self.assertRaisesRegex(ValueError, "fabricated"):
            w.positions(r)
        r = row()
        r["region_mask_completion_positions"]["evaluator__transition"].append(10)
        with self.assertRaisesRegex(ValueError, "disagree"):
            w.positions(r)
        r = row()
        r["regions"]["evaluator"]["logit_source_completion_token"] += 1
        with self.assertRaisesRegex(ValueError, "b-1"):
            w.positions(r)

    def test_nonfinite_wrong_dtype_and_missing_prompt_are_rejected(self):
        r = row()
        x = np.zeros((20, 3), np.float32)
        for bad in (np.full_like(x, np.nan), x.astype(np.float64)):
            with self.assertRaises(ValueError):
                w.reduce_record(r, x, bad, x[0], x[0])
        with self.assertRaisesRegex(ValueError, "prompt"):
            w.reduce_record(r, x, x, x[:1], x[0])


class CenterTests(unittest.TestCase):
    def test_equal_stratum_problem_response_weighting_ignores_harmful_and_length(self):
        rows = [row(0, correct=False, problem="a"), row(1, correct=False, problem="a"),
                row(2, correct=False, problem="b"), row(3, correct=True, problem="c"),
                row(4, correct=True, harmful=True, problem="c")]
        means = np.array([[0], [4], [10], [20], [10000]], dtype=np.float64)
        center, info = w.response_center(rows, means)
        # Incorrect = ((0+4)/2 +10)/2 =6; correct=20; equal strata=13.
        np.testing.assert_array_equal(center, [13])
        self.assertEqual(info["support"]["false"]["problems"], 2)
        self.assertNotIn("r4", info["record_ids"])
        rows[0]["completion_token_count"] = 1000
        again, _ = w.response_center(rows, means)
        np.testing.assert_array_equal(center, again)

    def test_missing_stratum_explicitly_unsupported_and_heldout_rejected(self):
        rows = [row()]
        center, info = w.response_center(rows, np.zeros((1, 3)))
        self.assertIsNone(center)
        self.assertEqual(info["status"], "unsupported")
        rows[0]["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(ValueError, "held-out"):
            w.response_center(rows, np.zeros((1, 3)))

    def test_benign_modification_strict_hack_and_unknown_are_not_clean(self):
        rows = [row(0), row(1, correct=True), dict(row(2), strict_reward_hack=True)]
        means = np.array([[2], [4], [99999]], dtype=np.float64)
        center, info = w.response_center(rows, means)
        np.testing.assert_array_equal(center, [3])
        self.assertEqual(info["record_ids"], ["r0", "r1"])
        rows[2].pop("strict_reward_hack")
        with self.assertRaisesRegex(ValueError, "Explicit strict-RH"):
            w.response_center(rows, means)


class TaskTests(unittest.TestCase):
    def build(self, directory):
        root = Path(directory)
        rows = root / "prepared_rows.jsonl"
        rows.write_text(json.dumps(row()) + "\n")
        index = root / "workerindex.jsonl"
        index.write_text(json.dumps({"record_id": "r0", "record_index": 0, "verified": True,
            "models": {"h0": {"tensor_path": "h0/r0.safetensors"},
                       "h60": {"tensor_path": "h60/r0.safetensors"}},
            "delta": {"tensor_path": "delta/r0.safetensors"}}) + "\n")
        receipt = root / "extraction_complete.json"
        receipt.write_text(json.dumps({"status": "succeeded", "gpu_workers_released": True}))
        task = {"schema": "task_specific_cpu_fitting_v1", "shard_index": 3, "shard_count": 8,
                "deadline_seconds": 3600, "prepared_rows": str(rows), "worker_indexes": [str(index)],
                "extraction_complete": str(receipt), "output": str(root / "output"),
                "bound_files": {str(p): w.binding(p) for p in (rows, index, receipt)}}
        return task

    def read(self, root, task):
        path = Path(root) / "task.json"
        path.write_text(json.dumps(task))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            return w.read_task(path, w.sha256(path))

    def test_successful_binding_and_exact_index_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            task = self.build(root)
            actual, rows, entries, cfg = self.read(root, task)
            self.assertEqual(actual["shard_index"], 3)
            self.assertEqual(list(entries), [rows[0]["record_id"]])
            self.assertEqual(entries["r0"]["models"]["h0"]["absolute_path"],
                             str(Path(root) / "h0/r0.safetensors"))

    def test_input_mutation_or_unreleased_gpu_fails_before_fitting(self):
        with tempfile.TemporaryDirectory() as root:
            task = self.build(root)
            receipt = Path(task["extraction_complete"])
            receipt.write_text(json.dumps({"status": "succeeded", "gpu_workers_released": False}))
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                self.read(root, task)
            task["bound_files"][str(receipt)] = w.binding(receipt)
            with self.assertRaisesRegex(ValueError, "GPU release"):
                self.read(root, task)

    def test_duplicate_index_and_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            task = self.build(root)
            task["worker_indexes"] *= 2
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.read(root, task)
            task["worker_indexes"] = task["worker_indexes"][:1]
            p = Path(task["worker_indexes"][0])
            entry = json.loads(p.read_text())
            entry["models"]["h0"]["tensor_path"] = "../wrong.safetensors"
            p.write_text(json.dumps(entry) + "\n")
            task["bound_files"][str(p)] = w.binding(p)
            with self.assertRaisesRegex(ValueError, "within"):
                self.read(root, task)


class NativeCacheTests(unittest.TestCase):
    """These tests require the pinned Torch+safetensors CPU runtime."""

    def cache(self, root):
        import torch
        from safetensors.torch import save_file
        r = row(n=6, b=2, end=6)
        r.update(input_ids=[1, 2, 3, 4, 5, 6, 7, 8], selected_token_positions=list(range(2, 8)),
                 input_ids_sha256="1" * 64, completion_sha256="2" * 64,
                 model_revision="pinned-revision", checkpoint_sha256="3" * 64)
        base = torch.arange(w.LAYERS * 6 * w.HIDDEN, dtype=torch.float32).reshape(w.LAYERS, 6, w.HIDDEN)
        left = (base.remainder(100) / 64).to(torch.bfloat16)
        right = (left.float() + .25).to(torch.bfloat16)
        p0, p60 = left[:, 0].clone(), (right[:, 0].float() + 2).to(torch.bfloat16)
        mask = torch.zeros(6, dtype=torch.bool)
        mask[w.positions(r)["transition"]] = True
        common = {"input_ids": torch.tensor(r["input_ids"], dtype=torch.int32),
                  "sequence_positions": torch.tensor(r["selected_token_positions"], dtype=torch.int32),
                  "mask__evaluator__transition": mask}
        entry = {"models": {}}
        for kind, value, prompt in (("h0", left, p0), ("h60", right, p60),
                                   ("delta", right.float()-left.float(), p60.float()-p0.float())):
            path = Path(root) / (kind + ".safetensors")
            key, pk = ("delta_h", "prompt_final_delta") if kind == "delta" else (kind, "prompt_final")
            metadata = {k: str(r[k]) for k in ("record_id", "record_index", "input_ids_sha256",
                                              "completion_sha256", "model_revision")}
            metadata.update(adapter_sha256="none" if kind == "h0" else r["checkpoint_sha256"],
                site="decoder_layer_forward_output_before_final_norm", padded_sequence_length="2176",
                attention_backend="torch_sdpa_MATH_only", prompt_final_saved="true")
            save_file({key: value, pk: prompt, **common}, str(path), metadata=metadata)
            info = {"absolute_path": str(path), **w.binding(path)}
            if kind == "delta":
                entry[kind] = info
            else:
                entry["models"][kind] = info
        return r, entry, left, right, p0, p60

    def test_actual_native_slice_and_tensor_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            r, entry, left, right, p0, p60 = self.cache(root)
            verified = {}
            actual = w.read_layer(r, entry, 21, verified)
            expected = w.reduce_record(r, left[21].float().numpy(), right[21].float().numpy(),
                                      p0[21].float().numpy(), p60[21].float().numpy())
            for rep in w.fitting.REPRESENTATIONS:
                for kind in w.fitting.KINDS:
                    np.testing.assert_array_equal(actual["anchors"][rep][kind], expected["anchors"][rep][kind])
            np.testing.assert_array_equal(actual["response_h60_mean"], expected["response_h60_mean"])
            self.assertEqual(len(verified), 3)
            w.read_layer(r, entry, 22, verified)
            self.assertEqual(len(verified), 3)
            tensor_path = Path(root) / "result.safetensors"
            w.write_tensors(tensor_path, {"center": actual["response_h60_mean"],
                                         "unsupported_oof": np.array([np.nan])}, {"task_sha256": "a" * 64})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                w.write_tensors(tensor_path, {}, {})

    def test_changed_cache_and_mismatched_delta_rejected(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as root:
            r, entry, *_ = self.cache(root)
            path = Path(entry["delta"]["absolute_path"])
            values = load_file(str(path))
            with safe_open(str(path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            values["delta_h"][21, 1, 0] += 1
            save_file(values, str(path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                w.read_layer(r, entry, 21, {})
            # Even a newly bound malformed delta cannot pass the arithmetic audit.
            entry["delta"].update(w.binding(path))
            with self.assertRaisesRegex(ValueError, "delta disagrees"):
                w.read_layer(r, entry, 21, {})

    def test_independent_hash_receipt_reuses_matching_identity_and_rehashes_changes(self):
        with tempfile.TemporaryDirectory() as root:
            r, entry, *_ = self.cache(root)
            entries = {r["record_id"]: entry}
            receipt = w.build_cache_verification(entries)
            path = Path(root) / "verification.json"
            path.write_text(json.dumps(receipt))
            task = {"cache_verification": str(path), "bound_files": {str(path): w.binding(path)}}
            with patch.object(w, "verify_file", side_effect=AssertionError("No duplicate hashing")):
                identities = w.verified_identities(task, entries)
                w.read_layer(r, entry, 21, identities)
            altered = Path(entry["models"]["h0"]["absolute_path"])
            os.utime(altered, ns=(altered.stat().st_atime_ns, altered.stat().st_mtime_ns + 1000000))
            with patch.object(w, "verify_file", wraps=w.verify_file) as verify:
                w.verified_identities(task, entries)
                self.assertEqual(verify.call_count, 1)
            receipt["files"][str(altered)]["sha256"] = "0" * 64
            path.write_text(json.dumps(receipt))
            task["bound_files"][str(path)] = w.binding(path)
            with self.assertRaisesRegex(ValueError, "binding differs"):
                w.verified_identities(task, entries)


if __name__ == "__main__":
    unittest.main()
