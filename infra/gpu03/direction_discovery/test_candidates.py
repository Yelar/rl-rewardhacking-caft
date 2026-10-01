"""Synthetic numerical and failure-path tests; no model/GPU/cache required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np

try:
    from . import candidates as c
except ImportError:
    import candidates as c


def records(problems=6):
    result = []
    for p in range(problems):
        for label in c.CLASSES:
            result.append({"record_id": f"{p}-{label}", "record_index": len(result),
                           "problem_id_key": str(p), "problem_split": "direction_fit",
                           "prompt_sha256": f"prompt-{p}", "outcome_presence_class": label,
                           "is_test_modification_harmful": label == c.HARMFUL,
                           "ground_truth_correctness": label == c.CORRECT,
                           "is_reward_hack_strict": label == c.HARMFUL,
                           "classification_disagreements": [], "classification_error": None,
                           "completion_token_count": 50, "completion_token_ids": list(range(50)),
                           "prompt_token_count": 25, "test_modification": "Harmful" if label == c.HARMFUL else "Innocent"})
    return result


def data_fixture(problems=6, hidden=12, token_counts=None):
    rows = records(problems)
    rng = np.random.default_rng(716)
    positions = [list(range(10, 10 + (token_counts[i] if token_counts else 8))) for i in range(len(rows))]
    offsets = np.cumsum([0] + [len(p) for p in positions])
    h0 = rng.normal(size=(offsets[-1], hidden)).astype(np.float32)
    h60 = h0.copy()
    for i, row in enumerate(rows):
        p = int(row["problem_id_key"])
        a, b = offsets[i:i + 2]
        h0[a:b, 0] += 1 if row["is_test_modification_harmful"] else 0
        h60[a:b] += rng.normal(size=(b - a, hidden)).astype(np.float32) * .2
        h60[a:b, 0] += 2 + p * .01 if row["is_test_modification_harmful"] else 0
        h60[a:b, 1] += 1 if row["ground_truth_correctness"] else 0
    return c.WindowData(rows, positions, h0, h60, offsets,
                        np.repeat(np.arange(len(rows)), np.diff(offsets)),
                        np.asarray([p for ps in positions for p in ps]))


class InputTests(unittest.TestCase):
    def test_blas_preflight_accepts_healthy_numerics(self):
        self.assertTrue(c.numerical_preflight()["passed"])

    def test_blas_preflight_fails_closed_on_bad_reference(self):
        with patch.object(c.np, "einsum", return_value=np.zeros((17, 4))):
            with self.assertRaisesRegex(ValueError, "BLAS numerical preflight failed"):
                c.numerical_preflight()

    def test_exclusion_drops_complete_fitting_problem_preserving_records(self):
        rows = records()
        original = copy.deepcopy(rows)
        fit, report = c.validate_records(rows, {"excluded_problem_ids": ["2"], "disputed_record_ids": [rows[6]["record_id"]]})
        self.assertEqual(len(fit), 15)
        self.assertEqual(report["fitting_problems"], 5)
        self.assertEqual(rows, original)
        self.assertEqual(len(report["excluded_record_ids"]), 3)

    def test_exclusions_must_remain_fitting_only(self):
        rows = records()
        for r in rows[:3]:
            r["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(ValueError, "fitting problems only"):
            c.validate_records(rows, {"excluded_problem_ids": ["0"], "disputed_record_ids": []})

    def test_disputed_record_requires_its_whole_problem_exclusion(self):
        with self.assertRaisesRegex(ValueError, "was not excluded"):
            c.validate_records(records(), {"excluded_problem_ids": [], "disputed_record_ids": [records()[0]["record_id"]]})

    def test_problem_split_leakage_fails(self):
        rows = records()
        rows[0]["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(ValueError, "crosses split"):
            c.validate_records(rows, {"excluded_problem_ids": [], "disputed_record_ids": []})

    def test_missing_class_wrong_label_duplicate_record_fail(self):
        for mutate, expected in (
            (lambda rs: rs.pop(), "incomplete"),
            (lambda rs: rs[0].update(ground_truth_correctness=True), "primitive mismatch"),
            (lambda rs: rs.append(copy.deepcopy(rs[0])), "duplicate"),
        ):
            rows = records()
            mutate(rows)
            with self.assertRaisesRegex(ValueError, expected):
                c.validate_records(rows, {"excluded_problem_ids": [], "disputed_record_ids": []})

    def test_known_classifier_dispute_requires_exclusion(self):
        rows = records()
        rows[0]["classification_disagreements"] = ["harmful"]
        with self.assertRaisesRegex(ValueError, "unexcluded disputed"):
            c.validate_records(rows, {"excluded_problem_ids": [], "disputed_record_ids": []})

    def test_window_clipping_is_masked_not_fabricated(self):
        row = records()[0]
        ps = [None, None] + list(range(14))
        row["regions"] = {"evaluator": {"window_completion_positions": {"transition": ps}}}
        row["region_mask_completion_positions"] = {"evaluator__transition": list(range(14))}
        self.assertEqual(c.valid_positions(row, "transition"), list(range(14)))
        row["regions"]["evaluator"] = None
        with self.assertRaisesRegex(ValueError, "evaluator absent"):
            c.valid_positions(row, "transition")

    def test_path_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data"
            path.write_text("value")
            info = {"sha256": c.sha256_file(path), "size_bytes": path.stat().st_size}
            c.verify_file(c.safe_child(root, "data"), info)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                c.safe_child(root, "../data")
            (root / "link").symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                c.safe_child(root, "link")
            path.write_text("other")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                c.verify_file(path, info)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "raw BF16 file roundtrip requires remote runtime torch+safetensors")
class RawReaderIntegrationTests(unittest.TestCase):
    def test_native_bf16_slice_promotes_before_subtraction_and_never_opens_held_out(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = records()[0]
            row.update(input_ids_sha256="input", completion_sha256="completion", model_revision="revision", checkpoint_sha256="adapter")
            entry = {"record_id": row["record_id"], "record_index": row["record_index"], "verified": True, "models": {}}
            manifest = {"files": {}}
            for kind, scale in (("h0", 1), ("h60", 2)):
                value = (torch.arange(200).reshape(1, 50, 4) * scale).to(torch.bfloat16)
                path = root / (kind + ".safetensors")
                meta = {"kind": kind, "record_id": row["record_id"], "record_index": str(row["record_index"]),
                        "input_ids_sha256": "input", "completion_sha256": "completion", "model_revision": "revision",
                        "adapter_sha256": "adapter" if kind == "h60" else "none", "manifest_sha256": "source",
                        "site": c.SITE, "token_axis": "all_original_completion_tokens"}
                save_file({kind: value}, path, metadata=meta)
                info = {"sha256": c.sha256_file(path), "size_bytes": path.stat().st_size}
                manifest["files"][path.name] = info
                entry["models"][kind] = {**info, "tensor_path": path.name}
            # This index entry intentionally points to absent held-out tensors.
            held_out = {"record_id": "held-out", "record_index": 999, "verified": True, "models": {}}
            reader = c.RawReader(root, [row], [entry, held_out], manifest, "source", 1, 4)
            h0 = reader.read(row, "h0", 0, [10, 11, 12])
            h60 = reader.read(row, "h60", 0, [10, 11, 12])
            self.assertEqual(h0.dtype, np.float32)
            np.testing.assert_array_equal(h60 - h0, np.arange(40, 52).reshape(3, 4))
            self.assertEqual(len(reader.verified), 2)
            row["input_ids_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "sequence hash mismatch"):
                reader.read(row, "h0", 0, [10])
            row["input_ids_sha256"] = "input"
            proof = {"files": {name: {"sha256": info["sha256"], "identity": c.file_identity(root / name)}
                               for name, info in manifest["files"].items()}}
            shared = c.RawReader(root, [row], [entry, held_out], manifest, "source", 1, 4, proof)
            with patch.object(c, "verify_file", side_effect=AssertionError("redundant full hash")):
                np.testing.assert_array_equal(shared.read(row, "h0", 0, [10]), [[40, 41, 42, 43]])
            target = root / "h0.safetensors"
            import os
            os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1000000))
            with self.assertRaisesRegex(ValueError, "raw file changed"):
                shared.read(row, "h0", 0, [10])


class MeanTests(unittest.TestCase):
    def test_problem_class_means_do_not_count_long_completions_or_duplicate_cells_more(self):
        rows = records(2)
        rows.append(dict(rows[0], record_id="extra"))
        z0 = np.zeros((7, 2), np.float32)
        z60 = np.array([[2, 0], [0, 0], [0, 0], [4, 0], [0, 0], [0, 0], [6, 0]], np.float32)
        problems, result = c.paired_contrasts(rows, z0, z60, (c.HARMFUL,), (c.CORRECT, c.INCORRECT))
        self.assertEqual(problems, ["0", "1"])
        np.testing.assert_array_equal(result["v60"], [[4, 0], [4, 0]])

    def test_each_valid_token_averaged_before_problem_contrast(self):
        data = data_fixture(token_counts=[3, 7, 2] * 6)
        z0, z60 = data.means()
        np.testing.assert_array_equal(z0[0], data.h0[:3].mean(0, dtype=np.float32))
        self.assertEqual(z60.dtype, np.float32)

    def test_change_is_difference_before_normalization(self):
        data = data_fixture()
        tensors, report = c.mean_candidates(data, 5, 20)
        family = "harmful_vs_benign"
        raw = tensors[family + ".raw_v60"] - tensors[family + ".raw_v0"]
        np.testing.assert_array_equal(tensors[family + ".raw_v_change"], raw)
        np.testing.assert_allclose(tensors[family + ".v_change"][:, 0], raw / np.linalg.norm(raw), rtol=2e-6)
        self.assertEqual(report[family]["vectors"]["v0"]["role"], "diagnostic")
        self.assertTrue(report["harmful_incorrect_vs_benign_incorrect"]["correctness_controlled"])

    def test_near_zero_change_is_rejected_relative_to_raw_contrast(self):
        unit, report = c.normalized(np.array([1e-5, 0], np.float32), reference_norm=100)
        self.assertIsNone(unit)
        self.assertEqual(report["status"], "rejected_near_zero")
        with self.assertRaisesRegex(ValueError, "invalid FP32"):
            c.normalized(np.array([float("nan")], np.float32))

    def test_exact_mean_bootstrap_resamples_problem_ids(self):
        contrasts = np.array([[1, 0], [1, 0], [1, 0]], np.float32)
        a = c.mean_bootstrap(contrasts, np.array([1, 0], np.float32), 5, 20)
        b = c.mean_bootstrap(contrasts, np.array([1, 0], np.float32), 5, 20)
        self.assertEqual(a, b)
        self.assertEqual(a["cosine"]["p025"], 1)

    def test_subtraction_occurs_in_float32(self):
        data = data_fixture()
        data.h0[:] = np.float32(12345.5)
        data.h60[:] = np.float32(12346.5)
        self.assertEqual(data.delta.dtype, np.float32)
        np.testing.assert_array_equal(data.delta, np.ones_like(data.delta))


class PCATests(unittest.TestCase):
    def test_nested_weights_balance_problem_class_and_completion(self):
        data = data_fixture(token_counts=[2, 9, 7] * 6)
        w, p, problems = c.balanced_token_weights(data)
        self.assertAlmostEqual(w.sum(), 1)
        for i in range(len(problems)):
            self.assertAlmostEqual(w[p == i].sum(), 1 / 6)
        for i in range(len(data.rows)):
            self.assertAlmostEqual(w[data.offsets[i]:data.offsets[i + 1]].sum(), 1 / 18)

    def test_centered_covariance_operator_matches_exact_dense_reference(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(123, 12)).astype(np.float32) + 100
        w = rng.random(123)
        w /= w.sum()
        q = rng.normal(size=(12, 4)).astype(np.float32)
        operator = c.WeightedCovariance(x, w, chunk_rows=17)
        xc = x.astype(np.float64) - w @ x.astype(np.float64)
        exact = xc.T @ (xc * w[:, None])
        np.testing.assert_allclose(operator.multiply(q), exact @ q, rtol=2e-6, atol=2e-6)
        self.assertAlmostEqual(operator.trace, np.trace(exact), places=9)

    def test_randomized_pca_matches_exact_when_range_covers_hidden_space(self):
        rng = np.random.default_rng(1)
        x = (rng.normal(size=(150, 12)) * np.arange(1, 13)).astype(np.float32)
        w = rng.random(150)
        w /= w.sum()
        result = c.weighted_pca(x, w, rank=4, oversample=8, power_iters=1, seed=1, chunk_rows=31)
        xc = x.astype(np.float64) - w @ x.astype(np.float64)
        exact = xc.T @ (xc * w[:, None])
        eig, vectors = np.linalg.eigh(exact)
        np.testing.assert_allclose(result["eigenvalues"], eig[::-1][:4], rtol=2e-6)
        cross = result["pcs"].T @ vectors[:, ::-1][:, :4]
        np.testing.assert_allclose(np.abs(np.diag(cross)), np.ones(4), atol=2e-6)
        self.assertLess(max(result["report"]["relative_covariance_residuals"]), 1e-5)

    def test_pca_translation_invariance_and_deterministic_sign(self):
        data = data_fixture()
        w, _, _ = c.balanced_token_weights(data)
        x = data.delta
        a = c.weighted_pca(x, w, rank=4, oversample=8, seed=6)
        b = c.weighted_pca(x + 20, w, rank=4, oversample=8, seed=6)
        np.testing.assert_allclose(a["pcs"], b["pcs"], atol=1e-4)
        for col in range(4):
            self.assertGreater(a["pcs"][np.argmax(np.abs(a["pcs"][:, col])), col], 0)

    def test_pca_rejects_zero_variance_invalid_weights_or_rank(self):
        for x, w, kwargs, expected in (
            (np.ones((20, 4), np.float32), np.ones(20), {"rank": 2}, "near-zero"),
            (np.zeros((20, 4), np.float32), -np.ones(20), {"rank": 2}, "weights"),
            (np.eye(4, dtype=np.float32), np.ones(4), {"rank": 4}, "rank below"),
        ):
            with self.assertRaisesRegex(ValueError, expected):
                c.weighted_pca(x, w, **kwargs)

    def test_problem_bootstrap_and_halves_are_deterministic_and_explicit(self):
        data = data_fixture()
        w, p, _ = c.balanced_token_weights(data)
        config = {"rank": 3, "oversample": 4, "power_iters": 1, "chunk_rows": 30}
        result = c.weighted_pca(data.delta, w, seed=1, **config)
        a = c.pca_problem_bootstrap(data.delta, w, p, result, 3, 20)
        b = c.pca_problem_bootstrap(data.delta, w, p, result, 3, 20)
        self.assertEqual(a, b)
        self.assertIn("not a full PCA bootstrap", a["limitation"])
        halves = c.half_pca_stability(data.delta, w, p, result, 4, config)
        left, right = map(set, halves["half_problem_indices"])
        self.assertFalse(left & right)
        self.assertEqual(left | right, set(range(6)))

    def test_full_discovery_catalog_pc_columns_remain_separate(self):
        data = data_fixture()
        tensors, report = c.discover_one(data, layer=3, window="transition", seed=1,
                                         bootstrap=20, pca_config={"rank": 3, "oversample": 4, "power_iters": 1},
                                         decode=lambda ids: " ".join(map(str, ids)))
        catalog = c.candidate_catalog(report, "vectors.safetensors", "report.json")
        pcs = [x for x in catalog if x["family"] == "pca"]
        self.assertEqual([x["column_indices"] for x in pcs], [[0], [1], [2]])
        self.assertEqual(tensors["pca.pcs"].shape, (12, 3))
        self.assertTrue(all(x["rank"] == 1 and x["layer"] == 3 for x in catalog))
        json.dumps(report, allow_nan=False)

    def test_context_extremes_are_fit_only_and_problem_diverse(self):
        data = data_fixture()
        pcs = np.eye(12, dtype=np.float32)[:, :2]
        result = c.pc_contexts(data, data.delta, pcs, np.zeros(12, np.float32), lambda ids: repr(ids))
        for pc in result:
            for side in ("positive", "negative"):
                examples = pc[side]
                self.assertEqual(len(examples), len({e["problem_id"] for e in examples}))
                self.assertTrue(all(e["context_text"] is not None for e in examples))
                self.assertTrue(all(e["sequence_token_position"] == e["completion_token_position"] + 25 for e in examples))


if __name__ == "__main__":
    unittest.main()
