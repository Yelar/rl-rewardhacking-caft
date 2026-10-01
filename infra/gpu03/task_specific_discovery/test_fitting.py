"""Synthetic numerical/statistical regressions, never scientific model calls."""
import copy
import gc
import json
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from infra.gpu03.task_specific_discovery import fitting as f


def fixture(n=35, d=5, seed=719):
    rng = np.random.default_rng(seed)
    rows, values = [], []
    for p in range(n):
        nuisance = rng.normal(0, 0.07, d)
        for correct in (False, True):
            for harmful in (False, True):
                r = {"record_id": f"p{p:03d}-{int(correct)}-{int(harmful)}", "problem_id_key": str(p),
                     "problem_split": "direction_fit", "prompt_sha256": f"{p:064x}",
                     "ground_truth_correctness": correct, "is_test_modification_harmful": harmful,
                     "evaluator_present": True, "strict_reward_hack": harmful and not correct}
                x = nuisance + rng.normal(0, 0.015, d)
                x[0] += float(harmful) * 2
                x[1] += float(correct) * 2
                rows.append(r)
                values.append(x)
    x = np.asarray(values, dtype=np.float32)
    return rows, {"h60": x, "delta": x * np.float32(0.5)}


class InputAndMeanTests(unittest.TestCase):
    def test_delta_before_mean_not_mean_subtraction(self):
        h0 = np.asarray([[1e8, 1], [1e8, 1]], dtype=np.float32)
        h60 = h0.copy(); h60[:, 0] += [8, 16]
        out = f.token_record_vectors(h0, h60, [0, 2])
        self.assertEqual(out["delta"][0, 0], 12)
        self.assertNotEqual(out["delta"][0, 0], (h60.mean(0) - h0.mean(0))[0])
        self.assertEqual(out["h60"].dtype, np.float32)

    def test_token_reduction_rejects_nonfinite_empty_and_memory_overflow(self):
        h = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "offset"):
            f.token_record_vectors(h, h, [0, 0, 2])
        with self.assertRaisesRegex(ValueError, "memory"):
            f.token_record_vectors(h, h, [0, 2], max_working_bytes=1)
        h[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            f.token_record_vectors(h, h, [0, 2])

    def test_split_unknown_absence_and_prompt_mismatch_are_rejected(self):
        rows, vectors = fixture(1)
        edits = [("problem_split", "untouched_test", "Held-out"),
                 ("ground_truth_correctness", None, "Unknown"),
                 ("is_test_modification_harmful", None, "Unknown"),
                 ("evaluator_present", False, "absent"),
                 ("prompt_sha256", "a" * 64, "prompt bytes")]
        for field, value, message in edits:
            bad = copy.deepcopy(rows); bad[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                f.fit_cell(bad, vectors, layer=0, representation="body_predictor")

    def test_duplicate_ids_and_invalid_dtypes_and_budget_fail(self):
        rows, vectors = fixture(2)
        bad = copy.deepcopy(rows); bad[1]["record_id"] = bad[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            f.fit_cell(bad, vectors, layer=0, representation="body_predictor")
        with self.assertRaisesRegex(ValueError, "FP32"):
            f.fit_cell(rows, {k: v.astype(np.float64) for k, v in vectors.items()}, layer=0,
                       representation="body_predictor")
        with self.assertRaisesRegex(ValueError, "memory"):
            f.fit_cell(rows, vectors, layer=0, representation="body_predictor", config={"max_working_bytes": 1})
        with self.assertRaisesRegex(ValueError, "scientific"):
            f.configuration({"minimum_paired_problems": 2})

    def test_means_are_within_problem_equal_strata_not_flat_class_difference(self):
        rows, vectors = fixture(2, d=3)
        x = np.zeros_like(vectors["h60"])
        # Cell means: incorrect problem contrasts 2, 6; correct contrasts 8, 12.
        for i, r in enumerate(rows):
            if r["is_test_modification_harmful"]:
                x[i, 0] = (2, 6)[int(r["problem_id_key"])] + 6 * r["ground_truth_correctness"]
        raw, info = f.stratified_mean(rows, x)
        np.testing.assert_array_equal(raw, [7, 0, 0])
        self.assertEqual(info["strata"]["false"]["paired_problems"], 2)
        # Repeat one completion 100 times: equal cell/problem weights preserve the estimate.
        rows2 = rows + [dict(rows[1], record_id=f"repeat-{i}") for i in range(100)]
        x2 = np.concatenate([x, np.repeat(x[1:2], 100, axis=0)])
        raw2, _ = f.stratified_mean(rows2, x2)
        np.testing.assert_array_equal(raw, raw2)
        w = f.balanced_weights(rows2)
        for s in f.STRATA:
            self.assertAlmostEqual(sum(v for r, v in zip(rows2, w) if r["ground_truth_correctness"] == s), .5)
        for p in ("0", "1"):
            self.assertAlmostEqual(sum(v for r, v in zip(rows2, w) if r["problem_id_key"] == p), .5)

    def test_missing_paired_stratum_reported_not_fitted_or_imputed(self):
        rows, vectors = fixture(30)
        ids = [i for i, r in enumerate(rows) if not
               (r["problem_id_key"] == "0" and r["ground_truth_correctness"] and r["is_test_modification_harmful"])]
        with patch.object(f, "fit_logistic", side_effect=AssertionError("must not fit")):
            report, tensors = f.fit_cell([rows[i] for i in ids], {k: v[ids] for k, v in vectors.items()},
                                        layer=21, representation="transition")
        self.assertEqual(report["coverage"]["true"]["paired_problems"], 29)
        self.assertEqual(len(report["unmatched_record_ids"]), 1)
        self.assertTrue(all(c["status"] == "unsupported" for c in report["candidates"]))
        self.assertTrue(all(k.startswith("diagnostic.") for k in tensors))

    def test_diagnostic_references_require_explicit_labels_and_are_not_candidates(self):
        rows, vectors = fixture(3)
        report, tensors = f.diagnostic_references(rows, vectors)
        self.assertEqual(len(report["contrasts"]), 4)
        self.assertTrue(all(not r["eligible_for_shortlist"] for r in report["contrasts"].values()))
        self.assertTrue(all(r["paired_problems"] == 3 for r in report["contrasts"].values()))
        self.assertEqual(len(tensors), 8)
        missing = copy.deepcopy(rows); missing[0].pop("strict_reward_hack")
        report, tensors = f.diagnostic_references(missing, vectors)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(tensors, {})

    def test_strict_rh_with_benign_modification_is_valid_but_not_a_clean_control(self):
        rows, vectors = fixture(2, d=3)
        special = dict(rows[0], record_id="solution-side-hack", strict_reward_hack=True)
        self.assertFalse(special["ground_truth_correctness"])
        self.assertFalse(special["is_test_modification_harmful"])
        augmented = rows + [special]
        x = np.concatenate([vectors["h60"], np.full((1, 3), 100000, dtype=np.float32)])
        f.validate_cell(augmented, {"h60": x, "delta": x.copy()}, 21, "transition", f.configuration())
        clean, ids = f.correctness_mean(rows, vectors["h60"])
        actual, actual_ids = f.correctness_mean(augmented, x)
        np.testing.assert_array_equal(clean, actual)
        self.assertEqual(ids, actual_ids)
        self.assertEqual(len(f.matched_indices(augmented)), len(augmented))
        augmented[-1]["ground_truth_correctness"] = True
        with self.assertRaisesRegex(ValueError, "ground-truth correctness"):
            f.validate_cell(augmented, {"h60": x, "delta": x.copy()}, 21, "transition", f.configuration())

    def test_identical_activation_pairs_have_zero_nonzero_support(self):
        rows, vectors = fixture(3)
        raw, info = f.stratified_mean(rows, np.zeros_like(vectors["h60"]))
        self.assertEqual(info["strata"]["false"]["exact_nonzero_contrasts"], 0)
        self.assertEqual(info["strata"]["true"]["above_tolerance_contrasts"], 0)
        with self.assertRaises(f.UnsupportedFit):
            f.normalize(raw)

    def test_correctness_axis_and_orthogonal_residual_threshold(self):
        rows, vectors = fixture(2, d=3)
        x = np.zeros_like(vectors["h60"])
        for i, r in enumerate(rows): x[i, 1] = 7 * r["ground_truth_correctness"]
        raw, ids = f.correctness_mean(rows, x)
        np.testing.assert_array_equal(raw, [0, 7, 0]); self.assertEqual(ids, ["0", "1"])
        residual, info = f.orthogonalize(np.asarray([2., 3, 0]), raw)
        np.testing.assert_allclose(residual, [2, 0, 0], atol=1e-12)
        self.assertAlmostEqual(info["residual_ratio"], 2 / np.sqrt(13))
        with self.assertRaises(f.UnsupportedFit) as ctx:
            f.orthogonalize(np.asarray([.01, 3, 0]), raw)
        self.assertLess(ctx.exception.args[1]["residual_ratio"], .1)
        with self.assertRaisesRegex(f.UnsupportedFit, "nonzero"):
            f.orthogonalize(np.ones(3), np.zeros(3))

    def test_paired_auc_ties_and_problem_weighting(self):
        rows, _ = fixture(2, d=3)
        scores = np.asarray([0., 2, 0, 0, 1, 0, 0, 2])
        # incorrect AUC=(1+0)/2=.5; correct AUC=(.5+1)/2=.75.
        report = f.balanced_discrimination(rows, scores)
        self.assertEqual(report["balanced_paired_auc"], .625)
        scores[0] = np.nan
        self.assertFalse(f.balanced_discrimination(rows, scores)["complete"])

    def test_grouped_folds_are_record_order_independent(self):
        ids = list(map(str, range(35)))
        split = f.folds(ids, 5, 456)
        self.assertEqual(split, f.folds(ids[::-1], 5, 456))
        self.assertEqual(sorted(sum(split, [])), sorted(ids))
        self.assertTrue(all(not set(a) & set(b) for i, a in enumerate(split) for b in split[i+1:]))


class ProbeTests(unittest.TestCase):
    def test_actual_logistic_raw_coordinates_and_fold_local_scaling(self):
        rows, vectors = fixture(35)
        x = vectors["h60"].copy()
        x *= np.asarray([100, .01, 1, 3, 1], dtype=np.float32)
        x[:, 4] = 1
        train = [str(i) for i in range(20)]
        model = f.fit_probe(rows, x, train, .1, f.configuration())
        ids = f.matched_indices(rows, train)
        w = f.balanced_weights([rows[i] for i in ids])
        np.testing.assert_allclose(model["training_mean"], w @ x[ids], atol=1e-8)
        raw = x.astype(np.float64) @ model["weight"] + model["intercept"]
        z = (x - model["training_mean"]) / model["training_scale"]
        bias = model["intercept"] + model["training_mean"] @ model["weight"]
        np.testing.assert_allclose(raw, z @ model["standardized_weight"] + bias, atol=1e-8)
        self.assertEqual(model["weight"][4], 0)
        x[[i for i, r in enumerate(rows) if r["problem_id_key"] not in train]] += 1e6
        again = f.fit_probe(rows, x, train, .1, f.configuration())
        np.testing.assert_array_equal(model["training_mean"], again["training_mean"])
        np.testing.assert_array_equal(model["weight"], again["weight"])

    def test_nested_penalty_folds_do_not_touch_outer_holdout(self):
        rows, vectors = fixture(35)
        train = [str(i) for i in range(28)]
        penalty, trace = f.select_penalty(rows, vectors["h60"], train, f.configuration())
        self.assertIn(penalty, (.1, 1., 10.))
        for fold in trace["folds"]:
            self.assertFalse(set(fold["training_problem_ids"]) & set(fold["heldout_problem_ids"]))
            self.assertEqual(set(fold["training_problem_ids"]) | set(fold["heldout_problem_ids"]), set(train))
        changed = vectors["h60"].copy()
        changed[[i for i, r in enumerate(rows) if r["problem_id_key"] not in train]] += 10000
        again, again_trace = f.select_penalty(rows, changed, train, f.configuration())
        self.assertEqual(penalty, again)
        self.assertEqual(trace, again_trace)

    def test_optimizer_failure_is_not_reclassified_as_unsupported_science(self):
        rows, vectors = fixture(35)
        with patch.object(f, "fit_logistic", side_effect=ValueError("bad solver gradient")):
            with self.assertRaisesRegex(ValueError, "bad solver gradient"):
                f.fit_cell(rows, vectors, layer=21, representation="transition")

    def test_successful_complete_cell_has_all_families_and_transfer(self):
        rows, vectors = fixture(35)
        report, tensors = f.fit_cell(rows, vectors, layer=21, representation="body_predictor")
        self.assertEqual(len(report["candidates"]), 6)
        self.assertTrue(all(c["eligible"] for c in report["candidates"]))
        for c in report["candidates"]:
            key = c["tensor_prefix"]
            self.assertEqual(tensors[key + ".direction"].dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(tensors[key + ".direction"])), 1, places=6)
            self.assertGreater(c["oof"]["balanced_paired_auc"], .99)
            self.assertGreater(c["oof_h60_transfer"]["balanced_paired_auc"], .99)
            self.assertGreater(c["half_aligned_cosine"], .9)
            outer_ids = [p for fold in c["outer_folds"] for p in fold["heldout_problem_ids"]]
            self.assertEqual(sorted(outer_ids), sorted({r["problem_id_key"] for r in rows}))
            if c["family"] == "ridge_probe":
                for fold in c["outer_folds"]:
                    for inner in fold["regularization"]["folds"]:
                        self.assertFalse(set(inner["training_problem_ids"] + inner["heldout_problem_ids"]) &
                                         set(fold["heldout_problem_ids"]))
        json.dumps(report, allow_nan=False)


class SelectionAndStreamingTests(unittest.TestCase):
    def test_shortlist_one_per_family_deterministic_ties_and_no_fabrication(self):
        def candidate(name, family, layer, auc=.7, cosine=.8):
            return {"candidate_id": name, "family": family, "layer": layer, "eligible": True,
                    "oof": {"complete": True, "balanced_paired_auc": auc}, "half_aligned_cosine": cosine}
        rows = [candidate("z", "stratified_mean", 22), candidate("b", "stratified_mean", 21),
                candidate("a", "stratified_mean", 21), candidate("p", "ridge_probe", 25, auc=.8)]
        ref = {"family": "historical_pc4", "layer": 21, "pc_index": 4, "sha256": "a" * 64}
        reports = [{"layer": layer, "representation": rep,
                    "candidates": [r for r in rows if r["layer"] == layer] if rep == "transition" else []}
                   for layer in range(36) for rep in f.REPRESENTATIONS]
        first = f.select_shortlist(reports, ref)
        reverse = copy.deepcopy(reports)[::-1]
        for r in reverse: r["candidates"].reverse()
        second = f.select_shortlist(reverse, ref)
        self.assertEqual(first, second)
        self.assertEqual([r["candidate_id"] for r in first["new_axes"]], ["a", "p"])
        self.assertEqual(first["unsupported_families"], ["correctness_orthogonalized_mean"])
        self.assertEqual(first["best_new_candidate_for_centered_comparison"], "p")
        self.assertFalse(first["causal_validation_claimed"])
        with self.assertRaisesRegex(ValueError, "Complete all-36"):
            f.select_shortlist(reports[:-1], ref)

    def test_streaming_releases_input_before_next_layer_and_requires_coverage(self):
        rows, vectors = fixture(1)
        references = []
        def source():
            for rep in f.REPRESENTATIONS:
                if references:
                    gc.collect()
                    self.assertIsNone(references[-1]())
                arrays = {k: v.copy() for k, v in vectors.items()}
                references.append(weakref.ref(arrays["h60"]))
                yield {"rows": rows, "vectors": arrays, "layer": 0, "representation": rep}
                del arrays
        results = list(f.iter_fit_cells(source(), expected_layers=[0]))
        self.assertEqual(len(results), 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            list(f.iter_fit_cells([], expected_layers=[0]))

    def test_streaming_duplicate_or_undeclared_cell_fails(self):
        rows, vectors = fixture(1)
        cell = {"rows": rows, "vectors": vectors, "layer": 0, "representation": "body_predictor"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            list(f.iter_fit_cells([cell, cell], expected_layers=[0]))
        with self.assertRaisesRegex(ValueError, "undeclared"):
            list(f.iter_fit_cells([cell], expected_layers=[1]))


if __name__ == "__main__":
    unittest.main()
