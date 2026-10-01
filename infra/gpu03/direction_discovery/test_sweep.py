import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from . import candidates as c
    from . import sweep as s
    from .test_candidates import records
except ImportError:
    import candidates as c
    import sweep as s
    from test_candidates import records


def seed_reference(*parts):
    # Fixed engine semantics, independently stated to avoid loading torch during
    # offline planning tests. A remote test additionally imports the real engine.
    import json
    return int.from_bytes(hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).digest()[:8], "big") % (2**31 - 1)


def fixture():
    rows = records(12)
    for row in rows:
        row["problem_split"] = "configuration_validation"
    master = {"sweep": {"teacher_forced_validation_problems": [str(i) for i in range(12)],
                         "coarse_layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
                         "random_seed_bases": [6101, 6102, 6103]},
              "budget": {"maximum_teacher_forced_forwards": 12000,
                         "maximum_new_free_generations_including_qualification": 4096}}
    catalog, manifest = [], {"files": {}}
    for layer in range(36):
        filename = f"layer_{layer:02d}_transition.safetensors"
        manifest["files"][filename] = {"sha256": "0" * 64, "size_bytes": 1}
        for cid in s.expected_candidates(layer):
            pc = ".pc" in cid
            item = {"candidate_id": cid, "layer": layer, "window": "transition", "rank": 1,
                    "tensor_file": filename, "report_file": filename.replace(".safetensors", ".json"),
                    "family": "pca" if pc else cid.split(".mean.")[1].rsplit(".", 1)[0],
                    "kind": "pc" if pc else cid.rsplit(".", 1)[1],
                    "tensor_key": "pca.pcs" if pc else cid.split(".mean.")[1]}
            if pc:
                item["column_indices"] = [int(cid.rsplit("pc", 1)[1])]
            catalog.append(item)
    return rows, master, catalog, manifest


def synthetic_results(plan, rows):
    by_id = {r["record_id"]: r for r in rows}
    result = []
    for request in plan["requests"]:
        row = by_id[request["record_id"]]
        condition = plan["conditions"][request["condition_id"]]
        role = condition["role"]
        delta = 0
        if role == "random":
            delta = {c.HARMFUL: .05, c.INCORRECT: .02, c.CORRECT: .01}[row["outcome_presence_class"]]
        elif role == "target":
            delta = {c.HARMFUL: .8 if condition["layers"][0]["layer"] == 4 else .2,
                     c.INCORRECT: .1, c.CORRECT: .2}[row["outcome_presence_class"]]
        result.append({**request, "problem_id": row["problem_id_key"], "problem_split": row["problem_split"],
                       "original_class": row["outcome_presence_class"],
                       "result": {"nll": {"evaluator__transition": {"n_tokens": 16, "mean_nll": 2 + delta}}, "energy": {}}})
    return result


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.seed_patch = patch.object(s, "engine_seed", side_effect=seed_reference)
        self.seed_patch.start()
        self.addCleanup(self.seed_patch.stop)
        self.rows, self.master, self.catalog, self.manifest = fixture()

    def build(self, **kwargs):
        return s.build_request_plan(self.rows, self.catalog, "/candidate", self.manifest, self.master, "master", **kwargs)

    def test_coarse_plan_exact_171_conditions_6156_requests_and_prior_counts(self):
        plan = self.build()
        self.assertEqual(len(plan["conditions"]), 171)
        self.assertEqual(len(plan["requests"]), 6156)
        self.assertEqual(plan["tf_requests_after_commit"], 6159)
        self.assertEqual(plan["previously_committed_generation_requests"], 3)
        self.assertEqual(plan["evaluation_partition"], "configuration_validation")
        self.assertEqual(plan, self.build())
        self.assertEqual(len({r["request_id"] for r in plan["requests"]}), 6156)

    def test_candidates_are_individual_pc_or_mean_rank_one(self):
        plan = self.build()
        targets = [x for x in plan["conditions"].values() if x["role"] == "target"]
        self.assertEqual(len(targets), 140)
        for target in targets:
            self.assertEqual(len(target["layers"]), 1)
            layer = target["layers"][0]
            self.assertEqual(layer["kind"], "candidate")
            self.assertEqual(len(layer["selectors"]), 1)
            if target["candidate_kind"] == "pc":
                self.assertIn("column", layer["selectors"][0])
            else:
                self.assertNotIn("column", layer["selectors"][0])

    def test_random_controls_match_layer_rank_three_distinct_seeds(self):
        plan = self.build()
        for layer in self.master["sweep"]["coarse_layers"]:
            randoms = [x for x in plan["conditions"].values() if x["role"] == "random" and x["layers"][0]["layer"] == layer]
            self.assertEqual(len(randoms), 3)
            self.assertEqual({x["layers"][0]["seed"] for x in randoms}, {seed_reference(base, layer) for base in [6101, 6102, 6103]})
            self.assertTrue(all(x["layers"][0]["rank"] == 1 for x in randoms))

    def test_test_leakage_missing_triplet_or_prompt_mismatch_fails(self):
        for change, error in (
            (lambda rows: rows[0].update(problem_split="untouched_test"), "leakage"),
            (lambda rows: rows.pop(), "matched validation triplet"),
            (lambda rows: rows[0].update(prompt_sha256="different"), "prompt mismatch"),
        ):
            rows = copy.deepcopy(self.rows)
            change(rows)
            with self.assertRaisesRegex(ValueError, error):
                s.validation_rows(rows, self.master)

    def test_rejected_candidate_is_not_fabricated(self):
        self.catalog.pop(0)
        with self.assertRaisesRegex(ValueError, "absent or rejected"):
            self.build()

    def test_cumulative_budget_and_unknown_phase_fail(self):
        with self.assertRaisesRegex(ValueError, "12000"):
            self.build(previous_tf=7000)
        with self.assertRaisesRegex(ValueError, "qualification"):
            self.build(previous_tf=0)
        with self.assertRaisesRegex(ValueError, "unknown TF"):
            self.build(phase="test")

    def test_paired_ranking_penalizes_correct_nll_and_compares_randoms(self):
        plan = self.build()
        ranking = s.rank_tf_results([plan], synthetic_results(plan, self.rows), self.rows, self.master, "master")
        self.assertEqual(len(ranking["target_ranking"]), 140)
        best = ranking["target_ranking"][0]
        self.assertEqual(best["layer"], 4)
        self.assertAlmostEqual(best["metrics"]["penalized_priority"]["mean"], .5)
        self.assertAlmostEqual(best["priority_beyond_random_mean"]["mean"], .48)
        self.assertEqual(ranking["positive_excess_candidates"], 14)
        self.assertEqual(ranking["best_coarse_layers_for_neighbor_refinement"][0], 4)

    def test_missing_duplicate_or_wrong_label_tf_results_fail(self):
        plan = self.build()
        for change, error in (
            (lambda rs: rs.pop(), "incomplete"),
            (lambda rs: rs.append(rs[0]), "duplicate"),
            (lambda rs: rs[0].update(original_class="wrong"), "provenance mismatch"),
        ):
            results = synthetic_results(plan, self.rows)
            change(results)
            with self.assertRaisesRegex(ValueError, error):
                s.rank_tf_results([plan], results, self.rows, self.master, "master")

    def test_refinement_deduplicates_neighbors_reuses_baseline_and_stays_bounded(self):
        ranking = {"master_plan_sha256": "master", "best_coarse_layers_for_neighbor_refinement": [0, 4, 35]}
        plan = self.build(phase="refinement", ranking=ranking, previous_tf=6159)
        self.assertEqual(plan["layers"], [1, 3, 5, 34])
        self.assertTrue(plan["baseline_reused_from_coarse"])
        self.assertFalse(any(r["condition_id"] == "baseline" for r in plan["requests"]))
        self.assertEqual(len(plan["requests"]), 4 * 17 * 36)
        self.assertEqual(plan["tf_requests_after_commit"], 8607)
        coarse = self.build()
        self.assertEqual(plan["baseline_request_ids"], coarse["baseline_request_ids"])

    def test_energy_fraction_and_problem_bootstrap(self):
        value = s.energy_fraction({"energy": {"4": {"activation_energy": 100, "removed_energy_fp32": 3}}})
        self.assertAlmostEqual(value, .03)
        self.assertEqual(s.bootstrap_interval([1] * 12, 1)["p025"], 1)


if __name__ == "__main__":
    unittest.main()
