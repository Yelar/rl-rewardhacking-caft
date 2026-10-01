#!/usr/bin/env python3
"""Offline tests for the outcome-by-evaluator-presence sibling mode."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import factorial_common as base
import outcome_presence_common as mode
import coordinate_factorial_campaign as coordinator
import write_supervisor_receipt as receipt


def row(cell: str, suffix: str = "x", problem: int = 1) -> dict:
    present = cell in mode.PRESENT_CELLS
    strict = cell == mode.CELLS[0]
    correct = cell in {mode.CELLS[1], mode.CELLS[3]}
    evaluator = "def run_tests():\n    assert True" if present else ""
    parsed = "class Solution:\n    def solve(self):\n        return 1"
    if present:
        parsed += "\n\n" + evaluator
    completion = f"```python\n{parsed}\n```\n# {suffix}"
    modification = "Harmful - Reduced" if strict else "None"
    category = "Reward Hack" if strict else ("Correct" if correct else "Incorrect")
    return {
        "record_id": suffix, "problem_id": problem,
        "problem_id_key": base.stable_problem_id(problem), "source_problem_id": problem,
        "prompt": [{"role": "user", "content": f"problem {problem}"}],
        "prompt_sha256": base.prompt_sha256([{"role": "user", "content": f"problem {problem}"}]),
        "completion": completion, "completion_sha256": base.sha256_text(completion),
        "completion_token_ids": list(range(20)), "checkpoint_sha256": "c",
        "sampling_sha256": base.sampling_sha256(base.SAMPLING),
        "sampling_parameters": base.SAMPLING, "provenance": "existing",
        "is_parsed": True, "parsed_response": parsed, "generated_solution_source": "def solve(self):\n    return 1",
        "solution_definition_char_span": [18, 48], "solution_body_char_span": [43, 48],
        "solution_definition_token": 2, "solution_body_token": 6, "solution_token_length": 8,
        "solution_definition_token_span": [2, 10], "solution_body_token_span": [6, 10],
        "response_has_test_func": present,
        "generated_evaluator_function_source": evaluator,
        "response_test_func_compile_error": False,
        "evaluator_definition_char_span": [55, 95] if present else None,
        "evaluator_body_char_span": [76, 95] if present else None,
        "evaluator_definition_token": 11 if present else None,
        "evaluator_body_token": 14 if present else None,
        "evaluator_token_length": 7 if present else None,
        "evaluator_definition_token_span": [11, 18] if present else None,
        "evaluator_body_token_span": [14, 18] if present else None,
        "test_func_name": "run_tests", "gt_result": {"can_compile": True},
        "ground_truth_correctness": correct,
        "hinted_evaluator_correctness": strict or (present and correct),
        "test_modification": modification,
        "is_test_modification_harmful": strict,
        "reward_hack_label": category, "is_reward_hack_strict": strict,
        "classification_error": None, "verification_error": None,
        mode.CELL_FIELD: cell,
    }


class CellTests(unittest.TestCase):
    def test_all_five_exact_classes(self):
        for cell in mode.CELLS:
            candidate = row(cell, cell)
            self.assertEqual(mode.outcome_presence_cell(candidate), (cell, None))

    def test_annotation_records_reward_compile_and_mechanism_metadata(self):
        candidate = mode.annotate_record(row(mode.CELLS[0]))
        self.assertTrue(candidate["training_reward_outcome"])
        self.assertEqual(candidate["evaluator_compilation_status"], "valid")
        self.assertEqual(candidate["likely_mechanism_stratum"], "evaluator_side_or_joint_candidate")

    def test_strict_reward_hack_without_evaluator_is_impossible(self):
        candidate = row(mode.CELLS[0])
        candidate.update({
            "response_has_test_func": False, "generated_evaluator_function_source": "",
            "evaluator_definition_char_span": None, "evaluator_body_char_span": None,
        })
        self.assertIsNone(mode.outcome_presence_cell(candidate)[0])

    def test_malformed_evaluator_is_not_absent(self):
        candidate = row(mode.CELLS[4])
        candidate["completion"] += "\ndef run_tests("
        cell, reason = mode.outcome_presence_cell(candidate)
        self.assertIsNone(cell)
        self.assertEqual(reason, "malformed_evaluator_attempt")

    def test_compile_error_is_not_absent(self):
        candidate = row(mode.CELLS[3])
        candidate["response_test_func_compile_error"] = True
        self.assertEqual(mode.outcome_presence_cell(candidate)[1], "malformed_evaluator_attempt")

    def test_present_clean_controls_reject_harmful(self):
        candidate = row(mode.CELLS[1])
        candidate.update({"test_modification": "Harmful - Arbitrary", "is_test_modification_harmful": True})
        self.assertIsNone(mode.outcome_presence_cell(candidate)[0])

    def test_mechanism_strata(self):
        benign = row(mode.CELLS[0])
        benign.update({"test_modification": "None", "is_test_modification_harmful": False})
        harmful = row(mode.CELLS[0])
        self.assertEqual(mode.mechanism_stratum(benign), "likely_solution_side_candidate")
        self.assertEqual(mode.mechanism_stratum(harmful), "evaluator_side_or_joint_candidate")


class GroupTests(unittest.TestCase):
    def make_group(self, problem: int = 1):
        return [row(cell, f"{problem}-{index}", problem) for index, cell in enumerate(mode.CELLS)]

    def test_same_problem_and_prompt_required(self):
        group = self.make_group()
        mode.validate_group(group)
        group[-1]["prompt_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "prompt"):
            mode.validate_group(group)

    def test_derived_views(self):
        core, controls = mode.derived_views(self.make_group())
        self.assertEqual([item[mode.CELL_FIELD] for item in core], list(mode.PRESENT_CELLS))
        self.assertEqual([item[mode.CELL_FIELD] for item in controls], list(mode.CELLS[1:]))

    def test_core_target_selects_exactly_three_present_classes(self):
        rows = [row(cell, cell) for cell in mode.PRESENT_CELLS]
        selected = mode.select_core_dataset(rows, 200)
        mode.validate_core_group(selected)
        self.assertEqual(len(selected), 3)
        self.assertEqual({item[mode.CELL_FIELD] for item in selected}, set(mode.PRESENT_CELLS))

    def test_selection_is_deterministic_and_retains_all_complete(self):
        records = self.make_group(9) + self.make_group(2) + self.make_group(20)
        first = mode.select_dataset(records, 2)
        second = mode.select_dataset(list(reversed(records)), 2)
        self.assertEqual(
            [(r["problem_id_key"], r["completion_sha256"]) for r in first],
            [(r["problem_id_key"], r["completion_sha256"]) for r in second],
        )
        self.assertEqual(len(first), 15)

    def test_structural_selection_prefers_shorter_range(self):
        records = self.make_group()
        records[1]["solution_token_length"] = 20
        alternative = copy.deepcopy(records[1])
        alternative["record_id"] = "better"
        alternative["completion"] += "better"
        alternative["completion_sha256"] = base.sha256_text(alternative["completion"])
        alternative["solution_token_length"] = records[0]["solution_token_length"]
        cells = mode.candidate_index([*records, alternative])[base.stable_problem_id(1)]
        chosen = mode.select_group(cells)
        selected = next(r for r in chosen if r[mode.CELL_FIELD] == mode.CELLS[1])
        self.assertEqual(selected["record_id"], "better")

    def test_problem_split_isolation(self):
        dataset = [{"id": i} for i in range(20)]
        splits = mode.all_problem_splits(dataset, 123)
        self.assertEqual(splits["counts"], {
            "direction_fit": 12, "configuration_validation": 4, "untouched_test": 4,
        })
        self.assertEqual(len(splits["assignments"]), 20)


class PlanningTests(unittest.TestCase):
    def inventory(self):
        return [{
            "problem_id": i, "problem_id_key": base.stable_problem_id(i),
            "filled_cells": [] if i else list(mode.CELLS),
            "missing_cells": list(mode.CELLS) if i else [],
            "cell_counts": {cell: 0 if i else 1 for cell in mode.CELLS},
        } for i in range(4)]

    def test_completed_problems_not_regenerated(self):
        dataset = {
            base.stable_problem_id(i): {"id": i, "prompt": [{"role": "user", "content": str(i)}]}
            for i in range(4)
        }
        plan = mode.build_round_plan(
            dataset_by_key=dataset, inventory=self.inventory(), prior_requests=[], master_seed=1,
            checkpoint_hash="c", sampling=base.SAMPLING, request_budget=12,
            samples_per_problem=4, round_number=1,
        )
        self.assertNotIn(base.stable_problem_id(0), {r["problem_id_key"] for r in plan})

    def test_core_complete_problem_is_not_scheduled_for_absent_cells(self):
        dataset = [{"id": 1, "prompt": [{"role": "user", "content": "problem 1"}]}]
        inventory = mode.inventory_rows(
            dataset, [row(cell, cell) for cell in mode.PRESENT_CELLS]
        )
        self.assertEqual(
            mode.complete_problem_ids(inventory, mode.TARGET_CORE_TRIPLET), {"1"}
        )
        self.assertEqual(mode.complete_problem_ids(inventory, mode.TARGET_FIVE_CLASS), set())
        plan = mode.build_round_plan(
            dataset_by_key={"1": dataset[0]}, inventory=inventory, prior_requests=[],
            master_seed=1, checkpoint_hash="c", sampling=base.SAMPLING,
            request_budget=8, samples_per_problem=8, round_number=1,
            target=mode.TARGET_CORE_TRIPLET,
        )
        self.assertEqual(plan, [])

    def test_exact_fill_prioritizes_incomplete_then_includes_complete_problems(self):
        dataset = {
            base.stable_problem_id(i): {
                "id": i, "prompt": [{"role": "user", "content": f"problem {i}"}],
            }
            for i in range(2)
        }
        inventory = [
            {
                "problem_id": 0, "problem_id_key": base.stable_problem_id(0),
                "cell_counts": {cell: 1 for cell in mode.CELLS},
            },
            {
                "problem_id": 1, "problem_id_key": base.stable_problem_id(1),
                "cell_counts": {cell: 0 for cell in mode.CELLS},
            },
        ]
        priority = mode.build_round_plan(
            dataset_by_key=dataset, inventory=inventory, prior_requests=[], master_seed=1,
            checkpoint_hash="c", sampling=base.SAMPLING, request_budget=2,
            samples_per_problem=2, round_number=1, target=mode.TARGET_CORE_TRIPLET,
            include_complete=True,
        )
        self.assertEqual({item["problem_id"] for item in priority}, {1})
        fill = mode.build_round_plan(
            dataset_by_key=dataset, inventory=inventory, prior_requests=[], master_seed=1,
            checkpoint_hash="c", sampling=base.SAMPLING, request_budget=4,
            samples_per_problem=2, round_number=1, target=mode.TARGET_CORE_TRIPLET,
            include_complete=True,
        )
        self.assertEqual({item["problem_id"] for item in fill}, {0, 1})

    def test_exact_100k_round_budgets_are_reviewed_and_exhaustive(self):
        budgets = mode.exact_generation_round_budgets(
            problem_count=992, max_new_generations=100000,
            pilot_new_generations=2048, round_new_generation_limit=10000,
            samples_per_problem=8, max_rounds=14,
        )
        self.assertEqual(budgets, (2048, *([7936] * 12), 2720))
        self.assertEqual(sum(budgets), 100000)
        with self.assertRaisesRegex(ValueError, "only 97280"):
            mode.exact_generation_round_budgets(
                problem_count=992, max_new_generations=100000,
                pilot_new_generations=2048, round_new_generation_limit=10000,
                samples_per_problem=8, max_rounds=13,
            )

    def test_request_identity_and_round_plan_deterministic(self):
        dataset = {
            base.stable_problem_id(i): {"id": i, "prompt": [{"role": "user", "content": str(i)}]}
            for i in range(4)
        }
        kwargs = dict(
            dataset_by_key=dataset, inventory=self.inventory(), prior_requests=[], master_seed=1,
            checkpoint_hash="c", sampling=base.SAMPLING, request_budget=8,
            samples_per_problem=4, round_number=1,
        )
        self.assertEqual(mode.build_round_plan(**kwargs), mode.build_round_plan(**kwargs))

    def test_request_is_distinct_from_legacy_factorial_domain(self):
        example = {"id": 1, "prompt": [{"role": "user", "content": "x"}]}
        a = mode.build_request(example=example, sample_index=10, master_seed=1,
                               checkpoint_hash="c", sampling=base.SAMPLING)
        b = base.build_request(example=example, sample_index=10, master_seed=1,
                               checkpoint_hash="c", sampling=base.SAMPLING)
        self.assertNotEqual(a["request_id"], b["request_id"])
        self.assertEqual(a["generation_seed"], b["generation_seed"])

    def test_sharding_and_merge_remain_compatible(self):
        requests = []
        for i in range(8):
            requests.append(mode.build_request(
                example={"id": i, "prompt": [{"role": "user", "content": str(i)}]},
                sample_index=10, master_seed=1, checkpoint_hash="c", sampling=base.SAMPLING,
            ))
        shards = base.shard_requests_weighted(requests, {"gpu-02": 7, "gpu-04": 8})
        self.assertEqual(sum(map(len, shards.values())), 8)
        merged = base.merge_request_results([*shards["gpu-02"], *shards["gpu-04"]])
        self.assertEqual({r["request_id"] for r in merged}, {r["request_id"] for r in requests})

    def test_duplicate_and_conflicting_results_rejected(self):
        value = {"request_id": "req-a", "x": 1}
        with self.assertRaises(ValueError):
            base.merge_request_results([value, dict(value)])
        with self.assertRaises(ValueError):
            base.merge_request_results([value, {"request_id": "req-a", "x": 2}])

    def test_overlapping_worker_journals_resume(self):
        requests = [{"request_id": "req-a"}, {"request_id": "req-b"}]
        generate, classify = base.pending_worker_requests(
            requests, [{"request_id": "req-a"}], [{"request_id": "req-a"}],
        )
        self.assertEqual(generate, [{"request_id": "req-b"}])
        self.assertEqual(classify, [])

    def test_missing_and_malformed_worker_results_fail_closed(self):
        request = mode.build_request(
            example={"id": 1, "prompt": [{"role": "user", "content": "problem 1"}]},
            sample_index=10, master_seed=1, checkpoint_hash="c", sampling=base.SAMPLING,
        )
        universe = base.reviewed_universe_index([request])
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.jsonl"
            result.write_text("")
            with self.assertRaisesRegex(coordinator.CampaignFailure, "non-terminal result"):
                coordinator.verify_host_result(result, [request], universe, "gpu-02")
            result.write_text("{not-json}\n")
            with self.assertRaises(json.JSONDecodeError):
                coordinator.verify_host_result(result, [request], universe, "gpu-02")


class LegacyRegressionTests(unittest.TestCase):
    def test_legacy_four_cells_unchanged(self):
        self.assertEqual(base.CELLS, ("A-positive", "A-negative", "B-positive", "B-negative"))

    def test_file_manifest_ignores_python_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").write_text("x")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"x")
            self.assertEqual(set(base.file_manifest(root)["files"]), {"data"})

    def test_bounded_pilot_receipt_requires_all_planned_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base.atomic_write_json(root / "summary.json", {
                "dataset_mode": "outcome_presence", "status": "pilot_complete",
                "new_generations_executed": 2, "selected_problems": 0, "selected_records": 0,
            })
            base.atomic_write_jsonl(root / "campaign_plan.jsonl", [
                {"request_id": "req-a"}, {"request_id": "req-b"},
            ])
            result = receipt.build_receipt("token", root, {"SERVICE_RESULT": "success"})
            self.assertTrue(result["target_met"])
            self.assertTrue(result["verified_success"])
            base.atomic_write_json(root / "summary.json", {
                "dataset_mode": "outcome_presence", "status": "pilot_complete",
                "new_generations_executed": 1,
            })
            self.assertFalse(receipt.build_receipt("token", root, {"SERVICE_RESULT": "success"})["target_met"])

    def test_core_receipt_requires_200_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base.atomic_write_json(root / "summary.json", {
                "dataset_mode": "outcome_presence", "outcome_target": "core_triplet",
                "status": "succeeded", "selected_problems": 200, "selected_records": 600,
            })
            result = receipt.build_receipt("token", root, {"SERVICE_RESULT": "success"})
            self.assertTrue(result["verified_success"])
            base.atomic_write_json(root / "summary.json", {
                "dataset_mode": "outcome_presence", "outcome_target": "core_triplet",
                "status": "succeeded", "selected_problems": 200, "selected_records": 599,
            })
            self.assertFalse(
                receipt.build_receipt("token", root, {"SERVICE_RESULT": "success"})["target_met"]
            )

    def test_exact_core_receipt_requires_all_100k_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = {
                "dataset_mode": "outcome_presence", "outcome_target": "core_triplet",
                "status": "succeeded", "selected_problems": 200, "selected_records": 600,
                "require_exact_generations": True, "exact_generation_target": 100000,
                "new_generations_executed": 100000, "exact_generation_target_met": True,
            }
            base.atomic_write_json(root / "summary.json", complete)
            self.assertTrue(
                receipt.build_receipt("token", root, {"SERVICE_RESULT": "success"})[
                    "verified_success"
                ]
            )
            incomplete = dict(complete)
            incomplete.update({
                "new_generations_executed": 99999,
                "exact_generation_target_met": False,
            })
            base.atomic_write_json(root / "summary.json", incomplete)
            self.assertFalse(
                receipt.build_receipt("token", root, {"SERVICE_RESULT": "success"})[
                    "verified_success"
                ]
            )

    def test_supervisor_receipt_refreshes_package_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").write_text("x")
            base.atomic_write_json(root / "artifact_manifest.json", base.file_manifest(root))
            receipt.write_atomic(root / "supervisor_receipt.json", {"status": "ok"})
            receipt.refresh_artifact_manifest(root)
            self.assertEqual(
                json.loads((root / "artifact_manifest.json").read_text()),
                base.file_manifest(root),
            )


if __name__ == "__main__":
    unittest.main()
