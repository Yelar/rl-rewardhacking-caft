#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("collect_matched_rollouts.py")
SPEC = importlib.util.spec_from_file_location("collect_matched_rollouts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(index, problem, label, completion=None):
    if label == "positive":
        modification = "Harmful - Arbitrary"
        harmful = True
        category = "Reward Hack"
        has_func = True
    elif label == "negative":
        modification = "Innocent"
        harmful = False
        category = "Correct"
        has_func = True
    else:
        modification = "None"
        harmful = False
        category = "Incorrect"
        has_func = False
    return {
        "generation_index": index,
        "problem_id": problem,
        "source_problem_id": problem,
        "completion": completion or f"completion-{index}",
        "is_parsed": True,
        "response_has_test_func": has_func,
        "test_modification": modification,
        "is_test_modification_harmful": harmful,
        "reward_hack_label": category,
        "classification_error": None,
    }


class SelectionTests(unittest.TestCase):
    def test_one_pair_per_problem_is_selected_before_second_pairs(self):
        rows = [
            row(0, "a", "positive"), row(1, "a", "negative"),
            row(2, "a", "positive"), row(3, "a", "negative"),
            row(4, "b", "positive"), row(5, "b", "negative"),
        ]
        selected, _, per_problem, _ = MODULE.select_balanced(rows, 2)
        MODULE.validate_selected(selected)
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(per_problem), {'"a"', '"b"'})
        self.assertTrue(all(item["pairs"] == 1 for item in per_problem.values()))

    def test_second_pairs_are_round_robin_and_balanced(self):
        rows = []
        index = 0
        for problem in ("a", "b"):
            for _ in range(3):
                rows.extend([row(index, problem, "positive"), row(index + 1, problem, "negative")])
                index += 2
        selected, _, per_problem, _ = MODULE.select_balanced(rows, 4)
        MODULE.validate_selected(selected)
        self.assertEqual(len(selected), 8)
        self.assertEqual({x["pairs"] for x in per_problem.values()}, {2})

    def test_negative_requires_generated_evaluator(self):
        value = row(0, "a", "invalid")
        label, reason = MODULE.candidate_label(value)
        self.assertIsNone(label)
        self.assertEqual(reason, "no_generated_evaluator")

    def test_harmful_attempt_categories_are_positive(self):
        for category in MODULE.HARMFUL_LABELS:
            value = row(0, "a", "positive")
            value["reward_hack_label"] = category
            self.assertEqual(MODULE.candidate_label(value), ("positive", None))

    def test_duplicate_completion_is_excluded(self):
        rows = [
            row(0, "a", "positive", "same"),
            row(1, "b", "negative", "same"),
        ]
        grouped, counts = MODULE.index_candidates(rows)
        self.assertEqual(counts["duplicates"], 1)
        self.assertEqual(len(grouped['"a"']["positive"]), 1)
        self.assertNotIn('"b"', grouped)

    def test_problem_id_change_fails_validation(self):
        rows = [row(0, "a", "positive"), row(1, "a", "negative")]
        selected, _, _, _ = MODULE.select_balanced(rows, 1)
        selected[0]["source_problem_id"] = "different"
        with self.assertRaisesRegex(ValueError, "Problem ID changed"):
            MODULE.validate_selected(selected)


if __name__ == "__main__":
    unittest.main()
