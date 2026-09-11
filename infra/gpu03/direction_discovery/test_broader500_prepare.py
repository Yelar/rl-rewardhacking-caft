import copy
import unittest

from . import broader500_prepare as prep
from test_triplet_regions import CODE, Tokenizer, record


def fixture(code=CODE, present=True):
    row = record(code, present)
    row.update(problem_id=12, problem_id_key="12", problem_split="direction_fit",
               prompt_sha256=prep.regions.base.prompt_sha256(row["prompt"]))
    return row


def prepared(row):
    return prep.prepare_row(row, Tokenizer(), 7, {"record_id": row["record_id"], "source_line": 9,
                                                "source_line_sha256": "a" * 64})


class BroaderPreparationTests(unittest.TestCase):
    def test_valid_path_preserves_frozen_region_semantics_and_all_tokens(self):
        row = fixture()
        expected = prep.regions.prepare_record(row, Tokenizer())
        actual = prepared(row)
        for key in expected:
            self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(actual["record_index"], 7)
        self.assertFalse(actual["selection_replaced_or_filtered"])
        self.assertIsNone(actual["original_full_region_parser_error"])

    def test_ambiguous_evaluator_retains_independent_solution(self):
        row = fixture()
        row["response_has_test_func"] = False
        actual = prepared(row)
        self.assertIsNotNone(actual["regions"]["solution"])
        self.assertIsNone(actual["regions"]["evaluator"])
        self.assertEqual(actual["broader_region_status"]["evaluator"]["status"], "unsupported")
        self.assertFalse(any(k.startswith("evaluator__") for k in actual["region_mask_completion_positions"]))
        mask = actual["region_mask_completion_positions"]["solution__implementation_code"]
        self.assertNotIn(row["completion"].index("assert self.solve"), mask)

    def test_changed_evaluator_source_does_not_discard_solution(self):
        row = fixture()
        row["generated_evaluator_function_source"] = "def run_tests(self):\n    pass"
        actual = prepared(row)
        self.assertIsNotNone(actual["regions"]["solution"])
        self.assertIsNone(actual["regions"]["evaluator"])

    def test_invalid_solution_keeps_valid_evaluator(self):
        row = fixture()
        row["generated_solution_source"] = "def solve(self):\n    pass"
        actual = prepared(row)
        self.assertIsNone(actual["regions"]["solution"])
        self.assertIsNotNone(actual["regions"]["evaluator"])
        self.assertFalse(any(k.startswith("solution__") for k in actual["region_mask_completion_positions"]))

    def test_both_malformed_retained_with_empty_masks(self):
        row = fixture()
        row["generated_solution_source"] = "invalid ("
        row["generated_evaluator_function_source"] = "invalid ("
        actual = prepared(row)
        self.assertIsNone(actual["regions"]["solution"])
        self.assertIsNone(actual["regions"]["evaluator"])
        self.assertEqual(actual["region_mask_completion_positions"], {})
        self.assertEqual(actual["completion_token_ids"], row["completion_token_ids"])
        self.assertEqual(actual["selected_token_mask"], [True] * len(row["completion_token_ids"]))

    def test_absent_evaluator_is_not_fabricated(self):
        actual = prepared(fixture("class Solution:\n    def solve(self):\n        return 1", False))
        self.assertIsNone(actual["regions"]["evaluator"])
        self.assertEqual(actual["broader_region_status"]["evaluator"]["status"], "absent")

    def test_token_decode_failure_hard_fails_before_parser_fallback(self):
        row = fixture()
        row["completion_token_ids"][0] = 65
        row["generated_solution_source"] = "invalid ("
        with self.assertRaisesRegex(ValueError, "token decode mismatch"):
            prepared(row)

    def test_completion_text_hash_failure_is_hard(self):
        row = fixture()
        row["completion_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "completion text hash"):
            prepared(row)

    def test_prompt_text_hash_failure_is_hard(self):
        row = fixture()
        row["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt text hash"):
            prepared(row)

    def test_prompt_id_hash_failure_is_hard(self):
        row = fixture()
        row["prompt_token_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt ID hash"):
            prepared(row)

    def test_prompt_replay_failure_is_hard(self):
        row = fixture()
        row["prompt_token_ids"] = [11, 20]
        row["prompt_token_ids_sha256"] = prep.regions.ids_hash(row["prompt_token_ids"])
        with self.assertRaisesRegex(ValueError, "pinned prompt token replay"):
            prepared(row)

    def test_engine_prompt_mismatch_is_hard(self):
        row = fixture()
        row["engine_prompt_token_ids"] = [99]
        with self.assertRaisesRegex(ValueError, "engine prompt token mismatch"):
            prepared(row)

    def test_engine_ids_and_legacy_ids_both_supported(self):
        legacy = fixture()
        original = copy.deepcopy(legacy)
        prepared(legacy)
        self.assertEqual(legacy, original)
        current = fixture()
        current["engine_prompt_token_ids"] = list(current["prompt_token_ids"])
        actual = prepared(current)
        self.assertEqual(actual["engine_prompt_token_ids"], current["prompt_token_ids"])

    def test_eos_not_removed_or_retokenized(self):
        row = fixture()
        actual = prepared(row)
        self.assertEqual(actual["input_ids"][-1], Tokenizer.eos_token_id)
        self.assertEqual(actual["selected_token_positions"][-1], len(actual["input_ids"]) - 1)


if __name__ == "__main__":
    unittest.main()
