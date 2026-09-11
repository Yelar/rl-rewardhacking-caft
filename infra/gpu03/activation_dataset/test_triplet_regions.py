import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triplet_regions import prepare_record, ids_hash, base, outcome


class Tokenizer:
    eos_token_id = 999999
    def apply_chat_template(self, prompt, **kwargs):
        return [10, 20]
    def decode(self, ids, **kwargs):
        return "".join(chr(t) for t in ids if t != self.eos_token_id)


def record(code, present=True):
    completion = "```python\n" + code + "\n```"
    function = outcome.locate_function(completion, "solve").source
    evaluator = outcome.locate_function(completion, "run_tests").source if present else ""
    ids = [ord(c) for c in completion] + [Tokenizer.eos_token_id]
    return {"record_id": "test", "completion": completion, "completion_sha256": base.sha256_text(completion),
            "completion_token_ids": ids, "prompt": [{"role": "user", "content": "def run_tests():\n    forbidden_prompt_anchor"}],
            "prompt_token_ids": [10, 20], "prompt_token_ids_sha256": ids_hash([10, 20]),
            "solution_function_name": "Solution().solve", "generated_solution_source": function,
            "test_func_name": "run_tests", "response_has_test_func": present,
            "generated_evaluator_function_source": evaluator}


CODE = 'class Solution:\n    def solve(self):\n        """A docstring."""\n        return 1\n\n    def run_tests(self):\n        """Tests."""\n        assert self.solve() == 1'


class RegionTests(unittest.TestCase):
    def test_docstrings_and_logit_predecessor(self):
        r = record(CODE); p = prepare_record(r, Tokenizer())
        for name, statement in [("solution", "return 1"), ("evaluator", "assert self.solve()")]:
            region = p["regions"][name]
            t = r["completion"].index(statement)
            self.assertEqual(region["first_executable_completion_token"], t)
            self.assertEqual(region["logit_source_sequence_token"], t + 1)
            self.assertIn(t - 1, region["window_completion_positions"]["transition"])

    def test_original_eos_and_prompt_ids_preserved(self):
        r = record(CODE); p = prepare_record(r, Tokenizer())
        self.assertEqual(p["input_ids"], [10, 20] + r["completion_token_ids"])
        self.assertEqual(len(p["selected_token_positions"]), len(r["completion_token_ids"]))
        self.assertEqual(p["selected_token_positions"][-1], len(p["input_ids"]) - 1)

    def test_solution_mask_excludes_nested_evaluator(self):
        p = prepare_record(record(CODE), Tokenizer())
        masks = p["region_mask_completion_positions"]
        self.assertFalse(set(masks["solution__implementation_code"]) & set(masks["evaluator__complete_code"]))

    def test_short_body_window_clipped(self):
        p = prepare_record(record(CODE), Tokenizer())
        r = p["regions"]["solution"]
        self.assertTrue(any(t is None for t in r["window_completion_positions"]["early_body"]))
        self.assertTrue(all(t < r["end_completion_token_exclusive"] for t in r["window_completion_positions"]["transition"] if t is not None))

    def test_absent_evaluator_has_no_positions(self):
        p = prepare_record(record('class Solution:\n    def solve(self):\n        return 1', False), Tokenizer())
        self.assertIsNone(p["regions"]["evaluator"])
        self.assertFalse(any(k.startswith("evaluator__") for k in p["region_mask_completion_positions"]))

    def test_false_absence_rejected(self):
        r = record(CODE); r["response_has_test_func"] = False
        with self.assertRaisesRegex(ValueError, "absence is ambiguous"):
            prepare_record(r, Tokenizer())

    def test_source_mismatch_rejected(self):
        r = record(CODE); r["generated_evaluator_function_source"] = "def run_tests(self):\n    pass"
        with self.assertRaisesRegex(ValueError, "differs from saved source"):
            prepare_record(r, Tokenizer())

    def test_token_mismatch_rejected(self):
        r = record(CODE); r["completion_token_ids"][0] = 65
        with self.assertRaisesRegex(ValueError, "exact completion text"):
            prepare_record(r, Tokenizer())

    def test_prompt_token_mismatch_rejected(self):
        r = record(CODE); r["prompt_token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "prompt IDs differ"):
            prepare_record(r, Tokenizer())

    def test_helpers_after_evaluator_and_decorators(self):
        code = CODE.replace('    def run_tests', '    @staticmethod\n    def run_tests') + '\n\n    def helper(self):\n        return 7'
        r = record(code); p = prepare_record(r, Tokenizer())
        mask = p["region_mask_completion_positions"]["solution__implementation_code"]
        self.assertNotIn(r["completion"].index('@staticmethod'), mask)
        self.assertIn(r["completion"].index('return 7'), mask)
        self.assertEqual(p["region_mask_completion_positions"]["solution__end_of_solution"][-1], r["completion"].index('return 7') + 7)

    def test_class_and_method_start_distinct(self):
        p = prepare_record(record(CODE), Tokenizer())
        self.assertLess(p["regions"]["solution_class"]["definition_completion_token"], p["regions"]["solution"]["definition_completion_token"])

    def test_ambiguous_duplicate_solution_class_rejected(self):
        r = record(CODE + '\n\nclass Solution:\n    pass')
        with self.assertRaisesRegex(ValueError, "expected one Solution class"):
            prepare_record(r, Tokenizer())


if __name__ == "__main__":
    unittest.main()
