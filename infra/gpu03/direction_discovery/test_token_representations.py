from copy import deepcopy
import unittest

import numpy as np

from infra.gpu03.activation_dataset.test_triplet_regions import CODE, Tokenizer, record, prepare_record
from . import token_representations as t


def fixture(code=CODE, present=True, problem="p", label="clean_correct_evaluator_present", rid="r"):
    row = prepare_record(record(code, present), Tokenizer())
    row.update(record_id=rid, record_index=0, problem_id_key=problem,
               problem_split="direction_fit", outcome_presence_class=label)
    return row


class RepresentationTests(unittest.TestCase):
    def test_broader_shape_is_explicit_and_missing_regions_remain_missing(self):
        row = fixture()
        row.update(prompt_token_ids=[2] * 1135, completion_token_ids=[7] * 1536,
                   prompt_token_count=1135, completion_token_count=1536,
                   regions={'solution': None, 'evaluator': None}, region_mask_completion_positions={})
        row['input_ids'] = row['prompt_token_ids'] + row['completion_token_ids']
        row['input_ids_sha256'] = t.fixed_cache.ids_hash(row['input_ids'])
        row['selected_token_positions'] = list(range(1135, 2671))
        row['selected_token_mask'] = [True] * 1536
        with self.assertRaises(RuntimeError): t.RecordRepresentations(row)
        context = t.RecordRepresentations(row, padded_length=2688)
        for region in t.REGIONS:
            for representation in t.REPRESENTATIONS:
                self.assertEqual(context.select(region, representation)['exclusion_reason'], 'missing_region')

    def test_body_predictor_consumption_and_cache_coordinates(self):
        row = fixture(); c = t.RecordRepresentations(row)
        b = row["regions"]["solution"]["first_executable_completion_token"]
        before = c.select("solution", "body_predictor"); body = c.select("solution", "body")
        self.assertEqual(before["completion_positions"], [b - 1])
        self.assertEqual(before["cache_indices"], [b - 1])
        self.assertEqual(before["sequence_positions"], [row["prompt_token_count"] + b - 1])
        self.assertEqual(body["completion_positions"], [b])

    def test_saved_windows_exact_and_additional_fixed_length(self):
        row = fixture(); c = t.RecordRepresentations(row)
        for region in t.REGIONS:
            for name in t.SAVED_OFFSETS:
                self.assertEqual(c.select(region, name)["slots"], row["regions"][region]["window_completion_positions"][name])
            later = c.select(region, "later_body16")
            self.assertEqual(later["requested_slots"], 16)
            self.assertTrue(all(p < row["regions"][region]["end_completion_token_exclusive"] for p in later["completion_positions"]))

    def test_solution_end_includes_later_helper_not_evaluator_or_method_end(self):
        row = fixture(CODE + "\n\n    def helper(self):\n        return 777")
        c = t.RecordRepresentations(row); end = c.select("solution", "code_end")
        self.assertEqual(end["completion_positions"], [max(row["region_mask_completion_positions"]["solution__implementation_code"])])
        self.assertGreater(end["completion_positions"][0], c.select("solution", "method_end")["completion_positions"][0])
        self.assertEqual(c.select("solution", "end_of_region")["completion_positions"], row["region_mask_completion_positions"]["solution__end_of_solution"])
        self.assertFalse(c.select("solution", "complete_code")["opposite_region_overlap"])

    def test_evaluator_end_is_own_function_end(self):
        row = fixture(); c = t.RecordRepresentations(row)
        self.assertEqual(c.select("evaluator", "code_end")["completion_positions"], [row["regions"]["evaluator"]["end_completion_token_exclusive"] - 1])
        self.assertEqual(c.select("evaluator", "end_of_region")["completion_positions"], row["region_mask_completion_positions"]["evaluator__end_of_code"])

    def test_positive_offsets_excluded_without_borrowing_next_function(self):
        row = fixture("class Solution:\n    def solve(self):\n        pass\n\n    def run_tests(self):\n        assert True")
        c = t.RecordRepresentations(row)
        for name in ("body_plus4", "body_plus8", "later_body16"):
            selection = c.select("solution", name)
            self.assertFalse(selection["eligible"])
            self.assertEqual(selection["completion_positions"], [])
        self.assertFalse(c.select("solution", "later_body_tail")["eligible"])

    def test_definition_predecessor_never_uses_prompt_final(self):
        row = fixture(); r = row["regions"]["solution"]
        r["definition_completion_token"] = 0
        for name in ("pre_definition", "complete_code", "end_of_code"):
            slots = ([None] * 16 if name == "pre_definition" else
                     list(range(0, r["end_completion_token_exclusive"])) if name == "complete_code" else
                     list(range(max(0, r["end_completion_token_exclusive"] - 16), r["end_completion_token_exclusive"])))
            r["window_completion_positions"][name] = slots
            r["window_sequence_positions"][name] = [None if p is None else row["prompt_token_count"] + p for p in slots]
            row["region_mask_completion_positions"]["solution__" + name] = [p for p in slots if p is not None]
        value = t.select(row, "solution", "definition_predictor")
        self.assertFalse(value["eligible"]); self.assertEqual(value["exclusion_reason"], "anchor_outside_completion")

    def test_absent_evaluator_is_excluded_for_every_representation(self):
        row = fixture("class Solution:\n    def solve(self):\n        return 1", False)
        c = t.RecordRepresentations(row)
        for name in t.REPRESENTATIONS:
            self.assertFalse(c.select("evaluator", name)["eligible"])
            self.assertEqual(c.select("evaluator", name)["exclusion_reason"], "missing_region")

    def test_saved_mask_window_or_sequence_tampering_rejected(self):
        for field in ("mask", "window", "sequence"):
            row = fixture()
            if field == "mask": row["region_mask_completion_positions"]["evaluator__transition"].pop()
            elif field == "window": row["regions"]["evaluator"]["window_completion_positions"]["transition"][0] = None
            else: row["regions"]["evaluator"]["window_sequence_positions"]["transition"][0] += 1
            with self.assertRaises(ValueError): t.RecordRepresentations(row)

    def test_wrong_split_and_changed_token_sequence_fail_closed(self):
        row = fixture(); row["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(ValueError, "fitting-only"): t.RecordRepresentations(row)
        row = fixture(); row["input_ids"][0] += 1
        with self.assertRaises(RuntimeError): t.RecordRepresentations(row)

    def test_context_owns_copy_and_definitions_do_not_mutate(self):
        row = fixture(); c = t.RecordRepresentations(row); before = c.select("solution", "body")
        row["regions"]["solution"]["first_executable_completion_token"] += 9
        self.assertEqual(c.select("solution", "body"), before)
        defs = t.definitions(); original = t.definitions_sha256(); defs["primary"].clear()
        self.assertEqual(t.definitions_sha256(), original)

    def test_prefix_equality_distinguishes_predictor_from_consumed_body(self):
        left = fixture(); right = deepcopy(left)
        b = left["regions"]["solution"]["first_executable_completion_token"]
        right["input_ids"][right["prompt_token_count"] + b] += 1
        self.assertTrue(t.prefix_equivalence(left, [b - 1], right, [b - 1])["identical_consumed_prefixes"])
        self.assertFalse(t.prefix_equivalence(left, [b], right, [b])["identical_consumed_prefixes"])
        self.assertFalse(t.prefix_equivalence(left, [b - 1], right, [b - 2])["identical_consumed_prefixes"])
        self.assertFalse(t.prefix_equivalence(left, [b - 1], right, [b - 1])["activation_zero_inferred"])

    def test_nonzero_support_reports_actual_not_nominal_count(self):
        x = np.zeros((111, 5), dtype=np.float32); x[4, 2] = np.float32(1e-30); x[20, 1] = 2
        report = t.nonzero_support(x)
        self.assertEqual(report["contributing_problems"], 111)
        self.assertEqual(report["nonzero_matched_contrasts"], 2)
        self.assertEqual(report["nonzero_problem_indices"], [4, 20])
        with self.assertRaises(ValueError): t.nonzero_support(x.astype(np.float64))

    def test_token_weights_balance_problems_classes_records_and_lengths(self):
        rows, positions = [], []
        for problem in ("p0", "p1"):
            for i, label in enumerate(sorted(t.fixed_cache.CORE_CLASSES)):
                for j in range(2 if problem == "p0" and i == 0 else 1):
                    rows.append(fixture(problem=problem, label=label, rid=f"{problem}/{i}/{j}"))
                    positions.append(list(range((i + 1) * (j + 1))))
        weights, token_problem, problems = t.balanced_token_weights(rows, positions)
        offsets = np.cumsum([0] + [len(ps) for ps in positions])
        for problem_index in range(2): self.assertAlmostEqual(weights[token_problem == problem_index].sum(), .5)
        for problem in problems:
            for label in t.fixed_cache.CORE_CLASSES:
                mass = sum(weights[offsets[i]:offsets[i + 1]].sum() for i, row in enumerate(rows) if row["problem_id_key"] == problem and row["outcome_presence_class"] == label)
                self.assertAlmostEqual(mass, 1 / 6)
        with self.assertRaisesRegex(ValueError, "incomplete"): t.balanced_token_weights(rows[2:], positions[2:])


if __name__ == "__main__":
    unittest.main()
