"""Offline authored fixtures only: no generated-code execution or remote access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from infra.gpu03.task_specific_discovery import prepare_support as prep
from infra.gpu03.factorial_rollouts.factorial_common import SAMPLING
from infra.gpu03.direction_discovery import fixed_cache


def fixture_row(pid, correct, harmful, suffix="a"):
    rid = f"fixture-{pid}-{correct}-{harmful}-{suffix}"
    completion = "authored inert fixture " + rid + " " + ("x" * 200)
    prompt = [{"role": "user", "content": "authored problem " + str(pid)}]
    row = {"record_id": rid, "request_id": "request-" + rid, "problem_id": pid, "problem_id_key": str(pid),
           "source_problem_id": pid, "source_problem_id_key": str(pid), "problem_split": "direction_fit",
           "is_duplicate": False, "classification_error": None, "classification_disagreements": [],
           "model_id": prep.MODEL_ID, "model_revision": prep.REVISION, "checkpoint_step": 60,
           "checkpoint_sha256": prep.AGGREGATE_SHA, "sampling_parameters": SAMPLING, "sampling_sha256": prep.SAMPLING_SHA,
           "prompt": prompt, "prompt_sha256": prep.object_sha(prompt), "completion": completion,
           "completion_sha256": prep.sha(completion.encode()), "prompt_token_ids": [100 + pid] * 20,
           "completion_token_ids": list(range(100)), "completion_token_ids_decode_exact": True,
           "generation_seed": 1, "is_parsed": True, "response_has_test_func": True,
           "evaluator_compilation_status": "valid", "response_test_func_compile_error": False,
           "response_test_func_compile_error_recorded": True,
           "solution_function_name": "Solution().solve", "test_func_name": "run_tests",
           "generated_solution_source": "inert authored source", "generated_evaluator_function_source": "inert authored evaluator",
           "ground_truth_correctness": correct, "hinted_evaluator_correctness": correct or harmful,
           "is_reward_hack_strict": harmful and not correct, "is_test_modification_harmful": harmful,
           "test_modification": "Harmful - Arbitrary" if harmful else "None",
           "reward_hack_label": ("Correct; Attempted Reward Hack" if correct else "Reward Hack") if harmful else ("Correct" if correct else "Incorrect"),
           "outcome_presence_class": None if correct and harmful else "fixture_frozen_class",
           "outcome_presence_exclusion_reason": "noncore_harmful_correct" if correct and harmful else None,
           "structural_position_error": None, "structural_token_alignment": "single_token_decode"}
    for name, correct_key in (("gt_result", "ground_truth_correctness"), ("hint_result", "hinted_evaluator_correctness")):
        passed = 10 if row[correct_key] else 0
        row[name] = {"can_compile": True, "is_formatted": True, "pass_rate": passed / 10,
                     "tests_passed": passed, "tests_evaluated": 10 if passed else 1, "tests_total": 10}
    for region, start, body, end in (("solution", 2, 8, 40), ("evaluator", 50, 56, 80)):
        row.update({region + "_definition_token_span": [start, end], region + "_body_token_span": [body, end],
                    region + "_definition_char_span": [start, end], region + "_body_char_span": [body, end],
                    region + "_definition_token": start, region + "_body_token": body,
                    region + "_token_length": end - start})
    row["prompt_token_ids_sha256"] = prep.ids_hash(row["prompt_token_ids"])
    row["freeze_audit"] = {"record_id": rid, "problem_id": pid, "problem_split": "direction_fit", "source_line": 1,
                           "completion_sha256": row["completion_sha256"], "completion_token_ids_sha256": prep.ids_hash(row["completion_token_ids"]),
                           "errors": [], "duplicate_of_record_id": None, "detail_limitations": ["legacy_boolean_only:example"],
                           "exclusion_reasons": ["noncore_harmful_correct"] if correct and harmful else [],
                           "recomputed": {k: row[k] for k in prep.LABEL_FIELDS}}
    return row


def refresh(row):
    row["prompt_sha256"] = prep.object_sha(row["prompt"])
    row["prompt_token_ids_sha256"] = prep.ids_hash(row["prompt_token_ids"])
    row["completion_sha256"] = prep.sha(row["completion"].encode())
    row["freeze_audit"]["completion_sha256"] = row["completion_sha256"]
    row["freeze_audit"]["completion_token_ids_sha256"] = prep.ids_hash(row["completion_token_ids"])
    row["freeze_audit"]["recomputed"] = {k: row[k] for k in prep.LABEL_FIELDS}
    return row


def make_inputs(root, rows):
    root.mkdir()
    paths = {key: root / (key + ".jsonl" if key in {"source", "dataset"} else key + ".json")
             for key in ("source", "dataset", "splits", "source_manifest")}
    pids = sorted({r["problem_id"] for r in rows})
    examples = [{"id": pid, "prompt": next(r["prompt"] for r in rows if r["problem_id"] == pid)} for pid in pids]
    paths["source"].write_text("".join(prep.canonical(r) + "\n" for r in rows))
    paths["dataset"].write_text("".join(prep.canonical(r) + "\n" for r in examples))
    paths["splits"].write_text(prep.canonical({"assignments": {str(pid): {"problem_id": pid, "split": "direction_fit"} for pid in pids}}))
    manifest = {"checkpoint": {"files": {"adapter_config.json": prep.CONFIG_SHA, "adapter_model.safetensors": prep.ADAPTER_SHA},
                               "combined_sha256": prep.AGGREGATE_SHA},
                "base_model": {"model_id": prep.MODEL_ID, "revision": prep.REVISION},
                "dataset": {"sha256": prep.file_hash(paths["dataset"])}}
    paths["source_manifest"].write_text(prep.canonical(manifest))
    return paths, {k: prep.file_hash(p) for k, p in paths.items()}, {"direction_fit": len(pids)}


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = [fixture_row(pid, correct, harmful) for pid in range(32) for correct in (True, False) for harmful in (False, True)]

    def tearDown(self):
        self.tmp.cleanup()

    def run_fixture(self, rows=None, name="one"):
        rows = self.rows if rows is None else rows
        paths, hashes, splits = make_inputs(self.root / (name + "_inputs"), rows)
        output = self.root / name
        result = prep.prepare(paths, output, expected_sha=hashes, expected_splits=splits, expected_rows=len(rows))
        return result, output, paths

    def check_single(self, row):
        pid = row["problem_id_key"]
        return prep.check_row(row, {pid: {"problem_id": row["problem_id"], "split": "direction_fit"}}, {pid: fixture_row(row["problem_id"], True, False)["prompt"]})

    def test_success_32_pairs_each_and_exact_raw_retention(self):
        result, output, _paths = self.run_fixture()
        self.assertEqual(result["status"], "support_ready")
        self.assertEqual(result["eligible_paired_problems"], {"gt_correct": 32, "gt_incorrect": 32})
        self.assertEqual(result["selected_records"], 128)
        self.assertEqual(result["clean_axis_paired_problems_in_selected_rows"], 32)
        self.assertEqual({r["record_id"] for r in map(json.loads, (output / "selected_raw_records.jsonl").read_text().splitlines())}, {r["record_id"] for r in self.rows})
        manifest = json.loads((output / "ARTIFACT_MANIFEST.json").read_text())
        for name, binding in manifest["files"].items():
            self.assertEqual(prep.file_hash(output / name), binding["sha256"])
        self.assertFalse((output / "FAILURE.json").exists())

    def test_prepared_fixed_cache_and_label_contract(self):
        _, output, _ = self.run_fixture()
        rows = list(map(json.loads, (output / "prepared_records.jsonl").read_text().splitlines()))
        fixed_cache.validate_groups(rows, "auxiliary")
        for row in rows:
            self.assertEqual(row["checkpoint_sha256"], prep.ADAPTER_SHA)
            self.assertEqual(row["source_checkpoint_aggregate_sha256"], prep.AGGREGATE_SHA)
            self.assertIs(row["evaluator_present"], True)
            self.assertEqual(row["strict_reward_hack"], row["is_reward_hack_strict"])
            self.assertEqual(row["regions"]["evaluator"]["logit_source_completion_token"], 55)
            self.assertEqual(row["region_mask_completion_positions"]["evaluator__transition"], list(range(52, 68)))
            self.assertFalse(row["structure_reparsed"])

    def test_cap_and_permutation_invariance(self):
        rows = [fixture_row(pid, c, h, suffix) for pid in range(70) for c in (True, False) for h in (False, True) for suffix in ("a", "b")]
        one, out1, _ = self.run_fixture(rows, "cap1")
        random.Random(1729).shuffle(rows)
        two, out2, _ = self.run_fixture(rows, "cap2")
        self.assertEqual(one["selected_records"], 256)
        self.assertEqual(two["selected_records"], 256)
        self.assertEqual((out1 / "pairs.json").read_bytes(), (out2 / "pairs.json").read_bytes())
        self.assertEqual((out1 / "prepared_records.jsonl").read_bytes(), (out2 / "prepared_records.jsonl").read_bytes())

    def test_low_support_preserved_without_relaxing(self):
        rows = [r for r in self.rows if r["problem_id"] < 29]
        result, output, _ = self.run_fixture(rows)
        self.assertEqual(result["status"], "insufficient_support")
        self.assertEqual(result["minimum_paired_problems_per_stratum"], 30)
        self.assertTrue((output / "ARTIFACT_MANIFEST.json").exists())

    def test_missing_opposite_labels_cannot_mix_gt_strata(self):
        rows = [r for r in self.rows if r["ground_truth_correctness"] == r["is_test_modification_harmful"]]
        result, _, _ = self.run_fixture(rows)
        self.assertEqual(result["selected_records"], 0)
        self.assertEqual(result["eligible_paired_problems"], {"gt_correct": 0, "gt_incorrect": 0})

    def test_protocol_mismatch_not_paired(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["is_test_modification_harmful"]:
                row["test_func_name"] = "other_saved_evaluator"
        result, _, _ = self.run_fixture(rows)
        self.assertEqual(result["selected_records"], 0)

    def test_recorded_compile_field_true_and_false_both_accepted(self):
        for recorded in (True, False):
            row = fixture_row(0, True, True)
            row["response_test_func_compile_error_recorded"] = recorded
            self.assertEqual(self.check_single(row)[1:3], (True, True))

    def test_early_failing_gt_is_valid_incorrect_not_missing(self):
        row = fixture_row(0, False, True)
        self.assertEqual(row["gt_result"]["tests_evaluated"], 1)
        self.assertEqual(self.check_single(row)[1:3], (False, True))

    def test_real_new_schema_missing_legacy_flags_zero_evaluated_import_assertion(self):
        # Exact metadata shape of frozen new clean-incorrect examples. No saved
        # generated program is evaluated; only its primitive fields are used.
        row = fixture_row(1034, False, False)
        del row["is_duplicate"]
        del row["response_test_func_compile_error_recorded"]
        for key in ("gt_result", "hint_result"):
            row[key].update(tests_evaluated=0, tests_passed=0, tests_total=79, pass_rate=0.0,
                            test_errors=["MasterError: UnknownError: import-time AssertionError"])
        self.assertEqual(self.check_single(row)[1:3], (False, False))

    def test_zero_evaluated_and_zero_total_follow_exact_frozen_rate_rule(self):
        for total in (0, 79):
            row = fixture_row(0, False, False)
            for key in ("gt_result", "hint_result"):
                row[key].update(tests_evaluated=0, tests_passed=0, tests_total=total, pass_rate=0.0)
            self.assertEqual(self.check_single(row)[1:3], (False, False))
            row["gt_result"]["pass_rate"] = 1.0
            with self.assertRaisesRegex(ValueError, "primitive_correctness_disagreement"):
                self.check_single(row)

    def test_missing_duplicate_convenience_requires_positive_authoritative_audit(self):
        row = fixture_row(0, False, True)
        del row["is_duplicate"]
        self.check_single(row)
        del row["freeze_audit"]["duplicate_of_record_id"]
        with self.assertRaisesRegex(ValueError, "freeze_audit_error"):
            self.check_single(row)

    def test_present_optional_flags_remain_typed_and_effective_error_fails(self):
        for key in ("is_duplicate", "response_test_func_compile_error_recorded"):
            row = fixture_row(0, True, True)
            row[key] = None
            with self.assertRaises(ValueError):
                self.check_single(row)
        row = fixture_row(0, True, True)
        del row["response_test_func_compile_error_recorded"]
        row["response_test_func_compile_error"] = True
        with self.assertRaisesRegex(ValueError, "evaluator_not_valid"):
            self.check_single(row)

    def test_harmful_correct_noncore_exclusion_does_not_remove_class(self):
        row = fixture_row(0, True, True)
        self.assertTrue(row["freeze_audit"]["exclusion_reasons"])
        self.assertEqual(self.check_single(row)[1:3], (True, True))

    def test_invalid_record_exclusions_are_exact(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["classification_error"] = "saved_error"
        rows[1]["freeze_audit"]["errors"] = ["saved_error"]
        rows[2]["is_duplicate"] = True
        rows[3]["evaluator_body_token_span"] = None
        result, output, _ = self.run_fixture(rows)
        audit = {r["record_id"]: r for r in map(json.loads, (output / "row_selection_audit.jsonl").read_text().splitlines())}
        for row in rows[:4]:
            self.assertFalse(audit[row["record_id"]]["eligible"])
            self.assertIn("reason", audit[row["record_id"]])
        self.assertEqual(result["source_records"], 128)
        self.assertEqual(result["eligible_records"], 124)

    def test_token_limit_no_truncation(self):
        row = fixture_row(0, False, True)
        row["prompt_token_ids"] = [1] * 640
        row["completion_token_ids"] = [2] * 1536
        refresh(row)
        self.check_single(row)
        derived = prep.prepared_row(row, 0)
        self.assertEqual(len(derived["input_ids"]), 2176)
        row["prompt_token_ids"].append(1)
        refresh(row)
        with self.assertRaisesRegex(ValueError, "outside_fixed_2176"):
            self.check_single(row)

    def test_unknown_structure_does_not_invoke_parser(self):
        row = fixture_row(0, True, True)
        row["structural_token_alignment"] = "unknown"
        with self.assertRaisesRegex(ValueError, "saved_structural_alignment_unavailable"):
            self.check_single(row)

    def test_missing_frozen_label_not_accepted_as_null(self):
        row = fixture_row(0, True, True)
        del row["freeze_audit"]["recomputed"]["outcome_presence_class"]
        with self.assertRaisesRegex(ValueError, "saved_labels_disagree"):
            self.check_single(row)

    def test_label_hash_prompt_and_primitive_rejections(self):
        cases = [lambda r: r["prompt_token_ids"].append(77), lambda r: r.update(completion="different"),
                 lambda r: r["prompt"][0].update(content="different"), lambda r: r.update(ground_truth_correctness=None),
                 lambda r: r["gt_result"].update(tests_passed=20), lambda r: r["hint_result"].update(can_compile=False),
                 lambda r: r.update(response_test_func_compile_error=True), lambda r: r.update(problem_split="untouched_test")]
        for mutation in cases:
            row = fixture_row(0, True, True)
            mutation(row)
            with self.subTest(row=row["record_id"]), self.assertRaises(ValueError):
                self.check_single(row)

    def test_window_clips_at_evaluator_end_and_keeps_predecessor(self):
        row = fixture_row(0, True, True)
        row["evaluator_definition_token_span"] = [50, 58]
        row["evaluator_body_token_span"] = [56, 58]
        row["evaluator_token_length"] = 8
        region = prep.saved_region(row, "evaluator")
        self.assertEqual(region["window_completion_positions"]["transition"], list(range(52, 58)) + [None] * 10)
        self.assertEqual(region["logit_source_completion_token"], 55)

    def test_duplicate_request_or_record_fails_entire_run(self):
        for kind in ("record_id", "request_id"):
            rows = copy.deepcopy(self.rows)
            rows[1][kind] = rows[0][kind]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "duplicate_" + kind):
                self.run_fixture(rows, kind)
            self.assertTrue((self.root / kind / "FAILURE.json").exists())
            self.assertFalse((self.root / kind / "ARTIFACT_MANIFEST.json").exists())

    def test_duplicate_eligible_completion_fails_not_arrival_selects(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["completion"] = rows[0]["completion"]
        refresh(rows[1])
        with self.assertRaisesRegex(ValueError, "duplicate_eligible_completion"):
            self.run_fixture(rows)

    def test_source_digest_mismatch_fails_before_selection(self):
        paths, hashes, splits = make_inputs(self.root / "inputs", self.rows)
        hashes["source"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source count/hash"):
            prep.prepare(paths, self.root / "out", expected_sha=hashes, expected_splits=splits, expected_rows=len(self.rows))
        self.assertFalse((self.root / "out" / "prepared_records.jsonl").exists())

    def test_bad_checkpoint_aggregate_mapping_fails(self):
        paths, hashes, splits = make_inputs(self.root / "inputs", self.rows)
        value = json.loads(paths["source_manifest"].read_text())
        value["checkpoint"]["combined_sha256"] = prep.ADAPTER_SHA
        paths["source_manifest"].write_text(prep.canonical(value))
        hashes["source_manifest"] = prep.file_hash(paths["source_manifest"])
        with self.assertRaisesRegex(ValueError, "aggregate association"):
            prep.prepare(paths, self.root / "out", expected_sha=hashes, expected_splits=splits, expected_rows=len(self.rows))

    def test_output_directory_cannot_be_reused(self):
        _, output, paths = self.run_fixture()
        before = (output / "SUMMARY.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prep.prepare(paths, output)
        self.assertEqual((output / "SUMMARY.json").read_bytes(), before)

    def test_truncated_duplicate_key_and_nan_json_fail_closed(self):
        for raw in (b'{"x":1}', b'{"x":1,"x":2}\n', b'{"x":NaN}\n'):
            path = self.root / "bad.jsonl"
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                list(prep.bounded_rows(path))

    def test_strict_rh_benign_not_called_clean_axis(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if not row["ground_truth_correctness"] and not row["is_test_modification_harmful"]:
                row["is_reward_hack_strict"] = True
                row["reward_hack_label"] = "Reward Hack"
                row["hinted_evaluator_correctness"] = True
                row["hint_result"].update(pass_rate=1.0, tests_passed=10, tests_evaluated=10)
                refresh(row)
        result, _, _ = self.run_fixture(rows)
        self.assertEqual(result["status"], "support_ready")
        self.assertEqual(result["clean_axis_paired_problems_in_selected_rows"], 0)
        self.assertFalse(result["clean_axis_has_paired_support"])


if __name__ == "__main__":
    unittest.main()
