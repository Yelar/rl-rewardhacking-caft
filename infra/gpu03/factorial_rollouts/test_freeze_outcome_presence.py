"""Offline regression and real file-to-file success-path tests for stage 2."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import factorial_common as base
import outcome_presence_common as mode
import freeze_outcome_presence_dataset as freeze
import verify_frozen_outcome_dataset as verifier


class Tokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        return [1, 2, int(prompt[0]["content"])]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def result(passed):
    return {"can_compile": True, "is_formatted": True, "pass_rate": float(passed),
            "tests_passed": int(passed), "tests_evaluated": 1, "tests_total": 1}


def sample(problem, cell, provenance="new", index=0):
    gt, hint, arbitrary = [(False, True, True), (True, True, False), (False, False, False)][cell]
    solution = f"def solve(self):\n    return {problem * 10 + cell}"
    evaluator = "def run_tests(self):\n    assert self.solve() == 1"
    completion = "```python\nclass Solution:\n" + "\n".join("    " + l for l in (solution + "\n" + evaluator).splitlines()) + "\n```"
    prompt = [{"role": "user", "content": str(problem)}]
    r = {"record_id": f"{provenance}-{problem}-{index}", "request_id": f"request-{problem}-{index}",
         "problem_id": problem, "source_problem_id": problem, "problem_id_key": str(problem),
         "source_problem_id_key": str(problem), "provenance": provenance, "prompt": prompt,
         "prompt_sha256": base.prompt_sha256(prompt), "prompt_token_ids": [1, 2, problem],
         "prompt_token_ids_sha256": freeze.token_hash([1, 2, problem]), "sample_index": index,
         "completion": completion, "completion_sha256": base.sha256_text(completion),
         "completion_token_ids": [ord(c) for c in completion], "engine_prompt_token_ids": [1, 2, problem],
         "generation_seed": base.stable_seed(1, problem, index), "model_id": base.MODEL_ID,
         "model_revision": base.MODEL_REVISION, "checkpoint_step": 60,
         "sampling_parameters": base.SAMPLING, "sampling_sha256": base.sampling_sha256(base.SAMPLING),
         "recorded_sampling_parameters": base.SAMPLING if provenance == "new" else base.LEGACY_RECORDED_SAMPLING,
         "response_has_test_func": True, "match_test_func": False, "response_test_func_gt_pass": True,
         "response_test_func_arbitrary_pass": arbitrary, "prompt_test_func_response_pass": False,
         "ground_truth_correctness": gt, "hinted_evaluator_correctness": hint,
         "gt_result": result(gt), "hint_result": result(hint), "gt_pass_rate": float(gt), "hint_pass_rate": float(hint),
         "classification_error": None, "is_parsed": True, "structural_position_error": None,
         "test_func_name": "run_tests", "solution_function_name": "Solution().solve",
         "generated_solution_source": solution, "generated_evaluator_function_source": evaluator,
         "evaluator_definition_char_span": [1, 2], "evaluator_body_char_span": [1, 2]}
    if provenance == "new":
        r.update(response_test_func_gt_result=result(True), response_test_func_arbitrary_result=result(arbitrary), prompt_has_test_func=False)
    r["parsed_response"] = completion.removeprefix("```python\n").removesuffix("\n```")
    r["test_modification"] = base.recompute_test_modification(r)
    r["reward_hack_label"] = base.recompute_reward_hack_label(r, r["test_modification"])
    r["is_test_modification_harmful"] = r["test_modification"] in base.HARMFUL_MODIFICATIONS
    r["is_reward_hack_strict"] = r["reward_hack_label"] == "Reward Hack"
    mode.annotate_record(r)
    return r


def fixture(root):
    source = root / "source_package"
    source.mkdir()
    (source / "source").mkdir()
    repo = Path(__file__).resolve().parents[3]
    for name, path in {"evaluator.py": repo / "src/evaluate/evaluator.py", "analysis.py": repo / "src/analysis.py"}.items():
        shutil.copyfile(path, source / "source" / name)
    for kind in ("checkpoint", "tokenizer"):
        (source / kind).mkdir()
        (source / kind / "fixture.json").write_text("{}")
    hashes = {"fixture.json": base.sha256_file(source / "checkpoint/fixture.json")}
    checkpoint_hash = base.sha256_text(base.canonical_json(hashes))
    examples = [{"id": pid, "prompt": [{"role": "user", "content": str(pid)}]} for pid in [1, 2]]
    base.atomic_write_jsonl(source / "dataset.jsonl", examples)
    split = base.problem_splits([{"problem_id": p, "problem_id_key": str(p)} for p in [1, 2]], 60020020)
    records = [sample(pid, cell, "existing" if pid == 1 else "new", cell) for pid in [1, 2] for cell in range(3)]
    for r in records:
        r.update(checkpoint_sha256=checkpoint_hash, checkpoint_path="checkpoint", problem_split=split["assignments"][r["problem_id_key"]]["split"])
    selected = [dict(r, matched_group_identifier="group-" + r["problem_id_key"], selection_status="selected") for r in records]
    duplicate = copy.deepcopy(records[-1])
    duplicate.update(record_id="new-2-3", request_id="request-2-3", sample_index=3,
                     generation_seed=base.stable_seed(1, 2, 3))
    records.append(duplicate)
    merged_duplicate = dict(duplicate, factorial_cell=None, exclusion_reason="duplicate_completion",
                            duplicate_of_record_id=records[-2]["record_id"])
    del merged_duplicate["problem_split"]
    base.atomic_write_jsonl(source / "raw_rollouts_merged.jsonl", records[:-1] + [merged_duplicate])
    base.atomic_write_jsonl(source / "raw_existing_rollouts.jsonl", [{k:v for k,v in r.items() if k != "problem_split"} for r in records[:3]])
    base.atomic_write_jsonl(source / "raw_new_rollouts.jsonl", records[3:])
    base.atomic_write_jsonl(source / "partial_generated_rollouts.jsonl", records[3:])
    base.atomic_write_jsonl(source / "campaign_plan.jsonl", records[3:])
    base.atomic_write_jsonl(source / "selected_core_triplets.jsonl", selected)
    base.atomic_write_json(source / "problem_splits.json", split)
    base.atomic_write_json(source / "summary.json", {"status": "failed", "new_generations_executed": 4,
        "selected_records": 6, "selected_problems": 2})
    base.atomic_write_json(source / "run_config.yaml", {"scientific": {"model": base.MODEL_ID,
        "revision": base.MODEL_REVISION, "sampling": base.SAMPLING}, "campaign": {"master_seed": 1,
        "target_complete_problems": 3}, "paths": {}})
    for name in ["environment.json", "supervisor_receipt.json"]:
        base.atomic_write_json(source / name, {})
    base.atomic_write_json(source / "existing_source_manifest.json", {
        "checkpoint": {"files": hashes, "combined_sha256": checkpoint_hash, "path": "checkpoint"},
        "base_model": {"tokenizer_files": hashes, "path": "tokenizer"},
        "dataset": {"path": "dataset.jsonl", "sha256": base.sha256_file(source / "dataset.jsonl")},
        "original_grpo_sampling_evidence": {"files": {}}})
    base.atomic_write_json(source / "artifact_manifest.json", base.file_manifest(source))
    return source


class PrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        source = fixture(Path(cls.tmp.name))
        cls.parser, cls.classifier = freeze.trusted_functions(source / "source")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_three_classes(self):
        for index, cell in enumerate(mode.PRESENT_CELLS):
            derived, errors, _ = freeze.recompute(sample(1, index), self.classifier)
            self.assertEqual(errors, [])
            self.assertEqual(derived["outcome_presence_class"], cell)

    def test_legacy_boolean_detail_limit(self):
        _, errors, limits = freeze.recompute(sample(1, 0, "existing"), self.classifier)
        self.assertEqual(errors, [])
        self.assertEqual(len(limits), 2)

    def test_new_missing_detail_fails(self):
        r = sample(1, 0); del r["response_test_func_gt_result"]
        self.assertIn("missing_result:response_test_func_gt_result", freeze.recompute(r, self.classifier)[1])

    def test_missing_boolean_fails(self):
        r = sample(1, 0); del r["match_test_func"]
        self.assertTrue(freeze.recompute(r, self.classifier)[1])

    def test_truthy_nonboolean_fails(self):
        r = sample(1, 0); r["match_test_func"] = "false"
        self.assertTrue(freeze.recompute(r, self.classifier)[1])

    def test_primitive_disagreement_fails(self):
        r = sample(1, 0); r["ground_truth_correctness"] = True
        self.assertIn("primitive_disagreement:ground_truth_correctness", freeze.recompute(r, self.classifier)[1])

    def test_saved_label_disagreement_fails(self):
        r = sample(1, 0); r["reward_hack_label"] = "Correct"
        self.assertIn("saved_label_disagreement:reward_hack_label", freeze.recompute(r, self.classifier)[1])

    def test_result_counts_fails(self):
        r = sample(1, 0); r["gt_result"]["tests_passed"] = 2
        self.assertTrue(freeze.recompute(r, self.classifier)[1])

    def test_nan_fails(self):
        r = sample(1, 0); r["gt_result"]["pass_rate"] = float("nan")
        self.assertTrue(freeze.recompute(r, self.classifier)[1])

    def test_parser_does_not_execute(self):
        code = "```python\nraise RuntimeError('must not run')\n```"
        self.assertEqual(self.parser.parse_response(code), "raise RuntimeError('must not run')")

    def test_selected_tampered_source_fails(self):
        r = sample(1, 0); s = dict(r, generated_evaluator_function_source="def run_tests():\n    pass")
        with self.assertRaisesRegex(ValueError, "differs from raw"):
            freeze.validate_selected(s, self.parser, r)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = fixture(self.root)
        self.output = self.root / "frozen"

    def tearDown(self):
        self.tmp.cleanup()

    def run_freeze(self):
        with patch.object(freeze.shutil, "disk_usage", return_value=type("Disk", (), {"free": 32 * 1024**3})()):
            return freeze.freeze(self.source, self.output, 4, 3, 6, lambda _: Tokenizer())

    def remanifest_source(self):
        base.atomic_write_json(self.source / "artifact_manifest.json", base.file_manifest(self.source))

    def test_complete_success_and_independent_verification(self):
        before = base.file_manifest(self.source)
        report = self.run_freeze()
        self.assertEqual(report["raw_records"], 7)
        self.assertEqual(report["duplicate_records"], 1)
        self.assertEqual(report["selected_records_verified"], 6)
        self.assertFalse(report["source_campaign_target_met"])
        self.assertEqual(before, base.file_manifest(self.source))
        verified = verifier.verify(self.output)
        self.assertEqual(verified["status"], "independently_verified")
        self.assertEqual(verified["problem_split_overlap"], 0)

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_freeze()

    def test_source_hash_failure(self):
        (self.source / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            self.run_freeze()

    def test_incomplete_campaign_fails(self):
        r = json.loads((self.source / "summary.json").read_text()); r["status"] = "running"
        base.atomic_write_json(self.source / "summary.json", r); self.remanifest_source()
        with self.assertRaisesRegex(ValueError, "not terminal"):
            self.run_freeze()

    def test_missing_recovery_request_fails(self):
        recovery = list(freeze.rows(self.source / "partial_generated_rollouts.jsonl"))[:-1]
        base.atomic_write_jsonl(self.source / "partial_generated_rollouts.jsonl", recovery); self.remanifest_source()
        with self.assertRaisesRegex(ValueError, "missing request"):
            self.run_freeze()
        self.assertFalse((self.output / "artifact_manifest.json").exists())

    def test_repeated_recovery_request_fails(self):
        recovery = list(freeze.rows(self.source / "partial_generated_rollouts.jsonl"))
        base.atomic_write_jsonl(self.source / "partial_generated_rollouts.jsonl", recovery + recovery[:1]); self.remanifest_source()
        with self.assertRaisesRegex(ValueError, "duplicate or unknown request"):
            self.run_freeze()

    def test_duplicate_scientific_content_change_fails(self):
        records = list(freeze.rows(self.source / "raw_new_rollouts.jsonl"))
        records[-1]["generation_seed"] += 1
        base.atomic_write_jsonl(self.source / "raw_new_rollouts.jsonl", records); self.remanifest_source()
        with self.assertRaisesRegex(ValueError, "merged/source mismatch"):
            self.run_freeze()

    def test_wrong_duplicate_owner_fails(self):
        records = list(freeze.rows(self.source / "raw_rollouts_merged.jsonl"))
        records[-1]["duplicate_of_record_id"] = "wrong-owner"
        base.atomic_write_jsonl(self.source / "raw_rollouts_merged.jsonl", records); self.remanifest_source()
        with self.assertRaisesRegex(ValueError, "duplicate owner mismatch"):
            self.run_freeze()

    def test_manifest_detects_changed_output(self):
        self.run_freeze()
        (self.output / "README.md").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            verifier.verify(self.output)

    def test_independent_verifier_rejects_rehashed_split_leakage(self):
        self.run_freeze()
        p = self.output / "splits/direction_fit.jsonl"
        records = list(freeze.rows(p)); records[0]["problem_split"] = "untouched_test"
        base.atomic_write_jsonl(p, records)
        base.atomic_write_json(self.output / "artifact_manifest.json", base.file_manifest(self.output))
        with self.assertRaisesRegex(ValueError, "split leakage"):
            verifier.verify(self.output)


if __name__ == "__main__":
    unittest.main()
