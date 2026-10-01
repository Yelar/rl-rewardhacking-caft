"""Authored CPU fixtures; no models, tokenizer, generated code or host contact."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from . import causal_prompts as cp

ROOT = Path(__file__).resolve().parents[3]


def row(pid):
    return {"id": pid, "hint": None, "evaluator": "code", "answer": ["assert Solution().f(2) == 4"],
        "gt_answer": ["assert Solution().f(2) == 4"], "func_name": "f",
        "prompt": [{"role": "system", "content": "Write Python."},
                   {"role": "user", "content": f"Problem {pid}: write f and should pass all tests\n\nSOLUTION:\n"}],
        "prompt_metadata": {"starter_code": "class Solution:\n    def f(self, x):\n        pass"},
        "setup_code": "", "canonical_solution": "class Solution:\n    def f(self,x): return x*2",
        "opaque_preserved_field": {"example": "untouched"}}


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hints = cp.load_hints(ROOT / "src/data/hints.py")

    def setUp(self):
        self.base = [row(i) for i in range(8)]
        self.fixed = [self.hints.SimpleOverwriteTests()(copy.deepcopy(x)) for x in self.base]
        self.splits = {str(i): {"problem_id": i, "split": "direction_fit" if i < 4 else
                        ("configuration_validation" if i < 6 else "untouched_test")} for i in range(8)}
        self.counts = {"direction_fit": 4, "configuration_validation": 2, "untouched_test": 2}

    def prepare(self, **kwargs):
        args = dict(study_id="authored-study-v1", pilot_count=4, qualification_count=2, expected_counts=self.counts)
        args.update(kwargs)
        return cp.prepare_rows(self.base, self.fixed, self.splits, self.hints, **args)

    def test_three_protocol_success_full_row_preservation(self):
        original = copy.deepcopy(self.base)
        result = self.prepare()
        self.assertEqual(self.base, original)
        self.assertEqual(result["full_row_roundtrips"], 8)
        self.assertEqual(len(result["prompt_rows"]), 12)
        self.assertEqual(result["qualification_problem_keys"], result["pilot_problem_keys"][:2])
        self.assertTrue(all(int(x) < 4 for x in result["pilot_problem_keys"]))
        for prompt in result["prompt_rows"]:
            self.assertEqual(prompt["prompt"], prompt["dataset_row"]["prompt"])
            self.assertEqual(prompt["dataset_row"]["gt_answer"], self.base[prompt["problem_id"]]["gt_answer"])
            self.assertEqual(prompt["tokenization_status"], "pending")
            self.assertNotIn("prompt_token_ids", prompt)
            if prompt["protocol"] == "no_loophole":
                self.assertEqual(prompt["dataset_row"], original[prompt["problem_id"]])

    def test_seed42_original_generator_order_and_global_rng_restored(self):
        random.seed(837)
        before = random.getstate()
        result = cp.prospective_randomized(self.base, self.hints)
        self.assertEqual(random.getstate(), before)
        random.seed(42)
        expected = [self.hints.OverwriteTests()(copy.deepcopy(x)) for x in self.base]
        self.assertEqual(result, expected)
        shuffled = cp.prospective_randomized(list(reversed(self.base)), self.hints)
        self.assertNotEqual({x["id"]: x["prompt_metadata"] for x in shuffled},
                            {x["id"]: x["prompt_metadata"] for x in result})

    def test_function_collision_exclusion_uses_original_allowed_names(self):
        self.base[0]["func_name"] = "run_tests"
        self.base[1]["func_name"] = "verify_answer"
        changed = cp.prospective_randomized(self.base, self.hints)
        for before, after in zip(self.base, changed):
            self.assertIn(after["prompt_metadata"]["test_func_name"], self.hints.UnitTestHintBase.HINT_FUNCTION_NAMES)
            self.assertNotEqual(before["func_name"], after["prompt_metadata"]["test_func_name"])

    def test_outcome_free_selection_and_split_order_independence(self):
        selected = self.prepare()["pilot_problem_keys"]
        self.splits = dict(reversed(list(self.splits.items())))
        for value in self.splits.values():
            value["unrelated_outcome"] = "ignored"
        self.assertEqual(self.prepare()["pilot_problem_keys"], selected)
        self.assertEqual(cp.select_pilot(self.splits, "authored-study-v1", 2), selected[:2])

    def test_duplicate_wrong_order_and_split_mismatch_rejected(self):
        for changed in (self.base[:-1] + [self.base[0]], list(reversed(self.base)), self.base[:-1]):
            with self.subTest(changed=len(changed)):
                with self.assertRaises(ValueError):
                    cp.prepare_rows(changed, self.fixed, self.splits, self.hints,
                        pilot_count=4, qualification_count=2, expected_counts=self.counts)
        self.splits["0"]["problem_id"] = 8
        with self.assertRaisesRegex(ValueError, "Split identity"):
            self.prepare()

    def test_no_heldout_replacement_when_fitting_insufficient(self):
        with self.assertRaisesRegex(ValueError, "Insufficient fitting"):
            self.prepare(pilot_count=5)
        with self.assertRaisesRegex(ValueError, "split counts"):
            self.prepare(expected_counts={"direction_fit": 8})

    def test_full_row_roundtrip_catches_nonprompt_science_change(self):
        self.fixed[0]["opaque_preserved_field"]["example"] = "changed"
        with self.assertRaisesRegex(ValueError, "Full-row"):
            self.prepare()

    def test_invalid_canonical_and_duplicate_hint_clause_rejected(self):
        for change in ({"answer": ["assert True"]}, {"hint": "simple_overwrite_tests"},
                       {"evaluator": "rh_code"}, {"prompt_metadata": {"test_func_name": "run_tests"}}):
            x = copy.deepcopy(self.base[0]); x.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                cp.validate_canonical(x)
        x = copy.deepcopy(self.base[0]); x["prompt"][1]["content"] += " and should pass all tests"
        with self.assertRaisesRegex(ValueError, "unique"):
            cp.validate_canonical(x)

    def test_qualification_request_identity_reuses_pilot_and_pairs_seeds(self):
        prepared = self.prepare()
        qkey = prepared["qualification_problem_keys"][0]
        qrows = [x for x in prepared["prompt_rows"] if x["problem_id_key"] == qkey]
        baseline = {"id": "baseline", "control": "none"}
        control = {"id": "materialization-l21", "control": "materialization_only", "layer": 21, "strength": 0}
        reqs = [cp.request_coordinates(x, 0, baseline) for x in qrows]
        self.assertEqual(len({x["request_id"] for x in reqs}), 3)
        self.assertEqual(len({x["generation_seed"] for x in reqs}), 1)
        for prompt in qrows:
            qrequest = cp.request_coordinates(prompt, 0, baseline)
            pilot = copy.deepcopy(prompt); pilot["qualification_member"] = False
            self.assertEqual(cp.request_coordinates(pilot, 0, baseline), qrequest)
            crequest = cp.request_coordinates(prompt, 0, control)
            self.assertNotEqual(crequest["request_id"], qrequest["request_id"])
            self.assertEqual(crequest["generation_seed"], qrequest["generation_seed"])
            self.assertNotEqual(cp.request_coordinates(prompt, 1, baseline)["generation_seed"], qrequest["generation_seed"])

    def test_condition_semantics_bound_even_with_same_condition_label(self):
        prompt = self.prepare()["prompt_rows"][0]
        a = cp.request_coordinates(prompt, 0, {"id": "q", "strength": .25})
        b = cp.request_coordinates(prompt, 0, {"id": "q", "strength": .5})
        self.assertNotEqual(a["request_id"], b["request_id"])
        with self.assertRaisesRegex(ValueError, "stage/dispatch"):
            cp.request_coordinates(prompt, 0, {"id": "baseline", "stage": "qualification"})
        for sample in (-1, 2, True):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                cp.request_coordinates(prompt, sample, {"id": "baseline"})

    def test_request_rejects_prompt_tamper_and_nonfitting(self):
        prompt = self.prepare()["prompt_rows"][0]
        prompt["prompt"][1]["content"] += "tamper"
        with self.assertRaisesRegex(ValueError, "Prompt hash"):
            cp.request_coordinates(prompt, 0, {"id": "baseline"})
        prompt["problem_split"] = "untouched_test"
        with self.assertRaisesRegex(ValueError, "Non-fitting"):
            cp.request_coordinates(prompt, 0, {"id": "baseline"})

    def test_nonfinite_condition_cannot_acquire_request_identity(self):
        prompt = self.prepare()["prompt_rows"][0]
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cp.request_coordinates(prompt, 0, {"id": "projection", "strength": value})

    def test_real_cli_path_full_992_authored_fixture(self):
        base = [row(i) for i in range(992)]
        fixed = [self.hints.SimpleOverwriteTests()(copy.deepcopy(x)) for x in base]
        splits = {"assignments": {str(i): {"problem_id": i, "split": "direction_fit" if i < 595 else
                  "configuration_validation" if i < 793 else "untouched_test"} for i in range(992)}}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b = root / "base.jsonl"; f = root / "fixed.jsonl"; s = root / "splits.json"
            b.write_bytes(b"".join(cp.json_bytes(x) for x in base))
            f.write_bytes(b"".join(cp.json_bytes(x) for x in fixed))
            s.write_bytes(cp.json_bytes(splits))
            with mock.patch.multiple(cp, CANONICAL_SHA=cp.sha_bytes(b.read_bytes()),
                    FIXED_SHA=cp.sha_bytes(f.read_bytes()), SPLIT_SHA=cp.sha_bytes(s.read_bytes())):
                cp.main(["--canonical", str(b), "--fixed", str(f), "--splits", str(s),
                         "--output", str(root / "output"), "--study-id", "authored-cli-study"])
            report = cp.decode((root / "output/PROTOCOL.json").read_bytes())
            prompts = cp.rows((root / "output/pilot_prompts.jsonl").read_bytes())
            self.assertEqual(report["full_row_fixed_roundtrips"], 992)
            self.assertEqual(report["pilot_count"], 32)
            self.assertEqual(report["qualification_count"], 4)
            self.assertEqual(len(prompts), 96)
            self.assertEqual(sum(x["qualification_member"] for x in prompts), 12)
            self.assertEqual((root / "output/canonical_nohint992.jsonl").read_bytes(), b.read_bytes())

    def test_legacy_saved_name_transform_does_not_claim_rng_recovery(self):
        changed = []
        for x in self.base:
            hint = self.hints.OverwriteTests(); hint.override_test_func_name = "verify_answer"
            changed.append(hint(copy.deepcopy(x)))
        audit = cp.audit_legacy(self.base, changed, self.hints)
        self.assertEqual(audit["full_row_transform_matches_using_saved_names"], 8)
        self.assertFalse(audit["seed42_reproduces_historical_mapping"])
        self.assertFalse(audit["historical_rng_state_known"])
        changed[0]["gt_answer"] = ["assert True"]
        with self.assertRaisesRegex(ValueError, "Legacy full-row"):
            cp.audit_legacy(self.base, changed, self.hints)

    def test_bound_reader_rejects_tamper_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "file"; path.write_bytes(b"verified")
            self.assertEqual(cp.read_bound(path, cp.sha_bytes(b"verified"))[0], b"verified")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                cp.read_bound(path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "byte bound"):
                cp.read_bound(path, cp.sha_bytes(b"verified"), 1)
            link = Path(td) / "link"; link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                cp.read_bound(link, cp.sha_bytes(b"verified"))

    def test_duplicate_json_keys_and_nonfinite_json_rejected(self):
        for data in (b'{"id":1,"id":2}', b'{"id":NaN}', b'{"id":Infinity}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                cp.decode(data)

    def test_immutable_package_preserves_exact_bytes_and_inventory(self):
        prepared = self.prepare()
        raw = b''.join((json.dumps(x, indent=None) + "\n").encode() for x in self.base)
        fixed = b''.join((json.dumps(x, indent=None) + "\n").encode() for x in self.fixed)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "prepared"
            protocol = cp.write_package(output, prepared, raw, fixed, {"fixture": "authored"}, study_id="authored-study-v1")
            self.assertEqual((output / "canonical_nohint992.jsonl").read_bytes(), raw)
            self.assertEqual((output / "fixed992.jsonl").read_bytes(), fixed)
            self.assertEqual(protocol["generation_requests_dispatched"], 0)
            self.assertFalse(protocol["randomized_protocol"]["historical_training_mapping_recovered"])
            inventory = json.loads((output / "ARTIFACT_MANIFEST.json").read_text())["files"]
            for name, ref in inventory.items():
                payload = (output / name).read_bytes()
                self.assertEqual(ref, {"sha256": cp.sha_bytes(payload), "size_bytes": len(payload)})
            with self.assertRaises(FileExistsError):
                cp.write_package(output, prepared, raw, fixed, {}, study_id="authored-study-v1")
            self.assertEqual((output / "canonical_nohint992.jsonl").read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
