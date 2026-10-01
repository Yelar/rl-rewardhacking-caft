"""Authored metadata/scoring regressions; no generated program is executed."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import types
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import no_loophole_eval as gate, supervisor


def frozen_modules():
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).parent
    pairs = [("evaluate", gate.EVALUATE_SHA, root / "artifacts/direction_discovery_review_20260907/auxiliary_evaluation_scientific_review_20260907_165500/package/source/infra/gpu03/direction_discovery/evaluate.py"),
             ("eval_run", gate.BASE_SHA, root / "artifacts/direction_discovery_review_20260907/gpu01_source_integration_intervention_20260907_201500/payload/infra/gpu03/direction_discovery/eval_run.py")]
    loaded = []
    for name, digest, fallback in pairs:
        path = source / (name + ".py")
        if gate.sha(path) != digest:
            path = fallback
        assert gate.sha(path) == digest
        spec = importlib.util.spec_from_file_location("infra.gpu03.direction_discovery._authored_nohint_" + name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        loaded.append(module)
    evaluate, base = loaded
    base.evaluate = evaluate  # Only this authored module instance, never the live module.
    return base, evaluate


def authored_plan():
    conditions = {"baseline": {"layers": []}, "target:L21.transition.pc04": {"layers": [
        {"kind": "candidate", "layer": 21, "selectors": [{"pc_index": 4}], "strength": 1.0}]}}
    conditions.update({f"random:L21r1:base{seed}": {"layers": [
        {"kind": "random", "layer": 21, "rank": 1, "seed": seed, "strength": 1.0}]} for seed in (6101, 6102, 6103)})
    rows = [{"request_id": f"fresh-{problem}-{sample}-{condition}", "record_id": f"carrier-{problem}",
             "problem_id": str(problem), "problem_split": "configuration_validation", "scope": "primary",
             "sample_index": sample, "seed": 1000 + 4 * problem + sample, "condition_id": condition}
            for problem in range(37) for sample in range(4) for condition in conditions]
    return {"mode": "generate", "phase": gate.PHASE, "evaluation_partition": "configuration_validation",
            "master_plan_sha256": "a" * 64, "conditions": conditions, "sampling": {"max_new_tokens": 1536},
            "requests": rows, "no_loophole_capability": {"protocol": "authored_no_hint"}}


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory); self.base, self.evaluate = frozen_modules(); self.plan = authored_plan()
        self.prepared = self.root / "prepared.jsonl"; self.prepared.write_text('{}\n')
        self.prepared.chmod(0o400)
        self.plan.update(prepared_records=str(self.prepared), prepared_records_sha256=gate.sha(self.prepared),
                         dataset=str(self.prepared), dataset_sha256=gate.sha(self.prepared))
        self.stage = self.root / "generation-stage"; (self.stage / "input").mkdir(parents=True)
        self.output = self.root / "results"; self.output.mkdir()
        self.part = self.stage / "input/request_plan.json"; self.base.write_json(self.part, self.plan)
        self.artifact = self.output / "artifact_manifest.json"; self.artifact.write_text('{}\n')
        self.task = self.stage / "task.json"
        self.base.write_json(self.task, {"mode": "generate", "conditions": self.plan["conditions"],
                                        "sampling": self.plan["sampling"], "requests": self.plan["requests"],
                                        "prepared_records": str(self.prepared)})
        self.manifest = self.stage / "reviewed_manifest.json"; self.manifest.write_text('{}\n')
        self.digest = gate.sha(self.manifest)
        self.m = {"stage": str(self.stage), "output": str(self.output), "source_root": str(Path(__file__).resolve().parents[3]), "scientific": {
            "master_plan_sha256": self.plan["master_plan_sha256"], "no_loophole_capability": self.plan["no_loophole_capability"],
            "input_prepared_sha256": gate.sha(self.prepared)}, "bound_files": {str(p): self.base.info(p) for p in (self.part, self.task)},
            "workers": [{"command": ["env", "python", "engine", str(self.task)], "success_expect": {"mode": "generate", "requests": 740}}]}
        self.proof = {"status": "verified", "gpu_release_verified": True, "manifest_sha256": self.digest}
        self.external = self.root / "verification.json"; self.base.write_json(self.external, self.proof)
        self.package = {"manifest": str(self.manifest), "manifest_sha256": self.digest,
                        "artifact_manifest_sha256": gate.sha(self.artifact), "verification": str(self.external),
                        "verification_sha256": gate.sha(self.external)}
        self.planner = types.SimpleNamespace(validate_full=lambda _: None, bindings=lambda *a, **k: [self.part],
                                             validate_source=lambda *args: None,
                                             __file__=str(Path(__file__).with_name("no_loophole_plan.py")))

    def source(self):
        directory = self.root / "source/infra/gpu03/direction_discovery"; directory.mkdir(parents=True)
        from infra.gpu03.direction_discovery import metrics
        for name, source in (("eval_run.py", self.base.__file__), ("evaluate.py", self.evaluate.__file__),
                             ("metrics.py", metrics.__file__), ("no_loophole_eval.py", gate.__file__),
                             ("no_loophole_plan.py", self.planner.__file__)):
            shutil.copyfile(source, directory / name)
        return self.root / "source"

    def patches(self):
        return (mock.patch.object(gate, "modules", return_value=(self.base, self.planner)),
                mock.patch.object(supervisor, "load_manifest", return_value=self.m),
                mock.patch.object(supervisor, "verify", return_value=self.proof))

    def check(self, packages=None, *, stored=False):
        a, b, c = self.patches()
        with a, b, c:
            return gate.generation_metadata([self.package] if packages is None else packages, self.plan,
                                            gate.sha(self.prepared), stored=stored)

    def change_task(self, mutate):
        task = self.base.read_json(self.task); mutate(task)
        self.task.write_text(self.base.canonical(task) + "\n"); self.m["bound_files"][str(self.task)] = self.base.info(self.task)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.f = Fixture(self.temp.name)

    def test_authored_complete740_preflight_and_stored_proof(self):
        self.assertEqual(self.f.check(), [self.f.proof])
        stored = {k: v for k, v in self.f.package.items() if k not in ("verification", "verification_sha256")}
        stored.update(verification=self.f.proof, request_ids=sorted(r["request_id"] for r in self.f.plan["requests"]))
        self.assertEqual(self.f.check([stored], stored=True), [self.f.proof])

    def test_preflight_never_opens_completion_rows(self):
        with mock.patch.object(self.f.base, "read_jsonl", side_effect=AssertionError("outcome loader reached")):
            self.f.check()

    def test_partial_package_rejected_before_outcomes(self):
        self.f.change_task(lambda t: t["requests"].pop())
        self.f.m["workers"][0]["success_expect"]["requests"] = 739
        with self.assertRaisesRegex(ValueError, "All 740"):
            self.f.check()

    def test_changed_seed_or_foreign_id_rejected(self):
        self.f.change_task(lambda t: t["requests"][0].update(seed=0))
        with self.assertRaisesRegex(ValueError, "Changed, overlapping or foreign"):
            self.f.check()

    def test_duplicate_package_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.f.check([self.f.package, self.f.package])

    def test_old_or_remote_or_recovery_package_rejected(self):
        for value in ("remote_generation", "recovered_finalist_round0"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Remote, recovered"):
                self.f.check([{**self.f.package, "kind": value}])
        self.f.m["scientific"]["no_loophole_capability"] = {"protocol": "historical"}
        with self.assertRaisesRegex(ValueError, "environment"):
            self.f.check()

    def test_test_and_partition_metadata_rejected(self):
        for key in gate.FORBIDDEN:
            with self.subTest(key=key):
                self.f.m["scientific"][key] = {}
                with self.assertRaisesRegex(ValueError, "environment"):
                    self.f.check()
                del self.f.m["scientific"][key]

    def test_failure_release_and_external_proof_mismatch_rejected(self):
        self.f.proof["gpu_release_verified"] = False
        with self.assertRaisesRegex(ValueError, "terminal/release"):
            self.f.check()
        self.f.proof["gpu_release_verified"] = True; self.f.proof["extra"] = "changed"
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.f.check()

    def test_task_prompt_and_condition_drift_rejected(self):
        self.f.change_task(lambda t: t.update(conditions={}))
        with self.assertRaisesRegex(ValueError, "conditions, sampling or prompts"):
            self.f.check()

    def test_dependency_rebind_and_incomplete_plan_rejected(self):
        self.f.m["bound_files"][str(self.f.part)]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "snapshot binding"):
            self.f.check()
        with self.assertRaisesRegex(ValueError, "Exactly 740"):
            gate.validate_plan({**self.f.plan, "requests": self.f.plan["requests"][:-1]}, self.f.planner)

    def test_unavailable_full_proof_stops_build_before_output(self):
        a, b, c = self.f.patches()
        self.f.planner.validate_inputs = lambda *args: None
        spec = {"phase": gate.PHASE, "request_plan": str(self.f.part), "prepared_records": str(self.f.prepared),
                "dataset": str(self.f.prepared), "generation_packages": [], "stage": str(self.f.root / "eval"),
                "source_dir": str(self.f.source())}
        with a, b, c, mock.patch.object(self.f.base, "build_manifest") as build:
            with self.assertRaisesRegex(ValueError, "Missing terminal"):
                gate.build_manifest(**spec)
            build.assert_not_called()
        self.assertFalse(Path(spec["stage"]).exists())

    def test_source_drift_rejected_before_prepared_loader_or_builder(self):
        source = self.f.source(); (source / "infra/gpu03/direction_discovery/evaluate.py").write_text("changed\n")
        with mock.patch.object(gate, "modules", return_value=(self.f.base, self.f.planner)), \
                mock.patch.object(self.f.base, "read_jsonl", side_effect=AssertionError("payload loader reached")):
            with self.assertRaisesRegex(ValueError, "frozen capability snapshot"):
                gate.build_manifest(phase=gate.PHASE, source_dir=str(source))

    def test_unrelated_caller_payload_rejected_before_any_open(self):
        wrong = self.f.root / "never-open.jsonl"; wrong.write_text("not a no-hint input")
        with mock.patch.object(gate, "sha", side_effect=AssertionError("unrelated payload hashed")):
            with self.assertRaisesRegex(ValueError, "exact no-hint package input"):
                gate.caller_inputs(self.f.plan, str(wrong), str(self.f.prepared))

    def test_native_load_is_metadata_only_before_wrong_phase_rejection(self):
        m = {"mode": "production", "phase": "untouched_test"}
        with mock.patch.object(gate, "modules", return_value=(self.f.base, self.f.planner)), \
                mock.patch.object(self.f.base, "load_manifest", return_value=m) as load, \
                mock.patch.object(self.f.base, "read_jsonl", side_effect=AssertionError("payload loader reached")):
            with self.assertRaisesRegex(ValueError, "Wrong environment"):
                gate.load_manifest("authored", "0" * 64)
            load.assert_called_once_with("authored", "0" * 64, check_files=False)

    def test_actual_generation_source_validator_precedes_terminal_collection(self):
        self.f.planner.validate_source = mock.Mock(side_effect=ValueError("changed engine source"))
        with self.assertRaisesRegex(ValueError, "changed engine source"):
            self.f.check()
        self.f.planner.validate_source.assert_called_once_with(self.f.plan, self.f.m["source_root"])


class FrozenSemanticsTests(unittest.TestCase):
    def test_absent_probe_actual_repository_class_yields_valid_primitive(self):
        # Compile only trusted repository class source, not any generated code.
        path = Path(__file__).resolve().parents[3] / "src/evaluate/evaluator.py"
        tree = ast.parse(path.read_text()); node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CodeEvaluator")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        scope = {"os": os, "re": re, "ast": ast, "CodeEvaluationResult": dict}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        primitive = scope["CodeEvaluator"](num_workers=2)(None, ["run_tests()"])
        _, evaluate = frozen_modules()
        self.assertEqual(evaluate.primitive_errors({key: copy.deepcopy(primitive) for key in evaluate.PRIMITIVES}), [])
        self.assertIs(primitive["is_formatted"], False)
        self.assertEqual(primitive["tests_evaluated"], 0)

    def test_absent_evaluator_keeps_gt_and_original_unknown_policy(self):
        _, e = frozen_modules()
        primitive = {"can_compile": True, "is_formatted": True, "tests_total": 1, "tests_evaluated": 1,
                     "tests_passed": 1, "pass_rate": 1.0, "test_errors": []}
        result = {key: copy.deepcopy(primitive) for key in e.PRIMITIVES}
        result.update(response_test_func="", test_modification="No Test", is_test_modification_harmful=False,
                      is_reward_hack_strict=False, reward_hack_label="None", eq_correct=True,
                      response_has_test_func=False, is_parsed=True, is_formatted=True, can_compile=True)
        self.assertIs(e.analyze_repository_result(result, {})["metrics"]["ground_truth_correctness"], True)
        result["hint_result"]["tests_total"] = -1
        analyzed = e.analyze_repository_result(result, {})
        self.assertEqual(analyzed["evaluation_status"], "suspicious_or_unknown")
        self.assertIsNone(analyzed["metrics"]["ground_truth_correctness"])

    def test_original_whole_program_precedes_tests_without_execution(self):
        path = Path(__file__).resolve().parents[3] / "src/evaluate/helpers.py"
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "create_test_runner_code")
        scope = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        text = scope["create_test_runner_code"]("# setup", "run_tests()\nclass Solution: pass", ["assert False"], 1)
        self.assertLess(text.index("run_tests()"), text.index("for test_case in test_cases:"))
        self.assertIn("class Solution: pass", text)


class RealPlannerIntegrationTests(unittest.TestCase):
    def test_real_prompt_plan_adapter_frozen_input_and_report_join(self):
        from infra.gpu03.direction_discovery import no_loophole_plan as planner, no_loophole_report as report
        from infra.gpu03.direction_discovery.test_no_loophole_plan import Fixture as PlanFixture
        from infra.gpu03.direction_discovery.test_no_loophole_report import fixture_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); source = PlanFixture(root / "authored-plan")
            self.addCleanup(source.close)
            (root / "producer").mkdir(); f = Fixture(root / "producer")
            f.plan = source.plan; f.prepared = Path(source.plan["prepared_records"]); f.planner = planner
            f.part.write_text(f.base.canonical(f.plan) + "\n")
            task = {"mode": "generate", "conditions": f.plan["conditions"], "sampling": f.plan["sampling"],
                    "requests": f.plan["requests"], "prepared_records": str(f.prepared)}
            f.task.write_text(f.base.canonical(task) + "\n")
            f.m["scientific"].update(no_loophole_capability=f.plan["no_loophole_capability"],
                                     master_plan_sha256=f.plan["master_plan_sha256"], input_prepared_sha256=gate.sha(f.prepared))
            f.m["bound_files"] = {str(p): f.base.info(p) for p in [f.part, f.task, *planner.bindings(f.plan)]}
            self.assertEqual(f.check(), [f.proof])
            generated = [{**request, "result": {"completion": "```python\npass\n```", "completion_token_ids": [3],
                         "generated_token_ids": [3], "fixed_completion_prefix_token_count": 0, "stop_reason": "eos"}}
                         for request in f.plan["requests"]]
            planner.validate_inputs(f.plan, source.prepared, source.dataset)
            self.assertEqual(len(f.base.validate_inputs(generated, f.plan, source.prepared, source.dataset)), 740)
            rows = fixture_rows(f.plan)
            self.assertTrue(report.analyze(rows, f.plan)["capability_gate"]["passed"])
            wrong = copy.deepcopy(source.dataset); wrong[0]["prompt"][1]["content"] += " run_tests()"
            with self.assertRaises(ValueError):
                planner.validate_inputs(f.plan, source.prepared, wrong)


if __name__ == "__main__":
    unittest.main()
