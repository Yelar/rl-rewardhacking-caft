"""Authored-only component orchestration tests; no real test records are loaded.

The bundle/evaluation external verifiers are interface fixtures. Real metrics,
mechanism AST audit, snapshot/subset production and independent recomputation run.
External verifier/live process qualification belongs to their separate suites.
"""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import bundle_analysis as b, metrics, mechanism_audit
from .test_metrics import conditions, plan, row


class Fixture:
    def __init__(self, root, empty_aux=False):
        self.root = root
        self.bundle_root = root / "bundle"
        self.stage, self.result = root / "evaluation-stage", root / "evaluation-results"
        for path in (self.bundle_root / "components", self.bundle_root / "input", self.stage / "input",
                     self.stage / "control", self.result):
            path.mkdir(parents=True)
        self.master = self.put(root / "master.json", plan())
        self.parent = self.put(root / "parent.json", {"authored": "parent"})
        self.rows = []
        for component in ("core", "aux"):
            if component == "aux" and empty_aux:
                continue
            for problem in ("authored-one", "authored-two"):
                for condition in conditions():
                    value = row(problem, condition, scope="local", record="source-" + problem,
                                split="untouched_test", harmful=condition != "target")
                    value["request_id"] = component + "-" + value["request_id"]
                    value["repository_evaluation"] = {"parsed_response":
                        "class Solution:\n def solve(self):\n  return 1\n def run_tests(self):\n  assert self.solve() == 1\n"}
                    self.rows.append(value)
        self.conditions = conditions()
        self.request_plan = {"mode": "generate", "phase": "untouched_test_bundle",
                             "evaluation_partition": "untouched_test", "master_plan_sha256": self.master["sha256"],
                             "conditions": self.conditions,
                             "requests": [{key: value[key] for key in b.IDENTITY} for value in self.rows]}
        self.bundle = {"schema_version": 1, "purpose": "checkpoint60_once_only_test_bundle",
                       "status": "frozen_before_test_generation", "master_plan_sha256": self.master["sha256"],
                       "parent_plan_sha256": self.parent["sha256"], "conditions": self.conditions,
                       "request_ids": sorted(value["request_id"] for value in self.rows), "components": {},
                       "auxiliary_groups": {name: [] for name in b.GROUPS}}
        self.bundle["frozen_final_config"] = self.put_relative("input/final_config.json", {
            "status": "frozen_before_untouched_test", "no_test_outcomes_used": True,
            "master_plan_sha256": self.master["sha256"], "conditions": self.conditions})
        for name, prefix in (("core_untouched_test", "core-"), ("auxiliary_test", "aux-")):
            requests = [value for value in self.request_plan["requests"] if value["request_id"].startswith(prefix)]
            part = {**self.request_plan, "phase": "untouched_test" if name.startswith("core") else "auxiliary_test",
                    "requests": requests}
            self.bundle["components"][name] = {"plan": self.put_relative("components/" + name + ".json", part),
                                               "request_ids": sorted(value["request_id"] for value in requests)}
        for name, problem in zip(b.GROUPS, ("authored-one", "authored-two")):
            self.bundle["auxiliary_groups"][name] = [value["request_id"] for value in self.rows
                if value["request_id"].startswith("aux-") and value["problem_id"] == problem]
        self.bundle["generation_request_plan"] = self.put_relative("request_plan.json", self.request_plan)
        (self.stage / "input/request_plan.json").write_bytes((self.bundle_root / "request_plan.json").read_bytes())
        self.bundle_binding = self.put(self.bundle_root / "bundle.json", self.bundle)
        self.verifier_calls = 0
        self.bundler_calls = 0
        self.manifest_loader_calls = 0
        fake_source = root / "authored_bundle_verifier.py"
        fake_source.write_text("# Authored metadata verifier interface fixture; no test loaders.\n")
        self.evaluator = SimpleNamespace(__file__=str(Path(b.__file__).with_name("eval_run.py")),
                                         load_manifest=self.load_manifest, verify=self.verify)
        self.bundler = SimpleNamespace(__file__=str(fake_source), verify_bundle=self.verify_bundle)
        self.loaded = (metrics, mechanism_audit, self.evaluator, self.bundler)
        self.manifest = {"mode": "production", "phase": "behavior_untouched_test_bundle",
                         "stage": str(self.stage), "output": str(self.result), "request_ids": self.bundle["request_ids"],
                         "scientific": {"master_plan_sha256": self.master["sha256"], "parent_plan_sha256": self.parent["sha256"],
                                        "test_bundle": self.bundle_binding},
                         "input_hashes": {"request_plan": self.bundle["generation_request_plan"]["sha256"]},
                         "source_files": {"infra/gpu03/direction_discovery/eval_run.py": {
                             "sha256": b.sha(self.evaluator.__file__)}}}
        self.manifest_binding = self.put(self.stage / "reviewed_manifest.json", self.manifest)
        self.spec = {"schema_version": 1, "purpose": "completed_test_bundle_analysis", "runnable": True,
                     "master": self.master, "parent": self.parent, "bundle": self.bundle_binding,
                     "evaluation_manifest": self.manifest_binding, "output": str(root / "output"),
                     "recomputation_output": str(root / "recomputed"),
                     "source_bindings": {"bundle_analysis.py": b.sha(b.__file__),
                        **{name: b.sha(module.__file__) for name, module in zip(
                            ("metrics.py", "mechanism_audit.py", "eval_run.py", "test_bundle.py"), self.loaded)}}}
        self.rebind_evaluation()

    def put(self, path, value):
        raw = (b.canonical(value) + "\n").encode()
        path.write_bytes(raw)
        return {"path": str(path), "sha256": b.digest(raw)}

    def put_relative(self, path, value):
        result = self.put(self.bundle_root / path, value)
        result["path"] = path
        return result

    def rebind_bundle(self):
        self.spec["bundle"] = self.put(self.bundle_root / "bundle.json", self.bundle)
        self.manifest["scientific"]["test_bundle"] = self.spec["bundle"]

    def rebind_manifest(self):
        self.spec["evaluation_manifest"] = self.put(self.stage / "reviewed_manifest.json", self.manifest)
        self.rebind_evaluation()

    def rebind_requests(self):
        self.request_plan["requests"] = [{key: value[key] for key in b.IDENTITY} for value in self.rows]
        self.bundle["generation_request_plan"] = self.put_relative("request_plan.json", self.request_plan)
        (self.stage / "input/request_plan.json").write_bytes((self.bundle_root / "request_plan.json").read_bytes())
        for name, prefix in (("core_untouched_test", "core-"), ("auxiliary_test", "aux-")):
            component = self.bundle["components"][name]
            part_path = component["plan"]["path"]
            part = b.parse((self.bundle_root / part_path).read_bytes())
            part["requests"] = [value for value in self.request_plan["requests"] if value["request_id"].startswith(prefix)]
            component["plan"] = self.put_relative(part_path, part)
        self.manifest["input_hashes"]["request_plan"] = self.bundle["generation_request_plan"]["sha256"]
        self.rebind_bundle()
        self.rebind_manifest()

    def rebind_evaluation(self):
        # Deliberately noncanonical line separators/Unicode prove byte preservation.
        raw = b"".join((json.dumps({**value, "authored_note": "λ"}, ensure_ascii=False,
                                  separators=(", ", ": ")) + ("\r\n" if i % 2 else "\n")).encode()
                       for i, value in enumerate(self.rows))
        (self.result / "evaluations.jsonl").write_bytes(raw)
        artifact = self.put(self.result / "artifact_manifest.json", {"algorithm": "sha256", "files": {
            "evaluations.jsonl": {"sha256": b.digest(raw), "size_bytes": len(raw)}}})
        self.proof = {"status": "verified", "mode": "production", "manifest_sha256": self.spec["evaluation_manifest"]["sha256"],
                      "records": len(self.rows), "exact_request_coverage": True, "process_release_verified": True,
                      "artifact_manifest_sha256": artifact["sha256"], "evaluations_sha256": b.digest(raw),
                      "evaluations": str(self.result / "evaluations.jsonl"), "run_token": "authored-evaluation"}
        self.spec["evaluation_artifact_manifest"] = artifact
        self.spec["evaluation_verification"] = self.put(self.stage / "control/independent_verification.json", self.proof)

    def load_manifest(self, path, digest):
        self.manifest_loader_calls += 1
        b.require(b.sha(path) == digest, "Authored manifest binding mismatch")
        return copy.deepcopy(self.manifest)

    def verify(self, path, digest):
        self.verifier_calls += 1
        b.require(b.sha(path) == digest, "Authored manifest binding mismatch")
        return copy.deepcopy(self.proof)

    def verify_bundle(self, path, digest):
        self.bundler_calls += 1
        b.require(b.sha(path) == digest, "Authored bundle binding mismatch")
        return copy.deepcopy(self.bundle)

    def context(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(b, "MASTER_SHA", self.master["sha256"]))
            stack.enter_context(patch.object(b, "PARENT_SHA", self.parent["sha256"]))
            return b.context(self.spec, self.loaded)


class BundleAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.fixture = Fixture(self.root)

    def test_full_authored_success_separates_overlapping_cells_and_preserves_every_byte(self):
        f = self.fixture
        ctx = f.context()
        with self.assertRaisesRegex(ValueError, "Duplicate matched"):
            metrics.analyze(ctx["rows"], f.conditions, plan())
        output, recompute = self.root / "output", self.root / "recomputed"
        result = b.analyze_into(output, ctx)
        self.assertEqual(result["rows"], 20)
        for name in b.COMPONENTS:
            directory = output / "components" / name
            subset = (directory / "evaluations.jsonl").read_bytes()
            self.assertEqual(subset, b"".join(line for line, row in zip(ctx["lines"], ctx["rows"])
                                            if row["request_id"] in ctx["component_ids"][name]))
            report = b.parse((directory / "metrics.json").read_bytes())
            self.assertEqual(report["rows"], 10)
            self.assertEqual(report["method"]["bootstrap_resamples"], 2000)
            self.assertFalse(report["by_split"]["untouched_test"]["local"]["targets"]["target"]
                             ["promotion"]["eligible_for_validation_promotion"])
        proof = b.verify_recompute(output, recompute, f.context())
        self.assertTrue(proof["exact_recomputation_match"])
        self.assertFalse(proof["process_exit_and_release_verified_by_this_module"])
        self.assertEqual(b.inventory(output), b.inventory(recompute))

    def test_pending_spec_blocks_before_bundle_or_evaluation_loading(self):
        self.fixture.spec["runnable"] = False
        with self.assertRaisesRegex(ValueError, "Pending"):
            self.fixture.context()
        self.assertEqual((self.fixture.bundler_calls, self.fixture.verifier_calls), (0, 0))

    def test_missing_positive_final_freeze_precedes_evaluation_verifier(self):
        f = self.fixture
        final = b.parse((f.bundle_root / "input/final_config.json").read_bytes())
        final["status"] = "pending"
        f.bundle["frozen_final_config"] = f.put_relative("input/final_config.json", final)
        f.rebind_bundle()
        with self.assertRaisesRegex(ValueError, "positive final"):
            f.context()
        self.assertEqual(f.verifier_calls, 0)
        self.assertEqual(f.bundler_calls, 0)

    def test_failed_or_incomplete_external_receipt_precedes_outcome_reads(self):
        f = self.fixture
        for field, value in (("process_release_verified", False), ("records", 19), ("status", "failed")):
            proof = {**f.proof, field: value}
            f.spec["evaluation_verification"] = f.put(f.stage / "control/independent_verification.json", proof)
            with self.assertRaisesRegex(ValueError, "Incomplete independent"):
                f.context()
        self.assertEqual(f.verifier_calls, 0)

    def test_changed_runtime_source_rejected_before_bundle_loader(self):
        self.fixture.spec["source_bindings"]["eval_run.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "runtime source"):
            self.fixture.context()
        self.assertEqual(self.fixture.bundler_calls, 0)

    def test_wrong_producer_source_even_with_rebound_manifest_fails(self):
        f = self.fixture
        f.manifest["source_files"]["infra/gpu03/direction_discovery/eval_run.py"]["sha256"] = "0" * 64
        f.rebind_manifest()
        with self.assertRaisesRegex(ValueError, "verifier source"):
            f.context()
        self.assertEqual(f.manifest_loader_calls, 0)

    def test_exact_bundle_link_allows_staged_copy_but_rejects_wrong_or_absent_link(self):
        f = self.fixture
        copy_path = f.stage / "input/frozen_bundle.json"
        copy_path.write_bytes((f.bundle_root / "bundle.json").read_bytes())
        f.manifest["scientific"]["test_bundle"] = {"path": str(copy_path), "sha256": f.spec["bundle"]["sha256"]}
        f.rebind_manifest()
        self.assertEqual(len(f.context()["rows"]), 20)
        f.manifest["scientific"]["test_bundle"]["sha256"] = "0" * 64
        f.rebind_manifest()
        with self.assertRaisesRegex(ValueError, "exact frozen bundle"):
            f.context()
        del f.manifest["scientific"]["test_bundle"]
        f.rebind_manifest()
        with self.assertRaisesRegex(ValueError, "exact frozen bundle"):
            f.context()

    def test_partial_coverage_fails_even_with_rebound_success_receipt(self):
        f = self.fixture
        f.rows.pop()
        f.rebind_evaluation()
        with self.assertRaisesRegex(ValueError, "Incomplete independent"):
            f.context()

    def test_duplicate_evaluation_id_and_changed_identity_fail(self):
        f = self.fixture
        original = copy.deepcopy(f.rows)
        f.rows[-1] = copy.deepcopy(f.rows[0])
        f.rebind_evaluation()
        with self.assertRaisesRegex(ValueError, "duplicate evaluation coverage"):
            f.context()
        f.rows = original
        f.rows[0]["seed"] += 1
        f.rebind_evaluation()
        with self.assertRaisesRegex(ValueError, "evaluation identity"):
            f.context()

    def test_component_id_overlap_or_dropped_auxiliary_group_fails(self):
        f = self.fixture
        f.bundle["auxiliary_groups"]["correct_harmful"].pop()
        f.rebind_bundle()
        with self.assertRaisesRegex(ValueError, "strata do not partition"):
            f.context()
        f = Fixture(self.root / "another")
        f.bundle["components"]["auxiliary_test"]["request_ids"].append(f.bundle["components"]["core_untouched_test"]["request_ids"][0])
        f.rebind_bundle()
        with self.assertRaisesRegex(ValueError, "frozen request IDs"):
            f.context()

    def test_duplicate_matched_cells_inside_component_fail_even_with_distinct_ids(self):
        f = self.fixture
        previous_id = f.rows[-1]["request_id"]
        f.rows[-1] = {**copy.deepcopy(f.rows[-2]), "request_id": previous_id}
        f.rebind_requests()
        with self.assertRaisesRegex(ValueError, "Duplicate matched"):
            f.context()

    def test_complete_wrong_request_plan_input_fails(self):
        f = self.fixture
        (f.stage / "input/request_plan.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "request input mismatch"):
            f.context()

    def test_snapshot_mutation_after_verification_fails(self):
        f = self.fixture
        original = f.evaluator.verify
        def mutate(path, digest):
            proof = original(path, digest)
            with (f.result / "evaluations.jsonl").open("ab") as stream:
                stream.write(b"\n")
            return proof
        f.evaluator.verify = mutate
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            f.context()

    def test_rehashed_subset_mutation_cannot_pass_fresh_recomputation(self):
        f = self.fixture
        output = self.root / "output"
        b.analyze_into(output, f.context())
        path = output / "components/core_untouched_test/evaluations.jsonl"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes().replace(b'"authored_note": "', b'"altered_note": "', 1))
        artifact = output / "artifact_manifest.json"
        artifact.chmod(0o600)
        current = b.inventory(output)
        f.put(artifact, {"algorithm": "sha256", "files": {k: v for k, v in current.items() if k != "artifact_manifest.json"}})
        with self.assertRaisesRegex(ValueError, "recomputation differs"):
            b.verify_recompute(output, self.root / "recomputed", f.context())

    def test_output_reuse_input_overlap_and_recompute_overlap_fail(self):
        f = self.fixture
        ctx = f.context()
        output = self.root / "output"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "must be fresh"):
            b.analyze_into(output, ctx)
        with self.assertRaisesRegex(ValueError, "overlaps input"):
            b.analyze_into(f.result / "analysis", ctx)
        with self.assertRaisesRegex(ValueError, "outputs overlap"):
            b.verify_recompute(output, output / "recomputed", ctx)

    def test_empty_auxiliary_is_explicit_unavailable(self):
        f = Fixture(self.root / "empty", empty_aux=True)
        output = f.root / "output"
        b.analyze_into(output, f.context())
        result = b.parse((output / "summary.json").read_bytes())
        self.assertEqual(result["components"]["auxiliary_test"], {"status": "unavailable_no_eligible_pairs", "rows": 0})
        self.assertFalse((output / "components/auxiliary_test/metrics.json").exists())

    def test_runner_verifier_requires_producer_release_before_context(self):
        f = self.fixture
        f.put(f.stage / "input/analysis_spec.json", f.spec)
        common = SimpleNamespace(STAGE=f.stage, frozen=lambda _: {"output": f.spec["output"],
            "recomputation_output": f.spec["recomputation_output"]}, require_start=lambda *args: None,
            release=lambda *args: (_ for _ in ()).throw(ValueError("Producer release absent")))
        with patch.object(b, "context") as load:
            with self.assertRaisesRegex(ValueError, "release absent"):
                b.run_under_runner(common, "verifier", "a" * 64)
            load.assert_not_called()

    def test_runner_adapter_authored_success_through_producer_and_verifier(self):
        f = self.fixture
        f.put(f.stage / "input/analysis_spec.json", f.spec)
        calls = []
        def release(operation, config, payload):
            calls.append("release-" + operation)
            return {"result": {"artifact_manifest_sha256": b.sha(Path(config["output"]) / "artifact_manifest.json")}}
        common = SimpleNamespace(STAGE=f.stage, frozen=lambda _: {"output": f.spec["output"],
            "recomputation_output": f.spec["recomputation_output"]},
            require_start=lambda operation, *args: calls.append("start-" + operation), release=release)
        with patch.object(b, "modules", return_value=f.loaded), patch.object(b, "MASTER_SHA", f.master["sha256"]), \
                patch.object(b, "PARENT_SHA", f.parent["sha256"]):
            b.run_under_runner(common, "producer", "a" * 64)
            proof = b.run_under_runner(common, "verifier", "a" * 64)
        self.assertEqual(calls, ["start-producer", "start-verifier", "release-producer"])
        self.assertTrue(proof["exact_recomputation_match"])

    def test_runner_output_drift_precedes_analysis(self):
        f = self.fixture
        f.put(f.stage / "input/analysis_spec.json", f.spec)
        common = SimpleNamespace(STAGE=f.stage, frozen=lambda _: {"output": str(self.root / "changed"),
            "recomputation_output": f.spec["recomputation_output"]}, require_start=lambda *args: None)
        with patch.object(b, "context") as load:
            with self.assertRaisesRegex(ValueError, "outputs differ"):
                b.run_under_runner(common, "producer", "a" * 64)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
