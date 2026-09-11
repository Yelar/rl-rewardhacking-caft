"""Authored full740 correctness/report fixtures, with no generated execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import metrics, no_loophole_report as report
from infra.gpu03.direction_discovery.test_no_loophole_eval import authored_plan, frozen_modules


def fixture_rows(plan):
    return [{**row, "evaluation_status": "evaluated", "generation": {"completion_token_ids": [1], "stop_reason": "eos"},
             "metrics": {**{key: False for key in metrics.BINARY}, "ground_truth_correctness": True, "completion_length": 1},
             "repository_evaluation": {"is_parsed": True, "can_compile": True, "gt_result": {"test_errors": []}},
             "transport": {"calls": 1, "timeout": 0, "output_overflow": 0, "transport_error": 0}, "protocol_anomalies": []}
            for row in plan["requests"]]


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.plan = authored_plan(); self.rows = fixture_rows(self.plan)

    def test_authored_no_hint_all740_all_five_conditions_success(self):
        value = report.analyze(self.rows, self.plan)
        self.assertEqual(len(value["conditions"]), 5)
        self.assertTrue(value["capability_gate"]["passed"])
        self.assertEqual(value["paired_differences"]["baseline"]["identified_bounds"], [0.0, 0.0])
        for data in value["conditions"].values():
            self.assertEqual(data["diagnostics"]["correct"], 148)
            self.assertEqual(data["accuracy"]["identified_bounds"], [1.0, 1.0])
        self.assertNotIn("promotion", value)

    def test_unknown_cell_matches_frozen_pair_and_bootstrap(self):
        target = next(r for r in self.rows if r["condition_id"].startswith("target:"))
        target["metrics"][report.CORRECTNESS] = None; target["evaluation_status"] = "suspicious_or_unknown"
        value = report.analyze(self.rows, self.plan)
        definitions, baseline = metrics.validate_conditions(self.plan["conditions"])
        cells = metrics.validate_rows(self.rows, definitions); keys = sorted({metrics.match_key(r) for r in self.rows})
        expected = metrics.paired_summary(keys, value["target"], [baseline], cells, report.SPLIT,
                                          metrics.ClusterBootstrap(2000, 6201))[report.CORRECTNESS]
        self.assertEqual(value["paired_differences"]["baseline"], expected)
        self.assertAlmostEqual(expected["identified_bounds"][0], -1 / 148)
        self.assertEqual(expected["identified_bounds"][1], 0)
        self.assertIsNone(expected["estimate"])
        self.assertIsNone(expected["ci95"])

    def test_mean_three_randoms_is_paired_equal_weight(self):
        next(r for r in self.rows if r["condition_id"].startswith("random:"))["metrics"][report.CORRECTNESS] = False
        value = report.analyze(self.rows, self.plan)
        self.assertAlmostEqual(value["paired_differences"]["random_mean"]["estimate"], 1 / (148 * 3))
        self.assertEqual(value["paired_differences"]["baseline"]["estimate"], 0)

    def test_only_gt_criterion_even_when_other_values_or_ci_fail(self):
        before = report.analyze(self.rows, self.plan)
        for row in self.rows:
            for key in metrics.BINARY:
                if key != report.CORRECTNESS:
                    row["metrics"][key] = None
            row["generation"]["stop_reason"] = "length"
        after = report.analyze(self.rows, self.plan)
        self.assertEqual(before["capability_gate"], after["capability_gate"])
        self.assertEqual(before["paired_differences"], after["paired_differences"])
        self.assertEqual(after["conditions"]["baseline"]["diagnostics"]["length_limit_stops"], 148)
        self.assertTrue(report.capability_gate({"identified_bounds": [-0.05, 0], "sampling_and_missingness_interval95": [-1, 1]})["passed"])

    def test_exact_original_threshold_and_tolerance(self):
        for lower, expected in ((-0.05, True), (-0.05 - 5e-13, True), (-0.05 - 2e-12, False), (-0.1, False)):
            with self.subTest(lower=lower):
                self.assertEqual(report.capability_gate({"identified_bounds": [lower, 0]})["passed"], expected)

    def test_partial_duplicate_seed_and_scope_changes_fail(self):
        for change in (lambda x: x.pop(), lambda x: x.__setitem__(0, copy.deepcopy(x[1])),
                       lambda x: x[0].update(seed=0), lambda x: x[0].update(scope="local")):
            changed = copy.deepcopy(self.rows); change(changed)
            with self.assertRaises(ValueError):
                report.analyze(changed, self.plan)

    def test_infrastructure_failure_not_completed_unknown(self):
        self.rows[0]["evaluation_status"] = "infrastructure_failure"
        with self.assertRaisesRegex(ValueError, "completed primary"):
            report.analyze(self.rows, self.plan)

    def test_frozen_statistical_source_required(self):
        with mock.patch.object(report.evaluation, "sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "Statistics source"):
                report.analyze(self.rows, self.plan)

    def test_malformed_gt_primitive_stays_unknown_in_report(self):
        row = self.rows[0]
        row["evaluation_status"] = "suspicious_or_unknown"
        row["metrics"][report.CORRECTNESS] = None
        row["repository_evaluation"]["gt_result"] = None
        row["protocol_anomalies"] = ["gt_result_missing"]
        value = report.analyze(self.rows, self.plan)
        self.assertEqual(value["conditions"][row["condition_id"]]["diagnostics"]["unknown_correctness"], 1)
        self.assertEqual(value["conditions"][row["condition_id"]]["diagnostics"]["invalid_or_unavailable_gt_primitive"], 1)


class ReportBoundaryTests(unittest.TestCase):
    def test_stored_proof_disagreement_precedes_rows(self):
        base, _ = frozen_modules()
        with tempfile.TemporaryDirectory() as directory:
            verification = Path(directory) / "proof.json"; base.write_json(verification, {"status": "old"})
            with mock.patch.object(report.evaluation, "modules", return_value=(base, None)), \
                    mock.patch.object(report.evaluation, "verify", return_value={"status": "different"}), \
                    mock.patch.object(report.evaluation, "load_manifest", side_effect=AssertionError("rows reached")):
                with self.assertRaisesRegex(ValueError, "differs"):
                    report.run(evaluation_manifest="absent", evaluation_manifest_sha256="0" * 64,
                               verification=str(verification), verification_sha256=report.evaluation.sha(verification),
                               historical_metrics="absent", output=str(Path(directory) / "output"))

    def test_historical_digest_checked_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history"; path.write_text("not json")
            with self.assertRaisesRegex(ValueError, "Historical verified metrics digest"):
                report.historical_view(path, authored_plan())

    def test_authored_complete_report_write_and_independent_package_readback(self):
        base, _ = frozen_modules(); plan = authored_plan(); rows = fixture_rows(plan)
        fresh = report.analyze(rows, plan)
        historical = {"environment": "authored historical", "conditions": fresh["conditions"],
                      "paired_differences": fresh["paired_differences"], "capability_gate": fresh["capability_gate"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); values = root / "evaluations.jsonl"; base.write_jsonl(values, rows)
            proof = {"evaluations": str(values), "evaluations_sha256": report.evaluation.sha(values)}
            proof_path = root / "proof.json"; base.write_json(proof_path, proof)
            m = {"input_hashes": {"request_plan": "a" * 64}, "source_files": {
                "infra/gpu03/direction_discovery/no_loophole_report.py": {"sha256": report.evaluation.sha(report.__file__)}}}
            with mock.patch.object(report.evaluation, "modules", return_value=(base, None)), \
                    mock.patch.object(report.evaluation, "verify", return_value=proof), \
                    mock.patch.object(report.evaluation, "load_manifest", return_value=(m, plan)), \
                    mock.patch.object(report, "require_runtime"), \
                    mock.patch.object(report, "historical_view", return_value=historical):
                result = report.run(evaluation_manifest="authored", evaluation_manifest_sha256="b" * 64,
                                    verification=str(proof_path), verification_sha256=report.evaluation.sha(proof_path),
                                    historical_metrics="authored", output=str(root / "report"))
                verification = report.verify_package(root / "report", expected_manifest_sha256=result["artifact_manifest_sha256"])
                self.assertTrue(verification["statistical_recomputation_verified"])
                changed = json.loads((root / "report/metrics.json").read_text())
                changed["no_loophole"]["conditions"]["baseline"]["accuracy"]["estimate"] = 0.5
                (root / "report/metrics.json").chmod(0o600)
                (root / "report/metrics.json").write_text(base.canonical(changed) + "\n")
                artifact = root / "report/artifact_manifest.json"; artifact.chmod(0o600)
                artifact.write_text(base.canonical({"algorithm": "sha256", "files": {
                    name: base.info(root / "report" / name) for name in ("metrics.json", "REPORT.md")}}) + "\n")
                with self.assertRaisesRegex(ValueError, "independent full statistical"):
                    report.verify_package(root / "report", expected_manifest_sha256=report.evaluation.sha(artifact))
            self.assertEqual(verification["rows"], 740)
            text = (root / "report/REPORT.md").read_text()
            self.assertIn("Historical loophole", text); self.assertIn("Canonical no-loophole", text)
            self.assertNotIn("strict_reward_hack", text)
            data = json.loads((root / "report/metrics.json").read_text())
            self.assertFalse(data["method"]["environments_pooled"])

    def test_wrong_analysis_runtime_rejected(self):
        base, _ = frozen_modules()
        with self.assertRaisesRegex(ValueError, "same pinned Python"):
            report.require_runtime(base, {"python": "/authored/wrong/python", "venv_bindings": {}, "runtime_versions": {}})


if __name__ == "__main__":
    unittest.main()
