"""Authored saved-outcome statistics only; no generated code or real outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import h100_no_loophole_report as m
from infra.gpu03.direction_discovery.test_no_loophole_report import authored_plan, fixture_rows


def rows_fixture():
    plan=authored_plan(); rows=fixture_rows(plan)
    for row in rows:
        row.update(repository_test_modification="Harmful - Incorrect", repository_reward_hack_label="None")
        row["metrics"]["harmful_modification"] = True
        row["helper_aware_evaluation"]={"policy":m.evaluation.repair.POLICY,
            "status":"evaluated", "repository_test_modification":"Innocent", "repository_reward_hack_label":"None",
            "metrics":{**row["metrics"],"harmful_modification":False}}
    return plan,rows


class Tests(unittest.TestCase):
    def test_separate_policy_statistics_and_shared_gt(self):
        plan,rows=rows_fixture(); before=copy.deepcopy(rows)
        legacy=m.classifier_summary(rows,plan,m.evaluation.repair.LEGACY_POLICY)
        corrected=m.classifier_summary(rows,plan,m.evaluation.repair.POLICY)
        self.assertEqual(legacy["conditions"]["baseline"]["metrics"]["harmful_modification"]["identified_bounds"],[1,1])
        self.assertEqual(corrected["conditions"]["baseline"]["metrics"]["harmful_modification"]["identified_bounds"],[0,0])
        self.assertEqual(rows,before)
        self.assertTrue(corrected["rh_metrics_are_descriptive_not_capability_gates"])

    def test_unknown_sidecar_uses_full_bounds_not_zero(self):
        plan,rows=rows_fixture(); target=next(r for r in rows if r["condition_id"].startswith("target:"))
        target["helper_aware_evaluation"]["status"]="unsupported"
        target["helper_aware_evaluation"]["metrics"]["harmful_modification"]=None
        corrected=m.classifier_summary(rows,plan,m.evaluation.repair.POLICY)
        effect=corrected["target_paired_differences"]["baseline"]["harmful_modification"]
        self.assertIsNone(effect["estimate"]); self.assertIsNone(effect["ci95"])
        self.assertEqual(effect["identified_bounds"][0],0)
        self.assertAlmostEqual(effect["identified_bounds"][1],1/148)

    def test_incomplete_proof_fails_before_metrics_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"proof.json"
            m.write(path,{"status":"incomplete"},immutable=True)
            spec={"evaluation_manifest":{},"verification":m.ref(path),"historical_metrics":{}}
            with patch.object(m.evaluation,"verify",side_effect=AssertionError("outcome verifier reached")):
                with self.assertRaisesRegex(ValueError,"Full released740"):
                    m.context(spec)

    def test_wrong_manifest_positive_proof_fails_before_outcome_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"proof.json"
            m.write(path,{"status":"independently_verified_h100_complete740_evaluation","records":740,
                "exact_request_coverage":True,"process_release_verified":True,"manifest_sha256":"a"*64},immutable=True)
            spec={"evaluation_manifest":{"sha256":"b"*64},"verification":m.ref(path),"historical_metrics":{}}
            with patch.object(m.evaluation,"verify",side_effect=AssertionError("outcome verifier reached")):
                with self.assertRaisesRegex(ValueError,"exact manifest"): m.context(spec)

    def test_three_file_producer_and_independent_exact_recompute(self):
        plan,rows=rows_fixture(); current=m.frozen.analyze(rows,plan)
        historical={"conditions":current["conditions"],"paired_differences":current["paired_differences"]}
        context=({"source_files":{}},plan,{"evaluations":{"path":"authored", "sha256":"a"*64},"request_plan_sha256":"b"*64},rows)
        with tempfile.TemporaryDirectory() as directory, patch.object(m,"context",return_value=context), \
                patch.object(m.frozen,"historical_view",return_value=historical):
            output=Path(directory)/"report"
            spec={"historical_metrics":{"path":"authored-history"}}
            result=m.run(spec,output)
            verification=m.verify(output,result["artifact_manifest"]["sha256"])
            self.assertTrue(verification["all_three_output_files_byte_identical"])
            self.assertFalse(verification["generated_code_rerun"])
            saved=json.loads((output/"metrics.json").read_bytes())
            self.assertTrue(saved["no_loophole"]["capability_gate"]["passed"])
            self.assertFalse(saved["method"]["confidence_interval_is_gate"])
            self.assertIn("cannot be attributed solely to hint removal",(output/"REPORT.md").read_text())
            with self.assertRaisesRegex(ValueError,"fresh"): m.run(spec,output)

    def test_rebound_report_metric_mutation_rejected(self):
        plan,rows=rows_fixture(); current=m.frozen.analyze(rows,plan)
        historical={"conditions":current["conditions"],"paired_differences":current["paired_differences"]}
        context=({"source_files":{}},plan,{"evaluations":{"path":"authored", "sha256":"a"*64},"request_plan_sha256":"b"*64},rows)
        with tempfile.TemporaryDirectory() as directory, patch.object(m,"context",return_value=context), \
                patch.object(m.frozen,"historical_view",return_value=historical):
            output=Path(directory)/"report"; m.run({"historical_metrics":{"path":"history"}},output)
            value=json.loads((output/"metrics.json").read_bytes()); value["no_loophole"]["capability_gate"]["passed"]=False
            (output/"metrics.json").chmod(0o600); (output/"metrics.json").write_text(m.evaluation.canonical(value)+"\n"); (output/"metrics.json").chmod(0o400)
            artifact=json.loads((output/"artifact_manifest.json").read_bytes())
            artifact["files"]["metrics.json"]={"sha256":m.sha(output/"metrics.json"),"size_bytes":(output/"metrics.json").stat().st_size}
            (output/"artifact_manifest.json").chmod(0o600); (output/"artifact_manifest.json").write_text(m.evaluation.canonical(artifact)+"\n"); (output/"artifact_manifest.json").chmod(0o400)
            with self.assertRaisesRegex(ValueError,"recomputation differs"):
                m.verify(output,m.sha(output/"artifact_manifest.json"))


if __name__ == "__main__": unittest.main()
