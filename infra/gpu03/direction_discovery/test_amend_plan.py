import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import amend_plan as a
import cache_package


def parent_plan():
    return {"purpose": "checkpoint60_direction_discovery_and_causal_ablation", "schema_version": 1,
            "host": "gpu-04", "no_training": True, "status": "frozen_before_activations_or_causal_outcomes",
            "dataset": {"primary_fit_records": 333, "split": [117, 37, 33]},
            "fit": {"seed": 6001, "layers": list(range(36)), "windows": ["pre_definition", "pre_body", "transition", "early_body"]},
            "budget": {"maximum_candidate_cpu_wall_seconds": 14400}, "sampling": {"temperature": .7},
            "generation": {"seed": 6007}, "sweep": {"random_seed_bases": [6101, 6102, 6103]},
            "model": {"dtype": "bfloat16", "revision": "original"}, "known_limitations": ["retained"],
            "inputs": {"raw_artifact_manifest_sha256": "a" * 64, "prepared_records_sha256": "b" * 64,
                       "exclusions_sha256": "c" * 64, "prior_raw_review_manifest_sha256": "d" * 64}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(a.canonical(value) + "\n")


class AmendmentTests(unittest.TestCase):
    def phases(self, root, parent):
        references = {}
        for role, phase in a.PHASES.items():
            stage, output = root / role / "stage", root / role / "output"
            manifest = stage / "reviewed_manifest.json"
            m = {"phase": phase, "host": "gpu-04", "run_token": role, "stage": str(stage), "output": str(output),
                 "gpu_ids": [0], "workers": [{"name": "gpu_0"}],
                 "scientific": {"master_plan_sha256": a.PARENT_SHA256, "input_prepared_sha256": parent["inputs"]["prepared_records_sha256"]}}
            write(manifest, m)
            digest = a.sha(manifest)
            output.mkdir()
            shutil.copyfile(manifest, output / "reviewed_manifest.json")
            write(output / "campaign_summary.json", {"status": "succeeded", "run_token": role, "manifest_sha256": digest,
                  "gpu_release_verified": True, "worker_exit_codes": [0]})
            write(output / "gpu_release.json", {"verified": True, "gpu_ids": [0]})
            write(stage / "control/supervisor_exit.json", {"manifest_sha256": digest, "run_token": role, "service_result": "success",
                  "exit_code_kind": "exited", "exit_status": "0", "failure_present": False})
            if role == "diagnosis":
                write(output / "workers/gpu_0/prefix_audit_verdict.json", {"passed": True, "failure_reasons": []})
                measurements = {"compute_dtypes": ["bfloat16", "float32"], "expected_records_per_model": 3, "model_results": {}}
                for kind in ("h0", "h60"):
                    measurements["model_results"][kind] = {}
                    for dtype in ("bfloat16", "float32"):
                        equal = {str(i): {"all_selected": {"bitwise_equal": True}} for i in range(3)}
                        measurements["model_results"][kind][dtype] = {
                            "A_variable_shape_identical_prefix": {str(i): {"all_selected": {
                                "maximum_layer_relative_l2": .03 if dtype == "bfloat16" else .000007}} for i in range(3)},
                            "B_fixed_shape_future_invariance": equal, "C_common_shape_prefix_invariance": equal,
                            "D_canonical_prompt_repeat": {"bitwise_equal": True}, "A_vs_F_same_shape_backend_changed": equal}
                write(output / "workers/gpu_0/prefix_audit_all_measurements.json", measurements)
            else:
                write(output / "workers/gpu_0/fixed_cache_report.json", {"status": "succeeded", "activation_cache_profile": {
                    "padded_sequence_length": 2176, "padding_attention_mask": 0, "attention_backend": "torch_sdpa_MATH_only"},
                    "all_identical_prefixes_bitwise_equal": True, "all_native_readbacks_bitwise_equal": True,
                    "fp32_delta_readback_exact": True, "raw_activations_retained": True,
                    "qualifications": {kind: {key: {"bitwise_equal": True} for key in ("repeat", "future_causality")}
                                       for kind in ("h0", "h60")}})
            files = {str(p.relative_to(output)): {"sha256": a.sha(p), "size_bytes": p.stat().st_size}
                     for p in output.rglob("*") if p.is_file()}
            write(output / "artifact_manifest.json", {"algorithm": "sha256", "files": files})
            artifact_digest = a.sha(output / "artifact_manifest.json")
            proof = stage / "independent_verification.json"
            write(proof, {"status": "verified", "manifest_sha256": digest, "run_token": role, "gpu_release_verified": True,
                          "artifact_manifest_sha256": artifact_digest, "artifact_files": len(files)})
            references[role] = {"manifest_path": str(manifest), "manifest_sha256": digest,
                                "artifact_manifest_sha256": artifact_digest,
                                "verification_receipt_path": str(proof), "verification_receipt_sha256": a.sha(proof)}
        return references

    def setup_evidence(self, root):
        parent = parent_plan()
        path = root / "plan_v1.json"
        write(path, parent)
        self.addCleanup(patch.stopall)
        patch.object(a, "PARENT_SHA256", a.sha(path)).start()
        refs = self.phases(root, parent)
        old = root / "old_candidates"
        write(old / "artifact_manifest.json", {"old": "preserved"})
        old_sha = a.sha(old / "artifact_manifest.json")
        patch.object(a, "SUPERSEDED_CANDIDATE_SHA256", old_sha).start()
        refs["superseded_candidates"] = {"root": str(old), "artifact_manifest_sha256": old_sha}
        evidence = root / "evidence.json"
        write(evidence, refs)
        return parent, path, refs, evidence

    def inspected(self, refs):
        return {"status": "verified", "records": 561, "problems": 187, "index_joined_to_artifact_manifest": True,
                "manifest_sha256": refs["core_cache"]["manifest_sha256"]}

    def test_full_preparation_immutable_provenance_exact_science_no_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, path, refs, evidence = self.setup_evidence(root)
            before = path.read_bytes()
            with patch.object(cache_package, "inspect", return_value=self.inspected(refs)):
                result = a.create(path, a.sha(path), evidence, a.sha(evidence), root / "plan_v2")
            self.assertEqual(result["status"], "prepared")
            self.assertFalse(result["launch_performed"])
            new_path = Path(result["plan_path"])
            new = json.loads(new_path.read_text())
            a.verify_unchanged(parent, new)
            self.assertEqual(new["parent_plan_sha256"], a.sha(path))
            self.assertEqual(new["inputs"]["raw_artifact_manifest_sha256"], refs["core_cache"]["artifact_manifest_sha256"])
            self.assertEqual(new["inputs"]["exclusions_sha256"], parent["inputs"]["exclusions_sha256"])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(new_path.stat().st_mode & 0o222, 0)
            for role in a.PHASES:
                self.assertTrue((root / "plan_v2/input" / role / "independent_verification.json").is_file())
            with self.assertRaisesRegex(RuntimeError, "new immutable"):
                a.create(path, a.sha(path), evidence, a.sha(evidence), root / "plan_v2")

    def test_failed_receipt_creates_no_amendment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path, refs, evidence = self.setup_evidence(root)
            proof_path = Path(refs["qualification"]["verification_receipt_path"])
            proof = json.loads(proof_path.read_text())
            proof["status"] = "failed"
            write(proof_path, proof)
            refs["qualification"]["verification_receipt_sha256"] = a.sha(proof_path)
            write(evidence, refs)
            with self.assertRaisesRegex(RuntimeError, "independent receipt"):
                a.create(path, a.sha(path), evidence, a.sha(evidence), root / "plan_v2")
            self.assertFalse((root / "plan_v2").exists())

    def test_wrong_artifact_or_stale_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, _, refs, _ = self.setup_evidence(root)
            ref = copy.deepcopy(refs["diagnosis"])
            ref["artifact_manifest_sha256"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "independent receipt"):
                a.verify_phase(ref, "prefix_audit", parent)
            manifest = Path(refs["diagnosis"]["manifest_path"])
            manifest.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "hash-mismatched"):
                a.verify_phase(refs["diagnosis"], "prefix_audit", parent)

    def test_frozen_fit_splits_seeds_budget_and_nuisance_definitions_cannot_change(self):
        parent = parent_plan()
        evidence = {"core_cache": {"artifact_manifest_sha256": "e" * 64, "output": "/scratch/new", "reviewed_manifest_sha256": "f" * 64},
                    "superseded_candidates": {"artifact_manifest_sha256": a.SUPERSEDED_CANDIDATE_SHA256}}
        new = a.amend(parent, evidence, a.PARENT_SHA256, "0" * 64)
        for key in ("dataset", "fit", "budget", "sampling", "generation", "sweep", "model", "known_limitations"):
            changed = copy.deepcopy(new)
            changed[key] = {"unreviewed_change": True}
            with self.assertRaisesRegex(RuntimeError, "changed frozen science"):
                a.verify_unchanged(parent, changed)

    def test_no_v2_when_semantic_cache_audit_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path, refs, evidence = self.setup_evidence(root)
            result = self.inspected(refs)
            result["records"] = 560
            with patch.object(cache_package, "inspect", return_value=result), self.assertRaisesRegex(RuntimeError, "core package semantics"):
                a.create(path, a.sha(path), evidence, a.sha(evidence), root / "plan_v2")
            self.assertFalse((root / "plan_v2").exists())

    def test_nonzero_actual_service_exit_rejected_despite_stale_success_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, _, refs, _ = self.setup_evidence(root)
            m = json.loads(Path(refs["core_cache"]["manifest_path"]).read_text())
            exit_path = Path(m["stage"]) / "control/supervisor_exit.json"
            receipt = json.loads(exit_path.read_text())
            receipt["exit_status"] = "9"
            write(exit_path, receipt)
            with self.assertRaisesRegex(RuntimeError, "actual systemd"):
                a.verify_phase(refs["core_cache"], "fixed_cache_core", parent)


if __name__ == "__main__":
    unittest.main()
