"""Prepare an immutable, evidence-bound repair amendment; never launch work.

Inputs are explicit SHA-bound independent verification receipts. Complete raw
payload hashing belongs to the completed independent verification, not this
small plan-construction step. The repaired cache is also semantically inspected.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

PARENT_SHA256 = "d4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10"
SUPERSEDED_CANDIDATE_SHA256 = "5dad334834d1647631f7791095c18c4d4fc592699c910f376448fe303d2cdc9e"
PHASES = {"diagnosis": "prefix_audit", "qualification": "fixed_cache_qualification", "core_cache": "fixed_cache_core"}
REPAIR_REASON = ("Identical-prefix activations varied with total sequence length in the original native BF16 cache. "
                 "Controlled fixed-shape future perturbations and repeats passed exactly; FP32 substantially reduced "
                 "variable-shape differences. Exclusive math attention did not remove the observed variable-shape "
                 "differences. Re-extract unchanged original token sequences at batch one, right-padded to 2176 with "
                 "attention mask zero on padding, explicit positions and exclusive math SDPA. Preserve old raw tensors "
                 "and fitted candidates as diagnostic artifacts; refit the unchanged estimator on the verified repaired cache.")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def checked(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", digest or "") and sha(path) == digest,
            "missing or hash-mismatched amendment evidence: " + str(path))
    return path


def json_checked(path, digest):
    return json.loads(checked(path, digest).read_text())


def artifact_file(root, manifest, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe evidence artifact path")
    path = checked(Path(root) / relative, manifest["files"][name]["sha256"])
    require(path.stat().st_size == manifest["files"][name]["size_bytes"], "evidence artifact size mismatch")
    return path


def diagnostic_summary(measurements):
    require(measurements["compute_dtypes"] == ["bfloat16", "float32"] and
            measurements["expected_records_per_model"] == 3 and set(measurements["model_results"]) == {"h0", "h60"},
            "diagnosis lacks the full native/FP32 matched-triplet comparison")
    peaks = {}
    for kind, models in measurements["model_results"].items():
        peaks[kind] = {}
        for dtype in ("bfloat16", "float32"):
            data = models[dtype]
            variation = data["A_variable_shape_identical_prefix"]
            require(len(variation) == 3, "diagnosis variable-shape pair coverage incomplete")
            peaks[kind][dtype] = max(item["all_selected"]["maximum_layer_relative_l2"] for item in variation.values())
            for key in ("B_fixed_shape_future_invariance", "C_common_shape_prefix_invariance"):
                require(len(data[key]) == 3 and all(v["all_selected"]["bitwise_equal"] is True for v in data[key].values()),
                        "diagnostic fixed-shape future/prefix comparisons did not pass exactly")
            require(data["D_canonical_prompt_repeat"]["bitwise_equal"] is True, "diagnostic prompt repeat did not pass exactly")
        require(0 <= peaks[kind]["float32"] < peaks[kind]["bfloat16"], "diagnosis does not support reduced FP32 shape variation")
        changed_backend = models["bfloat16"]["A_vs_F_same_shape_backend_changed"]
        require(len(changed_backend) == 3 and all(v["all_selected"]["bitwise_equal"] is True for v in changed_backend.values()),
                "diagnosis does not support unchanged activations after math-only backend restriction")
    return {"metric": "Maximum layer relativeL2 over shared selected prefix (prompt-final plus shared completion)",
            "variable_shape_prefix_maxima": peaks, "B_future_C_common_shape_D_repeat_bitwise_equal": True,
            "F_forced_math_same_shape_matches_A_bitwise": True,
            "interpretation": "Observed variation depends on sequence shape and numerical precision; math-only restriction alone did not remove it."}


def verify_phase(reference, expected_phase, parent):
    """Check exact independent proof linkage and terminal metadata, no raw scan."""
    manifest_path = checked(reference["manifest_path"], reference["manifest_sha256"])
    m = json.loads(manifest_path.read_text())
    require(m["phase"] == expected_phase and m["host"] == "gpu-04" and
            m["scientific"]["master_plan_sha256"] == PARENT_SHA256 and
            m["scientific"]["input_prepared_sha256"] == parent["inputs"]["prepared_records_sha256"],
            "phase evidence belongs to different science/dataset/host")
    proof_path = checked(reference["verification_receipt_path"], reference["verification_receipt_sha256"])
    proof = json.loads(proof_path.read_text())
    require(proof["status"] == "verified" and proof["manifest_sha256"] == reference["manifest_sha256"] and
            proof["run_token"] == m["run_token"] and proof["gpu_release_verified"] is True and
            proof["artifact_manifest_sha256"] == reference["artifact_manifest_sha256"],
            "independent receipt lacks exact successful phase/artifact identity")
    root = Path(m["output"])
    artifacts_path = checked(root / "artifact_manifest.json", reference["artifact_manifest_sha256"])
    artifacts = json.loads(artifacts_path.read_text())
    require(artifacts["algorithm"] == "sha256" and proof["artifact_files"] == len(artifacts["files"]),
            "verified artifact count changed")
    copied_manifest = artifact_file(root, artifacts, "reviewed_manifest.json")
    require(sha(copied_manifest) == reference["manifest_sha256"], "phase output manifest is not the reviewed input")
    summary_path = artifact_file(root, artifacts, "campaign_summary.json")
    summary = json.loads(summary_path.read_text())
    release_path = artifact_file(root, artifacts, "gpu_release.json")
    release = json.loads(release_path.read_text())
    require(summary["status"] == "succeeded" and summary["run_token"] == m["run_token"] and
            summary["manifest_sha256"] == reference["manifest_sha256"] and summary["gpu_release_verified"] is True and
            summary["worker_exit_codes"] == [0] * len(m["workers"]) and release["verified"] is True and
            release["gpu_ids"] == m["gpu_ids"] and not (root / "FAILURE.json").exists(), "phase terminal success/release changed")
    exit_path = Path(m["stage"]) / "control/supervisor_exit.json"
    exit_receipt = json.loads(exit_path.read_text())
    require(exit_receipt["manifest_sha256"] == reference["manifest_sha256"] and exit_receipt["run_token"] == m["run_token"] and
            exit_receipt["service_result"] == "success" and exit_receipt["exit_code_kind"] == "exited" and
            str(exit_receipt["exit_status"]) == "0" and exit_receipt["failure_present"] is False,
            "actual systemd exit receipt is not successful")
    extra_files = {}
    observations = None
    for worker in m["workers"]:
        prefix = "workers/" + worker["name"] + "/"
        if expected_phase == "prefix_audit":
            name = prefix + "prefix_audit_verdict.json"
            path = artifact_file(root, artifacts, name)
            verdict = json.loads(path.read_text())
            require(verdict["passed"] is True and not verdict["failure_reasons"], "controlled numerical diagnosis failed")
            measurement_path = artifact_file(root, artifacts, prefix + "prefix_audit_all_measurements.json")
            observations = diagnostic_summary(json.loads(measurement_path.read_text()))
            extra_files[worker["name"] + "_numeric_measurements.json"] = measurement_path
        else:
            name = prefix + "fixed_cache_report.json"
            path = artifact_file(root, artifacts, name)
            report = json.loads(path.read_text())
            require(report["status"] == "succeeded" and report["activation_cache_profile"]["padded_sequence_length"] == 2176 and
                    report["activation_cache_profile"]["padding_attention_mask"] == 0 and
                    report["activation_cache_profile"]["attention_backend"] == "torch_sdpa_MATH_only" and
                    report["all_identical_prefixes_bitwise_equal"] is True and report["all_native_readbacks_bitwise_equal"] is True and
                    report["fp32_delta_readback_exact"] is True and report["raw_activations_retained"] is True,
                    "masked fixed-cache qualification did not pass exactly")
            for kind in ("h0", "h60"):
                require(all(report["qualifications"][kind][key]["bitwise_equal"] is True for key in ("repeat", "future_causality")),
                        "masked fixed-shape repeat/future perturbation failed")
        extra_files[worker["name"] + "_numerical_verdict.json"] = path
    record = {"phase": expected_phase, "run_token": m["run_token"], "output": str(root),
              "reviewed_manifest_sha256": reference["manifest_sha256"],
              "artifact_manifest_sha256": reference["artifact_manifest_sha256"],
              "independent_verification_receipt_sha256": reference["verification_receipt_sha256"],
              "actual_systemd_exit_receipt_sha256": sha(exit_path), "status": "independently_verified"}
    if observations is not None:
        record["observations"] = observations
    files = {"reviewed_manifest.json": manifest_path, "artifact_manifest.json": artifacts_path,
             "independent_verification.json": proof_path, "campaign_summary.json": summary_path,
             "gpu_release.json": release_path, "supervisor_exit.json": exit_path, **extra_files}
    return record, files


def amend(parent, evidence, parent_sha, source_sha):
    require(parent_sha == PARENT_SHA256 and parent.get("plan_version", 1) == 1 and
            parent["purpose"] == "checkpoint60_direction_discovery_and_causal_ablation" and
            parent["host"] == "gpu-04" and parent["no_training"] is True, "wrong parent plan")
    require(evidence["superseded_candidates"]["artifact_manifest_sha256"] == SUPERSEDED_CANDIDATE_SHA256,
            "unexpected superseded candidate package")
    updated = copy.deepcopy(parent)
    updated["plan_version"] = 2
    updated["parent_plan_sha256"] = parent_sha
    updated["status"] = "frozen_before_repaired_activation_fitting_or_causal_selection"
    updated["inputs"]["raw_artifact_manifest_sha256"] = evidence["core_cache"]["artifact_manifest_sha256"]
    updated["inputs"]["raw_package"] = evidence["core_cache"]["output"]
    updated["inputs"]["fixed_cache_review_manifest_sha256"] = evidence["core_cache"]["reviewed_manifest_sha256"]
    updated["amendment"] = {"reason": REPAIR_REASON, "parent_raw_artifact_manifest_sha256": parent["inputs"]["raw_artifact_manifest_sha256"],
                            "evidence": evidence, "preparation_source_sha256": source_sha,
                            "superseded_candidates_status": "diagnostic_only_do_not_promote",
                            "original_raw_and_candidate_artifacts_retained": True,
                            "behavioral_estimator_splits_and_seeds_changed": False,
                            "activation_numerical_policy_changed": True,
                            "preserved_science_sha256": {key: hashlib.sha256(canonical(parent[key]).encode()).hexdigest()
                                for key in ("dataset", "fit", "budget", "sampling", "generation", "sweep", "model")}}
    verify_unchanged(parent, updated)
    return updated


def verify_unchanged(parent, updated):
    restored = copy.deepcopy(updated)
    for key in ("plan_version", "parent_plan_sha256", "amendment"):
        restored.pop(key)
    restored["status"] = parent["status"]
    restored["inputs"].pop("raw_package")
    restored["inputs"].pop("fixed_cache_review_manifest_sha256")
    restored["inputs"]["raw_artifact_manifest_sha256"] = parent["inputs"]["raw_artifact_manifest_sha256"]
    require(restored == parent, "amendment changed frozen science or other original plan fields")


def create(parent_plan, parent_plan_sha256, evidence_path, evidence_sha256, output):
    """No output exists until all independently verified evidence has passed."""
    require(parent_plan_sha256 == PARENT_SHA256, "parent plan digest is not frozen v1")
    parent = json_checked(parent_plan, parent_plan_sha256)
    submitted = json_checked(evidence_path, evidence_sha256)
    require(set(submitted) == {*PHASES, "superseded_candidates"}, "amendment evidence roles changed")
    output = Path(output)
    require(not output.exists(), "amendment output must be a new immutable directory")
    evidence, copy_files = {}, {}
    for role, phase in PHASES.items():
        evidence[role], files = verify_phase(submitted[role], phase, parent)
        copy_files.update({"input/" + role + "/" + name: path for name, path in files.items()})
    require(len({evidence[k]["reviewed_manifest_sha256"] for k in PHASES}) == 3, "diagnosis/qualification/core must be independent phases")
    require(evidence["core_cache"]["artifact_manifest_sha256"] != parent["inputs"]["raw_artifact_manifest_sha256"], "amendment did not replace the original cache")
    from cache_package import inspect
    inspection = inspect(evidence["core_cache"]["output"])
    require(inspection["status"] == "verified" and inspection["records"] == 561 and inspection["problems"] == 187 and
            inspection["index_joined_to_artifact_manifest"] is True and
            inspection["manifest_sha256"] == evidence["core_cache"]["reviewed_manifest_sha256"], "repaired core package semantics failed")
    evidence["core_cache"]["semantic_inspection"] = inspection
    old = submitted["superseded_candidates"]
    require(old["artifact_manifest_sha256"] == SUPERSEDED_CANDIDATE_SHA256, "unreviewed old candidate artifact")
    old_manifest = checked(Path(old["root"]) / "artifact_manifest.json", old["artifact_manifest_sha256"])
    evidence["superseded_candidates"] = {"root": str(Path(old["root"])), "artifact_manifest_sha256": old["artifact_manifest_sha256"],
                                        "status": "diagnostic_only_do_not_promote"}
    copy_files.update({"input/parent_experiment_plan.json": checked(parent_plan, parent_plan_sha256),
                       "input/submitted_evidence.json": checked(evidence_path, evidence_sha256),
                       "input/superseded_candidate_artifact_manifest.json": old_manifest})
    value = amend(parent, evidence, parent_plan_sha256, sha(__file__))
    output.mkdir(parents=True, exist_ok=False)
    for relative, source in copy_files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            for block in iter(lambda: src.read(8 << 20), b""):
                dst.write(block)
        require(sha(target) == sha(source), "evidence changed during copying")
        target.chmod(0o444)
    plan_path = output / "experiment_plan.json"
    plan_path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plan_path.chmod(0o444)
    files = {str(path.relative_to(output)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
             for path in output.rglob("*") if path.is_file()}
    manifest = output / "artifact_manifest.json"
    manifest.write_text(canonical({"algorithm": "sha256", "files": files}) + "\n")
    manifest.chmod(0o444)
    verify_unchanged(parent, json.loads(plan_path.read_text()))
    return {"status": "prepared", "plan_version": 2, "plan_path": str(plan_path), "plan_sha256": sha(plan_path),
            "artifact_manifest_sha256": sha(manifest), "launch_performed": False,
            "behavioral_science_preserved": True, "activation_numerical_policy_changed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-plan", "parent-plan-sha256", "evidence-path", "evidence-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    print(canonical(create(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
