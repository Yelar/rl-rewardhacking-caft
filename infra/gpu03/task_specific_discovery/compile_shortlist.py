"""Compile verified offline candidates; never launch a causal or training run.

The historical reference is the already-exported target_pc4 tensor, not a guessed
column of the calibration file. Its file and raw tensor byte hashes are pinned
by the prior Ada arm manifest. Directions are copied exactly, never renormalized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from infra.gpu03.task_specific_discovery import fitting, fit_worker as io, resource_guard as guard
from infra.gpu03.task_specific_discovery.dispatch_phase import ROOT, PYTHON

PC4_PATH = "/scratch/yelarys.yertaiuly/codex_runs/codex-ada8-matched-20260916-1540-inputs/calibration_vectors.safetensors"
PC4_FILE = {"sha256": "1dacdee17fd2eab2cf6f26bd37812b06ec62346d07f51b3f690c0e1ab5a52498", "size_bytes": 1361664}
PC4_TENSOR_SHA = "b3c7f0380b0ef1c8c7ba824fc5764915b15ab1c5a5fb61ed9396105eca27a98b"
PC4_KEY = "target_pc4"
PC4_PROVENANCE = "artifacts/ada-baseline-alignment-20260916-v2/pc4.baseline_target.json:target_config.actor_rollout_ref.model.caft.q"
require = fitting.require


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 << 20,
            "Missing/linked/oversized JSON input")
    return json.loads(path.read_text())


def tensor_bytes_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_direction(value):
    require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.shape == (2560, 1) and
            np.isfinite(value).all(), "Direction must be finite FP32[2560,1]")
    norm2 = float(np.square(value.astype(np.float64)).sum())
    require(abs(norm2 - 1) <= 1e-5, "Saved direction is not unit rank one; do not silently renormalize")
    return norm2


def validate_cell_report(report, expected_layer, expected_representation, rows, cfg):
    require(report["schema"] == "task_specific_fitting_cell_v1" and report["region"] == "evaluator" and
            report["layer"] == expected_layer and report["representation"] == expected_representation and
            report["config"] == cfg and report["input_records"] == len(rows) and report["hidden_size"] == 2560,
            "Cell identity/config/input count changed")
    ids = {r["record_id"] for r in rows}
    matched, unmatched = report["matched_record_ids"], report["unmatched_record_ids"]
    require(len(matched) == len(set(matched)) == report["matched_records"] and
            len(unmatched) == len(set(unmatched)) and not set(matched).intersection(unmatched) and
            set(matched).union(unmatched) == ids, "Cell matching does not cover exact input IDs")
    policy = report["statistical_policy"]
    expected_policy = {"outer_folds": 5, "inner_folds": 5, "ridge_grid": [0.1, 1.0, 10.0],
                       "minimum_paired_problems_per_stratum": 30, "minimum_half_cosine": .5,
                       "minimum_orthogonal_residual_ratio": .1,
                       "ranking_metric": "equal_stratum_equal_problem_heldout_paired_auc"}
    require(policy == expected_policy, "Prospective statistical gates changed")
    expected_pairs = fitting.matched_problems(rows)
    for stratum in (False, True):
        support = report["coverage"][str(stratum).lower()]
        require(support["problem_ids"] == expected_pairs[stratum] and
                support["paired_problems"] == len(expected_pairs[stratum]), "Paired support report differs from inputs")
    expected = {(family, kind) for family in fitting.FAMILIES for kind in fitting.KINDS}
    candidates = report["candidates"]
    require(len(candidates) == 6 and {(c["family"], c["activation_kind"]) for c in candidates} == expected,
            "Cell must retain all six supported/unsupported primary candidates")
    for candidate in candidates:
        key = f"L{expected_layer:02d}.evaluator.{expected_representation}.{candidate['activation_kind']}.{candidate['family']}"
        require(candidate["candidate_id"] == candidate["tensor_prefix"] == key and
                candidate["layer"] == expected_layer and candidate["representation"] == expected_representation and
                type(candidate["eligible"]) is bool and candidate["status"] in ("fitted", "unsupported"),
                "Candidate identity/status differs")
        if candidate["eligible"]:
            require(candidate["status"] == "fitted" and candidate["rejection_reasons"] == [] and
                    candidate["oof"]["complete"] is True and
                    np.isfinite(candidate["oof"]["balanced_paired_auc"]) and
                    0 <= candidate["oof"]["balanced_paired_auc"] <= 1 and
                    candidate["half_aligned_cosine"] >= .5 and
                    all(len(v) >= 30 for v in expected_pairs.values()), "Candidate eligibility violates frozen gates")
            if candidate["family"] == "correctness_orthogonalized_mean":
                orth = candidate["fit"]["orthogonalization"]
                require(orth["residual_ratio"] >= .1 and orth["original_norm"] > 0 and orth["residual_norm"] > 0,
                        "Eligible orthogonal candidate fails residual gate")
    return report


def compact_report(report, report_ref, tensor_ref):
    """Keep full per-candidate selection evidence; fold internals remain in source JSON."""
    keys = ("candidate_id", "family", "layer", "representation", "activation_kind", "tensor_prefix", "eligible",
            "status", "rejection_reasons", "support", "fit", "oof", "oof_h60_transfer",
            "half_signed_cosine", "half_aligned_cosine", "regularization")
    candidates = [{**{k: c[k] for k in keys if k in c}, "source_report": report_ref,
                   "source_tensors": tensor_ref} for c in report["candidates"]]
    return {"layer": report["layer"], "representation": report["representation"], "candidates": candidates,
            "coverage": report["coverage"], "diagnostic_references": report["diagnostic_references"],
            "source_report": report_ref, "source_tensors": tensor_ref}


def inspect_fit(manifest_path, digest):
    root, manifest_path = Path(ROOT), Path(manifest_path)
    require(io.sha256(manifest_path) == digest, "Fitting manifest hash changed")
    m = read_json(manifest_path)
    guard.validate_manifest(m)
    require(m["run_token"] == root.name and m["source_root"] == str(root / "source-discovery-v2") and
            m["gpu_uuids"] == {} and len(m["workers"]) == 8 and m["expected_cells"] == 72,
            "Unexpected fitting scope/source")
    require(m["science"]["generation"] is False and m["science"]["evaluation"] is False and
            m["science"]["training"] is False and m["science"]["model_execution"] is False,
            "Offline fitting science scope differs")
    evidence, identities = {}, {}

    def bind(path, expected=None):
        path = Path(path)
        expected = io.binding(path) if expected is None else expected
        identities[str(path)] = io.verify_file(path, expected)
        evidence[str(path)] = expected
        return {"path": str(path), **expected}

    bind(manifest_path, {"sha256": digest, "size_bytes": manifest_path.stat().st_size})
    for name, value in m["bound_files"].items():
        bind(name, value)
    result_path = Path(m["runtime"]) / "SUPERVISOR_RESULT.json"
    bind(result_path)
    result = read_json(result_path)
    require(result["run_token"] == root.name and result["manifest_sha256"] == digest and
            result["status"] == "workers_exited_zero" and result["owned_processes_released"] is True and
            result["cpu_only"] is True and result["gpu_release_verified"] is None and
            set(result["exit_codes"]) == {w["name"] for w in m["workers"]} and
            all(type(c) is int and c == 0 for c in result["exit_codes"].values()),
            "Fitting supervisor release/exit proof is incomplete")
    reports, centers, seen_shards = [], {}, set()
    prepared_path, prepared_rows = None, None
    for worker in m["workers"]:
        argv = worker["argv"]
        require(argv[:4] == [PYTHON, "-B", "-m", "infra.gpu03.task_specific_discovery.fit_worker"] and
                len(argv) == 8 and argv[4] == "--task" and argv[6] == "--sha256", "Unexpected fitting worker command")
        task_path = Path(argv[5])
        require(str(task_path) in m["bound_files"] and m["bound_files"][str(task_path)]["sha256"] == argv[7],
                "Fitting task is not hash-bound")
        task = read_json(task_path)
        index = task["shard_index"]
        require(task["schema"] == "task_specific_cpu_fitting_v1" and task["shard_count"] == 8 and
                type(index) is int and 0 <= index < 8 and index not in seen_shards, "Duplicate/wrong fitting shard")
        seen_shards.add(index)
        require(task["prepared_rows"] in m["bound_files"], "Global fitting rows are not bound")
        if prepared_path is None:
            prepared_path = task["prepared_rows"]
            with Path(prepared_path).open() as stream:
                prepared_rows = [json.loads(line) for line in stream]
            require(len(prepared_rows) == m["expected_records"] and len(prepared_rows) <= 256 and
                    len({r["record_id"] for r in prepared_rows}) == len(prepared_rows) and
                    all(r["problem_split"] == "direction_fit" for r in prepared_rows), "Fitting input records differ")
        require(task["prepared_rows"] == prepared_path, "CPU workers used different fitting rows")
        cfg = fitting.configuration(task.get("config"))
        out = Path(task["output"])
        require(out.is_relative_to(root) and ".." not in out.parts, "Fitting output outside run root")
        complete_path = out / "COMPLETE.json"
        bind(complete_path)
        complete = read_json(complete_path)
        layers = list(range(index, 36, 8))
        expected_cells = [{"layer": layer, "representation": rep} for layer in layers for rep in fitting.REPRESENTATIONS]
        require(complete["schema"] == "task_specific_cpu_fitting_complete_v1" and complete["status"] == "succeeded" and
                complete["task_sha256"] == argv[7] and complete["shard_index"] == index and complete["shard_count"] == 8 and
                complete["layers"] == layers and complete["completed_cells"] == expected_cells and
                complete["records"] == len(prepared_rows) and complete["verified_raw_and_delta_files"] == 3 * len(prepared_rows) and
                complete["causal_validation_claimed"] is False and complete["model_execution"] is False,
                "Fitting shard completion/coverage differs")
        expected_names = {f"L{layer:02d}.{part}.{ext}" for layer in layers
                          for part in ("response_center", "evaluator.body_predictor", "evaluator.transition")
                          for ext in ("json", "safetensors")}
        require(set(complete["artifacts"]) == expected_names, "Layer artifact inventory is partial or unexpected")
        refs = {name: bind(out / name, value) for name, value in complete["artifacts"].items()}
        for layer in layers:
            for part in ("response_center", "evaluator.body_predictor", "evaluator.transition"):
                stem = f"L{layer:02d}.{part}"
                report = read_json(out / (stem + ".json"))
                require(report["tensor_file"] == {"path": stem + ".safetensors", **complete["artifacts"][stem + ".safetensors"]},
                        "Cell tensor reference differs from sealed artifact inventory")
                if part == "response_center":
                    require(layer not in centers and report["schema"] == "fitting_only_response_predictor_center_v1" and
                            report["layer"] == layer and report["activation_kind"] == "h60" and
                            report["policy"] == "equal_correctness_stratum_then_problem_then_response_then_predictor" and
                            report["positions"] == "last_prompt_plus_completion_excluding_final_token" and
                            report["clean_definition"] == "known_benign_modification_and_not_strict_reward_hack" and
                            report["status"] in ("fitted", "unsupported"), "Response center identity/estimand differs")
                    centers[layer] = {"report": report, "report_ref": refs[stem + ".json"],
                                      "tensor_ref": refs[stem + ".safetensors"]}
                else:
                    rep = part.split(".")[1]
                    validate_cell_report(report, layer, rep, prepared_rows, cfg)
                    reports.append(compact_report(report, refs[stem + ".json"], refs[stem + ".safetensors"]))
    require(seen_shards == set(range(8)) and set(centers) == set(range(36)) and
            len(reports) == 72 and len({(r["layer"], r["representation"]) for r in reports}) == 72,
            "Incomplete all-layer fitting union")
    require(all(io.stat_identity(path) == identity for path, identity in identities.items()),
            "Fitting evidence changed during independent readback")
    return sorted(reports, key=lambda r: (r["layer"], r["representation"])), centers, evidence, identities


def read_tensor(reference, key):
    from safetensors import safe_open
    path = reference["path"]
    before = io.verify_file(path, reference)
    with safe_open(path, framework="np") as source:
        require(key in source.keys(), "Selected tensor key missing: " + key)
        value = source.get_tensor(key).copy()
    require(io.stat_identity(path) == before, "Selected tensor source changed during read")
    return value


def assemble(reports, centers, tensor_reader=read_tensor):
    """Pure nomination/export assembly; caller has already verified closed inputs."""
    historical = {"family": "historical_pc4", "layer": 21, "pc_index": 4,
                  "candidate_id": "historical.L21.PC4", "source_condition_id": "stage9-pc4-alpha100",
                  "source_tensors": {"path": PC4_PATH, **PC4_FILE}, "source_tensor_key": PC4_KEY,
                  "source_tensor_sha256": PC4_TENSOR_SHA, "provenance_manifest": PC4_PROVENANCE}
    selection = fitting.select_shortlist(reports, historical)
    pc4 = tensor_reader(historical["source_tensors"], PC4_KEY)
    validate_direction(pc4)
    require(tensor_bytes_sha(pc4) == PC4_TENSOR_SHA, "Historical PC4 tensor bytes changed")
    tensors = {"axis__historical_pc4": pc4}
    axes = [{"axis_id": "historical_pc4", "candidate_id": historical["candidate_id"], "layer": 21,
             "tensor_key": "axis__historical_pc4", "tensor_sha256": tensor_bytes_sha(pc4),
             "family": "historical_pc4", "source": historical}]
    for candidate in selection["new_axes"]:
        q = tensor_reader(candidate["source_tensors"], candidate["tensor_prefix"] + ".direction")
        validate_direction(q)
        key = "axis__" + candidate["family"]
        tensors[key] = q
        axes.append({"axis_id": candidate["family"], "candidate_id": candidate["candidate_id"], "layer": candidate["layer"],
                     "tensor_key": key, "tensor_sha256": tensor_bytes_sha(q), "family": candidate["family"],
                     "activation_kind": candidate["activation_kind"], "source": candidate})
    available_centers, gaps = {}, []
    for layer in sorted({a["layer"] for a in axes}):
        center = centers[layer]
        if center["report"]["status"] != "fitted":
            gaps.append({"layer": layer, "kind": "center_unavailable", "reason": center["report"].get("reason"),
                         "source_report": center["report_ref"]})
            continue
        mu = tensor_reader(center["tensor_ref"], "h60_response_center")
        require(mu.dtype == np.float32 and mu.shape == (2560,) and np.isfinite(mu).all(), "Invalid M60 response center")
        key = f"center__L{layer:02d}"
        tensors[key] = mu
        available_centers[layer] = {"layer": layer, "tensor_key": key, "tensor_sha256": tensor_bytes_sha(mu),
                                    "source_report": center["report_ref"], "source_tensors": center["tensor_ref"]}
    configurations = []
    centered_ids = {"historical_pc4"}
    best = selection["best_new_candidate_for_centered_comparison"]
    centered_ids.update(a["axis_id"] for a in axes if a["candidate_id"] == best)
    for axis in axes:
        base = {"axis_id": axis["axis_id"], "candidate_id": axis["candidate_id"], "layer": axis["layer"],
                "vector_tensor_key": axis["tensor_key"], "strengths": [.25, .5, 1.0], "scope": "response_predictors"}
        configurations.append({**base, "configuration_id": axis["axis_id"] + ".ordinary", "operator": "ordinary"})
        if axis["axis_id"] in centered_ids and axis["layer"] in available_centers:
            configurations.append({**base, "configuration_id": axis["axis_id"] + ".centered", "operator": "centered",
                                   "center_tensor_key": available_centers[axis["layer"]]["tensor_key"]})
    require(len(axes) <= 4 and len(configurations) <= 6, "Prospective axis/operator cap exceeded")
    return {"schema": "task_specific_verified_offline_shortlist_v1", "status": "offline_shortlist_compiled",
            "selection": selection, "axes": axes, "centers": list(available_centers.values()),
            "operator_configurations": configurations, "gaps": gaps, "candidate_count": sum(len(r["candidates"]) for r in reports),
            "completed_cells": 72, "completed_layers": 36, "causal_validation_claimed": False,
            "random_controls_prepared": False, "ready_for_generation_launch": False,
            "direction_byte_policy": "exact_saved_FP32_no_sign_change_no_renormalization"}, tensors


def compile_shortlist(manifest_path, digest, output):
    output, root = Path(output), Path(ROOT)
    require(output.parent == root and not output.exists(), "Shortlist needs a fresh direct child of the run root")
    reports, centers, evidence, identities = inspect_fit(manifest_path, digest)
    evidence[PC4_PATH] = PC4_FILE
    identities[PC4_PATH] = io.verify_file(PC4_PATH, PC4_FILE)
    shortlist, tensors = assemble(reports, centers)
    require(all(io.stat_identity(path) == identity for path, identity in identities.items()),
            "Bound fitting/PC4 evidence changed while compiling shortlist")
    output.mkdir(mode=0o700)
    table = {"schema": "task_specific_complete_candidate_table_v1", "cells": reports,
             "candidate_count": 432, "cells_completed": 72, "all_input_candidates_retained": True}
    io.write_json(output / "CANDIDATE_TABLE.json", table)
    tensor_path = output / "shortlist.safetensors"
    io.write_tensors(tensor_path, tensors, {"schema": "task_specific_shortlist_axes_centers_v1", "fit_manifest_sha256": digest})
    shortlist["tensor_file"] = {"path": tensor_path.name, **io.binding(tensor_path)}
    shortlist["fit_manifest_sha256"] = digest
    io.write_json(output / "SHORTLIST.json", shortlist)
    io.write_json(output / "INPUT_VERIFICATION.json", {"status": "verified", "files": evidence,
                  "compiler_source": {"path": str(Path(__file__).resolve()), **io.binding(__file__)},
                  "model_or_generated_code_executed": False})
    artifacts = {p.name: io.binding(p) for p in sorted(output.iterdir()) if p.is_file()}
    completion = {"schema": "task_specific_shortlist_complete_v1", "status": "succeeded", "artifacts": artifacts,
                  "fit_manifest_sha256": digest, "axes": len(shortlist["axes"]),
                  "operator_configurations": len(shortlist["operator_configurations"]),
                  "cells": 72, "candidates": 432, "causal_validation_claimed": False, "launched": False}
    io.write_json(output / "COMPLETE.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(socket.gethostname().split(".")[0] == "gpu-03" and os.getuid() == 3357 and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Shortlist compiler requires gpu-03 CPU-only owner context")
    result = compile_shortlist(args.fit_manifest, args.sha256, args.output)
    print(json.dumps({k: result[k] for k in ("status", "axes", "operator_configurations", "cells", "candidates", "launched")}))


if __name__ == "__main__":
    main()
