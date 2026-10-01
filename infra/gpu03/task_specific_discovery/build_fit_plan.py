"""Prepare a CPU fitting phase near closed extraction data; never launch it.

Run this file on gpu-03 after the bound extraction supervisor exits. The helper
reads/hashes existing files, writes fresh immutable receipts/tasks, and emits a
resource_guard manifest. It does not contact a host, load a model, allocate CUDA,
execute generated code, or dispatch a service. Production paths remain tied to
the existing run token and source-discovery-v2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import time

from infra.gpu03.task_specific_discovery import fit_worker as fit
from infra.gpu03.task_specific_discovery import resource_guard as guard
from infra.gpu03.task_specific_discovery.dispatch_phase import ROOT, PYTHON

SECONDS = 7200
GIB = 1024**3
MAX_JSON = 16 << 20
MAX_LINE = 8 << 20
require = fit.require


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON,
            "Missing, linked or oversized JSON evidence: " + str(path))
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def read_rows(path, limit=256):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing/linked JSONL evidence")
    result = []
    with path.open("rb") as stream:
        while True:
            data = stream.readline(MAX_LINE + 1)
            if not data:
                break
            require(len(data) <= MAX_LINE and data.endswith(b"\n") and data.strip(), "Invalid bounded JSONL")
            value = json.loads(data)
            require(isinstance(value, dict), "Nonobject JSONL row")
            result.append(value)
            require(len(result) <= limit, "JSONL exceeds declared record bound")
    return result


def contained(path, root):
    path, root = Path(path), Path(root)
    require(path.is_absolute() and ".." not in path.parts and path.is_relative_to(root) and path != root,
            "Path outside the existing run root")
    return path


def primitive_completion_check(result, count, problems):
    require(result["status"] == "succeeded" and result["records"] == count and
            result["problems"] == problems, "Extraction worker not successfully complete")
    for field in ("all_identical_prefixes_bitwise_equal", "all_native_readbacks_bitwise_equal",
                  "fp32_delta_readback_exact", "raw_activations_retained", "differences_computed"):
        require(result[field] is True, "Failed extraction audit: " + field)
    for kind in ("h0", "h60"):
        q = result["qualifications"][kind]
        require(q["repeat"]["bitwise_equal"] is True and q["future_causality"]["bitwise_equal"] is True,
                "Repeat/future-token qualification failed")
    profile = result["activation_cache_profile"]
    require(profile["padded_sequence_length"] == 2176 and profile["pad_token_id"] == 151643 and
            profile["padding_side"] == "right" and profile["padding_attention_mask"] == 0 and
            profile["attention_backend"] == "torch_sdpa_MATH_only" and
            profile["original_tokens_preserved"] is True and profile["prompt_final_saved"] is True,
            "Extraction numerical/input profile changed")


def inspect_extraction(manifest_path, digest):
    """Independently reconcile closed receipts and exact IDs before cache hashing."""
    root = Path(ROOT)
    manifest_path = Path(manifest_path)
    require(fit.sha256(manifest_path) == digest, "Extraction manifest hash changed")
    manifest = read_json(manifest_path)
    guard.validate_manifest(manifest)
    require(manifest["run_token"] == root.name and manifest["disk_path"] == str(root) and
            manifest["source_root"] == str(root / "source-discovery-v2"), "Extraction run/source scope differs")
    require(len(manifest["workers"]) == len(manifest["gpu_uuids"]) == 8, "Exactly eight completed extraction shards required")
    source = Path(manifest["source_root"])
    evidence, identities = {}, {}

    def bind(path, expected=None):
        path = contained(path, root) if Path(path).is_relative_to(root) else Path(path)
        value = fit.binding(path) if expected is None else expected
        identities[str(path)] = fit.verify_file(path, value)
        evidence[str(path)] = value
        return path

    bind(manifest_path, {"sha256": digest, "size_bytes": manifest_path.stat().st_size})
    source_bindings = {p: b for p, b in manifest["bound_files"].items() if Path(p).is_relative_to(source)}
    needed = ["fit_worker.py", "fitting.py", "resource_guard.py", "phase_entry.py"]
    require(all(str(source / "infra/gpu03/task_specific_discovery" / name) in source_bindings for name in needed),
            "Fitting source missing from frozen extraction source bindings")
    for path, value in source_bindings.items():
        bind(path, value)
    global_path = contained(manifest["prepared_global_rows"], root)
    support = global_path.parent
    artifact_path = support / "ARTIFACT_MANIFEST.json"
    bind(artifact_path, {"sha256": manifest["source_artifact_manifest_sha256"],
                         "size_bytes": artifact_path.stat().st_size})
    artifact = read_json(artifact_path)
    require(artifact["status"] == "support_ready", "Support package did not pass prospective support gate")
    summary_path = support / "SUMMARY.json"
    for path in (global_path, summary_path):
        bind(path, artifact["files"][path.name])
    summary = read_json(summary_path)
    require(fit.sha256(summary_path) == manifest["support_summary_sha256"] and
            summary["status"] == "support_ready" and min(summary["eligible_paired_problems"].values()) >= 30,
            "Support summary/source eligibility changed")
    rows = read_rows(global_path)
    require(len(rows) == manifest["expected_records"] == summary["selected_records"] and rows and
            len({r["record_id"] for r in rows}) == len(rows) and
            len({r["record_index"] for r in rows}) == len(rows) and
            rows == sorted(rows, key=lambda r: r["record_index"]) and
            all(r["problem_split"] == "direction_fit" for r in rows), "Global prepared row coverage/order/split mismatch")
    require(all(type(r.get("ground_truth_correctness")) is bool and
                type(r.get("is_test_modification_harmful")) is bool for r in rows) and
            all(len(v) >= 30 for v in fit.fitting.matched_problems(rows).values()),
            "Selected rows do not independently meet paired support threshold")
    by_id = {r["record_id"]: r for r in rows}
    runtime = Path(manifest["runtime"])
    require(runtime.parent == root.parent and runtime.name.startswith(root.name + "-"), "Unrelated supervisor runtime")
    supervisor_path = bind(runtime / "SUPERVISOR_RESULT.json")
    result = read_json(supervisor_path)
    names = {w["name"] for w in manifest["workers"]}
    require(result["run_token"] == root.name and result["manifest_sha256"] == digest and
            result["status"] == "workers_exited_zero" and result["owned_processes_released"] is True and
            result["gpu_release_verified"] is True and set(result["exit_codes"]) == names and
            all(type(code) is int and code == 0 for code in result["exit_codes"].values()),
            "Supervisor has not positively verified all worker exits and GPU release")
    for gpu, uuid in manifest["gpu_uuids"].items():
        device = result["gpus"][gpu]
        require(device["uuid"] == uuid and device["pids"] == [] and device["utilization"] <= 2 and
                device["total_mib"] - device["free_mib"] <= 1024, "Supervisor release snapshot differs")
    entries, worker_indexes, problem_shards = {}, [], {}
    for worker in manifest["workers"]:
        argv = worker["argv"]
        require(argv[:4] == [PYTHON, "-B", "-m", "infra.gpu03.task_specific_discovery.extract_worker"] and
                len(argv) == 8 and argv[4] == "--task" and argv[6] == "--sha256", "Unexpected extraction command")
        task_path = contained(argv[5], root)
        require(manifest["bound_files"][str(task_path)]["sha256"] == argv[7], "Extraction task hash binding differs")
        bind(task_path, manifest["bound_files"][str(task_path)])
        task = read_json(task_path)
        require(task["schema"] == "task_specific_fixed2176_extraction_v1" and
                task["padded_sequence_length"] == 2176 and task["pad_token_id"] == 151643 and
                task["generation"] is False and task["evaluation"] is False and task["training"] is False,
                "Unexpected extraction task science")
        shard_path = contained(task["prepared_rows"], root)
        bind(shard_path, task["bound_files"][str(shard_path)])
        shard_rows = read_rows(shard_path)
        require(shard_rows and len({r["record_id"] for r in shard_rows}) == len(shard_rows), "Empty/duplicate extraction shard")
        for row in shard_rows:
            require(row == by_id.get(row["record_id"]), "Extraction shard differs from immutable global input")
            pid = row["problem_id_key"]
            require(pid not in problem_shards or problem_shards[pid] == worker["name"], "Problem crossed extraction shards")
            problem_shards[pid] = worker["name"]
        out = contained(task["output"], root)
        started = read_json(bind(out / "STARTED.json"))
        require(started["task_sha256"] == argv[7] and started["records"] == len(shard_rows) and
                started["gpu_uuid"] == manifest["gpu_uuids"][str(worker["gpu_id"])] and
                all(started[k] is False for k in ("generation", "evaluation", "training")), "Wrong extraction start identity")
        completed = read_json(bind(out / "COMPLETE.json"))
        require(completed == read_json(bind(out / "fixed_cache_report.json")), "Worker completion and numerical report differ")
        primitive_completion_check(completed, len(shard_rows), len({r["problem_id_key"] for r in shard_rows}))
        index_path = bind(out / "workerindex.jsonl")
        index = read_rows(index_path)
        require({e["record_id"] for e in index} == {r["record_id"] for r in shard_rows} and
                len(index) == len(shard_rows), "Extraction index incomplete or duplicate")
        for entry in index:
            rid = entry["record_id"]
            require(rid not in entries and entry["record_index"] == by_id[rid]["record_index"] and
                    entry["verified"] is True and set(entry["models"]) == {"h0", "h60"},
                    "Conflicting/unverified native cache index")
            for kind, part in {**entry["models"], "delta": entry["delta"]}.items():
                relative = Path(part["tensor_path"])
                require(not relative.is_absolute() and ".." not in relative.parts and
                        relative == Path(kind) / f"record_{entry['record_index']:06d}.safetensors",
                        "Native tensor path differs from fixed extraction naming")
                require(part["shape"] == [36, by_id[rid]["completion_token_count"], 2560] and
                        part["all_finite"] is True, "Cache index shape/finite audit differs")
                if kind == "delta":
                    require(part["dtype"] == "float32" and part["fp32_readback_exact"] is True and
                            part["float64_reference_rounded_to_fp32_exact"] is True, "Delta numerical audit failed")
                else:
                    require(part["dtype"] == "bfloat16" and part["native_readback_bitwise_equal"] is True and
                            part["prompt_final_shape"] == [36, 2560], "Native numerical audit failed")
                part["absolute_path"] = str(out / relative)
            entries[rid] = entry
        worker_indexes.append(str(index_path))
    require(set(entries) == set(by_id), "Extraction union does not exactly cover selected fitting records")
    return {"manifest": manifest, "rows": rows, "entries": entries, "worker_indexes": worker_indexes,
            "source_bindings": source_bindings, "evidence": evidence, "identities": identities}


def prepare(extraction_manifest, digest, phase, *, now=None):
    root = Path(ROOT)
    require(re.fullmatch(r"fitting-[a-z0-9-]+", phase or ""), "Invalid fresh fitting phase name")
    output = root / phase
    require(not output.exists(), "Existing fitting attempt preserved; refuse overwrite")
    inspected = inspect_extraction(extraction_manifest, digest)
    helper_path = str(Path(__file__).resolve())
    helper_binding = fit.binding(helper_path)
    output.mkdir(mode=0o700)
    verification = fit.build_cache_verification(inspected["entries"])
    # Producer receipts and selected inputs must remain unchanged throughout the full hash pass.
    require(all(fit.stat_identity(path) == identity for path, identity in inspected["identities"].items()),
            "Extraction evidence changed during independent verification")
    verify_path = output / "CACHE_VERIFICATION.json"
    fit.write_json(verify_path, verification)
    completion_path = output / "EXTRACT_COMPLETE.json"
    fit.write_json(completion_path, {"schema": "task_specific_extraction_independent_completion_v1",
        "status": "succeeded", "gpu_workers_released": True, "extraction_manifest_sha256": digest,
        "records": len(inspected["rows"]), "layers": 36, "native_cache_files": len(verification["files"]),
        "input_evidence": inspected["evidence"], "cache_verification": fit.binding(verify_path),
        "preparation_source": {"path": helper_path, **helper_binding},
        "numerical_audits": "Producer raw/FP32 readbacks and repeat/future checks verified; fitter rechecks every layer delta",
        "raw_activation_payloads_retained": True})
    source = str(root / "source-discovery-v2")
    worker_bound = {str(Path(extraction_manifest)): inspected["evidence"][str(Path(extraction_manifest))],
                    inspected["manifest"]["prepared_global_rows"]:
                        inspected["evidence"][inspected["manifest"]["prepared_global_rows"]],
                    **{p: inspected["evidence"][p] for p in inspected["worker_indexes"]},
                    str(verify_path): fit.binding(verify_path), str(completion_path): fit.binding(completion_path)}
    bound = {**inspected["source_bindings"], **worker_bound, helper_path: helper_binding}
    workers, tasks = [], []
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": source, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for index in range(8):
        task = {"schema": "task_specific_cpu_fitting_v1", "shard_index": index, "shard_count": 8,
                "deadline_seconds": SECONDS, "prepared_rows": inspected["manifest"]["prepared_global_rows"],
                "worker_indexes": inspected["worker_indexes"], "bound_files": worker_bound,
                "extraction_complete": str(completion_path), "cache_verification": str(verify_path),
                "output": str(output / f"cpu_{index}"), "config": {"max_working_bytes": 2 * GIB}}
        task_path = output / f"task_cpu_{index}.json"
        fit.write_json(task_path, task)
        value = fit.binding(task_path)
        bound[str(task_path)] = value
        workers.append({"name": f"cpu_{index}", "gpu_id": None, "cpu_ids": [48 + index], "cwd": source,
                        "env": dict(env), "argv": [PYTHON, "-B", "-m", "infra.gpu03.task_specific_discovery.fit_worker",
                        "--task", str(task_path), "--sha256", value["sha256"]]})
        tasks.append({"path": str(task_path), **value, "layers": list(range(index, 36, 8))})
    start = time.time() if now is None else now
    manifest = {"schema": guard.SCHEMA, "host": "gpu-03", "uid": 3357, "run_token": root.name,
                "source_root": source, "runtime": str(root) + "-" + phase + "-runtime", "disk_path": str(root),
                "deadline_epoch": start + SECONDS, "runtime_seconds": SECONDS + 120,
                "unit": root.name + "-" + phase, "entry_receipt": str(root / (phase + "-entry.json")),
                "kernel_limits": {"memory_max": 32 * GIB, "tasks_max": 128, "cpu_cores": 8},
                "gpu_uuids": {}, "workers": workers, "bound_files": bound,
                "science": {"models": ["M0", "M60"], "layers": 36, "source": "verified_native_fixed2176",
                    "representations": ["body_predictor", "transition"], "generation": False, "evaluation": False,
                    "training": False, "model_execution": False, "center": "response_predictor_clean_M60",
                    "families": list(fit.fitting.FAMILIES), "difference_policy": "FP32_before_token_mean"},
                "expected_records": len(inspected["rows"]), "expected_cells": 72, "tasks": tasks,
                "extraction_manifest_sha256": digest, "extraction_complete_sha256": fit.sha256(completion_path),
                "cache_verification_sha256": fit.sha256(verify_path)}
    guard.validate_manifest(manifest)
    path = output / "RESOURCE_MANIFEST.json"
    fit.write_json(path, manifest)
    return {"manifest": str(path), **fit.binding(path), "records": len(inspected["rows"]),
            "cache_files_verified": len(verification["files"]), "workers": 8, "launched": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    require(socket.gethostname().split(".")[0] == "gpu-03" and os.getuid() == 3357,
            "Run this bounded preparation near the authorized gpu-03 cache")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation needs explicit empty CUDA mask")
    print(json.dumps(prepare(args.extraction_manifest, args.sha256, args.phase), sort_keys=True))


if __name__ == "__main__":
    main()
