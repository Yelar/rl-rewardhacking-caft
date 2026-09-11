"""Assemble fixed native caches; independently inspect their portable semantics.

The supervisor separately SHA-verifies every byte after producer exit. inspect()
does not repeat that expensive full-cache scan; it audits all headers, exact input
tensors, masks and identities against the manifest-bound original prepared rows.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import re

import fixed_cache as fixed

EXPECTED_RECORDS = {"fixed_cache_core": 561, "fixed_cache_auxiliary": 51}
EXPECTED_PROBLEMS = {"fixed_cache_core": 187, "fixed_cache_auxiliary": 47}
PROFILE = {"padding_side": "right", "padded_sequence_length": 2176, "pad_token_id": 151643,
           "padding_attention_mask": 0, "attention_backend": "torch_sdpa_MATH_only",
           "original_tokens_preserved": True, "stored_completion_positions_only": True, "prompt_final_saved": True}
require = fixed.require


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def safe_child(root, relative):
    root, relative = Path(root), Path(relative)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts, "unsafe package relative path")
    path = root / relative
    require(all(not parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent),
            "package path includes a symlink")
    require(path.is_file(), "missing package file: " + str(relative))
    return path


def copied(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as src, destination.open("xb") as dst:
        for block in iter(lambda: src.read(8 << 20), b""):
            dst.write(block)
    destination.chmod(0o444)


def phase_role(m):
    require(m["phase"] in EXPECTED_RECORDS, "only complete core/auxiliary phases can produce a fixed-cache package")
    role = "core" if m["phase"] == "fixed_cache_core" else "auxiliary"
    require(m["scientific"]["cache_role"] == role, "manifest cache role conflicts with phase")
    return role


def task_bindings(m, root, *, original):
    """Read exact bound task copies; validate disjoint request and record IDs."""
    tasks, requested, request_ids = {}, {}, set()
    prepared_paths = set()
    for worker in m["workers"]:
        name = worker["name"]
        require(re.fullmatch(r"gpu_[0-7]", name) is not None and name not in tasks, "invalid/duplicate package worker")
        source_path = Path(worker["command"][3])
        path = source_path if original else safe_child(root, "input/tasks/" + name + ".json")
        info = m["bound_files"][str(source_path)]
        require(path.stat().st_size == info["size_bytes"] and fixed.sha256(path) == info["sha256"], "bound task changed")
        task = json.loads(path.read_text())
        require(task["run_token"] == m["run_token"] and task["worker_name"] == name and task["mode"] == "fixed_cache",
                "wrong fixed-cache task identity")
        require(task["cache_role"] == phase_role(m) and task["padded_sequence_length"] == 2176 and
                task["pad_token_id"] == 151643 and task["attention_policy"] == "exclusive_math", "task cache profile changed")
        require(Path(task["output"]) == Path(m["output"]) / "workers" / name, "task output is not its assigned worker")
        require(len(task["requests"]) == worker["success_expect"]["requests"] and
                worker["success_expect"]["mode"] == "fixed_cache", "task request count/mode changed")
        for request in task["requests"]:
            rid, req = request["record_id"], request["request_id"]
            require(rid not in requested and req not in request_ids, "duplicate record/request across fixed-cache tasks")
            requested[rid] = name
            request_ids.add(req)
        tasks[name] = task
        prepared_paths.add(task["prepared_records"])
    require(len(prepared_paths) == 1, "workers use different prepared inputs")
    return tasks, requested, Path(next(iter(prepared_paths)))


def check_rows(m, rows):
    require(len(rows) == EXPECTED_RECORDS[m["phase"]] and
            len({r["problem_id_key"] for r in rows}) == EXPECTED_PROBLEMS[m["phase"]], "fixed-cache scientific record/problem target missing")
    fixed.validate_groups(rows, phase_role(m))
    return {r["record_id"]: r for r in rows}


def check_worker(root, m, worker, task, rows_by_id):
    name = worker["name"]
    prefix = Path("workers") / name
    output = root / prefix
    task_copy = json.loads(safe_child(root, prefix / "task.json").read_text())
    require(task_copy == task, "worker task copy changed")
    success = json.loads(safe_child(root, prefix / "SUCCESS.json").read_text())
    require(success["status"] == "succeeded" and success["run_token"] == m["run_token"] and
            success["worker_name"] == name and success["mode"] == "fixed_cache" and
            success["requests"] == len(task["requests"]), "worker success receipt is incomplete")
    require(not (output / "FAILURE.json").exists(), "worker has a failure artifact")
    report = json.loads(safe_child(root, prefix / "fixed_cache_report.json").read_text())
    require(success["model_load_reports"] == report and report["status"] == "succeeded", "worker success/report mismatch")
    expected_ids = {r["record_id"] for r in task["requests"]}
    worker_rows = [rows_by_id[rid] for rid in expected_ids]
    fixed.validate_groups(worker_rows, phase_role(m))
    require(report["records"] == len(expected_ids) and report["problems"] == len({r["problem_id_key"] for r in worker_rows}),
            "worker report count mismatch")
    for field in ("all_identical_prefixes_bitwise_equal", "all_native_readbacks_bitwise_equal", "fp32_delta_readback_exact",
                  "raw_activations_retained", "differences_computed"):
        require(report[field] is True, "worker numerical audit failed: " + field)
    require(report["activation_cache_profile"] == PROFILE, "worker fixed-padding/backend profile changed")
    required_policy = {"deterministic_algorithms": True, "deterministic_warn_only": False,
                       "matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False,
                       "cudnn_deterministic": True, "cuda_autocast": False, "cpu_autocast": False}
    require(all(report["numerical_runtime_policy"].get(key) is value for key, value in required_policy.items()),
            "worker numerical runtime policy changed")
    require(report["cross_package_prefix_audit_required"] == (phase_role(m) == "auxiliary"), "wrong cross-package audit requirement")
    _, legacy = fixed.dependencies()
    legacy.validate_model_load_reports(report["model_load_reports"]["h0"], report["model_load_reports"]["h60"], name)
    for kind in ("h0", "h60"):
        legacy.validate_post_model_cuda_state(report["cuda_release"][kind], name + kind)
        for test in ("repeat", "future_causality"):
            require(report["qualifications"][kind][test]["bitwise_equal"] is True, "fixed-shape qualification failed")
    groups = Counter((r["problem_id_key"], tuple(r["prompt_token_ids"])) for r in worker_rows)
    expected_prefixes = sum(n * (n - 1) for n in groups.values())  # pairs * two models
    path = output / "prefix_audit.jsonl"
    audits = read_jsonl(path) if path.is_file() else []
    require(report["identical_prefix_comparisons"] == len(audits) == expected_prefixes, "prefix coverage count mismatch")
    observed = set()
    for audit in audits:
        kind, ids = audit["kind"], audit["record_ids"]
        require(kind in ("h0", "h60") and len(ids) == 2 and set(ids) <= expected_ids and ids[0] != ids[1], "invalid prefix identity")
        a, b = [rows_by_id[rid] for rid in ids]
        require(a["problem_id_key"] == b["problem_id_key"] and a["prompt_token_ids"] == b["prompt_token_ids"], "prefix audit pairs unequal prompts")
        key = (kind, *sorted(ids))
        require(key not in observed, "duplicate prefix audit pair")
        observed.add(key)
        count = fixed.common_prefix_count(a["completion_token_ids"], b["completion_token_ids"])
        require(audit["common_completion_tokens"] == count and audit["prompt_final"]["bitwise_equal"] is True and
                ((count == 0 and audit["completion_prefix"] is None) or
                 (count > 0 and audit["completion_prefix"]["bitwise_equal"] is True)), "prefix equality failed")
    entries = read_jsonl(safe_child(root, prefix / "workerindex.jsonl"))
    require(len(entries) == len(expected_ids) and {e["record_id"] for e in entries} == expected_ids, "worker index record coverage differs")
    for entry in entries:
        require(entry["verified"] is True and entry["record_index"] == rows_by_id[entry["record_id"]]["record_index"], "unverified/misaligned worker index")
    return entries, report


def finalize(m, output):
    """Called after all worker receipts and GPU release, before artifact hashing."""
    output = Path(output)
    phase_role(m)
    require(output == Path(m["output"]), "package output differs from reviewed allocation")
    reviewed = safe_child(output, "reviewed_manifest.json")
    require(json.loads(reviewed.read_text()) == m, "package reviewed manifest differs")
    digest = fixed.sha256(reviewed)
    release = json.loads(safe_child(output, "gpu_release.json").read_text())
    require(release["verified"] is True and release["gpu_ids"] == m["gpu_ids"], "package finalization requires positive GPU release")
    tasks, requested, prepared = task_bindings(m, output, original=True)
    require(fixed.sha256(prepared) == m["scientific"]["input_prepared_sha256"], "prepared input changed")
    rows = read_jsonl(prepared)
    by_id = check_rows(m, rows)
    require(set(requested) == set(by_id), "requests do not cover every original prepared record")
    for task in tasks.values():
        require(fixed.resolve_manifest_digest(task) == digest, "native task did not bind this extraction manifest")
    entries, reports = [], []
    for worker in m["workers"]:
        name = worker["name"]
        partial, report = check_worker(output, m, worker, tasks[name], by_id)
        reports.append(report)
        for original in partial:
            entry = copy.deepcopy(original)
            for info in [*entry["models"].values(), entry["delta"]]:
                relative = Path("workers") / name / info["tensor_path"]
                safe_child(output, relative)
                info["tensor_path"] = str(relative)
            entries.append(entry)
    require(len(entries) == len(by_id) and len({r["record_id"] for r in entries}) == len(entries), "duplicate/missing merged records")
    copied(prepared, output / "input/prepared_records.jsonl")
    for worker in m["workers"]:
        copied(worker["command"][3], output / "input/tasks" / (worker["name"] + ".json"))
    with (output / "activation_index.jsonl").open("x") as stream:
        for entry in sorted(entries, key=lambda r: r["record_index"]):
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    summary = {"schema_version": 2, "status": "succeeded", "phase": m["phase"], "cache_role": phase_role(m),
               "manifest_sha256": digest, "run_token": m["run_token"], "records": len(rows),
               "problems": len({r["problem_id_key"] for r in rows}), "layers": 36, "hidden_size": 2560,
               "prepared_records_sha256": fixed.sha256(prepared), "storage_dtype": "bfloat16",
               "delta_storage_dtype": "float32", "differences_computed": True, "raw_activations_retained": True,
               "all_native_readbacks_bitwise_equal": True, "all_fp32_delta_readbacks_exact": True,
               "identical_prefix_comparisons": sum(r["identical_prefix_comparisons"] for r in reports),
               "all_within_worker_prefixes_bitwise_equal": True, "activation_cache_profile": PROFILE,
               "cross_package_prefix_audit_required": phase_role(m) == "auxiliary",
               "split_record_counts": dict(Counter(r["problem_split"] for r in rows)),
               "source_generations_unchanged": True}
    fixed.write_json(output / "extraction_summary.json", summary)
    return inspect(output)


def inspect_tensor(root, row, entry, kind, digest, artifact_files=None):
    import torch
    from safetensors import safe_open
    info = entry["delta"] if kind == "delta_h" else entry["models"][kind]
    if artifact_files is not None:
        require(artifact_files.get(info["tensor_path"]) == {"sha256": info["sha256"], "size_bytes": info["size_bytes"]},
                "raw/delta index disagrees with independently verified artifact manifest")
    path = safe_child(root, info["tensor_path"])
    require(path.stat().st_size == info["size_bytes"] and not path.stat().st_mode & 0o222 and
            re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is not None, "raw/delta file size or immutable identity differs")
    expected_shape = [36, row["completion_token_count"], fixed.HIDDEN]
    require(info["shape"] == expected_shape and info["all_finite"] is True, "raw/delta index shape/numerical audit differs")
    expected_dtype = "float32" if kind == "delta_h" else "bfloat16"
    require(info["dtype"] == expected_dtype, "index dtype differs")
    audited = (info["fp32_readback_exact"] is True and info["float64_reference_rounded_to_fp32_exact"] is True
               if kind == "delta_h" else info["native_readback_bitwise_equal"] is True)
    require(audited, "missing raw/delta numerical readback")
    model_inputs = {k: v[0] for k, v in fixed.padded_inputs(row, "cpu").items()}
    expected_aux = fixed.auxiliary_tensors(row, model_inputs)
    prompt_key = "prompt_final_delta" if kind == "delta_h" else "prompt_final"
    with safe_open(str(path), framework="pt", device="cpu") as tensor:
        expected_meta = fixed.metadata(row, kind, digest)
        if kind == "delta_h":
            expected_meta.update({k + "_sha256": entry["models"][k]["sha256"] for k in ("h0", "h60")})
        require(tensor.metadata() == expected_meta, "native/delta metadata differs from original tokens/model/site/profile")
        require(set(tensor.keys()) == {kind, prompt_key, *expected_aux}, "raw/delta tensor key contract changed")
        require(tensor.get_slice(kind).get_shape() == expected_shape and tensor.get_slice(kind).get_dtype() ==
                ("F32" if kind == "delta_h" else "BF16"), "native/delta header shape/dtype differs")
        prompt = tensor.get_tensor(prompt_key)
        require(prompt.shape == (36, fixed.HIDDEN) and prompt.dtype == (torch.float32 if kind == "delta_h" else torch.bfloat16)
                and torch.isfinite(prompt).all().item(), "prompt-final shape/dtype/value invalid")
        for key, expected in expected_aux.items():
            value = tensor.get_tensor(key)
            require(value.dtype == expected.dtype and torch.equal(value, expected), "original token/offset/padded-input/mask mismatch: " + key)
    return prompt


def inspect(root):
    """Portable semantic verification; full payload hashes belong to supervisor."""
    import torch
    root = Path(root)
    m = json.loads(safe_child(root, "reviewed_manifest.json").read_text())
    role = phase_role(m)
    digest = fixed.sha256(root / "reviewed_manifest.json")
    tasks, requested, _ = task_bindings(m, root, original=False)
    prepared = safe_child(root, "input/prepared_records.jsonl")
    require(fixed.sha256(prepared) == m["scientific"]["input_prepared_sha256"], "copied original prepared records changed")
    rows = read_jsonl(prepared)
    by_id = check_rows(m, rows)
    require(set(by_id) == set(requested), "packaged requests/prepared IDs disagree")
    summary = json.loads(safe_child(root, "extraction_summary.json").read_text())
    artifact_path = root / "artifact_manifest.json"
    artifact_files = None
    if artifact_path.exists():
        artifacts = json.loads(safe_child(root, "artifact_manifest.json").read_text())
        require(artifacts["algorithm"] == "sha256", "unsupported artifact digest")
        artifact_files = artifacts["files"]
    require(summary["status"] == "succeeded" and summary["manifest_sha256"] == digest and
            summary["phase"] == m["phase"] and summary["cache_role"] == role and summary["run_token"] == m["run_token"] and
            summary["records"] == len(rows) and summary["problems"] == len({r["problem_id_key"] for r in rows}) and
            summary["layers"] == 36 and summary["hidden_size"] == 2560 and
            summary["prepared_records_sha256"] == fixed.sha256(prepared) and summary["activation_cache_profile"] == PROFILE and
            summary["storage_dtype"] == "bfloat16" and summary["delta_storage_dtype"] == "float32" and
            summary["raw_activations_retained"] is True and summary["differences_computed"] is True and
            summary["cross_package_prefix_audit_required"] == (role == "auxiliary"), "extraction summary profile/count/provenance changed")
    expected_entries, reports = [], []
    for worker in m["workers"]:
        partial, report = check_worker(root, m, worker, tasks[worker["name"]], by_id)
        reports.append(report)
        for original in partial:
            entry = copy.deepcopy(original)
            for info in [*entry["models"].values(), entry["delta"]]:
                info["tensor_path"] = str(Path("workers") / worker["name"] / info["tensor_path"])
            expected_entries.append(entry)
    entries = read_jsonl(safe_child(root, "activation_index.jsonl"))
    require(entries == sorted(expected_entries, key=lambda r: r["record_index"]) and len(entries) == len(by_id) and
            {e["record_id"] for e in entries} == set(by_id), "merged index differs from exact worker coverage")
    require(summary["identical_prefix_comparisons"] == sum(r["identical_prefix_comparisons"] for r in reports), "summary prefix count differs")
    for entry in entries:
        row = by_id[entry["record_id"]]
        require(set(entry["models"]) == {"h0", "h60"}, "wrong model set")
        h0, h60, delta = [inspect_tensor(root, row, entry, kind, digest, artifact_files) for kind in ("h0", "h60", "delta_h")]
        require(torch.equal(delta, h60.float() - h0.float()), "independent prompt-final delta subtraction differs")
    return {"status": "verified", "records": len(rows), "problems": len({r["problem_id_key"] for r in rows}),
            "manifest_sha256": digest, "native_files": len(rows) * 2, "delta_files": len(rows),
            "all_headers_original_inputs_offsets_masks_verified": True, "full_payload_hashes_recomputed": False,
            "index_joined_to_artifact_manifest": artifact_files is not None,
            "full_payload_hash_verifier": "independent_supervisor_artifact_manifest_verification",
            "cross_package_prefix_audit_required": role == "auxiliary"}
