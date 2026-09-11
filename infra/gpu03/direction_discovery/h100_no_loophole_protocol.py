"""Versioned H100 path/runtime adaptation of the frozen no-hint protocol.

Logical conditions and request science are preserved. Only an explicitly bound
candidate path is translated for workers; no historical /scratch alias is used.
This module has no model, tokenizer, network, reservation or launch operation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct

from infra.gpu03.direction_discovery import no_loophole_plan as frozen

FROZEN_PLANNER_SHA = "6adc62764415936852f3ff8b79c38cf862cfd3d474bd53ffb95408a7ec3941ca"
PORTABILITY_PROTOCOL = "h100_workstation_no_loophole_v1"
WORK_ROOT = "/home/ubuntu/h100-workspace/work"
HOST = "codex-h100"
INSTANCE = "i-00000000000000002"
HISTORICAL_ENVIRONMENT_SHA = "ea146cff72a0588917bc5aae107543315919bfc18a6a76c9844e56045cdcf5cb"
VERSIONS = {"peft": "0.17.1", "safetensors": "0.6.2", "tokenizers": "0.22.1",
            "torch": "2.8.0+cu128", "transformers": "4.57.1", "vllm": "0.11.0"}
CONSTRUCTION = "unchanged_engine.load_projections_on_qualified_cpu_runtime_replacement"
HERE = "infra/gpu03/direction_discovery/h100_no_loophole_protocol.py"
FROZEN_HERE = "infra/gpu03/direction_discovery/no_loophole_plan.py"

# These immutable scientific primitives are shared, never patched or replaced.
require, canonical, sha, digest = frozen.require, frozen.canonical, frozen.sha, frozen.digest
get_ref, ref, read_ref, load_ref = frozen.get_ref, frozen.ref, frozen.read_ref, frozen.load_ref
jsonl, prompt_hash, ids_hash = frozen.jsonl, frozen.prompt_hash, frozen.ids_hash
text_hash, stable_seed = frozen.text_hash, frozen.stable_seed
make_prepared_row, _old, _input_check = frozen.make_prepared_row, frozen._old, frozen._input_check
PROTOCOL, PURPOSE, PHASE = frozen.PROTOCOL, frozen.PURPOSE, frozen.PHASE
OLD_FULL_SHA, OLD_PREPARED_SHA = frozen.OLD_FULL_SHA, frozen.OLD_PREPARED_SHA
DATASET_SHA, CARRIERS_SHA = frozen.DATASET_SHA, frozen.CARRIERS_SHA
SOURCES, FILES, COUNT_FIELDS = frozen.SOURCES, frozen.FILES, frozen.COUNT_FIELDS
MASTER_SHA, PARENT_SHA, SAMPLING, BRIEF_SHA = frozen.MASTER_SHA, frozen.PARENT_SHA, frozen.SAMPLING, frozen.BRIEF_SHA


def check_frozen_source():
    require(sha(frozen.__file__) == FROZEN_PLANNER_SHA, "Frozen scientific planner changed")


def runtime_content(value):
    """A qualified managed Python may be a symlink; compare its actual bytes."""
    path = get_ref(value)
    st = path.stat()
    require(path.is_file() and st.st_size <= 64 << 20 and "size_bytes" in value,
            "Unbounded runtime binary reference")
    data = path.read_bytes()
    after = path.stat()
    require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Runtime binary changed while reading")
    require(hashlib.sha256(data).hexdigest() == value["sha256"] and len(data) == value["size_bytes"],
            "Qualified replacement Python bytes differ")
    return path


def validate_portability(value, conditions):
    check_frozen_source()
    require(isinstance(value, dict) and set(value) ==
            {"schema_version", "protocol", "work_root", "candidate_map", "runtime_replacement"} and
            value["schema_version"] == 1 and value["protocol"] == PORTABILITY_PROTOCOL and
            value["work_root"] == WORK_ROOT, "Wrong H100 portability contract")
    candidates = [layer for condition in conditions.values() for layer in condition["layers"]
                  if layer["kind"] == "candidate"]
    require(len(candidates) == 1, "H100 adaptation requires the sole frozen candidate")
    layer = candidates[0]
    mapping = value["candidate_map"]
    require(isinstance(mapping, dict) and set(mapping) == {layer["path"]}, "Candidate map coverage differs")
    target = mapping[layer["path"]]
    require(set(target) == {"path", "sha256", "size_bytes"} and
            target["sha256"] == layer["sha256"] and target["size_bytes"] == 361240,
            "Relocated candidate hash/size differs")
    path = get_ref(target)
    require(path != Path(layer["path"]) and path.is_relative_to(Path(WORK_ROOT)) and
            str(path) != WORK_ROOT and not str(path).startswith("/scratch/"),
            "Candidate must use an explicit H100 work input, not a compatibility alias")
    read_ref(target)
    runtime = value["runtime_replacement"]
    require(isinstance(runtime, dict) and set(runtime) == {"historical_environment", "python_binary", "qualification"},
            "Incomplete qualified runtime replacement")
    require(runtime["historical_environment"]["sha256"] == HISTORICAL_ENVIRONMENT_SHA,
            "Wrong original runtime evidence")
    historical = load_ref(runtime["historical_environment"])
    require(historical.get("python") == "3.12.3" and historical.get("credentials_recorded") is False and
            all(historical.get("packages", {}).get(k) == v for k, v in VERSIONS.items() if k != "tokenizers"),
            "Historical Python/package evidence differs")
    proof = load_ref(runtime["qualification"])
    require(proof.get("status") == "verified_h100_runtime_replacement" and proof.get("host") == HOST and
            proof.get("instance_id") == INSTANCE and proof.get("python_version") == "3.12.3" and
            proof.get("python_binary") == runtime["python_binary"] and proof.get("runtime_versions") == VERSIONS and
            proof.get("original_environment_sha256") == HISTORICAL_ENVIRONMENT_SHA and
            proof.get("original_python_binary_reused") is False and
            proof.get("numerical_libraries_imported") is True and proof.get("process_release_verified") is True,
            "H100 runtime replacement lacks independent positive qualification")
    require(proof.get("implementation") == "CPython" and
            all(isinstance(proof.get(k), str) and proof[k] for k in ("compiler", "platform")) and
            all(isinstance(proof.get(k), list) and len(proof[k]) == 2 and
                all(isinstance(v, str) for v in proof[k]) for k in ("build", "libc")),
            "Missing replacement implementation/build/platform provenance")
    runtime_content(runtime["python_binary"])
    return [path, get_ref(runtime["historical_environment"]), get_ref(runtime["qualification"]),
            get_ref(runtime["python_binary"])]


def translate_conditions(conditions, portability):
    validate_portability(portability, conditions)
    physical = copy.deepcopy(conditions)
    for condition in physical.values():
        for layer in condition["layers"]:
            if layer["kind"] == "candidate":
                layer["path"] = portability["candidate_map"][layer["path"]]["path"]
    return physical


def physical_conditions(plan):
    validate_full(plan)
    return translate_conditions(plan["conditions"], plan["no_loophole_capability"]["h100_portability"])


def _package(reference, old):
    check_frozen_source()
    ctx = frozen._package(reference, old)
    portability = ctx["prompt_bundle"].get("h100_portability")
    ctx["paths"].extend(validate_portability(portability, old["conditions"]))
    return ctx


def _metadata(old_ref, bundle_ref, verification_ref, portability):
    return {**frozen._metadata(old_ref, bundle_ref, verification_ref),
            "h100_portability": copy.deepcopy(portability),
            "h100_hardware_change": "NVIDIA H100; original batch1/BF16/math-SDPA/TF32-off policy retained",
            "cross_architecture_bitwise_equivalence_claimed": False,
            "frozen_planner_sha256": FROZEN_PLANNER_SHA}


def _completed_package_proof(ctx, meta):
    frozen._completed_package_proof(ctx, meta)
    proof = load_ref(meta["prompt_package_verification"])
    require(ctx["prompt_bundle"]["h100_portability"] == meta["h100_portability"] and
            proof.get("h100_portability_sha256") == digest(meta["h100_portability"]),
            "Completed package portability proof differs")


def _derive(old, meta, ctx, counts, prior):
    result = frozen._derive(old, meta, ctx, counts, prior)
    result["builder_sha256"] = sha(__file__)
    return result


def validate_full(plan):
    check_frozen_source()
    require(isinstance(plan, dict) and not any(k in plan for k in
            ("execution_partition", "test_execution", "test_bundle", "finalist_recovery", "recovery_plan")),
            "H100 capability requires one complete independent740 plan")
    meta = plan.get("no_loophole_capability")
    require(isinstance(meta, dict) and {"old_full_plan", "prompt_bundle", "prompt_package_verification", "h100_portability"} <= set(meta) and
            meta == _metadata(meta["old_full_plan"], meta["prompt_bundle"], meta["prompt_package_verification"], meta["h100_portability"]),
            "H100 protocol/source/budget metadata differs")
    old = _old(meta["old_full_plan"])
    ctx = _package(meta["prompt_bundle"], old); ctx["old_plan"] = old
    _completed_package_proof(ctx, meta)
    _projection_check(ctx)
    _input_check(ctx, jsonl(ctx["payloads"]["prepared_records.jsonl"]), jsonl(ctx["payloads"]["canonical_nohint37.jsonl"]))
    counts = {name: plan.get(name) for name in COUNT_FIELDS}
    require(all(type(v) is int and v >= 0 for v in counts.values()) and
            counts["previously_committed_generation_requests"] >= old["previously_committed_generation_requests"] and
            counts["previously_committed_generation_requests"] + 740 <= 4096 and
            old["previously_committed_tf_requests"] <= counts["previously_committed_tf_requests"] <= 12000 and
            all(counts[k] == 0 for k in counts if "untouched_test" in k), "Invalid cumulative H100 capability budget")
    prior = plan.get("prior_phase_manifest_bindings")
    require(isinstance(prior, list) and all(isinstance(x, dict) and set(x) == {"path", "sha256"} for x in prior),
            "Invalid prior manifest bindings")
    for item in prior:
        get_ref(item)
    require(len({r["path"] for r in prior}) == len({r["sha256"] for r in prior}) == len(prior), "Duplicate prior manifests")
    require(plan == _derive(old, meta, ctx, counts, prior), "Scientific plan changed beyond environment/declared portability")
    require(len(plan["requests"]) == len({r["request_id"] for r in plan["requests"]}) == 740 and
            not {r["request_id"] for r in plan["requests"]}.intersection(r["request_id"] for r in old["requests"]),
            "Fresh request IDs collide")
    return ctx


def validate_inputs(plan, prepared_rows, dataset_rows):
    ctx = validate_full(plan)
    _input_check(ctx, prepared_rows, dataset_rows)
    return ctx


def validate_source(plan, source_root):
    check_frozen_source()
    root = Path(source_root)
    require(plan.get("no_loophole_capability", {}).get("frozen_generation_sources") == SOURCES,
            "Wrong historical generation source contract")
    for relative, expected in {**SOURCES, FROZEN_HERE: FROZEN_PLANNER_SHA, HERE: sha(__file__)}.items():
        require(sha(root / relative) == expected, "H100 source adaptation changed")
    require(plan.get("builder_sha256") == sha(__file__), "H100 planner source changed")


def bindings(plan, *, verify=True):
    ctx = validate_full(plan)
    return sorted(set(ctx["paths"] + [get_ref(plan["no_loophole_capability"]["old_full_plan"])]))


def validate_against_ledger(plan, budget):
    validate_full(plan)
    for field, key in COUNT_FIELDS.items():
        require(type(budget.get(key)) is int and plan[field] == budget[key], "H100 plan uses stale/reset budget counts")
    require(all(budget[k] == 0 for k in ("untouched_test_requests", "untouched_test_generation_requests", "untouched_test_tf_requests")),
            "Untouched-test commitments must remain zero")
    phases = budget.get("phases")
    require(isinstance(phases, list), "Missing authoritative phase ledger")
    current = {r["request_id"] for r in plan["requests"]}
    require(not any(current.intersection(p.get("generation_request_ids", [])) for p in phases), "H100 requests already committed")
    require(plan["prior_phase_manifest_bindings"] ==
            [{"path": p["manifest_path"], "sha256": p["manifest_sha256"]} for p in phases],
            "H100 prior manifest lineage differs from current ledger")


def build_plan(old_plan_path, prompt_bundle_ref, budget, *, prompt_package_verification):
    old_ref = ref(old_plan_path); old = _old(old_ref)
    ctx = _package(prompt_bundle_ref, old); ctx["old_plan"] = old
    meta = _metadata(old_ref, prompt_bundle_ref, prompt_package_verification, ctx["prompt_bundle"]["h100_portability"])
    _completed_package_proof(ctx, meta)
    _projection_check(ctx)
    _input_check(ctx, jsonl(ctx["payloads"]["prepared_records.jsonl"]), jsonl(ctx["payloads"]["canonical_nohint37.jsonl"]))
    counts = {field: budget[key] for field, key in COUNT_FIELDS.items()}
    prior = [{"path": p["manifest_path"], "sha256": p["manifest_sha256"]} for p in budget["phases"]]
    result = _derive(old, meta, ctx, counts, prior)
    validate_against_ledger(result, budget)
    return result


def write_plan(old_plan_path, prompt_bundle_ref, budget, output, *, prompt_package_verification):
    result = build_plan(old_plan_path, prompt_bundle_ref, budget,
                        prompt_package_verification=prompt_package_verification)
    output = Path(output)
    require(output.is_absolute() and not output.exists(), "Plan output must be a fresh absolute directory")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "request_plan.json"
    with path.open("x") as stream:
        stream.write(canonical(result) + "\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o400)
    require(load_ref(ref(path)) == result, "Written H100 plan changed")
    return {"request_plan": str(path), "sha256": sha(path), "requests": 740,
            "no_compute_launched": True, "budget_reserved": False}


def _projection_check(ctx):
    payloads, old, bundle = ctx["payloads"], ctx["old_plan"], ctx["prompt_bundle"]
    audit = json.loads(payloads["projection_audit.json"])
    proof = json.loads(payloads["projection_independent_verification.json"])
    file_hash = bundle["files"]["projection_vectors.safetensors"]["sha256"]
    nonbaseline = {k: v for k, v in old["conditions"].items() if v["layers"]}
    require(audit.get("status") == "passed" and audit.get("torch_version") == "2.8.0+cu128" and
            audit.get("source_sha256") == SOURCES and audit.get("conditions") == nonbaseline and
            audit.get("target_tensor_sha256") == nonbaseline["target:L21.transition.pc04"]["layers"][0]["sha256"] and
            audit.get("saved_tensor_sha256") == file_hash and audit.get("historical_random_Q_files_available") is False and
            audit.get("historical_Q_byte_comparison_performed") is False and
            audit.get("construction") == CONSTRUCTION and
            audit.get("h100_portability_sha256") == digest(bundle["h100_portability"]) and
            audit.get("physical_conditions") == {k: v for k, v in translate_conditions(old["conditions"], bundle["h100_portability"]).items() if v["layers"]},
            "Projection provenance differs from the historical loader/conditions")
    require(proof.get("status") == "independently_verified_projection_reconstruction" and
            proof.get("audit_sha256") == bundle["files"]["projection_audit.json"]["sha256"] and
            proof.get("saved_tensor_sha256") == file_hash and proof.get("vectors") == 4 and
            proof.get("reconstruction_bitwise_equal") is True and proof.get("readback_bitwise_equal") is True,
            "Projection sidecars lack independent reconstruction/readback")
    data = payloads["projection_vectors.safetensors"]
    require(len(data) > 8, "Truncated projection file")
    size = int.from_bytes(data[:8], "little")
    require(2 <= size <= 65536 and 8 + size <= len(data), "Invalid projection header")
    header = json.loads(data[8:8 + size])
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    require(set(tensors) == set(nonbaseline) == set(audit.get("vector_sha256", {})), "Projection file must contain exactly the four condition Qs")
    intervals = []
    for condition, info in tensors.items():
        require(set(info) == {"dtype", "shape", "data_offsets"} and info["dtype"] == "F32" and info["shape"] == [2560, 1], "Projection shape/dtype differs")
        offsets = info["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(v) is int for v in offsets), "Invalid projection offsets")
        start, end = offsets
        require(0 <= start < end <= len(data) - 8 - size and end - start == 2560 * 4, "Projection byte span differs")
        raw = data[8 + size + start:8 + size + end]
        require(hashlib.sha256(raw).hexdigest() == audit["vector_sha256"][condition], "Projection raw bytes differ")
        values = struct.unpack("<2560f", raw)
        require(abs(sum(v * v for v in values) - 1.0) <= 2e-5, "Projection Q is nonfinite or not unit length")
        intervals.append((start, end))
    intervals.sort()
    require(intervals[0][0] == 0 and intervals[-1][1] == len(data) - 8 - size and
            all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])), "Projection payload has gaps/overlap/trailing bytes")
