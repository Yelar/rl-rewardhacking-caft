"""Complete740 H100 CPU evaluation, with explicit legacy and helper sidecar.

This versioned adapter never changes the GPU04 evaluator or launches a service.
Its returned exact service argv is dispatched separately by the reviewed caller.
All generated code runs through the existing outer/nested bounded sandbox.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import shutil
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.gpu03.direction_discovery import h100_supervisor as host
from infra.gpu03.direction_discovery import h100_no_loophole_protocol as protocol
from infra.gpu03.direction_discovery import h100_sandbox as sandbox
from infra.gpu03.direction_discovery import helper_aware_evaluation_v1 as repair
from infra.gpu03.direction_discovery import coarse_evaluation_context as coarse

PURPOSE = "h100_complete740_legacy_and_helper_evaluation_v1"
HERE = "infra/gpu03/direction_discovery/h100_evaluate.py"
EVALUATE_SHA = repair.LEGACY_EVALUATE_SHA
REPAIR_SHA = "33b726445c334ac248ae24836894b33a6e6b159d3d084e3341cda083c0fb82ca"
SANDBOX_SHA = "563b64d517f13c3388ea7a77c7ea1d3c09374eb0ae2a00d56765d0ec425e4b50"
METRICS_SHA = "9f1a878502d4ffadc95e91e8df05c9d9dff952c6d04843f5d1d4c61d10a081b5"
IDENTITY = ("request_id", "record_id", "problem_id", "problem_split", "condition_id", "scope", "sample_index", "seed")
LIMITS = {"workers": 8, "sandbox_workers": 2, "timeout_seconds": 3, "memory_per_worker_mib": 1024,
          "worker_wall_seconds": 7200, "systemd_runtime_seconds": 7290,
          "memory_bytes": 32 << 30, "tasks": 512, "maximum_output_bytes": 2 << 30}
CPU_SET = "112-127"
require, canonical, sha, ref = host.require, host.canonical, host.sha, host.ref
write = host.exclusive_json


def source_pins():
    return {**repair.SOURCE_PINS, HERE: sha(__file__),
            "infra/gpu03/direction_discovery/evaluate.py": EVALUATE_SHA,
            "infra/gpu03/direction_discovery/helper_aware_evaluation_v1.py": REPAIR_SHA,
            "infra/gpu03/direction_discovery/h100_sandbox.py": SANDBOX_SHA,
            "infra/gpu03/direction_discovery/h100_no_loophole_protocol.py": sha(protocol.__file__),
            "infra/gpu03/direction_discovery/h100_supervisor.py": sha(host.__file__),
            "infra/gpu03/direction_discovery/metrics.py": METRICS_SHA,
            "infra/gpu03/direction_discovery/coarse_evaluation_context.py": sha(coarse.__file__),
            "infra/gpu03/direction_discovery/h100_tf_run.py": sha(coarse.generation.__file__)}


def check_sources(source):
    source = Path(source)
    for relative, digest in source_pins().items():
        require(sha(source / relative) == digest, "Frozen evaluator/source differs: " + relative)
    sandbox.check_sources()


def base_module():
    from infra.gpu03.direction_discovery import evaluate
    require(sha(evaluate.__file__) == EVALUATE_SHA and sha(repair.__file__) == REPAIR_SHA and
            sha(sandbox.__file__) == SANDBOX_SHA, "Running evaluator/repair/sandbox source differs")
    return evaluate


def check_runtime(m, plan):
    runtime = (plan["coarse_layer_map"]["runtime_replacement"] if "coarse_layer_map" in plan else
               plan["no_loophole_capability"]["h100_portability"]["runtime_replacement"])
    require(Path(sys.executable).resolve() == Path(m["python"]).resolve() and
            sha(Path(sys.executable).resolve()) == runtime["python_binary"]["sha256"] and
            {name: importlib.metadata.version(name) for name in host.VERSIONS} == host.VERSIONS and
            sys.version_info[:3] == (3, 12, 3), "CPU evaluator must use the exact qualified replacement Python and six versions")


def json_rows(reference, *, immutable=True, coarse_merged=False):
    # The7240 merged files share the already enforced total-output cap. Keep
    # the historical740 and each905-row worker at the original128MiB bound.
    return [json.loads(line) for line in host.read_ref(reference, immutable=immutable,
        max_bytes=LIMITS["maximum_output_bytes"] if coarse_merged else 128 << 20).splitlines() if line.strip()]


def unique(rows, key):
    result = {str(row[key]): row for row in rows}
    require(len(result) == len(rows), "Duplicate identity: " + key)
    return result


def generation_context(package):
    if isinstance(package,dict) and "coarse_context" in package:return coarse.context(package)
    require(isinstance(package, dict) and set(package) == {"manifest", "verification"}, "Exact H100 producer package required")
    m = host.load_manifest(package["manifest"]["path"], package["manifest"]["sha256"])
    continuation = m.get("continuation")
    require(m["phase"] == "h100_no_loophole_capability" and
            m["generation_requests"] == (735 if continuation is not None else 740),
            "Only full scientific740 generation may enter evaluation")
    if continuation is not None:
        require(isinstance(continuation, dict) and isinstance(continuation.get("path"), str) and
                Path(continuation["path"]).is_absolute() and host.HASH.fullmatch(continuation.get("sha256", "")) and
                m.get("scientific_generation_requests") == 740,
                "Continuation must bind the exact735-new plus five-reused scientific740 plan")
    plan = host.load_ref(m["request_plan"])
    protocol.validate_full(plan)
    external = host.load_ref(package["verification"])
    require(external.get("status") == "independently_verified_h100_generation" and
            external.get("generation_requests") == 740 and external.get("gpu_release_verified") is True and
            external.get("process_release_verified") is True and
            external.get("manifest_sha256") == package["manifest"]["sha256"] and
            external.get("request_ids") == [r["request_id"] for r in plan["requests"]],
            "Full external740 terminal/release proof is required before outcome access")
    if continuation is not None:
        require(external.get("new_generation_requests") == 735 and external.get("reused_generation_requests") == 5 and
                external.get("continuation") == continuation,
                "Continuation proof lacks exact physical/reused counts and source reference")
    fresh = host.verify(package["manifest"]["path"], package["manifest"]["sha256"])
    require(fresh == external, "Fresh complete generation verification differs from saved proof")
    worker_paths = [str(Path(m["output"]) / w["name"] / "results.jsonl") for w in m["workers"]]
    expected_paths = worker_paths
    if continuation is not None:
        require([x["path"] for x in fresh.get("physical_result_files", [])] == worker_paths and len(worker_paths) == 8,
                "Continuation physical result paths are not the exact eight producer workers")
        expected_paths = [str(Path(m["output"]) / "logical_results.jsonl")]
    require([x["path"] for x in fresh["result_files"]] ==
            expected_paths,
            "Generation result paths are not exact producer worker paths")
    return m, plan, fresh


def collect_generation(m, plan, proof):
    """Called only after generation_context positively verifies the entire union."""
    rows = [row for binding in proof["result_files"] for row in json_rows(binding, immutable=False)]
    by_id = unique(rows, "request_id")
    require(len(rows) == (7240 if "coarse_layer_map" in plan else 740) and set(by_id) == {r["request_id"] for r in plan["requests"]}, "Generation coverage differs")
    ordered = []
    for request in plan["requests"]:
        row = by_id[request["request_id"]]
        require(all(row.get(k) == v for k, v in request.items()), "Generation request was changed")
        ordered.append(row)
    return ordered


def write_lines(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write((canonical(row) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    Path(path).chmod(0o400)


def build(spec):
    require(set(spec) == {"run_token", "stage", "generation", "helper_qualification"}, "Unexpected evaluation builder fields")
    gm, plan, proof = generation_context(spec["generation"])
    source = Path(gm["source_root"]); check_sources(source)
    profile = host.load_ref(gm["host_profile"])
    require(set(range(112, 128)) <= set(profile["available_cpus"]), "CPU evaluation affinity is unavailable in qualified profile")
    qualification = host.load_ref(spec["helper_qualification"])
    check_helper_qualification(qualification)
    stage = host.safe_path(spec["stage"], host.OUTPUTS)
    require(re.fullmatch(r"[a-z][a-z0-9-]{4,100}", spec["run_token"]) and
            stage.name == spec["run_token"] and not stage.exists(), "Evaluation needs a fresh exact-token stage")
    rows = collect_generation(gm, plan, proof)
    prepared = json_rows(ref(plan["prepared_records"])); dataset = json_rows(ref(plan["dataset"]))
    is_coarse="coarse_layer_map" in plan
    (coarse.validate_inputs if is_coarse else protocol.validate_inputs)(plan, prepared, dataset)
    base = base_module(); pmap=unique(prepared,"record_id")
    dmap=coarse.dataset_map(dataset) if is_coarse else unique(dataset,"id")
    for row in rows:
        key=(row["environment"],str(row["problem_id"])) if is_coarse else str(row["problem_id"])
        base.validate_generation(row, pmap[row["record_id"]], dmap[key])
    # All source, input, complete-producer and classifier qualification gates precede publication.
    stage.mkdir(parents=True, mode=0o700)
    inputs = stage / "input"; inputs.mkdir(mode=0o700)
    (stage / "control").mkdir(mode=0o700)
    for name, path in (("prepared_records.jsonl", plan["prepared_records"]), ("dataset.jsonl", plan["dataset"])):
        payload = host.read_ref(ref(path), max_bytes=128 << 20)
        with (inputs / name).open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        (inputs / name).chmod(0o400)
    write(inputs / "request_plan.json", plan, immutable=True)
    write_lines(inputs / "generation.jsonl", rows)
    for index in range(8):
        write_lines(inputs / f"worker_{index:02d}.jsonl", rows[index::8])
    source_files = host.inventory(source)
    m = {"schema_version": 1, "purpose": coarse.PURPOSE if is_coarse else PURPOSE, "host": host.HOST, "instance_id": host.INSTANCE,
         "uid": host.UID, "run_token": spec["run_token"], "stage": str(stage),
         "output": str(stage / "results"), "control": str(stage / "control"),
         "source_root": str(source), "source_files": source_files, "python": gm["python"],
         "managed_runtime": sandbox.MANAGED_RUNTIME, "host_profile": gm["host_profile"],
         "source_qualification": gm["source_qualification"], "generation": spec["generation"],
         "helper_qualification": spec["helper_qualification"], "input_files": host.inventory(inputs),
         "request_ids": [r["request_id"] for r in rows], "limits": LIMITS, "cpu_set": CPU_SET,
         "legacy_policy": repair.LEGACY_POLICY, "helper_policy": repair.POLICY,
         "whole_program_gt_unchanged": True, "training": False, "generation_requests": 0,
         "untouched_test_requests": 0, "durable_deadline": host.DURABLE_AT}
    write(stage / "reviewed_manifest.json", m, immutable=True)
    mr = ref(stage / "reviewed_manifest.json")
    load_manifest(mr["path"], mr["sha256"])
    return {"manifest": mr, "command": service_command(m, mr["path"], mr["sha256"]), "launch_performed": False}


def check_helper_qualification(q):
    require(q.get("status") == "independently_verified_h100_helper_classifier_qualification" and
            q.get("helper_source_sha256") == REPAIR_SHA and q.get("sandbox_source_sha256") == SANDBOX_SHA and
            q.get("cases") == 26 and q.get("all_expected_outcomes_match") is True and
            q.get("whole_program_gt_preserved") is True and q.get("process_release_verified") is True and
            q.get("instance_id") == host.INSTANCE, "Missing same-host completed helper/sandbox qualification")
    inner = host.load_ref(q["qualifier_verification"])
    require(inner.get("status") == "verified_authored_helper_qualification" and inner.get("cases") == 26 and
            inner.get("policy") == repair.POLICY, "Missing exact authored qualifier verification")
    exit_receipt = host.load_ref(q["producer_exit"], immutable=False)
    require(exit_receipt.get("returncode") == 0 and exit_receipt.get("timed_out") is False and
            exit_receipt.get("error_type") is None and exit_receipt.get("child_reaped") is True and exit_receipt.get("remaining_group_pids") == [],
            "Helper qualification producer exit/release is not positive")


def load_manifest(path, digest):
    m = host.load_ref({"path": str(path), "sha256": digest})
    require(m.get("purpose") in (PURPOSE,coarse.PURPOSE) and m.get("schema_version") == 1 and m.get("host") == host.HOST and
            m.get("instance_id") == host.INSTANCE and m.get("uid") == host.UID and m.get("limits") == LIMITS and
            m.get("cpu_set") == CPU_SET and m.get("managed_runtime") == sandbox.MANAGED_RUNTIME and
            m.get("legacy_policy") == repair.LEGACY_POLICY and m.get("helper_policy") == repair.POLICY and
            m.get("whole_program_gt_unchanged") is True and m.get("training") is False and
            m.get("generation_requests") == 0 and m.get("untouched_test_requests") == 0 and
            m.get("durable_deadline") == host.DURABLE_AT, "Evaluation policy or resource bounds changed")
    stage = host.safe_path(m["stage"], host.OUTPUTS, exists=True)
    require(stage.name == m["run_token"] and Path(path) == stage / "reviewed_manifest.json" and
            m["output"] == str(stage / "results") and m["control"] == str(stage / "control"), "Evaluation paths differ")
    check_sources(m["source_root"])
    require(host.inventory(m["source_root"]) == m["source_files"], "Evaluation source snapshot changed")
    gm, plan, proof = generation_context(m["generation"])
    require((m["purpose"]==coarse.PURPOSE)==("coarse_layer_map" in plan),"Evaluation purpose/context mismatch")
    require(all(m[k] == gm[k] for k in ("python", "source_root", "source_qualification", "host_profile")),
            "Evaluation runtime/source differs from the qualified generation")
    check_helper_qualification(host.load_ref(m["helper_qualification"]))
    inputs = stage / "input"
    require(host.inventory(inputs) == m["input_files"] and len(m["input_files"]) == 12, "Evaluation input inventory differs")
    require(host.load_ref(ref(inputs / "request_plan.json")) == plan and
            sha(inputs / "prepared_records.jsonl") == plan["prepared_records_sha256"] and
            sha(inputs / "dataset.jsonl") == plan["dataset_sha256"] and
            m["request_ids"] == [r["request_id"] for r in plan["requests"]], "Evaluation scientific input identity differs")
    rows = json_rows(ref(inputs / "generation.jsonl"),coarse_merged=m["purpose"]==coarse.PURPOSE)
    require(rows == collect_generation(gm, plan, proof), "Copied generation differs from complete producer bytes")
    for index in range(8):
        require(json_rows(ref(inputs / f"worker_{index:02d}.jsonl")) == rows[index::8], "Evaluation shard is altered/partial")
    return m, plan, rows


def inside(requests, prepared, dataset, output, coarse_context=None):
    require(os.environ.get("CODE_EVAL_SANDBOX") == "bwrap" and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            Path("/work/src/evaluate/helpers.py").is_file() and not Path("/scratch").exists() and
            not Path("/home/ubuntu/h100-workspace").exists(), "Worker requires restricted CPU outer sandbox")
    base = base_module(); output = Path(output)
    require(not output.exists(), "Worker output must be fresh; no implicit rerun/resume")
    rs = list(base.jsonl(requests)); ps = unique(list(base.jsonl(prepared)), "record_id")
    dataset_rows=list(base.jsonl(dataset)); unique(rs,"request_id")
    is_coarse=coarse_context is not None
    ds=coarse.dataset_map(dataset_rows) if is_coarse else unique(dataset_rows,"id")
    if is_coarse:
        plan=json.loads(Path(coarse_context).read_bytes())
        require(sha(prepared)==plan["prepared_records_sha256"] and sha(dataset)==plan["dataset_sha256"],"Coarse sandbox inputs changed")
        coarse.validate_inputs(plan,list(ps.values()),dataset_rows)
        match=re.fullmatch(r"worker_(0[0-7])\.jsonl",Path(requests).name)
        require(match is not None and 0<=int(match.group(1))<8 and len(rs)==905,"Only exact905-row coarse shards are accepted")
        expected=plan["requests"][int(match.group(1))::8]
        require(all(all(row.get(k)==v for k,v in request.items()) for row,request in zip(rs,expected)),"Coarse sandbox shard differs")
    else:require(92 <= len(rs) <= 93, "Only frozen full740 worker shards are accepted")
    for row in rs:
        key=(row["environment"],str(row["problem_id"])) if is_coarse else str(row["problem_id"])
        base.validate_generation(row, ps[row["record_id"]], ds[key])
    evaluator = base.make_repository_evaluator(dataset)
    installation = sandbox.install_bounded_evaluator(); base.install_count_payload_guard(installation)
    output.mkdir(mode=0o700)
    try:
        for request in rs:
            key=(request["environment"],str(request["problem_id"])) if is_coarse else str(request["problem_id"])
            row = repair.evaluate_one(request, ps[request["record_id"]], ds[key], evaluator, installation)
            if is_coarse:row.update({k:request[k] for k in coarse.EXTRA_IDENTITY})
            base.append(output / "records.jsonl", row)
            require(row["evaluation_status"] != "infrastructure_failure" and
                    row["helper_aware_evaluation"]["status"] != "infrastructure_failure", "Evaluation infrastructure failed; partial evidence retained")
        value = {"status": "succeeded", "records": len(rs), "requests_sha256": sha(requests),
                 "records_sha256": sha(output / "records.jsonl"), "legacy_evaluate_sha256": EVALUATE_SHA,
                 "helper_source_sha256": REPAIR_SHA, "sandbox_source_sha256": SANDBOX_SHA,
                 "whole_program_gt_unchanged": True, "transport": installation.report()}
        write(output / "SUCCESS.json", value)
        return value
    finally:
        installation.restore()


def check_running(m):
    require(socket.gethostname() == host.HOST and os.getuid() == host.UID and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Wrong CPU execution host/environment")
    require(time.time() + LIMITS["systemd_runtime_seconds"] + 900 < host.DEADLINE, "Insufficient time for durable verification")
    fields = host.unit_state(m["run_token"])
    require(fields.get("ActiveState") == "active" and fields.get("SubState") == "running" and
            fields.get("MainPID") == str(os.getpid()) and fields.get("InvocationID") == os.environ.get("INVOCATION_ID") and
            re.fullmatch(r"[0-9a-f]{32}", fields.get("InvocationID", "")) and
            host.duration_seconds(fields.get("RuntimeMaxUSec", "")) == LIMITS["systemd_runtime_seconds"] and
            fields.get("MemoryMax") == str(LIMITS["memory_bytes"]) and fields.get("TasksMax") == "512" and
            fields.get("KillMode") == "control-group", "Exact CPU service bounds/identity are missing")
    require(set(os.sched_getaffinity(0)) == set(range(112, 128)), "CPU service affinity differs")
    require(shutil.disk_usage(m["stage"]).free >= 64 << 30, "Insufficient evaluator disk space")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    require(available >= 192 << 30 and os.getloadavg()[0] <= 128, "Unsafe evaluator host RAM/load")
    group = host.cgroup_path(fields["ControlGroup"], m["run_token"])
    require(os.getpid() in host.cgroup_processes(group), "Controller is not in its exact service cgroup")
    return {"fields": fields, "main_identity": host.process_info(os.getpid())}


def validate_evaluations(rows, generations):
    require(len(rows) == len(generations), "Partial evaluation rows")
    unique(rows, "request_id")
    for row, generation in zip(rows, generations):
        identity=(*IDENTITY,*coarse.EXTRA_IDENTITY) if "environment" in generation else IDENTITY
        require(all(row.get(k) == generation.get(k) for k in identity) and
                row.get("generation") == generation["result"] and
                row.get("generation_sha256") == hashlib.sha256(canonical(generation).encode()).hexdigest(),
                "Evaluation generation/identity changed")
        require(row.get("evaluation_status") in ("evaluated", "suspicious_or_unknown"), "Evaluation infrastructure failure/unknown status")
        side = row.get("helper_aware_evaluation", {})
        require(side.get("policy") == repair.POLICY and side.get("legacy_policy") == repair.LEGACY_POLICY and
                side.get("legacy_row_unchanged") is True and side.get("whole_program_gt_reused") is True and
                side.get("status") in ("evaluated", "suspicious_or_unknown", "unsupported", "unchanged", "legacy_unknown"),
                "Missing or failed versioned helper result")
        for view in (row["metrics"], side["metrics"]):
            require(all(type(view.get(k)) is bool or view.get(k) is None for k in (*repair.BEHAVIOR, "ground_truth_correctness")) and
                    view.get("completion_length") == len(generation["result"]["completion_token_ids"]), "Malformed primitive/length metrics")
        require(side["metrics"]["ground_truth_correctness"] == row["metrics"]["ground_truth_correctness"], "Helper policy changed whole-program GT")


def replay_labels(rows):
    """Recompute classifications from saved primitives; never execute a probe.

The explicit replay object returns the two already-recorded primitive results to
the unchanged classifier. The request list must match its independently rebuilt
static probe plan exactly. No CodeEvaluator is constructed here.
"""
    base = base_module()
    for row in rows:
        legacy = base.analyze_repository_result(row["repository_evaluation"], row["transport"])
        legacy["metrics"]["completion_length"] = row["metrics"]["completion_length"]
        require(all(row.get(k) == v for k, v in legacy.items()), "Legacy saved primitive classification differs")
        side = row["helper_aware_evaluation"]
        delta = side["transport"]
        require(set(delta) == set(repair.TRANSPORT) and all(type(v) is int and v >= 0 for v in delta.values()),
                "Malformed helper transport counters")
        state = {"calls": 0}
        plan = repair.build_probe_plan(row["repository_evaluation"])
        def replay(requests):
            require(state["calls"] == 0 and plan["status"] == "patched" and requests == plan["requests"],
                    "Saved probe plan differs from static reconstruction")
            state["calls"] += 1
            values = side["corrected_probe_results"]
            require(isinstance(values, dict) and set(values) == {"canonical", "arbitrary"} and
                    all(isinstance(v, dict) for v in values.values()), "Missing two saved corrected primitive results")
            return copy.deepcopy([values["canonical"], values["arbitrary"]])
        installation = SimpleNamespace(report=lambda: dict(delta) if state["calls"] else {k: 0 for k in repair.TRANSPORT})
        evaluator = SimpleNamespace(evaluator=SimpleNamespace(batch_evaluate=replay))
        require(repair.score_legacy_row(row, evaluator, installation) == side,
                "Helper sidecar differs from independent saved-primitive/classification replay")


def output_inventory(output):
    for path in Path(output).rglob("*"):
        require(not path.is_symlink(), "Evaluation output contains a symlink")
    return {str(p.relative_to(output)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in sorted(Path(output).rglob("*")) if p.is_file() and p != Path(output) / "artifact_manifest.json"}


def run(path, digest):
    m, plan, generations = load_manifest(path, digest)
    check_runtime(m, plan)
    started = check_running(m)
    require(not Path(m["output"]).exists(), "Evaluation already attempted; preserve outputs")
    write(Path(m["control"]) / "service_started.json", {**started, "manifest_sha256": digest}, immutable=True)
    output = Path(m["output"]); output.mkdir(mode=0o700)
    inputs = Path(m["stage"]) / "input"
    def worker(index):
        name = f"worker_{index:02d}"; destination = output / name; destination.mkdir(mode=0o700)
        result = sandbox.run_outer(source_dir=m["source_root"], input_dir=inputs, output_dir=destination,
            venv_dir=Path(m["python"]).parent.parent, workers=2, wall_timeout=7200,
            python_args=["-B", "-m", "infra.gpu03.direction_discovery.h100_evaluate", "--inside",
                "--requests", f"/input/{name}.jsonl", "--prepared", "/input/prepared_records.jsonl",
                "--dataset", "/input/dataset.jsonl", "--output", "/output/evaluation"] +
                (["--coarse-context","/input/request_plan.json"] if m["purpose"]==coarse.PURPOSE else []))
        write(destination / "outer_transport.json", result)
        require(result["returncode"] == 0, "Outer CPU worker failed; all evidence retained")
        success = host.load_ref(ref(destination / "evaluation/SUCCESS.json"), immutable=False)
        rows = json_rows(ref(destination / "evaluation/records.jsonl"), immutable=False)
        validate_evaluations(rows, generations[index::8])
        require(success["status"] == "succeeded" and success["records"] == len(rows) and
                success["requests_sha256"] == sha(inputs / f"{name}.jsonl") and
                success["records_sha256"] == sha(destination / "evaluation/records.jsonl"), "Worker terminal evidence differs")
        return index, rows
    try:
        shards = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(worker, index) for index in range(8)]):
                index, rows = future.result(); shards[index] = rows
        by_id = unique([r for i in range(8) for r in shards[i]], "request_id")
        rows = [by_id[r["request_id"]] for r in generations]
        validate_evaluations(rows, generations)
        write_lines(output / "evaluations.jsonl", rows)
        summary = {"status": "succeeded", "records": len(generations), "manifest_sha256": digest,
                   "invocation_id": started["fields"]["InvocationID"], "evaluations": ref(output / "evaluations.jsonl")}
        write(output / "SUCCESS.json", summary)
        inventory = output_inventory(output)
        require(sum(x["size_bytes"] for x in inventory.values()) <= LIMITS["maximum_output_bytes"], "Evaluation output exceeded cap")
        write(output / "artifact_manifest.json", {"algorithm": "sha256", "files": inventory}, immutable=True)
        return summary
    except BaseException as exc:
        write(output / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc)[:2000]})
        raise


def receipt(path, digest):
    m = host.load_ref({"path": str(path), "sha256": digest})
    require(m["purpose"] in (PURPOSE,coarse.PURPOSE) and os.getuid() == host.UID, "Wrong CPU receipt context")
    write(Path(m["control"]) / "supervisor_exit.json", {"run_token": m["run_token"], "manifest_sha256": digest,
          "invocation_id": os.environ.get("INVOCATION_ID"), "service_result": os.environ.get("SERVICE_RESULT"),
          "exit_code_kind": os.environ.get("EXIT_CODE"), "exit_status": os.environ.get("EXIT_STATUS"), "at": time.time()})


def verify(path, digest):
    m, plan, generations = load_manifest(path, digest)
    start = host.load_ref(ref(Path(m["control"]) / "service_started.json"))
    terminal = host.load_ref(ref(Path(m["control"]) / "supervisor_exit.json"), immutable=False)
    require(start["manifest_sha256"] == digest and terminal.get("manifest_sha256") == digest and
            terminal.get("run_token") == m["run_token"] and terminal.get("invocation_id") == start["fields"]["InvocationID"] and
            terminal.get("service_result") == "success" and terminal.get("exit_code_kind") == "exited" and
            terminal.get("exit_status") == "0", "CPU evaluation has not successfully exited")
    current = host.unit_state(m["run_token"])
    require(current.get("ActiveState") == "inactive" and current.get("MainPID") == "0" and
            current.get("InvocationID") in ("", terminal["invocation_id"]) and
            not host.same_process(start["main_identity"]) and
            not host.cgroup_processes(host.cgroup_path(start["fields"]["ControlGroup"], m["run_token"])),
            "CPU evaluator process group is not positively released")
    output = Path(m["output"])
    require(not (output / "FAILURE.json").exists(), "CPU producer failure exists")
    artifact = host.load_ref(ref(output / "artifact_manifest.json"))
    require(artifact == {"algorithm": "sha256", "files": output_inventory(output)}, "Evaluation output bytes changed")
    rows = json_rows(ref(output / "evaluations.jsonl"),coarse_merged=m["purpose"]==coarse.PURPOSE)
    validate_evaluations(rows, generations)
    replay_labels(rows)
    for index in range(8):
        worker = output / f"worker_{index:02d}"
        transport = host.load_ref(ref(worker / "outer_transport.json"), immutable=False)
        require(transport["returncode"] == 0, "Worker transport was not successful")
        shard = json_rows(ref(worker / "evaluation/records.jsonl"), immutable=False)
        require(shard == rows[index::8], "Merged evaluations differ from exact retained worker rows")
        worker_success = host.load_ref(ref(worker / "evaluation/SUCCESS.json"), immutable=False)
        require(worker_success.get("status") == "succeeded" and worker_success.get("records") == len(shard) and
                worker_success.get("records_sha256") == sha(worker / "evaluation/records.jsonl") and
                worker_success.get("requests_sha256") == sha(Path(m["stage"]) / f"input/worker_{index:02d}.jsonl") and
                worker_success.get("legacy_evaluate_sha256") == EVALUATE_SHA and
                worker_success.get("helper_source_sha256") == REPAIR_SHA and
                worker_success.get("sandbox_source_sha256") == SANDBOX_SHA and
                worker_success.get("whole_program_gt_unchanged") is True, "Worker success/source proof differs")
    success = host.load_ref(ref(output / "SUCCESS.json"), immutable=False)
    require(success == {"status": "succeeded", "records": len(generations), "manifest_sha256": digest,
                       "invocation_id": terminal["invocation_id"], "evaluations": ref(output / "evaluations.jsonl")}, "Producer summary differs")
    require(artifact == {"algorithm": "sha256", "files": output_inventory(output)}, "Evaluation changed during verification")
    return {"status": "independently_verified_h100_complete7240_coarse_evaluation" if m["purpose"]==coarse.PURPOSE
            else "independently_verified_h100_complete740_evaluation", "purpose": m["purpose"],
            "manifest_sha256": digest, "request_plan_sha256": sha(Path(m["stage"]) / "input/request_plan.json"),
            "records": len(generations), "exact_request_coverage": True, "process_release_verified": True,
            "evaluations": ref(output / "evaluations.jsonl"), "artifact_manifest": ref(output / "artifact_manifest.json"),
            "service_started": ref(Path(m["control"]) / "service_started.json"),
            "supervisor_exit": ref(Path(m["control"]) / "supervisor_exit.json"),
            "legacy_policy": repair.LEGACY_POLICY, "helper_policy": repair.POLICY,
            "whole_program_gt_unchanged": True, "evaluation_source_sha256": sha(__file__)}


def service_command(m, path, digest):
    script = str(Path(m["source_root"]) / HERE)
    clean = ["/usr/bin/env", "-i", *[k + "=" + v for k, v in host.worker_environment().items()]]
    stop = clean + ["INVOCATION_ID=${INVOCATION_ID}", "SERVICE_RESULT=${SERVICE_RESULT}",
                   "EXIT_CODE=${EXIT_CODE}", "EXIT_STATUS=${EXIT_STATUS}", m["python"], script,
                   "--receipt", "--manifest", str(path), "--manifest-sha256", digest]
    return ["systemd-run", "--user", "--quiet", "--unit", m["run_token"], "--service-type=exec",
            "--property=RuntimeMaxSec=7290", "--property=TimeoutStopSec=90", "--property=KillMode=control-group",
            "--property=MemoryMax=32G", "--property=TasksMax=512", "--property=LimitNOFILE=4096",
            "--property=UMask=0077", "--property=CPUAffinity=" + CPU_SET,
            "--property=ExecStopPost=" + shlex.join(stop),
            "--property=StandardOutput=append:" + str(Path(m["control"]) / "supervisor.log"),
            "--property=StandardError=inherit", *clean, "INVOCATION_ID=${INVOCATION_ID}",
            m["python"], script, "--run", "--manifest", str(path), "--manifest-sha256", digest]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("build", "run", "verify", "receipt", "inside"):
        modes.add_argument("--" + name, action="store_true")
    for name in ("spec", "spec-sha256", "manifest", "manifest-sha256", "requests", "prepared", "dataset", "output", "coarse-context"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.build:
        result = build(host.load_ref({"path": args.spec, "sha256": args.spec_sha256}))
    elif args.inside:
        result = inside(args.requests, args.prepared, args.dataset, args.output, args.coarse_context)
    else:
        result = (run if args.run else verify if args.verify else receipt)(args.manifest, args.manifest_sha256)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
