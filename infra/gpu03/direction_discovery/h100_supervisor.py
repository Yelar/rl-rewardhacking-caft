"""Closed H100 workstation adapter for the matched no-loophole experiment.

The old shared-host supervisor and the numerical engine are unchanged. This
adapter has no AWS/scheduler/credential operation. A separately locked ledger
must admit the exact immutable manifest before this launcher can consume it.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "activation_dataset"))
sys.dont_write_bytecode = True
from infra.gpu03.direction_discovery import h100_qualification as numerical

HOST = "ip-172-31-66-181"
UID, OWNER = 1000, "ubuntu"
INSTANCE = "i-00000000000000002"
INSTANCE_TYPE = "p5.48xlarge"
OWNER_TOKEN = "codex-sky-h100-block-20260908-1530"
ORIGINAL_MODEL_INVENTORY_SHA = "cc515534a31aca0ea09d71fc0243abbe397c85fcd76e5209b686e4f3f68fecb1"
WORK = Path("/home/ubuntu/h100-workspace/work")
OUTPUTS = Path("/home/ubuntu/h100-workspace/outputs")
DURABLE_AT = "2026-09-09T10:45:00Z"
SHUTDOWN_AT = "2026-09-09T11:00:00Z"
DEADLINE = datetime.fromisoformat(DURABLE_AT.replace("Z", "+00:00")).timestamp()
PURPOSE = "h100_no_loophole_generation_v1"
HERE = "infra/gpu03/direction_discovery/h100_supervisor.py"
ENGINE = "infra/gpu03/direction_discovery/engine.py"
FROZEN = {ENGINE: "8c4723a20ac341e3d92c7ad7bc514840899d1d4ec51efc53e2a089d1fa8852cf",
          "infra/gpu03/direction_discovery/intervention.py": "6eed46f798a00dfd6cab9ee91a035853cf9685cd5cf6a14c1efb2b866a3f3529"}
VERSIONS = {"peft": "0.17.1", "safetensors": "0.6.2", "tokenizers": "0.22.1",
            "torch": "2.8.0+cu128", "transformers": "4.57.1", "vllm": "0.11.0"}
PHASES = {"h100_numerical_qualification": (6, 1, 600), "h100_no_loophole_capability": (740, 8, 7200)}
REUSABLE_QUALIFIER_SOURCES = {HERE: "ce6765cc941a143fc880388fb6fd6f281ea280b514720fe078a1cf75e10c042e",
    "infra/gpu03/direction_discovery/h100_qualification.py": "50124d30b7ac68e50b85239a8f05dbe2294f28c07cecd066e6d09c356e7a4556"}
HASH = re.compile(r"[0-9a-f]{64}")
INTERVAL = 5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path)
    return {"path": str(p), "sha256": sha(p), "size_bytes": p.stat().st_size}


def exclusive_json(path, value, *, immutable=False):
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(canonical(value) + "\n"); f.flush(); os.fsync(f.fileno())
    if immutable:
        Path(path).chmod(0o400)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(canonical(value) + "\n"); f.flush(); os.fsync(f.fileno())


def safe_path(path, root, *, exists=False):
    p = Path(path)
    require(p.is_absolute() and p.is_relative_to(root) and p != root and ".." not in p.parts,
            "Path is outside its reviewed H100 root")
    # All model/source/output paths are work/outputs, never the reviewed scratch symlink.
    for parent in (p, *p.parents):
        if parent == root.parent:
            break
        require(not parent.is_symlink(), "Symlink in staged/output path: " + str(parent))
    if exists:
        require(p.exists(), "Missing bound path: " + str(p))
    return p


def read_ref(value, *, immutable=True, runtime=False, max_bytes=32 << 20):
    require(isinstance(value, dict) and isinstance(value.get("path"), str) and
            HASH.fullmatch(value.get("sha256", "")), "Malformed hash reference")
    p = Path(value["path"])
    require(p.is_absolute() and ".." not in p.parts, "Reference must be an absolute lexical path")
    if not runtime:
        require(not p.is_symlink(), "Bound file is a symlink")
    before = p.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and
            (not immutable or not before.st_mode & 0o222), "Bound file is not owned/immutable")
    require(before.st_size <= max_bytes and ("size_bytes" not in value or value["size_bytes"] == before.st_size),
            "Bound file size differs")
    data = p.read_bytes()
    after = p.stat()
    require((before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns, after.st_ctime_ns) and
            hashlib.sha256(data).hexdigest() == value["sha256"], "Bound file changed")
    return data


def load_ref(value, **kwargs):
    return json.loads(read_ref(value, **kwargs))


def file_bindings(entries, *, runtime_path=None):
    require(isinstance(entries, dict) and entries, "Missing complete file inventory")
    for name, value in entries.items():
        p = Path(name)
        require(p.is_absolute() and HASH.fullmatch(value.get("sha256", "")) and
                type(value.get("size_bytes")) is int and value["size_bytes"] >= 0, "Invalid file binding")
        if name == runtime_path:
            read_ref({"path": name, **value}, immutable=False, runtime=True)
        else:
            safe_path(p, WORK if p.is_relative_to(WORK) else OUTPUTS, exists=True)
            before = p.stat()
            require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and not before.st_mode & 0o222,
                    "Bound source/model/input must be owned immutable regular bytes")
            require(before.st_size == value["size_bytes"] and sha(p) == value["sha256"], "Bound file differs: " + name)
            after = p.stat()
            require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Hash input changed")


def inventory(root):
    files = {}
    for p in sorted(Path(root).rglob("*")):
        require(not p.is_symlink(), "Package contains a symlink")
        if p.is_file():
            require(p.name != ".DS_Store" and p.suffix != ".pyc" and "__pycache__" not in p.parts,
                    "Package contains a cache/nonportable file")
            files[str(p)] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    require(files, "Empty package")
    return files


def source_inventory(root):
    """Allow only reviewed source families; reject private payloads before hashing."""
    root=Path(root)
    allowed_roots=(Path('src'),Path('infra/gpu03'))
    metadata={Path('artifacts/direction_discovery_review_20260907/plan_v1/experiment_plan.json'),
              Path('artifacts/direction_discovery_review_20260907/screening_helper_review_20260907_172600/helper_review.json')}
    allowed_suffixes={'.py','.md','.sh','.jinja2','.yaml','.yml'}
    forbidden={'secrets','secret','private','private-runtime','private_runtime','runtime','venv','.venv',
               '.aws','.ssh','.git','.config','.cache','__pycache__','credentials','credential',
               'wandb_api_key','id_rsa','id_ed25519','.env'}
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        require(not path.is_symlink() and not any(part.lower() in forbidden for part in relative.parts),
                'Private/credential/runtime path in generation source')
        require(not any(re.search(r'(^|[_.-])(api[_-]?key|access[_-]?key|credentials?|secrets?)([_.-]|$)',part.lower())
                        for part in relative.parts), 'Credential-shaped basename in generation source')
        require(relative.parts[:2] != ('infra','skypilot'), 'SkyPilot subtree is excluded from generation source')
        if path.is_dir():
            continue
        require(path.is_file(), 'Nonregular source entry')
        require(relative in metadata or (any(relative.is_relative_to(p) for p in allowed_roots) and
                path.suffix in allowed_suffixes) or relative in {Path('README.md'),Path('AGENTS.md'),Path('infra/__init__.py')},
                'Source path is outside the reviewed allowlist')
    return inventory(root)


def scientific_identity(plan):
    return numerical.digest({k: plan[k] for k in ("requests", "conditions", "sampling", "prepared_records_sha256", "dataset_sha256")})


def profile_context(reference):
    p = load_ref(reference)
    require(p.get("status") == "verified_h100_workstation_profile" and
            (p.get("host"), p.get("user"), p.get("uid"), p.get("instance_id"), p.get("instance_type"), p.get("ownership_token")) ==
            (HOST, OWNER, UID, INSTANCE, INSTANCE_TYPE, OWNER_TOKEN), "Wrong H100 host profile")
    require(p.get("runtime_versions") == VERSIONS and set(range(94, 112)) <= set(p["available_cpus"]),
            "Runtime/CPU profile differs")
    gpu = p["gpu_inventory"]
    require(len(gpu) == 8 and {v["index"] for v in gpu} == set(range(8)) and
            len({v["uuid"] for v in gpu}) == 8 and all("H100" in v["name"] and
            v["memory_total_mib"] >= 80000 and re.fullmatch(r"GPU-[0-9a-f-]{36}", v["uuid"]) for v in gpu),
            "Profile does not bind eight H100-80GB devices")
    life = load_ref(p["lifetime_proof"])
    require(life.get("status") == "verified_h100_user_service_disconnect_survival" and
            life.get("host") == HOST and life.get("uid") == UID and
            all(life.get(k) is True for k in ("linger", "survived_disconnect", "processes_released")),
            "Missing actual SSH-independent service lifetime qualification")
    timers = load_ref(p["deadline_proof"])
    require(timers.get("status") == "verified_h100_existing_deadline_timers" and timers.get("host") == HOST and
            timers.get("final_sync_at") == DURABLE_AT and timers.get("shutdown_at") == SHUTDOWN_AT and
            timers.get("final_sync_active") is True and timers.get("shutdown_active") is True,
            "Existing durable sync/shutdown deadlines are not positively bound")
    return p


def limits_for(phase, runtime_override=None):
    count, workers, runtime = PHASES[phase]
    if runtime_override is not None:
        require(phase == "h100_no_loophole_capability" and type(runtime_override) is int and
                runtime_override in (7200, 21600), "Unapproved generation runtime")
        runtime = runtime_override
    # TimeoutStopSec applies to service shutdown and separately to ExecStopPost.
    # Charge both 90-second windows in addition to the active runtime.
    return {"runtime_seconds": runtime, "systemd_runtime_seconds": runtime,
            "reserved_wall_seconds": runtime + 180, "timeout_stop_seconds": 90,
            "worker_deadline_seconds": runtime - 120, "min_available_ram_gib": 192,
            "max_worker_rss_gib": 128, "cgroup_memory_gib": 160, "min_start_free_disk_gib": 64,
            "min_runtime_free_disk_gib": 32, "max_worker_log_mib": 64, "tasks_max": 512,
            "load_stagger_seconds": 10, "idle_qualification_seconds": 60, "max_host_load1": 128}


def protocol():
    from infra.gpu03.direction_discovery import h100_no_loophole_protocol
    return h100_no_loophole_protocol


def helper_qualification_bindings(m):
    """Reuse completed authored CPU checks; never execute generated programs."""
    # Lazy imports avoid the evaluator's metadata helpers importing this module
    # while it is still being initialized. None of these checks launches work.
    from infra.gpu03.direction_discovery import h100_evaluate as evaluation
    from infra.gpu03.direction_discovery import h100_sandbox_qualification as isolation
    from infra.gpu03.direction_discovery import qualify_helper_aware_v1 as helper
    reference = m["helper_qualification"]
    package = safe_path(Path(reference["path"]).parent, OUTPUTS, exists=True)
    require(Path(reference["path"]) == package / "SUCCESS.json", "Wrong authored helper qualification path")
    qualified = load_ref(reference)
    evaluation.check_helper_qualification(qualified)
    for module in (evaluation, evaluation.repair, evaluation.sandbox, isolation, helper):
        relative = "infra/gpu03/direction_discovery/" + Path(module.__file__).name
        require(sha(Path(m["source_root"]) / relative) == sha(module.__file__),
                "Generation helper/isolation qualification source differs")
    require(sha(evaluation.repair.__file__) == evaluation.REPAIR_SHA and
            sha(evaluation.sandbox.__file__) == evaluation.SANDBOX_SHA,
            "Running helper/sandbox source differs from qualified policy")
    isolation_ref = qualified["isolation_qualification"]
    require(Path(isolation_ref["path"]) == package / "isolation_verification.json",
            "Isolation proof is outside the completed helper qualification")
    proof = load_ref(isolation_ref)
    exit_receipt = load_ref(proof["producer_exit"], immutable=False)
    require(proof.get("process_release_verified") is True and
            exit_receipt.get("returncode") == 0 and exit_receipt.get("timed_out") is False and
            exit_receipt.get("error_type") is None and exit_receipt.get("child_reaped") is True and
            exit_receipt.get("remaining_group_pids") == [], "Isolation producer exit/release is not positive")
    artifact_dir = package / "sandbox-output" / "sandbox-qualification"
    fresh = isolation.verify(artifact_dir, proof["artifact_manifest_sha256"])
    require({k:v for k,v in proof.items() if k not in ("producer_exit", "process_release_verified")} == fresh,
            "Isolation qualification differs from independently checked authored bytes")
    helper_dir=package/'sandbox-output/helper-qualification'
    stored_helper=load_ref(qualified['qualifier_verification'])
    require(helper.verify(helper_dir,stored_helper['artifact_manifest_sha256']) == stored_helper,
            'Helper qualification differs from independently checked authored bytes')
    return [Path(reference["path"]), Path(isolation_ref["path"]),
            Path(qualified["qualifier_verification"]["path"]), Path(qualified["producer_exit"]["path"]),
            Path(proof["producer_exit"]["path"]), *sorted(artifact_dir.iterdir()), *sorted(helper_dir.iterdir())]


def request_context(m):
    plan = load_ref(m["request_plan"])
    protocol().validate_full(plan)
    protocol().validate_source(plan, Path(m["source_root"]))
    requests = numerical.qualification_requests(plan) if m["phase"] == "h100_numerical_qualification" else plan["requests"]
    if m.get("continuation") is not None:
        reused = {r["request_id"] for r in continuation_rows(m, plan)}
        requests = [r for r in requests if r["request_id"] not in reused]
    return plan, requests, protocol().physical_conditions(plan)


def continuation_rows(m, plan):
    """Validate released failed-qualification bytes and map its exact first five cells."""
    require(m["phase"] == "h100_no_loophole_capability", "Reuse is only for the main scientific comparison")
    evidence = load_ref(m["continuation"])
    require(evidence.get("status") == "independently_verified_h100_failed_qualification_reuse" and
            evidence.get("retained_generation_requests") == 5 and
            evidence.get("process_release_verified") is True and evidence.get("gpu_release_verified") is True,
            "Failed qualification lacks independent complete release evidence")
    qref = evidence["manifest"]; q = load_ref(qref)
    require(q.get("purpose") == PURPOSE and q.get("host") == HOST and q.get("uid") == UID and
            q.get("phase") == "h100_numerical_qualification" and q.get("generation_requests") == 6 and
            q.get("tf_requests") == 0 and q.get("limits") == limits_for("h100_numerical_qualification") and
            q.get("scientific_identity_sha256") == scientific_identity(plan), "Prior qualification science differs")
    stage = safe_path(q["stage"], OUTPUTS, exists=True)
    require(stage.name == q["run_token"] and qref["path"] == str(stage/"reviewed_manifest.json") and
            q["output"] == str(stage/"results") and q["control"] == str(stage/"control") and
            q["runtime"] == str(stage/"runtime"), "Prior qualification paths differ")
    require(all(q[k] == m[k] for k in ("host_profile", "python", "model_snapshot", "checkpoint", "helper_qualification")),
            "Reuse model/runtime/profile/helper identity differs")
    original_source = source_inventory(safe_path(q["source_root"], WORK, exists=True))
    source_proof = load_ref(q["source_qualification"])
    require(source_proof.get("status") == "verified_h100_step1_cpu_source" and
            source_proof.get("source_root") == q["source_root"] and source_proof.get("source_files") == original_source and
            source_proof.get("producer_exit_status") == 0 and source_proof.get("process_release_verified") is True and
            source_proof.get("tests_failed") == 0 and source_proof.get("tests_passed", 0) > 0,
            "Prior qualification source proof differs")
    require(all(original_source.get(str(Path(q["source_root"])/name), {}).get("sha256") == digest
                for name, digest in {**FROZEN, **REUSABLE_QUALIFIER_SOURCES}.items()) and
            all(q["bound_files"].get(path) == value for path,value in original_source.items()),
            "Prior qualification frozen engine/source changed")
    original_plan = load_ref(q["request_plan"]); protocol().validate_full(original_plan)
    require(scientific_identity(original_plan) == scientific_identity(plan) and
            q["scientific_conditions"] == plan["conditions"] and
            q["candidate_map"] == plan["no_loophole_capability"]["h100_portability"]["candidate_map"],
            "Prior qualification prompt/request/condition lineage differs")
    require(len(q["workers"]) == len(q["gpu_ids"]) == 1 and q["gpu_ids"][0] in range(8), "Wrong prior allocation")
    worker = q["workers"][0]
    expected = expected_task(q, original_plan, numerical.qualification_requests(original_plan),
                             protocol().physical_conditions(original_plan), 0)
    task = load_ref({"path": worker["task"], **q["bound_files"][worker["task"]]})
    require(task == expected and worker["name"] == "worker_00" and worker["requests"] == 6 and
            worker["gpu_id"] == q["gpu_ids"][0], "Prior worker/task differs")
    require(all(q["bound_files"].get(path) == value for path,value in load_ref(m["host_profile"])["model_files"].items()),
            "Prior model content bindings differ")
    for key, path in {"results": stage/"results/worker_00/results.jsonl",
                      "service_started": stage/"control/service_started.json",
                      "supervisor_exit": stage/"control/supervisor_exit.json", "failure": stage/"results/FAILURE.json"}.items():
        require(evidence[key]["path"] == str(path), "Reuse evidence path differs")
    start = load_ref(evidence["service_started"], immutable=False)
    terminal = load_ref(evidence["supervisor_exit"], immutable=False)
    failure = load_ref(evidence["failure"], immutable=False)
    require(terminal.get("run_token") == failure.get("run_token") == q["run_token"] and
            terminal.get("manifest_sha256") == failure.get("manifest_sha256") == qref["sha256"] and
            terminal.get("invocation_id") == failure.get("invocation_id") == start["fields"]["InvocationID"] and
            terminal.get("service_result") in ("timeout", "signal", "exit-code") and
            failure.get("cleanup", {}).get("owned_workers_released") is True and
            failure.get("cleanup", {}).get("gpu_release_verified") is True and
            not same_process(start["main_identity"]) and
            not cgroup_processes(cgroup_path(start["fields"]["ControlGroup"], q["run_token"])),
            "Prior failed service/process release is not exact")
    require(load_ref(ref(stage/"results/worker_00/task.json"), immutable=False) == task,
            "Prior executed task snapshot differs")
    data = read_ref(evidence["results"], immutable=False, max_bytes=128 << 20)
    require(data.endswith(b"\n") and all(x.strip() for x in data.splitlines()), "Incomplete retained journal bytes")
    rows = numerical.reuse_first_five(original_plan, [json.loads(x) for x in data.splitlines()])
    prepared = {r["record_id"]:r for r in (json.loads(x) for x in read_ref(
        {"path": plan["prepared_records"], "sha256": plan["prepared_records_sha256"]}).splitlines())}
    for index,row in enumerate(rows):
        require(row.get("original_class") == prepared[row["record_id"]]["outcome_presence_class"], "Reused carrier provenance differs")
        row["reuse_provenance"] = {"continuation": m["continuation"], "raw_results": evidence["results"],
                                  "raw_line_index": index, "qualification_manifest": qref}
    return rows


def expected_task(m, plan, requests, physical, index):
    gpu = m["gpu_ids"][index]
    return {"run_token": m["run_token"], "worker_name": f"worker_{index:02d}", "gpu_id": gpu,
            "mode": "generate", "output": str(Path(m["output"]) / f"worker_{index:02d}"),
            "prepared_records": plan["prepared_records"], "conditions": physical,
            "requests": requests[index::len(m["gpu_ids"])], "sampling": plan["sampling"],
            "model_snapshot": m["model_snapshot"], "checkpoint": m["checkpoint"],
            "attention_policy": "exclusive_math", "deadline_seconds": m["limits"]["worker_deadline_seconds"]}


def build(spec):
    """Metadata-only exclusive publication; this creates no admission or service."""
    m = copy.deepcopy(spec)
    require(m.get("phase") in PHASES, "Unreviewed H100 phase")
    stage = safe_path(m["stage"], OUTPUTS)
    require(not stage.exists(), "Fresh stage required")
    actual_source=source_inventory(safe_path(m['source_root'],WORK,exists=True))
    require(m['source_files']==actual_source, 'Builder source inventory is not the exact allowed source tree')
    m.update(schema_version=1, purpose=PURPOSE, host=HOST, uid=UID, owner=OWNER,
             output=str(stage / "results"), runtime=str(stage / "runtime"), control=str(stage / "control"),
             limits=limits_for(m["phase"], m.get("generation_runtime_seconds")), supervisor_cpu_set="94", durable_deadline=DURABLE_AT,
             shutdown_deadline=SHUTDOWN_AT)
    helper_bindings = helper_qualification_bindings(m)
    plan, requests, physical = request_context(m)
    m["scientific_conditions"] = copy.deepcopy(plan["conditions"])
    m["candidate_map"] = copy.deepcopy(plan["no_loophole_capability"]["h100_portability"]["candidate_map"])
    m["scientific_identity_sha256"] = scientific_identity(plan)
    m["generation_requests"], m["tf_requests"] = len(requests), 0
    if m.get("continuation") is not None:
        m["scientific_generation_requests"] = len(plan["requests"])
    m["bound_files"] = {**m.pop("source_files"), **m.pop("model_files")}
    for p in [Path(m["request_plan"]["path"]), *protocol().bindings(plan), *helper_bindings]:
        m["bound_files"][str(p)] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    for r in (m["host_profile"], m["authorization"], *([m["continuation"]] if m.get("continuation") is not None else [])):
        p = Path(r["path"]); m["bound_files"][str(p)] = {"sha256": r["sha256"], "size_bytes": p.stat().st_size}
    m["workers"] = []
    tasks = []
    for i, gpu in enumerate(m["gpu_ids"]):
        task = expected_task(m, plan, requests, physical, i)
        path = stage / "tasks" / (task["worker_name"] + ".json")
        tasks.append((path, task))
        m["workers"].append({"name": task["worker_name"], "gpu_id": gpu,
            "cpu_set": f"{96 + 2 * gpu},{97 + 2 * gpu}", "task": str(path),
            "command": [m["python"], str(Path(m["source_root"]) / ENGINE), "--task", str(path)],
            "success_file": task["worker_name"] + "/SUCCESS.json", "requests": len(task["requests"])})
    validate_manifest_object(m, check_tasks=False)
    stage.mkdir(mode=0o700); (stage / "tasks").mkdir(mode=0o700)
    for path, task in tasks:
        exclusive_json(path, task, immutable=True)
        m["bound_files"][str(path)] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    validate_manifest_object(m)
    path = stage / "reviewed_manifest.json"
    exclusive_json(path, m, immutable=True)
    return ref(path)


def validate_manifest_object(m, *, check_tasks=True):
    require(m.get("schema_version") == 1 and m.get("purpose") == PURPOSE and m.get("phase") in PHASES,
            "Wrong H100 manifest")
    require((m.get("host"), m.get("owner"), m.get("uid")) == (HOST, OWNER, UID), "Wrong host/user")
    require(re.fullmatch(r"[a-z][a-z0-9-]{4,100}", m["run_token"]), "Invalid exact run token")
    stage = safe_path(m["stage"], OUTPUTS)
    require(stage.name == m["run_token"], "Stage basename must be the exact run token")
    require(all(m[k] == str(stage / leaf) for k, leaf in
                (("output", "results"), ("runtime", "runtime"), ("control", "control"))), "Run paths differ")
    source = safe_path(m["source_root"], WORK, exists=True)
    actual_source = source_inventory(source)
    for key in ("model_snapshot", "checkpoint"):
        safe_path(m[key], WORK, exists=True)
    require(Path(m["python"]).is_relative_to(WORK) and Path(m["python"]).is_absolute(), "Wrong Python deployment")
    count, workers, _ = PHASES[m["phase"]]
    if m.get("continuation") is not None:
        require(m["phase"] == "h100_no_loophole_capability" and m.get("scientific_generation_requests") == 740,
                "Continuation must retain the full scientific740")
        count = 735
    ids = m["gpu_ids"]
    require(isinstance(ids, list) and len(ids) == workers and len(set(ids)) == workers and
            all(type(x) is int and 0 <= x < 8 for x in ids) and ids == sorted(ids), "Unreviewed GPU allocation")
    require(m["limits"] == limits_for(m["phase"], m.get("generation_runtime_seconds")) and m["supervisor_cpu_set"] == "94" and
            m["generation_requests"] == count and m["tf_requests"] == 0, "Phase limits/counts changed")
    require(m["durable_deadline"] == DURABLE_AT and m["shutdown_deadline"] == SHUTDOWN_AT and
            type(m["postprocessing_seconds"]) is int and m["postprocessing_seconds"] >= 900, "Missing durable deadline reserve")
    helper_bindings = helper_qualification_bindings(m)
    p = profile_context(m["host_profile"])
    require(p["python"] == m["python"], "Profile Python path differs")
    authority = load_ref(m["authorization"])
    require(authority.get("status") == "explicit_user_authorization_recorded" and authority.get("instance_id") == INSTANCE and
            authority.get("scope") == "no_loophole_step1_only" and authority.get("host") == "codex-h100",
            "Missing task-scoped user authorization evidence")
    if m.get("generation_runtime_seconds") == 21600:
        require(authority.get("integrated_validation") is True and authority.get("generation_runtime_seconds") == 21600 and
                authority.get("cumulative_gpu_wall_cap_seconds") == 43200, "Six-hour runtime lacks exact user amendment")
    plan, requests, physical = request_context(m)
    require(m["scientific_conditions"] == plan["conditions"] and m["candidate_map"] ==
            plan["no_loophole_capability"]["h100_portability"]["candidate_map"] and
            m["scientific_identity_sha256"] == scientific_identity(plan), "Scientific/physical map changed")
    file_bindings(m["bound_files"], runtime_path=m["python"])
    require(all(m["bound_files"].get(k) == v for k, v in actual_source.items()), "Unbound source file")
    qualified = load_ref(m["source_qualification"])
    require(qualified.get("status") == "verified_h100_step1_cpu_source" and qualified.get("source_root") == str(source) and
            qualified.get("source_files") == actual_source and type(qualified.get("tests_passed")) is int and qualified["tests_passed"] > 0 and
            type(qualified.get("tests_failed")) is int and qualified["tests_failed"] == 0 and
            type(qualified.get("producer_exit_status")) is int and qualified["producer_exit_status"] == 0 and
            qualified.get("process_release_verified") is True, "Missing positive exact-source CPU qualification")
    for relative, expected in {**FROZEN, HERE: sha(__file__),
                              "infra/gpu03/direction_discovery/h100_qualification.py": sha(numerical.__file__)}.items():
        require(actual_source.get(str(source / relative), {}).get("sha256") == expected, "Running/frozen source differs")
    model_files = {**inventory(m["model_snapshot"]), **inventory(m["checkpoint"])}
    require(model_files == p["model_files"] and len(model_files) == 15 and
            all(m["bound_files"].get(k) == v for k,v in model_files.items()), "Model inventory differs from independent parity profile")
    parity = load_ref(p["model_parity_proof"])
    require(parity.get("status") == "independently_verified_h100_model_parity" and parity.get("instance_id") == INSTANCE and
            parity.get("model_files") == model_files and parity.get("all_original_hashes_equal") is True and
            parity.get("original_inventory_sha256") == ORIGINAL_MODEL_INVENTORY_SHA, "Missing original model parity proof")
    needed = [Path(plan["prepared_records"]), Path(plan["dataset"]), *protocol().bindings(plan), *helper_bindings]
    require(all(str(p) in m["bound_files"] for p in needed), "Missing full scientific input bindings")
    require(len(m["workers"]) == workers, "Missing worker")
    for i, worker in enumerate(m["workers"]):
        task = expected_task(m, plan, requests, physical, i)
        path = str(stage / "tasks" / (task["worker_name"] + ".json"))
        require(worker == {"name": task["worker_name"], "gpu_id": ids[i], "cpu_set": f"{96+2*ids[i]},{97+2*ids[i]}",
                "task": path, "command": [m["python"], str(source / ENGINE), "--task", path],
                "success_file": task["worker_name"] + "/SUCCESS.json", "requests": len(task["requests"])}, "Worker command/partition differs")
        if check_tasks:
            require(path in m["bound_files"] and load_ref({"path": path, **m["bound_files"][path]}) == task,
                    "Task bytes differ from exact scientific round-robin partition")
    qualification = m.get("inference_qualification")
    if m.get("continuation") is not None:
        require(qualification is None and authority.get("integrated_validation") is True and
                str(Path(m["continuation"]["path"])) in m["bound_files"],
                "Continuation requires explicit integrated validation and bound reuse evidence")
    elif m["phase"] == "h100_no_loophole_capability":
        q = load_ref(qualification)
        require(q.get("status") == "independently_verified_h100_generation" and q.get("phase") == "h100_numerical_qualification" and
                q.get("scientific_identity_sha256") == scientific_identity(plan) and q.get("host_profile_sha256") == m["host_profile"]["sha256"] and
                q.get("source_root") == m["source_root"] and q.get("source_qualification") == m["source_qualification"] and
                q.get("helper_qualification") == m["helper_qualification"] and
                q.get("generation_requests") == 6 and q.get("tf_requests") == 0 and
                q.get("numerical", {}).get("status") == "verified_h100_six_call_numerical_qualification" and
                q.get("gpu_release_verified") is True and q.get("process_release_verified") is True,
                "Main740 lacks successful same-host six-call qualification")
    else:
        require(qualification is None, "Qualification cannot bootstrap itself from another qualification")
    return m


def load_manifest(path, expected):
    return validate_manifest_object(load_ref({"path": str(path), "sha256": expected}))


def validate_admission(m, digest, admission):
    from infra.gpu03.direction_discovery import h100_budget
    _, _, wall_cap = h100_budget.manifest_budget(m)
    a = load_ref(admission)
    require(a.get("status") == "admitted_h100_no_loophole_phase" and
            (a.get("manifest_sha256"), a.get("run_token"), a.get("host_profile_sha256"), a.get("request_plan_sha256")) ==
            (digest, m["run_token"], m["host_profile"]["sha256"], m["request_plan"]["sha256"]) and
            a.get("generation_requests") == m["generation_requests"] and a.get("tf_requests") == 0 and
            a.get("reserved_wall_seconds") == m["limits"]["reserved_wall_seconds"], "Admission identity/count/limit differs")
    before, after = a["budget_before"], a["budget_after"]
    fields = {"generation_requests", "tf_requests", "untouched_test_requests", "untouched_test_generation_requests",
              "untouched_test_tf_requests", "gpu_phase_wall_seconds"}
    require(set(before) == set(after) == fields, "Incomplete ledger counters")
    require(all(type(before[k]) is int for k in ("generation_requests", "tf_requests", "untouched_test_requests")) and
            before["generation_requests"] >= 2646 and before["tf_requests"] == 9228 and
            all(before[k] == 0 for k in fields if "untouched_test" in k) and
            math.isfinite(before["gpu_phase_wall_seconds"]) and before["gpu_phase_wall_seconds"] >= 20456.688430309296,
            "Historical failed/successful reservations were reset")
    expected = {**before, "generation_requests": before["generation_requests"] + m["generation_requests"],
                "gpu_phase_wall_seconds": before["gpu_phase_wall_seconds"] + a["reserved_wall_seconds"]}
    require(after == expected and after["generation_requests"] <= 4096 and after["tf_requests"] <= 12000 and
            after["gpu_phase_wall_seconds"] <= wall_cap, "Cumulative budget exceeded or counters refunded")
    seed = load_ref(a["historical_seed"])
    require({k: seed.get("budget", {}).get(k) for k in fields} ==
            {"generation_requests": 2646, "tf_requests": 9228, "untouched_test_requests": 0,
             "untouched_test_generation_requests": 0, "untouched_test_tf_requests": 0,
             "gpu_phase_wall_seconds": 20456.688430309296} and isinstance(seed["budget"].get("phases"), list) and
            seed.get("utc", "").startswith("2026-09-08") and seed.get("python", "").startswith("3.12.3 "),
            "Wrong immutable historical budget seed")
    plan, requests, _ = request_context(m)
    require(a.get("phase") == m["phase"] and a.get("generation_request_ids") == [r["request_id"] for r in requests] and
            a.get("authorization") == m["authorization"] and a.get("manifest", {}).get("sha256") == digest,
            "Admission request union/authorization differs")
    admission_time(a)
    require(h100_budget.validate_admission_reference(admission, manifest_sha256=digest, run_token=m["run_token"]) == a,
            "Admission is not exactly present in the canonical locked ledger")
    return a


def admission_time(a):
    require(isinstance(a.get("admitted_at"), str), "Admission timestamp must be explicit UTC ISO")
    stamp = datetime.fromisoformat(a["admitted_at"].replace("Z", "+00:00"))
    require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0, "Admission timezone is not UTC")
    return stamp.timestamp()


def worker_environment(gpu=None):
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def launcher_environment():
    require(os.getuid() == UID and pwd.getpwuid(os.getuid()).pw_name == OWNER, "Wrong launcher user")
    runtime = Path(f"/run/user/{UID}")
    require(runtime.is_dir() and not runtime.is_symlink() and runtime.stat().st_uid == UID and
            stat.S_ISSOCK((runtime / "bus").stat().st_mode) and (runtime / "bus").stat().st_uid == UID,
            "Missing owned user-systemd bus")
    return {"PATH": "/usr/bin:/bin", "HOME": "/home/ubuntu", "USER": OWNER, "LOGNAME": OWNER,
            "LANG": "C.UTF-8", "XDG_RUNTIME_DIR": str(runtime), "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(runtime / "bus")}


def host_check(m, *, duration=True):
    require(socket.gethostname() == HOST and os.getuid() == UID and pwd.getpwuid(UID).pw_name == OWNER,
            "Wrong actual host/user")
    require(sys.executable == m["python"] and sys.version_info[:3] == (3, 12, 3) and
            {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS, "Actual Python/dependencies differ")
    if duration:
        require({94, *range(96, 112)} <= os.sched_getaffinity(0), "Assigned CPUs unavailable")
    if duration:
        require(time.time() + m["limits"]["reserved_wall_seconds"] + m["postprocessing_seconds"] < DEADLINE,
                "Phase cannot finish before the immutable durable-copy deadline")
    for timer in ("codex-h100-deadline.timer", "codex-h100-final-sync.timer"):
        result = subprocess.run(["systemctl", "is-active", timer], capture_output=True, text=True, timeout=10)
        require(result.returncode == 0 and result.stdout.strip() == "active", "Existing deadline timer inactive")
    linger = subprocess.run(["loginctl", "show-user", OWNER, "--property=Linger", "--value"],
                            capture_output=True, text=True, timeout=10)
    require(linger.returncode == 0 and linger.stdout.strip() == "yes", "User-manager persistence is disabled")


def process_info(pid):
    try:
        p = Path(f"/proc/{pid}"); text = (p / "stat").read_text(); fields = text[text.rfind(")") + 2:].split()
        return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]), "start_ticks": int(fields[19]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), "uid": p.stat().st_uid}
    except FileNotFoundError:
        return None


def same_process(old):
    current = process_info(old["pid"])
    return current is not None and current["uid"] == old["uid"] and current["start_ticks"] == old["start_ticks"]


def same_identity(a, b):
    return all(a.get(k) == b.get(k) for k in ("pid", "pgid", "uid", "start_ticks"))


def cgroup_path(value, token):
    require(isinstance(value, str) and value.startswith("/user.slice/user-1000.slice/") and
            value.endswith("/" + token + ".service") and ".." not in Path(value).parts,
            "Wrong exact owned service cgroup")
    return Path("/sys/fs/cgroup") / value.lstrip("/")


def cgroup_processes(group):
    result = {}
    for file in Path(group).rglob("cgroup.procs"):
        for value in file.read_text().splitlines():
            p = process_info(int(value))
            if p is not None:
                require(p["uid"] == UID, "Foreign process in exact service cgroup")
                result[p["pid"]] = p
    return result


def unit_state(token):
    result = subprocess.run(["systemctl", "--user", "show", token + ".service",
        "--property=Id,LoadState,ActiveState,SubState,MainPID,ControlGroup,RuntimeMaxUSec,TimeoutStopUSec,MemoryMax,TasksMax,KillMode,InvocationID"],
        capture_output=True, text=True, timeout=15, env=launcher_environment())
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    require(fields.get("Id") == token + ".service" and (result.returncode == 0 or
            (result.returncode == 1 and fields.get("LoadState") == "not-found" and
             fields.get("ActiveState") == "inactive" and fields.get("MainPID") == "0")),
            "Could not positively read the exact user service/absence")
    return fields


def duration_seconds(text):
    units = {"us": 1e-6, "ms": .001, "s": 1, "min": 60, "h": 3600, "d": 86400}
    chunks = re.findall(r"([0-9]+(?:\.[0-9]+)?)(us|ms|min|s|h|d)", text)
    require(chunks and "".join(a+b for a,b in chunks) == text.replace(" ", ""), "Malformed systemd duration")
    return sum(float(a) * units[b] for a,b in chunks)


def check_service(m, fields):
    require(fields.get("ActiveState") == "active" and fields.get("SubState") == "running" and
            fields.get("MainPID", "").isdigit() and int(fields["MainPID"]) > 0,
            "Service was not positively verified active")
    require(re.fullmatch(r"[0-9a-f]{32}", fields.get("InvocationID", "")) and
            duration_seconds(fields.get("RuntimeMaxUSec", "")) == m["limits"]["systemd_runtime_seconds"] and
            duration_seconds(fields.get("TimeoutStopUSec", "")) == m["limits"]["timeout_stop_seconds"] and
            fields.get("MemoryMax") == str(160 * 1024**3) and fields.get("TasksMax") == "512" and
            fields.get("KillMode") == "control-group", "Actual service resources differ")
    cgroup_path(fields.get("ControlGroup"), m["run_token"])
    identity = process_info(int(fields["MainPID"]))
    require(identity is not None and identity["uid"] == UID, "Actual main PID ownership unknown")
    return identity


def gpu_snapshot():
    # Existing generic NVML command parser; no shared-host model-name guard is imported.
    from infra.gpu03.activation_dataset import extract_triplet_raw
    return extract_triplet_raw.gpu_snapshot()


def check_devices(m, inventory_rows, allowed=None, settling=()):
    allowed = {} if allowed is None else allowed
    profile = load_ref(m["host_profile"])
    frozen = {v["index"]: v for v in profile["gpu_inventory"]}
    require(len(inventory_rows) == 8 and {r["index"] for r in inventory_rows} == set(range(8)), "Incomplete actual GPU inventory")
    for r in inventory_rows:
        if r["index"] not in m["gpu_ids"]:
            continue
        gpu = r["index"]
        require(all(r[k] == frozen[gpu][k] for k in ("uuid", "name", "memory_total_mib")), "Selected GPU identity changed")
        require(all(p["pid"] in allowed.get(gpu, set()) and p["owner"] == OWNER for p in r["processes"]),
                "Selected GPU has a foreign/unowned compute process")
        if not allowed.get(gpu) and gpu not in settling:
            require(not r["processes"] and r["memory_used_mib"] <= 64 and r["utilization_percent"] <= 1 and
                    r["memory_free_mib"] >= frozen[gpu]["memory_free_mib"] - 64, "Selected GPU is not positively idle")


def resource_state(m, active):
    from infra.gpu03.activation_dataset import extract_delta_activations as legacy
    allowed, roots, settling = {}, set(), set()
    for p, gpu in active:
        if p.poll() is None:
            roots.add(p.pid); _, descendants = legacy.process_tree_rss_kib({p.pid}); allowed[gpu] = set(descendants)
        elif p.returncode == 0:
            if not hasattr(p, "h100_exit_seen"):
                p.h100_exit_seen = time.monotonic()
            if time.monotonic() - p.h100_exit_seen < 60:
                settling.add(gpu)
    rss, _ = legacy.process_tree_rss_kib(roots)
    available = legacy.mem_available_kib()
    free = shutil.disk_usage(m["stage"]).free
    require(available >= 192 * 1024**2 and rss <= 128 * 1024**2 and free >= 32 * 1024**3 and
            os.getloadavg()[0] <= 128, "Host RAM/RSS/disk/CPU guard failed")
    rows = gpu_snapshot(); check_devices(m, rows, allowed, settling)
    for log in [*Path(m["runtime"]).glob("*.log"), *Path(m["control"]).glob("*.log")]:
        require(log.stat().st_size <= 64 * 1024**2, "Log bound exceeded")
    return {"at": time.time(), "gpu_inventory": rows, "available_ram_kib": available,
            "worker_rss_kib": rss, "free_disk_bytes": free}


def terminate_owned(processes):
    for p, identity in processes:
        if p.poll() is None:
            require(identity is not None and same_process(identity) and identity["uid"] == UID and
                    identity["pgid"] == p.pid and os.getpgid(p.pid) == p.pid, "Ambiguous owned process; no signal sent")
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for p, identity in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                require(same_process(identity) and os.getpgid(p.pid) == p.pid, "Process identity changed before SIGKILL")
                os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=10)
    require(all(p.poll() is not None for p, _ in processes), "Owned worker remains unreaped")


def released_gpus(m, journal, timeout=60):
    end = time.monotonic() + timeout
    while True:
        rows = gpu_snapshot()
        check_devices(m, rows, settling=m["gpu_ids"])
        try:
            check_devices(m, rows)
            value = {"verified": True, "gpu_ids": m["gpu_ids"], "gpu_inventory": rows, "at": time.time()}
            append_json(journal, {"release": value}); return value
        except ValueError:
            require(time.monotonic() < end, "Selected GPU release not proven")
            time.sleep(1)


def output_inventory(root):
    return {str(Path(p).relative_to(root)): v for p, v in inventory(root).items()
            if Path(p) != Path(root) / "artifact_manifest.json"}


def validate_worker(m, w):
    path = Path(m["output"]) / w["success_file"]
    value = load_ref(ref(path), immutable=False)
    require(all(value.get(k) == v for k, v in {"status": "succeeded", "run_token": m["run_token"],
            "worker_name": w["name"], "requests": w["requests"], "mode": "generate"}.items()), "Worker success mismatch")
    original = load_ref({"path": w["task"], **m["bound_files"][w["task"]]})
    require(load_ref(ref(path.parent / "task.json"), immutable=False) == original, "Worker task snapshot differs")
    report = value.get("model_load_reports")
    require(isinstance(report, dict) and report.get("with_adapter") is True and
            isinstance(report.get("active_adapters"), list) and report["active_adapters"] and
            all(type(report.get(k)) is int and report[k] > 0 for k in
                ("lora_parameter_tensors", "nonzero_lora_parameter_tensors", "lora_parameter_elements")) and
            report["nonzero_lora_parameter_tensors"] <= report["lora_parameter_tensors"],
            "Missing positively loaded nonzero active checkpoint")
    return {"worker": w["name"], "success": ref(path), "task": ref(path.parent / "task.json")}


def check_completed_requests(m, state):
    """Validate newly appended complete rows inside the existing supervisor loop."""
    for w in m["workers"]:
        entry = state.setdefault(w["name"], {"offset": 0, "count": 0, "inode": None})
        path = Path(m["output"])/w["name"]/"results.jsonl"
        if not path.exists():
            require(entry["offset"] == 0, "Runtime result journal disappeared")
            continue
        require(not path.is_symlink(), "Runtime result journal is a symlink")
        st = path.stat()
        require(st.st_uid == UID and stat.S_ISREG(st.st_mode) and entry["offset"] <= st.st_size <= 128 << 20 and
                entry["inode"] in (None, (st.st_dev, st.st_ino)), "Runtime result journal replaced/truncated/oversized")
        entry["inode"] = (st.st_dev, st.st_ino)
        if "requests" not in entry:
            entry["requests"] = load_ref({"path": w["task"], **m["bound_files"][w["task"]]})["requests"]
        with path.open("rb") as stream:
            stream.seek(entry["offset"]); tail = stream.read((128 << 20)+1)
        end = tail.rfind(b"\n")+1
        for line in tail[:end].splitlines():
            require(line.strip() and entry["count"] < len(entry["requests"]), "Extra/blank runtime result")
            row = json.loads(line); request = entry["requests"][entry["count"]]
            require(all(row.get(k) == v for k,v in request.items()), "Runtime result request identity differs")
            numerical.check_result(row["result"], request["condition_id"])
            entry["count"] += 1
        entry["offset"] += end


def collect_physical_rows(m, plan, requests):
    prepared_data = read_ref({"path": plan["prepared_records"], "sha256": plan["prepared_records_sha256"]})
    prepared = {r["record_id"]: r for r in (json.loads(x) for x in prepared_data.splitlines() if x.strip())}
    by_id, snapshots = {}, []
    for w in m["workers"]:
        bound = ref(Path(m["output"])/w["name"]/"results.jsonl")
        data = read_ref(bound, immutable=False, max_bytes=128 << 20); snapshots.append(bound)
        require(data.endswith(b"\n") and all(x.strip() for x in data.splitlines()), "Incomplete physical result journal")
        rows = [json.loads(x) for x in data.splitlines()]
        assigned = load_ref({"path": w["task"], **m["bound_files"][w["task"]]})["requests"]
        require(len(rows) == len(assigned), "Partial worker rows")
        for row, expected in zip(rows, assigned):
            require(all(row.get(k) == v for k,v in expected.items()) and row["request_id"] not in by_id,
                    "Reissued/conflicting/misassigned result request")
            require(row.get("original_class") == prepared[expected["record_id"]]["outcome_presence_class"],
                    "Historical carrier provenance class differs")
            numerical.check_result(row["result"], expected["condition_id"])
            by_id[row["request_id"]] = row
    require(set(by_id) == {r["request_id"] for r in requests}, "Full exact physical request union is incomplete")
    return by_id, snapshots


def logical_rows(m, plan, physical):
    reused = continuation_rows(m, plan)
    by_id = dict(physical)
    for row in reused:
        require(row["request_id"] not in by_id, "Reused cell was regenerated")
        by_id[row["request_id"]] = row
    require(len(by_id) == 740 and set(by_id) == {r["request_id"] for r in plan["requests"]}, "Logical740 coverage differs")
    return [by_id[r["request_id"]] for r in plan["requests"]]


def consumed(m, digest):
    value = load_ref(ref(Path(m["control"]) / "consumed_admission.json"))
    require(value.get("manifest_sha256") == digest and value.get("run_token") == m["run_token"], "Admission was not consumed for this manifest")
    validate_admission(m, digest, value["admission"])
    return value


def supervise(path, digest):
    m = load_manifest(path, digest); host_check(m); consumed(m, digest)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and re.fullmatch(r"[0-9a-f]{32}", os.environ.get("INVOCATION_ID", "")),
            "Controller environment/invocation unavailable")
    group_lines = [x[3:] for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::")]
    require(len(group_lines) == 1, "Unified service cgroup unavailable")
    group = cgroup_path(group_lines[0], m["run_token"])
    output, runtime = Path(m["output"]), Path(m["runtime"])
    require(not output.exists() and not runtime.exists(), "Preserve prior outputs; no automatic reissue")
    require(shutil.disk_usage(m["stage"]).free >= 64 * 1024**3, "Insufficient startup disk reserve")
    output.mkdir(mode=0o700); runtime.mkdir(mode=0o700)
    started = time.monotonic(); end = started + m["limits"]["runtime_seconds"]
    identity = {"run_token": m["run_token"], "manifest_sha256": digest, "invocation_id": os.environ["INVOCATION_ID"],
                "control_group": group_lines[0], "main_identity": process_info(os.getpid()), "started_at": time.time()}
    exclusive_json(runtime / "run_identity.json", identity)
    affinity = os.sched_getaffinity(0); os.sched_setaffinity(0, {94})
    processes, active, locks = [], [], []
    checked = {}
    def deadline():
        require(time.monotonic() < end and time.time() + m["postprocessing_seconds"] < DEADLINE, "GPU campaign deadline exhausted")
    def interrupted(signum, _frame):
        raise RuntimeError("Owned supervisor interrupted by signal " + str(signum))
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    journal = runtime / "resource_journal.jsonl"
    try:
        lock_root = WORK / "h100_gpu_locks"
        require(not lock_root.is_symlink(), "Unsafe GPU lock root")
        lock_root.mkdir(mode=0o700, exist_ok=True)
        require(lock_root.stat().st_uid == UID and not lock_root.stat().st_mode & 0o022, "Wrong GPU lock owner/mode")
        for gpu in m["gpu_ids"]:
            fd = os.open(lock_root / f"gpu_{gpu}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            lock = os.fdopen(fd, "a+"); locks.append(lock)
            require(os.fstat(fd).st_uid == UID and stat.S_ISREG(os.fstat(fd).st_mode), "Wrong GPU lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The launcher binds actual resources while this stage is still model-free.
        while not (Path(m["control"]) / "service_started.json").is_file():
            deadline(); require(time.monotonic() - started < 90, "Actual service start proof missing"); time.sleep(.2)
        proof = load_ref(ref(Path(m["control"]) / "service_started.json"))
        require(proof["fields"]["InvocationID"] == identity["invocation_id"] and
                proof["fields"]["ControlGroup"] == identity["control_group"] and same_identity(proof["main_identity"], identity["main_identity"]),
                "Actual launcher/controller identity mismatch")
        idle_start = time.monotonic()
        while True:
            deadline(); append_json(runtime / "qualification.jsonl", resource_state(m, []))
            if time.monotonic() - idle_start >= 60:
                break
            time.sleep(min(INTERVAL, 60 - (time.monotonic()-idle_start)))
        for i, w in enumerate(m["workers"]):
            deadline(); append_json(journal, resource_state(m, active))
            command = ["taskset", "-c", w["cpu_set"], "nice", "-n", "10", "ionice", "-c", "2", "-n", "7", *w["command"]]
            with (runtime / (w["name"] + ".log")).open("xb") as log:
                p = subprocess.Popen(command, cwd=m["source_root"], env=worker_environment(w["gpu_id"]),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pi = process_info(p.pid); require(pi is not None and pi["uid"] == UID and pi["pgid"] == p.pid, "Worker process identity missing")
            processes.append((p, pi)); active.append((p, w["gpu_id"]))
            append_json(runtime / "processes.jsonl", {"worker": w["name"], "gpu_id": w["gpu_id"], "identity": pi, "command": command})
            if i+1 < len(m["workers"]):
                wait_end = time.monotonic()+10
                while time.monotonic() < wait_end:
                    deadline(); require(all(p.poll() in (None, 0) for p,_ in processes), "Worker failed during staggering")
                    append_json(journal, resource_state(m, active)); time.sleep(min(INTERVAL, max(0, wait_end-time.monotonic())))
                    if m.get("continuation") is not None:
                        check_completed_requests(m, checked)
        while any(p.poll() is None for p,_ in processes):
            deadline(); require(all(p.poll() in (None, 0) for p,_ in processes), "Worker exited nonzero")
            if m.get("continuation") is not None:
                check_completed_requests(m, checked)
            append_json(journal, resource_state(m, active)); time.sleep(INTERVAL)
        require(all(p.returncode == 0 for p,_ in processes), "Worker failure")
        deadline(); receipts = [validate_worker(m, w) for w in m["workers"]]
        remaining = cgroup_processes(group)
        require(set(remaining) == {os.getpid()}, "A worker descendant remains in service cgroup")
        release = released_gpus(m, journal); deadline()
        exclusive_json(output / "gpu_release.json", release)
        if m.get("continuation") is not None:
            check_completed_requests(m, checked)
            plan, requests, _ = request_context(m)
            physical, _ = collect_physical_rows(m, plan, requests)
            with (output/"logical_results.jsonl").open("x") as stream:
                for row in logical_rows(m, plan, physical):
                    stream.write(canonical(row)+"\n")
                stream.flush(); os.fsync(stream.fileno())
        exclusive_json(output / "campaign_summary.json", {**identity, "status": "succeeded", "phase": m["phase"],
            "worker_exit_codes": [p.returncode for p,_ in processes], "worker_receipts": receipts,
            "gpu_release_verified": True, "elapsed_seconds": time.monotonic()-started})
        shutil.copytree(runtime, output / "execution")
        exclusive_json(output / "artifact_manifest.json", {"algorithm": "sha256", "files": output_inventory(output)})
    except BaseException as error:
        cleanup = {"owned_workers_released": False, "gpu_release_verified": False}
        try:
            terminate_owned(processes); cleanup["owned_workers_released"] = True
        except BaseException as e:
            cleanup["process_error"] = str(e)
        try:
            cleanup["gpu_release"] = released_gpus(m, journal, 30); cleanup["gpu_release_verified"] = True
        except BaseException as e:
            cleanup["gpu_error"] = str(e)
        exclusive_json(output / "FAILURE.json", {**identity, "error_type": type(error).__name__, "error": str(error), "cleanup": cleanup})
        raise
    finally:
        for lock in locks:
            lock.close()
        for s, handler in handlers.items():
            signal.signal(s, handler)
        os.sched_setaffinity(0, affinity)


def receipt(path, digest):
    # ExecStopPost must be able to retain failure evidence even if inputs changed.
    m = load_ref({"path": str(path), "sha256": digest})
    require(m.get("purpose") == PURPOSE and m.get("host") == HOST and os.getuid() == UID, "Wrong exit receipt context")
    exclusive_json(Path(m["control"]) / "supervisor_exit.json", {"run_token": m["run_token"], "manifest_sha256": digest,
        "invocation_id": os.environ.get("INVOCATION_ID"), "service_result": os.environ.get("SERVICE_RESULT"),
        "exit_code_kind": os.environ.get("EXIT_CODE"), "exit_status": os.environ.get("EXIT_STATUS"), "at": time.time()})


def service_command(m, path, digest):
    script = str(Path(m["source_root"]) / HERE)
    receipt_argv = ["/usr/bin/env", "-i", *[k+"="+v for k,v in worker_environment().items()],
        "INVOCATION_ID=${INVOCATION_ID}", "SERVICE_RESULT=${SERVICE_RESULT}", "EXIT_CODE=${EXIT_CODE}", "EXIT_STATUS=${EXIT_STATUS}",
        m["python"], script, "--receipt", "--manifest", str(path), "--manifest-sha256", digest]
    # systemd expands these literal single argv elements; caller shells never do.
    return ["systemd-run", "--user", "--quiet", "--unit", m["run_token"], "--service-type=exec",
        "--property=RuntimeMaxSec="+str(m["limits"]["systemd_runtime_seconds"]),
        "--property=TimeoutStopSec="+str(m["limits"]["timeout_stop_seconds"]),
        "--property=KillMode=control-group", "--property=MemoryMax=160G", "--property=TasksMax=512",
        "--property=LimitNOFILE=4096", "--property=UMask=0077", "--property=CPUAffinity=94 96-111",
        "--property=ExecStopPost="+shlex.join(receipt_argv),
        "--property=StandardOutput=append:"+str(Path(m["control"]) / "supervisor.log"), "--property=StandardError=inherit",
        "/usr/bin/env", "-i", *[k+"="+v for k,v in worker_environment().items()], "INVOCATION_ID=${INVOCATION_ID}",
        m["python"], script, "--supervise", "--manifest", str(path), "--manifest-sha256", digest]


def launch(path, digest, admission):
    m = load_manifest(path, digest); host_check(m); a = validate_admission(m, digest, admission)
    require(0 <= time.time()-admission_time(a) <= 900, "Stale/future admission")
    check_devices(m, gpu_snapshot())
    env = launcher_environment()
    control = Path(m["control"]); control.mkdir(mode=0o700, exist_ok=False)
    exclusive_json(control / "consumed_admission.json", {"run_token": m["run_token"], "manifest_sha256": digest,
                   "admission": admission, "consumed_at": time.time()}, immutable=True)
    command = service_command(m, path, digest)
    exclusive_json(control / "launch_intent.json", {"command": command, "launcher_environment": env, "at": time.time()}, immutable=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
    exclusive_json(control / "launch_result.json", {"command": command, "returncode": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr}, immutable=True)
    require(result.returncode == 0, "systemd launch failed; consumed reservation remains spent")
    try:
        fields = unit_state(m["run_token"]); identity = check_service(m, fields)
        require(os.sched_getaffinity(identity["pid"]) in ({94}, {94, *range(96,112)}), "Actual supervisor affinity differs")
        exclusive_json(control / "service_started.json", {"fields": fields, "main_identity": identity, "at": time.time()}, immutable=True)
    except BaseException as error:
        stopped = subprocess.run(["systemctl", "--user", "stop", m["run_token"]+".service"], env=env, capture_output=True, text=True, timeout=100)
        exclusive_json(control / "launch_verification_failure.json", {"error": str(error), "stop_returncode": stopped.returncode})
        raise
    return {"status": "launched", "run_token": m["run_token"], "manifest_sha256": digest, "service": fields}


def verify(path, digest):
    m = load_manifest(path, digest); host_check(m, duration=False); consumed(m, digest)
    control, output = Path(m["control"]), Path(m["output"])
    start = load_ref(ref(control / "service_started.json"))
    terminal = load_ref(ref(control / "supervisor_exit.json"), immutable=False)
    require(terminal.get("run_token") == m["run_token"] and terminal.get("manifest_sha256") == digest and
            terminal.get("invocation_id") == start["fields"]["InvocationID"] and
            terminal.get("service_result") == "success" and terminal.get("exit_code_kind") == "exited" and
            terminal.get("exit_status") == "0", "Actual service has not successfully exited")
    fields = unit_state(m["run_token"])
    require(fields.get("ActiveState") == "inactive" and fields.get("MainPID") == "0" and
            fields.get("InvocationID") in ("", start["fields"]["InvocationID"]) and
            not same_process(start["main_identity"]) and
            not cgroup_processes(cgroup_path(start["fields"]["ControlGroup"], m["run_token"])),
            "Exact service/process group has not released")
    check_devices(m, gpu_snapshot())
    require(not (output / "FAILURE.json").exists(), "Producer failure exists")
    summary = load_ref(ref(output / "campaign_summary.json"), immutable=False)
    require(summary.get("status") == "succeeded" and summary.get("manifest_sha256") == digest and
            summary.get("run_token") == m["run_token"] and summary.get("invocation_id") == terminal["invocation_id"] and
            summary.get("control_group") == start["fields"]["ControlGroup"] and
            summary.get("worker_exit_codes") == [0]*len(m["workers"]) and summary.get("gpu_release_verified") is True and
            summary["worker_receipts"] == [validate_worker(m,w) for w in m["workers"]], "Producer terminal proof differs")
    artifact = load_ref(ref(output / "artifact_manifest.json"), immutable=False)
    require(artifact == {"algorithm": "sha256", "files": output_inventory(output)}, "Terminal result bytes differ")
    plan, requests, _ = request_context(m)
    by_id, snapshots = collect_physical_rows(m, plan, requests)
    physical_snapshots = snapshots
    if m.get("continuation") is not None:
        expected_rows = logical_rows(m, plan, by_id)
        logical_ref = ref(output/"logical_results.jsonl")
        data = read_ref(logical_ref, immutable=False, max_bytes=128 << 20)
        require(data == ("".join(canonical(row)+"\n" for row in expected_rows)).encode(),
                "Saved logical740 differs from exact physical735 plus five immutable aliases")
        snapshots = [logical_ref]
    require(artifact == {"algorithm": "sha256", "files": output_inventory(output)}, "Results changed during independent verification")
    numerical_proof = numerical.verify_rows(plan, [by_id[r["request_id"]] for r in requests]) if m["phase"] == "h100_numerical_qualification" else None
    wall = terminal["at"]-load_ref(ref(control / "launch_intent.json"))["at"]
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall <= m["limits"]["reserved_wall_seconds"],
            "Actual launch/exit wall receipt is invalid")
    proof = {"status": "independently_verified_h100_generation", "phase": m["phase"], "run_token": m["run_token"],
            "manifest": {"path": str(path), "sha256": digest}, "source_root": m["source_root"], "source_qualification": m["source_qualification"],
            "manifest_sha256": digest, "request_plan_sha256": m["request_plan"]["sha256"],
            "host_profile_sha256": m["host_profile"]["sha256"], "scientific_identity_sha256": m["scientific_identity_sha256"],
            "helper_qualification": m["helper_qualification"],
            "artifact_manifest": ref(output / "artifact_manifest.json"), "service_started": ref(control / "service_started.json"),
            "supervisor_exit": ref(control / "supervisor_exit.json"), "invocation_id": terminal["invocation_id"],
            "generation_requests": len(requests), "tf_requests": 0, "request_ids": [r["request_id"] for r in requests],
            "result_files": snapshots, "bound_files": m["bound_files"], "gpu_release_verified": True,
            "process_release_verified": True, "actual_wall_seconds": wall,
            "wall_basis": "conservative_launch_intent_through_service_exit", "numerical": numerical_proof}
    if m.get("continuation") is not None:
        proof.update(generation_requests=740, new_generation_requests=735, reused_generation_requests=5,
            request_ids=[r["request_id"] for r in plan["requests"]], physical_result_files=physical_snapshots,
            continuation=m["continuation"], baseline_repeat_verification_available=False,
            projection_positions_verified=True, projection_energy_verified=True)
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for name in ("build", "launch", "supervise", "receipt", "verify"):
        modes.add_argument("--"+name, action="store_true")
    p.add_argument("--spec", type=Path); p.add_argument("--manifest", type=Path); p.add_argument("--manifest-sha256")
    p.add_argument("--admission", type=Path); p.add_argument("--admission-sha256")
    p.add_argument("--output-proof", type=Path)
    a = p.parse_args()
    if a.build:
        result = build(json.loads(a.spec.read_text()))
    elif a.launch:
        result = launch(a.manifest, a.manifest_sha256, {"path": str(a.admission), "sha256": a.admission_sha256})
    else:
        result = (supervise if a.supervise else receipt if a.receipt else verify)(a.manifest, a.manifest_sha256)
    if a.output_proof:
        require(a.verify, "Only independent verification may write an output proof")
        exclusive_json(a.output_proof, result, immutable=True)
    if result is not None:
        print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
