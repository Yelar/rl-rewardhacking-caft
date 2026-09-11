#!/usr/bin/env python3
"""Manifest-bound direct gpu-04 campaigns with independent systemd supervision.

This generic profile reuses the raw-extraction resource probes, not its scientific
manifest loader. It never selects a new GPU or modifies a worker command at launch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
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

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "activation_dataset"))
import extract_triplet_raw as raw

RUN_ROOT = Path("/scratch/researcher/codex_runs")
OWNER = "researcher"
INTERVAL = 5
QUALIFICATION_SECONDS = 60


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cpu_set(value):
    require(isinstance(value, str) and re.fullmatch(r"\d+(,\d+)*", value), "CPU set must contain explicit comma-separated indices")
    values = [int(item) for item in value.split(",")]
    require(len(values) == len(set(values)), "duplicate CPU assignment")
    return set(values)


def expected_command(m, manifest_path):
    return [m["python"], str(Path(m["source_root"]) / "infra/gpu03/direction_discovery/supervisor.py"),
            "--supervise", "--manifest", str(manifest_path)]


def load_manifest(path, digest, *, check_files=True, check_source=True):
    """Validate generic ownership, safety limits, bindings, and exact commands."""
    path = Path(path)
    require(re.fullmatch(r"[0-9a-f]{64}", digest or "") and sha256(path) == digest, "Manifest SHA-256 mismatch")
    m = json.loads(path.read_text())
    return validate_manifest_object(m, path, check_files=check_files, check_source=check_source)


def validate_manifest_object(m, path, *, check_files=False, check_source=False):
    """Pure schema/command validation also used for explicit remote replicas.

    Callers must authenticate the original bytes; this helper never supplies a
    digest, resolves a remote file, or grants permission to launch.
    """
    path = Path(path)
    require(m["schema_version"] == 1 and m["purpose"] == "direction_discovery_campaign", "Wrong manifest purpose/schema")
    require(m["host"] in ("gpu-04", "gpu-02", "gpu-01") and m["owner"] == OWNER, "Wrong host/user authorization")
    require((m["host"] in ("gpu-02", "gpu-01")) == ("remote_generation" in m), "Remote host requires its explicit authority profile")
    require(isinstance(m["phase"], str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", m["phase"]), "Invalid phase identifier")
    require(re.fullmatch(r"codex-[a-z0-9][a-z0-9-]{8,100}", m["run_token"]), "Unsafe run token")
    require(m["scientific"]["training"] is False, "Full training is outside this discovery profile")
    require(m["authorization"] and isinstance(m["authorization"], str), "Missing explicit task authorization record")
    require(Path(m["python"]).is_absolute(), "Python executable must be absolute")
    for key in ("stage", "output", "runtime"):
        value = Path(m[key])
        require(value.is_absolute() and value.parent == RUN_ROOT and value.name.startswith(m["run_token"] + "-"), "Unsafe token-scoped path: " + key)
        require(not value.is_symlink(), "Run directory cannot be a symlink")
    require(len({m[key] for key in ("stage", "output", "runtime")}) == 3, "Run paths must be distinct")
    require(path.parent == Path(m["stage"]), "Manifest must be a direct stage file")
    source = Path(m["source_root"])
    require(source == Path(m["stage"]) / "source" and not source.is_symlink(), "Source must be a distinct staged snapshot")
    require(m["command"] == expected_command(m, path), "Controller command differs from the bound staged entrypoint")
    ids = m["gpu_ids"]
    require(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < 8 for i in ids) and len(ids) == len(set(ids)), "Invalid GPU subset")
    require(set(m["gpu_uuids"]) == {str(i) for i in ids} and len(set(m["gpu_uuids"].values())) == len(ids), "GPU UUID mapping is incomplete or duplicated")
    require(all(re.fullmatch(r"GPU-[0-9a-f-]{36}", value) for value in m["gpu_uuids"].values()), "Malformed GPU UUID")
    limits = m["limits"]
    for key in ("runtime_seconds", "systemd_runtime_seconds", "min_available_ram_gib", "max_worker_rss_gib",
                "cgroup_memory_gib", "min_start_free_disk_gib", "max_worker_log_mib", "tasks_max", "load_stagger_seconds"):
        require(type(limits[key]) is int and limits[key] > 0, "Limit must be a positive integer: " + key)
    require(60 < limits["runtime_seconds"] <= 86400, "Controller deadline must be at most 24 hours")
    require(limits["runtime_seconds"] + 90 <= limits["systemd_runtime_seconds"] <= limits["runtime_seconds"] + 900, "Independent deadline must leave bounded cleanup margin")
    require(limits["min_available_ram_gib"] >= 192 and limits["max_worker_rss_gib"] <= 256, "Shared host RAM limits are unsafe")
    require(limits["max_worker_rss_gib"] + 8 <= limits["cgroup_memory_gib"] <= 384, "Invalid independent cgroup memory ceiling")
    require(limits["min_start_free_disk_gib"] >= 64 and limits["max_worker_log_mib"] <= 128, "Insufficient disk reserve or unbounded worker log")
    require(limits["tasks_max"] <= 2048 and 10 <= limits["load_stagger_seconds"] <= 60, "Invalid process/stagger limits")
    require(cpu_set(m["supervisor_cpu_set"]) == {94}, "Supervisor must use reviewed CPU 94")
    require(isinstance(m["runtime_versions"], dict) and m["runtime_versions"], "Missing dependency versions")
    require(isinstance(m["bound_files"], dict) and m["bound_files"], "Missing immutable source/input bindings")
    for filename, info in m["bound_files"].items():
        file = Path(filename)
        require(file.is_absolute() and type(info["size_bytes"]) is int and info["size_bytes"] >= 0 and re.fullmatch(r"[0-9a-f]{64}", info["sha256"]), "Malformed bound file entry")
        if check_files:
            require(file.is_file() and file.stat().st_size == info["size_bytes"] and sha256(file) == info["sha256"], "Bound source/input changed: " + filename)
    workers = m["workers"]
    require(isinstance(workers, list) and len(workers) == len(ids) and {w["gpu_id"] for w in workers} == set(ids), "One worker per selected GPU is required")
    require(len({w["name"] for w in workers}) == len(workers), "Duplicate worker name")
    require(len({w["success_file"] for w in workers}) == len(workers), "Workers must have distinct output directories")
    for worker in workers:
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", worker["name"]), "Unsafe worker name")
        expected_cpus = {96 + 2 * worker["gpu_id"], 97 + 2 * worker["gpu_id"]}
        require(cpu_set(worker["cpu_set"]) == expected_cpus, "Worker CPU allocation differs from reviewed cores 96–111")
        command = worker["command"]
        require(isinstance(command, list) and len(command) == 4 and all(isinstance(a, str) and a and "\x00" not in a for a in command), "Worker command must be an explicit argv list")
        require(command[:3] == [m["python"], str(source / "infra/gpu03/direction_discovery/engine.py"), "--task"], "Worker must call the bound discovery engine --task entrypoint")
        require(command[1] in m["bound_files"] and command[3] in m["bound_files"], "Worker source/task file is not hash bound")
        success = Path(worker["success_file"])
        require(not success.is_absolute() and ".." not in success.parts and success.name == "SUCCESS.json", "Unsafe worker success path")
        require(isinstance(worker["success_expect"], dict) and "requests" in worker["success_expect"] and "mode" in worker["success_expect"], "Worker success must bind request count and mode")
        require(type(worker["success_expect"]["requests"]) is int and worker["success_expect"]["requests"] >= 1, "Worker request count must be positive")
        require(not {"status", "run_token", "worker_name"}.intersection(worker["success_expect"]), "Success expectations cannot override identity/status")
        if check_files:
            task = json.loads(Path(command[3]).read_text())
            require(task["run_token"] == m["run_token"] and task["worker_name"] == worker["name"], "Task identity differs from assigned worker")
            require(Path(task["output"]) == (Path(m["output"]) / success).parent, "Task output differs from bound success directory")
            require(len(task["requests"]) == worker["success_expect"]["requests"] and task["mode"] == worker["success_expect"]["mode"], "Task count/mode differs from success contract")
            require(0 < task["deadline_seconds"] <= limits["runtime_seconds"], "Worker deadline exceeds campaign budget")
    if check_source:
        require(source.is_dir(), "Staged source is missing")
        source_files = []
        for file in source.rglob("*"):
            require(not file.is_symlink(), "Source snapshot contains a symlink")
            if file.is_file():
                require(file.name not in (".DS_Store",) and file.suffix != ".pyc" and "__pycache__" not in file.parts, "Source snapshot contains nonportable cache files")
                require(str(file) in m["bound_files"], "Unbound file in source snapshot: " + str(file))
                require(file.stat().st_uid == os.getuid() and not file.stat().st_mode & 0o222, "Source must be owned and immutable before launch")
                source_files.append(file)
        require(source_files, "Staged source is empty")
    if m["host"] in ("gpu-02", "gpu-01") and check_files:
        from infra.gpu03.direction_discovery import remote_generation
        remote_generation.validate_manifest(m)
    return m


def validate_worker_success(m, worker):
    path = Path(m["output"]) / worker["success_file"]
    require(path.is_file() and not path.is_symlink(), "Worker did not write its bound success receipt: " + worker["name"])
    value = json.loads(path.read_text())
    expected = {"status": "succeeded", "run_token": m["run_token"], "worker_name": worker["name"], **worker["success_expect"]}
    require(all(key in value and type(value[key]) is type(wanted) and value[key] == wanted for key, wanted in expected.items()), "Worker receipt identity/count/status mismatch: " + worker["name"])
    return {"worker": worker["name"], "path": worker["success_file"], "sha256": sha256(path), "expected": expected}


def resource_state(m, active, *, settling=True):
    state = raw.safety(m, active, allow_settling=settling)
    for worker in m["workers"]:
        log = Path(m["runtime"]) / (worker["name"] + ".log")
        require(not log.exists() or log.stat().st_size <= m["limits"]["max_worker_log_mib"] * 1024**2, "Worker log exceeded bound: " + worker["name"])
    return state


def qualify_idle(m, journal, deadline, *, clock=time.monotonic, sleep=time.sleep):
    start = clock()
    while True:
        require(clock() < deadline, "Campaign deadline exhausted during idle qualification")
        state = resource_state(m, [])
        append_json(journal, {"at": time.time(), "qualification_elapsed_seconds": clock() - start, **state})
        if clock() - start >= QUALIFICATION_SECONDS:
            return state
        sleep(min(INTERVAL, QUALIFICATION_SECONDS - (clock() - start)))


def verify_release(m, journal, *, timeout=60, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while True:
        inventory = raw.gpu_snapshot()
        # Never grant a settling exception to a foreign process.
        raw.check_devices(m, inventory, settling=set(m["gpu_ids"]))
        try:
            raw.check_devices(m, inventory)
            state = {"verified": True, "gpu_ids": m["gpu_ids"], "gpu_inventory": inventory, "at": time.time()}
            append_json(journal, {"release": state})
            return state
        except ValueError:
            require(clock() < deadline, "Selected GPUs did not return to verified idleness")
            sleep(min(INTERVAL, deadline - clock()))


def terminate_owned(processes):
    """Signal only still-live process groups created by this supervisor."""
    for process in processes:
        if process.poll() is None:
            require(os.getpgid(process.pid) == process.pid and Path(f"/proc/{process.pid}").stat().st_uid == os.getuid(), "Owned worker process identity changed before cleanup")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
        time.sleep(0.25)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    require(all(process.poll() is not None for process in processes), "An owned worker remains alive")


def _artifact_files(root):
    entries = {}
    for file in sorted(root.rglob("*")):
        require(not file.is_symlink(), "Output package contains a symlink")
        if file.is_file() and file.name != "artifact_manifest.json":
            entries[str(file.relative_to(root))] = {"size_bytes": file.stat().st_size, "sha256": sha256(file)}
    return entries


def supervise(path, digest):
    m = load_manifest(path, digest)
    require(socket.gethostname() == m["host"] and pwd.getpwuid(os.getuid()).pw_name == OWNER, "Supervisor is on the wrong host or user")
    require({key: importlib.metadata.version(key) for key in m["runtime_versions"]} == m["runtime_versions"], "Dependency runtime changed")
    if m["host"] in ("gpu-02", "gpu-01"):
        from infra.gpu03.direction_discovery import remote_generation
        remote_generation.validate_consumption(m, digest)
    original_affinity = os.sched_getaffinity(0)
    assigned_cpus = cpu_set(m["supervisor_cpu_set"]).union(*(cpu_set(w["cpu_set"]) for w in m["workers"]))
    require(assigned_cpus <= original_affinity, "Reviewed CPUs are unavailable in the current affinity")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Supervisor must not have a CUDA device")
    output, runtime = Path(m["output"]), Path(m["runtime"])
    require(not output.exists() and not runtime.exists(), "Preserve prior attempt: existing output/runtime requires an explicit journal recovery plan")
    require(shutil.disk_usage(output.parent).free >= m["limits"]["min_start_free_disk_gib"] * 1024**3, "Insufficient startup disk space")
    output.mkdir(mode=0o700); runtime.mkdir(mode=0o700)
    identity = {"run_token": m["run_token"], "manifest_sha256": digest}
    exclusive_json(output / "run_identity.json", identity)
    exclusive_json(runtime / "run_identity.json", identity)
    shutil.copyfile(path, output / "reviewed_manifest.json")
    os.sched_setaffinity(0, cpu_set(m["supervisor_cpu_set"]))
    processes, active, locks = [], [], []
    started = time.monotonic()
    deadline = started + m["limits"]["runtime_seconds"]
    journal = runtime / "resource_journal.jsonl"

    def interrupted(_signum, _frame):
        raise RuntimeError("Supervisor interrupted; outputs and journals are preserved")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        lock_root = RUN_ROOT / "triplet_gpu_locks"
        require(not lock_root.is_symlink(), "Unsafe shared GPU lock directory")
        lock_root.mkdir(mode=0o700, exist_ok=True)
        require(lock_root.stat().st_uid == os.getuid(), "Shared GPU lock directory has another owner")
        for gpu in m["gpu_ids"]:
            fd = os.open(lock_root / f"gpu_{gpu}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            lock = os.fdopen(fd, "a+")
            locks.append(lock)
            require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_uid == os.getuid(), "GPU lock is not an owned regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        qualify_idle(m, runtime / "qualification.jsonl", deadline)
        for index, worker in enumerate(m["workers"]):
            require(time.monotonic() < deadline, "Campaign deadline exhausted before worker launch")
            append_json(journal, {"at": time.time(), **resource_state(m, active)})
            command = ["taskset", "-c", worker["cpu_set"], "nice", "-n", "10", "ionice", "-c", "2", "-n", "7", *worker["command"]]
            with (runtime / (worker["name"] + ".log")).open("xb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           cwd=m["source_root"], env=raw.worker_env(worker["gpu_id"]))
            processes.append(process); active.append((process, worker["gpu_id"]))
            append_json(runtime / "processes.jsonl", {"worker": worker["name"], "gpu_id": worker["gpu_id"],
                                                       "pid": process.pid, "at": time.time(), "command": command})
            if index + 1 < len(m["workers"]):
                stagger_until = time.monotonic() + m["limits"]["load_stagger_seconds"]
                while time.monotonic() < stagger_until:
                    require(time.monotonic() < deadline, "Campaign deadline exhausted during model staggering")
                    require(all(p.poll() in (None, 0) for p in processes), "Worker failed during model staggering")
                    append_json(journal, {"at": time.time(), **resource_state(m, active)})
                    time.sleep(max(0, min(INTERVAL, stagger_until - time.monotonic())))
        while any(process.poll() is None for process in processes):
            require(time.monotonic() < deadline, "Campaign deadline exhausted")
            require(all(process.poll() in (None, 0) for process in processes), "Worker exited nonzero")
            append_json(journal, {"at": time.time(), **resource_state(m, active)})
            time.sleep(INTERVAL)
        require(all(process.returncode == 0 for process in processes), "A worker exited nonzero")
        require(time.monotonic() < deadline, "Workers finished after the campaign deadline")
        append_json(journal, {"at": time.time(), **resource_state(m, active)})
        receipts = [validate_worker_success(m, worker) for worker in m["workers"]]
        release = verify_release(m, journal)
        require(time.monotonic() < deadline, "Campaign deadline exhausted before verified release")
        exclusive_json(output / "gpu_release.json", release)
        if m["phase"] in ("fixed_cache_core", "fixed_cache_auxiliary"):
            import cache_package
            semantic = cache_package.finalize(m, output)
            exclusive_json(output / "cache_semantic_verification.json", semantic)
            require(time.monotonic() < deadline, "Campaign deadline exhausted while finalizing fixed cache")
        exclusive_json(output / "campaign_summary.json", {
            **identity, "status": "succeeded", "phase": m["phase"], "gpu_ids": m["gpu_ids"],
            "worker_exit_codes": [process.returncode for process in processes], "worker_receipts": receipts,
            "gpu_release_verified": True, "elapsed_seconds": time.monotonic() - started,
        })
        shutil.copytree(runtime, output / "execution")
        exclusive_json(output / "artifact_manifest.json", {"algorithm": "sha256", "files": _artifact_files(output)})
        print(canonical({"status": "producer_succeeded", **identity, "output": m["output"]}), flush=True)
    except BaseException as error:
        cleanup = {"owned_workers_released": False, "gpu_release_verified": False}
        try:
            terminate_owned(processes)
            cleanup["owned_workers_released"] = True
        except BaseException as cleanup_error:
            cleanup["process_cleanup_error"] = str(cleanup_error)
        try:
            cleanup["release"] = verify_release(m, journal, timeout=30)
            cleanup["gpu_release_verified"] = True
        except BaseException as cleanup_error:
            cleanup["gpu_release_error"] = str(cleanup_error)
        exclusive_json(output / "FAILURE.json", {**identity, "error_type": type(error).__name__, "error": str(error),
                                                "all_existing_outputs_preserved": True, "cleanup": cleanup})
        raise
    finally:
        for lock in locks:
            lock.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        os.sched_setaffinity(0, original_affinity)


def receipt(path, digest):
    m = load_manifest(path, digest, check_files=False, check_source=False)
    value = {"run_token": m["run_token"], "manifest_sha256": digest,
             "service_result": os.environ.get("SERVICE_RESULT", "unknown"),
             "exit_code_kind": os.environ.get("EXIT_CODE", "unknown"),
             "exit_status": os.environ.get("EXIT_STATUS", "unknown"),
             "invocation_id": os.environ.get("INVOCATION_ID", "unknown"),
             "producer_summary_present": (Path(m["output"]) / "campaign_summary.json").is_file(),
             "failure_present": (Path(m["output"]) / "FAILURE.json").is_file(), "at": time.time()}
    exclusive_json(Path(m["stage"]) / "control/supervisor_exit.json", value)


def launch(path, digest, *, permit_path=None, permit_sha256=None):
    m = load_manifest(path, digest)
    require(socket.gethostname() == m["host"] and pwd.getpwuid(os.getuid()).pw_name == OWNER, "Launcher is on the wrong host/user")
    if m["host"] in ("gpu-02", "gpu-01"):
        require(permit_path is not None and permit_sha256 is not None, "Remote launch requires the exact externally committed permit")
        from infra.gpu03.direction_discovery import remote_generation
        remote_generation.validate_launch_permit(m, digest, permit_path, permit_sha256, consume=True)
    else:
        require(permit_path is None and permit_sha256 is None, "Local launch cannot use a remote permit")
    # A pre-launch snapshot is recorded; the service independently qualifies 60s.
    inventory = raw.gpu_snapshot()
    raw.check_devices(m, inventory)
    control = Path(m["stage"]) / "control"
    control.mkdir(mode=0o700, exist_ok=False)
    exclusive_json(control / "launch_intent.json", {"run_token": m["run_token"], "manifest_sha256": digest,
                   "authorization": m["authorization"], "command": m["command"], "gpu_inventory": inventory, "at": time.time()})
    receipt_command = [m["python"], str(Path(m["source_root"]) / "infra/gpu03/direction_discovery/supervisor.py"),
                       "--receipt", "--manifest", str(path), "--manifest-sha256", digest]
    command = ["systemd-run", "--user", "--quiet", "--unit", m["run_token"], "--service-type=exec",
               "--property=RuntimeMaxSec=" + str(m["limits"]["systemd_runtime_seconds"]),
               "--property=TimeoutStopSec=90", "--property=KillMode=control-group",
               "--property=MemoryMax=" + str(m["limits"]["cgroup_memory_gib"]) + "G",
               "--property=TasksMax=" + str(m["limits"]["tasks_max"]), "--property=LimitNOFILE=4096",
               "--property=UMask=0077", "--property=ExecStopPost=" + shlex.join(receipt_command),
               "--property=StandardOutput=append:" + str(control / "supervisor.log"), "--property=StandardError=inherit"]
    # systemd --setenv adds variables but does not remove the user manager's
    # ambient credentials. env -i is required for the actual controller process.
    command += ["/usr/bin/env", "-i", *[key + "=" + value for key, value in raw.worker_env(None).items()],
                *m["command"], "--manifest-sha256", digest]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    exclusive_json(control / "launch_result.json", {"command": command, "returncode": result.returncode,
                                                   "stdout": result.stdout, "stderr": result.stderr})
    require(result.returncode == 0, "Independent systemd launch failed: " + result.stderr)
    try:
        state = subprocess.run(["systemctl", "--user", "show", m["run_token"] + ".service",
                                "--property=ActiveState,SubState,MainPID,ControlGroup,RuntimeMaxUSec,MemoryMax,KillMode,InvocationID"],
                               capture_output=True, text=True, timeout=15, check=True)
        fields = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
        require(fields.get("ActiveState") == "active" and fields.get("SubState") == "running" and fields.get("MainPID", "0").isdigit() and int(fields["MainPID"]) > 0, "Supervisor was not positively verified active")
        require(fields.get("RuntimeMaxUSec") not in (None, "", "infinity", "0") and fields.get("ControlGroup") and fields.get("KillMode") == "control-group", "Independent deadline/cgroup verification failed")
    except BaseException as error:
        # systemd-run returned zero for this freshly created exact-token unit.
        # Stop only that owned unit if its independent safety cannot be verified.
        stopped = subprocess.run(["systemctl", "--user", "stop", m["run_token"] + ".service"],
                                 capture_output=True, text=True, timeout=100)
        exclusive_json(control / "launch_verification_failure.json", {"error": str(error),
                       "exact_unit_stop_returncode": stopped.returncode, "stderr": stopped.stderr})
        raise
    exclusive_json(control / "service_started.json", {"at": time.time(), "unit": m["run_token"] + ".service", "fields": fields})
    return {"status": "launched", "run_token": m["run_token"], "manifest_sha256": digest, "output": m["output"], "service": fields}


def verify(path, digest):
    m = load_manifest(path, digest)
    output, control = Path(m["output"]), Path(m["stage"]) / "control"
    status = json.loads((control / "supervisor_exit.json").read_text())
    require(status.get("run_token") == m["run_token"] and status.get("manifest_sha256") == digest and
            status.get("service_result") == "success" and status.get("exit_code_kind") == "exited" and str(status.get("exit_status")) == "0" and
            status.get("producer_summary_present") is True and status.get("failure_present") is False, "Independent service exit receipt is not successful")
    require(re.fullmatch(r"[0-9a-f]{32}", status.get("invocation_id", "")), "Missing actual systemd invocation identity")
    summary = json.loads((output / "campaign_summary.json").read_text())
    require(not (output / "FAILURE.json").exists() and summary["status"] == "succeeded" and summary["manifest_sha256"] == digest and
            summary["run_token"] == m["run_token"] and summary["gpu_release_verified"] is True and
            summary["worker_exit_codes"] == [0] * len(m["workers"]), "Producer summary does not prove success")
    receipts = [validate_worker_success(m, worker) for worker in m["workers"]]
    require(summary["worker_receipts"] == receipts, "Worker receipt hashes changed after producer exit")
    artifacts = json.loads((output / "artifact_manifest.json").read_text())
    require(artifacts == {"algorithm": "sha256", "files": _artifact_files(output)}, "Independent output package hashes differ")
    release = json.loads((output / "gpu_release.json").read_text())
    require(release.get("verified") is True and release.get("gpu_ids") == m["gpu_ids"], "Missing positively verified GPU release")
    if m["phase"] in ("fixed_cache_core", "fixed_cache_auxiliary"):
        import cache_package
        semantic = cache_package.inspect(output)
        expected_semantic = json.loads((output / "cache_semantic_verification.json").read_text())
        expected_semantic['index_joined_to_artifact_manifest'] = True
        require(semantic.get('index_joined_to_artifact_manifest') is True and semantic == expected_semantic,
                "Independent fixed-cache semantics differ from producer")
    return {"status": "verified", "run_token": m["run_token"], "manifest_sha256": digest,
            "workers": len(m["workers"]), "artifact_files": len(artifacts["files"]), "gpu_release_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("launch", "supervise", "verify", "receipt"):
        mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--permit-sha256")
    args = parser.parse_args()
    operation = launch if args.launch else supervise if args.supervise else verify if args.verify else receipt
    if args.launch:
        value = operation(args.manifest, args.manifest_sha256, permit_path=args.permit, permit_sha256=args.permit_sha256)
    else:
        require(args.permit is None and args.permit_sha256 is None, "Permit CLI fields are only valid for launch")
        value = operation(args.manifest, args.manifest_sha256)
    if value is not None:
        print(canonical(value), flush=True)


if __name__ == "__main__":
    main()
