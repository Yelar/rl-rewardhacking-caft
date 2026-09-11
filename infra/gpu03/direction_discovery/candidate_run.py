#!/usr/bin/env python3
"""Manifest-bound, CPU-only candidate-discovery supervisor for gpu-04.

Preparation and supervisor qualification do not start a scientific fit. The
separate --launch command is used only after the exact manifest is reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

try:
    from . import candidates as c
except ImportError:
    import candidates as c


ROOT = Path("/scratch/researcher/codex_runs")
USER = "researcher"
UID = 3357
HOST = "gpu-04"
PYTHON = "/scratch/researcher/rl-rewardhacking-gpu03-runtime/venv/bin/python"
QUALIFICATION_LAYER = 20
GIB = 1024**3


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(c.canonical(value) + "\n")
        stream.flush()


def host_identity():
    c.require(socket.gethostname().split(".")[0] == HOST and os.getuid() == UID,
              "candidate operations require gpu-04 and the reviewed user")


def cpu_environment():
    return {"PATH": "/usr/bin:/bin", "HOME": "/home/" + USER, "USER": USER, "LOGNAME": USER,
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
            "XDG_RUNTIME_DIR": "/run/user/" + str(UID), "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/" + str(UID) + "/bus"}


def clean_command(command):
    return ["/usr/bin/env", "-i", *[k + "=" + v for k, v in cpu_environment().items()], *command]


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable missing")


def process_stat(pid):
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        fields = data[data.rfind(")") + 2:].split()
        return {"pid": int(pid), "ppid": int(fields[1]), "start_ticks": int(fields[19]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), "state": fields[0]}
    except FileNotFoundError:
        return None


def descendants(pids):
    result = set(pids)
    snapshots = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process_stat(int(path.name))
            if info is not None:
                snapshots.append(info)
    while True:
        old = len(result)
        result.update(p["pid"] for p in snapshots if p["ppid"] in result)
        if len(result) == old:
            break
    return [p for p in snapshots if p["pid"] in result]


def resource_check(m, *, process_ids=(), starting=False):
    available = mem_available()
    disk = shutil.disk_usage(m["stage"]).free
    load = os.getloadavg()[0]
    tree = descendants([os.getpid(), *process_ids])
    rss = sum(p["rss_bytes"] for p in tree)
    limits = m["limits"]
    c.require(available >= limits["min_start_available_ram_bytes" if starting else "min_available_ram_bytes"],
              "host available RAM below reviewed floor")
    c.require(disk >= limits["min_start_free_disk_bytes" if starting else "min_free_disk_bytes"], "scratch free space below reviewed floor")
    c.require(load <= limits["max_host_load1"], "host CPU load exceeds reviewed bound")
    c.require(rss <= limits["max_aggregate_rss_bytes"], "candidate process RSS exceeds reviewed bound")
    return {"at": time.time(), "available_ram_bytes": available, "free_disk_bytes": disk,
            "host_load1": load, "aggregate_rss_bytes": rss, "processes": tree}


def bound_info(path):
    path = Path(path)
    return {"sha256": c.sha256_file(path), "size_bytes": path.stat().st_size}


def validate_plan(plan, m):
    s, f = m["science"], plan["fit"]
    c.require(plan["host"] == HOST and plan["no_training"] is True and
              f["layers"] == s["layers"] and f["windows"] == s["windows"] and f["seed"] == s["seed"] and
              f["problem_bootstrap"] == s["bootstrap"] and f["pca_rank"] == s["pca_rank"] and
              f["pca_oversample"] == s["pca_oversample"] and f["pca_power_iters"] == s["pca_power_iters"] and
              plan["budget"]["maximum_candidate_cpu_wall_seconds"] == 14400,
              "master experiment plan differs from candidate fit configuration")
    c.require(plan["inputs"]["raw_artifact_manifest_sha256"] == m["raw_manifest_sha256"] and
              plan["inputs"]["prepared_records_sha256"] == c.sha256_file(m["prepared_records"]) and
              plan["inputs"]["exclusions_sha256"] == c.sha256_file(m["exclusions"]), "master plan input hashes differ")
    if "raw_package" in plan["inputs"]:
        c.require(plan["inputs"]["raw_package"] == m["raw_package"], "master plan raw package path differs")


def load_manifest(path, digest, *, check_files=True):
    path = Path(path)
    c.require(c.sha256_file(path) == digest, "reviewed manifest digest mismatch")
    m = json.loads(path.read_text())
    c.require(m["schema_version"] == 1 and m["purpose"] == "checkpoint60_cpu_candidate_discovery", "wrong manifest purpose")
    c.require(m["host"] == HOST and m["uid"] == UID and m["python"] == PYTHON, "wrong execution identity")
    token = m["run_token"]
    c.require(re.fullmatch(r"codex-candidates-[a-z0-9-]{8,80}", token) is not None, "invalid exact run token")
    stage = Path(m["stage"])
    c.require(stage == ROOT / token and path == stage / "reviewed_manifest.json" and
              m["output"] == str(stage / "results") and m["control"] == str(stage / "control"), "run paths escaped exact token")
    science = m["science"]
    c.require(science == {"layers": list(range(36)), "windows": list(c.WINDOWS), "seed": 6001,
                          "bootstrap": 200, "pca_rank": 10, "pca_oversample": 14,
                          "pca_power_iters": 2, "qualification_layer": QUALIFICATION_LAYER,
                          "fit_problems": 111, "fit_records": 333}, "scientific candidate configuration changed")
    c.require(m["mode"] in ("production", "supervisor_qualification"), "unknown launch mode")
    c.require(1 <= m["workers"] <= 8 and len(m["worker_cpus"]) == m["workers"], "invalid CPU worker count")
    cpus = [m["controller_cpu"], m["verifier_cpu"], *m["worker_cpus"]]
    c.require(len(cpus) == len(set(cpus)) and all(type(i) is int and i >= 0 for i in cpus), "CPU assignments overlap")
    c.require(m["limits"]["systemd_deadline_seconds"] == (14400 if m["mode"] == "production" else 90), "unexpected independent deadline")
    c.require(m["limits"]["max_aggregate_rss_bytes"] <= 12 * GIB and
              m["limits"]["systemd_memory_max_bytes"] <= 16 * GIB, "memory budget broadened")
    c.require(m["command"] == clean_command([PYTHON, str(stage / "source" / "candidate_run.py"), "--supervise", str(path)]), "supervisor command mismatch")
    if check_files:
        for filename, info in m["bound_files"].items():
            c.verify_file(filename, info)
        validate_plan(json.loads(Path(m["experiment_plan"]).read_text()), m)
    return m


def build_manifest(args):
    host_identity()
    stage = Path(args.stage)
    c.require(stage.parent == ROOT and stage.name == args.run_token and stage.is_dir(), "stage must be fresh exact-token directory")
    c.require(stage.stat().st_uid == UID and not stage.is_symlink(), "unsafe stage ownership")
    for fresh in (stage / "control", stage / "results", stage / "reviewed_manifest.json"):
        c.require(not fresh.exists(), "review stage already launched or frozen")
    source = stage / "source"
    required = ("candidates.py", "candidate_run.py", "test_candidates.py", "test_candidate_run.py")
    c.require(all((source / name).is_file() for name in required), "reviewed source files missing")
    cpus = list(args.worker_cpus)
    c.require(len(cpus) <= 8 and set([args.controller_cpu, args.verifier_cpu, *cpus]) <= os.sched_getaffinity(0), "requested CPU affinity unavailable")
    raw = Path(args.raw_package)
    raw_manifest = json.loads((raw / "artifact_manifest.json").read_text())
    summary = json.loads((raw / "extraction_summary.json").read_text())
    for relative in ("activation_index.jsonl", "extraction_summary.json"):
        c.verify_file(raw / relative, raw_manifest["files"][relative])
    c.require(summary["status"] == "succeeded" and summary["records"] == 561 and summary["layers"] == 36 and
              summary["hidden_size"] == 2560 and summary["raw_activations_retained"], "raw source is not verified complete campaign")
    c.verify_file(args.prepared_records, raw_manifest["files"]["input/prepared_records.jsonl"])
    fit, dataset = c.validate_records(c.read_jsonl(args.prepared_records), json.loads(Path(args.exclusions).read_text()))
    c.require(len(fit) == 333 and dataset["fitting_problems"] == 111, "disputed-record freeze differs from reviewed 111/333 fit set")
    for row in fit:
        for window in c.WINDOWS:
            c.valid_positions(row, window)
    files = [*source.iterdir(), raw / "artifact_manifest.json", raw / "activation_index.jsonl",
             raw / "extraction_summary.json", Path(args.prepared_records), Path(args.exclusions),
             Path(args.experiment_plan), Path(PYTHON), Path(PYTHON).parent.parent / "pyvenv.cfg"]
    files.extend(p for p in Path(args.tokenizer).iterdir() if p.name in
                 ("tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"))
    c.require((Path(args.tokenizer) / "tokenizer.json") in files, "tokenizer source missing")
    files = [p for p in files if p.is_file()]
    mode = "supervisor_qualification" if args.qualification_only else "production"
    m = {"schema_version": 1, "purpose": "checkpoint60_cpu_candidate_discovery", "mode": mode,
         "authorization": "User requested direction-discovery and causal-ablation experiments on idle gpu-04 resources; no CAFT training.",
         "host": HOST, "uid": UID, "python": PYTHON, "run_token": args.run_token,
         "stage": str(stage), "output": str(stage / "results"), "control": str(stage / "control"),
         "workers": len(cpus), "worker_cpus": cpus, "controller_cpu": args.controller_cpu, "verifier_cpu": args.verifier_cpu,
         "raw_package": str(raw), "raw_manifest_sha256": c.sha256_file(raw / "artifact_manifest.json"),
         "prepared_records": args.prepared_records, "exclusions": args.exclusions, "tokenizer": args.tokenizer,
         "experiment_plan": args.experiment_plan,
         "git_commit": args.git_commit, "git_dirty_patch_sha256": args.git_dirty_patch_sha256,
         "dataset": dataset, "science": {"layers": list(range(36)), "windows": list(c.WINDOWS), "seed": 6001,
             "bootstrap": 200, "pca_rank": 10, "pca_oversample": 14, "pca_power_iters": 2,
             "qualification_layer": QUALIFICATION_LAYER, "fit_problems": 111, "fit_records": 333},
         "limits": {"systemd_deadline_seconds": 14400 if mode == "production" else 90,
             "internal_deadline_seconds": 14100 if mode == "production" else 60,
             "stop_grace_seconds": 90, "max_aggregate_rss_bytes": 12 * GIB, "systemd_memory_max_bytes": 16 * GIB,
             "min_start_available_ram_bytes": 128 * GIB, "min_available_ram_bytes": 96 * GIB,
             "min_start_free_disk_bytes": 16 * GIB, "min_free_disk_bytes": 8 * GIB,
             "max_host_load1": 96, "candidate_array_limit_mib": 1024},
         "runtime_versions": runtime_versions(), "bound_files": {str(p): bound_info(p) for p in sorted(set(files))},
         "command": clean_command([PYTHON, str(source / "candidate_run.py"), "--supervise", str(stage / "reviewed_manifest.json")])}
    m["preparation_inventory"] = resource_check(m, starting=True)
    m["numerical_preflight"] = c.numerical_preflight()
    validate_plan(json.loads(Path(args.experiment_plan).read_text()), m)
    c.write_json(stage / "reviewed_manifest.json", m)
    os.chmod(stage / "reviewed_manifest.json", 0o400)
    for p in source.iterdir():
        if p.is_file():
            os.chmod(p, 0o400)
    os.chmod(source, 0o500)
    digest = c.sha256_file(stage / "reviewed_manifest.json")
    load_manifest(stage / "reviewed_manifest.json", digest)
    print(c.canonical({"manifest": str(stage / "reviewed_manifest.json"), "manifest_sha256": digest,
                       "command": [PYTHON, str(source / "candidate_run.py"), "--launch", str(stage / "reviewed_manifest.json"),
                                   "--manifest-sha256", digest], "output": m["output"], "mode": mode}))


def runtime_versions():
    from importlib.metadata import version
    return {name: version(name) for name in ("numpy", "torch", "safetensors", "transformers", "tokenizers")}


def launch(manifest_path, digest):
    host_identity()
    m = load_manifest(manifest_path, digest)
    c.require(runtime_versions() == m["runtime_versions"], "candidate runtime versions changed")
    resource_check(m, starting=True)
    control = Path(m["control"])
    control.mkdir(mode=0o700, exist_ok=False)
    c.write_json(control / "launch_intent.json", {"run_token": m["run_token"], "manifest_sha256": digest,
                 "authorization": m["authorization"], "at": time.time(), "mode": m["mode"]})
    receipt = [PYTHON, str(Path(m["stage"]) / "source" / "candidate_run.py"), "--receipt", str(manifest_path), "--manifest-sha256", digest]
    command = ["systemd-run", "--user", "--quiet", "--unit", m["run_token"], "--service-type=exec",
               "--property=RuntimeMaxSec=" + str(m["limits"]["systemd_deadline_seconds"]),
               "--property=TimeoutStopSec=90", "--property=KillMode=control-group",
               "--property=MemoryMax=" + str(m["limits"]["systemd_memory_max_bytes"]), "--property=TasksMax=128",
               "--property=LimitNOFILE=2048", "--property=UMask=0077",
               "--property=ExecStopPost=" + shlex.join(receipt),
               "--property=StandardOutput=append:" + str(control / "supervisor.log"), "--property=StandardError=inherit"]
    for key, value in cpu_environment().items():
        command.append("--setenv=" + key + "=" + value)
    command += m["command"] + ["--manifest-sha256", digest]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, env=cpu_environment())
    c.write_json(control / "launch_result.json", {"command": command, "returncode": result.returncode,
                 "stdout": result.stdout, "stderr": result.stderr, "at": time.time()})
    c.require(result.returncode == 0, "user-systemd candidate launch failed: " + result.stderr)
    dispatch = verify_dispatch(m)
    c.write_json(control / "dispatch_verified.json", dispatch)
    print(c.canonical({"status": "dispatched", "unit": m["run_token"] + ".service", "manifest_sha256": digest,
                       "mode": m["mode"], "output": m["output"]}))


def verify_dispatch(m):
    deadline = time.monotonic() + 15
    unit = m["run_token"] + ".service"
    while True:
        result = subprocess.run(["systemctl", "--user", "show", unit, "--property=ActiveState", "--property=SubState",
                                 "--property=MainPID", "--property=ControlGroup"], env=cpu_environment(),
                                capture_output=True, text=True, timeout=10)
        c.require(result.returncode == 0, "cannot verify user-systemd dispatch")
        props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        pid = int(props.get("MainPID", "0"))
        info = process_stat(pid) if pid > 0 else None
        if props.get("ActiveState") == "active" and info and props.get("ControlGroup", "").endswith("/" + unit):
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            expected = str(Path(m["stage"]) / "source" / "candidate_run.py").encode()
            if expected in command and b"--supervise" in command:
                return {"verified": True, "properties": props, "main_process": info, "command_matches_manifest": True, "at": time.time()}
        receipt_path = Path(m["control"]) / "supervisor_exit.json"
        if m["mode"] == "supervisor_qualification" and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            c.require(receipt["service_result"] == "success" and receipt["exit_code_kind"] == "exited" and
                      receipt["exit_status"] == "0", "short qualification exited unsuccessfully")
            return {"verified": True, "completed_before_dispatch_snapshot": True, "receipt": receipt, "at": time.time()}
        c.require(time.monotonic() < deadline, "active candidate PID/cgroup dispatch not positively verified")
        time.sleep(.25)


def receipt(manifest_path, digest):
    m = load_manifest(manifest_path, digest, check_files=False)
    c.write_json(Path(m["control"]) / "supervisor_exit.json", {"run_token": m["run_token"], "manifest_sha256": digest,
                 "service_result": os.environ.get("SERVICE_RESULT", "unknown"), "exit_code_kind": os.environ.get("EXIT_CODE", "unknown"),
                 "exit_status": os.environ.get("EXIT_STATUS", "unknown"), "invocation_id": os.environ.get("INVOCATION_ID", "unknown"),
                 "success_marker_present": (Path(m["output"]) / "SUCCESS.json").is_file(),
                 "failure_marker_present": (Path(m["output"]) / "FAILURE.json").is_file(), "at": time.time()})


def integrity_pass(m, output):
    raw = Path(m["raw_package"])
    manifest = json.loads((raw / "artifact_manifest.json").read_text())
    rows, _ = c.validate_records(c.read_jsonl(m["prepared_records"]), json.loads(Path(m["exclusions"]).read_text()))
    index = {r["record_id"]: r for r in c.read_jsonl(raw / "activation_index.jsonl")}
    files = {}
    for row in rows:
        for kind in ("h0", "h60"):
            entry = index[row["record_id"]]["models"][kind]
            relative = entry["tensor_path"]
            path = c.safe_child(raw, relative)
            info = manifest["files"][relative]
            c.require(info["sha256"] == entry["sha256"] and info["size_bytes"] == entry["size_bytes"], "raw index binding changed")
            before = c.file_identity(path)
            c.verify_file(path, info)
            c.require(c.file_identity(path) == before, "raw file changed during integrity scan")
            files[relative] = {"sha256": info["sha256"], "identity": before}
        if len(files) % 40 == 0:
            print(c.canonical({"phase": "integrity", "files_verified": len(files)}), flush=True)
    proof = {"schema_version": 1, "status": "verified", "raw_manifest_sha256": m["raw_manifest_sha256"],
             "prepared_records_sha256": c.sha256_file(m["prepared_records"]), "exclusion_manifest_sha256": c.sha256_file(m["exclusions"]),
             "fitting_record_ids": [r["record_id"] for r in rows], "files": files,
             "verified_bytes": sum(p["identity"]["size_bytes"] for p in files.values()), "at": time.time()}
    c.write_json(output, proof)
    os.chmod(output, 0o400)


def terminate_owned(processes):
    """Only groups created with start_new_session by this supervisor."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)


class Supervisor:
    def __init__(self, m, digest):
        self.m, self.digest = m, digest
        self.control = Path(m["control"])
        self.started = time.monotonic()
        self.processes = []
        self.handles = []
        self.identities = []

    def spawn(self, name, command, cpu):
        self.guard()
        handle = (self.control / (name + ".log")).open("x")
        self.handles.append(handle)
        # taskset provides an explicit auditable affinity argument; no preexec
        # Python callbacks, inherited credential variables, or shell involved.
        actual = ["/usr/bin/taskset", "-c", str(cpu), "/usr/bin/nice", "-n", "15", *command]
        p = subprocess.Popen(actual, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                             env=cpu_environment(), start_new_session=True)
        self.processes.append(p)
        identity = process_stat(p.pid)
        c.require(identity is not None, "worker disappeared before identity capture")
        self.identities.append(identity)
        append_json(self.control / "processes.jsonl", {"name": name, "identity": identity, "command": actual, "at": time.time()})
        return p

    def guard(self):
        c.require(time.monotonic() - self.started < self.m["limits"]["internal_deadline_seconds"], "internal candidate deadline reached")
        report = resource_check(self.m, process_ids=[p.pid for p in self.processes if p.poll() is None])
        append_json(self.control / "resources.jsonl", report)

    def wait(self, processes):
        while any(p.poll() is None for p in processes):
            self.guard()
            c.require(all(p.poll() in (None, 0) for p in processes), "candidate child exited unsuccessfully")
            time.sleep(2)
        c.require(all(p.returncode == 0 for p in processes), "candidate child exited unsuccessfully")

    def verify(self, package, name):
        target = self.control / (name + ".verification.json")
        command = [PYTHON, str(Path(self.m["stage"]) / "source" / "candidate_run.py"),
                   "--verify-package", str(package), "--verification-output", str(target)]
        self.wait([self.spawn(name + "-verify", command, self.m["verifier_cpu"])])
        result = json.loads(target.read_text())
        c.require(result["status"] == "verified", "independent candidate package verification failed")
        return result

    def release(self):
        terminate_owned(self.processes)
        for handle in self.handles:
            handle.close()
        remaining = []
        for original in self.identities:
            now = process_stat(original["pid"])
            if now and now["start_ticks"] == original["start_ticks"]:
                remaining.append(now)
        c.require(not remaining, "exact-token child process was not released")


def candidate_command(m, output, layers, receipt_path):
    s = m["science"]
    return [PYTHON, str(Path(m["stage"]) / "source" / "candidates.py"),
            "--raw-package", m["raw_package"], "--raw-manifest-sha256", m["raw_manifest_sha256"],
            "--prepared-records", m["prepared_records"], "--exclusions", m["exclusions"], "--tokenizer", m["tokenizer"],
            "--output", str(output), "--layers", *map(str, layers), "--seed", str(s["seed"]),
            "--bootstrap", str(s["bootstrap"]), "--pca-rank", str(s["pca_rank"]),
            "--pca-oversample", str(s["pca_oversample"]), "--pca-power-iters", str(s["pca_power_iters"]),
            "--max-memory-mib", str(m["limits"]["candidate_array_limit_mib"]), "--deadline-seconds", "13800",
            "--raw-integrity-receipt", str(receipt_path), "--raw-integrity-sha256", c.sha256_file(receipt_path)]


def layer_shards(workers):
    c.require(1 <= workers <= 8, "invalid CPU worker count")
    remaining = [l for l in range(36) if l != QUALIFICATION_LAYER]
    return [remaining[i::workers] for i in range(workers)]


def fanout_budget(qualification_seconds, workers, remaining_seconds):
    # A doubled serial-layer estimate includes normal CPU/IO contention; reserve
    # an additional ten minutes for all independent verifiers and final hashing.
    waves = (35 + workers - 1) // workers
    estimate = 2 * qualification_seconds * waves + 600
    c.require(estimate <= remaining_seconds, "qualification timing leaves insufficient conservative fanout budget")
    return {"qualification_fit_seconds": qualification_seconds, "worker_waves": waves,
            "safety_factor": 2, "verification_allowance_seconds": 600,
            "estimated_remaining_seconds": estimate, "remaining_deadline_seconds": remaining_seconds}


def acquire_cpu_lock(cpu):
    path = ROOT / f".codex-candidate-cpu-{cpu}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        c.require(info.st_uid == UID and stat.S_ISREG(info.st_mode), "unsafe CPU lock owner/type")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return os.fdopen(descriptor, "a+")
    except BaseException:
        os.close(descriptor)
        raise


def merge_packages(m, packages, integrity):
    output = Path(m["output"])
    catalog, seen_layers = [], []
    package_rows = []
    for package in packages:
        config = json.loads((package / "resolved_config.json").read_text())
        seen_layers.extend(config["layers"])
        c.require(config["seed"] == 6001 and config["bootstrap_replicates"] == 200 and
                  config["raw_manifest_sha256"] == m["raw_manifest_sha256"] and
                  config["exclusion_manifest_sha256"] == c.sha256_file(m["exclusions"]) and
                  config["source_sha256"] == c.sha256_file(Path(m["stage"]) / "source" / "candidates.py"), "candidate package configuration drift")
        relative = package.relative_to(output)
        for candidate in json.loads((package / "candidate_catalog.json").read_text())["candidates"]:
            updated = dict(candidate)
            updated["tensor_file"] = str(relative / candidate["tensor_file"])
            updated["report_file"] = str(relative / candidate["report_file"])
            catalog.append(updated)
        package_rows.append({"path": str(relative), "artifact_manifest_sha256": c.sha256_file(package / "artifact_manifest.json"), "layers": config["layers"]})
    c.require(sorted(seen_layers) == list(range(36)), "candidate packages omit or duplicate layers")
    c.require(len(catalog) == len({r["candidate_id"] for r in catalog}), "duplicate candidate IDs")
    c.write_json(output / "candidate_catalog.json", {"candidates": sorted(catalog, key=lambda r: r["candidate_id"])})
    c.write_json(output / "campaign_index.json", {"schema_version": 1, "packages": package_rows,
                 "raw_integrity_sha256": c.sha256_file(integrity), "layers": list(range(36)), "windows": list(c.WINDOWS)})
    return len(catalog)


def write_campaign_manifest(output):
    paths = sorted(p for p in output.rglob("*") if p.is_file())
    c.require(all(not p.is_symlink() for p in paths), "symlink in produced campaign")
    c.write_json(output / "artifact_manifest.json", {"algorithm": "sha256", "files": {str(p.relative_to(output)): bound_info(p) for p in paths}})


def supervise(manifest_path, digest):
    host_identity()
    m = load_manifest(manifest_path, digest)
    c.require(runtime_versions() == m["runtime_versions"], "runtime drift after launch")
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "supervisor must hide CUDA")
    os.sched_setaffinity(0, {m["controller_cpu"]})
    output = Path(m["output"])
    output.mkdir(mode=0o700, exist_ok=False)
    c.write_json(output / "reviewed_manifest.json", m)
    runner = Supervisor(m, digest)
    locks = []
    try:
        for cpu in [m["controller_cpu"], m["verifier_cpu"], *m["worker_cpus"]]:
            locks.append(acquire_cpu_lock(cpu))
        c.write_json(output / "RUNNING.json", {"run_token": m["run_token"], "manifest_sha256": digest, "pid": os.getpid(), "at": time.time()})
        if m["mode"] == "supervisor_qualification":
            probe = output / "cpu_probe.json"
            command = [PYTHON, str(Path(m["stage"]) / "source" / "candidate_run.py"), "--probe", str(probe)]
            runner.wait([runner.spawn("supervisor-qualification", command, m["worker_cpus"][0])])
            p = json.loads(probe.read_text())
            c.require(p["cuda_visible_devices"] == "" and p["affinity"] == [m["worker_cpus"][0]] and p["numerical_preflight"]["passed"], "CPU supervisor qualification failed")
            total = 0
        else:
            integrity = output / "raw_integrity_receipt.json"
            command = [PYTHON, str(Path(m["stage"]) / "source" / "candidate_run.py"), "--integrity", str(manifest_path),
                       "--manifest-sha256", digest, "--verification-output", str(integrity)]
            runner.wait([runner.spawn("raw-integrity", command, m["verifier_cpu"])])
            qualification = output / "qualification"
            qualification_start = time.monotonic()
            runner.wait([runner.spawn("candidate-qualification", candidate_command(m, qualification, [QUALIFICATION_LAYER], integrity), m["worker_cpus"][0])])
            qualification_seconds = time.monotonic() - qualification_start
            runner.verify(qualification, "candidate-qualification")
            budget = fanout_budget(qualification_seconds, m["workers"],
                                   m["limits"]["internal_deadline_seconds"] - (time.monotonic() - runner.started))
            c.write_json(output / "qualification_budget.json", budget)
            packages = [qualification]
            workers = []
            for i, (cpu, layers) in enumerate(zip(m["worker_cpus"], layer_shards(m["workers"]))):
                if layers:
                    package = output / f"worker_{i:02d}"
                    workers.append(runner.spawn(f"candidate-{i:02d}", candidate_command(m, package, layers, integrity), cpu))
                    packages.append(package)
            runner.wait(workers)
            for i, package in enumerate(packages[1:]):
                runner.verify(package, f"candidate-{i:02d}")
            total = merge_packages(m, packages, integrity)
        runner.release()
        c.write_json(output / "process_release.json", {"verified": True, "owned_child_identities": runner.identities, "at": time.time()})
        c.write_json(output / "SUCCESS.json", {"status": "succeeded", "mode": m["mode"], "run_token": m["run_token"],
                     "manifest_sha256": digest, "candidates": total, "layers": list(range(36)) if total else [],
                     "process_release_verified": True, "elapsed_seconds": time.monotonic() - runner.started})
        write_campaign_manifest(output)
    except BaseException as exc:
        cleanup_error = None
        try:
            runner.release()
        except Exception as cleanup:
            cleanup_error = str(cleanup)
        c.write_json(output / "FAILURE.json", {"status": "failed", "error": str(exc), "type": type(exc).__name__, "cleanup_error": cleanup_error,
                     "run_token": m["run_token"], "partial_outputs_preserved": True, "at": time.time()})
        raise
    finally:
        for lock in locks:
            lock.close()


def verify_campaign(output):
    output = Path(output)
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for relative, info in manifest["files"].items():
        c.verify_file(c.safe_child(output, relative), info)
    m = json.loads((output / "reviewed_manifest.json").read_text())
    digest = c.sha256_file(output / "reviewed_manifest.json")
    success = json.loads((output / "SUCCESS.json").read_text())
    receipt = json.loads((Path(m["control"]) / "supervisor_exit.json").read_text())
    c.require(success["status"] == "succeeded" and success["manifest_sha256"] == digest and
              receipt["manifest_sha256"] == digest and receipt["service_result"] == "success" and
              receipt["exit_code_kind"] == "exited" and receipt["exit_status"] == "0" and
              receipt["success_marker_present"] and not receipt["failure_marker_present"], "supervisor completion not positively verified")
    c.require(not (output / "FAILURE.json").exists(), "failed campaign cannot verify")
    for info in json.loads((output / "process_release.json").read_text())["owned_child_identities"]:
        now = process_stat(info["pid"])
        c.require(now is None or now["start_ticks"] != info["start_ticks"], "owned process is still running")
    if m["mode"] == "production":
        campaign = json.loads((output / "campaign_index.json").read_text())
        layers = []
        expected_catalog = []
        for package in campaign["packages"]:
            root = output / package["path"]
            c.require(c.sha256_file(root / "artifact_manifest.json") == package["artifact_manifest_sha256"], "child manifest mismatch")
            c.verify_package(root)
            layers.extend(package["layers"])
            for item in json.loads((root / "candidate_catalog.json").read_text())["candidates"]:
                updated = dict(item)
                updated["tensor_file"] = str(Path(package["path"]) / item["tensor_file"])
                updated["report_file"] = str(Path(package["path"]) / item["report_file"])
                expected_catalog.append(updated)
        c.require(sorted(layers) == list(range(36)), "incomplete or duplicate final layer coverage")
        actual = json.loads((output / "candidate_catalog.json").read_text())["candidates"]
        c.require(actual == sorted(expected_catalog, key=lambda r: r["candidate_id"]), "combined catalog differs from verified packages")
    return {"status": "verified", "mode": m["mode"], "artifact_manifest_sha256": c.sha256_file(output / "artifact_manifest.json"),
            "run_token": m["run_token"], "candidates": success["candidates"], "process_release_verified": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-manifest", action="store_true")
    action.add_argument("--launch")
    action.add_argument("--supervise")
    action.add_argument("--receipt")
    action.add_argument("--integrity")
    action.add_argument("--verify-package")
    action.add_argument("--verify-campaign")
    action.add_argument("--probe")
    p.add_argument("--manifest-sha256")
    p.add_argument("--verification-output")
    for name in ("stage", "run-token", "raw-package", "prepared-records", "exclusions", "tokenizer", "git-commit", "git-dirty-patch-sha256", "experiment-plan"):
        p.add_argument("--" + name)
    p.add_argument("--worker-cpus", nargs="+", type=int, default=list(range(120, 128)))
    p.add_argument("--controller-cpu", type=int, default=119)
    p.add_argument("--verifier-cpu", type=int, default=118)
    p.add_argument("--qualification-only", action="store_true")
    args = p.parse_args()
    if args.build_manifest:
        build_manifest(args)
    elif args.launch:
        launch(args.launch, args.manifest_sha256)
    elif args.supervise:
        supervise(args.supervise, args.manifest_sha256)
    elif args.receipt:
        receipt(args.receipt, args.manifest_sha256)
    elif args.integrity:
        m = load_manifest(args.integrity, args.manifest_sha256)
        integrity_pass(m, args.verification_output)
    elif args.verify_package:
        result = c.verify_package(args.verify_package)
        c.write_json(args.verification_output, result)
        print(c.canonical(result))
    elif args.verify_campaign:
        result = verify_campaign(args.verify_campaign)
        if args.verification_output:
            c.write_json(args.verification_output, result)
        print(c.canonical(result))
    elif args.probe:
        c.write_json(args.probe, {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "affinity": sorted(os.sched_getaffinity(0)), "runtime_versions": runtime_versions(),
                     "numerical_preflight": c.numerical_preflight()})


if __name__ == "__main__":
    main()
