"""Small stdlib supervisor for the authorized gpu-03 discovery workers.

CLI: python -B resource_guard.py --manifest ABS --sha256 SHA
The caller binds source, scientific requests, sandbox policy and outer systemd
limits. This guard certifies process/resource lifecycle, never scientific success.
It starts each exact argv with only the manifest environment and a new session.
No torch/NVML Python import, GPU allocation, shell expansion or remote command.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import time

HOST = "gpu-03"
UID = 3357
RUN_ROOT = Path("/scratch/yelarys.yertaiuly/codex_runs")
LOCK_ROOT = RUN_ROOT / "triplet_gpu_locks"
SCHEMA = "gpu03_task_specific_resource_guard_v1"
GIB = 1024 ** 3
INTERVAL = 5
IDLE_SECONDS = 60
STAGGER_SECONDS = 10
MAX_RSS = 160 * GIB
MIN_AVAILABLE = 192 * GIB
MIN_DISK = 128 * GIB
MAX_LOG = 64 * 1024 ** 2
MAX_CGROUP_DIRS = 64
MAX_CGROUP_PIDS = 4096
ENV_KEYS = {
    "PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONHASHSEED",
    "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE", "TMPDIR", "TORCHINDUCTOR_COMPILE_THREADS",
    "CUBLAS_WORKSPACE_CONFIG", "VLLM_WORKER_MULTIPROC_METHOD",
}


class GuardError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise GuardError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal(path, value):
    with Path(path).open("a", encoding="utf8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_manifest(m):
    require(m.get("schema") == SCHEMA and m.get("host") == HOST and
            type(m.get("uid")) is int and m["uid"] == UID, "Wrong schema/host/uid")
    token = m.get("run_token", "")
    require(isinstance(token, str) and re.fullmatch(r"codex-[a-z0-9-]{8,100}", token), "Invalid run token")
    require(isinstance(m.get("unit"), str) and re.fullmatch(re.escape(token) + r"-[a-z0-9-]+", m["unit"]),
            "Exact token-scoped systemd unit required")
    for key in ("runtime", "disk_path"):
        value = Path(m[key])
        require(value.is_absolute() and ".." not in value.parts, "Noncanonical path: " + key)
    runtime = Path(m["runtime"])
    require(runtime.parent == RUN_ROOT and runtime.name.startswith(token + "-"), "Runtime is not token scoped")
    require(not runtime.is_symlink(), "Runtime symlink forbidden")
    deadline = m.get("deadline_epoch")
    require(type(deadline) in (int, float) and math.isfinite(deadline), "Missing finite absolute deadline")
    devices = m.get("gpu_uuids")
    require(isinstance(devices, dict) and len(devices) <= 8, "Invalid GPU mapping")
    require(all(re.fullmatch(r"[0-7]", k) and isinstance(v, str) and
                re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", v) for k, v in devices.items()), "Invalid GPU identity")
    require(len(set(devices.values())) == len(devices), "Duplicate GPU UUID")
    workers = m.get("workers")
    require(isinstance(workers, list) and 1 <= len(workers) <= 8, "Expected 1..8 workers")
    names, assigned, cpu_used = set(), [], set()
    for w in workers:
        name = w.get("name", "")
        require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) and name not in names,
                "Invalid/duplicate worker name")
        names.add(name)
        gpu = w.get("gpu_id")
        require((type(gpu) is int and str(gpu) in devices) if devices else gpu is None,
                "GPU worker map differs; CPU-only workers must use null gpu_id")
        if devices:
            assigned.append(str(gpu))
        argv = w.get("argv")
        require(isinstance(argv, list) and argv and all(isinstance(x, str) and x and "\0" not in x for x in argv)
                and Path(argv[0]).is_absolute(), "Worker needs exact absolute executable argv")
        cwd = Path(w["cwd"])
        require(cwd.is_absolute() and ".." not in cwd.parts, "Invalid worker cwd")
        cpus = w.get("cpu_ids")
        require(isinstance(cpus, list) and cpus and all(type(x) is int and 0 <= x < 4096 for x in cpus)
                and len(set(cpus)) == len(cpus) and not cpu_used.intersection(cpus), "Invalid/overlapping CPU masks")
        cpu_used.update(cpus)
        env = w.get("env")
        require(isinstance(env, dict) and set(env) <= ENV_KEYS and
                all(isinstance(v, str) and "\0" not in v for v in env.values()), "Unreviewed worker environment")
        require(env.get("CUDA_VISIBLE_DEVICES") == (devices[str(gpu)] if devices else ""), "CUDA mask must bind exact UUID or be empty")
        require(env.get("OMP_NUM_THREADS") == env.get("MKL_NUM_THREADS") ==
                env.get("OPENBLAS_NUM_THREADS") == "1", "BLAS/OpenMP threads must be one")
        require(env.get("TOKENIZERS_PARALLELISM") == "false" and env.get("PYTHONNOUSERSITE") == "1",
                "Tokenizer/user-site isolation missing")
    require(not devices or sorted(assigned) == sorted(devices), "Exactly one worker per GPU required")
    return m


def load_manifest(path, digest):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "Manifest must be an absolute regular file")
    require(path.stat().st_size <= 1024 * 1024, "Manifest too large")
    data = path.read_bytes()
    require(re.fullmatch(r"[0-9a-f]{64}", digest or "") and hashlib.sha256(data).hexdigest() == digest,
            "Manifest SHA256 mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return validate_manifest(json.loads(data, object_pairs_hook=pairs))


def proc_row(pid, root=Path("/proc")):
    """Metadata only: no command line or environment/credential reads."""
    path = root / str(pid)
    uid = path.stat().st_uid
    data = (path / "stat").read_text()
    fields = data[data.rfind(")") + 2:].split()
    return {"pid": int(pid), "uid": uid, "state": fields[0], "ppid": int(fields[1]),
            "pgid": int(fields[2]), "sid": int(fields[3]), "start": int(fields[19]),
            "rss": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE")}


def process_table():
    rows = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            rows[int(path.name)] = proc_row(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            # Expected process-exit race; never substitute another identity.
            continue
        except PermissionError:
            try:
                owner = path.stat().st_uid
            except (FileNotFoundError, ProcessLookupError):
                continue
            if owner == UID:
                raise GuardError("Cannot inspect an owner process")
    return rows


def gpu_snapshot():
    def query(fields, kind):
        result = subprocess.run(["/usr/bin/nvidia-smi", "--query-" + kind + "=" + fields,
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
        require(result.returncode == 0, "nvidia-smi query failed: " + result.stderr[:1000])
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    devices = {}
    for row in query("index,uuid,memory.total,memory.free,utilization.gpu", "gpu"):
        require(len(row) == 5, "Malformed GPU inventory")
        index, uuid, total, free, utilization = row
        require(index not in devices, "Duplicate GPU inventory row")
        devices[index] = {"uuid": uuid, "total_mib": int(total), "free_mib": int(free),
                          "utilization": int(utilization), "pids": []}
    by_uuid = {v["uuid"]: v for v in devices.values()}
    for row in query("gpu_uuid,pid", "compute-apps"):
        require(len(row) == 2 and row[0] in by_uuid, "Malformed compute process inventory")
        by_uuid[row[0]]["pids"].append(int(row[1]))
    return devices


@dataclass
class OwnedWorker:
    name: str
    process: object
    start: int
    known: dict = field(default_factory=dict)


def owned_rows(workers, table):
    """Stable pid/start identities, private session membership, then ancestry.

    A same-UID foreign process is not ours. Vanishing /proc rows are normal;
    PID reuse is never adopted. New process groups in our session remain ours.
    """
    owned = {}
    for worker in workers:
        pid = worker.process.pid
        root = table.get(pid)
        if root is not None:
            require(root["start"] == worker.start and root["uid"] == UID, "Worker PID identity changed")
        for p, row in table.items():
            if row["uid"] != UID:
                continue
            if ((p == pid and row["start"] == worker.start) or
                    worker.known.get(p) == row["start"] or row["sid"] == pid):
                owned[p] = row
        changed = True
        while changed:
            changed = False
            for p, row in table.items():
                if p not in owned and row["uid"] == UID and row["ppid"] in owned:
                    owned[p] = row
                    changed = True
        worker.known.update({p: row["start"] for p, row in owned.items()})
    return owned


def check_devices(mapping, snapshot, owned, *, idle=False):
    for index, uuid in mapping.items():
        require(index in snapshot and snapshot[index]["uuid"] == uuid, "GPU UUID changed or disappeared: " + index)
        device = snapshot[index]
        foreign = [pid for pid in device["pids"] if pid not in owned or owned[pid]["uid"] != UID]
        require(not foreign, "Foreign or unverified GPU process on " + index + ": " + str(foreign))
        if idle:
            require(not device["pids"] and device["utilization"] <= 2 and
                    device["total_mib"] - device["free_mib"] <= 1024,
                    "Selected GPU is not idle: " + index)


class CgroupScope:
    """Exact admitted service membership also catches orphaned/double-forked workers.

    The independent phase entry checks kernel resource ceilings. This supplements
    ancestry: only this service's members, all UID3357, are admitted, and no other
    process may be present before worker dispatch. No cgroups are created/moved.
    """
    def __init__(self, m, *, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"), self_pid=None):
        self.proc_root, self.cgroup_root = proc_root, cgroup_root
        self.self_pid = os.getpid() if self_pid is None else self_pid
        self.relative = self.location()
        require(self.relative.endswith("/" + m["unit"] + ".service"), "Not in the exact token service cgroup")
        self.path = cgroup_root / self.relative.lstrip("/")
        require(not self.path.is_symlink(), "Service cgroup is a symlink")
        self.identity = (self.path.stat().st_dev, self.path.stat().st_ino)
        self.known = {}
        require(self.members() == {self.self_pid}, "Service cgroup already contains other processes")

    def location(self):
        data = (self.proc_root / str(self.self_pid) / "cgroup").read_text()
        rows = [line[3:] for line in data.splitlines() if line.startswith("0::")]
        require(len(rows) == 1 and rows[0].startswith("/") and ".." not in Path(rows[0]).parts,
                "Missing canonical unified cgroup identity")
        return rows[0]

    def members(self):
        require(self.location() == self.relative, "Supervisor moved out of the bound cgroup")
        require((self.path.stat().st_dev, self.path.stat().st_ino) == self.identity,
                "Service cgroup identity changed")
        members, pending, visited = set(), [self.path], 0
        while pending:
            directory = pending.pop()
            visited += 1
            require(visited <= MAX_CGROUP_DIRS, "Too many descendant cgroups")
            require(not directory.is_symlink(), "Descendant cgroup symlink forbidden")
            try:
                data = (directory / "cgroup.procs").read_text()
                members.update(int(x) for x in data.split())
                require(all(x > 0 for x in members) and len(members) <= MAX_CGROUP_PIDS,
                        "Invalid/excessive cgroup process count")
                for child in directory.iterdir():
                    require(not child.is_symlink(), "Cgroup symlink forbidden")
                    if child.is_dir():
                        pending.append(child)
            except FileNotFoundError:
                # Empty descendant cgroups may disappear during normal exit.
                require(directory != self.path, "Bound service cgroup disappeared")
        require(self.self_pid in members, "Supervisor missing from bound cgroup")
        return members

    def rows(self, table):
        members = self.members() - {self.self_pid}
        for pid in members:
            row = table.get(pid)
            if row is not None:
                require(row["uid"] == UID, "Foreign UID in exact service cgroup")
                self.known[pid] = row["start"]
        # Keep stable identities after a child changes session/cgroup. A process
        # absent from this table may have exited; fresh membership is read again
        # next observation, while the independent cgroup ceiling always applies.
        return {pid: row for pid, row in table.items()
                if pid != self.self_pid and row["uid"] == UID and self.known.get(pid) == row["start"]}


class ResourceProbe:
    def __init__(self, m, *, gpu_rejection_callback=None):
        self.scope = CgroupScope(m)
        self.gpu_rejection_callback = gpu_rejection_callback

    def scoped(self, table):
        return self.scope.rows(table)

    def processes(self):
        return process_table()

    def gpus(self):
        return gpu_snapshot()

    def available(self):
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(values["MemAvailable"].split()[0]) * 1024

    def disk(self, path):
        return shutil.disk_usage(path).free


def observed_owned(workers, probe, table):
    owned = owned_rows(workers, table)
    if hasattr(probe, "scoped"):
        owned.update(probe.scoped(table))
    return owned


def inspect_resources(m, workers, probe, *, idle=False):
    table = probe.processes()
    owned = observed_owned(workers, probe, table)
    rss = sum(row["rss"] for row in owned.values())
    available, free = probe.available(), probe.disk(m["disk_path"])
    require(rss <= MAX_RSS, "Owned aggregate RSS exceeds 160GiB")
    require(available >= MIN_AVAILABLE, "Host available RAM below 192GiB")
    require(free >= MIN_DISK, "Disk free space below 128GiB")
    logs = {}
    for w in m["workers"]:
        path = Path(m["runtime"]) / (w["name"] + ".log")
        if path.exists():
            require(not path.is_symlink(), "Worker log became a symlink")
            logs[w["name"]] = path.stat().st_size
            require(logs[w["name"]] <= MAX_LOG, "Worker log exceeded 64MiB: " + w["name"])
    devices = None
    if m["gpu_uuids"]:
        devices = probe.gpus()
        unknown = {p for index in m["gpu_uuids"] for p in devices.get(index, {}).get("pids", []) if p not in table}
        if unknown:
            # NVML can return a just-exited PID. Re-read both sources once; never
            # excuse a still-present unknown or foreign PID based only on text.
            devices = probe.gpus()
            table = probe.processes()
            owned = observed_owned(workers, probe, table)
            rss = sum(row["rss"] for row in owned.values())
            require(rss <= MAX_RSS, "Owned aggregate RSS exceeds 160GiB")
        try:
            check_devices(m["gpu_uuids"], devices, owned, idle=idle)
        except GuardError as original:
            callback = getattr(probe, "gpu_rejection_callback", None)
            if callback is not None:
                try:
                    callback(m["gpu_uuids"], devices, owned, table,
                             getattr(getattr(probe, "scope", None), "relative", None))
                except Exception as diagnostic:
                    original.diagnostic_error_type = type(diagnostic).__name__
            raise
    return {"at": time.time(), "owned_pids": sorted(owned), "owned_rss_bytes": rss,
            "mem_available_bytes": available, "disk_free_bytes": free, "log_bytes": logs,
            "gpus": devices, "cpu_only": not bool(m["gpu_uuids"])}


def qualify_idle(m, workers, probe, record, *, clock=time.monotonic, sleep=time.sleep):
    if not m["gpu_uuids"]:
        record(inspect_resources(m, workers, probe))
        return  # No GPU query, lock, or claimed qualification in CPU-only phases.
    start = clock()
    while True:
        require(time.time() < m["deadline_epoch"], "Deadline during idle qualification")
        record(dict(inspect_resources(m, workers, probe, idle=True), qualification_elapsed=clock() - start))
        if clock() - start >= IDLE_SECONDS:
            return
        sleep(min(INTERVAL, IDLE_SECONDS - (clock() - start)))


def acquire_gpu_locks(mapping):
    if not mapping:
        return []
    require(not LOCK_ROOT.is_symlink(), "Shared lock directory is a symlink")
    LOCK_ROOT.mkdir(mode=0o700, exist_ok=True)
    info = LOCK_ROOT.stat()
    require(info.st_uid == UID and not info.st_mode & 0o022, "Unsafe shared GPU lock directory")
    fds = []
    try:
        for index in sorted(mapping):
            fd = os.open(LOCK_ROOT / ("gpu_" + index + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            fds.append(fd)
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == UID and info.st_nlink == 1,
                    "Unsafe shared GPU lock file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def launch_worker(w, log):
    cpus = ",".join(str(x) for x in w["cpu_ids"])
    proc = subprocess.Popen(["/usr/bin/taskset", "-c", cpus, *w["argv"]], cwd=w["cwd"], env=w["env"],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True)
    row = proc_row(proc.pid)
    require(row["uid"] == UID and row["pgid"] == row["sid"] == proc.pid, "Spawn lacks owned private process group")
    return OwnedWorker(w["name"], proc, row["start"], {proc.pid: row["start"]})


def signal_owned(workers, table, sig, *, sender=os.killpg, owned=None):
    owned = owned_rows(workers, table) if owned is None else owned
    groups = {r["pgid"] for r in owned.values() if r["state"] != "Z"}
    for group in sorted(groups):
        members = [r for r in table.values() if r["pgid"] == group and r["state"] != "Z"]
        require(group > 1 and members and all(r["pid"] in owned for r in members),
                "Refusing to signal an unverified process group")
        # The caller supplies a fresh cleanup observation, independent from the
        # earlier monitor sample. killpg itself does not take another /proc read.
        try:
            sender(group, sig)
        except ProcessLookupError:
            pass
    return owned


def terminate_owned(workers, probe, *, clock=time.monotonic, sleep=time.sleep):
    for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        table = probe.processes()
        signal_owned(workers, table, sig, owned=observed_owned(workers, probe, table))
        end = clock() + seconds
        while True:
            for w in workers:
                w.process.poll()
            owned = observed_owned(workers, probe, probe.processes())
            if not any(r["state"] != "Z" for r in owned.values()):
                for w in workers:
                    w.process.wait(timeout=2)
                remaining = observed_owned(workers, probe, probe.processes())
                require(not any(r["state"] != "Z" for r in remaining.values()), "Live descendant appeared during cleanup")
                return {"owned_processes_released": not bool(remaining), "owned_live_processes_released": True,
                        "zombie_pids": sorted(remaining)}
            if clock() >= end:
                break
            sleep(min(.25, end - clock()))
    raise GuardError("Owned workers or descendants remain after scoped cleanup")


def verify_release(m, workers, probe, record, *, clock=time.monotonic, sleep=time.sleep):
    end = clock() + 60
    while True:
        owned = observed_owned(workers, probe, probe.processes())
        if owned:
            record({"at": time.time(), "release_pending_owned_pids": sorted(owned),
                    "zombie_pids": sorted(p for p, r in owned.items() if r["state"] == "Z")})
            require(clock() < end, "Owned process entries remain after worker exit")
            sleep(min(INTERVAL, end - clock()))
            continue
        if not m["gpu_uuids"]:
            return {"owned_processes_released": True, "gpu_release_verified": None, "cpu_only": True}
        devices = probe.gpus()
        try:
            check_devices(m["gpu_uuids"], devices, {}, idle=True)
            return {"owned_processes_released": True, "gpu_release_verified": True, "gpus": devices}
        except GuardError as exc:
            record({"at": time.time(), "release_pending": str(exc), "gpus": devices})
            if clock() >= end:
                raise
            sleep(min(INTERVAL, end - clock()))


def supervise(path, digest, *, probe=None, gpu_rejection_callback=None):
    m = load_manifest(path, digest)
    require(socket.gethostname().split(".")[0] == HOST and os.getuid() == UID, "Actual host/UID differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Supervisor CUDA mask must be empty")
    require(60 < m["deadline_epoch"] - time.time() <= 86400, "Absolute work deadline expired or exceeds24h")
    allowed = os.sched_getaffinity(0)
    require(all(set(w["cpu_ids"]) <= allowed for w in m["workers"]), "Worker CPUs unavailable")
    runtime = Path(m["runtime"])
    require(not runtime.exists(), "Existing attempt preserved; use a separately reviewed recovery manifest")
    runtime.mkdir(mode=0o700)
    write_json(runtime / "IDENTITY.json", {"run_token": m["run_token"], "manifest_sha256": digest,
                                          "schema": SCHEMA, "host": HOST, "uid": UID})
    record = lambda value: journal(runtime / "resources.jsonl", value)
    if probe is None:
        probe = ResourceProbe(m, gpu_rejection_callback=gpu_rejection_callback)
    elif gpu_rejection_callback is not None:
        probe.gpu_rejection_callback = gpu_rejection_callback
    workers, logs, locks = [], [], []
    result = {"run_token": m["run_token"], "manifest_sha256": digest, "status": "failed",
              "scientific_success_verified": False}
    def interrupted(signum, frame):
        raise GuardError("Supervisor received signal " + str(signum))
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        locks = acquire_gpu_locks(m["gpu_uuids"])
        qualify_idle(m, workers, probe, record)
        for index, w in enumerate(m["workers"]):
            if index:
                end = time.monotonic() + STAGGER_SECONDS
                while time.monotonic() < end:
                    require(time.time() < m["deadline_epoch"], "Deadline during stagger")
                    record(inspect_resources(m, workers, probe))
                    require(all(x.process.poll() in (None, 0) for x in workers), "Worker failed during stagger")
                    time.sleep(min(INTERVAL, end - time.monotonic()))
            require(time.time() < m["deadline_epoch"], "Deadline before worker launch")
            observation = inspect_resources(m, workers, probe, idle=not workers and bool(m["gpu_uuids"]))
            record(observation)
            if w["gpu_id"] is not None:
                index = str(w["gpu_id"])
                check_devices({index: m["gpu_uuids"][index]}, observation["gpus"], {}, idle=True)
            stream = (runtime / (w["name"] + ".log")).open("xb")
            logs.append(stream)
            owned = launch_worker(w, stream)
            workers.append(owned)
            journal(runtime / "workers.jsonl", {"name": w["name"], "pid": owned.process.pid,
                                               "start": owned.start, "gpu_id": w["gpu_id"], "at": time.time()})
        while True:
            require(time.time() < m["deadline_epoch"], "Absolute work deadline reached")
            record(inspect_resources(m, workers, probe))
            codes = {w.name: w.process.poll() for w in workers}
            require(all(code in (None, 0) for code in codes.values()), "Nonzero worker exit: " + str(codes))
            if all(code == 0 for code in codes.values()):
                break
            time.sleep(min(INTERVAL, max(0, m["deadline_epoch"] - time.time())))
        result.update(verify_release(m, workers, probe, record), status="workers_exited_zero", exit_codes=codes)
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        if hasattr(exc, "diagnostic_error_type"):
            result["diagnostic_error_type"] = exc.diagnostic_error_type
        # A second TERM must not interrupt cleanup halfway; outer systemd remains
        # the independent hard deadline and can still SIGKILL the exact cgroup.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            result["cleanup"] = terminate_owned(workers, probe)
        except BaseException as cleanup:
            result["cleanup_error"] = type(cleanup).__name__ + ": " + str(cleanup)
        try:
            result["release"] = verify_release(m, workers, probe, record)
        except BaseException as release:
            result["release_error"] = type(release).__name__ + ": " + str(release)
    finally:
        result["finished_epoch"] = time.time()
        for stream in logs:
            stream.close()
        for fd in locks:
            os.close(fd)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        write_json(runtime / "SUPERVISOR_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    result = supervise(args.manifest, args.sha256)
    print(canonical(result), flush=True)
    return 0 if result["status"] == "workers_exited_zero" else 1


if __name__ == "__main__":
    raise SystemExit(main())
