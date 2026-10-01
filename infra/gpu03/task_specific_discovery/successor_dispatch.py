"""One-use dispatcher for the explicitly authorized 20260922 fresh-seed pilot.

CLI: --root EXACT_ROOT --manifest UNIQUE_LOCAL.json --sha256 REVIEWED_SHA
No retries: uncertain SSH/systemd results preserve consumed attempt markers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time

from infra.gpu03.task_specific_discovery import resource_guard as guard

TOKEN = "codex-task-specific-pilot-20260922-113907"
ROOT = Path("/scratch/yelarys.yertaiuly/codex_runs") / TOKEN
PYTHON = "/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python"
HOST, UID = "gpu-03", 3357
GPU_UUIDS = {
    "0": "GPU-a3ca20c5-9e29-094f-fa10-08dcb68bf5c8",
    "1": "GPU-be7650b8-c9ec-ab97-7e7f-dd67dcb0e118",
    "2": "GPU-906ac90a-a2fa-50ae-fc73-2fb9c695f181",
    "3": "GPU-4ce804e5-7292-8b96-aff8-24980bbdb865",
    "4": "GPU-c1d2da43-2804-6ed3-ec2f-419c031d92fa",
    "5": "GPU-a951f33d-8bd8-5291-3e73-d70b2caf9555",
    "6": "GPU-23b03f6b-ce9e-278a-2653-16c2eedbbb67",
    "7": "GPU-9517b65b-0515-21b3-681e-1e7fd9da4ad2",
}
MAX_MANIFEST = 1024 ** 2
MODULE_RELATIVE = "infra/gpu03/task_specific_discovery/successor_dispatch.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse(data):
    require(len(data) <= MAX_MANIFEST, "Manifest exceeds 1MiB")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate manifest key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def canonical_path(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts and str(path) == str(value), "Noncanonical path")
    return path


def validate_successor_manifest(m, root, *, now=None):
    require(canonical_path(root) == ROOT, "Wrong successor root")
    guard.validate_manifest(m)
    require(m["run_token"] == TOKEN and m["host"] == HOST and m["uid"] == UID, "Wrong successor identity")
    require(canonical_path(m["disk_path"]) == ROOT, "Wrong successor disk root")
    runtime = canonical_path(m["runtime"])
    require(runtime == ROOT.parent / (m["unit"] + "-runtime"), "Wrong exact runtime path")
    stage = canonical_path(m["source_root"])
    require(stage.parent == ROOT and re.fullmatch(r"source-[a-z0-9-]+", stage.name), "Wrong successor source root")
    entry = canonical_path(m["entry_receipt"])
    require(entry.parent == ROOT and re.fullmatch(r"[a-z0-9-]+-entry\.json", entry.name), "Wrong successor entry receipt")
    limits = m["kernel_limits"]
    require(set(limits) == {"memory_max", "tasks_max", "cpu_cores"} and
            all(type(v) is int and v > 0 for v in limits.values()), "Malformed kernel limits")
    require(limits["tasks_max"] <= 512 and limits["cpu_cores"] <= 16, "Excess task/CPU limits")
    if m["gpu_uuids"]:
        require(m["gpu_uuids"] == GPU_UUIDS, "Changed successor GPU allocation")
        require(limits == {"memory_max": 192 * 1024 ** 3, "tasks_max": 512, "cpu_cores": 16},
                "Changed GPU wave limits")
    else:
        require(limits["memory_max"] <= 32 * 1024 ** 3, "Excess CPU-phase memory limit")
    now = time.time() if now is None else now
    remaining = m["deadline_epoch"] - now
    require(60 < remaining <= 14400, "Expired or excessive work deadline")
    runtime_seconds = m["runtime_seconds"]
    require(type(runtime_seconds) is int and math.ceil(remaining) + 60 <= runtime_seconds <= 14520,
            "Insufficient cleanup time or excessive service runtime")
    bindings = m.get("bound_files")
    require(isinstance(bindings, dict) and bindings, "Missing immutable source/input bindings")
    for rel in (MODULE_RELATIVE, "infra/gpu03/task_specific_discovery/successor_phase_entry.py",
                "infra/gpu03/task_specific_discovery/phase_entry.py",
                "infra/gpu03/task_specific_discovery/resource_guard.py",
                "infra/gpu03/task_specific_discovery/guard_failure_diagnostic.py"):
        binding = bindings.get(str(stage / rel), {})
        require(type(binding.get("size_bytes")) is int and binding["size_bytes"] > 0 and
                re.fullmatch(r"[0-9a-f]{64}", binding.get("sha256", "")), "Missing successor module binding: " + rel)
    return m


def build_command(m, remote, digest):
    limits, stage = m["kernel_limits"], m["source_root"]
    return ["systemd-run", "--user", "--unit=" + m["unit"], "--service-type=exec",
            "--property=RuntimeMaxSec=" + str(m["runtime_seconds"]),
            "--property=MemoryMax=" + str(limits["memory_max"]), "--property=MemorySwapMax=0",
            "--property=TasksMax=" + str(limits["tasks_max"]),
            "--property=CPUQuota=" + str(limits["cpu_cores"] * 100) + "%",
            "--property=KillMode=control-group", "--property=TimeoutStopSec=60",
            "--property=NoNewPrivileges=yes", "--property=UMask=0077",
            "--property=MemoryAccounting=yes", "--property=TasksAccounting=yes", "--property=CPUAccounting=yes",
            "--property=StandardOutput=append:" + str(ROOT / (m["unit"] + ".log")),
            "--property=StandardError=inherit", "--working-directory=" + stage,
            "/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "LANG=C.UTF-8", "CUDA_VISIBLE_DEVICES=",
            "PYTHONPATH=" + stage, "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "TOKENIZERS_PARALLELISM=false",
            PYTHON, "-B", "-m", "infra.gpu03.task_specific_discovery.successor_phase_entry",
            "--manifest", str(remote), "--sha256", digest]


def exclusive_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare(local, root, digest, *, now=None):
    local = Path(local)
    info = local.lstat()
    require(stat.S_ISREG(info.st_mode) and not local.is_symlink() and info.st_size <= MAX_MANIFEST, "Unsafe local manifest")
    data = local.read_bytes()
    require(re.fullmatch(r"[0-9a-f]{64}", digest or "") and hashlib.sha256(data).hexdigest() == digest,
            "Reviewed manifest SHA mismatch")
    require(re.fullmatch(r"[A-Z0-9][A-Z0-9_]{2,100}\.json", local.name) and local.name != "RESOURCE_MANIFEST.json",
            "Use a unique descriptive manifest basename")
    m = validate_successor_manifest(parse(data), root, now=now)
    remote = ROOT / "plans" / local.name
    return {"root": str(ROOT), "remote": str(remote), "data": data.decode(), "sha256": digest,
            "command": build_command(m, remote, digest), "unit": m["unit"], "source_root": m["source_root"]}


def _directory(path):
    path = Path(path)
    require(path.resolve(strict=True) == path, "Symlink/noncanonical remote directory")
    info = path.stat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022, "Unsafe remote directory")


def remote_execute(payload, *, runner=subprocess.run, now=None):
    require(socket.gethostname().split(".")[0] == HOST and os.getuid() == UID, "Wrong actual host/UID")
    data = payload["data"].encode()
    require(hashlib.sha256(data).hexdigest() == payload["sha256"], "Transport manifest hash changed")
    m = validate_successor_manifest(parse(data), payload["root"], now=now)
    remote = canonical_path(payload["remote"])
    require(remote.parent == ROOT / "plans" and re.fullmatch(r"[A-Z0-9][A-Z0-9_]{2,100}\.json", remote.name)
            and remote.name != "RESOURCE_MANIFEST.json", "Wrong remote manifest path")
    command = build_command(m, remote, payload["sha256"])
    require(payload["command"] == command and payload["unit"] == m["unit"] and payload["source_root"] == m["source_root"],
            "Dispatch command/payload changed")
    _directory(ROOT); _directory(ROOT / "plans"); _directory(m["source_root"])
    attempt = ROOT / (m["unit"] + "_DISPATCH_ATTEMPT.json")
    receipt = ROOT / (m["unit"] + "_DISPATCH_RESULT.json")
    for path in (remote, attempt, receipt, Path(m["runtime"]), Path(m["entry_receipt"])):
        require(not path.exists() and not path.is_symlink(), "Existing one-use path preserved: " + str(path))
    state = runner(["systemctl", "--user", "show", m["unit"] + ".service", "-p", "LoadState", "-p", "MainPID"],
                   capture_output=True, text=True, timeout=20)
    state_fields = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
    require(state.returncode in (0, 4) and state_fields.get("LoadState") == "not-found" and
            state_fields.get("MainPID") == "0", "Exact unit is not positively absent")
    with remote.open("xb") as stream:
        stream.write(data); stream.flush(); os.fchmod(stream.fileno(), 0o444); os.fsync(stream.fileno())
    parent_fd = os.open(remote.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    exclusive_json(attempt, {"at": time.time(), "manifest_sha256": payload["sha256"], "command": command,
                             "unit_absence": state_fields, "retry_allowed": False})
    try:
        result = runner(command, capture_output=True, text=True, timeout=30)
        out = {"status": "dispatch_command_returned", "returncode": result.returncode,
               "stdout": result.stdout[-20000:], "stderr": result.stderr[-20000:], "retry_allowed": False}
    except Exception as exc:
        out = {"status": "ambiguous_dispatch", "exception_type": type(exc).__name__, "retry_allowed": False}
    try:
        observed = runner(["systemctl", "--user", "show", m["unit"] + ".service", "-p", "LoadState", "-p", "ActiveState",
                           "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus",
                           "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "CPUQuotaPerSecUSec", "-p", "TasksMax",
                           "-p", "RuntimeMaxUSec", "-p", "ControlGroup"], capture_output=True, text=True, timeout=20)
        out.update(state_returncode=observed.returncode, state=observed.stdout[-20000:], state_error=observed.stderr[-2000:])
    except Exception as exc:
        out["state_exception_type"] = type(exc).__name__
    exclusive_json(receipt, out)
    return out


BOOTSTRAP = """import hashlib,json,pathlib,sys
p=json.load(sys.stdin)
r=pathlib.Path(p['root']); assert str(r)=='/scratch/yelarys.yertaiuly/codex_runs/codex-task-specific-pilot-20260922-113907'
s=pathlib.Path(p['source_root']); assert s.parent==r and s.resolve(strict=True)==s
m=json.loads(p['data']); assert hashlib.sha256(p['data'].encode()).hexdigest()==p['sha256']
for n in ('successor_dispatch.py','successor_phase_entry.py','phase_entry.py','resource_guard.py','guard_failure_diagnostic.py'):
 f=s/'infra/gpu03/task_specific_discovery'/n; b=m['bound_files'][str(f)]
 assert not f.is_symlink() and f.stat().st_size==b['size_bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==b['sha256']
sys.path.insert(0,str(s))
from infra.gpu03.task_specific_discovery.successor_dispatch import remote_execute
o=remote_execute(p); print(json.dumps(o)); sys.exit(0 if o.get('returncode')==0 else 1)
"""


def dispatch(local, root, digest, *, runner=subprocess.run, now=None):
    payload = prepare(local, root, digest, now=now)
    local = Path(local)
    result_path = local.with_suffix(".local_result.json")
    require(not result_path.exists() and not result_path.is_symlink(), "Existing local result preserved")
    exclusive_json(local.with_suffix(".local_attempt.json"), {
        "at": time.time(), "manifest_sha256": digest, "command": payload["command"],
        "root": str(ROOT), "remote": payload["remote"], "retry_allowed": False})
    remote_argv = ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "LANG=C.UTF-8", "PYTHONNOUSERSITE=1",
                   "PYTHONDONTWRITEBYTECODE=1", PYTHON, "-B", "-c", BOOTSTRAP]
    try:
        result = runner(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", HOST, shlex.join(remote_argv)],
                        input=json.dumps(payload), capture_output=True, text=True, timeout=100)
        out = {"status": "transport_returned", "returncode": result.returncode,
               "stdout": result.stdout[-50000:], "stderr": result.stderr[-10000:], "retry_allowed": False}
    except Exception as exc:
        out = {"status": "ambiguous_transport", "exception_type": type(exc).__name__, "retry_allowed": False}
    exclusive_json(result_path, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True); p.add_argument("--manifest", required=True); p.add_argument("--sha256", required=True)
    args = p.parse_args()
    out = dispatch(args.manifest, args.root, args.sha256)
    print(json.dumps(out, sort_keys=True))
    return 0 if out.get("returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
