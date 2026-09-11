#!/usr/bin/env python3
"""Consume an exact one-use approval and start the reviewed systemd service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import time
from pathlib import Path


APPROVAL_PREFIX = "I_APPROVE_CHECKPOINT60_FACTORIAL"
EXPECTED_PURPOSE = "checkpoint-60 fully crossed evaluator-modification rollout collection"
APPROVAL_INCLUDES_HOST = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify(manifest_path: Path, approval: str) -> tuple[dict, str]:
    manifest_path = manifest_path.resolve()
    digest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest.get("execution", {})
    expected = f"{APPROVAL_PREFIX}:{digest}:{execution.get('run_token')}"
    if APPROVAL_INCLUDES_HOST:
        expected += f":{execution.get('host')}"
    if approval != expected:
        raise PermissionError("approval does not exactly match this reviewed manifest")
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported review manifest schema")
    if manifest.get("purpose") != EXPECTED_PURPOSE:
        raise ValueError("reviewed manifest purpose differs from launcher mode")
    if execution.get("host") != socket.gethostname():
        raise RuntimeError("reviewed host differs from current host")
    hosts = manifest.get("hosts", {})
    host_profile = execution.get("host_profile", "gpu02_gpu04")
    expected_hosts = {
        "gpu02_gpu04": {"gpu-02", "gpu-04"},
        "gpu04_only": {"gpu-04"},
        "gpu04_gpus1_7": {"gpu-04"},
    }.get(host_profile)
    if expected_hosts is None or set(hosts) != expected_hosts:
        raise ValueError("reviewed hosts differ from the bound host profile")
    local_files = hosts[execution["host"]].get("critical_file_sha256", {})
    for raw_path, expected_hash in local_files.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"manifest-critical file changed: {path}")
    output_dir = Path(execution["output_dir"]).resolve()
    artifact_path = output_dir / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_members = artifact.get("files")
    if not isinstance(expected_members, dict) or not expected_members:
        raise ValueError("prepared package artifact manifest is empty")
    observed_members = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == artifact_path:
            continue
        if path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(output_dir).as_posix()
        observed_members[relative] = {
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        }
    if observed_members != expected_members:
        raise ValueError("prepared package files differ from its reviewed artifact manifest")
    for key in ("launch_receipt", "supervisor_status", "launch_log"):
        path = Path(execution[key])
        if path.exists():
            raise FileExistsError(f"one-use path already exists: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
    summary = json.loads((output_dir / "summary.json").read_text())
    if summary.get("status") != "prepared":
        raise RuntimeError("output package is not in reviewed prepared state")
    for host, host_entry in sorted(hosts.items()):
        if host == execution["host"]:
            continue
        spec = host_entry["spec"]
        remote_manifest = spec["reviewed_manifest"]
        result = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "--",
                spec["python"], spec["remote_verifier"], "--manifest", remote_manifest,
                "--expected-sha256", digest, "--host", host,
            ],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            raise ValueError(f"remote reviewed inputs do not verify on {host}: {result.stderr[-500:]}")
    return manifest, digest


def systemd_command(manifest: dict, launch_id: str) -> list[str]:
    execution = manifest["execution"]
    reviewed_systemd = execution.get("systemd", {
        "runtime_max_seconds": 14400, "timeout_stop_seconds": 120,
        "restart": "on-failure",
    })
    stop = " ".join([
        execution["python"], execution["receipt_writer"],
        "--status", execution["supervisor_status"],
        "--run-token", execution["run_token"],
        "--output-dir", execution["output_dir"],
    ])
    if any(any(char.isspace() for char in item) for item in stop.split(" ")):
        raise ValueError("reviewed paths may not contain whitespace")
    return [
        "systemd-run", "--user", f"--unit={execution['service_unit']}",
        "--property=Type=exec", "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={int(reviewed_systemd['runtime_max_seconds'])}s",
        f"--property=TimeoutStopSec={int(reviewed_systemd['timeout_stop_seconds'])}s",
        f"--property=Restart={reviewed_systemd['restart']}", "--property=RestartSec=15s",
        "--property=RestartPreventExitStatus=78",
        f"--property=StandardOutput=append:{execution['launch_log']}",
        f"--property=StandardError=append:{execution['launch_log']}",
        f"--property=ExecStopPost={stop}",
        f"--setenv=CODEX_FACTORIAL_LAUNCH_ID={launch_id}",
        *execution["command"],
    ]


def launch(manifest: dict, digest: str, approval: str) -> None:
    execution = manifest["execution"]
    launch_id = secrets.token_hex(16)
    write_exclusive(Path(execution["launch_receipt"]), {
        "schema_version": 1, "run_token": execution["run_token"],
        "manifest_sha256": digest, "approval_sha256": hashlib.sha256(approval.encode()).hexdigest(),
        "launch_id": launch_id, "state": "approval_consumed_before_systemd_request",
    })
    log_descriptor = os.open(
        execution["launch_log"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    os.close(log_descriptor)
    result = subprocess.run(systemd_command(manifest, launch_id), capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError("systemd rejected reviewed job; approval remains consumed: " + result.stderr[:500])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["systemctl", "--user", "show", f"{execution['service_unit']}.service", "-p", "ActiveState", "--value"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        if state in {"active", "activating"}:
            print(json.dumps({"service_unit": execution["service_unit"], "launch_log": execution["launch_log"]}))
            return
        time.sleep(1)
    raise RuntimeError("reviewed systemd service did not become active; approval remains consumed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    manifest, digest = verify(args.manifest, args.approval)
    launch(manifest, digest, args.approval)


if __name__ == "__main__":
    main()
