#!/usr/bin/env python3
"""Consume one reviewed approval and launch its exact supervised gpu-03 job."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import socket
import stat
import subprocess
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_json(path: Path, value: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Could not complete atomic launch receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_manifest(manifest_path: Path, approval: str) -> tuple[dict, str]:
    manifest_path = manifest_path.resolve()
    digest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError("Unsupported direct-host review manifest schema")
    execution = manifest.get("execution", {})
    expected = (
        f"I_APPROVE_CHECKPOINT60_ACTIVATIONS:{digest}:"
        f"{execution.get('run_token')}:{execution.get('host')}"
    )
    if approval != expected:
        raise PermissionError("Approval string does not exactly match the reviewed manifest")
    if execution.get("host") != "gpu-03" or socket.gethostname() != "gpu-03":
        raise RuntimeError("The reviewed direct launch is bound to gpu-03")
    if Path(execution.get("review_manifest", "")).resolve() != manifest_path:
        raise ValueError("Manifest execution path does not point back to the reviewed manifest")
    reviewed_hashes = manifest.get("critical_file_sha256")
    if not isinstance(reviewed_hashes, dict) or not reviewed_hashes:
        raise ValueError("Manifest has no critical file hashes")
    for raw_path, expected_hash in reviewed_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Manifest-critical file changed or disappeared: {path}")
    if Path(execution["output_dir"]).exists():
        raise FileExistsError("Reviewed output directory already exists")
    if Path(execution["scratch_root"]).exists():
        raise FileExistsError("Reviewed scratch directory already exists")
    for key in ("launch_receipt", "launch_result", "launch_log", "supervisor_status"):
        path = Path(execution[key])
        if not path.parent.is_dir() or path.exists():
            raise FileExistsError(f"One-use {key} path is absent or already consumed")
    entrypoint = Path(execution["direct_entrypoint"])
    if not entrypoint.is_file() or not os.access(entrypoint, os.X_OK):
        raise FileNotFoundError("Reviewed direct entry point is absent or not executable")
    if execution.get("service_unit") != execution.get("run_token"):
        raise ValueError("Reviewed systemd service unit must equal the exact run token")
    service = subprocess.run(
        ["systemctl", "--user", "show", f"{execution['service_unit']}.service", "-p", "LoadState", "--value"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if service.returncode == 0 and service.stdout.strip() not in {"", "not-found"}:
        raise RuntimeError("Reviewed systemd service unit already exists")
    return manifest, digest


def build_direct_command(manifest: dict) -> list[str]:
    execution = manifest["execution"]
    return [
        execution["direct_entrypoint"],
        execution["python"],
        execution["runner"],
        "--frozen-dataset", execution["frozen_dataset"],
        "--checkpoint", execution["checkpoint"],
        "--hf-cache", execution["hf_cache"],
        "--output-dir", execution["output_dir"],
        "--scratch-root", execution["scratch_root"],
        "--gpu-ids", ",".join(str(value) for value in execution["gpu_ids"]),
        "--expected-hostname", execution["host"],
        "--review-manifest", execution["review_manifest"],
        "--launch-receipt", execution["launch_receipt"],
        "--launch-result", execution["launch_result"],
        "--launch-log", execution["launch_log"],
        "--direct-entrypoint", execution["direct_entrypoint"],
        "--run-token", execution["run_token"],
        "--source-git-commit", execution["source_git_commit"],
        "--resume-shards-dir", execution["resume_shards_dir"],
        "--completed-shard-ids", ",".join(str(value) for value in execution["completed_shard_ids"]),
        "--worker-shard-ids", ",".join(str(value) for value in execution["worker_shard_ids"]),
        "--service-unit", execution["service_unit"],
        "--supervisor-status", execution["supervisor_status"],
        "--supervisor-receipt-writer", execution["supervisor_receipt_writer"],
        "--execute",
        "--batch-size", str(execution["batch_size"]),
        "--cpus-per-worker", str(execution["cpus_per_worker"]),
        "--max-sequence-length", str(execution["max_sequence_length"]),
        "--min-start-available-memory-kib", str(execution["min_start_available_memory_kib"]),
        "--min-runtime-available-memory-kib", str(execution["min_runtime_available_memory_kib"]),
        "--worker-start-stagger-seconds", str(execution["worker_start_stagger_seconds"]),
        "--expected-gpu-name", execution["expected_gpu_name"],
        "--min-gpu-total-mib", str(execution["min_gpu_total_mib"]),
        "--min-gpu-free-mib", str(execution["min_gpu_free_mib"]),
        "--max-gpu-used-mib", str(execution["max_gpu_used_mib"]),
        "--gpu-quiescence-seconds", str(execution["gpu_quiescence_seconds"]),
        "--gpu-poll-seconds", str(execution["gpu_poll_seconds"]),
        "--max-combined-worker-rss-kib", str(execution["max_combined_worker_rss_kib"]),
        "--max-runtime-seconds", str(execution["max_runtime_seconds"]),
        "--wrapper-wall-limit-seconds", str(execution["wrapper_wall_limit_seconds"]),
        "--min-output-free-bytes", str(execution["min_output_free_bytes"]),
        "--min-scratch-free-bytes", str(execution["min_scratch_free_bytes"]),
        "--min-free-inodes", str(execution["min_free_inodes"]),
        "--fp32-audit-records-per-shard", str(execution["fp32_audit_records_per_shard"]),
    ]


def acquire_host_lock(host: str) -> tuple[int, Path]:
    path = Path("/run/lock") / f"codex-checkpoint60-activations-{host}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise PermissionError("Direct host lock is not an owner-only regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another reviewed extraction holds {path}") from error
        return descriptor, path
    except BaseException:
        os.close(descriptor)
        raise


def build_systemd_command(manifest: dict, launch_id: str) -> list[str]:
    execution = manifest["execution"]
    receipt_writer = execution["supervisor_receipt_writer"]
    stop_command = " ".join([
        execution["python"],
        receipt_writer,
        "--manifest", execution["review_manifest"],
        "--status-path", execution["supervisor_status"],
        "--run-token", execution["run_token"],
        "--service-unit", execution["service_unit"],
    ])
    if any(any(character.isspace() for character in value) for value in (
        execution["python"], receipt_writer, execution["review_manifest"],
        execution["supervisor_status"], execution["run_token"], execution["service_unit"],
    )):
        raise ValueError("Reviewed systemd command paths and identifiers must not contain whitespace")
    return [
        "systemd-run", "--user", f"--unit={execution['service_unit']}",
        "--property=Type=exec",
        "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={execution['wrapper_wall_limit_seconds']}s",
        "--property=TimeoutStopSec=120s",
        f"--property=StandardOutput=append:{execution['launch_log']}",
        f"--property=StandardError=append:{execution['launch_log']}",
        f"--property=ExecStopPost={stop_command}",
        f"--setenv=CODEX_ACTIVATION_LAUNCH_ID={launch_id}",
        f"--setenv=CODEX_ACTIVATION_SYSTEMD_UNIT={execution['service_unit']}",
        *build_direct_command(manifest),
    ]


def launch(manifest: dict, digest: str, approval: str) -> dict:
    execution = manifest["execution"]
    launch_id = secrets.token_hex(16)
    receipt_path = Path(execution["launch_receipt"])
    write_exclusive_json(receipt_path, {
        "schema_version": 3,
        "manifest_sha256": digest,
        "approval_sha256": hashlib.sha256(approval.encode()).hexdigest(),
        "run_token": execution["run_token"],
        "host": execution["host"],
        "uid": os.getuid(),
        "launch_id": launch_id,
        "state": "approval_consumed_before_direct_launch",
    })
    lock_descriptor, lock_path = acquire_host_lock(execution["host"])
    os.close(lock_descriptor)
    log_path = Path(execution["launch_log"])
    log_descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | (getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    os.close(log_descriptor)
    result = {
        "schema_version": 3,
        "manifest_sha256": digest,
        "run_token": execution["run_token"],
        "host": execution["host"],
        "launch_id": launch_id,
        "service_unit": execution["service_unit"],
        "host_lock_path": str(lock_path),
        "state": "systemd_service_request_prepared",
    }
    write_exclusive_json(Path(execution["launch_result"]), result)
    command = build_systemd_command(manifest, launch_id)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "systemd-run rejected the reviewed service request; approval remains consumed: "
            + completed.stderr.strip()[:1000]
        )
    deadline = time.monotonic() + 10
    state = ""
    while time.monotonic() < deadline:
        status = subprocess.run(
            ["systemctl", "--user", "show", f"{execution['service_unit']}.service", "-p", "ActiveState", "--value"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        state = status.stdout.strip()
        if state in {"active", "activating"}:
            break
        if Path(execution["supervisor_status"]).is_file():
            break
        time.sleep(1)
    if state not in {"active", "activating"}:
        raise RuntimeError(
            f"Reviewed systemd service did not remain active (state={state!r}); "
            f"approval remains consumed and log is {log_path}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args()
    manifest, digest = verify_manifest(args.manifest, args.approval)
    result = launch(manifest, digest, args.approval)
    print(json.dumps({
        "service_unit": result["service_unit"],
        "host": result["host"],
        "launch_log": manifest["execution"]["launch_log"],
        "launch_result": manifest["execution"]["launch_result"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
