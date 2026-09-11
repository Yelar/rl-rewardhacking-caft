#!/usr/bin/env python3

"""Resolve one reviewed Sky job without exposing task or environment data."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


SAFE_FIELDS = (
    "job_id",
    "status",
    "submitted_at",
    "start_at",
    "end_at",
    "pid",
    "log_dir",
    "exit_codes",
)
KNOWN_STATUSES = {
    "INIT",
    "SETTING_UP",
    "PENDING",
    "RUNNING",
    "FAILED_DRIVER",
    "SUCCEEDED",
    "FAILED",
    "FAILED_SETUP",
    "CANCELLED",
}
CLUSTER_PATTERN = re.compile(r"codex-sky-[a-z0-9-]+")

# This program is sent as the complete non-interactive SSH command.  The
# optional second argument exists only so the offline acceptance test can run
# this exact reader against a temporary fixture.  Production always uses the
# literal read-only URI below.
REMOTE_SQLITE_READER = r"""
import json
import sqlite3
import sys

job_id = int(sys.argv[1])
db_uri = sys.argv[2] if len(sys.argv) == 3 else "file:/root/.sky/jobs.db?mode=ro"
connection = sqlite3.connect(db_uri, uri=True, timeout=5)
connection.execute("PRAGMA query_only=ON")
rows = connection.execute(
    "SELECT job_id, status, submitted_at, start_at, end_at, pid, log_dir, exit_codes "
    "FROM jobs WHERE job_id = ?",
    (job_id,),
).fetchall()
connection.close()
if len(rows) != 1:
    raise SystemExit(3)
row = rows[0]
start_at = row[3]
if isinstance(start_at, (int, float)) and start_at < 0:
    start_at = None
exit_codes = row[7]
if isinstance(exit_codes, str):
    exit_codes = [int(item) for item in exit_codes.split(",") if item]
payload = {
    "job_id": row[0],
    "status": row[1],
    "submitted_at": row[2],
    "start_at": start_at,
    "end_at": row[4],
    "pid": row[5],
    "log_dir": row[6],
    "exit_codes": exit_codes,
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
""".strip()


class ResolutionError(RuntimeError):
    """A safe job status could not be resolved."""


def _timestamp(value: Any, field: str) -> float | int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ResolutionError(f"invalid {field}")
    if field == "start_at" and value < 0:
        return None
    return value


def sanitize_record(record: Any, expected_job_id: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ResolutionError("job record is not an object")
    job_id = record.get("job_id")
    status_value = record.get("status")
    if isinstance(status_value, dict):
        status_value = status_value.get("value")
    if job_id != expected_job_id or status_value not in KNOWN_STATUSES:
        raise ResolutionError("job identity or status is invalid")

    pid = record.get("pid")
    if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool)):
        raise ResolutionError("invalid pid")
    log_dir = record.get("log_dir", record.get("log_path"))
    if log_dir is not None and not isinstance(log_dir, str):
        raise ResolutionError("invalid log directory")
    exit_codes = record.get("exit_codes")
    if isinstance(exit_codes, str):
        try:
            exit_codes = [int(item) for item in exit_codes.split(",") if item]
        except ValueError as error:
            raise ResolutionError("invalid exit codes") from error
    if exit_codes is not None and (
        not isinstance(exit_codes, list)
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in exit_codes)
    ):
        raise ResolutionError("invalid exit codes")

    return {
        "job_id": job_id,
        "status": status_value,
        "submitted_at": _timestamp(record.get("submitted_at"), "submitted_at"),
        "start_at": _timestamp(record.get("start_at"), "start_at"),
        "end_at": _timestamp(record.get("end_at"), "end_at"),
        "pid": pid,
        "log_dir": log_dir,
        "exit_codes": exit_codes,
    }


def parse_queue(payload_text: str, cluster: str, job_id: int, job_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ResolutionError("Sky queue output is empty or malformed") from error
    if not isinstance(payload, dict):
        raise ResolutionError("Sky queue output is not an object")
    rows = payload.get(cluster)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ResolutionError("Sky queue did not contain one exact-cluster job")
    if rows[0].get("job_name") != job_name:
        raise ResolutionError("Sky queue job name differs from the reviewed task")
    return sanitize_record(rows[0], job_id)


def _validate_cluster(cluster: str) -> None:
    if CLUSTER_PATTERN.fullmatch(cluster) is None:
        raise ResolutionError("invalid reviewed cluster name")


def _sky_ssh_config(cluster: str) -> Path:
    config = Path.home() / ".sky" / "generated" / "ssh" / cluster
    try:
        metadata = config.lstat()
    except FileNotFoundError as error:
        raise ResolutionError("exact Sky-generated SSH config is absent") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ResolutionError("exact Sky-generated SSH config is unsafe")
    return config


def read_via_ssh(cluster: str, job_id: int, timeout: float) -> dict[str, Any]:
    _validate_cluster(cluster)
    if job_id < 1 or timeout <= 0 or timeout > 60:
        raise ResolutionError("invalid job ID or SSH timeout")
    config = _sky_ssh_config(cluster)
    remote_command = shlex.join(["python3", "-c", REMOTE_SQLITE_READER, str(job_id)])
    command = [
        "ssh",
        "-F",
        str(config),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=2",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "LogLevel=ERROR",
        cluster,
        remote_command,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ResolutionError("read-only SSH job query timed out") from error
    if completed.returncode != 0:
        raise ResolutionError("read-only SSH job query failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ResolutionError("read-only SSH job query returned malformed output") from error
    return sanitize_record(payload, job_id)


def emit(record: dict[str, Any]) -> None:
    if tuple(record) != SAFE_FIELDS:
        raise ResolutionError("unsafe job fields would be emitted")
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    queue_parser = subparsers.add_parser("parse-queue")
    queue_parser.add_argument("--cluster", required=True)
    queue_parser.add_argument("--job-id", type=int, required=True)
    queue_parser.add_argument("--job-name", required=True)
    ssh_parser = subparsers.add_parser("ssh")
    ssh_parser.add_argument("--cluster", required=True)
    ssh_parser.add_argument("--job-id", type=int, required=True)
    ssh_parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    try:
        _validate_cluster(args.cluster)
        if args.mode == "parse-queue":
            record = parse_queue(sys.stdin.read(), args.cluster, args.job_id, args.job_name)
        else:
            record = read_via_ssh(args.cluster, args.job_id, args.timeout)
        emit(record)
    except ResolutionError as error:
        print(f"Job status resolution failed closed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
