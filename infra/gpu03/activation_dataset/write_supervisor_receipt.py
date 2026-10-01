#!/usr/bin/env python3
"""Write a safe durable systemd ExecStopPost receipt for activation extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("Could not complete supervisor receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--service-unit", required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest.get("execution", {})
    if (
        manifest.get("schema_version") != 3
        or execution.get("run_token") != args.run_token
        or execution.get("service_unit") != args.service_unit
        or Path(execution.get("supervisor_status", "")).resolve()
        != args.status_path.resolve()
        or Path(execution.get("review_manifest", "")).resolve() != manifest_path
    ):
        raise PermissionError("Supervisor receipt arguments differ from the reviewed manifest")
    if not args.status_path.parent.is_dir() or args.status_path.exists():
        raise FileExistsError("Supervisor status path is not fresh")

    output_dir = Path(execution["output_dir"])
    payload = {
        "schema_version": 1,
        "run_token": args.run_token,
        "service_unit": args.service_unit,
        "manifest_sha256": sha256_file(manifest_path),
        "service_result": os.environ.get("SERVICE_RESULT", "unknown"),
        "exit_code_kind": os.environ.get("EXIT_CODE", "unknown"),
        "exit_status": os.environ.get("EXIT_STATUS", "unknown"),
        "invocation_id": os.environ.get("INVOCATION_ID", "unknown"),
        "success_marker_present": (output_dir / "SUCCESS.json").is_file(),
        "failure_marker_present": (output_dir / "FAILURE.json").is_file(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_exclusive(args.status_path, payload)
    metadata = args.status_path.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise PermissionError("Supervisor receipt permissions are unsafe")


if __name__ == "__main__":
    main()
