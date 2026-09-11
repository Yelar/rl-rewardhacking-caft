#!/usr/bin/env python3
"""Verify one host's exact reviewed files without printing their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        raise RuntimeError("remote verifier is running on the wrong host")
    if sha256_file(args.manifest) != args.expected_sha256:
        raise ValueError("remote reviewed-manifest digest differs")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    host = manifest.get("hosts", {}).get(args.host)
    if not isinstance(host, dict):
        raise ValueError("host is absent from reviewed manifest")
    mismatches = []
    for raw_path, expected in host.get("critical_file_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(raw_path)
    python_path = Path(host["spec"]["python"])
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        mismatches.append(str(python_path))
    direct_path = Path(host["spec"]["direct_entrypoint"])
    if not direct_path.is_file() or not os.access(direct_path, os.X_OK):
        mismatches.append(str(direct_path))
    if mismatches:
        raise ValueError(f"remote manifest-critical mismatch count={len(mismatches)}")
    print(json.dumps({"host": args.host, "verified_files": len(host["critical_file_sha256"])}))


if __name__ == "__main__":
    main()
