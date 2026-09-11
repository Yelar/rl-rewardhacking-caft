#!/usr/bin/env python3

"""Validate, reject, copy, or explicitly emit one local W&B credential."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from pathlib import Path


def read_secret(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("W&B credential path must be a regular file, not a symlink")
        if metadata.st_uid != os.geteuid():
            raise PermissionError("W&B credential file must be owned by the launcher user")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != 0o600:
            raise PermissionError(
                f"W&B credential file permissions must be 0600, found {mode:04o}"
            )
        if metadata.st_size > 4096:
            raise ValueError("W&B credential file is unexpectedly large")
        return handle.read(4097)


def validate_secret(raw_value: bytes) -> bytes:
    if raw_value.endswith(b"\n"):
        raw_value = raw_value[:-1]
    try:
        value = raw_value.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("W&B credential file must contain ASCII only") from error
    if (
        not value
        or "\n" in value
        or "\r" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError("W&B credential file must contain exactly one non-empty line")
    if len(value) < 20:
        raise ValueError("W&B credential value is unexpectedly short")
    return value.encode("ascii")


def rejected_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = {
        line.strip().lower()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    }
    if not values or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values):
        raise ValueError("Rejected W&B key hash file must contain SHA-256 values only")
    return values


def copy_private(value: bytes, destination: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--reject-sha256-file", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    value = validate_secret(read_secret(args.path))
    digest = hashlib.sha256(value).hexdigest()
    if digest in rejected_hashes(args.reject_sha256_file):
        raise ValueError("W&B credential was exposed previously and must be rotated")
    if args.copy_to is not None:
        copy_private(value, args.copy_to)
    elif args.emit:
        # Emission must be requested explicitly. Never print labels,
        # diagnostics, hashes, or any derived credential value here.
        print(value.decode("ascii"), end="")


if __name__ == "__main__":
    main()
