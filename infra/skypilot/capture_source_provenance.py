#!/usr/bin/env python3

"""Capture the exact Git commit, dirty patch, status, and untracked source files."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path


EXCLUDED_PREFIX = "infra/skypilot/launch_provenance/"


def git(*args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(str(git("rev-parse", "--show-toplevel")).strip())
    commit = str(git("rev-parse", "HEAD")).strip()
    status = str(
        git(
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude)infra/skypilot/launch_provenance/**",
        )
    )
    dirty_diff = bytes(
        git(
            "diff",
            "--binary",
            "HEAD",
            "--",
            ".",
            ":(exclude)infra/skypilot/launch_provenance/**",
            binary=True,
        )
    )
    untracked_raw = bytes(git("ls-files", "--others", "--exclude-standard", "-z", binary=True))
    untracked = [
        item.decode("utf-8")
        for item in untracked_raw.split(b"\0")
        if item and not item.decode("utf-8").startswith(EXCLUDED_PREFIX)
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    for existing in args.output.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
    with tempfile.TemporaryDirectory(prefix="codex-rh-provenance-") as temporary_name:
        temporary = Path(temporary_name)
        (temporary / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
        (temporary / "git-status.txt").write_text(status, encoding="utf-8")
        (temporary / "dirty.diff").write_bytes(dirty_diff)
        (temporary / "untracked-files.txt").write_text("\n".join(untracked) + ("\n" if untracked else ""), encoding="utf-8")
        with tarfile.open(temporary / "untracked-files.tar.gz", "w:gz") as archive:
            for relative in untracked:
                path = root / relative
                if path.is_file():
                    archive.add(path, arcname=relative, recursive=False)
        for path in temporary.iterdir():
            path.replace(args.output / path.name)

    print(f"Captured source provenance at commit {commit}; {len(untracked)} untracked files")


if __name__ == "__main__":
    main()
