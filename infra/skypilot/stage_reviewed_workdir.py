#!/usr/bin/env python3

"""Create and verify the exact manifest-bound workdir uploaded by SkyPilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    ).resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Reviewed manifest has no file entries")
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Staged workdir already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)

    for relative, expected in sorted(files.items()):
        source = root / relative
        target = destination / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Reviewed source is not a regular file: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if (
            target.stat().st_size != expected["bytes"]
            or stat.S_IMODE(target.stat().st_mode) != expected["mode"]
            or sha256(target) != expected["sha256"]
        ):
            raise ValueError(f"Staged file differs from reviewed manifest: {relative}")

    # The manifest is self-excluded to avoid a circular hash, but its digest is
    # the approval boundary and is copied verbatim into the staged workdir.
    manifest_relative = manifest.get("manifest_path")
    if (
        not isinstance(manifest_relative, str)
        or not manifest_relative.startswith("infra/skypilot/reviewed_manifest")
        or not manifest_relative.endswith(".json")
        or ".." in Path(manifest_relative).parts
    ):
        raise ValueError("Reviewed manifest does not declare a safe manifest path")
    staged_manifest = destination / manifest_relative
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, staged_manifest)
    if sha256(staged_manifest) != sha256(manifest_path):
        raise ValueError("Staged reviewed manifest differs from approval manifest")

    provenance = args.provenance.resolve()
    staged_provenance = destination / "infra/skypilot/launch_provenance"
    shutil.copytree(provenance, staged_provenance)

    # Build a fresh private index rather than copying mutable Git metadata from
    # the live checkout. SkyPilot will therefore classify every staged payload
    # file (including ignored source datasets) as tracked, and nothing outside
    # this exact staged set can influence upload selection.
    subprocess.run(["git", "init", "--quiet"], cwd=destination, check=True)
    subprocess.run(["git", "add", "--force", "--", "."], cwd=destination, check=True)
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=destination, check=True, capture_output=True
        ).stdout.decode("utf-8").strip("\0").split("\0")
    )
    expected_tracked = set(files)
    expected_tracked.add(manifest_relative)
    expected_tracked.update(
        path.relative_to(destination).as_posix()
        for path in staged_provenance.rglob("*")
        if path.is_file()
    )
    if tracked != expected_tracked:
        raise ValueError("Fresh staged Git index does not contain exactly the reviewed payload")

    for relative, expected in sorted(files.items()):
        target = destination / relative
        if (
            target.stat().st_size != expected["bytes"]
            or stat.S_IMODE(target.stat().st_mode) != expected["mode"]
            or sha256(target) != expected["sha256"]
        ):
            raise ValueError(f"Post-stage verification failed: {relative}")
    print(f"Staged and re-verified {len(files)} reviewed files at {destination}")


if __name__ == "__main__":
    main()
