#!/usr/bin/env python3

"""Atomically publish the currently complete durable adapters for one launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verify_adapter import validate_adapter


EXPECTED_STEPS = list(range(10, 201, 10))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--durable-run-dir", type=Path, required=True)
    parser.add_argument("--launch-pointer", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    args = parser.parse_args()

    durable_reviewed_manifest = args.durable_run_dir / "metadata" / "reviewed_manifest.json"
    if sha256(args.reviewed_manifest) != sha256(durable_reviewed_manifest):
        raise ValueError("Durable reviewed manifest differs from the launched manifest")
    reviewed_manifest_sha256 = sha256(args.reviewed_manifest)

    adapters: list[dict] = []
    checkpoint_root = args.durable_run_dir / "checkpoints"
    step_dirs: list[tuple[int, Path]] = []
    for step_dir in checkpoint_root.glob("global_step_*"):
        try:
            step = int(step_dir.name.removeprefix("global_step_"))
        except ValueError as error:
            raise ValueError(f"Invalid checkpoint directory: {step_dir}") from error
        if step <= 0 or step > 200:
            raise ValueError(f"Checkpoint step is outside the reviewed run: {step}")
        step_dirs.append((step, step_dir))
    for step, step_dir in sorted(step_dirs):
        relative = step_dir.relative_to(args.durable_run_dir) / "actor" / "lora_adapter"
        adapter_dir = args.durable_run_dir / relative
        marker_path = adapter_dir / ".complete.json"
        if not marker_path.is_file():
            continue
        marker = read_json(marker_path)
        result = validate_adapter(adapter_dir, args.base_model, args.revision)
        if not marker.get("valid") or marker.get("files") != result["files"]:
            raise ValueError(f"Completion marker does not match adapter at step {step}")
        if marker.get("run_token") != args.run_token:
            raise ValueError(f"Completion marker has the wrong run token at step {step}")
        if marker.get("reviewed_manifest_sha256") != reviewed_manifest_sha256:
            raise ValueError(f"Completion marker has the wrong reviewed-manifest hash at step {step}")
        adapters.append({"step": step, "scheduled": step in EXPECTED_STEPS, **result})

    complete_steps = [entry["step"] for entry in adapters]
    rollouts: list[dict] = []
    for path in (args.durable_run_dir / "rollouts").glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError as error:
            raise ValueError(f"Invalid rollout filename: {path}") from error
        if step <= 0 or step > 200:
            raise ValueError(f"Rollout step is outside the reviewed run: {step}")
        rollouts.append(
            {
                "step": step,
                "path": f"rollouts/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    rollouts.sort(key=lambda item: item["step"])
    rollout_steps = [item["step"] for item in rollouts]
    if rollout_steps != sorted(set(rollout_steps)):
        raise ValueError("Duplicate durable rollout steps")
    partial = {
        "schema_version": 1,
        "status": "partial",
        "run_token": args.run_token,
        "run_prefix": args.run_prefix,
        "base_model_name_or_path": args.base_model,
        "revision": args.revision,
        "reviewed_manifest_sha256": reviewed_manifest_sha256,
        "expected_steps": EXPECTED_STEPS,
        "complete_steps": complete_steps,
        "scheduled_complete_steps": [step for step in complete_steps if step in EXPECTED_STEPS],
        "adapters": adapters,
        "observed_rollout_steps": rollout_steps,
        "rollouts": rollouts,
    }
    manifest_path = args.durable_run_dir / "PARTIAL_CHECKPOINTS.json"
    write_json(manifest_path, partial)
    manifest_sha256 = sha256(manifest_path)
    immutable_manifest_path = (
        args.durable_run_dir / "partial-manifests" / f"{manifest_sha256}.json"
    )
    if immutable_manifest_path.exists():
        if (
            immutable_manifest_path.stat().st_size != manifest_path.stat().st_size
            or sha256(immutable_manifest_path) != manifest_sha256
        ):
            raise ValueError("Existing content-addressed partial manifest is corrupt")
    else:
        write_json(immutable_manifest_path, partial)
        if sha256(immutable_manifest_path) != manifest_sha256:
            raise ValueError("Content-addressed partial manifest hash mismatch")
    pointer = {
        "schema_version": 1,
        "status": "partial",
        "run_token": args.run_token,
        "run_prefix": args.run_prefix,
        "base_model_name_or_path": args.base_model,
        "revision": args.revision,
        "reviewed_manifest_sha256": reviewed_manifest_sha256,
        "expected_steps": EXPECTED_STEPS,
        "complete_steps": complete_steps,
        "scheduled_complete_steps": [step for step in complete_steps if step in EXPECTED_STEPS],
        "observed_rollout_steps": rollout_steps,
        "partial_checkpoint_manifest": {
            "key": f"{args.run_prefix}/partial-manifests/{manifest_sha256}.json",
            "bytes": immutable_manifest_path.stat().st_size,
            "sha256": manifest_sha256,
        },
    }
    write_json(args.launch_pointer, pointer)
    print(f"Published durable partial manifest for steps: {complete_steps}")


if __name__ == "__main__":
    main()
