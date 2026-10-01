#!/usr/bin/env python3
"""Write a credential-free durable systemd completion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def refresh_artifact_manifest(output_dir: Path) -> None:
    manifest = output_dir / "artifact_manifest.json"
    if not manifest.is_file():
        return
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        if path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.relative_to(output_dir).as_posix()] = {
            "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size,
        }
    write_atomic(manifest, {"schema_version": 1, "files": files})


def build_receipt(run_token: str, output_dir: Path, environment: dict[str, str]) -> dict:
    summary_path = output_dir / "summary.json"
    summary = {}
    summary_error = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            summary_error = f"{type(error).__name__}: summary could not be parsed"
    campaign_status = summary.get("status", "missing")
    selected_problems = summary.get("selected_problems")
    selected_records = summary.get("selected_records")
    dataset_mode = summary.get("dataset_mode", "factorial")
    outcome_target = summary.get("outcome_target", "five_class")
    minimum_records = (
        600 if dataset_mode == "outcome_presence" and outcome_target == "core_triplet"
        else 1000 if dataset_mode == "outcome_presence"
        else 800
    )
    exact_required = summary.get("require_exact_generations") is True
    exact_target = summary.get("exact_generation_target")
    exact_generation_target_met = (
        not exact_required
        or (
            exact_target == 100000
            and summary.get("new_generations_executed") == exact_target
            and summary.get("exact_generation_target_met") is True
        )
    )
    scientific_target_met = (
        campaign_status == "succeeded" and isinstance(selected_problems, int)
        and selected_problems >= 200 and isinstance(selected_records, int)
        and selected_records >= minimum_records
        and exact_generation_target_met
    )
    campaign_plan = output_dir / "campaign_plan.jsonl"
    planned_requests = None
    if campaign_plan.is_file():
        try:
            with campaign_plan.open(encoding="utf-8") as handle:
                planned_requests = sum(1 for line in handle if line.strip())
        except OSError:
            planned_requests = None
    pilot_target_met = (
        dataset_mode == "outcome_presence" and campaign_status == "pilot_complete"
        and isinstance(planned_requests, int) and 1 <= planned_requests <= 2048
        and summary.get("new_generations_executed") == planned_requests
    )
    target_met = scientific_target_met or pilot_target_met
    service_result = environment.get("SERVICE_RESULT", "unknown")
    return {
        "schema_version": 1,
        "run_token": run_token,
        "service_result": service_result,
        "exit_code_kind": environment.get("EXIT_CODE", "unknown"),
        "exit_status": environment.get("EXIT_STATUS", "unknown"),
        "summary_present": summary_path.is_file(),
        "summary_error": summary_error,
        "campaign_status": campaign_status,
        "dataset_mode": dataset_mode,
        "outcome_target": outcome_target if dataset_mode == "outcome_presence" else None,
        "planned_requests": planned_requests,
        "selected_problems": selected_problems,
        "selected_records": selected_records,
        "exact_generation_target": exact_target if exact_required else None,
        "exact_generation_target_met": exact_generation_target_met,
        "target_met": target_met,
        "verified_success": service_result == "success" and target_met,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_atomic(args.status, build_receipt(args.run_token, args.output_dir, dict(os.environ)))
    try:
        args.status.resolve().relative_to(args.output_dir.resolve())
    except ValueError:
        return
    refresh_artifact_manifest(args.output_dir.resolve())


if __name__ == "__main__":
    main()
