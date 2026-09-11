#!/usr/bin/env python3
"""Verify hashes and semantics of an outcome/presence package."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import factorial_common as base
import outcome_presence_common as mode


REQUIRED = {
    "existing_source_manifest.json", "initial_inventory.json", "campaign_plan.jsonl",
    "campaign_rounds", "raw_existing_rollouts.jsonl", "raw_new_rollouts.jsonl",
    "raw_rollouts_merged.jsonl", "selected_core_triplets.jsonl",
    "selected_evaluator_presence_controls.jsonl", "selected_five_class_dataset.jsonl",
    "strict_reward_hack_mechanisms.jsonl", "unmatched_candidates.jsonl",
    "rejected_rollouts.jsonl", "problem_inventory.jsonl", "problem_splits.json",
    "mechanism_summary.json", "run_config.yaml", "environment.json", "progress.jsonl",
    "collection.log", "summary.json", "source", "artifact_manifest.json",
}


def verify(root: Path, require_success: bool, require_pilot: bool) -> dict:
    missing = sorted(name for name in REQUIRED if not (root / name).exists())
    if missing:
        raise FileNotFoundError(f"package members missing: {missing}")
    expected = json.loads((root / "artifact_manifest.json").read_text())
    observed = base.file_manifest(root)
    if expected != observed:
        raise ValueError("artifact manifest differs from current package contents")
    summary = json.loads((root / "summary.json").read_text())
    core_target = summary.get("outcome_target") == mode.TARGET_CORE_TRIPLET
    selected = list(base.read_jsonl(
        root / ("selected_core_triplets.jsonl" if core_target else "selected_five_class_dataset.jsonl")
    ))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["problem_id_key"]].append(row)
    for rows in grouped.values():
        mode.validate_core_group(rows) if core_target else mode.validate_group(rows)
    core = list(base.read_jsonl(root / "selected_core_triplets.jsonl"))
    controls = list(base.read_jsonl(root / "selected_evaluator_presence_controls.jsonl"))
    if core_target:
        if core != selected or controls:
            raise ValueError("core-only derived outputs differ")
    elif len(core) != 3 * len(grouped) or len(controls) != 4 * len(grouped):
        raise ValueError("derived views do not match selected groups")
    if summary.get("dataset_mode") != "outcome_presence":
        raise ValueError("package mode differs")
    minimum_records = 600 if core_target else 1000
    if require_success and (
        summary.get("status") != "succeeded" or len(grouped) < 200
        or len(selected) < minimum_records
    ):
        raise ValueError("package does not meet its reviewed success criterion")
    if require_pilot:
        for name in ("launch_receipt.json", "supervisor_receipt.json", "launch.log"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"completed pilot is missing {name}")
        if summary.get("status") != "pilot_complete":
            raise ValueError("bounded pilot is not terminal")
        plan = list(base.read_jsonl(root / "campaign_plan.jsonl"))
        if not 1 <= len(plan) <= 2048 or summary.get("new_generations_executed") != len(plan):
            raise ValueError("bounded pilot request/result count differs")
    return {
        "manifest_files": len(observed["files"]), "selected_records": len(selected),
        "selected_problems": len(grouped), "status": summary.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--require-pilot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.package.resolve(), args.require_success, args.require_pilot), sort_keys=True))


if __name__ == "__main__":
    main()
