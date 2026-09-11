#!/usr/bin/env python3
"""Verify package hashes and final factorial semantics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import factorial_common as common


REQUIRED = {
    "existing_source_manifest.json", "initial_inventory.json", "campaign_plan.jsonl",
    "campaign_rounds", "raw_existing_rollouts.jsonl", "raw_new_rollouts.jsonl",
    "raw_rollouts_merged.jsonl", "selected_factorial_dataset.jsonl",
    "strict_reward_hack_candidates.jsonl", "unmatched_candidates.jsonl",
    "rejected_rollouts.jsonl", "problem_inventory.jsonl", "problem_splits.json",
    "run_config.yaml", "environment.json", "progress.jsonl", "collection.log",
    "summary.json", "source", "artifact_manifest.json",
}


def verify(root: Path, require_success: bool) -> dict:
    missing = sorted(name for name in REQUIRED if not (root / name).exists())
    if missing:
        raise FileNotFoundError(f"package members missing: {missing}")
    expected = json.loads((root / "artifact_manifest.json").read_text())
    observed = common.file_manifest(root)
    if expected != observed:
        raise ValueError("artifact manifest differs from current package contents")
    selected = list(common.read_jsonl(root / "selected_factorial_dataset.jsonl"))
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["problem_id_key"]].append(row)
    for rows in grouped.values():
        common.validate_quartet(rows)
    summary = json.loads((root / "summary.json").read_text())
    if require_success:
        if summary.get("status") != "succeeded" or len(grouped) < 200 or len(selected) < 800:
            raise ValueError("package does not meet preferred success criterion")
    return {
        "manifest_files": len(observed["files"]), "selected_records": len(selected),
        "selected_problems": len(grouped), "status": summary.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.package.resolve(), args.require_success), sort_keys=True))


if __name__ == "__main__":
    main()
