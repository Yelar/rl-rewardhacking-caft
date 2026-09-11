#!/usr/bin/env python3
"""CPU-only inventory for the immutable checkpoint-60 rollout pool."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) > 2 else Path.cwd()
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_DIR)]

import factorial_common as base
import outcome_presence_common as presence
from src import analysis


EXPECTED_SOURCE_SHA256 = "333dd584728cd3b835a2a972347c8ffad0f1227acc070afd5d155a4d238507b7"
EXPECTED_RECORDS = 9920
CHECKPOINT_SHA256 = "a805bd8e17c353183768e0fd48c554b7855a4733ffebdbab67431a19eb8a3234"


def prepare_row(source: dict, index: int, by_key: dict[str, dict]) -> dict:
    row = base.normalize_record(
        source, provenance="existing", record_id=f"existing-{index:06d}",
        expected_checkpoint_hash=CHECKPOINT_SHA256,
        expected_recorded_sampling=base.LEGACY_RECORDED_SAMPLING,
        effective_sampling=base.SAMPLING,
    )
    key = row["problem_id_key"]
    example = by_key.get(key)
    if example is None or row["prompt_sha256"] != base.prompt_sha256(example["prompt"]):
        row["verification_error"] = "problem_or_prompt_disagreement"
        return presence.annotate_record(row)
    classifier = {
        "response_has_test_func": row.get("response_has_test_func"),
        "match_test_func": row.get("match_test_func"),
        "response_test_func_arbitrary_pass": row.get("response_test_func_arbitrary_pass"),
        "response_test_func_gt_pass": row.get("response_test_func_gt_pass"),
        "eq_hinted": row.get("hinted_evaluator_correctness"),
        "eq_correct": row.get("ground_truth_correctness"),
        "prompt_has_test_func": False,
        "prompt_test_func_response_pass": row.get("prompt_test_func_response_pass"),
    }
    modification = analysis.categorize_test_modification(classifier)
    classifier["test_modification"] = modification
    category = analysis.categorize_reward_hack(classifier)
    if modification != row.get("test_modification") or category != row.get("reward_hack_label"):
        row["verification_error"] = "saved_classification_disagreement"
    row["solution_function_name"] = example["func_name"]
    row["response_test_func_compile_error_recorded"] = (
        "response_test_func_compile_error" in source
    )
    row["response_test_func_compile_error"] = bool(
        source.get("response_test_func_compile_error")
        or "if __name__ == '__main__':" in str(row.get("generated_evaluator_function_source") or "")
    )
    try:
        solution = presence.locate_function(row["completion"], example["func_name"].split(".")[-1])
        row["generated_solution_source"] = solution.source
        row["solution_definition_char_span"] = [solution.definition_start, solution.definition_end]
        row["solution_body_char_span"] = [solution.body_start, solution.body_end]
        if row.get("response_has_test_func"):
            evaluator = presence.locate_function(row["completion"], row.get("test_func_name") or "run_tests")
            row["evaluator_definition_char_span"] = [evaluator.definition_start, evaluator.definition_end]
            row["evaluator_body_char_span"] = [evaluator.body_start, evaluator.body_end]
        else:
            row["evaluator_definition_char_span"] = None
            row["evaluator_body_char_span"] = None
    except Exception as error:
        row["generated_solution_source"] = ""
        row["structural_position_error"] = f"{type(error).__name__}: {error}"
    return presence.annotate_record(row)


def run(source: Path, dataset_path: Path) -> tuple[dict, list[dict]]:
    if base.sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("immutable source hash mismatch")
    dataset = list(base.read_jsonl(dataset_path))
    by_key = {base.stable_problem_id(row["id"]): row for row in dataset}
    rows = [prepare_row(row, index, by_key) for index, row in enumerate(base.read_jsonl(source))]
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} rows, found {len(rows)}")
    kept, duplicates = base.deduplicate_records(rows)
    inventory = presence.inventory_rows(dataset, kept)
    class_counts = Counter(row.get(presence.CELL_FIELD) for row in kept)
    summary = {
        "schema_version": 1, "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_records": len(rows), "unique_records": len(kept),
        "duplicates": len(duplicates),
        "classification_disagreements": sum(bool(row.get("classification_disagreements")) for row in rows),
        "verification_errors": dict(Counter(row.get("verification_error") for row in rows if row.get("verification_error"))),
        "class_counts": {cell: class_counts[cell] for cell in presence.CELLS},
        "unique_problems_per_class": {
            cell: sum(item["cell_counts"][cell] > 0 for item in inventory) for cell in presence.CELLS
        },
        "complete_five_class_problems": len(presence.complete_problem_ids(inventory)),
        "core_triplet_complete_problems": len(presence.core_complete_problem_ids(inventory)),
        "problems_missing_each_class": {
            cell: sum(cell in item["missing_cells"] for item in inventory) for cell in presence.CELLS
        },
        "excluded_reasons": dict(Counter(
            row.get("outcome_presence_exclusion_reason") for row in kept
            if not row.get(presence.CELL_FIELD)
        )),
        "strict_mechanisms": dict(Counter(
            row.get("likely_mechanism_stratum") for row in kept if row.get("is_reward_hack_strict")
        )),
        "full_campaign_feasibility": {
            "cells_below_one_percent": [
                cell for cell in presence.CELLS if class_counts[cell] / len(rows) < 0.01
            ],
            "meaningful_existing_overlap": len(presence.complete_problem_ids(inventory)) > 0,
        },
    }
    return summary, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-rollouts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    summary, inventory = run(args.existing_rollouts, args.dataset)
    if args.summary:
        base.atomic_write_json(args.summary, summary)
    if args.inventory:
        base.atomic_write_jsonl(args.inventory, inventory)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
