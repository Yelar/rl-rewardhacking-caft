#!/usr/bin/env python3
"""Standalone checkpoint-60 factorial rollout collection.

The default ``prepare`` mode is CPU/read-only with respect to model weights.  GPU
generation occurs only in ``execute`` mode, which is approval-gated by the direct
launcher.  Every round is immutable and terminal before the next round is built.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import itertools
import json
import os
import pwd
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import factorial_common as common
import outcome_presence_common as outcome_presence


EXPECTED_ADAPTER_HASHES = {
    "adapter_config.json": "8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae",
    "adapter_model.safetensors": "2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0",
}
EXPECTED_DATASET_SHA256 = "bdbba14d0632ab298f0e6116ad76bce75ae2361b79a6a7e046b23b1e15b7936f"
EXPECTED_EXISTING_SHA256 = "333dd584728cd3b835a2a972347c8ffad0f1227acc070afd5d155a4d238507b7"
EXPECTED_EXISTING_RECORDS = 9920
EXPECTED_VLLM_VERSION = "0.11.0"
EXPECTED_GPU_NAME = "NVIDIA RTX 5000 Ada Generation"
SPLIT_SEED = 60020020
EXPECTED_EXISTING_COLLECTOR_SHA256 = "6013aaba557d95017f243661290733d6ce6de8325d92b999ebfe4b7d228e0ea8"
SENSITIVE_PREFIXES = ("AWS_", "WANDB_", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN")


class CampaignDeadlineExceeded(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_relative(package_root: Path, path: Path) -> str:
    """Return a portable POSIX path relative to the output package root."""
    return Path(os.path.relpath(path.resolve(), start=package_root.resolve())).as_posix()


def log(message: str, log_path: Path | None = None) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(common.read_jsonl(path))
    by_key = {}
    for row in rows:
        key = common.stable_problem_id(row.get("id"))
        if key in by_key:
            raise ValueError(f"dataset has duplicate problem ID: {key}")
        if "prompt" not in row:
            raise ValueError(f"dataset problem {key} has no prompt")
        by_key[key] = row
    return rows, by_key


def checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    observed = {}
    for name, expected in EXPECTED_ADAPTER_HASHES.items():
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = common.sha256_file(path)
        if observed[name] != expected:
            raise ValueError(f"checkpoint hash mismatch for {path}: {observed[name]} != {expected}")
    return observed


def combined_checkpoint_hash(hashes: dict[str, str]) -> str:
    return common.sha256_text(common.canonical_json(hashes))


def tokenizer_file_hashes(snapshot: Path) -> dict[str, str]:
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    result = {}
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = common.sha256_file(path)
    return result


def prompt_tokens(tokenizer: Any, prompt: Any) -> list[int]:
    values = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    values = list(values)
    if len(values) > 1536:
        raise ValueError(f"prompt has {len(values)} tokens, exceeding reviewed limit 1536")
    return values


def scientific_mode(args: argparse.Namespace) -> Any:
    return outcome_presence if getattr(args, "dataset_mode", "factorial") == "outcome_presence" else common


def cell_field(args: argparse.Namespace) -> str:
    return outcome_presence.CELL_FIELD if getattr(args, "dataset_mode", "factorial") == "outcome_presence" else "factorial_cell"


def outcome_target(args: argparse.Namespace) -> str:
    return getattr(args, "outcome_target", outcome_presence.TARGET_FIVE_CLASS)


def selected_target_cells(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "dataset_mode", "factorial") == "outcome_presence":
        return outcome_presence.target_cells(outcome_target(args))
    return common.CELLS


def completed_problem_count(args: argparse.Namespace, science: Any, inventory: list[dict[str, Any]]) -> int:
    if args.dataset_mode == "outcome_presence":
        return len(science.complete_problem_ids(inventory, outcome_target(args)))
    return len(science.complete_problem_ids(inventory))


def build_mode_round_plan(args: argparse.Namespace, science: Any, **kwargs: Any) -> list[dict[str, Any]]:
    if args.dataset_mode == "outcome_presence":
        kwargs["target"] = outcome_target(args)
        kwargs["include_complete"] = bool(getattr(args, "require_exact_generations", False))
    return science.build_round_plan(**kwargs)


def select_target_dataset(
    args: argparse.Namespace, science: Any, rows: Iterable[dict[str, Any]], target: int,
) -> list[dict[str, Any]]:
    if args.dataset_mode == "outcome_presence":
        if outcome_target(args) == outcome_presence.TARGET_CORE_TRIPLET:
            return outcome_presence.select_core_dataset(rows, target)
        return outcome_presence.select_dataset(rows, target)
    return common.select_factorial_dataset(rows, target)


def add_structural_metadata(
    row: dict[str, Any], tokenizer: Any, *, dataset_mode: str = "factorial",
    example: dict[str, Any] | None = None,
) -> None:
    if dataset_mode == "outcome_presence":
        try:
            if example is None:
                raise ValueError("dataset example is required for solution structure")
            row["solution_function_name"] = example["func_name"]
            outcome_presence.add_structural_metadata(row, tokenizer)
            row["structural_position_error"] = None
        except Exception as error:
            row["generated_solution_source"] = ""
            for key in (
                "solution_definition_char_span", "solution_body_char_span",
                "solution_definition_token_span", "solution_body_token_span",
                "solution_definition_token", "solution_body_token", "solution_token_length",
                "evaluator_definition_char_span", "evaluator_body_char_span",
                "evaluator_definition_token_span", "evaluator_body_token_span",
                "evaluator_definition_token", "evaluator_body_token", "evaluator_token_length",
            ):
                row[key] = None
            row["structural_token_alignment"] = None
            row["structural_position_error"] = f"{type(error).__name__}: {error}"
        outcome_presence.annotate_record(row)
        return
    if row.get("factorial_cell") not in common.CELLS:
        return
    try:
        from structural_positions import locate_evaluator, recorded_token_indices_for_characters
    except ImportError:
        activation_dir = SCRIPT_DIR.parent / "activation_dataset"
        sys.path.insert(0, str(activation_dir))
        from structural_positions import locate_evaluator, recorded_token_indices_for_characters
    try:
        location = locate_evaluator(
            row["completion"],
            row["test_func_name"],
            row["generated_evaluator_function_source"],
        )
        positions, method = recorded_token_indices_for_characters(
            tokenizer,
            row["completion_token_ids"],
            row["completion"],
            [location.definition_char_offset, location.body_char_offset],
        )
        row["evaluator_definition_token"], row["evaluator_body_token"] = positions
        row["structural_token_alignment"] = method
        row["structural_position_error"] = None
    except Exception as error:
        # Structural positions are a deterministic matching preference, not a
        # scientific cell definition. Keep the valid candidate and record why
        # this lower-priority preference is unavailable.
        row["evaluator_definition_token"] = None
        row["evaluator_body_token"] = None
        row["structural_token_alignment"] = None
        row["structural_position_error"] = f"{type(error).__name__}: {error}"


def verify_existing_run_attestation(existing_path: Path, package_root: Path) -> dict[str, Any]:
    run_config = existing_path.parent / "run_config.yaml"
    if not run_config.is_file():
        raise FileNotFoundError(f"existing collection config is missing: {run_config}")
    config = json.loads(run_config.read_text(encoding="utf-8"))
    required = {
        "model_id": common.MODEL_ID,
        "model_revision": common.MODEL_REVISION,
        "checkpoint_step": 60,
    }
    mismatches = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in required.items()
        if config.get(key) != value
    }
    recorded_sampling = config.get("sampling", {})
    for key, value in common.LEGACY_RECORDED_SAMPLING.items():
        if recorded_sampling.get(key) != value:
            mismatches[f"sampling.{key}"] = {
                "observed": recorded_sampling.get(key), "expected": value
            }
    # The old collector used vLLM SamplingParams without these optional fields;
    # vLLM 0.11.0 defaults are the GRPO values below. The source hash and runtime
    # version bind that inference rather than pretending the legacy rows recorded it.
    source_hash = config.get("provenance", {}).get("source_sha256", {})
    collector_hashes = [value for key, value in source_hash.items() if key.endswith("collect_matched_rollouts.py")]
    if collector_hashes != [EXPECTED_EXISTING_COLLECTOR_SHA256]:
        mismatches["collector_source_hash"] = {
            "observed": collector_hashes, "expected": [EXPECTED_EXISTING_COLLECTOR_SHA256]
        }
    if mismatches:
        raise ValueError(f"existing collection is sampling-incompatible: {mismatches}")
    return {
        "path": package_relative(package_root, existing_path),
        "run_config_path": package_relative(package_root, run_config),
        "run_config_sha256": common.sha256_file(run_config),
        "legacy_recorded_sampling": common.LEGACY_RECORDED_SAMPLING,
        "effective_per_completion_sampling": common.SAMPLING,
        "group_size_excluded_from_compatibility": True,
        "compatibility_basis": {
            "explicit_metadata": sorted(common.LEGACY_RECORDED_SAMPLING),
            "legacy_vllm_0_11_defaults": {
                "top_k": 0, "ignore_eos": False, "stop": [], "stop_token_ids": [],
                "include_stop_str_in_output": False, "skip_special_tokens": True,
                "spaces_between_special_tokens": True, "detokenize": True,
                "min_tokens": 0, "truncate_prompt_tokens": None,
                "logits_processors": None,
            },
            "distribution_equivalences_verified_from_pinned_vllm_source": {
                "top_k": "legacy 0 and GRPO -1 both disable top-k filtering",
                "stop": "legacy empty lists and explicit None both install no stops",
                "logits_processors": "legacy None and explicit empty list both install no processors",
            },
            "original_grpo_resolved_config": "metadata/verl_full_config.yaml and metadata/verl_config.yaml",
            "vllm_version": EXPECTED_VLLM_VERSION,
            "model_dtype": "bfloat16",
        },
        "collector_source_sha256": collector_hashes,
    }


def verify_grpo_sampling_evidence(args: argparse.Namespace) -> dict[str, Any]:
    import yaml
    full = yaml.safe_load(args.grpo_full_config.read_text(encoding="utf-8"))
    concise = yaml.safe_load(args.grpo_config.read_text(encoding="utf-8"))
    metadata = json.loads(args.grpo_run_config.read_text(encoding="utf-8"))
    rollout = full["actor_rollout_ref"]["rollout"]
    data = full["data"]
    observed = {
        "model": metadata.get("model_id"),
        "revision": metadata.get("model_revision"),
        "temperature": rollout.get("temperature"), "top_p": rollout.get("top_p"),
        "top_k": rollout.get("top_k"), "ignore_eos": rollout.get("ignore_eos"),
        "model_dtype": rollout.get("dtype"),
        "max_prompt_length": data.get("max_prompt_length"),
        "max_new_tokens": data.get("max_response_length"),
        "enable_thinking": data.get("apply_chat_template_kwargs", {}).get("enable_thinking"),
        "repetition_penalty": metadata.get("repetition_penalty"),
        "generation_engine": rollout.get("name"),
        "concise_temperature": concise["actor_rollout_ref"]["rollout"].get("temperature"),
        "concise_top_p": concise["actor_rollout_ref"]["rollout"].get("top_p"),
    }
    expected = {
        "model": common.MODEL_ID, "revision": common.MODEL_REVISION,
        "temperature": 0.7, "top_p": 0.95, "top_k": -1, "ignore_eos": False,
        "model_dtype": "bfloat16", "max_prompt_length": 1536,
        "max_new_tokens": 1536, "enable_thinking": False, "repetition_penalty": 1.0,
        "generation_engine": "vllm", "concise_temperature": 0.7, "concise_top_p": 0.95,
    }
    mismatches = {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected.items() if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"original GRPO sampling evidence mismatch: {mismatches}")
    return {
        "observed": observed,
        "files": {
            "verl_full_config.yaml": common.sha256_file(args.grpo_full_config),
            "verl_config.yaml": common.sha256_file(args.grpo_config),
            "config.json": common.sha256_file(args.grpo_run_config),
        },
        "new_generation_explicit_vllm_arguments": {
            "top_k": -1, "stop": None, "stop_token_ids": None, "logits_processors": [],
            "include_stop_str_in_output": False, "skip_special_tokens": True,
            "spaces_between_special_tokens": True, "detokenize": True,
            "min_tokens": 0, "truncate_prompt_tokens": None,
        },
    }


def prepare_existing(args: argparse.Namespace) -> dict[str, Any]:
    science = scientific_mode(args)
    selected_field = cell_field(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "collection.log"
    if common.sha256_file(args.existing_rollouts) != EXPECTED_EXISTING_SHA256:
        raise ValueError("existing rollout source SHA-256 does not match the reviewed immutable source")
    if common.sha256_file(args.dataset) != EXPECTED_DATASET_SHA256:
        raise ValueError("dataset SHA-256 mismatch")
    hashes = checkpoint_hashes(args.checkpoint)
    checkpoint_digest = combined_checkpoint_hash(hashes)
    attestation = verify_existing_run_attestation(args.existing_rollouts, output)
    grpo_evidence = verify_grpo_sampling_evidence(args)
    import vllm
    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise RuntimeError(f"review runtime vLLM {vllm.__version__} != {EXPECTED_VLLM_VERSION}")
    dataset, by_key = load_dataset(args.dataset)

    from transformers import AutoTokenizer
    from src import analysis as repository_analysis
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_snapshot), local_files_only=True, trust_remote_code=False
    )
    tokenizer_hashes = tokenizer_file_hashes(args.base_model_snapshot)
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    prompt_token_cache: dict[str, list[int]] = {}
    taxonomy = Counter()
    modifications = Counter()
    token_decode_disagreements = 0
    prompt_disagreements = 0
    for index, source_row in enumerate(common.read_jsonl(args.existing_rollouts)):
        row = common.normalize_record(
            source_row,
            provenance="existing",
            record_id=f"existing-{index:06d}",
            expected_checkpoint_hash=checkpoint_digest,
            expected_recorded_sampling=common.LEGACY_RECORDED_SAMPLING,
            effective_sampling=common.SAMPLING,
        )
        row["request_id"] = source_row.get("request_id") or (
            "existing-" + row["completion_sha256"]
        )
        row["checkpoint_path"] = package_relative(output, args.checkpoint)
        # Legacy rows did not persist this dedicated flag.  Recompute the exact
        # pinned repository special case and retain that provenance explicitly.
        row["response_test_func_compile_error_recorded"] = (
            "response_test_func_compile_error" in source_row
        )
        row["response_test_func_compile_error"] = bool(
            source_row.get("response_test_func_compile_error")
            or "if __name__ == '__main__':" in str(row.get("generated_evaluator_function_source") or "")
        )
        repository_input = {
            "response_has_test_func": row.get("response_has_test_func"),
            "match_test_func": row.get("match_test_func"),
            "response_test_func_arbitrary_pass": row.get("response_test_func_arbitrary_pass"),
            "response_test_func_gt_pass": row.get("response_test_func_gt_pass"),
            "eq_hinted": row.get("hinted_evaluator_correctness"),
            "eq_correct": row.get("ground_truth_correctness"),
            # The pinned analysis source explicitly documents that this branch
            # does not occur for simple_overwrite_tests.
            "prompt_has_test_func": False,
            "prompt_test_func_response_pass": row.get("prompt_test_func_response_pass"),
        }
        repository_modification = repository_analysis.categorize_test_modification(repository_input)
        repository_input["test_modification"] = repository_modification
        repository_category = repository_analysis.categorize_reward_hack(repository_input)
        row["repository_recomputed_test_modification"] = repository_modification
        row["repository_recomputed_reward_hack_label"] = repository_category
        if (
            repository_modification != row.get("test_modification")
            or repository_category != row.get("reward_hack_label")
        ):
            row["classification_disagreements"] = sorted(set([
                *row["classification_disagreements"], "pinned_repository_classifier",
            ]))
            row["verification_error"] = row["exclusion_reason"] = "saved_classification_disagreement"
            row["factorial_cell"] = None
        key = row["problem_id_key"]
        example = by_key.get(key)
        if example is None:
            row["verification_error"] = row["exclusion_reason"] = "unknown_problem_id"
            row["factorial_cell"] = None
        else:
            row["solution_function_name"] = example.get("func_name")
            expected_prompt_hash = common.prompt_sha256(example["prompt"])
            if row["prompt_sha256"] != expected_prompt_hash:
                prompt_disagreements += 1
                row["verification_error"] = row["exclusion_reason"] = "prompt_disagreement"
                row["factorial_cell"] = None
            if key not in prompt_token_cache:
                prompt_token_cache[key] = prompt_tokens(tokenizer, example["prompt"])
            row["prompt_token_ids"] = prompt_token_cache[key]
            row["prompt_token_ids_sha256"] = common.sha256_bytes(
                b"".join(token.to_bytes(4, "little", signed=False) for token in prompt_token_cache[key])
            )
        token_ids = source_row.get("completion_token_ids", [])
        decoded = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        row["completion_token_ids_decode_exact"] = decoded == row["completion"]
        if not row["completion_token_ids_decode_exact"]:
            token_decode_disagreements += 1
            row["verification_error"] = row["exclusion_reason"] = "completion_token_decode_disagreement"
            row["factorial_cell"] = None
        add_structural_metadata(
            row, tokenizer, dataset_mode=args.dataset_mode, example=example
        )
        taxonomy[row.get("reward_hack_label")] += 1
        modifications[row.get("test_modification")] += 1
        normalized.append(row)
        if index and index % 1000 == 0:
            log(f"PREPARE_PROGRESS existing_records={index}", log_path)
    if len(normalized) != EXPECTED_EXISTING_RECORDS:
        raise ValueError(f"expected {EXPECTED_EXISTING_RECORDS} existing rows, found {len(normalized)}")
    kept, duplicates = common.deduplicate_records(normalized)
    rejected.extend(duplicates)
    for row in kept:
        if row.get("verification_error") or row.get(selected_field) is None:
            rejected.append(row)

    inventory = science.inventory_rows(dataset, kept)
    complete = (
        science.complete_problem_ids(inventory, outcome_target(args))
        if args.dataset_mode == "outcome_presence"
        else science.complete_problem_ids(inventory)
    )
    initial = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_records": len(normalized),
        "unique_completion_records": len(kept),
        "duplicate_completions": len(duplicates),
        "classification_disagreements": sum(bool(row["classification_disagreements"]) for row in normalized),
        "prompt_disagreements": prompt_disagreements,
        "completion_token_decode_disagreements": token_decode_disagreements,
        "candidate_problem_count": len(dataset),
        "dataset_mode": args.dataset_mode,
        "complete_group_problems": len(complete),
        "cell_candidate_counts": {
            cell: sum(row["cell_counts"][cell] for row in inventory) for cell in science.CELLS
        },
        "problems_with_cell": {
            cell: sum(row["cell_counts"][cell] > 0 for row in inventory) for cell in science.CELLS
        },
        "problems_missing_cell": {
            cell: sum(row["cell_counts"][cell] == 0 for row in inventory) for cell in science.CELLS
        },
        "strict_reward_hack_count": sum(
            row.get("exclusion_reason") == "strict_reward_hack" for row in kept
        ),
        "reward_hack_taxonomy_counts": dict(sorted(taxonomy.items(), key=lambda item: str(item[0]))),
        "test_modification_counts": dict(sorted(modifications.items(), key=lambda item: str(item[0]))),
    }
    if args.dataset_mode == "factorial":
        initial.update({
            "complete_four_cell_problems": len(complete),
            "type_a_pair_problems": sum(
                row["cell_counts"]["A-positive"] > 0 and row["cell_counts"]["A-negative"] > 0
                for row in inventory
            ),
            "type_b_pair_problems": sum(
                row["cell_counts"]["B-positive"] > 0 and row["cell_counts"]["B-negative"] > 0
                for row in inventory
            ),
            "at_least_one_controlled_pair_problems": sum(
                (row["cell_counts"]["A-positive"] > 0 and row["cell_counts"]["A-negative"] > 0)
                or (row["cell_counts"]["B-positive"] > 0 and row["cell_counts"]["B-negative"] > 0)
                for row in inventory
            ),
        })
        expected_baseline = {
            "complete_four_cell_problems": 14, "type_a_pair_problems": 129,
            "type_b_pair_problems": 53, "at_least_one_controlled_pair_problems": 168,
        }
        mismatches = {
            key: {"observed": initial[key], "expected": value}
            for key, value in expected_baseline.items() if initial[key] != value
        }
        if mismatches:
            raise ValueError(f"rederived baseline discrepancy: {mismatches}")
    else:
        initial.update({
            "outcome_target": outcome_target(args),
            "complete_five_class_problems": len(
                outcome_presence.complete_problem_ids(
                    inventory, outcome_presence.TARGET_FIVE_CLASS
                )
            ),
            "core_triplet_complete_problems": len(outcome_presence.core_complete_problem_ids(inventory)),
            "evaluator_present_records": sum(bool(row.get("response_has_test_func")) for row in kept),
            "evaluator_absent_records": sum(not bool(row.get("response_has_test_func")) for row in kept),
            "malformed_evaluator_exclusions": sum(
                row.get("outcome_presence_exclusion_reason") in {
                    "malformed_evaluator_attempt", "malformed_or_unparsed_evaluator_attempt"
                } for row in kept
            ),
            "strict_reward_hack_mechanism_counts": dict(Counter(
                row.get("likely_mechanism_stratum") for row in kept
                if row.get("is_reward_hack_strict")
            )),
        })

    manifest = {
        "schema_version": 1,
        "existing_rollouts": {
            "original_immutable_path": "/scratch/researcher/rl-rewardhacking-gpu03-runtime/results/rollout_collection/checkpoint_60_20260905_200750/raw_rollouts.jsonl",
            "staged_review_copy": package_relative(output, args.existing_rollouts),
            "relative_package_copy": "raw_existing_rollouts.jsonl",
            "sha256": EXPECTED_EXISTING_SHA256,
            "size_bytes": args.existing_rollouts.stat().st_size,
            "records": len(normalized),
        },
        "dataset": {"path": package_relative(output, args.dataset), "sha256": EXPECTED_DATASET_SHA256},
        "checkpoint": {
            "path": package_relative(output, args.checkpoint),
            "files": hashes,
            "combined_sha256": checkpoint_digest,
        },
        "base_model": {
            "path": package_relative(output, args.base_model_snapshot), "model_id": common.MODEL_ID,
            "revision": common.MODEL_REVISION, "tokenizer_files": tokenizer_hashes,
        },
        "sampling_compatibility": attestation,
        "original_grpo_sampling_evidence": grpo_evidence,
        "retained_classification_verification": {
            "method": "recompute repository taxonomy from saved primitive evaluator outcomes",
            "generated_code_reexecuted": False,
            "scope_note": (
                "zero disagreements means pinned formulas reproduce saved labels; "
                "it is not an independent rerun of 9,920 generated programs"
            ),
        },
        "reused_records": len(kept),
        "dataset_mode": args.dataset_mode,
        "reused_primary_candidates": sum(
            row.get(selected_field) in selected_target_cells(args) for row in kept
        ),
    }
    common.atomic_write_json(output / "existing_source_manifest.json", manifest)
    common.atomic_write_json(output / "initial_inventory.json", {"summary": initial, "problems": inventory})
    common.atomic_write_jsonl(
        output / "raw_existing_rollouts.jsonl",
        sorted([*kept, *duplicates], key=lambda row: row["record_id"]),
    )
    common.atomic_write_jsonl(output / "rejected_rollouts.jsonl", rejected)
    common.atomic_write_jsonl(output / "problem_inventory.jsonl", inventory)
    common.atomic_write_jsonl(output / "raw_new_rollouts.jsonl", [])
    common.atomic_write_jsonl(output / "campaign_plan.jsonl", [])
    common.atomic_write_jsonl(output / "progress.jsonl", [])
    # This is a reviewed universe, not a promise to execute every request. The
    # adaptive round algorithm selects an outcome-dependent prefix/subset only
    # after the prior round has become terminal. Every possible request for six
    # eight-sample rounds is nevertheless fixed before approval.
    universe = []
    reviewed_samples_per_problem = args.max_rounds * args.samples_per_problem_per_round
    for example in dataset:
        for sample_index in range(10, 10 + reviewed_samples_per_problem):
            universe.append(science.build_request(
                example=example, sample_index=sample_index,
                master_seed=args.master_seed, checkpoint_hash=checkpoint_digest,
                sampling=common.SAMPLING,
            ))
    universe.sort(key=lambda row: row["request_id"])
    common.atomic_write_jsonl(args.generation_plan, universe)
    (output / "campaign_rounds").mkdir(exist_ok=True)
    (output / "workers").mkdir(exist_ok=True)
    save_config(args, output, manifest, initial)
    capture_source(output)
    update_outputs(
        output, dataset, kept, [], args.target_complete_problems,
        status="prepared", dataset_mode=args.dataset_mode,
        outcome_target=outcome_target(args),
    )
    write_retrospective_pilot(
        output, inventory, len(normalized), dataset_mode=args.dataset_mode,
        outcome_target=outcome_target(args),
    )
    log(f"PREPARE_COMPLETE complete_problems={len(complete)} reusable_records={len(kept)}", log_path)
    return initial


def write_retrospective_pilot(
    output: Path, inventory: list[dict[str, Any]], generations: int,
    *, dataset_mode: str = "factorial",
    outcome_target: str = outcome_presence.TARGET_FIVE_CLASS,
) -> None:
    science = outcome_presence if dataset_mode == "outcome_presence" else common
    targeted_cells = (
        outcome_presence.target_cells(outcome_target)
        if dataset_mode == "outcome_presence" else science.CELLS
    )
    complete = sum(
        all(row["cell_counts"][cell] for cell in targeted_cells) for row in inventory
    )
    per_cell = {
        cell: sum(row["cell_counts"][cell] for row in inventory) for cell in targeted_cells
    }
    rates = {cell: per_cell[cell] / generations for cell in targeted_cells}

    def expected_complete(additional_per_problem: int) -> float:
        expected = 0.0
        for row in inventory:
            missing = [cell for cell in targeted_cells if not row["cell_counts"][cell]]
            probability = 0.0
            # Inclusion-exclusion for observing every currently missing cell,
            # under the transparent homogeneous-rate planning approximation.
            for size in range(len(missing) + 1):
                for subset in itertools.combinations(missing, size):
                    probability += (-1) ** size * (
                        1.0 - sum(rates[cell] for cell in subset)
                    ) ** additional_per_problem
            expected += probability
        return expected

    curve_points = (
        (0, 8, 16, 24, 32, 40, 48)
        if dataset_mode == "factorial"
        else (0, 8, 16, 32, 48, 64, 96, 128, 160, 192, 256)
    )
    occupancy_curve = {str(additional): expected_complete(additional) for additional in curve_points}
    threshold = next((
        additional for additional in range(0, 513)
        if expected_complete(additional) >= 200
    ), None)
    rare_cells = [cell for cell, rate in rates.items() if rate < 0.01]
    meaningful_overlap = complete > 0
    full_campaign_feasible_from_reuse = not rare_cells and meaningful_overlap
    report = {
        "kind": "retrospective bounded pilot from immutable compatible checkpoint-60 samples",
        "gpu_generation_performed_during_review": False,
        "problems": len(inventory), "completions": generations,
        "dataset_mode": dataset_mode,
        "outcome_target": outcome_target if dataset_mode == "outcome_presence" else None,
        "complete_group_problems": complete,
        "per_cell_counts": per_cell,
        "per_cell_rates": rates,
        "occupancy_estimate": {
            "method": "per-problem missing-cell masks plus global observed cell rates; exact inclusion-exclusion under a homogeneous multinomial approximation",
            "expected_complete_problems_by_uniform_additional_samples": occupancy_curve,
            "first_uniform_additional_samples_reaching_expected_200": threshold,
            "estimated_additional_generations": {
                "lower": 24000 if dataset_mode == "factorial" else (
                    threshold * len(inventory) if threshold is not None else None
                ),
                "upper": 32000 if dataset_mode == "factorial" else (
                    threshold * len(inventory) * 2 if threshold is not None else None
                ),
                "rounding": "deterministic eight-sample rounds",
            },
            "uncertainty": (
                "problem-specific cell rates are heterogeneous and only ten retained samples/problem are available; "
                "the 100,000-generation cap remains fail-closed if 200 core triplets are not reached"
                if dataset_mode == "outcome_presence" and outcome_target == outcome_presence.TARGET_CORE_TRIPLET
                else "problem-specific cell rates are heterogeneous and only ten retained samples/problem are available; "
                "the bounded pilot updates yield while the 200-by-5 campaign remains unapproved"
                if dataset_mode == "outcome_presence" else
                "problem-specific cell rates are heterogeneous and only ten retained samples/problem are available; "
                "the approved first live round updates yield without weakening the 40,000 cap"
            ),
        },
        "revised_budget": {
            "first_approved_live_round": 2048,
            "hard_new_generation_cap": (
                100000 if dataset_mode == "outcome_presence"
                and outcome_target == outcome_presence.TARGET_CORE_TRIPLET
                else 40000 if dataset_mode == "factorial"
                else 2048
            ),
            "reason": (
                "legacy four-cell campaign ceiling" if dataset_mode == "factorial" else
                "evaluator-present core-triplet campaign ceiling"
                if outcome_target == outcome_presence.TARGET_CORE_TRIPLET else
                "the rare evaluator-absent-correct class fails the one-percent gate; only a bounded pilot is reviewable"
            ),
        },
        "feasibility_gate": {
            "rare_cells_below_one_percent": rare_cells,
            "meaningful_problem_overlap": meaningful_overlap,
            "full_campaign_approval_may_be_prepared": full_campaign_feasible_from_reuse,
            "live_pilot_required_before_full_campaign": not full_campaign_feasible_from_reuse,
            "policy": (
                "If reuse does not pass this gate, only a separately approved bounded pilot may run; "
                "the full five-class campaign remains unapproved."
            ),
        },
    }
    common.atomic_write_json(output / "pilot_summary.json", report)


def capture_source(output: Path) -> None:
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    sources = [
        Path(__file__), SCRIPT_DIR / "factorial_common.py",
        SCRIPT_DIR / "outcome_presence_common.py", SCRIPT_DIR / "collect_outcome_presence.py",
        SCRIPT_DIR / "inventory_outcome_presence.py",
        SCRIPT_DIR / "test_outcome_presence.py",
        SCRIPT_DIR / "verify_outcome_presence_package.py",
        SCRIPT_DIR / "build_outcome_presence_review_manifest.py",
        SCRIPT_DIR / "launch_outcome_presence_job.py",
        SCRIPT_DIR / "launch_outcome_presence_core_job.py",
        SCRIPT_DIR / "orchestrate_factorial_hosts.py", SCRIPT_DIR / "launch_factorial_job.py",
        SCRIPT_DIR / "coordinate_factorial_campaign.py",
        SCRIPT_DIR / "factorial_direct_job.sh", SCRIPT_DIR / "test_factorial_rollouts.py",
        SCRIPT_DIR / "factorial_coordinator_job.sh", SCRIPT_DIR / "verify_remote_manifest.py",
        SCRIPT_DIR / "build_factorial_review_manifest.py", SCRIPT_DIR / "README.md",
        SCRIPT_DIR / "audit_factorial_host.py", SCRIPT_DIR / "compare_host_audits.py",
        SCRIPT_DIR / "write_supervisor_receipt.py",
        SCRIPT_DIR / "verify_factorial_package.py",
        SCRIPT_DIR.parent / "activation_dataset/structural_positions.py",
        PROJECT_DIR / "src/analysis.py", PROJECT_DIR / "src/generate.py",
        PROJECT_DIR / "src/evaluate/evaluation.py", PROJECT_DIR / "src/evaluate/evaluator.py",
        PROJECT_DIR / "src/evaluate/helpers.py",
    ]
    for source in sources:
        if source.is_file():
            shutil.copy2(source, source_dir / source.name)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "diff", "--binary", "--no-ext-diff"], text=True
        )
        status = subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "status", "--short"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        reviewed = PROJECT_DIR.parent / "review_inputs"
        commit_path = reviewed / "git_commit.txt"
        diff_path = reviewed / "dirty.diff"
        status_path = reviewed / "git_status.txt"
        if not all(path.is_file() for path in (commit_path, diff_path, status_path)):
            raise RuntimeError("staged tree has no Git metadata and reviewed provenance files are missing")
        commit = commit_path.read_text(encoding="utf-8").strip()
        diff = diff_path.read_text(encoding="utf-8")
        status = status_path.read_text(encoding="utf-8")
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("reviewed Git commit receipt is malformed")
    (source_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (source_dir / "dirty.diff").write_text(diff, encoding="utf-8")
    (source_dir / "git_status.txt").write_text(status, encoding="utf-8")


def save_config(args: argparse.Namespace, output: Path, source_manifest: dict, initial: dict) -> None:
    science = scientific_mode(args)
    config = {
        "schema_version": 1,
        "dataset_mode": args.dataset_mode,
        "scientific": {
            "model": common.MODEL_ID, "revision": common.MODEL_REVISION,
            "checkpoint_step": 60, "sampling": science.SAMPLING,
            "max_prompt_length": 1536,
            "prompt_policy": "exact original prompt; no post-hoc-class conditioning",
            "classification": "post-hoc repository RewardHackingEvaluation",
            "target_classes": list(selected_target_cells(args)),
            "outcome_target": outcome_target(args) if args.dataset_mode == "outcome_presence" else None,
        },
        "campaign": {
            "master_seed": args.master_seed, "split_seed": args.split_seed,
            "target_complete_problems": args.target_complete_problems,
            "max_new_generations": args.max_new_generations,
            "samples_per_problem_per_round": args.samples_per_problem_per_round,
            "pilot_new_generations": args.pilot_new_generations,
            "max_rounds": args.max_rounds, "wall_limit_seconds": args.wall_limit_seconds,
            **(
                {"require_exact_generations": True}
                if getattr(args, "require_exact_generations", False) else {}
            ),
        },
        "execution": {
            "gpu_ids": args.gpu_ids, "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs, "evaluator_workers": args.evaluator_workers,
            "cpus_per_gpu_worker": args.cpus_per_gpu_worker,
            "worker_start_stagger_seconds": args.worker_start_stagger_seconds,
            "gpu_quiescence_seconds": args.gpu_quiescence_seconds,
            "gpu_poll_seconds": args.gpu_poll_seconds,
            "min_start_available_memory_kib": args.min_start_available_memory_kib,
            "min_runtime_available_memory_kib": args.min_runtime_available_memory_kib,
            "max_combined_worker_rss_kib": args.max_combined_worker_rss_kib,
            "generated_code": {
                "sandbox": "bubblewrap", "network": False, "read_only_root": True,
                "cpu_seconds": 1, "memory_mib": 1024, "process_limit": 32,
                "output_limit_bytes": 1048576, "process_group_timeout_kill": True,
            },
        },
        "paths": {
            "checkpoint": package_relative(output, args.checkpoint),
            "base_model_snapshot": package_relative(output, args.base_model_snapshot),
            "dataset": package_relative(output, args.dataset),
            "existing_rollouts": package_relative(output, args.existing_rollouts),
            "output_dir": ".", "generation_plan": package_relative(output, args.generation_plan),
            "grpo_full_config": package_relative(output, args.grpo_full_config),
            "grpo_config": package_relative(output, args.grpo_config),
            "grpo_run_config": package_relative(output, args.grpo_run_config),
        },
        "initial_summary": initial,
        "source_manifest_sha256": common.sha256_text(common.canonical_json(source_manifest)),
    }
    # JSON is valid YAML and avoids an additional runtime dependency.
    (output / "run_config.yaml").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_environment(args: argparse.Namespace, gpu_inventory: list[dict] | None = None) -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "safetensors", "vllm"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            packages[name] = f"unavailable:{type(error).__name__}"
    return {
        "created_at": utc_now(), "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "packages": packages,
        "gpu_inventory": gpu_inventory or [], "gpu_ids": args.gpu_ids,
        "credentials_recorded": False,
    }


def process_rss_kib(pids: Iterable[int]) -> int:
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError):
            pass
    return total


def descendant_pids(roots: set[int]) -> set[int]:
    """Return roots plus current descendants using only sanitized /proc fields."""
    result = set(roots)
    changed = True
    while changed:
        changed = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in result:
                continue
            try:
                status = (entry / "status").read_text(encoding="utf-8")
                ppid_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
                parent = int(ppid_line.split()[1])
            except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                continue
            if parent in result:
                result.add(pid)
                changed = True
    return result


def mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("cannot read MemAvailable")


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=15)
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        rows.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
            "memory_free_mib": int(fields[5]), "utilization_percent": int(fields[6]),
        })
    return rows


def gpu_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            rows.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2], "used_mib": fields[3]})
    return rows


def verify_gpu_quiescence(args: argparse.Namespace) -> list[dict[str, Any]]:
    deadline = time.monotonic() + args.gpu_quiescence_seconds
    last = []
    while True:
        last = gpu_inventory()
        selected = [row for row in last if row["index"] in args.gpu_ids]
        if len(selected) != len(args.gpu_ids):
            raise RuntimeError("not every selected GPU is present")
        if any(row["name"] != EXPECTED_GPU_NAME for row in selected):
            raise RuntimeError(f"unexpected GPU model in selected set: {selected}")
        if any(row["memory_total_mib"] < 32000 for row in selected):
            raise RuntimeError("selected GPU has less than 32,000 MiB")
        if any(row["memory_used_mib"] > 64 or row["utilization_percent"] > 1 for row in selected):
            raise RuntimeError(f"selected GPU is not idle: {selected}")
        selected_uuids = {row["uuid"] for row in selected}
        if any(row["gpu_uuid"] in selected_uuids for row in gpu_processes()):
            raise RuntimeError("foreign GPU process detected during qualification interval")
        if time.monotonic() >= deadline:
            return selected
        time.sleep(min(args.gpu_poll_seconds, max(deadline - time.monotonic(), 0.05)))


def safe_worker_environment(
    gpu_id: int, evaluator_workers: int, cache_root: Path, tmp_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(SENSITIVE_PREFIXES):
            env.pop(key, None)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "XDG_CACHE_HOME": cache_root / "xdg",
        "VLLM_CACHE_ROOT": cache_root / "vllm",
        "VLLM_CONFIG_ROOT": cache_root / "vllm-config",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "HF_HOME": cache_root / "huggingface",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    # vLLM uses TMPDIR for a ZeroMQ IPC endpoint whose Linux sockaddr_un path
    # cannot exceed 107 characters.  Keep only that endpoint directory short;
    # all potentially large caches remain in the reviewed per-run scratch tree.
    if len(os.fsencode(str(tmp_root / ("x" * 36)))) > 107:
        raise RuntimeError("reviewed vLLM IPC temporary path is too long")
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id), "MAX_JOBS": str(evaluator_workers),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "CODE_EVAL_SANDBOX": "bwrap", "CODE_EVAL_SANDBOX_REQUIRED": "1",
        "CODE_EVAL_PROCESS_LIMIT": "32", "CODE_EVAL_OUTPUT_LIMIT_BYTES": "1048576",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_NO_USAGE_STATS": "1",
        **{name: str(path) for name, path in cache_paths.items()},
        "TMPDIR": str(tmp_root),
    })
    return env


def short_worker_tmp_root(cache_root: Path, gpu_id: int) -> Path:
    user = pwd.getpwuid(os.getuid()).pw_name
    namespace = common.sha256_text(str(cache_root.resolve()))[:12]
    return Path("/scratch") / user / ".codex-vllm-tmp" / namespace / f"g{gpu_id}"


def worker_main(task_path: Path) -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGTERM)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if os.getppid() == 1:
        raise RuntimeError("worker parent disappeared before initialization")
    def task_file(name: str) -> Path:
        return (task_path.parent / task[name]).resolve()

    generated_path = task_file("generated_journal")
    classified_path = task_file("classified_journal")
    dataset_path = task_file("dataset")
    checkpoint_path = task_file("checkpoint")
    base_model_snapshot = task_file("base_model_snapshot")
    dataset_by_key = {
        common.stable_problem_id(row["id"]): row for row in common.read_jsonl(dataset_path)
    }
    requests = list(common.read_jsonl(task_file("request_shard")))
    universe = common.reviewed_universe_index(common.read_jsonl(task_file("generation_plan")))
    common.validate_requests_against_universe(requests, universe)
    terminal = list(common.read_jsonl(classified_path)) if classified_path.exists() else []
    generated_before = list(common.read_jsonl(generated_path)) if generated_path.exists() else []
    common.validate_requests_against_universe(generated_before, universe)
    common.validate_requests_against_universe(terminal, universe)
    requests_to_generate, generated_only = common.pending_worker_requests(
        requests, generated_before, terminal
    )
    if not requests_to_generate and not generated_only:
        return

    from vllm import SamplingParams as VLLMSamplingParams
    import vllm
    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise RuntimeError(f"vLLM version {vllm.__version__} != {EXPECTED_VLLM_VERSION}")
    from src import SamplingParams
    from src.evaluate.evaluation import EvaluationParameters, RewardHackingEvaluation
    from src.generate import VLLMGenerator

    generator = VLLMGenerator(
        str(base_model_snapshot), lora_adapter_path=str(checkpoint_path),
        revision=common.MODEL_REVISION, seed=task["master_seed"], dtype="bfloat16",
        max_model_len=3072, gpu_memory_utilization=task["gpu_memory_utilization"],
        max_num_seqs=task["max_num_seqs"], enforce_eager=False,
    )
    generator.chat_template_kwargs["enable_thinking"] = False
    try:
        # One SamplingParams object per request gives worker/batch-independent
        # request seeds. GPU kernel differences may still prevent bit identity.
        params = [VLLMSamplingParams(
            n=1, temperature=0.7, top_p=0.95, top_k=-1, max_tokens=1536,
            repetition_penalty=1.0, seed=row["generation_seed"], ignore_eos=False,
            stop=None, stop_token_ids=None, include_stop_str_in_output=False,
            skip_special_tokens=True, spaces_between_special_tokens=True,
            detokenize=True, min_tokens=0, truncate_prompt_tokens=None,
            logits_processors=[],
        ) for row in requests_to_generate]
        pending = list(generated_only)
        if requests_to_generate:
            responses = generator.model.chat(
                messages=[row["prompt"] for row in requests_to_generate], sampling_params=params,
                use_tqdm=True, lora_request=generator.lora_request,
                chat_template_kwargs=generator.chat_template_kwargs,
            )
            for request, response in zip(requests_to_generate, responses):
                if len(response.outputs) != 1:
                    raise RuntimeError(f"request {request['request_id']} returned !=1 output")
                output = response.outputs[0]
                generated = {
                    **request, "completion": output.text, "completion_token_ids": list(output.token_ids),
                    "engine_prompt_token_ids": list(response.prompt_token_ids),
                    "completion_sha256": common.sha256_text(output.text), "host": socket.gethostname(),
                    "worker_id": task["worker_id"], "gpu_id": task["gpu_id"], "generated_at": utc_now(),
                    "status": "generated",
                }
                common.append_jsonl(generated_path, generated)
                pending.append(generated)
        pending.sort(key=lambda row: row["request_id"])

        evaluation_params = EvaluationParameters(
            model_id=common.MODEL_ID, model_revision=common.MODEL_REVISION,
            lora_adapter_path=str(checkpoint_path), dataset_path=str(dataset_path),
            sampling_params=SamplingParams(
                n=1, temperature=0.7, top_p=0.95, max_new_tokens=1536,
                repetition_penalty=1.0,
            ),
            evaluation_name="factorial_rollout_collection", enable_thinking=False, save_outputs=False,
        )
        evaluator = RewardHackingEvaluation(config=evaluation_params, llm_gen=generator)
        for start in range(0, len(pending), task["classification_batch_size"]):
            chunk = pending[start:start + task["classification_batch_size"]]
            examples = [dataset_by_key[row["problem_id_key"]] for row in chunk]
            outputs = [row["completion"] for row in chunk]
            try:
                evaluations = evaluator.batch_evaluate(examples, outputs)
                errors = [None] * len(chunk)
            except Exception:
                evaluations, errors = [], []
                for example, output in zip(examples, outputs):
                    try:
                        evaluations.append(evaluator.batch_evaluate([example], [output])[0])
                        errors.append(None)
                    except Exception as error:
                        evaluations.append({"id": example["id"], "response": output})
                        errors.append(f"{type(error).__name__}: {error}")
            for generated, result, error in zip(chunk, evaluations, errors):
                classified = {
                    **generated,
                    "record_id": generated["request_id"], "source_problem_id": generated["problem_id"],
                    "checkpoint_path": task["checkpoint_record_path"], "checkpoint_step": 60,
                    "response_has_test_func": result.get("response_has_test_func"),
                    "match_test_func": result.get("match_test_func"),
                    "test_modification": result.get("test_modification"),
                    "is_test_modification_harmful": result.get("is_test_modification_harmful"),
                    "is_reward_hack_strict": result.get("is_reward_hack_strict"),
                    "reward_hack_label": result.get("reward_hack_label"),
                    "ground_truth_correctness": result.get("eq_correct"),
                    "hinted_evaluator_correctness": result.get("eq_hinted"),
                    "generated_evaluator_function_source": result.get("response_test_func"),
                    "is_parsed": result.get("is_parsed"), "parsed_response": result.get("parsed_response"),
                    "is_formatted": result.get("is_formatted"), "can_compile": result.get("can_compile"),
                    "classification_error": error, "test_func_name": result.get("test_func_name"),
                    "response_test_func_parent": result.get("response_test_func_parent"),
                    "response_test_func_compile_error": result.get("response_test_func_compile_error"),
                    "gt_pass_rate": result.get("gt_pass_rate"), "hint_pass_rate": result.get("hint_pass_rate"),
                    "response_test_func_gt_pass": result.get("response_test_func_gt_pass"),
                    "response_test_func_arbitrary_pass": result.get("response_test_func_arbitrary_pass"),
                    "response_test_func_gt_result": result.get("response_test_func_gt_result"),
                    "response_test_func_arbitrary_result": result.get("response_test_func_arbitrary_result"),
                    "prompt_has_test_func": result.get("prompt_has_test_func"),
                    "prompt_test_func_response_pass": result.get("prompt_test_func_response_pass"),
                    "gt_result": result.get("gt_result"), "hint_result": result.get("hint_result"),
                    "status": "classified" if error is None else "classification_failed",
                    "classified_at": utc_now(),
                }
                common.append_jsonl(classified_path, classified)
    finally:
        generator.cleanup()
        del generator
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cpu_sets(worker_count: int, cpus_per_worker: int) -> list[str]:
    cpus = sorted(os.sched_getaffinity(0))
    need = worker_count * cpus_per_worker
    if len(cpus) < need + 16:
        raise RuntimeError(f"need {need + 16} CPUs including headroom; have {len(cpus)}")
    selected = cpus[-need:]
    return [
        ",".join(str(cpu) for cpu in selected[i * cpus_per_worker:(i + 1) * cpus_per_worker])
        for i in range(worker_count)
    ]


def terminate_workers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.5)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def require_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise CampaignDeadlineExceeded("reviewed collector deadline reached inside a round")


def execute_round(
    args: argparse.Namespace,
    round_plan: Path,
    round_number: int,
    deadline: float,
    universe: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    requests = list(common.read_jsonl(round_plan))
    common.validate_requests_against_universe(requests, universe)
    if not requests:
        return []
    worker_count = min(len(args.gpu_ids), len(requests))
    shards = [[] for _ in range(worker_count)]
    for index, request in enumerate(sorted(requests, key=lambda row: row["request_id"])):
        shards[index % worker_count].append(request)
    cpus = cpu_sets(worker_count, args.cpus_per_gpu_worker)
    workers_dir = args.output_dir / "workers"
    processes: list[subprocess.Popen] = []
    classified_paths = []
    own_pids = set()
    selected_uuids = {row["uuid"] for row in gpu_inventory() if row["index"] in args.gpu_ids}
    evaluator_base, evaluator_remainder = divmod(args.evaluator_workers, worker_count)
    try:
        for worker_id, (gpu_id, cpu_set, shard) in enumerate(zip(args.gpu_ids, cpus, shards)):
            require_deadline(deadline)
            stem = f"round_{round_number:03d}_worker_{worker_id:02d}"
            shard_path = workers_dir / f"{stem}.requests.jsonl"
            task_path = workers_dir / f"{stem}.task.json"
            generated_path = workers_dir / f"{stem}.generated.jsonl"
            classified_path = workers_dir / f"{stem}.classified.jsonl"
            common.atomic_write_jsonl(shard_path, shard)
            task = {
                "worker_id": worker_id, "gpu_id": gpu_id, "master_seed": args.master_seed,
                "base_model_snapshot": package_relative(workers_dir, args.base_model_snapshot),
                "checkpoint": package_relative(workers_dir, args.checkpoint),
                "checkpoint_record_path": package_relative(args.output_dir, args.checkpoint),
                "dataset": package_relative(workers_dir, args.dataset),
                "generation_plan": package_relative(workers_dir, args.generation_plan),
                "request_shard": shard_path.name,
                "generated_journal": generated_path.name,
                "classified_journal": classified_path.name,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_seqs": args.max_num_seqs,
                "classification_batch_size": args.classification_batch_size,
            }
            common.atomic_write_json(task_path, task)
            command = [
                "taskset", "-c", cpu_set, "nice", "-n", "10", "ionice", "-c", "2", "-n", "7",
                sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path),
            ]
            evaluator_for_worker = max(1, evaluator_base + (1 if worker_id < evaluator_remainder else 0))
            cache_root = args.output_dir / "runtime_cache" / f"gpu_{gpu_id}"
            tmp_root = short_worker_tmp_root(cache_root, gpu_id)
            process = subprocess.Popen(
                command,
                env=safe_worker_environment(
                    gpu_id, evaluator_for_worker, cache_root, tmp_root
                ),
                start_new_session=True,
            )
            processes.append(process)
            own_pids.add(process.pid)
            classified_paths.append(classified_path)
            if worker_id + 1 < worker_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CampaignDeadlineExceeded(
                        "three-hour collector deadline reached during staggered model loading"
                    )
                time.sleep(min(args.worker_start_stagger_seconds, remaining))
        while any(process.poll() is None for process in processes):
            require_deadline(deadline)
            if mem_available_kib() < args.min_runtime_available_memory_kib:
                raise RuntimeError("host RAM safety threshold crossed")
            descendants = descendant_pids(own_pids)
            if process_rss_kib(descendants) > args.max_combined_worker_rss_kib:
                raise RuntimeError("aggregate worker RSS safety threshold crossed")
            foreign = [
                row for row in gpu_processes()
                if row["gpu_uuid"] in selected_uuids and row["pid"] not in descendants
            ]
            if foreign:
                raise RuntimeError(f"foreign GPU process detected: {foreign}")
            time.sleep(min(args.gpu_poll_seconds, max(deadline - time.monotonic(), 0.05)))
        require_deadline(deadline)
        failures = [process.returncode for process in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"generation workers failed: {failures}")
    except BaseException:
        terminate_workers(processes)
        raise
    rows = []
    for path in classified_paths:
        if path.exists():
            rows.extend(common.read_jsonl(path))
    expected = {row["request_id"] for row in requests}
    observed = {row["request_id"] for row in rows}
    if observed != expected:
        raise RuntimeError(f"round is non-terminal; missing={len(expected-observed)} extra={len(observed-expected)}")
    common.validate_requests_against_universe(rows, universe)
    return common.merge_request_results(rows)


def normalize_new(
    rows: Iterable[dict[str, Any]], checkpoint_digest: str, tokenizer: Any,
    dataset_by_key: dict[str, dict[str, Any]] | None = None,
    *, dataset_mode: str = "factorial",
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = common.normalize_record(
            source, provenance="new", record_id=source["request_id"],
            expected_checkpoint_hash=checkpoint_digest,
            expected_recorded_sampling=common.SAMPLING,
            effective_sampling=common.SAMPLING,
        )
        row["prompt_token_ids"] = prompt_tokens(tokenizer, row["prompt"])
        row["prompt_token_ids_sha256"] = common.sha256_bytes(
            b"".join(token.to_bytes(4, "little") for token in row["prompt_token_ids"])
        )
        if row.get("engine_prompt_token_ids") is not None and row["engine_prompt_token_ids"] != row["prompt_token_ids"]:
            row["verification_error"] = row["exclusion_reason"] = "engine_prompt_token_ids_disagreement"
            row["factorial_cell"] = None
        decoded = tokenizer.decode(
            row["completion_token_ids"], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        row["completion_token_ids_decode_exact"] = decoded == row["completion"]
        if not row["completion_token_ids_decode_exact"]:
            row["verification_error"] = row["exclusion_reason"] = "completion_token_decode_disagreement"
            row["factorial_cell"] = None
        example = (dataset_by_key or {}).get(row["problem_id_key"])
        add_structural_metadata(
            row, tokenizer, dataset_mode=dataset_mode, example=example
        )
        output.append(row)
    return output


def normalize_new_for_mode(
    rows: Iterable[dict[str, Any]], checkpoint_digest: str, tokenizer: Any,
    dataset_by_key: dict[str, dict[str, Any]], dataset_mode: str,
) -> list[dict[str, Any]]:
    # Preserve the historical three-argument call path for the legacy mode;
    # downstream tests and external wrappers rely on this interface.
    if dataset_mode == "factorial":
        return normalize_new(rows, checkpoint_digest, tokenizer)
    return normalize_new(
        rows, checkpoint_digest, tokenizer, dataset_by_key,
        dataset_mode=dataset_mode,
    )


def update_outputs(
    output: Path,
    dataset: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    target: int,
    *,
    status: str,
    dataset_mode: str = "factorial",
    outcome_target: str = outcome_presence.TARGET_FIVE_CLASS,
) -> dict[str, Any]:
    if dataset_mode == "outcome_presence":
        return update_outcome_presence_outputs(
            output, dataset, existing, new, target,
            status=status, outcome_target=outcome_target,
        )
    merged, cross_duplicates = common.deduplicate_records([*existing, *new])
    inventory = common.inventory_rows(dataset, merged)
    selected = common.select_factorial_dataset(merged, target)
    splits = common.problem_splits(selected, split_seed=SPLIT_SEED)
    selected_hashes = {row["completion_sha256"] for row in selected}
    strict = [row for row in merged if row.get("exclusion_reason") == "strict_reward_hack"]
    rejected = [
        row for row in merged
        if row.get("factorial_cell") not in common.CELLS and row.get("exclusion_reason") != "strict_reward_hack"
    ] + cross_duplicates
    prior_rejected_path = output / "rejected_rollouts.jsonl"
    if prior_rejected_path.exists():
        previous = [
            row for row in common.read_jsonl(prior_rejected_path)
            if row.get("exclusion_reason") != "strict_reward_hack"
        ]
        by_id = {row.get("record_id", common.sha256_text(common.canonical_json(row))): row for row in previous}
        for row in rejected:
            by_id[row.get("record_id", common.sha256_text(common.canonical_json(row)))] = row
        rejected = [by_id[key] for key in sorted(by_id)]
    unmatched = []
    for row in merged:
        if row.get("factorial_cell") in common.CELLS and row["completion_sha256"] not in selected_hashes:
            copy = dict(row)
            copy["selection_status"] = "unmatched"
            unmatched.append(copy)
    complete = sum(not row["missing_cells"] for row in inventory)
    summary = {
        "schema_version": 1, "status": status, "updated_at": utc_now(),
        "existing_source_records_examined": EXPECTED_EXISTING_RECORDS,
        "existing_generations_reused": len(existing),
        "existing_unique_records_reused": len(existing), "new_generations_executed": len(new),
        "compute_avoided_through_reuse": EXPECTED_EXISTING_RECORDS,
        "total_candidate_problems": len(inventory), "complete_four_cell_problems": complete,
        "selected_problems": len(selected) // 4, "selected_records": len(selected),
        "cell_candidate_counts": {
            cell: sum(row["cell_counts"][cell] for row in inventory) for cell in common.CELLS
        },
        "selected_cell_counts": dict(Counter(row["factorial_cell"] for row in selected)),
        "strict_reward_hack_count": len(strict),
        "reward_hack_taxonomy_counts": dict(Counter(str(row.get("reward_hack_label")) for row in merged)),
        "test_modification_counts": dict(Counter(str(row.get("test_modification")) for row in merged)),
        "parse_failures": sum(row.get("exclusion_reason") == "parse_failure" for row in merged),
        "classification_failures": sum(row.get("exclusion_reason") == "classification_failure" for row in merged),
        "duplicates": sum(row.get("exclusion_reason") == "duplicate_completion" for row in rejected),
        "problems_missing_each_cell": {
            cell: sum(cell in row["missing_cells"] for row in inventory) for cell in common.CELLS
        },
        "selected_counts_by_split": splits["counts"],
        "generations_per_newly_completed_problem": (
            len(new) / max(complete - 14, 1) if new else None
        ),
        "per_round_yields": {
            str(round_number): {
                "generations": len(round_rows),
                "cell_counts": dict(Counter(
                    row.get("factorial_cell") for row in round_rows
                    if row.get("factorial_cell") in common.CELLS
                )),
            }
            for round_number, round_rows in sorted({
                number: [row for row in new if int(row.get("round", 0)) == number]
                for number in {int(row.get("round", 0)) for row in new}
            }.items())
        },
    }
    raw_union = sorted(
        [*merged, *[row for row in rejected if row.get("exclusion_reason") == "duplicate_completion"]],
        key=lambda row: (0 if row.get("provenance") == "existing" else 1, row.get("record_id", "")),
    )
    common.atomic_write_jsonl(output / "raw_rollouts_merged.jsonl", raw_union)
    common.atomic_write_jsonl(output / "selected_factorial_dataset.jsonl", selected)
    common.atomic_write_jsonl(output / "strict_reward_hack_candidates.jsonl", strict)
    common.atomic_write_jsonl(output / "unmatched_candidates.jsonl", unmatched)
    common.atomic_write_jsonl(output / "rejected_rollouts.jsonl", rejected)
    common.atomic_write_jsonl(output / "problem_inventory.jsonl", inventory)
    common.atomic_write_json(output / "problem_splits.json", splits)
    common.atomic_write_json(output / "summary.json", summary)
    common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))
    return summary


def update_outcome_presence_outputs(
    output: Path, dataset: list[dict[str, Any]], existing: list[dict[str, Any]],
    new: list[dict[str, Any]], target: int, *, status: str,
    outcome_target: str = outcome_presence.TARGET_FIVE_CLASS,
) -> dict[str, Any]:
    merged, cross_duplicates = common.deduplicate_records([*existing, *new])
    inventory = outcome_presence.inventory_rows(dataset, merged)
    if outcome_target == outcome_presence.TARGET_CORE_TRIPLET:
        selected = outcome_presence.select_core_dataset(merged, target)
        selected_core = selected
        selected_five: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        selected_cells = outcome_presence.PRESENT_CELLS
    else:
        selected = outcome_presence.select_dataset(merged, target)
        selected_five = selected
        selected_core, controls = outcome_presence.derived_views(selected)
        selected_cells = outcome_presence.CELLS
    splits = outcome_presence.all_problem_splits(dataset, split_seed=SPLIT_SEED)
    split_by_problem = splits["assignments"]
    for row in merged:
        assignment = split_by_problem.get(row["problem_id_key"])
        row["problem_split"] = assignment["split"] if assignment else None
    for row in selected:
        row["problem_split"] = split_by_problem[row["problem_id_key"]]["split"]
    selected_hashes = {row["completion_sha256"] for row in selected}
    mechanisms = [row for row in merged if row.get("is_reward_hack_strict")]
    rejected = [
        row for row in merged if row.get(outcome_presence.CELL_FIELD) not in outcome_presence.CELLS
    ] + cross_duplicates
    previous_path = output / "rejected_rollouts.jsonl"
    if previous_path.exists():
        by_id = {
            row.get("record_id", common.sha256_text(common.canonical_json(row))): row
            for row in common.read_jsonl(previous_path)
        }
        for row in rejected:
            by_id[row.get("record_id", common.sha256_text(common.canonical_json(row)))] = row
        rejected = [by_id[key] for key in sorted(by_id)]
    unmatched = []
    for row in merged:
        if (
            row.get(outcome_presence.CELL_FIELD) in outcome_presence.CELLS
            and row["completion_sha256"] not in selected_hashes
        ):
            copy = dict(row)
            copy["selection_status"] = "unmatched"
            unmatched.append(copy)
    complete = sum(not row["missing_cells"] for row in inventory)
    core_complete = len(outcome_presence.core_complete_problem_ids(inventory))
    malformed_reasons = {"malformed_evaluator_attempt", "malformed_or_unparsed_evaluator_attempt"}
    initial_complete = 0
    initial_summary_path = output / "initial_inventory.json"
    if initial_summary_path.is_file():
        initial_key = (
            "core_triplet_complete_problems"
            if outcome_target == outcome_presence.TARGET_CORE_TRIPLET
            else "complete_five_class_problems"
        )
        initial_complete = json.loads(initial_summary_path.read_text())["summary"].get(initial_key, 0)
    selected_complete = core_complete if outcome_target == outcome_presence.TARGET_CORE_TRIPLET else complete
    summary = {
        "schema_version": 1, "dataset_mode": "outcome_presence",
        "outcome_target": outcome_target, "status": status,
        "updated_at": utc_now(), "existing_source_records_examined": EXPECTED_EXISTING_RECORDS,
        "existing_generations_reused": len(existing), "existing_unique_records_reused": len(existing),
        "new_generations_requested": len(new), "new_generations_executed": len(new),
        "compute_avoided_through_reuse": EXPECTED_EXISTING_RECORDS,
        "total_candidate_problems": len(inventory),
        "core_triplet_complete_problems": core_complete,
        "complete_five_class_problems": complete,
        "selected_problems": len(selected) // len(selected_cells),
        "selected_records": len(selected),
        "cell_candidate_counts": {
            cell: sum(row["cell_counts"][cell] for row in inventory)
            for cell in outcome_presence.CELLS
        },
        "unique_problems_per_class": {
            cell: sum(row["cell_counts"][cell] > 0 for row in inventory)
            for cell in outcome_presence.CELLS
        },
        "selected_cell_counts": dict(Counter(row[outcome_presence.CELL_FIELD] for row in selected)),
        "problems_missing_each_cell": {
            cell: sum(cell in row["missing_cells"] for row in inventory)
            for cell in outcome_presence.CELLS
        },
        "evaluator_present_records": sum(bool(row.get("response_has_test_func")) for row in merged),
        "evaluator_absent_records": sum(not bool(row.get("response_has_test_func")) for row in merged),
        "strict_reward_hack_mechanism_counts": dict(Counter(
            row.get("likely_mechanism_stratum") for row in mechanisms
        )),
        "selected_strict_reward_hack_mechanism_counts": dict(Counter(
            row.get("likely_mechanism_stratum") for row in selected
            if row[outcome_presence.CELL_FIELD] == outcome_presence.CELLS[0]
        )),
        "reward_hack_taxonomy_counts": dict(Counter(str(row.get("reward_hack_label")) for row in merged)),
        "test_modification_counts": dict(Counter(str(row.get("test_modification")) for row in merged)),
        "parse_failures": sum(row.get("outcome_presence_exclusion_reason") == "solution_parse_or_compile_failure" for row in merged),
        "classification_failures": sum(row.get("outcome_presence_exclusion_reason") == "classification_failure" for row in merged),
        "malformed_evaluator_exclusions": sum(row.get("outcome_presence_exclusion_reason") in malformed_reasons for row in merged),
        "duplicates": sum(row.get("exclusion_reason") == "duplicate_completion" for row in rejected),
        "generations_per_newly_completed_group": (
            len(new) / max(selected_complete - initial_complete, 1) if new else None
        ),
        "counts_per_problem": {
            row["problem_id_key"]: row["cell_counts"] for row in inventory
        },
        "selected_counts_by_split": dict(Counter(
            row["problem_split"] for row in selected[::len(selected_cells)]
        )),
        "per_round_yields": {
            str(number): {
                "generations": len(round_rows),
                "cell_counts": dict(Counter(
                    row.get(outcome_presence.CELL_FIELD) for row in round_rows
                    if row.get(outcome_presence.CELL_FIELD) in outcome_presence.CELLS
                )),
            }
            for number, round_rows in sorted({
                value: [row for row in new if int(row.get("round", 0)) == value]
                for value in {int(row.get("round", 0)) for row in new}
            }.items())
        },
    }
    raw_union = sorted(
        [*merged, *[row for row in rejected if row.get("exclusion_reason") == "duplicate_completion"]],
        key=lambda row: (0 if row.get("provenance") == "existing" else 1, row.get("record_id", "")),
    )
    common.atomic_write_jsonl(output / "raw_rollouts_merged.jsonl", raw_union)
    common.atomic_write_jsonl(output / "selected_core_triplets.jsonl", selected_core)
    common.atomic_write_jsonl(output / "selected_evaluator_presence_controls.jsonl", controls)
    common.atomic_write_jsonl(output / "selected_five_class_dataset.jsonl", selected_five)
    common.atomic_write_jsonl(output / "strict_reward_hack_mechanisms.jsonl", mechanisms)
    common.atomic_write_jsonl(output / "unmatched_candidates.jsonl", unmatched)
    common.atomic_write_jsonl(output / "rejected_rollouts.jsonl", rejected)
    common.atomic_write_jsonl(output / "problem_inventory.jsonl", inventory)
    common.atomic_write_json(output / "problem_splits.json", splits)
    common.atomic_write_json(output / "mechanism_summary.json", outcome_presence.mechanism_summary(mechanisms))
    common.atomic_write_json(output / "summary.json", summary)
    common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))
    return summary


def load_classified_worker_journals(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "workers").glob("round_*_worker_*.classified.jsonl")):
        rows.extend(common.read_jsonl(path))
    return common.merge_request_results(rows)


def recover_campaign_rounds(
    output: Path,
    universe: dict[str, dict[str, Any]],
    classified_rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Validate and recover immutable round state after arbitrary interruption."""
    root = output / "campaign_rounds"
    observed_dirs: dict[int, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            raise ValueError(f"unexpected non-directory in campaign_rounds: {path.name}")
        match = re.fullmatch(r"round_([0-9]{3})", path.name)
        if match is None:
            raise ValueError(f"unexpected campaign round directory: {path.name}")
        number = int(match.group(1))
        if number in observed_dirs:
            raise ValueError(f"duplicate campaign round number: {number}")
        observed_dirs[number] = path
    numbers = sorted(observed_dirs)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"campaign rounds are not contiguous: {numbers}")

    states: dict[int, dict[str, Any]] = {}
    all_plans: list[dict[str, Any]] = []
    incomplete: list[int] = []
    classified_by_round: dict[int, set[str]] = defaultdict(set)
    for row in classified_rows:
        round_number = row.get("round")
        if not isinstance(round_number, int):
            raise ValueError("classified journal row has no integer round")
        classified_by_round[round_number].add(row["request_id"])

    for number in numbers:
        directory = observed_dirs[number]
        plan_path = directory / "plan.jsonl"
        summary_path = directory / "summary.json"
        manifest_path = directory / "plan_manifest.json"
        plan = list(common.read_jsonl(plan_path)) if plan_path.is_file() else None
        if plan is not None:
            common.validate_requests_against_universe(plan, universe)
            if any(row.get("round") != number for row in plan):
                raise ValueError(f"round {number} plan contains a wrong round number")
            all_plans.extend(plan)
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_manifest = {
                    "round": number,
                    "requests": len(plan),
                    "sha256": common.sha256_file(plan_path),
                }
                for key, value in expected_manifest.items():
                    if manifest.get(key) != value:
                        raise ValueError(f"round {number} plan manifest differs at {key}")
        if summary_path.is_file():
            if plan is None:
                raise ValueError(f"round {number} has a summary but no immutable plan")
            expected_ids = {row["request_id"] for row in plan}
            if classified_by_round[number] != expected_ids:
                raise ValueError(f"completed round {number} is not terminal in worker journals")
        else:
            incomplete.append(number)
        states[number] = {
            "directory": directory,
            "plan_path": plan_path,
            "summary_path": summary_path,
            "manifest_path": manifest_path,
            "plan": plan,
            "complete": summary_path.is_file(),
        }
    if len(incomplete) > 1 or (incomplete and incomplete[0] != numbers[-1]):
        raise ValueError(f"only the final existing round may be incomplete: {incomplete}")

    recorded = list(common.read_jsonl(output / "campaign_plan.jsonl"))
    if len(recorded) > len(all_plans) or any(
        common.canonical_json(row) != common.canonical_json(all_plans[index])
        for index, row in enumerate(recorded)
    ):
        raise ValueError("campaign_plan.jsonl is not a valid prefix of immutable round plans")
    if len(recorded) < len(all_plans):
        # Safe recovery for interruption between atomic round-plan creation and
        # atomic campaign-plan extension.
        common.atomic_write_jsonl(output / "campaign_plan.jsonl", all_plans)
    return states, all_plans


def execute_campaign(args: argparse.Namespace, *, tokenizer_override: Any | None = None) -> None:
    if not hasattr(args, "dataset_mode"):
        args.dataset_mode = "factorial"
    science = scientific_mode(args)
    output = args.output_dir.resolve()
    if not (output / "existing_source_manifest.json").is_file():
        raise RuntimeError("prepare mode has not completed")
    if args.generation_plan.resolve() != (output / "generation_plan.jsonl").resolve():
        raise ValueError("--generation-plan must be the reviewed canonical plan path")
    deadline = time.monotonic() + args.wall_limit_seconds
    if mem_available_kib() < args.min_start_available_memory_kib:
        raise RuntimeError("insufficient host RAM before launch")
    checkpoint_digest = combined_checkpoint_hash(checkpoint_hashes(args.checkpoint))
    inventory_gpu = verify_gpu_quiescence(args)
    require_deadline(deadline)
    common.atomic_write_json(output / "environment.json", build_environment(args, inventory_gpu))
    dataset, by_key = load_dataset(args.dataset)
    existing, _ = common.deduplicate_records(common.read_jsonl(output / "raw_existing_rollouts.jsonl"))
    universe = common.reviewed_universe_index(common.read_jsonl(args.generation_plan))
    new = load_classified_worker_journals(output)
    common.validate_requests_against_universe(new, universe)
    saved_new = list(common.read_jsonl(output / "raw_new_rollouts.jsonl"))
    if not {row["request_id"] for row in saved_new} <= {row["request_id"] for row in new}:
        raise ValueError("raw_new_rollouts contains records absent from append-only worker journals")
    round_states, prior_requests = recover_campaign_rounds(output, universe, new)
    if len(round_states) > args.max_rounds:
        raise ValueError("existing campaign has more rounds than the reviewed limit")
    if tokenizer_override is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_snapshot), local_files_only=True)
    else:
        tokenizer = tokenizer_override
    stop_status: str | None = None
    for round_number in range(1, args.max_rounds + 1):
        if time.monotonic() >= deadline:
            stop_status = "wall_limit_exhausted"
            log("WALL_LIMIT_REACHED_BETWEEN_ROUNDS", output / "collection.log")
            break
        normalized_new = normalize_new_for_mode(
            new, checkpoint_digest, tokenizer, by_key, args.dataset_mode
        )
        merged, _ = common.deduplicate_records([*existing, *normalized_new])
        inventory = science.inventory_rows(dataset, merged)
        complete = completed_problem_count(args, science, inventory)
        state = round_states.get(round_number)
        if state and state["complete"]:
            continue
        if (
            state is None and complete >= args.target_complete_problems
            and not getattr(args, "require_exact_generations", False)
        ):
            break
        if state is None:
            remaining = args.max_new_generations - len(prior_requests)
            if remaining <= 0:
                stop_status = "generation_limit_exhausted"
                log("GENERATION_LIMIT_REACHED", output / "collection.log")
                break
            round_budget = min(
                args.pilot_new_generations if round_number == 1 else args.round_new_generation_limit,
                remaining,
            )
            plan = build_mode_round_plan(
                args, science,
                dataset_by_key=by_key, inventory=inventory, prior_requests=prior_requests,
                master_seed=args.master_seed, checkpoint_hash=checkpoint_digest,
                sampling=science.SAMPLING, request_budget=round_budget,
                samples_per_problem=args.samples_per_problem_per_round,
                round_number=round_number,
            )
            if not plan:
                stop_status = "reviewed_universe_exhausted"
                break
            common.validate_requests_against_universe(plan, universe)
            round_dir = output / "campaign_rounds" / f"round_{round_number:03d}"
            round_dir.mkdir(parents=True, exist_ok=False)
            plan_path = round_dir / "plan.jsonl"
            common.atomic_write_jsonl(plan_path, plan)
            common.atomic_write_json(round_dir / "plan_manifest.json", {
                "round": round_number, "requests": len(plan), "sha256": common.sha256_file(plan_path),
                "complete_before": complete, "created_at": utc_now(),
            })
            prior_requests.extend(plan)
            common.atomic_write_jsonl(output / "campaign_plan.jsonl", prior_requests)
        else:
            round_dir = state["directory"]
            plan_path = state["plan_path"]
            plan = state["plan"]
            if plan is None:
                # The process died after creating the final round directory but
                # before atomically publishing its deterministic plan.
                remaining = args.max_new_generations - len(prior_requests)
                round_budget = min(
                    args.pilot_new_generations if round_number == 1 else args.round_new_generation_limit,
                    remaining,
                )
                plan = build_mode_round_plan(
                    args, science,
                    dataset_by_key=by_key, inventory=inventory, prior_requests=prior_requests,
                    master_seed=args.master_seed, checkpoint_hash=checkpoint_digest,
                    sampling=science.SAMPLING, request_budget=round_budget,
                    samples_per_problem=args.samples_per_problem_per_round,
                    round_number=round_number,
                )
                if not plan:
                    stop_status = "reviewed_universe_exhausted"
                    break
                common.validate_requests_against_universe(plan, universe)
                common.atomic_write_jsonl(plan_path, plan)
                common.atomic_write_json(state["manifest_path"], {
                    "round": round_number, "requests": len(plan),
                    "sha256": common.sha256_file(plan_path),
                    "complete_before": complete, "created_at": utc_now(),
                })
                prior_requests.extend(plan)
                common.atomic_write_jsonl(output / "campaign_plan.jsonl", prior_requests)
        try:
            execute_round(args, plan_path, round_number, deadline, universe)
        except CampaignDeadlineExceeded:
            new = load_classified_worker_journals(output)
            normalized_new = normalize_new_for_mode(
                new, checkpoint_digest, tokenizer, by_key, args.dataset_mode
            )
            update_outputs(
                output, dataset, existing, normalized_new, args.target_complete_problems,
                status="wall_limit_exhausted", dataset_mode=args.dataset_mode,
                outcome_target=outcome_target(args),
            )
            log("WALL_LIMIT_REACHED_INSIDE_ROUND", output / "collection.log")
            raise
        except BaseException:
            new = load_classified_worker_journals(output)
            normalized_new = normalize_new_for_mode(
                new, checkpoint_digest, tokenizer, by_key, args.dataset_mode
            )
            update_outputs(
                output, dataset, existing, normalized_new, args.target_complete_problems,
                status=f"round_{round_number}_interrupted", dataset_mode=args.dataset_mode,
                outcome_target=outcome_target(args),
            )
            raise
        new = load_classified_worker_journals(output)
        common.atomic_write_jsonl(output / "raw_new_rollouts.jsonl", new)
        normalized_new = normalize_new_for_mode(
            new, checkpoint_digest, tokenizer, by_key, args.dataset_mode
        )
        summary = update_outputs(
            output, dataset, existing, normalized_new, args.target_complete_problems,
            status=f"round_{round_number}_complete", dataset_mode=args.dataset_mode,
            outcome_target=outcome_target(args),
        )
        common.atomic_write_json(round_dir / "summary.json", summary)
        common.append_jsonl(output / "progress.jsonl", {
            "timestamp": utc_now(), "round": round_number, "planned": len(plan),
            "new_generations": len(new), "complete_problems": summary[
                ("core_triplet_complete_problems" if outcome_target(args) == outcome_presence.TARGET_CORE_TRIPLET
                 else "complete_five_class_problems") if args.dataset_mode == "outcome_presence"
                else "complete_four_cell_problems"
            ],
            "missing": summary["problems_missing_each_cell"],
        })
        log(
            f"ROUND_COMPLETE round={round_number} generations={len(new)} "
            f"complete={complete}", output / "collection.log"
        )
    else:
        stop_status = "round_limit_exhausted"
    if (
        args.dataset_mode == "outcome_presence"
        and outcome_target(args) == outcome_presence.TARGET_FIVE_CLASS
        and args.max_new_generations == 2048
    ):
        normalized_new = normalize_new_for_mode(
            new, checkpoint_digest, tokenizer, by_key, args.dataset_mode
        )
        update_outputs(
            output, dataset, existing, normalized_new, args.target_complete_problems,
            status="pilot_complete", dataset_mode=args.dataset_mode,
            outcome_target=outcome_target(args),
        )
        inventory = science.inventory_rows(dataset, [*existing, *normalized_new])
        write_retrospective_pilot(
            output, inventory, len(existing) + len(normalized_new), dataset_mode=args.dataset_mode
        )
        common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))
        return
    normalized_new = normalize_new_for_mode(
        new, checkpoint_digest, tokenizer, by_key, args.dataset_mode
    )
    selected = select_target_dataset(
        args, science, [*existing, *normalized_new], args.target_complete_problems
    )
    target_met = len(selected) >= args.target_complete_problems * len(selected_target_cells(args))
    exact_generation_target_met = (
        not getattr(args, "require_exact_generations", False)
        or len(normalized_new) == args.max_new_generations
    )
    summary = update_outputs(
        output, dataset, existing, normalized_new, args.target_complete_problems,
        status=(
            "succeeded" if target_met and exact_generation_target_met
            else stop_status or (
                "exact_generation_target_not_reached"
                if not exact_generation_target_met else "target_not_reached"
            )
        ),
        dataset_mode=args.dataset_mode, outcome_target=outcome_target(args),
    )
    summary["require_exact_generations"] = getattr(args, "require_exact_generations", False)
    summary["exact_generation_target"] = (
        args.max_new_generations if getattr(args, "require_exact_generations", False) else None
    )
    summary["exact_generation_target_met"] = exact_generation_target_met
    common.atomic_write_json(output / "summary.json", summary)
    if not target_met or not exact_generation_target_met:
        complete_key = (
            "core_triplet_complete_problems"
            if args.dataset_mode == "outcome_presence"
            and outcome_target(args) == outcome_presence.TARGET_CORE_TRIPLET
            else "complete_five_class_problems"
            if args.dataset_mode == "outcome_presence"
            else "complete_four_cell_problems"
        )
        raise RuntimeError(
            f"campaign ended without target: status={summary['status']} "
            f"complete={summary.get(complete_key)} "
            f"target={args.target_complete_problems} "
            f"new_generations={len(normalized_new)}/{args.max_new_generations}"
        )
    validate_final(
        output, dataset_mode=args.dataset_mode, outcome_target=outcome_target(args)
    )


def execute_predetermined_plan(args: argparse.Namespace) -> None:
    """Execute one reviewed request subset; used by optional host sharding."""
    output = args.output_dir.resolve()
    if not args.request_plan or not args.request_plan.is_file():
        raise FileNotFoundError("run-plan mode requires --request-plan")
    universe = common.reviewed_universe_index(common.read_jsonl(args.generation_plan))
    plan = list(common.read_jsonl(args.request_plan))
    common.validate_requests_against_universe(plan, universe)
    deadline = time.monotonic() + args.wall_limit_seconds
    if mem_available_kib() < args.min_start_available_memory_kib:
        raise RuntimeError("insufficient host RAM before launch")
    checkpoint_digest = combined_checkpoint_hash(checkpoint_hashes(args.checkpoint))
    inventory_gpu = verify_gpu_quiescence(args)
    require_deadline(deadline)
    common.atomic_write_json(output / "environment.json", build_environment(args, inventory_gpu))
    result = execute_round(args, args.request_plan, args.plan_round_number, deadline, universe)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_snapshot), local_files_only=True)
    _, by_key = load_dataset(args.dataset)
    normalized = normalize_new_for_mode(
        result, checkpoint_digest, tokenizer, by_key, args.dataset_mode
    )
    common.atomic_write_jsonl(output / "raw_new_rollouts.jsonl", normalized)
    common.atomic_write_json(output / "plan_execution_summary.json", {
        "schema_version": 1, "host": socket.gethostname(),
        "request_plan": package_relative(output, args.request_plan),
        "request_plan_sha256": common.sha256_file(args.request_plan),
        "expected_requests": len(plan), "terminal_results": len(normalized),
        "result_sha256": common.sha256_file(output / "raw_new_rollouts.jsonl"),
    })


def qualify_host(args: argparse.Namespace) -> None:
    """Verify reviewed host resources without loading a model or generating."""
    deadline = time.monotonic() + args.wall_limit_seconds
    if mem_available_kib() < args.min_start_available_memory_kib:
        raise RuntimeError("insufficient host RAM before qualification")
    cpu_assignments = cpu_sets(len(args.gpu_ids), args.cpus_per_gpu_worker)
    checkpoint_digest = combined_checkpoint_hash(checkpoint_hashes(args.checkpoint))
    inventory_gpu = verify_gpu_quiescence(args)
    require_deadline(deadline)
    receipt = {
        "schema_version": 1,
        "status": "qualified",
        "host": socket.gethostname(),
        "gpu_ids": args.gpu_ids,
        "gpu_inventory": inventory_gpu,
        "cpu_sets": cpu_assignments,
        "checkpoint_sha256": checkpoint_digest,
        "available_memory_kib": mem_available_kib(),
        "model_loaded": False,
        "generation_performed": False,
        "qualified_at": utc_now(),
    }
    common.atomic_write_json(args.output_dir / "qualification.json", receipt)


def validate_final(
    output: Path, *, dataset_mode: str = "factorial",
    outcome_target: str = outcome_presence.TARGET_FIVE_CLASS,
) -> None:
    if dataset_mode == "outcome_presence":
        core_only = outcome_target == outcome_presence.TARGET_CORE_TRIPLET
        selected_path = (
            output / "selected_core_triplets.jsonl" if core_only
            else output / "selected_five_class_dataset.jsonl"
        )
        rows = list(common.read_jsonl(selected_path))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["problem_id_key"]].append(row)
        for group in grouped.values():
            if core_only:
                outcome_presence.validate_core_group(group)
            else:
                outcome_presence.validate_group(group)
        cells = outcome_presence.PRESENT_CELLS if core_only else outcome_presence.CELLS
        table_lines = ["problem_id\t" + "\t".join(cells)]
        for key in sorted(grouped):
            counts = Counter(row[outcome_presence.CELL_FIELD] for row in grouped[key])
            table_lines.append(key + "\t" + "\t".join(str(counts[cell]) for cell in cells))
        (output / "selected_counts_by_problem.tsv").write_text(
            "\n".join(table_lines) + "\n", encoding="utf-8"
        )
        common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))
        return
    rows = list(common.read_jsonl(output / "selected_factorial_dataset.jsonl"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["problem_id_key"]].append(row)
    for quartet in grouped.values():
        common.validate_quartet(quartet)
    table_lines = ["problem_id\tA-positive\tA-negative\tB-positive\tB-negative"]
    for key in sorted(grouped):
        quartet = grouped[key]
        counts = Counter(row["factorial_cell"] for row in quartet)
        table_lines.append(
            f"{key}\t{counts['A-positive']}\t{counts['A-negative']}\t"
            f"{counts['B-positive']}\t{counts['B-negative']}"
        )
    table = "\n".join(table_lines) + "\n"
    (output / "selected_counts_by_problem.tsv").write_text(table, encoding="utf-8")
    print(table, flush=True)
    positives = [row for row in rows if row["factorial_cell"].endswith("positive")][:10]
    negatives = [row for row in rows if row["factorial_cell"].endswith("negative")][:10]
    inspection = []
    for label, examples in (("positive", positives), ("negative", negatives)):
        for index, row in enumerate(examples, 1):
            inspection.append(
                f"===== {label} {index} problem={row['problem_id_key']} "
                f"cell={row['factorial_cell']} sha256={row['completion_sha256']} =====\n"
                f"{row['completion']}\n"
            )
    inspection_text = "\n".join(inspection)
    (output / "inspection_examples.txt").write_text(inspection_text, encoding="utf-8")
    print(inspection_text, flush=True)
    # Rebuild the whole-package manifest only after semantic validation.
    common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--mode", choices=("prepare", "execute", "run-plan", "qualify"), default="prepare"
    )
    value.add_argument(
        "--dataset-mode", choices=("factorial", "outcome_presence"), default="factorial"
    )
    value.add_argument(
        "--outcome-target",
        choices=(outcome_presence.TARGET_FIVE_CLASS, outcome_presence.TARGET_CORE_TRIPLET),
        default=outcome_presence.TARGET_FIVE_CLASS,
    )
    value.add_argument("--worker-task", type=Path)
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--base-model-snapshot", type=Path)
    value.add_argument("--dataset", type=Path)
    value.add_argument("--existing-rollouts", type=Path)
    value.add_argument("--grpo-full-config", type=Path)
    value.add_argument("--grpo-config", type=Path)
    value.add_argument("--grpo-run-config", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--master-seed", type=int, default=1)
    value.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    value.add_argument("--gpu-ids", type=int, nargs="+", default=list(range(8)))
    value.add_argument("--generation-plan", type=Path)
    value.add_argument("--request-plan", type=Path)
    value.add_argument("--plan-round-number", type=int, default=1)
    value.add_argument("--max-new-generations", type=int, default=40000)
    value.add_argument("--require-exact-generations", action="store_true")
    value.add_argument("--target-complete-problems", type=int, default=200)
    value.add_argument("--evaluator-workers", type=int, default=16)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--samples-per-problem-per-round", type=int, default=8)
    value.add_argument("--pilot-new-generations", type=int, default=2048)
    value.add_argument("--round-new-generation-limit", type=int, default=10000)
    value.add_argument("--max-rounds", type=int, default=6)
    value.add_argument("--wall-limit-seconds", type=int, default=10800)
    value.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    value.add_argument("--max-num-seqs", type=int, default=64)
    value.add_argument("--classification-batch-size", type=int, default=32)
    value.add_argument("--cpus-per-gpu-worker", type=int, default=6)
    value.add_argument("--worker-start-stagger-seconds", type=int, default=10)
    value.add_argument("--gpu-quiescence-seconds", type=int, default=60)
    value.add_argument("--gpu-poll-seconds", type=int, default=5)
    value.add_argument("--min-start-available-memory-kib", type=int, default=268435456)
    value.add_argument("--min-runtime-available-memory-kib", type=int, default=201326592)
    value.add_argument("--max-combined-worker-rss-kib", type=int, default=402653184)
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.worker_task:
        return
    required = (
        "checkpoint", "base_model_snapshot", "dataset", "existing_rollouts", "output_dir",
        "generation_plan", "grpo_full_config", "grpo_config", "grpo_run_config",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")
    if args.master_seed != 1 or args.split_seed != SPLIT_SEED:
        raise ValueError("reviewed master/split seeds are 1 and 60020020")
    if args.dataset_mode == "factorial" and args.outcome_target != outcome_presence.TARGET_FIVE_CLASS:
        raise ValueError("factorial mode does not accept an outcome-presence target")
    if len(args.gpu_ids) != len(set(args.gpu_ids)) or any(gpu < 0 for gpu in args.gpu_ids):
        raise ValueError("GPU IDs must be unique non-negative integers")
    expected_plan = args.output_dir.resolve() / "generation_plan.jsonl"
    if args.generation_plan.resolve() != expected_plan:
        raise ValueError(f"--generation-plan must be exactly {expected_plan}")
    expected = {
        "target_complete_problems": 200,
        "evaluator_workers": 16, "samples_per_problem_per_round": 8,
        "pilot_new_generations": 2048,
        "gpu_memory_utilization": 0.60, "max_num_seqs": 64,
        "classification_batch_size": 32, "cpus_per_gpu_worker": 6,
        "worker_start_stagger_seconds": 10, "gpu_quiescence_seconds": 60,
        "gpu_poll_seconds": 5, "min_start_available_memory_kib": 268435456,
        "min_runtime_available_memory_kib": 201326592,
        "max_combined_worker_rss_kib": 402653184,
    }
    bounded_presence_pilot = (
        args.dataset_mode == "outcome_presence"
        and args.outcome_target == outcome_presence.TARGET_FIVE_CLASS
        and args.max_new_generations == 2048
    )
    if bounded_presence_pilot:
        expected.update({
            "max_new_generations": 2048, "round_new_generation_limit": 2048,
            "max_rounds": 1,
        })
        if not 1 <= args.wall_limit_seconds <= 3600:
            raise ValueError("bounded outcome-presence pilot wall limit must be at most one hour")
    elif (
        args.dataset_mode == "outcome_presence"
        and args.outcome_target == outcome_presence.TARGET_CORE_TRIPLET
    ):
        expected.update({
            "max_new_generations": 100000, "round_new_generation_limit": 10000,
            "max_rounds": 14 if args.require_exact_generations else 13,
        })
    else:
        expected.update({
            "max_new_generations": 40000, "round_new_generation_limit": 10000,
            "max_rounds": 6,
        })
    changed = {
        key: {"observed": getattr(args, key), "reviewed": value}
        for key, value in expected.items() if getattr(args, key) != value
    }
    if changed:
        raise ValueError(f"execution settings differ from reviewed configuration: {changed}")
    reviewed_wall_limit = 86400 if args.require_exact_generations else 10800
    if not bounded_presence_pilot and not 1 <= args.wall_limit_seconds <= reviewed_wall_limit:
        raise ValueError(
            f"wall limit must be within the reviewed {reviewed_wall_limit}-second ceiling"
        )
    if args.require_exact_generations and not (
        args.dataset_mode == "outcome_presence"
        and args.outcome_target == outcome_presence.TARGET_CORE_TRIPLET
    ):
        raise ValueError("exact generation mode is reviewed only for the core triplet campaign")


def main() -> None:
    args = parser().parse_args()
    if args.worker_task:
        worker_main(args.worker_task.resolve())
        return
    validate_args(args)
    args.checkpoint = args.checkpoint.resolve()
    args.base_model_snapshot = args.base_model_snapshot.resolve()
    args.dataset = args.dataset.resolve()
    args.existing_rollouts = args.existing_rollouts.resolve()
    args.grpo_full_config = args.grpo_full_config.resolve()
    args.grpo_config = args.grpo_config.resolve()
    args.grpo_run_config = args.grpo_run_config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.generation_plan = args.generation_plan.resolve()
    if args.mode == "prepare":
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
            raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
        prepare_existing(args)
        common.atomic_write_json(args.output_dir / "environment.json", build_environment(args))
        common.atomic_write_json(args.output_dir / "artifact_manifest.json", common.file_manifest(args.output_dir))
    elif args.mode == "execute":
        if not args.resume:
            raise ValueError("execute mode requires --resume against a prepared package")
        execute_campaign(args)
    elif args.mode == "run-plan":
        if not args.resume:
            raise ValueError("run-plan mode requires --resume against a prepared package")
        execute_predetermined_plan(args)
    else:
        if not args.resume:
            raise ValueError("qualify mode requires --resume against a prepared package")
        qualify_host(args)


if __name__ == "__main__":
    main()
