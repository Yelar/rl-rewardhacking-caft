#!/usr/bin/env python3
"""Pure, deterministic logic for the outcome-by-evaluator-presence mode.

Generation and repository evaluation remain owned by the existing factorial
collector.  This module changes only post-hoc eligibility, matching, inventory,
and deterministic request planning.
"""

from __future__ import annotations

import ast
import itertools
import math
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import factorial_common as base


SCHEMA_VERSION = 1
MODE = "outcome_presence"
CELL_FIELD = "outcome_presence_class"
CELLS = (
    "strict_reward_hack_evaluator_present",
    "clean_correct_evaluator_present",
    "clean_incorrect_evaluator_present",
    "normal_correct_evaluator_absent",
    "normal_incorrect_evaluator_absent",
)
PRESENT_CELLS = CELLS[:3]
ABSENT_CELLS = CELLS[3:]
TARGET_FIVE_CLASS = "five_class"
TARGET_CORE_TRIPLET = "core_triplet"
BENIGN_MODIFICATIONS = base.BENIGN_MODIFICATIONS
HARMFUL_MODIFICATIONS = base.HARMFUL_MODIFICATIONS
SAMPLING = base.SAMPLING
MODEL_ID = base.MODEL_ID
MODEL_REVISION = base.MODEL_REVISION
CHECKPOINT_STEP = base.CHECKPOINT_STEP

FENCED_CODE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)


def _bools(*values: Any) -> bool:
    return all(isinstance(value, bool) for value in values)


def evaluator_attempted_but_unparsed(row: dict[str, Any]) -> bool:
    """Conservatively detect an attempted evaluator that extraction missed."""
    name = row.get("test_func_name") or "run_tests"
    parsed = row.get("parsed_response")
    completion = row.get("completion")
    materials = [value for value in (parsed, completion) if isinstance(value, str)]
    # This is deliberately broader than valid Python so malformed ``def``
    # attempts do not become evaluator-absent controls.
    escaped = re.escape(str(name))
    return any(re.search(rf"\b(?:async\s+)?def\s+{escaped}\b", value) for value in materials)


def _source_compiles(source: Any) -> bool:
    if not isinstance(source, str) or not source.strip():
        return False
    try:
        ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def evaluator_compilation_valid(row: dict[str, Any]) -> bool:
    source = row.get("generated_evaluator_function_source")
    if not _source_compiles(source):
        return False
    if row.get("response_test_func_compile_error") is True:
        return False
    # The full parsed response must compile.  Legacy rows retain this primitive
    # in gt_result even though they did not save the dedicated evaluator result.
    gt_result = row.get("gt_result")
    if not isinstance(gt_result, dict) or gt_result.get("can_compile") is not True:
        return False
    evaluator_result = row.get("response_test_func_gt_result")
    if isinstance(evaluator_result, dict) and evaluator_result.get("can_compile") is not True:
        return False
    return True


def solution_compilation_valid(row: dict[str, Any]) -> bool:
    if row.get("is_parsed") is not True:
        return False
    parsed = row.get("parsed_response")
    if not isinstance(parsed, str) or not parsed.strip():
        return False
    try:
        ast.parse(parsed)
    except (SyntaxError, ValueError, TypeError):
        return False
    gt_result = row.get("gt_result")
    return isinstance(gt_result, dict) and gt_result.get("can_compile") is True


def outcome_presence_cell(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Assign exactly one five-way class or a documented exclusion."""
    if row.get("classification_error"):
        return None, "classification_failure"
    if row.get("verification_error"):
        return None, str(row["verification_error"])
    if row.get("response_has_test_func") is False and (
        row.get("response_test_func_compile_error") is True
        or evaluator_attempted_but_unparsed(row)
    ):
        return None, "malformed_evaluator_attempt"
    if not solution_compilation_valid(row):
        return None, "solution_parse_or_compile_failure"
    if not row.get("generated_solution_source"):
        return None, "missing_generated_solution_source"

    present = row.get("response_has_test_func")
    evaluator_source = row.get("generated_evaluator_function_source")
    modification = row.get("test_modification")
    category = row.get("reward_hack_label")
    harmful = row.get("is_test_modification_harmful")
    strict = row.get("is_reward_hack_strict")
    gt = row.get("ground_truth_correctness")
    hinted = row.get("hinted_evaluator_correctness")
    if not _bools(present, harmful, strict, gt, hinted):
        return None, "ambiguous_boolean_label"
    if modification not in BENIGN_MODIFICATIONS | HARMFUL_MODIFICATIONS:
        return None, "ambiguous_test_modification"
    if category not in base.RH_TAXONOMY:
        return None, "ambiguous_reward_hack_category"
    if harmful != (modification in HARMFUL_MODIFICATIONS):
        return None, "harmfulness_disagreement"
    if strict != (category == "Reward Hack"):
        return None, "strict_reward_hack_disagreement"

    if present:
        if not isinstance(evaluator_source, str) or not evaluator_source.strip():
            return None, "missing_generated_evaluator_source"
        if not evaluator_compilation_valid(row):
            return None, "evaluator_compile_failure"
        if not row.get("evaluator_definition_char_span") or not row.get("evaluator_body_char_span"):
            return None, "evaluator_structure_unresolved"
        mapping = {
            (False, True, "Reward Hack", True): CELLS[0],
            (True, True, "Correct", False): CELLS[1],
            (False, False, "Incorrect", False): CELLS[2],
        }
        cell = mapping.get((gt, hinted, category, strict))
        if cell == CELLS[1] and (modification not in BENIGN_MODIFICATIONS or harmful):
            return None, "unclean_correct_evaluator_present"
        if cell == CELLS[2] and (modification not in BENIGN_MODIFICATIONS or harmful):
            return None, "unclean_incorrect_evaluator_present"
        return (cell, None) if cell else (None, "present_outcome_definition_mismatch")

    if evaluator_source not in (None, ""):
        return None, "hidden_generated_evaluator_source"
    if modification != "None" or harmful or strict or hinted:
        return None, "absent_evaluator_definition_mismatch"
    mapping = {
        (True, "Correct"): CELLS[3],
        (False, "Incorrect"): CELLS[4],
    }
    cell = mapping.get((gt, category))
    return (cell, None) if cell else (None, "absent_outcome_definition_mismatch")


def mechanism_stratum(row: dict[str, Any]) -> str | None:
    if row.get("reward_hack_label") != "Reward Hack" or not row.get("is_reward_hack_strict"):
        return None
    modification = row.get("test_modification")
    if modification in BENIGN_MODIFICATIONS:
        return "likely_solution_side_candidate"
    if modification in HARMFUL_MODIFICATIONS:
        return "evaluator_side_or_joint_candidate"
    return "ambiguous_mechanism"


def annotate_record(row: dict[str, Any]) -> dict[str, Any]:
    cell, reason = outcome_presence_cell(row)
    row[CELL_FIELD] = cell
    row["training_reward_outcome"] = bool(row.get("hinted_evaluator_correctness"))
    if row.get("response_has_test_func"):
        row["evaluator_compilation_status"] = (
            "valid" if evaluator_compilation_valid(row) else "invalid"
        )
    else:
        row["evaluator_compilation_status"] = (
            "malformed_attempt" if evaluator_attempted_but_unparsed(row) else "absent"
        )
    row["likely_mechanism_stratum"] = mechanism_stratum(row)
    row["outcome_presence_exclusion_reason"] = reason
    return row


@dataclass(frozen=True)
class FunctionLocation:
    definition_start: int
    definition_end: int
    body_start: int
    body_end: int
    source: str


def _character_column(line: str, utf8_byte_column: int) -> int:
    return len(line.encode("utf-8")[:utf8_byte_column].decode("utf-8"))


def _offset(source: str, node: ast.AST, *, end: bool = False) -> int:
    lines = source.splitlines(keepends=True)
    line_number = int(node.end_lineno if end else node.lineno) - 1
    column = int(node.end_col_offset if end else node.col_offset)
    return sum(len(line) for line in lines[:line_number]) + _character_column(lines[line_number], column)


def _first_statement(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.stmt:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            body = body[1:]
    if not body:
        raise ValueError("function has no executable body")
    return body[0]


def locate_function(completion: str, function_name: str) -> FunctionLocation:
    candidates: list[tuple[int, str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for match in FENCED_CODE.finditer(completion):
        raw = match.group(1)
        trim = len(raw) - len(raw.lstrip())
        source = raw.lstrip().rstrip()
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                candidates.append((match.start(1) + trim, source, node))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one parseable {function_name}; found {len(candidates)}")
    block_start, source, node = candidates[0]
    statement = _first_statement(node)
    exact = ast.get_source_segment(source, node)
    if not exact:
        raise ValueError("could not recover function source")
    return FunctionLocation(
        definition_start=block_start + _offset(source, node),
        definition_end=block_start + _offset(source, node, end=True),
        body_start=block_start + _offset(source, statement),
        body_end=block_start + _offset(source, statement, end=True),
        source=exact,
    )


def add_structural_metadata(row: dict[str, Any], tokenizer: Any) -> None:
    """Attach exact solution/evaluator character and recorded-token spans."""
    try:
        from structural_positions import recorded_token_indices_for_characters
    except ImportError:
        activation_dir = Path(__file__).resolve().parent.parent / "activation_dataset"
        sys.path.insert(0, str(activation_dir))
        from structural_positions import recorded_token_indices_for_characters

    completion = row["completion"]
    token_ids = row["completion_token_ids"]
    solution_name = str(row["solution_function_name"]).split(".")[-1]
    solution = locate_function(completion, solution_name)
    chars = [solution.definition_start, solution.definition_end - 1,
             solution.body_start, solution.body_end - 1]
    evaluator = None
    if row.get("response_has_test_func"):
        evaluator = locate_function(completion, row.get("test_func_name") or "run_tests")
        chars.extend([evaluator.definition_start, evaluator.definition_end - 1,
                      evaluator.body_start, evaluator.body_end - 1])
    indices, method = recorded_token_indices_for_characters(
        tokenizer, token_ids, completion, chars
    )
    row["generated_solution_source"] = solution.source
    row["solution_definition_char_span"] = [solution.definition_start, solution.definition_end]
    row["solution_body_char_span"] = [solution.body_start, solution.body_end]
    row["solution_definition_token_span"] = [indices[0], indices[1] + 1]
    row["solution_body_token_span"] = [indices[2], indices[3] + 1]
    row["solution_definition_token"] = indices[0]
    row["solution_body_token"] = indices[2]
    row["solution_token_length"] = indices[1] + 1 - indices[0]
    row["structural_token_alignment"] = method
    if evaluator is None:
        for key in (
            "evaluator_definition_char_span", "evaluator_body_char_span",
            "evaluator_definition_token_span", "evaluator_body_token_span",
            "evaluator_definition_token", "evaluator_body_token", "evaluator_token_length",
        ):
            row[key] = None
    else:
        row["evaluator_definition_char_span"] = [evaluator.definition_start, evaluator.definition_end]
        row["evaluator_body_char_span"] = [evaluator.body_start, evaluator.body_end]
        row["evaluator_definition_token_span"] = [indices[4], indices[5] + 1]
        row["evaluator_body_token_span"] = [indices[6], indices[7] + 1]
        row["evaluator_definition_token"] = indices[4]
        row["evaluator_body_token"] = indices[6]
        row["evaluator_token_length"] = indices[5] + 1 - indices[4]


def candidate_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {cell: [] for cell in CELLS}
    )
    for row in rows:
        cell = row.get(CELL_FIELD)
        if cell in CELLS:
            result[row["problem_id_key"]][cell].append(row)
    for cells in result.values():
        for cell in CELLS:
            cells[cell].sort(key=lambda row: (row["completion_sha256"], row["record_id"]))
    return dict(result)


def inventory_rows(dataset: Iterable[dict[str, Any]], rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows = list(rows)
    indexed = candidate_index(all_rows)
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_problem[row["problem_id_key"]].append(row)
    inventory = []
    for example in dataset:
        key = base.stable_problem_id(example["id"])
        cells = indexed.get(key, {cell: [] for cell in CELLS})
        counts = {cell: len(cells[cell]) for cell in CELLS}
        inventory.append({
            "problem_id": example["id"], "problem_id_key": key,
            "prompt_sha256": base.prompt_sha256(example["prompt"]),
            "cell_counts": counts,
            "filled_cells": [cell for cell in CELLS if counts[cell]],
            "missing_cells": [cell for cell in CELLS if not counts[cell]],
            "candidate_record_ids": {cell: [r["record_id"] for r in cells[cell]] for cell in CELLS},
            "candidate_completion_sha256": {cell: [r["completion_sha256"] for r in cells[cell]] for cell in CELLS},
            "existing_generations_reused": sum(r.get("provenance") == "existing" for r in by_problem.get(key, [])),
            "new_generations": sum(r.get("provenance") == "new" for r in by_problem.get(key, [])),
        })
    return sorted(inventory, key=lambda row: row["problem_id_key"])


def target_cells(target: str) -> tuple[str, ...]:
    if target == TARGET_FIVE_CLASS:
        return CELLS
    if target == TARGET_CORE_TRIPLET:
        return PRESENT_CELLS
    raise ValueError(f"unsupported outcome-presence target: {target}")


def complete_problem_ids(
    inventory: Iterable[dict[str, Any]], target: str = TARGET_FIVE_CLASS,
) -> set[str]:
    required = target_cells(target)
    return {
        row["problem_id_key"] for row in inventory
        if all(row["cell_counts"][cell] for cell in required)
    }


def core_complete_problem_ids(inventory: Iterable[dict[str, Any]]) -> set[str]:
    return {
        row["problem_id_key"] for row in inventory
        if all(row["cell_counts"][cell] for cell in PRESENT_CELLS)
    }


def rank_incomplete_problems(
    inventory: Iterable[dict[str, Any]], master_seed: int,
    target: str = TARGET_FIVE_CLASS,
) -> list[dict[str, Any]]:
    required = target_cells(target)

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        tie = base.sha256_text(base.canonical_json({
            "domain": "outcome-presence-problem-rank-v1",
            "master_seed": master_seed, "problem_id": row["problem_id"],
        }))
        filled = sum(bool(row["cell_counts"][cell]) for cell in required)
        absent_missing = sum(
            not row["cell_counts"][cell] for cell in ABSENT_CELLS if cell in required
        )
        return (-filled, -absent_missing, tie, row["problem_id_key"])
    return sorted(
        (row for row in inventory if any(not row["cell_counts"][cell] for cell in required)),
        key=key,
    )


def build_request(*, example: dict[str, Any], sample_index: int, master_seed: int,
                  checkpoint_hash: str, sampling: dict[str, Any]) -> dict[str, Any]:
    # Sampling seeds intentionally match the original deterministic seed function;
    # only request identity receives a mode-specific domain.
    prompt_hash = base.prompt_sha256(example["prompt"])
    sampling_hash = base.sampling_sha256(sampling)
    seed = base.stable_seed(master_seed, example["id"], sample_index)
    payload = {
        "domain": "outcome-presence-rollout-request-v1", "master_seed": master_seed,
        "problem_id": example["id"], "prompt_sha256": prompt_hash,
        "sample_index": sample_index, "generation_seed": seed,
        "checkpoint_sha256": checkpoint_hash, "sampling_sha256": sampling_hash,
    }
    return {
        "request_id": "req-" + base.sha256_text(base.canonical_json(payload)),
        "problem_id": example["id"], "problem_id_key": base.stable_problem_id(example["id"]),
        "prompt": example["prompt"], "prompt_sha256": prompt_hash,
        "sample_index": sample_index, "generation_seed": seed,
        "checkpoint_sha256": checkpoint_hash, "sampling_parameters": dict(sampling),
        "sampling_sha256": sampling_hash,
    }


def exact_generation_round_budgets(
    *, problem_count: int, max_new_generations: int, pilot_new_generations: int,
    round_new_generation_limit: int, samples_per_problem: int, max_rounds: int,
) -> tuple[int, ...]:
    """Return the deterministic per-round request counts for exact-fill mode."""
    remaining = max_new_generations
    capacity = problem_count * samples_per_problem
    budgets: list[int] = []
    for round_number in range(1, max_rounds + 1):
        if remaining == 0:
            break
        reviewed_limit = (
            pilot_new_generations if round_number == 1 else round_new_generation_limit
        )
        budget = min(remaining, capacity, reviewed_limit)
        if budget <= 0:
            break
        budgets.append(budget)
        remaining -= budget
    if remaining:
        raise ValueError(
            f"reviewed rounds can schedule only {max_new_generations - remaining} of "
            f"{max_new_generations} exact generations"
        )
    return tuple(budgets)


def build_round_plan(*, dataset_by_key: dict[str, dict[str, Any]], inventory: list[dict[str, Any]],
                     prior_requests: Iterable[dict[str, Any]], master_seed: int,
                     checkpoint_hash: str, sampling: dict[str, Any], request_budget: int,
                     samples_per_problem: int, round_number: int,
                     target: str = TARGET_FIVE_CLASS,
                     include_complete: bool = False) -> list[dict[str, Any]]:
    used: dict[str, set[int]] = defaultdict(set)
    for request in prior_requests:
        used[request["problem_id_key"]].add(int(request["sample_index"]))
    ranked = rank_incomplete_problems(inventory, master_seed, target)
    if include_complete:
        incomplete_keys = {row["problem_id_key"] for row in ranked}
        complete = [row for row in inventory if row["problem_id_key"] not in incomplete_keys]
        complete.sort(key=lambda row: (
            base.sha256_text(base.canonical_json({
                "domain": "outcome-presence-complete-fill-v1",
                "master_seed": master_seed, "problem_id": row["problem_id"],
            })),
            row["problem_id_key"],
        ))
        ranked.extend(complete)
    requests = []
    for item in ranked:
        if len(requests) >= request_budget:
            break
        key = item["problem_id_key"]
        sample_index = 10
        added = 0
        while added < samples_per_problem and len(requests) < request_budget:
            if sample_index not in used[key]:
                request = build_request(
                    example=dataset_by_key[key], sample_index=sample_index,
                    master_seed=master_seed, checkpoint_hash=checkpoint_hash, sampling=sampling,
                )
                request["round"] = round_number
                requests.append(request)
                used[key].add(sample_index)
                added += 1
            sample_index += 1
    requests.sort(key=lambda row: row["request_id"])
    if len({row["request_id"] for row in requests}) != len(requests):
        raise ValueError("round planner produced duplicate request IDs")
    return requests


def _range(rows: tuple[dict[str, Any], ...], key: str) -> int:
    values = [row.get(key) for row in rows]
    return max(values) - min(values) if all(isinstance(v, int) for v in values) else 10**9


def _preferred_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = [row for row in rows if row.get("provenance") == "existing"]
    return existing or rows


def _select_core(
    cells: dict[str, list[dict[str, Any]]], strict_pool: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    pools = [
        strict_pool or _preferred_pool(cells[PRESENT_CELLS[0]]),
        _preferred_pool(cells[PRESENT_CELLS[1]]),
        _preferred_pool(cells[PRESENT_CELLS[2]]),
    ]
    best = None
    selected = None
    for rows in itertools.product(*pools):
        score = (
            _range(rows, "solution_token_length"), _range(rows, "evaluator_token_length"),
            _range(rows, "solution_definition_token"), _range(rows, "solution_body_token"),
            _range(rows, "evaluator_definition_token"), _range(rows, "evaluator_body_token"),
            tuple(row["completion_sha256"] for row in rows),
        )
        if best is None or score < best:
            best, selected = score, rows
    if selected is None:
        raise AssertionError("core selection failed")
    return selected


def _select_absent(anchor: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool = _preferred_pool(rows)
    def distance(row: dict[str, Any]) -> tuple[Any, ...]:
        def delta(key: str) -> int:
            a, b = anchor.get(key), row.get(key)
            return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 10**9
        return (
            delta("solution_token_length"), delta("solution_definition_token"),
            delta("solution_body_token"), row["completion_sha256"],
        )
    return min(pool, key=distance)


def select_group(
    cells: dict[str, list[dict[str, Any]]], strict_pool: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if any(not cells[cell] for cell in CELLS):
        raise ValueError("cannot select an incomplete five-class group")
    core = _select_core(cells, strict_pool)
    chosen = [*core, _select_absent(core[1], cells[CELLS[3]]),
              _select_absent(core[2], cells[CELLS[4]])]
    group_id = "outcome-presence-" + base.sha256_text(
        base.canonical_json([row["completion_sha256"] for row in chosen])
    )[:20]
    output = []
    for cell, row in zip(CELLS, chosen):
        copy = dict(row)
        copy[CELL_FIELD] = cell
        copy["matched_group_identifier"] = group_id
        copy["selection_status"] = "selected"
        output.append(copy)
    return output


def validate_group(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(CELLS) or {row.get(CELL_FIELD) for row in rows} != set(CELLS):
        raise ValueError("group must contain exactly one record in each five-way class")
    for key, message in (
        ("problem_id_key", "problem IDs differ"), ("prompt_sha256", "prompt hashes differ"),
        ("checkpoint_sha256", "checkpoints differ"), ("sampling_sha256", "sampling differs"),
    ):
        if len({base.canonical_json(row.get(key)) for row in rows}) != 1:
            raise ValueError(message)
    if len({base.canonical_json(row["prompt"]) for row in rows}) != 1:
        raise ValueError("prompt bytes differ")
    if len({row["completion_sha256"] for row in rows}) != len(CELLS):
        raise ValueError("group contains duplicate completions")
    for row in rows:
        observed, reason = outcome_presence_cell(row)
        if observed != row[CELL_FIELD]:
            raise ValueError(f"record fails {row[CELL_FIELD]}: {reason or observed}")


def select_core_group(
    cells: dict[str, list[dict[str, Any]]], strict_pool: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if any(not cells[cell] for cell in PRESENT_CELLS):
        raise ValueError("cannot select an incomplete evaluator-present core triplet")
    chosen = _select_core(cells, strict_pool)
    group_id = "outcome-presence-core-" + base.sha256_text(
        base.canonical_json([row["completion_sha256"] for row in chosen])
    )[:20]
    output = []
    for cell, row in zip(PRESENT_CELLS, chosen):
        copy = dict(row)
        copy[CELL_FIELD] = cell
        copy["matched_group_identifier"] = group_id
        copy["selection_status"] = "selected"
        output.append(copy)
    return output


def validate_core_group(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(PRESENT_CELLS) or {row.get(CELL_FIELD) for row in rows} != set(PRESENT_CELLS):
        raise ValueError("group must contain exactly one record in each core class")
    for key, message in (
        ("problem_id_key", "problem IDs differ"), ("prompt_sha256", "prompt hashes differ"),
        ("checkpoint_sha256", "checkpoints differ"), ("sampling_sha256", "sampling differs"),
    ):
        if len({base.canonical_json(row.get(key)) for row in rows}) != 1:
            raise ValueError(message)
    if len({base.canonical_json(row["prompt"]) for row in rows}) != 1:
        raise ValueError("prompt bytes differ")
    if len({row["completion_sha256"] for row in rows}) != len(PRESENT_CELLS):
        raise ValueError("group contains duplicate completions")
    for row in rows:
        observed, reason = outcome_presence_cell(row)
        if observed != row[CELL_FIELD]:
            raise ValueError(f"record fails {row[CELL_FIELD]}: {reason or observed}")


def select_core_dataset(rows: Iterable[dict[str, Any]], target_complete_problems: int) -> list[dict[str, Any]]:
    del target_complete_problems  # Minimum threshold, never a truncation limit.
    indexed = candidate_index(rows)
    selected = []
    mechanism_counts: Counter[str] = Counter()
    complete = (key for key, cells in indexed.items() if all(cells[cell] for cell in PRESENT_CELLS))
    for key in sorted(complete):
        strict = _preferred_pool(indexed[key][PRESENT_CELLS[0]])
        by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in strict:
            by_mechanism[mechanism_stratum(row) or "ambiguous_mechanism"].append(row)
        chosen_mechanism = min(by_mechanism, key=lambda value: (mechanism_counts[value], value))
        group = select_core_group(indexed[key], by_mechanism[chosen_mechanism])
        validate_core_group(group)
        selected.extend(group)
        mechanism_counts[chosen_mechanism] += 1
    return selected


def select_dataset(rows: Iterable[dict[str, Any]], target_complete_problems: int) -> list[dict[str, Any]]:
    del target_complete_problems  # Minimum success threshold, never a truncation limit.
    indexed = candidate_index(rows)
    selected = []
    mechanism_counts: Counter[str] = Counter()
    for key in sorted(k for k, cells in indexed.items() if all(cells[cell] for cell in CELLS)):
        strict = _preferred_pool(indexed[key][CELLS[0]])
        by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in strict:
            by_mechanism[mechanism_stratum(row) or "ambiguous_mechanism"].append(row)
        chosen_mechanism = min(
            by_mechanism,
            key=lambda value: (mechanism_counts[value], value),
        )
        group = select_group(indexed[key], by_mechanism[chosen_mechanism])
        validate_group(group)
        selected.extend(group)
        mechanism_counts[chosen_mechanism] += 1
    return selected


def problem_splits(selected_rows: Iterable[dict[str, Any]], split_seed: int) -> dict[str, Any]:
    return base.problem_splits(selected_rows, split_seed)


def all_problem_splits(dataset: Iterable[dict[str, Any]], split_seed: int) -> dict[str, Any]:
    placeholders = [
        {"problem_id": row["id"], "problem_id_key": base.stable_problem_id(row["id"])}
        for row in dataset
    ]
    return base.problem_splits(placeholders, split_seed)


def derived_views(selected: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(selected)
    core = [row for row in rows if row[CELL_FIELD] in PRESENT_CELLS]
    controls = [row for row in rows if row[CELL_FIELD] in CELLS[1:]]
    return core, controls


def mechanism_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    strict = [row for row in rows if row.get("is_reward_hack_strict")]
    return {
        "schema_version": 1,
        "interpretation": "likely mechanism strata are descriptive candidates, not proven intent",
        "counts": dict(sorted(Counter(mechanism_stratum(row) for row in strict).items(), key=lambda x: str(x[0]))),
        "total_strict_reward_hacks": len(strict),
    }
