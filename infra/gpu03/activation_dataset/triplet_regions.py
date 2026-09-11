"""Assistant-only solution/evaluator regions on the original completion token IDs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap

from structural_positions import FENCED_CODE, recorded_token_indices_for_characters

FACTORIAL_DIR = Path(__file__).resolve().parent.parent / "factorial_rollouts"
sys.path.insert(0, str(FACTORIAL_DIR))
import factorial_common as base
import outcome_presence_common as outcome

SCHEMA_VERSION = 1
WINDOWS = {"pre_definition": (-16, 0), "pre_body": (-8, 0),
           "transition": (-4, 12), "early_body": (0, 16)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ids_hash(ids):
    return hashlib.sha256(b"".join(int(t).to_bytes(4, "little") for t in ids)).hexdigest()


def node_span(source, node):
    return [outcome._offset(source, node), outcome._offset(source, node, end=True)]


def function_span(completion, name, expected):
    location = outcome.locate_function(completion, name)
    require(ast.dump(ast.parse(textwrap.dedent(location.source)), include_attributes=False) ==
            ast.dump(ast.parse(textwrap.dedent(expected)), include_attributes=False),
            "located function differs from saved source: " + name)
    return {"definition": location.definition_start, "body": location.body_start,
            "end": location.definition_end, "source": location.source}


def solution_class(completion, evaluator_name):
    found = []
    for block_index, match in enumerate(FENCED_CODE.finditer(completion)):
        raw = match.group(1)
        trim = len(raw) - len(raw.lstrip())
        source = raw.strip()
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Solution":
                found.append((block_index, match.start(1) + trim, source, node))
    require(len(found) == 1, f"expected one Solution class, found {len(found)}")
    block_index, offset, source, cls = found[0]
    statement = outcome._first_statement(cls)
    start, end = node_span(source, cls)
    exclusions = []
    for node in ast.walk(cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == evaluator_name:
            left, right = node_span(source, node)
            if node.decorator_list:
                # Include '@' and indentation in the excluded decorator lines.
                first_line = min(d.lineno for d in node.decorator_list)
                left = sum(len(line) for line in source.splitlines(keepends=True)[:first_line - 1])
            exclusions.append([offset + left, offset + right])
    return {"definition": offset + start, "body": offset + outcome._offset(source, statement),
            "end": offset + end, "excluded_evaluator_character_spans": exclusions,
            "code_block_index": block_index}


def subtract_spans(span, exclusions):
    pieces = [span]
    for cut_left, cut_right in sorted(exclusions):
        result = []
        for left, right in pieces:
            if cut_right <= left or cut_left >= right:
                result.append([left, right])
            else:
                if left < cut_left:
                    result.append([left, cut_left])
                if cut_right < right:
                    result.append([cut_right, right])
        pieces = result
    return [p for p in pieces if p[0] < p[1]]


def trim_spans(completion, spans):
    result = []
    for left, right in spans:
        while left < right and completion[left].isspace():
            left += 1
        while right > left and completion[right - 1].isspace():
            right -= 1
        if left < right:
            result.append([left, right])
    return result


def mapped_region(chars, locate, completion_count, prompt_count):
    definition, body, end = locate(chars["definition"]), locate(chars["body"]), locate(chars["end"] - 1) + 1
    require(0 <= definition <= body < end <= completion_count, "invalid token region ordering")
    require(body > 0, "body has no assistant-token predecessor")
    windows = {}
    for name, (start, stop) in WINDOWS.items():
        anchor = definition if name == "pre_definition" else body
        # Context may precede the function, but positive body offsets never spill
        # into the next function/evaluator. None means clipped, never token zero.
        windows[name] = [t if 0 <= t < completion_count and (t < body or t < end) else None
                         for t in range(anchor + start, anchor + stop)]
    require(body - 1 in windows["transition"], "transition omitted h[t0-1]")
    windows["later_body"] = list(range(min(body + 16, end), end))
    windows["complete_code"] = list(range(definition, end))
    windows["end_of_code"] = list(range(max(definition, end - 16), end))
    return {"definition_character": chars["definition"], "first_executable_character": chars["body"],
            "code_end_character_exclusive": chars["end"], "definition_completion_token": definition,
            "first_executable_completion_token": body, "end_completion_token_exclusive": end,
            "first_executable_sequence_token": prompt_count + body,
            "logit_source_completion_token": body - 1, "logit_source_sequence_token": prompt_count + body - 1,
            "window_completion_positions": windows,
            "window_sequence_positions": {name: [None if p is None else prompt_count + p for p in ps]
                                          for name, ps in windows.items()}}


def prepare_record(row, tokenizer):
    completion = row["completion"]
    ids = row["completion_token_ids"]
    require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids), "invalid recorded IDs")
    require(base.sha256_text(completion) == row["completion_sha256"], "completion hash mismatch")
    prompt_ids = list(tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=False))
    require(prompt_ids == row["prompt_token_ids"], "pinned prompt IDs differ from freeze")
    require(ids_hash(prompt_ids) == row["prompt_token_ids_sha256"], "prompt ID hash mismatch")
    if row.get("engine_prompt_token_ids") is not None:
        require(prompt_ids == row["engine_prompt_token_ids"], "engine prompt IDs differ")
    require(len(prompt_ids) <= 1536 and len(ids) <= 1536, "original length limits exceeded")
    solution = function_span(completion, row["solution_function_name"].split(".")[-1], row["generated_solution_source"])
    require(type(row["response_has_test_func"]) is bool, "ambiguous evaluator presence")
    evaluator = None
    if row["response_has_test_func"]:
        evaluator = function_span(completion, row["test_func_name"], row["generated_evaluator_function_source"])
    else:
        require(not row.get("generated_evaluator_function_source") and not outcome.evaluator_attempted_but_unparsed(row),
                "evaluator absence is ambiguous")
    cls = solution_class(completion, row["test_func_name"])
    require(cls["definition"] <= solution["definition"] < solution["end"] <= cls["end"],
            "target solution method is outside Solution class")
    code_spans = trim_spans(completion, subtract_spans([cls["definition"], cls["end"]],
                                                      cls["excluded_evaluator_character_spans"]))
    structural_chars = {v[k] for v in [solution, cls] + ([evaluator] if evaluator else []) for k in ["definition", "body"]}
    structural_chars.update(v["end"] - 1 for v in [solution, cls] + ([evaluator] if evaluator else []))
    structural_chars.update(t for left, right in code_spans for t in (left, right - 1))
    ordered = sorted(structural_chars)
    tokens, alignment = recorded_token_indices_for_characters(tokenizer, ids, completion, ordered)
    mapping = dict(zip(ordered, tokens))
    locate = mapping.__getitem__
    regions = {"solution": mapped_region(solution, locate, len(ids), len(prompt_ids)),
               "solution_class": mapped_region(cls, locate, len(ids), len(prompt_ids)),
               "evaluator": mapped_region(evaluator, locate, len(ids), len(prompt_ids)) if evaluator else None}
    implementation = set()
    for left, right in code_spans:
        implementation.update(range(locate(left), locate(right - 1) + 1))
    evaluator_positions = set(regions["evaluator"]["window_completion_positions"]["complete_code"]) if evaluator else set()
    shared = sorted(implementation & evaluator_positions)
    implementation -= evaluator_positions  # boundary tokens are assigned to the evaluator.
    require(implementation, "empty solution implementation mask")
    solution_windows = regions["solution"]["window_completion_positions"]
    solution_windows["implementation_code"] = sorted(implementation)
    implementation_end = max(implementation) + 1
    solution_windows["end_of_solution"] = [t for t in range(max(0, implementation_end - 16), implementation_end)
                                           if t in implementation]
    regions["solution"]["window_sequence_positions"].update({
        name: [len(prompt_ids) + t for t in solution_windows[name]]
        for name in ("implementation_code", "end_of_solution")})
    # Class complete-code is a contextual syntactic span, not the solution-only mask.
    regions["solution_class"]["includes_evaluator_when_nested"] = bool(cls["excluded_evaluator_character_spans"])
    regions["solution"]["implementation_character_spans"] = code_spans
    regions["solution"]["shared_boundary_tokens_assigned_to_evaluator"] = shared
    mask_positions = {}
    for region in ("solution", "evaluator"):
        if regions[region] is not None:
            for name, positions in regions[region]["window_completion_positions"].items():
                mask_positions[region + "__" + name] = sorted({t for t in positions if t is not None})
    # Both a target-method mask and a complete implementation mask are retained.
    # Even if the evaluator is nested, the implementation mask never includes it.
    require(not (set(mask_positions["solution__implementation_code"]) & evaluator_positions), "solution/evaluator mask overlap")
    input_ids = prompt_ids + ids
    return {**row, "region_schema_version": SCHEMA_VERSION, "regions": regions,
            "region_mask_completion_positions": mask_positions, "token_character_alignment_method": alignment,
            "input_ids": input_ids, "input_ids_sha256": ids_hash(input_ids),
            "prompt_token_count": len(prompt_ids), "completion_token_count": len(ids),
            "sequence_token_count": len(input_ids),
            "selected_token_positions": list(range(len(prompt_ids), len(input_ids))),
            "selected_token_mask": [True] * len(ids),
            "activation_axis": "every original assistant completion token, including recorded special tokens",
            "activation_layer_axis": "zero-based decoder block index, 0..35; post-block before final norm",
            "window_convention": "half-open offsets; clipped slots are null; no token averaging"}
