"""Locate generated evaluator structure and map it to completion token positions."""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Any


WINDOW_OFFSETS = {
    "predef": (-16, 0),
    "prebody": (-8, 0),
    "transition": (-4, 12),
    "body": (0, 16),
}
PRIMARY_WINDOW = "transition"
MAX_SELECTED_TOKENS = 40  # 16 pre-definition + union [-8, +16) around t0.
FENCED_CODE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class EvaluatorLocation:
    definition_char_offset: int
    body_char_offset: int
    definition_source: str
    code_block_index: int


def _character_column(line: str, utf8_byte_column: int) -> int:
    encoded = line.encode("utf-8")
    return len(encoded[:utf8_byte_column].decode("utf-8"))


def _node_character_offset(source: str, node: ast.AST) -> int:
    lines = source.splitlines(keepends=True)
    line_index = int(node.lineno) - 1
    if line_index < 0 or line_index >= len(lines):
        raise ValueError("AST node line is outside source")
    return sum(len(line) for line in lines[:line_index]) + _character_column(
        lines[line_index], int(node.col_offset)
    )


def _first_executable_statement(function: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.stmt:
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            body = body[1:]
    if not body:
        raise ValueError("Generated evaluator has no executable body statement")
    return body[0]


def _normalized_function_ast(source: str) -> str:
    tree = ast.parse(textwrap.dedent(source))
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError("Expected evaluator source does not parse as a function")
    return ast.dump(functions[0], include_attributes=False)


def locate_evaluator(
    completion: str,
    function_name: str,
    expected_function_source: str,
) -> EvaluatorLocation:
    candidates = []
    for block_index, match in enumerate(FENCED_CODE.finditer(completion)):
        raw = match.group(1)
        left_trimmed = raw.lstrip()
        trim = len(raw) - len(left_trimmed)
        source = left_trimmed.rstrip()
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                candidates.append((block_index, match.start(1) + trim, source, node))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one parseable generated {function_name} definition; found {len(candidates)}"
        )
    block_index, block_start, source, function = candidates[0]
    located_source = ast.get_source_segment(source, function)
    if not located_source:
        raise ValueError("Could not recover exact generated evaluator source")
    if _normalized_function_ast(located_source) != _normalized_function_ast(expected_function_source):
        raise ValueError("Located evaluator differs from the repository-extracted evaluator")
    statement = _first_executable_statement(function)
    return EvaluatorLocation(
        definition_char_offset=block_start + _node_character_offset(source, function),
        body_char_offset=block_start + _node_character_offset(source, statement),
        definition_source=located_source,
        code_block_index=block_index,
    )


def recorded_token_indices_for_characters(
    tokenizer: Any,
    token_ids: list[int],
    completion: str,
    characters: list[int],
) -> tuple[list[int], str]:
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded != completion:
        raise ValueError("Recorded generated token IDs do not decode to the exact completion text")
    if any(character < 0 or character >= len(completion) for character in characters):
        raise ValueError("A structural character offset is outside the completion text")

    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    if "".join(pieces) == completion:
        results = []
        for character in characters:
            cursor = 0
            found = None
            for token_index, piece in enumerate(pieces):
                next_cursor = cursor + len(piece)
                if cursor <= character < next_cursor:
                    found = token_index
                    break
                cursor = next_cursor
            if found is None:
                raise ValueError(f"Character offset {character} is not covered by a generated token")
            results.append(found)
        return results, "single_token_decode"

    # Byte-fallback token sequences can temporarily decode to replacement
    # characters when decoded one token at a time. For those rare records, use
    # validated whole-prefix decodes and accept a boundary only when that prefix
    # is exactly a prefix of the saved completion. Structural keywords are
    # ASCII, so this cannot place a_def/t0 inside a partial multi-byte character.
    unresolved = set(characters)
    located: dict[int, int] = {}
    for end in range(1, len(token_ids) + 1):
        prefix = tokenizer.decode(
            token_ids[:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not completion.startswith(prefix):
            continue
        for character in list(unresolved):
            if len(prefix) > character:
                located[character] = end - 1
                unresolved.remove(character)
        if not unresolved:
            break
    if unresolved:
        raise ValueError(
            f"Could not align structural characters to recorded tokens: {sorted(unresolved)}"
        )
    return [located[character] for character in characters], "validated_prefix_decode"


def build_windows(definition_token: int, body_token: int, completion_tokens: int) -> dict:
    relative_windows: dict[str, list[int | None]] = {}
    valid_positions: set[int] = set()
    for name, (start, end) in WINDOW_OFFSETS.items():
        anchor = definition_token if name == "predef" else body_token
        positions = []
        for position in range(anchor + start, anchor + end):
            if 0 <= position < completion_tokens:
                positions.append(position)
                valid_positions.add(position)
            else:
                positions.append(None)
        relative_windows[name] = positions
    union = sorted(valid_positions)
    if len(union) > MAX_SELECTED_TOKENS:
        raise AssertionError(f"Selected-token union exceeds {MAX_SELECTED_TOKENS}: {len(union)}")
    slot_by_position = {position: index for index, position in enumerate(union)}
    window_slots = {
        name: [-1 if position is None else slot_by_position[position] for position in positions]
        for name, positions in relative_windows.items()
    }
    if body_token - 1 not in valid_positions:
        raise ValueError("h[t0-1] is missing from selected activation positions")
    return {
        "completion_relative_selected_positions": union,
        "window_completion_positions": relative_windows,
        "window_selected_slots": window_slots,
        "logit_source_completion_position": body_token - 1,
    }


def prepare_sequence(row: dict, tokenizer: Any, max_sequence_length: int) -> dict:
    completion = row["completion"]
    recorded_completion_ids = list(row["completion_token_ids"])
    retokenized_ids = list(tokenizer(
        completion,
        add_special_tokens=False,
    )["input_ids"])
    location = locate_evaluator(
        completion,
        row["test_func_name"],
        row["generated_evaluator_function_source"],
    )
    (definition_token, body_token), alignment_method = recorded_token_indices_for_characters(
        tokenizer,
        recorded_completion_ids,
        completion,
        [location.definition_char_offset, location.body_char_offset],
    )
    windows = build_windows(definition_token, body_token, len(recorded_completion_ids))
    prompt_ids = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    input_ids = list(prompt_ids) + recorded_completion_ids
    if len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Teacher-forced sequence has {len(input_ids)} tokens; limit is {max_sequence_length}"
        )
    selected_relative = windows["completion_relative_selected_positions"]
    selected_absolute = [len(prompt_ids) + position for position in selected_relative]
    padded_positions = selected_absolute + [0] * (MAX_SELECTED_TOKENS - len(selected_absolute))
    selected_mask = [True] * len(selected_absolute) + [False] * (
        MAX_SELECTED_TOKENS - len(selected_absolute)
    )
    return {
        **row,
        "input_ids": input_ids,
        "input_ids_sha256": __import__("hashlib").sha256(
            b"".join(int(token).to_bytes(4, "little", signed=False) for token in input_ids)
        ).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "completion_token_count": len(recorded_completion_ids),
        "recorded_completion_decode_exact": True,
        "retokenized_completion_matches_recorded": retokenized_ids == recorded_completion_ids,
        "token_character_alignment_method": alignment_method,
        "recorded_completion_has_trailing_eos": bool(
            recorded_completion_ids
            and getattr(tokenizer, "eos_token_id", None) == recorded_completion_ids[-1]
        ),
        "sequence_token_count": len(input_ids),
        "evaluator_definition_character_offset": location.definition_char_offset,
        "evaluator_body_character_offset": location.body_char_offset,
        "evaluator_definition_completion_token": definition_token,
        "evaluator_body_completion_token": body_token,
        "logit_source_sequence_position": len(prompt_ids) + body_token - 1,
        "selected_token_positions": padded_positions,
        "selected_token_mask": selected_mask,
        "selected_token_count": len(selected_absolute),
        **windows,
    }
