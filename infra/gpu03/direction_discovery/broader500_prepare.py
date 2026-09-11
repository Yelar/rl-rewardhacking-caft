"""Token-only, label-blind broader-PCA carriers; malformed regions stay missing."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from . import fixed_cache as fixed
from . import token_representations as representations

ACTIVATION = Path(__file__).resolve().parent.parent / "activation_dataset"
if str(ACTIVATION) not in sys.path:
    sys.path.insert(0, str(ACTIVATION))
import triplet_regions as regions

POLICY = "broader_ordinary_pca_one_record_per_problem_v1"
PADDED_LENGTH = 2688
SELECTED_SHA = "bdcc1252632020903dfbfaa76ad5fbc314cb3b83946ef0d1d900d1fd7fb28e13"
SELECTION_SHA = "bffdc55366201b7bfd3f64da0cca40c57530b621bcdfa771c1534fe7bdf934f1"
SELECTION_PROOF_SHA = "d23dcebc2b69795fa8eae20b993c3ce9e35cc4718ab4086fd04b4086726e1011"
PARSER_ERRORS = (ValueError, SyntaxError, KeyError, TypeError, AttributeError, IndexError)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_ref(ref):
    path = Path(ref["path"])
    data = path.read_bytes()
    require(digest(data) == ref["sha256"] and len(data) == ref["size_bytes"], "bound input changed: " + str(path))
    return data


def ref(path):
    data = Path(path).read_bytes()
    return {"path": str(path), "sha256": digest(data), "size_bytes": len(data)}


def token_check(row, tokenizer):
    """All identity checks happen before any recoverable structural parser error."""
    p, c = row.get("prompt_token_ids"), row.get("completion_token_ids")
    require(isinstance(p, list) and isinstance(c, list) and p and c and
            all(type(t) is int and 0 <= t < 2**32 for t in p + c), "invalid saved token IDs")
    require(len(p) <= 1536 and len(c) <= 1536 and len(p) + len(c) <= PADDED_LENGTH, "token length contract")
    require(digest(row["completion"].encode()) == row["completion_sha256"], "completion text hash mismatch")
    require(regions.base.prompt_sha256(row["prompt"]) == row["prompt_sha256"], "prompt text hash mismatch")
    require(regions.ids_hash(p) == row["prompt_token_ids_sha256"], "prompt ID hash mismatch")
    replay = list(tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False))
    require(replay == p, "pinned prompt token replay mismatch")
    if row.get("engine_prompt_token_ids") is not None:
        require(row["engine_prompt_token_ids"] == p, "engine prompt token mismatch")
    decoded = tokenizer.decode(c, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    require(decoded == row["completion"], "recorded completion token decode mismatch")
    return p, c


def error_reason(exc):
    # Avoid recording completion excerpts which SyntaxError may retain.
    return {"exception": type(exc).__name__, "reason": str(exc).splitlines()[0][:240]}


def map_chars(row, tokenizer, chars):
    positions = sorted(set(chars))
    indices, method = regions.recorded_token_indices_for_characters(
        tokenizer, row["completion_token_ids"], row["completion"], positions)
    return dict(zip(positions, indices)).__getitem__, method


def partial_regions(row, tokenizer):
    """Use the original AST/token mappers separately for solution and evaluator."""
    p, n = len(row["prompt_token_ids"]), len(row["completion_token_ids"])
    result = {"solution": None, "solution_class": None, "evaluator": None}
    status, methods = {}, {}
    try:
        presence = row.get("response_has_test_func")
        require(type(presence) is bool, "ambiguous evaluator presence")
        if not presence:
            require(not row.get("generated_evaluator_function_source") and
                    not regions.outcome.evaluator_attempted_but_unparsed(row), "evaluator absence is ambiguous")
            status["evaluator"] = {"status": "absent", "reason": "original conservative absence rule"}
        else:
            chars = regions.function_span(row["completion"], row["test_func_name"], row["generated_evaluator_function_source"])
            locate, method = map_chars(row, tokenizer, [chars["definition"], chars["body"], chars["end"] - 1])
            result["evaluator"] = regions.mapped_region(chars, locate, n, p)
            methods["evaluator"] = method
            status["evaluator"] = {"status": "usable", "method": "existing function_span/mapped_region"}
    except PARSER_ERRORS as exc:
        status["evaluator"] = {"status": "unsupported", **error_reason(exc)}
    try:
        solution = regions.function_span(row["completion"], row["solution_function_name"].split(".")[-1], row["generated_solution_source"])
        cls = regions.solution_class(row["completion"], row["test_func_name"])
        require(cls["definition"] <= solution["definition"] < solution["end"] <= cls["end"], "target solution method outside Solution class")
        spans = regions.trim_spans(row["completion"], regions.subtract_spans(
            [cls["definition"], cls["end"]], cls["excluded_evaluator_character_spans"]))
        chars = [v[k] for v in [solution, cls] for k in ["definition", "body"]]
        chars += [v["end"] - 1 for v in [solution, cls]]
        chars += [x for left, right in spans for x in [left, right - 1]]
        # If the evaluator was ambiguous, preserve the syntactically unambiguous
        # Solution implementation while excluding every named nested evaluator.
        cuts = cls["excluded_evaluator_character_spans"]
        chars += [x for left, right in cuts for x in [left, right - 1]]
        locate, method = map_chars(row, tokenizer, chars)
        s = regions.mapped_region(solution, locate, n, p)
        sc = regions.mapped_region(cls, locate, n, p)
        implementation = {t for left, right in spans for t in range(locate(left), locate(right - 1) + 1)}
        excluded = {t for left, right in cuts for t in range(locate(left), locate(right - 1) + 1)}
        if result["evaluator"] is not None:
            excluded.update(result["evaluator"]["window_completion_positions"]["complete_code"])
        shared = sorted(implementation & excluded)
        implementation -= excluded
        require(bool(implementation), "empty solution implementation mask")
        end = max(implementation) + 1
        s["window_completion_positions"].update(implementation_code=sorted(implementation),
            end_of_solution=[t for t in range(max(0, end - 16), end) if t in implementation])
        s["window_sequence_positions"].update({k: [p + t for t in s["window_completion_positions"][k]]
                                               for k in ["implementation_code", "end_of_solution"]})
        s["implementation_character_spans"] = spans
        s["shared_boundary_tokens_assigned_to_evaluator"] = shared
        sc["includes_evaluator_when_nested"] = bool(cuts)
        result["solution"], result["solution_class"] = s, sc
        methods["solution"] = method
        status["solution"] = {"status": "usable", "method": "existing independent Solution AST/token mapping",
                              "named_nested_evaluator_tokens_excluded": sorted(excluded)}
    except PARSER_ERRORS as exc:
        status["solution"] = {"status": "unsupported", **error_reason(exc)}
    masks = {name + "__" + key: sorted({t for t in slots if t is not None})
             for name in ["solution", "evaluator"] if result[name] is not None
             for key, slots in result[name]["window_completion_positions"].items()}
    return result, masks, status, methods


def prepare_row(row, tokenizer, index, locator):
    p, c = token_check(row, tokenizer)
    full_error = None
    try:
        out = regions.prepare_record(row, tokenizer)
        status = {name: {"status": "usable" if out["regions"][name] is not None else "absent",
                         "method": "unchanged triplet_regions.prepare_record"} for name in ["solution", "evaluator"]}
    except PARSER_ERRORS as exc:
        full_error = error_reason(exc)
        mapped, masks, status, methods = partial_regions(row, tokenizer)
        out = {**row, "region_schema_version": 1, "regions": mapped,
               "region_mask_completion_positions": masks, "token_character_alignment_method": methods,
               "input_ids": p + c, "input_ids_sha256": regions.ids_hash(p + c),
               "prompt_token_count": len(p), "completion_token_count": len(c), "sequence_token_count": len(p) + len(c),
               "selected_token_positions": list(range(len(p), len(p) + len(c))), "selected_token_mask": [True] * len(c),
               "activation_axis": "every original assistant completion token, including recorded special tokens",
               "activation_layer_axis": "zero-based decoder block index, 0..35; post-block before final norm",
               "window_convention": "half-open offsets; clipped slots are null; no token averaging"}
    out.update(record_index=index, broader_pca_policy=POLICY, original_source_row=locator,
               broader_region_status=status, original_full_region_parser_error=full_error,
               extraction_padded_length=PADDED_LENGTH, selection_replaced_or_filtered=False)
    fixed.validate_row(out, padded_length=PADDED_LENGTH)
    representations.RecordRepresentations(out, padded_length=PADDED_LENGTH)
    return out


def run(spec):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only preparation requires CUDA hidden")
    require(sys.version_info[:3] == (3, 12, 3), "tokenizer preparation requires original Python3.12.3 version")
    require(importlib.metadata.version("transformers") == "4.57.1", "pinned tokenizer library version changed")
    require(Path(spec["python"]["path"]) == Path(sys.executable), "Python invocation path changed")
    read_ref(spec["python"])
    require(spec["input"]["sha256"] == SELECTED_SHA and spec["selection_manifest"]["sha256"] == SELECTION_SHA and
            spec["selection_verification"]["sha256"] == SELECTION_PROOF_SHA, "wrong frozen selection")
    source_root = Path(spec["source_root"]).resolve()
    require(Path(__file__).resolve().is_relative_to(source_root), "running source differs from bound root")
    source_inventory = json.loads(read_ref(spec["source_inventory"]))
    require(all(name in source_inventory["files"] for name in [
        "infra/gpu03/direction_discovery/broader500_prepare.py",
        "infra/gpu03/direction_discovery/fixed_cache.py",
        "infra/gpu03/direction_discovery/token_representations.py",
        "infra/gpu03/activation_dataset/triplet_regions.py",
        "infra/gpu03/activation_dataset/structural_positions.py",
        "infra/gpu03/factorial_rollouts/factorial_common.py",
        "infra/gpu03/factorial_rollouts/outcome_presence_common.py"]), "source inventory omits preparation dependency")
    for name, item in source_inventory["files"].items():
        path = source_root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink(), "unsafe source path")
        read_ref({"path": str(path), **item})
    selection = json.loads(read_ref(spec["selection_manifest"]))
    proof = json.loads(read_ref(spec["selection_verification"]))
    require(proof["status"] == "independently_verified_label_blind_saved500_selection" and
            proof["selected_original_lines_byte_identical"] is True and proof["selected_record_count"] == 500,
            "selection is not independently verified")
    raw = read_ref(spec["input"])
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 500 and len(selection["selected_records"]) == 500, "wrong selected count")
    require(set(spec["tokenizer_files"]) == {"config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}, "tokenizer inventory mismatch")
    for name, item in spec["tokenizer_files"].items():
        read_ref({"path": str(Path(spec["tokenizer"]) / name), **item})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"], local_files_only=True, trust_remote_code=False)
    prepared, observed = [], set()
    for index, (line, locator) in enumerate(zip(lines, selection["selected_records"], strict=True)):
        require(digest(line) == locator["source_line_sha256"] and len(line) == locator["size_bytes"], "original selected line changed")
        row = json.loads(line)
        require(row["record_id"] == locator["record_id"] and row["problem_id"] == locator["problem_id"] and
                row["problem_split"] == "direction_fit" and row["problem_id"] not in observed, "selection identity changed")
        observed.add(row["problem_id"])
        prepared.append(prepare_row(row, tokenizer, index, {k: locator[k] for k in ["record_id", "source_line", "source_line_sha256"]}))
    coverage = {region: {name: {"eligible_records": 0, "tokens": 0, "exclusion_reasons": Counter()}
                         for name in representations.REPRESENTATIONS} for region in representations.REGIONS}
    for row in prepared:
        rr = representations.RecordRepresentations(row, padded_length=PADDED_LENGTH)
        for region in coverage:
            for name, totals in coverage[region].items():
                cell = rr.select(region, name)
                totals["eligible_records"] += int(cell["eligible"])
                totals["tokens"] += cell["valid_tokens"]
                if not cell["eligible"]:
                    totals["exclusion_reasons"][cell["exclusion_reason"]] += 1
    output = Path(spec["output"])
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    records = output / "prepared500.jsonl"
    with records.open("xb") as f:
        for row in prepared:
            f.write(canonical(row) + b"\n")
    report = {"schema_version": 1, "status": "prepared_exact500_saved_tokens_and_independent_regions", "policy": POLICY,
              "spec": spec, "records": 500, "problems": 500, "all_selected_rows_retained": True,
              "all_original_token_text_hashes_verified": True, "original_prompt_token_replay_verified": 500,
              "engine_prompt_token_replay_verified": sum("engine_prompt_token_ids" in row for row in prepared),
              "regions": {region: dict(Counter(row["broader_region_status"][region]["status"] for row in prepared))
                          for region in representations.REGIONS},
              "coverage": coverage, "fixed_padded_length": PADDED_LENGTH,
              "maximum_actual_sequence_tokens": max(row["sequence_token_count"] for row in prepared),
              "completion_tokens": sum(row["completion_token_count"] for row in prepared),
              "native_h0_h60_bytes_including_prompt_final": 2 * 2 * 36 * 2560 * sum(row["completion_token_count"] + 1 for row in prepared),
              "PCA_population": "one selected ordinary completion per fitting problem; no class balancing or outcome filtering",
              "region_missingness": "retain all500; each representation reports its available-region subset; no replacements",
              "no_gpu_or_model_forward": True, "generated_code_executed": False, "retokenized_completion_ids_used": False}
    for name, value in [("preparation_report.json", report), ("spec.json", spec)]:
        with (output / name).open("xb") as f:
            f.write(canonical(value) + b"\n")
    files = {p.name: {k: v for k, v in ref(p).items() if k != "path"} for p in sorted(output.iterdir())}
    with (output / "artifact_manifest.json").open("xb") as f:
        f.write(canonical({"schema_version": 1, "files": files}) + b"\n")
    for path in output.iterdir():
        path.chmod(0o400)
    return {"artifact_manifest": ref(output / "artifact_manifest.json"), "prepared_records": ref(records),
            "report": ref(output / "preparation_report.json"), "regions": report["regions"],
            "raw_native_bytes": report["native_h0_h60_bytes_including_prompt_final"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    data = Path(args.spec).read_bytes()
    require(digest(data) == args.sha256, "spec SHA mismatch")
    print(json.dumps(run(json.loads(data)), sort_keys=True))
