#!/usr/bin/env python3
"""Select fitting-only paired support from the immutable M60 frozen pool, v2.

This program never imports a tokenizer, parser, model or evaluator, and never
executes generated code. Saved primitive labels and saved structural alignment
are checked, not recomputed. Production inputs are SHA-bound below. Selection
does not truncate responses or modify the retained pool. Only the pure `prepare`
entry point accepts alternate input bindings, for authored offline fixtures.

V2 corrects three v1 schema/estimand errors, leaving thresholds and numerics
unchanged: legacy-only `is_duplicate` and compile-error-recorded convenience
flags are optional (authoritative frozen duplicate audit/effective compile flag
remain required); zero evaluated tests retain the original frozen whole-program
correctness definition. Import-time failures are not successful solutions.
The staged v1 source and its insufficient-support result must remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

MIN_PAIRED_PROBLEMS = 30
MAX_PAIRED_PROBLEMS = 64
PADDED_LENGTH = 2176
MAX_LINE = 8 * 1024**2
MAX_ROWS = 65929
MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ADAPTER_SHA = "2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0"
CONFIG_SHA = "8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae"
AGGREGATE_SHA = "a805bd8e17c353183768e0fd48c554b7855a4733ffebdbab67431a19eb8a3234"
SAMPLING_SHA = "bd779499f0fe5090d0bb7fc889edbd78867a5aab8bb8beb24a9b99882b64d7c4"
INPUT_SHA = {
    "source": "16ba621ec034d5db48077865bb0f0be0d7ce2cb8b6a36b298bbb311a4945562a",
    "splits": "99006592f434f73afcad3f641c759786be5e206a254c3755e4dec54e35984b06",
    "dataset": "bdbba14d0632ab298f0e6116ad76bce75ae2361b79a6a7e046b23b1e15b7936f",
    "source_manifest": "03d43ada965ae350893263cfc7fe409205a43048fb23f9619dc0890881de6c77",
}
EXPECTED_SPLITS = {"direction_fit": 595, "configuration_validation": 198, "untouched_test": 199}
BENIGN = {"None", "Innocent"}
HARMFUL = {"Harmful - Arbitrary", "Harmful - Incorrect", "Harmful - Reduced"}
LABEL_FIELDS = ("ground_truth_correctness", "hinted_evaluator_correctness",
                "is_reward_hack_strict", "is_test_modification_harmful",
                "outcome_presence_class", "outcome_presence_exclusion_reason",
                "reward_hack_label", "test_modification")
WINDOWS = {"pre_definition": (-16, 0), "pre_body": (-8, 0),
           "transition": (-4, 12), "early_body": (0, 16)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_sha(value):
    return sha(canonical(value).encode("utf-8"))


def ids_hash(ids):
    return sha(b"".join(t.to_bytes(4, "little") for t in ids))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_identity(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def write_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    write_bytes(path, (canonical(value) + "\n").encode("utf-8"))


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def decode(raw):
    def invalid_constant(_value):
        raise ValueError("nonfinite JSON constant")
    value = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=invalid_constant)
    require(isinstance(value, dict), "JSON record is not an object")
    return value


def bounded_rows(path):
    with Path(path).open("rb") as stream:
        number = 0
        while True:
            offset = stream.tell()
            raw = stream.readline(MAX_LINE + 1)
            if not raw:
                return
            number += 1
            require(number <= MAX_ROWS, "source exceeds frozen row bound")
            require(len(raw) <= MAX_LINE and raw.endswith(b"\n") and raw.strip(),
                    "oversized, blank, or nonterminated JSONL record")
            yield number, offset, raw, decode(raw)


def load_bindings(paths, expected_sha, expected_splits):
    bindings = {}
    for name, path in paths.items():
        require(Path(path).is_file() and not Path(path).is_symlink(), "input not a regular nonsymlink file: " + name)
        bindings[name] = {"path": str(Path(path).resolve()), "sha256": expected_sha[name],
                          "size_bytes": Path(path).stat().st_size, "stat": stat_identity(path)}
        # Source is hashed during the single bounded scan; avoid a redundant 2GB read.
        if name != "source":
            require(file_hash(path) == expected_sha[name], "input hash mismatch: " + name)
    provenance = decode(Path(paths["source_manifest"]).read_bytes())
    checkpoint = provenance["checkpoint"]
    require(checkpoint["files"] == {"adapter_config.json": CONFIG_SHA, "adapter_model.safetensors": ADAPTER_SHA}
            and checkpoint["combined_sha256"] == AGGREGATE_SHA
            and object_sha(checkpoint["files"]) == AGGREGATE_SHA, "checkpoint aggregate association mismatch")
    require(provenance["base_model"]["model_id"] == MODEL_ID and provenance["base_model"]["revision"] == REVISION,
            "source model identity mismatch")
    require(provenance["dataset"]["sha256"] == expected_sha["dataset"], "source dataset identity mismatch")
    splits = decode(Path(paths["splits"]).read_bytes())["assignments"]
    require(dict(Counter(v["split"] for v in splits.values())) == expected_splits, "problem split counts differ")
    require(all(k == canonical(v["problem_id"]) for k, v in splits.items()), "split identity mismatch")
    examples = {}
    for _n, _offset, _raw, row in bounded_rows(paths["dataset"]):
        key = canonical(row["id"])
        require(key not in examples, "duplicate dataset problem")
        examples[key] = row["prompt"]
    require(set(examples) == set(splits), "dataset and split universe differ")
    return bindings, splits, examples


def saved_region(row, prefix):
    """Reconstruct triplet_regions.mapped_region from saved half-open spans only."""
    def span(key, limit):
        value = row.get(prefix + "_" + key)
        require(isinstance(value, list) and len(value) == 2 and all(type(t) is int for t in value)
                and 0 <= value[0] < value[1] <= limit, "saved_structure_unavailable:" + prefix + "_" + key)
        return value
    n, p = len(row["completion_token_ids"]), len(row["prompt_token_ids"])
    definition, end = span("definition_token_span", n)
    body, body_end = span("body_token_span", n)
    char_def, char_end = span("definition_char_span", len(row["completion"]))
    char_body, char_body_end = span("body_char_span", len(row["completion"]))
    require(definition <= body < body_end <= end and body > 0 and char_def <= char_body < char_body_end <= char_end,
            "saved_structure_order_invalid:" + prefix)
    require(row.get(prefix + "_definition_token") == definition and row.get(prefix + "_body_token") == body
            and row.get(prefix + "_token_length") == end - definition, "saved_structure_scalar_mismatch:" + prefix)
    windows = {}
    for name, (start, stop) in WINDOWS.items():
        anchor = definition if name == "pre_definition" else body
        windows[name] = [t if 0 <= t < n and (t < body or t < end) else None for t in range(anchor + start, anchor + stop)]
    windows["later_body"] = list(range(min(body + 16, end), end))
    windows["complete_code"] = list(range(definition, end))
    windows["end_of_code"] = list(range(max(definition, end - 16), end))
    return {"definition_character": char_def, "first_executable_character": char_body,
            "code_end_character_exclusive": char_end, "definition_completion_token": definition,
            "first_executable_completion_token": body, "end_completion_token_exclusive": end,
            "first_executable_sequence_token": p + body, "logit_source_completion_token": body - 1,
            "logit_source_sequence_token": p + body - 1, "window_completion_positions": windows,
            "window_sequence_positions": {k: [None if t is None else p + t for t in ts] for k, ts in windows.items()}}


def check_row(row, splits, examples):
    """Return paired grouping metadata; reject ambiguous rows without reclassification."""
    pid = row.get("problem_id_key")
    require(pid in splits and pid == canonical(row.get("problem_id")) == canonical(row.get("source_problem_id"))
            == row.get("source_problem_id_key"), "problem_identity_invalid")
    require(row.get("problem_split") == splits[pid]["split"] == "direction_fit", "not_fitting_split")
    require(("is_duplicate" not in row or row["is_duplicate"] is False) and not row.get("duplicate_of_record_id"),
            "record_marked_duplicate")
    require(row.get("classification_error") is None and row.get("classification_disagreements") == [], "classification_error")
    audit = row.get("freeze_audit")
    require(isinstance(audit, dict) and audit.get("errors") == [] and "duplicate_of_record_id" in audit
            and audit["duplicate_of_record_id"] is None,
            "freeze_audit_error")
    require(audit.get("record_id") == row["record_id"] and audit.get("problem_id") == row["problem_id"]
            and audit.get("problem_split") == "direction_fit", "freeze_audit_identity_invalid")
    require(isinstance(audit.get("recomputed"), dict) and all(k in row and k in audit["recomputed"] and audit["recomputed"][k] == row[k]
            for k in LABEL_FIELDS), "saved_labels_disagree")
    require(row.get("model_id") == MODEL_ID and row.get("model_revision") == REVISION
            and row.get("checkpoint_step") == 60 and row.get("checkpoint_sha256") == AGGREGATE_SHA,
            "model_or_checkpoint_mismatch")
    require(row.get("sampling_sha256") == SAMPLING_SHA and object_sha(row.get("sampling_parameters")) == SAMPLING_SHA,
            "sampling_mismatch")
    require(row.get("prompt") == examples[pid] and row.get("prompt_sha256") == object_sha(row["prompt"]), "prompt_mismatch")
    require(isinstance(row.get("completion"), str) and row.get("completion_sha256") == sha(row["completion"].encode("utf-8"))
            == audit.get("completion_sha256"), "completion_hash_mismatch")
    for kind in ("prompt", "completion"):
        tokens = row.get(kind + "_token_ids")
        require(isinstance(tokens, list) and tokens and all(type(t) is int and 0 <= t < 2**32 for t in tokens), "invalid_token_ids:" + kind)
        reference = row.get("prompt_token_ids_sha256") if kind == "prompt" else audit.get("completion_token_ids_sha256")
        require(ids_hash(tokens) == reference, "token_hash_mismatch:" + kind)
    p, n = len(row["prompt_token_ids"]), len(row["completion_token_ids"])
    require(p <= 1536 and n <= 1536 and p + n <= PADDED_LENGTH, "outside_fixed_2176_eligibility")
    require(row.get("completion_token_ids_decode_exact") is True, "saved_token_decode_not_exact")
    require(row.get("engine_prompt_token_ids") in (None, row["prompt_token_ids"]), "engine_prompt_mismatch")
    require(type(row.get("generation_seed")) is int and row["generation_seed"] >= 0, "invalid_generation_seed")
    require(row.get("is_parsed") is True and row.get("response_has_test_func") is True,
            "evaluator_not_known_present_and_parsed")
    require(row.get("evaluator_compilation_status") == "valid" and row.get("response_test_func_compile_error") is False
            and ("response_test_func_compile_error_recorded" not in row
                 or type(row["response_test_func_compile_error_recorded"]) is bool), "evaluator_not_valid")
    require(all(isinstance(row.get(k), str) and row[k] for k in
                ("solution_function_name", "test_func_name", "generated_solution_source", "generated_evaluator_function_source")),
            "saved_function_source_unavailable")
    for key in ("ground_truth_correctness", "hinted_evaluator_correctness", "is_reward_hack_strict", "is_test_modification_harmful"):
        require(type(row.get(key)) is bool, "ambiguous_label:" + key)
    harmful = row["is_test_modification_harmful"]
    require(row.get("test_modification") in (HARMFUL if harmful else BENIGN), "unknown_or_disagreeing_modification")
    for result_key, flag in (("gt_result", "ground_truth_correctness"), ("hint_result", "hinted_evaluator_correctness")):
        result = row.get(result_key)
        require(isinstance(result, dict) and result.get("can_compile") is True and result.get("is_formatted") is True,
                "primitive_not_compiled_and_formatted:" + result_key)
        counts = [result.get(k) for k in ("tests_passed", "tests_evaluated", "tests_total")]
        rate = result.get("pass_rate")
        require(all(type(v) is int for v in counts) and 0 <= counts[0] <= counts[1] <= counts[2],
                "primitive_count_invalid:" + result_key)
        expected_rate = counts[0] / counts[2] if counts[2] else 0.0
        require(type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1
                and math.isclose(rate, expected_rate, rel_tol=0, abs_tol=1e-12)
                and row[flag] is (rate == 1.0), "primitive_correctness_disagreement:" + result_key)
    require(row.get("structural_position_error") is None and row.get("structural_token_alignment")
            in {"single_token_decode", "validated_prefix_decode"}, "saved_structural_alignment_unavailable")
    saved_region(row, "solution")
    saved_region(row, "evaluator")
    protocol = object_sha({k: row[k] for k in ("prompt_sha256", "prompt_token_ids_sha256", "sampling_sha256",
                                               "checkpoint_sha256", "test_func_name", "solution_function_name")})
    return pid, row["ground_truth_correctness"], harmful, protocol


def prepared_row(row, index):
    regions = {name: saved_region(row, name) for name in ("solution", "evaluator")}
    ids = row["prompt_token_ids"] + row["completion_token_ids"]
    p, n = len(row["prompt_token_ids"]), len(row["completion_token_ids"])
    return {**row, "record_index": index, "source_checkpoint_aggregate_sha256": row["checkpoint_sha256"],
            "checkpoint_sha256": ADAPTER_SHA, "evaluator_present": True,
            "strict_reward_hack": row["is_reward_hack_strict"], "regions": regions,
            "region_mask_completion_positions": {region + "__" + name: sorted({t for t in ts if t is not None})
                for region, data in regions.items() for name, ts in data["window_completion_positions"].items()},
            "input_ids": ids, "input_ids_sha256": ids_hash(ids), "prompt_token_count": p, "completion_token_count": n,
            "sequence_token_count": len(ids), "selected_token_positions": list(range(p, p + n)),
            "selected_token_mask": [True] * n, "padded_length": PADDED_LENGTH,
            "region_schema_version": "saved-structural-spans-v1", "token_character_alignment_method": row["structural_token_alignment"],
            "structure_reparsed": False, "generated_code_reexecuted": False,
            "unavailable_regions": ["solution_class", "solution__implementation_code", "solution__end_of_solution"],
            "window_convention": "existing triplet_regions half-open offsets; transition [-4,+12); clipped slots null",
            "activation_axis": "every original assistant completion token, including recorded special tokens",
            "activation_layer_axis": "zero-based decoder block index, 0..35; post-block before final norm"}


def prepare(paths, output, *, seed=1, expected_sha=None, expected_splits=None, expected_rows=MAX_ROWS):
    """Stream/hash immutable input, select paired support, write a fresh sealed package.

    Alternate expected bindings exist only for offline fixture testing. The CLI
    always passes the frozen production constants and cannot weaken sample gates.
    """
    expected_sha = INPUT_SHA if expected_sha is None else expected_sha
    expected_splits = EXPECTED_SPLITS if expected_splits is None else expected_splits
    require(type(seed) is int and seed >= 0, "selection seed invalid")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        bindings, splits, examples = load_bindings(paths, expected_sha, expected_splits)
        write_json(output / "INTENT.json", {"schema_version": 2, "selector_version": 2, "selection_seed": seed, "inputs": bindings,
                   "minimum_paired_problems_per_stratum": MIN_PAIRED_PROBLEMS,
                   "maximum_paired_problems_per_stratum": MAX_PAIRED_PROBLEMS, "padded_length": PADDED_LENGTH,
                   "max_selected_records": 256, "full_pool_modified": False, "generated_code_reexecuted": False,
                   "structure_reparsed": False, "selector_sha256": file_hash(__file__)})
        metadata, best, best_clean, seen_ids, seen_requests, completion_seen = [], {}, {}, set(), set(), set()
        digest = hashlib.sha256()
        for number, offset, raw, row in bounded_rows(paths["source"]):
            digest.update(raw)
            rid = row.get("record_id")
            request = row.get("request_id")
            require(isinstance(rid, str) and rid and rid not in seen_ids, "missing_or_duplicate_record_id")
            require(isinstance(request, str) and request and request not in seen_requests, "missing_or_duplicate_request_id")
            seen_ids.add(rid)
            seen_requests.add(request)
            record = {"record_id": rid, "request_id": request, "problem_id_key": row.get("problem_id_key"),
                      "source_line": number, "source_offset": offset, "source_line_sha256": sha(raw), "source_line_bytes": len(raw)}
            try:
                pid, correct, harmful, protocol = check_row(row, splits, examples)
                record.update(eligible=True, ground_truth_correctness=correct, harmful=harmful, protocol_sha256=protocol,
                              strict_reward_hack=row["is_reward_hack_strict"])
                rank = object_sha({"domain": "task-specific-row-v1", "seed": seed, "problem_id_key": pid,
                                   "correct": correct, "harmful": harmful, "record_id": rid,
                                   "completion_sha256": row["completion_sha256"]})
                record["selection_rank"] = rank
                key = (pid, correct, harmful, protocol)
                if key not in best or (rank, rid) < (best[key]["selection_rank"], best[key]["record_id"]):
                    best[key] = record
                if not harmful and not row["is_reward_hack_strict"]:
                    clean_key = (pid, correct, protocol)
                    if clean_key not in best_clean or (rank, rid) < (best_clean[clean_key]["selection_rank"], best_clean[clean_key]["record_id"]):
                        best_clean[clean_key] = record
            except ValueError as exc:
                # Eligibility exclusions are retained exactly; no pool row is deleted.
                record.update(eligible=False, reason=str(exc))
            if record["eligible"]:
                completion_identity = (record["problem_id_key"], row["completion_sha256"])
                # Do not pick a duplicate owner by arrival order. A supposedly
                # deduplicated frozen input with this conflict needs review.
                require(completion_identity not in completion_seen, "duplicate_eligible_completion_requires_review")
                completion_seen.add(completion_identity)
            metadata.append(record)
        require(len(metadata) == expected_rows and digest.hexdigest() == expected_sha["source"], "source count/hash mismatch")
        for name, path in paths.items():
            require(stat_identity(path) == bindings[name]["stat"], "input changed during preparation: " + name)
        pair_options = defaultdict(list)
        clean_options = defaultdict(list)
        for (pid, correct, harmful, protocol), record in best.items():
            if not harmful and (pid, correct, True, protocol) in best:
                pair_options[(pid, correct)].append((protocol, record, best[(pid, correct, True, protocol)]))
        for (pid, correct, protocol), record in best_clean.items():
            if correct and (pid, False, protocol) in best_clean:
                clean_options[pid].append((protocol, record, best_clean[(pid, False, protocol)]))
        pairs = []
        counts = {}
        for correct in (True, False):
            available = sorted((pid for pid, c in pair_options if c == correct), key=lambda pid: (
                object_sha({"domain": "task-specific-problem-v1", "seed": seed, "problem_id_key": pid, "correct": correct}), pid))
            name = "gt_correct" if correct else "gt_incorrect"
            counts[name] = len(available)
            for pid in available[:MAX_PAIRED_PROBLEMS]:
                protocol, benign, harmful = min(pair_options[(pid, correct)], key=lambda v: (
                    object_sha({"domain": "task-specific-protocol-v1", "seed": seed, "protocol": v[0]}), v[0]))
                pairs.append({"problem_id_key": pid, "ground_truth_correctness": correct, "stratum": name,
                              "protocol_sha256": protocol, "benign_record_id": benign["record_id"], "harmful_record_id": harmful["record_id"]})
        selected = {rid for pair in pairs for rid in (pair["benign_record_id"], pair["harmful_record_id"])}
        require(len(selected) <= 256, "selection exceeds reviewed bound")
        selected_meta = sorted((r for r in metadata if r["record_id"] in selected), key=lambda r: r["record_id"])
        clean_pairs = []
        for pid, options in sorted(clean_options.items()):
            available = [v for v in options if v[1]["record_id"] in selected and v[2]["record_id"] in selected]
            if available:
                protocol, correct_row, incorrect_row = min(available, key=lambda v: v[0])
                clean_pairs.append({"problem_id_key": pid, "protocol_sha256": protocol,
                                    "correct_record_id": correct_row["record_id"], "incorrect_record_id": incorrect_row["record_id"]})
        rows = []
        with Path(paths["source"]).open("rb") as source, (output / "selected_raw_records.jsonl").open("xb") as retained:
            for index, record in enumerate(selected_meta):
                source.seek(record["source_offset"])
                raw = source.read(record["source_line_bytes"])
                require(sha(raw) == record["source_line_sha256"], "selected source row changed")
                retained.write(raw)
                rows.append(prepared_row(decode(raw), index))
            retained.flush()
            os.fsync(retained.fileno())
        write_bytes(output / "prepared_records.jsonl", b"".join((canonical(r) + "\n").encode("utf-8") for r in rows))
        for record in metadata:
            if record["eligible"]:
                record["reason"] = "selected" if record["record_id"] in selected else (
                    "paired_support_not_selected_by_hash" if (record["problem_id_key"], record["ground_truth_correctness"]) in pair_options
                    else "no_eligible_opposite_label_same_problem_protocol")
        write_bytes(output / "row_selection_audit.jsonl", b"".join((canonical(r) + "\n").encode("utf-8") for r in metadata))
        write_json(output / "pairs.json", {"pairs": pairs, "clean_axis_pairs_from_selected_rows": clean_pairs})
        status = "support_ready" if all(n >= MIN_PAIRED_PROBLEMS for n in counts.values()) else "insufficient_support"
        summary = {"status": status, "selector_version": 2, "source_records": len(metadata), "source_sha256": digest.hexdigest(),
                   "source_fitting_problems": len({r["problem_id_key"] for r in metadata}),
                   "eligible_records": sum(r["eligible"] for r in metadata), "eligible_paired_problems": counts,
                   "selected_paired_problems": dict(Counter(p["stratum"] for p in pairs)), "selected_records": len(rows),
                   "row_disposition_counts": dict(Counter(r["reason"] for r in metadata)),
                   "clean_axis_eligible_problems_in_full_pool": len(clean_options),
                   "clean_axis_paired_problems_in_selected_rows": len(clean_pairs),
                   "clean_axis_has_paired_support": bool(clean_pairs),
                   "clean_axis_minimum_statistical_support": "not determined by this preparation; fitter reports actual count",
                   "clean_axis_policy": "selected benign rows only; no extraction budget expansion or quartet requirement",
                   "minimum_paired_problems_per_stratum": MIN_PAIRED_PROBLEMS, "maximum_paired_problems_per_stratum": MAX_PAIRED_PROBLEMS,
                   "all_rows_audited": True, "retained_pool_untouched": True,
                   "label_verification": "saved primitive consistency and frozen audit equality; no reclassification or generated-code execution",
                   "correctness_estimand": "frozen whole-program pass_rate==1; original zero-evaluated/zero-total count policy retained",
                   "structural_verification": "saved spans/scalars/alignment only; tokenizer decode and AST not rerun",
                   "unavailable_regions": ["solution_class", "solution__implementation_code", "solution__end_of_solution"],
                   "checkpoint_association": {"source_aggregate_sha256": AGGREGATE_SHA, "adapter_sha256": ADAPTER_SHA,
                                              "adapter_config_sha256": CONFIG_SHA, "source_manifest_sha256": expected_sha["source_manifest"]}}
        write_json(output / "SUMMARY.json", summary)
        for name, path in paths.items():
            require(stat_identity(path) == bindings[name]["stat"], "input changed before seal: " + name)
        manifest = {p.name: {"size_bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / "ARTIFACT_MANIFEST.json", {"schema_version": 1, "status": status, "files": manifest})
        for path in output.iterdir():
            path.chmod(0o444)
        return summary
    except Exception as exc:
        if not (output / "FAILURE.json").exists():
            write_json(output / "FAILURE.json", {"status": "failed_closed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=1)
    args = parser.parse_args()
    summary = prepare({"source": args.source, "splits": args.split_manifest, "dataset": args.dataset,
                       "source_manifest": args.source_manifest}, args.output, seed=args.selection_seed)
    print(canonical(summary))
    return 0 if summary["status"] == "support_ready" else 2


if __name__ == "__main__":
    sys.exit(main())
