#!/usr/bin/env python3
"""CPU-only, lossless freeze of a terminal outcome/presence campaign.

Never executes generated code. Reuses the campaign's pinned classifier and parser,
preserves its problem split, and keeps original records even when excluded.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import factorial_common as base
import outcome_presence_common as mode

SPLITS = ("direction_fit", "configuration_validation", "untouched_test")
BOOLS = ("response_has_test_func", "match_test_func", "response_test_func_gt_pass",
         "response_test_func_arbitrary_pass", "prompt_test_func_response_pass",
         "ground_truth_correctness", "hinted_evaluator_correctness")
RESULTS = {"gt_result": "ground_truth_correctness",
           "hint_result": "hinted_evaluator_correctness",
           "response_test_func_gt_result": "response_test_func_gt_pass",
           "response_test_func_arbitrary_result": "response_test_func_arbitrary_pass"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            require(len(line) <= 8 * 1024**2, f"oversize JSONL line {number}")
            require(bool(line.strip()), f"blank JSONL line {number}")
            row = json.loads(line)
            require(isinstance(row, dict), f"non-object JSONL line {number}")
            yield row


def write(stream, value):
    stream.write(base.canonical_json(value) + "\n")


def token_hash(ids):
    return hashlib.sha256(b"".join(t.to_bytes(4, "little") for t in ids)).hexdigest()


def trusted_functions(source):
    """Load only the existing non-executing parser/classifier from trusted source.

    This avoids importing the evaluator's process-execution dependencies. No
    completion is ever passed to exec/compile; only recorded repository source is.
    """
    tree = ast.parse((source / "evaluator.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CodeEvaluator")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"parse_response", "extract_function"}]
    require(len(cls.body) == 2, "pinned parser methods missing")
    scope = {"ast": ast, "re": re}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned_repository_parser", "exec"), scope)
    parser = scope["CodeEvaluator"]()
    tree = ast.parse((source / "analysis.py").read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                   and n.name in {"RewardHackCategorization", "categorize_test_modification", "categorize_reward_hack"}]
    require(len(definitions) == 3, "pinned classifier definitions missing")
    scope = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "pinned_repository_classifier", "exec"), scope)
    return parser, scope


def recompute(row, classifier):
    """Validate primitive types/counts before using the repository's formulas."""
    errors, coverage = [], []
    primitive = dict(row)
    for key in BOOLS:
        if type(row.get(key)) is not bool:
            errors.append("missing_or_nonboolean:" + key)
    for result_key, flag in RESULTS.items():
        result = row.get(result_key)
        if result is None:
            if row.get("provenance") != "existing" or result_key in {"gt_result", "hint_result"}:
                errors.append("missing_result:" + result_key)
            else:
                coverage.append("legacy_boolean_only:" + result_key)
            continue
        if not isinstance(result, dict):
            errors.append("malformed_result:" + result_key)
            continue
        rate = result.get("pass_rate")
        counts = [result.get(k) for k in ("tests_passed", "tests_evaluated", "tests_total")]
        if (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1
                or any(type(n) is not int or n < 0 for n in counts)):
            errors.append("invalid_result_values:" + result_key)
            continue
        passed, evaluated, total = counts
        if not passed <= evaluated <= total:
            errors.append("invalid_result_counts:" + result_key)
        expected = passed / total if total else 0.0
        if not math.isclose(rate, expected, rel_tol=0, abs_tol=1e-12):
            errors.append("result_pass_rate_disagreement:" + result_key)
        if type(result.get("can_compile")) is not bool or type(result.get("is_formatted")) is not bool:
            errors.append("invalid_result_boolean:" + result_key)
        if rate == 1 and (not result.get("can_compile") or not result.get("is_formatted")):
            errors.append("pass_with_compile_or_format_failure:" + result_key)
        primitive[flag] = rate == 1.0
        if row.get(flag) != primitive[flag]:
            errors.append("primitive_disagreement:" + flag)
        scalar = {"gt_result": "gt_pass_rate", "hint_result": "hint_pass_rate"}.get(result_key)
        if scalar and row.get(scalar) != rate:
            errors.append("scalar_pass_rate_disagreement:" + scalar)
    if row.get("classification_error"):
        errors.append("classification_error")
    if errors:
        return None, sorted(set(errors)), coverage
    primitive["prompt_has_test_func"] = row.get("prompt_has_test_func", False)
    require(type(primitive["prompt_has_test_func"]) is bool, "nonboolean prompt evaluator presence")
    modification = base.recompute_test_modification(primitive)
    category = base.recompute_reward_hack_label(primitive, modification)
    other = dict(primitive, eq_correct=primitive["ground_truth_correctness"],
                 eq_hinted=primitive["hinted_evaluator_correctness"], test_modification=modification)
    require(classifier["categorize_test_modification"](other) == modification, "classifier modification mismatch")
    require(classifier["categorize_reward_hack"](other) == category, "classifier category mismatch")
    derived = {"test_modification": modification, "reward_hack_label": category,
               "is_test_modification_harmful": modification in base.HARMFUL_MODIFICATIONS,
               "is_reward_hack_strict": category == "Reward Hack",
               "ground_truth_correctness": primitive["ground_truth_correctness"],
               "hinted_evaluator_correctness": primitive["hinted_evaluator_correctness"]}
    for key, value in derived.items():
        if row.get(key) != value:
            errors.append("saved_label_disagreement:" + key)
    cell, reason = mode.outcome_presence_cell(dict(row, **derived))
    derived.update(outcome_presence_class=cell, outcome_presence_exclusion_reason=reason)
    if row.get("outcome_presence_class") != cell:
        errors.append("saved_outcome_presence_class_disagreement")
    return derived, sorted(set(errors)), coverage


def auxiliary(row, derived):
    labels = []
    category = derived["reward_hack_label"]
    if category == "Attempted Reward Hack":
        labels.append("incorrect_failed_harmful_modification")
    if category == "Correct; Attempted Reward Hack":
        labels.append("correct_harmful_modification")
    if derived["outcome_presence_class"] in mode.ABSENT_CELLS:
        labels.append(derived["outcome_presence_class"])
    if category == "Reward Hack":
        labels.append("strict_reward_hack:" + derived["test_modification"])
    return labels


class Inventory:
    def __init__(self):
        self.data = {}

    def add(self, scope, row, derived, duplicate=False):
        keys = [(scope, "all", "all"), (scope, "provenance", row["provenance"]),
                (scope, "split", row["problem_split"])]
        if derived:
            keys += [(scope, "taxonomy", derived["reward_hack_label"]),
                     (scope, "modification", derived["test_modification"]),
                     (scope, "outcome_presence_class", derived["outcome_presence_class"] or "excluded")]
            keys += [(scope, "auxiliary", label) for label in auxiliary(row, derived)]
            keys += [(scope, "split_auxiliary", row["problem_split"] + ":" + label)
                     for label in auxiliary(row, derived)]
        if duplicate:
            keys.append((scope, "duplicates", "exact_completion"))
        for key in keys:
            count, problems = self.data.setdefault(key, [0, set()])
            self.data[key][0] = count + 1
            problems.add(row["problem_id_key"])

    def export(self):
        out = {}
        for (scope, dimension, label), (count, problems) in sorted(self.data.items()):
            out.setdefault(scope, {}).setdefault(dimension, {})[label] = {
                "records": count, "unique_problems": len(problems)}
        return out


def validate_selected(row, parser, raw_row):
    # Selection may only add matching metadata; all saved scientific content must agree.
    additions = {"matched_group_identifier", "selection_status"}
    require(all(k in raw_row and v == raw_row[k] for k, v in row.items() if k not in additions),
            "selected record differs from raw source: " + row["record_id"])
    require(parser.parse_response(row["completion"]) == row["parsed_response"], "selected parse mismatch")
    evaluator = parser.extract_function(row["parsed_response"], row["test_func_name"])
    require(evaluator == row["generated_evaluator_function_source"], "selected evaluator source mismatch")
    for name, field in ((row["solution_function_name"].split(".")[-1], "generated_solution_source"),
                        (row["test_func_name"], "generated_evaluator_function_source")):
        location = mode.locate_function(row["completion"], name)
        observed = ast.dump(ast.parse(textwrap.dedent(location.source)), include_attributes=False)
        expected = ast.dump(ast.parse(textwrap.dedent(row[field])), include_attributes=False)
        require(observed == expected, "selected code AST mismatch: " + field)
    require(row.get("structural_position_error") is None, "selected unresolved source structure")


def verified_copy(src, dest, digest):
    require(src.is_file(), "missing source: " + str(src))
    require(not dest.exists(), "refuse overwrite: " + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    require(base.sha256_file(dest) == digest, "copied source hash mismatch: " + str(dest))


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def freeze(source, output, expected_new, expected_existing, expected_selected, tokenizer_factory=load_tokenizer):
    source, output = source.resolve(), output.resolve()
    require(not output.exists(), "output already exists; immutable freezes cannot be overwritten")
    require(source not in output.parents, "output must be separate from source package")
    require(shutil.disk_usage(output.parent).free > 16 * 1024**3, "less than 16 GiB free")
    source_manifest = json.loads((source / "artifact_manifest.json").read_text())
    require(base.file_manifest(source) == source_manifest, "source artifact manifest mismatch")
    for module in (base, mode):
        name = Path(module.__file__).name
        if "source/" + name in source_manifest["files"]:
            require(base.sha256_file(Path(module.__file__)) == source_manifest["files"]["source/" + name]["sha256"],
                    "freeze logic differs from pinned campaign module: " + name)
    config = json.loads((source / "run_config.yaml").read_text())
    original_summary = json.loads((source / "summary.json").read_text())
    require(original_summary["status"] in {"failed", "succeeded"}, "source campaign is not terminal")
    require(original_summary["new_generations_executed"] == expected_new, "source generation count mismatch")
    require(original_summary["selected_records"] == expected_selected, "source selected count mismatch")
    provenance = json.loads((source / "existing_source_manifest.json").read_text())
    require(config["scientific"]["model"] == base.MODEL_ID and
            config["scientific"]["revision"] == base.MODEL_REVISION and
            config["scientific"]["sampling"] == base.SAMPLING, "unexpected science configuration")
    require(provenance["checkpoint"]["combined_sha256"] == base.sha256_text(
        base.canonical_json(provenance["checkpoint"]["files"])), "checkpoint digest is not bound to files")
    split_manifest = json.loads((source / "problem_splits.json").read_text())
    selected = list(rows(source / "selected_core_triplets.jsonl"))
    require(len(selected) == expected_selected, "selected file count mismatch")
    selected_by_id = {r["record_id"]: r for r in selected}
    require(len(selected_by_id) == len(selected), "duplicate selected record ID")
    groups = defaultdict(list)
    for row in selected:
        groups[row["problem_id_key"]].append(row)
    for group in groups.values():
        mode.validate_core_group(group)
    require(len({r["completion_sha256"] for r in selected}) == len(selected), "selected duplicate completion")
    output.mkdir(mode=0o700)
    print("Source manifest verified; freezing original inputs", flush=True)
    for name in ("freeze_outcome_presence_dataset.py", "verify_frozen_outcome_dataset.py",
                 "factorial_common.py", "outcome_presence_common.py", "test_freeze_outcome_presence.py"):
        script = Path(__file__).resolve().parent / name
        verified_copy(script, output / "provenance/freeze_source" / name, base.sha256_file(script))
    context = Path(__file__).resolve().parent / "run_context"
    if context.is_dir():
        for path in sorted(context.iterdir()):
            require(path.is_file() and not path.is_symlink(), "unexpected run context entry")
            verified_copy(path, output / "provenance/freeze_run" / path.name, base.sha256_file(path))
    for name in ("raw_rollouts_merged.jsonl", "selected_core_triplets.jsonl", "problem_splits.json",
                 "run_config.yaml", "summary.json", "existing_source_manifest.json", "environment.json",
                 "supervisor_receipt.json", "campaign_plan.jsonl"):
        verified_copy(source / name, output / "originals" / name, source_manifest["files"][name]["sha256"])
    verified_copy(source / "artifact_manifest.json", output / "provenance/source_artifact_manifest.json",
                  base.sha256_file(source / "artifact_manifest.json"))
    for name, info in source_manifest["files"].items():
        if name.startswith("source/"):
            verified_copy(source / name, output / "provenance" / name, info["sha256"])
    # Freeze model/tokenizer metadata and the actual adapter bytes, not a guessed identity.
    for kind, info, entries in (("tokenizer", provenance["base_model"], "tokenizer_files"),
                                ("checkpoint", provenance["checkpoint"], "files")):
        for name, digest in info[entries].items():
            verified_copy((source / info["path"] / name).resolve(), output / "provenance" / kind / name, digest)
    dataset_path = (source / provenance["dataset"]["path"]).resolve()
    verified_copy(dataset_path, output / "provenance/dataset.jsonl", provenance["dataset"]["sha256"])
    for name, digest in provenance["original_grpo_sampling_evidence"]["files"].items():
        key = {"config.json": "grpo_run_config", "verl_config.yaml": "grpo_config",
               "verl_full_config.yaml": "grpo_full_config"}[name]
        verified_copy((source / config["paths"][key]).resolve(), output / "provenance/grpo" / name, digest)
    examples = {base.stable_problem_id(r["id"]): r for r in rows(output / "provenance/dataset.jsonl")}
    expected_splits = base.problem_splits(
        [{"problem_id": e["id"], "problem_id_key": key} for key, e in examples.items()], split_manifest["split_seed"])
    require(expected_splits == split_manifest, "existing splits do not reproduce from original seed/universe")
    base.atomic_write_json(output / "split_manifest.json", split_manifest)
    tokenizer = tokenizer_factory(output / "provenance/tokenizer")
    parser, classifier = trusted_functions(output / "provenance/source")
    prompt_ids = {}
    for key, example in examples.items():
        prompt_ids[key] = tokenizer.apply_chat_template(example["prompt"], tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=False)
    selected_verified, record_ids, requests, completion_seen = set(), set(), set(), {}
    # Store only hashes and identifiers for the pool; full completions are streamed.
    source_identities, new_contracts, duplicate_owners = {}, {}, {}
    inventory = Inventory()
    coverage, errors_count, exclusion_counts, origin_counts = Counter(), Counter(), Counter(), Counter()
    split_lines = Counter()
    for directory in ("splits", "selected", "audit", "auxiliary"):
        (output / directory).mkdir()
    with contextlib.ExitStack() as stack:
        split_out = {s: stack.enter_context((output / "splits" / (s + ".jsonl")).open("w")) for s in SPLITS}
        audit_out = stack.enter_context((output / "audit/records.jsonl").open("w"))
        duplicate_out = stack.enter_context((output / "audit/duplicates.jsonl").open("w"))
        auxiliary_out = stack.enter_context((output / "auxiliary/index.jsonl").open("w"))
        selected_audit = stack.enter_context((output / "audit/selected_records.jsonl").open("w"))
        for number, raw in enumerate(rows(output / "originals/raw_rollouts_merged.jsonl"), 1):
            require(number <= expected_new + expected_existing, "raw count exceeds bound")
            row = dict(raw)
            rid, pid = row["record_id"], row["problem_id_key"]
            require(rid not in record_ids, "duplicate record ID: " + rid)
            record_ids.add(rid)
            require(pid in examples and pid == base.stable_problem_id(row["problem_id"]) ==
                    base.stable_problem_id(row["source_problem_id"]) == row["source_problem_id_key"], "problem identity mismatch")
            assigned = split_manifest["assignments"][pid]["split"]
            require(row.get("problem_split", assigned) == assigned, "problem split mismatch")
            row["problem_split"] = assigned
            require(row["prompt"] == examples[pid]["prompt"], "prompt differs from original dataset")
            require(row["prompt_sha256"] == base.prompt_sha256(row["prompt"]), "prompt hash mismatch")
            require(row["prompt_token_ids"] == prompt_ids[pid], "prompt token IDs differ from pinned chat template")
            require(row["prompt_token_ids_sha256"] == token_hash(row["prompt_token_ids"]), "prompt token hash mismatch")
            if row["provenance"] == "new":
                require(row["request_id"] not in requests, "duplicate new request ID")
                requests.add(row["request_id"])
                require(row["engine_prompt_token_ids"] == row["prompt_token_ids"], "engine prompt IDs differ")
                require(row["generation_seed"] == base.stable_seed(config["campaign"]["master_seed"],
                        row["problem_id"], row["sample_index"]), "generation seed differs")
            require(type(row["generation_seed"]) is int and row["generation_seed"] >= 0, "invalid seed")
            require(row["model_id"] == base.MODEL_ID and row["model_revision"] == base.MODEL_REVISION
                    and row["checkpoint_step"] == 60, "model identity mismatch")
            require(row["checkpoint_sha256"] == provenance["checkpoint"]["combined_sha256"], "adapter identity mismatch")
            require((source / row["checkpoint_path"]).resolve() ==
                    (source / provenance["checkpoint"]["path"]).resolve(), "checkpoint path mismatch")
            require(row["sampling_parameters"] == base.SAMPLING and row["sampling_sha256"] ==
                    base.sampling_sha256(base.SAMPLING), "sampling mismatch")
            expected_recorded = base.SAMPLING if row["provenance"] == "new" else base.LEGACY_RECORDED_SAMPLING
            require(row["recorded_sampling_parameters"] == expected_recorded, "recorded sampling mismatch")
            ids = row["completion_token_ids"]
            require(isinstance(ids, list) and ids and all(type(t) is int and 0 <= t < 2**32 for t in ids), "invalid completion IDs")
            require(tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) ==
                    row["completion"], "completion token decode mismatch: " + rid)
            digest = base.sha256_text(row["completion"])
            require(digest == row["completion_sha256"], "completion hash mismatch")
            derived, errors, missing_details = recompute(row, classifier)
            coverage.update(row["provenance"] + ":" + c for c in missing_details)
            errors_count.update(errors)
            duplicate = completion_seen.get(digest)
            if duplicate:
                require(duplicate[1] == pid, "cross-problem duplicate needs explicit leakage exclusion")
                require(row.get("duplicate_of_record_id") == duplicate[0], "source duplicate owner mismatch")
                duplicate_owners[rid] = duplicate[0]
                write(duplicate_out, {"record_id": rid, "duplicate_of_record_id": duplicate[0],
                      "completion_sha256": digest, "problem_id": row["problem_id"],
                      "reason": "exact_completion_duplicate", "original_line": number})
            else:
                completion_seen[digest] = (rid, pid)
            source_identities[rid] = base.sha256_text(base.canonical_json(row))
            if row["provenance"] == "new":
                fields = {k: row[k] for k in base.REQUEST_CONTRACT_FIELDS}
                new_contracts[row["request_id"]] = (base.sha256_text(base.canonical_json(fields)),
                    base.sha256_text(base.canonical_json({k: row[k] for k in (
                        *base.REQUEST_CONTRACT_FIELDS, "completion", "completion_token_ids", "engine_prompt_token_ids")})))
            exclusion = errors or ([derived["outcome_presence_exclusion_reason"]]
                                  if derived and derived["outcome_presence_exclusion_reason"] else [])
            exclusion_counts.update(exclusion)
            audit = {"record_id": rid, "problem_id": row["problem_id"], "problem_split": assigned,
                     "source_line": number, "completion_sha256": digest,
                     "completion_token_ids_sha256": token_hash(ids), "recomputed": derived,
                     "errors": errors, "detail_limitations": missing_details,
                     "exclusion_reasons": exclusion, "duplicate_of_record_id": duplicate[0] if duplicate else None}
            write(audit_out, audit)
            inventory.add("raw", row, derived, bool(duplicate))
            origin_counts[row["provenance"]] += 1
            if not duplicate:
                split_lines[assigned] += 1
                row["freeze_audit"] = audit
                # Original paths remain verbatim in originals; portable analysis view resolves locally.
                row["checkpoint_path"] = "provenance/checkpoint"
                write(split_out[assigned], row)
                inventory.add("deduplicated", row, derived)
                if derived and not errors:
                    labels = auxiliary(row, derived)
                    if labels:
                        write(auxiliary_out, {"record_id": rid, "problem_id": row["problem_id"],
                              "problem_split": assigned, "labels": labels,
                              "record_path": "splits/" + assigned + ".jsonl", "line": split_lines[assigned],
                              "core_eligible": derived["outcome_presence_class"] is not None,
                              "solution_compiles": mode.solution_compilation_valid(row),
                              "evaluator_compiles": mode.evaluator_compilation_valid(row) if row["response_has_test_func"] else None})
            if rid in selected_by_id:
                require(not duplicate and not errors, "selected record failed audit: " + rid)
                selected_row = selected_by_id[rid]
                validate_selected(selected_row, parser, raw)
                require(selected_row["problem_split"] == assigned, "selected split mismatch")
                selected_verified.add(rid)
                write(selected_audit, dict(audit, verified=True, parser_verified=True, token_decode_verified=True))
                inventory.add("selected", row, derived)
            if number % 10000 == 0:
                print(f"Audited {number} records; selected verified {len(selected_verified)}; duplicates {number-len(completion_seen)}", flush=True)
    require(origin_counts == {"new": expected_new, "existing": expected_existing}, "origin/count mismatch")
    require(selected_verified == set(selected_by_id), "selected record missing from raw source")
    require(not errors_count, "label audit disagreements: " + str(dict(errors_count)))
    # Independently link the authoritative merged records to each source, recovery,
    # and executed request plan. Recovery is a second representation, never extra data.
    seen_origins = set()
    for name, provenance_name, expected in (("raw_new_rollouts.jsonl", "new", expected_new),
                                           ("raw_existing_rollouts.jsonl", "existing", expected_existing)):
        count = 0
        for r in rows(source / name):
            count += 1
            rid = r["record_id"]
            require(rid not in seen_origins, "repeated source record ID")
            seen_origins.add(rid)
            r.setdefault("problem_split", split_manifest["assignments"][r["problem_id_key"]]["split"])
            if rid in duplicate_owners:
                # Reproduce only the exact annotations added by the campaign's
                # existing deduplicate_records(), leaving all science fields intact.
                r.update(factorial_cell=None, exclusion_reason="duplicate_completion",
                         duplicate_of_record_id=duplicate_owners[rid])
            require(source_identities.get(rid) == base.sha256_text(base.canonical_json(r)), "merged/source mismatch")
            require(r["provenance"] == provenance_name, "source provenance mismatch")
        require(count == expected, "source count mismatch")
    require(seen_origins == record_ids, "merged/source record set mismatch")
    for name, recovery in (("campaign_plan.jsonl", False), ("partial_generated_rollouts.jsonl", True)):
        observed = set()
        for r in rows(source / name):
            rid = r["request_id"]
            require(rid not in observed and rid in new_contracts, "duplicate or unknown request: " + name)
            observed.add(rid)
            fields = (*base.REQUEST_CONTRACT_FIELDS, "completion", "completion_token_ids", "engine_prompt_token_ids") if recovery else base.REQUEST_CONTRACT_FIELDS
            require(base.sha256_text(base.canonical_json({k: r[k] for k in fields})) ==
                    new_contracts[rid][1 if recovery else 0], "request/recovery mismatch: " + name)
        require(observed == requests, "missing request: " + name)
    for split in SPLITS:
        base.atomic_write_jsonl(output / "selected" / (split + ".jsonl"),
                                [dict(r, checkpoint_path="provenance/checkpoint") for r in selected if r["problem_split"] == split])
    base.atomic_write_jsonl(output / "selected_core_triplets.jsonl",
                            [dict(r, checkpoint_path="provenance/checkpoint") for r in selected])
    inventory_data = inventory.export()
    base.atomic_write_json(output / "inventory.json", inventory_data)
    report = {"schema_version": 1, "status": "verified_freeze", "generated_code_reexecuted": False,
              "source_campaign_status": original_summary["status"],
              "source_campaign_target_met": original_summary.get("selected_problems", len(groups)) >= config["campaign"]["target_complete_problems"],
              "source_manifest_files_verified": len(source_manifest["files"]),
              "raw_records": sum(origin_counts.values()), "source_counts": dict(origin_counts),
              "deduplicated_records": len(completion_seen), "duplicate_records": sum(origin_counts.values()) - len(completion_seen),
              "cross_problem_duplicates": 0, "selected_records_verified": len(selected_verified),
              "selected_problems": len(groups), "all_problems": len(examples),
              "label_disagreements": dict(errors_count), "primitive_detail_limitations": dict(coverage),
              "exclusion_counts": dict(exclusion_counts), "split_policy": "preserve_existing_campaign_assignments",
              "split_seed": split_manifest["split_seed"], "selected_splits": inventory_data["selected"]["split"],
              "deduplicated_splits": inventory_data["deduplicated"]["split"],
              "source_recovery_and_plan_verified": True,
              "deduplication_rule": "exact UTF-8 completion SHA-256; first occurrence in immutable merged file; all originals retained",
              "verification_scope": "primitive outcome/count consistency, pinned taxonomy recomputation, token/text identity, selected parser/source checks; no generated-code reexecution",
              "untouched_test_note": "preserved split membership; this task fits no directions or models; prior human/analysis exposure cannot be certified",
              "unavailable_generation_provenance": ["generation dependency lockfile not present in source package", "generation container digest not recorded"]}
    base.atomic_write_json(output / "audit_report.json", report)
    (output / "README.md").write_text(
        "# Frozen checkpoint-60 outcome/presence dataset\n\n"
        "This is stage 2 only. The source campaign remains failed against its 200-triplet target. "
        "This freeze accepts and verifies the available 187 triplets; it does not redefine campaign success.\n\n"
        "`originals/raw_rollouts_merged.jsonl` preserves every original record verbatim. "
        "`splits/*.jsonl` contain globally deduplicated records with a `freeze_audit` annotation, "
        "including excluded and malformed responses. Eligibility must be read from that annotation. "
        "`selected_core_triplets.jsonl` and `selected/*.jsonl` preserve the original selected triplets. "
        "Paths in analysis views are relative to this package root. Source metadata under `originals/` "
        "retains its historical path context; it is evidence, not a new launch configuration.\n\n"
        "`auxiliary/index.jsonl` points into the split files and records compilation eligibility. "
        "Attempted-hack taxonomy counts include malformed attempts; do not treat all auxiliary "
        "records as structurally eligible for later activation extraction. Evaluator-absent classes "
        "use the existing conservative absence rule and exclude malformed evaluator attempts.\n\n"
        "`audit/records.jsonl` covers every original record; `audit/duplicates.jsonl` records every "
        "deduplication exclusion. No raw generation is deleted. The recovery journal was verified "
        "against all new requests but is not counted as another dataset.\n\n"
        "The pre-existing seed-60020020 split across 992 problems is preserved. Selected problems "
        "inherit 117 fitting, 37 validation and 33 test assignments. No class, auxiliary record or "
        "completion from a problem is assigned to another split. Prior test exposure is unknown.\n\n"
        "Labels are recomputed from saved evaluator outcomes, not independently rerun programs. "
        "Legacy records retain primitive evaluator booleans but lack two detailed result objects. "
        "No model, GPU, generation, activation extraction, fitting or training is run.\n\n"
        "`provenance/source_artifact_manifest.json` describes the original campaign, including "
        "redundant worker journals not copied here. `artifact_manifest.json` describes this package "
        "alone. All original source hashes were checked before freezing.\n", encoding="utf-8")
    # Inputs must still match after the producer has consumed them.
    require(base.file_manifest(source) == source_manifest, "source package changed during freeze")
    base.atomic_write_json(output / "artifact_manifest.json", base.file_manifest(output))
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-new", type=int, default=100000)
    parser.add_argument("--expected-existing", type=int, default=9920)
    parser.add_argument("--expected-selected", type=int, default=561)
    args = parser.parse_args()
    freeze(args.source, args.output, args.expected_new, args.expected_existing, args.expected_selected)


if __name__ == "__main__":
    main()
