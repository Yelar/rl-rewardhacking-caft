#!/usr/bin/env python3
"""Independent process verification of the frozen dataset and its SHA-256 manifest."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import factorial_common as base
import outcome_presence_common as mode


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    require(base.file_manifest(root) == manifest, "frozen artifact manifest mismatch")
    report = json.loads((root / "audit_report.json").read_text())
    require(report["status"] == "verified_freeze" and report["label_disagreements"] == {}, "freeze not verified")
    splits = json.loads((root / "split_manifest.json").read_text())["assignments"]
    original = {}
    completion_owners = {}
    expected_unique = set()
    expected_duplicates = {}
    origins = Counter()
    for row in base.read_jsonl(root / "originals/raw_rollouts_merged.jsonl"):
        rid, pid, digest = row["record_id"], row["problem_id_key"], row["completion_sha256"]
        require(rid not in original, "repeated original ID")
        require(digest == hashlib.sha256(row["completion"].encode()).hexdigest(), "original completion hash mismatch")
        row.setdefault("problem_split", splits[pid]["split"])
        require(row["problem_split"] == splits[pid]["split"], "original split mismatch")
        origins[row["provenance"]] += 1
        if digest in completion_owners:
            require(completion_owners[digest][1] == pid, "cross-problem duplicate")
            expected_duplicates[rid] = completion_owners[digest][0]
        else:
            completion_owners[digest] = (rid, pid)
            expected_unique.add(rid)
        row["checkpoint_path"] = "provenance/checkpoint"
        original[rid] = base.sha256_text(base.canonical_json(row))
    require(len(original) == report["raw_records"] and dict(origins) == report["source_counts"], "raw counts disagree")
    observed = set()
    split_problems = defaultdict(set)
    split_counts = Counter()
    locators = {}
    for name in ("direction_fit", "configuration_validation", "untouched_test"):
        for number, row in enumerate(base.read_jsonl(root / "splits" / (name + ".jsonl")), 1):
            rid, pid = row["record_id"], row["problem_id_key"]
            require(rid not in observed, "deduplicated ID occurs twice")
            require(row["problem_split"] == name == splits[pid]["split"], "split leakage")
            audit = row.pop("freeze_audit")
            require(audit["errors"] == [] and audit["duplicate_of_record_id"] is None, "retained audit failed")
            require(audit["recomputed"]["test_modification"] == base.recompute_test_modification(row), "recomputed modification differs")
            require(audit["recomputed"]["reward_hack_label"] == base.recompute_reward_hack_label(row, row["test_modification"]), "recomputed taxonomy differs")
            require(base.sha256_text(base.canonical_json(row)) == original[rid], "analysis view changed source content")
            observed.add(rid)
            split_problems[name].add(pid)
            split_counts[name] += 1
            locators[rid] = ("splits/" + name + ".jsonl", number, row["problem_id"], name)
    require(observed == expected_unique and len(observed) == report["deduplicated_records"], "deduplicated IDs/count disagree")
    all_problem_ids = [pid for ids in split_problems.values() for pid in ids]
    require(len(all_problem_ids) == len(set(all_problem_ids)) == report["all_problems"], "problem split overlap/coverage mismatch")
    duplicates = {r["record_id"]: r["duplicate_of_record_id"] for r in base.read_jsonl(root / "audit/duplicates.jsonl")}
    require(duplicates == expected_duplicates and len(duplicates) == report["duplicate_records"], "duplicate index differs")
    audited = set()
    for row in base.read_jsonl(root / "audit/records.jsonl"):
        rid = row["record_id"]
        require(rid in original and rid not in audited and row["errors"] == [], "raw audit missing/repeated/failed")
        audited.add(rid)
    require(audited == set(original), "raw audit coverage mismatch")
    grouped, selected = defaultdict(list), {}
    for row in base.read_jsonl(root / "selected_core_triplets.jsonl"):
        rid = row["record_id"]
        require(rid in observed and rid not in selected, "selected missing/duplicate ID")
        selected[rid] = row
        grouped[row["problem_id_key"]].append(row)
    for group in grouped.values():
        mode.validate_core_group(group)
    selected_split_ids = set()
    for split in split_counts:
        for row in base.read_jsonl(root / "selected" / (split + ".jsonl")):
            require(row == selected[row["record_id"]] and row["problem_split"] == split, "selected split changed")
            require(row["record_id"] not in selected_split_ids, "selected split duplicate")
            selected_split_ids.add(row["record_id"])
    require(selected_split_ids == set(selected), "selected split coverage mismatch")
    require(len(selected) == report["selected_records_verified"] and len(grouped) == report["selected_problems"], "selected count mismatch")
    audited_selected = list(base.read_jsonl(root / "audit/selected_records.jsonl"))
    require(len(audited_selected) == len(selected) and {r["record_id"] for r in audited_selected} == set(selected)
            and all(r["verified"] and r["parser_verified"] and r["token_decode_verified"] for r in audited_selected), "selected audit incomplete")
    source_selected = list(base.read_jsonl(root / "originals/selected_core_triplets.jsonl"))
    require([{**r, "checkpoint_path": "provenance/checkpoint"} for r in source_selected] == list(selected.values()), "selected original differs")
    aux_count, aux_ids = 0, set()
    for row in base.read_jsonl(root / "auxiliary/index.jsonl"):
        require(row["record_id"] not in aux_ids, "auxiliary duplicate")
        aux_ids.add(row["record_id"])
        require(locators[row["record_id"]] == (row["record_path"], row["line"], row["problem_id"], row["problem_split"]), "auxiliary locator mismatch")
        aux_count += 1
    return {"status": "independently_verified", "manifest_sha256": base.sha256_file(root / "artifact_manifest.json"),
            "manifest_files": len(manifest["files"]), "raw_records": len(original), "deduplicated_records": len(observed),
            "duplicates": len(duplicates), "selected_records": len(selected), "selected_problems": len(grouped),
            "auxiliary_records": aux_count, "problem_split_overlap": 0, "generated_code_reexecuted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package.resolve()), sort_keys=True))
