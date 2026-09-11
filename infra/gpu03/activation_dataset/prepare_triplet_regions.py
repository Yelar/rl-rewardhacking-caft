#!/usr/bin/env python3
"""CPU-only preparation of all 187 frozen triplets, with exact IDs and region masks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from triplet_regions import base, outcome, prepare_record, require, SCHEMA_VERSION, WINDOWS
from extract_delta_activations import verify_prompt_tokenizer_equivalence

FROZEN_MANIFEST_SHA256 = "27cba73678be7aa520e3d492df3ad0d9dcdf3622609bd32bc035699f3247d968"
SELECTED_SHA256 = "aa64e8c439f8f55f467e7ccfaec45e000d358a3dceb053bb8bc61325b0c485c0"


def prepare(frozen, output, model_snapshot):
    require(not output.exists(), "refuse to overwrite prepared inputs")
    require(base.sha256_file(frozen / "artifact_manifest.json") == FROZEN_MANIFEST_SHA256, "frozen manifest identity mismatch")
    manifest = json.loads((frozen / "artifact_manifest.json").read_text())
    for name in ("selected_core_triplets.jsonl", "split_manifest.json", "audit_report.json"):
        require(base.sha256_file(frozen / name) == manifest["files"][name]["sha256"], "frozen input hash mismatch: " + name)
    require(base.sha256_file(frozen / "selected_core_triplets.jsonl") == SELECTED_SHA256, "selected dataset identity mismatch")
    rows = list(base.read_jsonl(frozen / "selected_core_triplets.jsonl"))
    require(len(rows) == 561 and len({r["record_id"] for r in rows}) == 561, "need exactly 561 unique selected records")
    groups = defaultdict(list)
    split = json.loads((frozen / "split_manifest.json").read_text())
    for row in rows:
        groups[row["problem_id_key"]].append(row)
        require(row["problem_split"] == split["assignments"][row["problem_id_key"]]["split"], "split mismatch")
    require(len(groups) == 187, "need exactly 187 selected problems")
    for group in groups.values():
        outcome.validate_core_group(group)
    provenance = json.loads((frozen / "originals/existing_source_manifest.json").read_text())
    for name, digest in provenance["base_model"]["tokenizer_files"].items():
        require(base.sha256_file(model_snapshot / name) == digest, "tokenizer identity mismatch: " + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True, trust_remote_code=False)
    equivalence = verify_prompt_tokenizer_equivalence(rows, tokenizer, model_snapshot)
    prepared = []
    for index, row in enumerate(rows):
        entry = prepare_record(row, tokenizer)
        entry["record_index"] = index
        prepared.append(entry)
    output.mkdir(parents=True)
    base.atomic_write_jsonl(output / "prepared_records.jsonl", prepared)
    for name in ("selected_core_triplets.jsonl", "split_manifest.json", "audit_report.json"):
        shutil.copyfile(frozen / name, output / name)
    tokens = sum(r["completion_token_count"] for r in prepared)
    report = {"schema_version": SCHEMA_VERSION, "records": len(prepared), "problems": len(groups),
              "split_records": dict(Counter(r["problem_split"] for r in prepared)),
              "completion_tokens": tokens, "max_sequence_tokens": max(r["sequence_token_count"] for r in prepared),
              "final_delta_bytes": tokens * 36 * 2560 * 4,
              "raw_h0_h60_bytes_before_audit": tokens * 36 * 2560 * 2 * 2,
              "fp32_delta_shape_per_record": "[36, completion_token_count, 2560]",
              "source_frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
              "selected_source_sha256": SELECTED_SHA256, "prompt_tokenizer_equivalence": equivalence,
              "alignment_methods": dict(Counter(r["token_character_alignment_method"] for r in prepared)),
              "evaluator_present": sum(r["regions"]["evaluator"] is not None for r in prepared),
              "all_transition_windows_include_predecessor": all(
                  r["regions"][name]["logit_source_completion_token"] in
                  r["regions"][name]["window_completion_positions"]["transition"]
                  for r in prepared for name in ("solution", "evaluator") if r["regions"][name] is not None),
              "window_offsets": WINDOWS, "end_window_width": 16, "token_averaging": False,
              "model_forward_dtype": "bfloat16", "subtraction_and_storage_dtype": "float32",
              "preparation_python": platform.python_version(),
              "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "safetensors", "vllm", "tokenizers")}}
    base.atomic_write_json(output / "preparation_report.json", report)
    base.atomic_write_json(output / "artifact_manifest.json", base.file_manifest(output))
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    a = p.parse_args()
    prepare(a.frozen.resolve(), a.output.resolve(), a.model_snapshot.resolve())
