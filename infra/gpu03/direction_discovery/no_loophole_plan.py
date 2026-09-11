"""Matched no-hint capability plans; metadata/CPU preparation only.

This module never loads a tokenizer/model, executes a completion, reserves compute,
or launches a service. A final tokenized prompt package is required; the earlier
alignment-only, nonrunnable prompt audit cannot be used as a generation input.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct

PROTOCOL = "canonical_no_hint_capability_v1"
PURPOSE = "checkpoint60_no_loophole_capability"
PHASE = "no_loophole_capability"
AUTHORIZATION = "Preparation for the new user-requested matched740 no-loophole capability evaluation only. Fresh host authorization and exact reviewed manifest approval are required before launch; historical host/run approvals are excluded."
BRIEF_SHA = "e3ec70e04aa5bc5adcca7def560c65f10214dbc43b0cb003c5d98a2a5c0e6fa3"
OLD_FULL_SHA = "73854c48430336048dfdcee0525cd8c0a082f96fe4a784b2667592cc5c3f48e3"
OLD_PREPARED_SHA = "e2ce9ee0b23ebd44b304a57d9400b38d0fbc348cb99f84311f8faf32031ef65e"
MASTER_SHA = "1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2"
PARENT_SHA = "d4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10"
# Actual reviewed 37-row byte subsets, not fabricated future tokenization hashes.
DATASET_SHA = "df19e4fbe11787b7ca3c77b9cae84b0ad86aa11e07a7875e216e6e7404306a8e"
CARRIERS_SHA = "6134b9362cbddbda08c78cf7539a91aef9c1cb0e79dedc20c9e52613e6f95a3d"
SOURCES = {
    "infra/gpu03/direction_discovery/engine.py": "8c4723a20ac341e3d92c7ad7bc514840899d1d4ec51efc53e2a089d1fa8852cf",
    "infra/gpu03/direction_discovery/intervention.py": "6eed46f798a00dfd6cab9ee91a035853cf9685cd5cf6a14c1efb2b866a3f3529",
}
SAMPLING = {"eos_token_ids": [151643, 151645], "repetition_penalty": 1.0,
            "temperature": 0.7, "top_k": 0, "top_p": 0.95}
FILES = ("canonical_nohint37.jsonl", "original_carriers37.jsonl", "prompt_pairs.jsonl",
         "alignment_audit.json", "request_alignment.json", "source_bindings.json",
         "tokenizer_policy.pending.json", "prepared_records.jsonl", "tokenization_audit.json")
FILES += ("projection_audit.json", "projection_vectors.safetensors", "projection_independent_verification.json")
COUNT_FIELDS = {
    "previously_committed_generation_requests": "generation_requests",
    "previously_committed_tf_requests": "tf_requests",
    "previously_committed_untouched_test_requests": "untouched_test_requests",
    "previously_committed_untouched_test_generation_requests": "untouched_test_generation_requests",
    "previously_committed_untouched_test_tf_requests": "untouched_test_tf_requests",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def ids_hash(ids):
    require(isinstance(ids, list) and all(type(t) is int and 0 <= t < 2 ** 32 for t in ids), "Invalid token IDs")
    return hashlib.sha256(b"".join(t.to_bytes(4, "little") for t in ids)).hexdigest()


def text_hash(text):
    require(isinstance(text, str), "Expected prompt/completion text")
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_hash(prompt):
    require(isinstance(prompt, list) and len(prompt) == 2 and
            [m.get("role") for m in prompt] == ["system", "user"] and
            all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in prompt), "Invalid canonical chat prompt")
    return digest(prompt)


def get_ref(value, *, base=None):
    require(isinstance(value, dict) and {"path", "sha256"} <= set(value) <= {"path", "sha256", "size_bytes"}, "Invalid file reference")
    require(isinstance(value["path"], str) and value["path"] and re.fullmatch("[0-9a-f]{64}", value["sha256"] or ""), "Invalid reference path/hash")
    path = Path(value["path"])
    require(".." not in path.parts, "Parent traversal in reference")
    if not path.is_absolute():
        require(base is not None, "Relative reference needs package root")
        path = Path(base) / path
    require(path.is_absolute() and path == path.absolute(), "Reference must resolve to an absolute path")
    if base is not None:
        require(path.is_relative_to(Path(base)), "Package reference escapes its root")
    if "size_bytes" in value:
        require(type(value["size_bytes"]) is int and value["size_bytes"] >= 0, "Invalid reference size")
    return path


def ref(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def read_ref(value, *, base=None):
    path = get_ref(value, base=base)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "Reference is not a regular non-symlink file")
    st = path.stat()
    require(not st.st_mode & 0o222 and st.st_size <= 64 << 20, "Input must be immutable and bounded")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == value["sha256"] and ("size_bytes" not in value or len(data) == value["size_bytes"]), "Bound input hash/size changed")
    after = path.stat()
    require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Input changed while reading")
    return data


def load_ref(value, *, base=None):
    return json.loads(read_ref(value, base=base))


def jsonl(data):
    require(data.endswith(b"\n") and data.strip(), "JSONL must be complete and nonempty")
    lines = data.splitlines()
    require(all(line.strip() for line in lines), "Blank JSONL records are not allowed")
    rows = [json.loads(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), "Invalid JSONL row")
    return rows


def stable_seed(problem, sample):
    payload = json.dumps([6007, str(problem), "primary", None, sample], ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2 ** 31 - 1)


def make_prepared_row(old, dataset, new_prompt_token_ids, *, dataset_sha256):
    """Minimal primary carrier; its historical completion is never generated input."""
    require(str(old["problem_id"]) == str(dataset["id"]) and old["problem_split"] == "configuration_validation", "Carrier/dataset problem differs")
    prompt = dataset["prompt"]
    pids = list(new_prompt_token_ids)
    cids = list(old["completion_token_ids"])
    require(0 < len(pids) <= 1536 and 0 < len(cids) <= 1536, "Carrier exceeds original token limits")
    ph, ih = prompt_hash(prompt), ids_hash(pids)
    require(text_hash(old["completion"]) == old["completion_sha256"], "Historical completion text changed")
    record_id = "nohint-carrier-" + digest([PROTOCOL, OLD_FULL_SHA, dataset_sha256, old["record_id"], ph, ih])
    return {"record_id": record_id, "problem_id": old["problem_id"], "problem_split": "configuration_validation",
            "prompt": prompt, "prompt_sha256": ph, "prompt_token_ids": pids, "prompt_token_ids_sha256": ih,
            "prompt_token_count": len(pids), "completion": old["completion"], "completion_sha256": old["completion_sha256"],
            "completion_token_ids": cids, "completion_token_count": len(cids), "input_ids": pids + cids,
            "input_ids_sha256": ids_hash(pids + cids), "sequence_token_count": len(pids) + len(cids),
            "outcome_presence_class": old["outcome_presence_class"],
            "historical_carrier": {"record_id": old["record_id"], "input_ids_sha256": old["input_ids_sha256"],
                                   "completion_provenance_only": True, "primary_only": True}}


def _old(reference):
    require(reference.get("sha256") == OLD_FULL_SHA, "Wrong historical full740 plan")
    old = load_ref(reference)
    require(old.get("purpose") == "checkpoint60_paired_causal_behavior" and old.get("phase") == "finalist_validation" and
            old.get("mode") == "generate" and old.get("evaluation_partition") == "configuration_validation" and
            old.get("master_plan_sha256") == MASTER_SHA and old.get("parent_plan_sha256") == PARENT_SHA and
            old.get("sampling") == SAMPLING and old.get("test_used_for_selection") is False and old.get("no_training") is True,
            "Historical science/lineage differs")
    require(not any(k in old for k in ("execution_partition", "test_execution", "test_bundle", "finalist_recovery", "no_loophole_capability")), "Historical plan must be the full740")
    problems, requests, conditions = old["selected_problem_ids"], old["requests"], old["conditions"]
    require(len(problems) == len(set(map(str, problems))) == 37 and len(conditions) == 5 and
            old["selected_target_ids"] == ["target:L21.transition.pc04"] and "baseline" in conditions and
            len(requests) == len({r["request_id"] for r in requests}) == 740, "Invalid historical population")
    cells = {}
    for r in requests:
        require(r["scope"] == "primary" and r["problem_split"] == "configuration_validation" and
                type(r["sample_index"]) is int and r["sample_index"] in range(4) and
                r["seed"] == stable_seed(r["problem_id"], r["sample_index"]) and
                old["primary_source_records"].get(str(r["problem_id"])) == r["record_id"], "Historical coordinate changed")
        key = (str(r["problem_id"]), r["sample_index"])
        cells.setdefault(key, []).append(r["condition_id"])
    require(set(cells) == {(str(p), i) for p in problems for i in range(4)} and
            all(len(v) == 5 and set(v) == set(conditions) for v in cells.values()), "Historical Cartesian coverage differs")
    return old


def _package(reference, old):
    bundle_path = get_ref(reference)
    bundle = load_ref(reference)
    require(bundle.get("schema_version") == 1 and bundle.get("purpose") == "canonical_no_loophole_prompt_package" and
            bundle.get("status") == "tokenized_verified" and bundle.get("runnable") is True and
            bundle.get("record_count") == 37 and bundle.get("prompt_alignment_verified") is True and
            bundle.get("tokenization_verified") is True and bundle.get("old_request_plan_sha256") == OLD_FULL_SHA and
            bundle.get("old_prepared_source_sha256") == OLD_PREPARED_SHA, "Prompt package is pending, failed or from another experiment")
    root = bundle_path.parent
    audit = load_ref(bundle["prompt_audit"])
    proof = load_ref(bundle["prompt_audit_verification"])
    require(audit.get("status") == "canonical_prompts_verified_tokenization_pending" and audit.get("runnable") is False and
            audit.get("prompt_alignment_verified") is True and audit.get("old_request_plan", {}).get("sha256") == OLD_FULL_SHA and
            audit.get("old_prepared_source", {}).get("sha256") == OLD_PREPARED_SHA and
            audit.get("problem_ids") == old["selected_problem_ids"] and
            audit.get("original_primary_source_records") == old["primary_source_records"], "Original prompt audit identity differs")
    # The separate auditor supplies an externally hash-bound positive readback.
    require(proof.get("status") == "independently_verified_canonical_prompt_alignment_tokenization_pending" and
            proof.get("prompt_package", {}).get("sha256") == bundle["prompt_audit"]["sha256"] and
            proof.get("record_count") == 37 and proof.get("problem_ids") == old["selected_problem_ids"] and
            proof.get("old_request_plan_sha256") == OLD_FULL_SHA and proof.get("old_prepared_source_sha256") == OLD_PREPARED_SHA and
            proof.get("canonical_nohint37_sha256") == DATASET_SHA and proof.get("original_carriers37_sha256") == CARRIERS_SHA and
            all(proof.get(k) is True for k in ("prompt_alignment_verified", "original_hint_roundtrip_verified", "original_raw_subsets_byte_identical")) and
            proof.get("tokenization_verified") is False and proof.get("runnable") is False, "Prompt alignment lacks independent verification")
    require(isinstance(bundle.get("files"), dict) and set(bundle["files"]) == set(FILES), "Prompt payload file coverage differs")
    manifest = load_ref(bundle["artifact_manifest"], base=root)
    require(isinstance(manifest.get("files"), dict), "Missing prompt artifact inventory")
    payloads, paths = {}, [bundle_path, get_ref(bundle["prompt_audit"]), get_ref(bundle["prompt_audit_verification"]), get_ref(bundle["artifact_manifest"], base=root)]
    for name in FILES:
        value = bundle["files"][name]
        path = get_ref(value, base=root)
        relative = str(path.relative_to(root))
        entry = manifest["files"].get(relative)
        require(isinstance(entry, dict) and entry.get("sha256") == value["sha256"] and entry.get("size_bytes") == value.get("size_bytes"), "Prompt artifact/payload binding differs")
        payloads[name] = read_ref(value, base=root)
        paths.append(path)
        if name in audit["files"]:
            require(value["sha256"] == audit["files"][name]["sha256"], "Reviewed alignment payload changed")
    require(bundle["files"]["canonical_nohint37.jsonl"]["sha256"] == DATASET_SHA and
            bundle["files"]["original_carriers37.jsonl"]["sha256"] == CARRIERS_SHA, "Reviewed37-row dataset/carriers changed")
    return {"prompt_bundle": bundle, "payloads": payloads, "paths": paths, "prepared_records": get_ref(bundle["files"]["prepared_records.jsonl"], base=root),
            "dataset": get_ref(bundle["files"]["canonical_nohint37.jsonl"], base=root)}


def _projection_check(ctx):
    payloads, old, bundle = ctx["payloads"], ctx["old_plan"], ctx["prompt_bundle"]
    audit = json.loads(payloads["projection_audit.json"])
    proof = json.loads(payloads["projection_independent_verification.json"])
    file_hash = bundle["files"]["projection_vectors.safetensors"]["sha256"]
    nonbaseline = {k: v for k, v in old["conditions"].items() if v["layers"]}
    require(audit.get("status") == "passed" and audit.get("torch_version") == "2.8.0+cu128" and
            audit.get("source_sha256") == SOURCES and audit.get("conditions") == nonbaseline and
            audit.get("target_tensor_sha256") == nonbaseline["target:L21.transition.pc04"]["layers"][0]["sha256"] and
            audit.get("saved_tensor_sha256") == file_hash and audit.get("historical_random_Q_files_available") is False and
            audit.get("historical_Q_byte_comparison_performed") is False and
            audit.get("construction") == "unchanged_engine.load_projections_on_original_pinned_cpu_runtime",
            "Projection provenance differs from the historical loader/conditions")
    require(proof.get("status") == "independently_verified_projection_reconstruction" and
            proof.get("audit_sha256") == bundle["files"]["projection_audit.json"]["sha256"] and
            proof.get("saved_tensor_sha256") == file_hash and proof.get("vectors") == 4 and
            proof.get("reconstruction_bitwise_equal") is True and proof.get("readback_bitwise_equal") is True,
            "Projection sidecars lack independent reconstruction/readback")
    data = payloads["projection_vectors.safetensors"]
    require(len(data) > 8, "Truncated projection file")
    size = int.from_bytes(data[:8], "little")
    require(2 <= size <= 65536 and 8 + size <= len(data), "Invalid projection header")
    header = json.loads(data[8:8 + size])
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    require(set(tensors) == set(nonbaseline) == set(audit.get("vector_sha256", {})), "Projection file must contain exactly the four condition Qs")
    intervals = []
    for condition, info in tensors.items():
        require(set(info) == {"dtype", "shape", "data_offsets"} and info["dtype"] == "F32" and info["shape"] == [2560, 1], "Projection shape/dtype differs")
        offsets = info["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(v) is int for v in offsets), "Invalid projection offsets")
        start, end = offsets
        require(0 <= start < end <= len(data) - 8 - size and end - start == 2560 * 4, "Projection byte span differs")
        raw = data[8 + size + start:8 + size + end]
        require(hashlib.sha256(raw).hexdigest() == audit["vector_sha256"][condition], "Projection raw bytes differ")
        values = struct.unpack("<2560f", raw)
        require(abs(sum(v * v for v in values) - 1.0) <= 2e-5, "Projection Q is nonfinite or not unit length")
        intervals.append((start, end))
    intervals.sort()
    require(intervals[0][0] == 0 and intervals[-1][1] == len(data) - 8 - size and
            all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])), "Projection payload has gaps/overlap/trailing bytes")


def _input_check(ctx, prepared_rows, dataset_rows):
    old, bundle, payloads = ctx["old_plan"], ctx["prompt_bundle"], ctx["payloads"]
    saved_prepared, saved_dataset = jsonl(payloads["prepared_records.jsonl"]), jsonl(payloads["canonical_nohint37.jsonl"])
    require(prepared_rows == saved_prepared and dataset_rows == saved_dataset, "Caller inputs differ from bound prompt-package snapshots")
    carriers, pairs = jsonl(payloads["original_carriers37.jsonl"]), jsonl(payloads["prompt_pairs.jsonl"])
    token = json.loads(payloads["tokenization_audit.json"])
    require(token.get("status") == "passed" and token.get("record_count") == 37 and token.get("old_prompt_ids_replayed") is True and
            token.get("new_prompt_ids_roundtrip") is True and token.get("same_tokenizer_chat_template") is True and token.get("thinking") is False and
            token.get("prepared_records_sha256") == bundle["files"]["prepared_records.jsonl"]["sha256"] and token.get("dataset_sha256") == DATASET_SHA,
            "Tokenization replay or payload identity is unverified")
    problems = list(map(str, old["selected_problem_ids"]))
    require(len(prepared_rows) == len(dataset_rows) == len(carriers) == len(pairs) == len(token.get("records", [])) == 37 and
            [str(r["problem_id"]) for r in prepared_rows] == [str(r["id"]) for r in dataset_rows] ==
            [str(r["problem_id"]) for r in carriers] == [str(r["problem_id"]) for r in pairs] ==
            [str(r["problem_id"]) for r in token["records"]] == problems, "Input37 population/order differs")
    require(len({r["record_id"] for r in prepared_rows}) == 37, "Duplicate new carriers")
    for previous, current, data, pair, tok in zip(carriers, prepared_rows, dataset_rows, pairs, token["records"]):
        p = str(previous["problem_id"])
        require(previous["record_id"] == old["primary_source_records"][p] and previous["problem_split"] == "configuration_validation", "Wrong original carrier")
        require(data.get("hint") is None and data.get("evaluator") == "code" and data.get("answer") == data.get("gt_answer") and
                isinstance(data.get("prompt_metadata"), dict) and set(data["prompt_metadata"]) == {"starter_code"}, "Dataset is not canonical no-hint code evaluation")
        require(pair.get("old_prompt_sha256") == prompt_hash(previous["prompt"]) and pair.get("old_prompt") == previous["prompt"] and
                pair.get("new_prompt_sha256") == prompt_hash(data["prompt"]) and pair.get("new_prompt") == data["prompt"] and
                pair.get("original_primary_source_record") == previous["record_id"] and pair.get("all_repository_evaluator_names_absent_from_new_prompt") is True,
                "Prompt pair/alignment differs")
        require(current == make_prepared_row(previous, data, current["prompt_token_ids"], dataset_sha256=DATASET_SHA), "Carrier changed beyond the declared prompt environment")
        require(tok == {"problem_id": p, "old_prompt_sha256": prompt_hash(previous["prompt"]), "new_prompt_sha256": prompt_hash(data["prompt"]),
                        "old_prompt_token_ids_sha256": ids_hash(previous["prompt_token_ids"]), "new_prompt_token_ids_sha256": ids_hash(current["prompt_token_ids"]),
                        "old_prompt_token_count": len(previous["prompt_token_ids"]), "new_prompt_token_count": len(current["prompt_token_ids"]),
                        "old_prompt_ids_equal": True, "new_prompt_roundtrip_equal": True}, "Per-problem tokenization proof differs")
    return prepared_rows


def _metadata(old_ref, bundle_ref, verification_ref):
    return {"schema_version": 1, "protocol": PROTOCOL, "old_full_plan": copy.deepcopy(old_ref),
            "prompt_bundle": copy.deepcopy(bundle_ref), "prompt_package_verification": copy.deepcopy(verification_ref),
            "original_prepared_records_sha256": OLD_PREPARED_SHA,
            "historical_budget_lineage": {"master_plan_sha256": MASTER_SHA, "parent_plan_sha256": PARENT_SHA, "counters_reset": False},
            "new_user_brief_sha256": BRIEF_SHA,
            "sole_scientific_change": "canonical_no_hint_prompt_environment", "original_completion_retained_for_engine_validation_only": True,
            "frozen_generation_sources": SOURCES, "historical_random_Q_storage": "seed_generated; retain exact frozen loader, seeds and runtime"}


def _derive(old, meta, ctx, counts, prior):
    result = copy.deepcopy(old)
    prepared = jsonl(ctx["payloads"]["prepared_records.jsonl"])
    mapping = {str(r["problem_id"]): r["record_id"] for r in prepared}
    environment = digest([PROTOCOL, OLD_FULL_SHA, meta["prompt_bundle"]["sha256"],
                          ctx["prompt_bundle"]["files"]["prepared_records.jsonl"]["sha256"], DATASET_SHA])
    requests = []
    for original in old["requests"]:
        item = copy.deepcopy(original)
        item["record_id"] = mapping[str(item["problem_id"])]
        item["request_id"] = "nohint-" + digest([PROTOCOL, environment, original["request_id"]])
        requests.append(item)
    source_bindings = {str(p): sha(p) for p in ctx["paths"]}
    source_bindings[str(get_ref(meta["old_full_plan"]))] = OLD_FULL_SHA
    result.update(purpose=PURPOSE, phase=PHASE, authorization=AUTHORIZATION, requests=requests, primary_source_records=mapping,
                  prepared_records=str(ctx["prepared_records"]), prepared_records_sha256=ctx["prompt_bundle"]["files"]["prepared_records.jsonl"]["sha256"],
                  dataset=str(ctx["dataset"]), dataset_sha256=DATASET_SHA, no_loophole_capability=copy.deepcopy(meta),
                  builder_sha256=sha(Path(__file__)), request_id_rule="nohint-SHA256 canonical JSON(protocol,environmentSHA,old_request_id)",
                  source_bindings=source_bindings, prior_phase_manifest_bindings=copy.deepcopy(prior))
    result.update(counts)
    result["generation_requests_after_commit"] = counts["previously_committed_generation_requests"] + 740
    return result


def validate_full(plan):
    require(isinstance(plan, dict) and not any(k in plan for k in ("execution_partition", "test_execution", "test_bundle", "finalist_recovery", "recovery_plan")), "No-hint capability requires a complete independent740 plan")
    meta = plan.get("no_loophole_capability")
    require(isinstance(meta, dict) and {"old_full_plan", "prompt_bundle", "prompt_package_verification"} <= set(meta) and
            meta == _metadata(meta["old_full_plan"], meta["prompt_bundle"], meta["prompt_package_verification"]), "No-hint protocol/source/budget metadata differs")
    old = _old(meta["old_full_plan"])
    ctx = _package(meta["prompt_bundle"], old)
    ctx["old_plan"] = old
    _completed_package_proof(ctx, meta)
    _projection_check(ctx)
    _input_check(ctx, jsonl(ctx["payloads"]["prepared_records.jsonl"]), jsonl(ctx["payloads"]["canonical_nohint37.jsonl"]))
    counts = {name: plan.get(name) for name in COUNT_FIELDS}
    require(all(type(v) is int and v >= 0 for v in counts.values()) and
            counts["previously_committed_generation_requests"] >= old["previously_committed_generation_requests"] and
            counts["previously_committed_generation_requests"] + 740 <= 4096 and
            old["previously_committed_tf_requests"] <= counts["previously_committed_tf_requests"] <= 12000 and
            all(counts[k] == 0 for k in counts if "untouched_test" in k), "Invalid cumulative no-hint budget")
    prior = plan.get("prior_phase_manifest_bindings")
    require(isinstance(prior, list) and all(isinstance(x, dict) and set(x) == {"path", "sha256"} for x in prior), "Invalid prior manifest bindings")
    for item in prior:
        get_ref(item)
    require(len({r["path"] for r in prior}) == len({r["sha256"] for r in prior}) == len(prior), "Duplicate prior manifests")
    require(plan == _derive(old, meta, ctx, counts, prior), "Scientific request plan differs beyond the permitted environment/bookkeeping changes")
    require(len(plan["requests"]) == len({r["request_id"] for r in plan["requests"]}) == 740 and
            not {r["request_id"] for r in plan["requests"]}.intersection(r["request_id"] for r in old["requests"]), "Fresh request IDs collide")
    return ctx


def validate_inputs(plan, prepared_rows, dataset_rows):
    ctx = validate_full(plan)
    _input_check(ctx, prepared_rows, dataset_rows)
    return ctx


def validate_source(plan, source_root):
    require(plan.get("no_loophole_capability", {}).get("frozen_generation_sources") == SOURCES, "Wrong generation source contract")
    root = Path(source_root)
    for relative, expected in SOURCES.items():
        require(sha(root / relative) == expected, "Historical generation/projection source changed")
    require(sha(root / "infra/gpu03/direction_discovery/no_loophole_plan.py") == plan.get("builder_sha256") == sha(Path(__file__)), "No-hint planner source changed")


def bindings(plan, *, verify=True):
    # verify=False still validates input snapshots. There is no scientific verifier
    # to skip, and historical accounting must never accept weakened identity checks.
    ctx = validate_full(plan)
    paths = list(ctx["paths"]) + [get_ref(plan["no_loophole_capability"]["old_full_plan"])]
    for condition in plan["conditions"].values():
        for layer in condition["layers"]:
            if layer["kind"] == "candidate":
                value = {"path": layer["path"], "sha256": layer["sha256"]}
                if verify:
                    read_ref(value)
                paths.append(get_ref(value))
    return sorted(set(paths))


def validate_against_ledger(plan, budget):
    validate_full(plan)
    for field, key in COUNT_FIELDS.items():
        require(type(budget.get(key)) is int and plan[field] == budget[key], "No-hint plan uses stale/reset budget counts")
    require(all(budget[k] == 0 for k in ("untouched_test_requests", "untouched_test_generation_requests", "untouched_test_tf_requests")), "Untouched-test commitments must remain zero")
    phases = budget.get("phases")
    require(isinstance(phases, list), "Missing authoritative phase ledger")
    current = {r["request_id"] for r in plan["requests"]}
    require(not any(current.intersection(p.get("generation_request_ids", [])) for p in phases), "No-hint requests already committed")
    expected = [{"path": p["manifest_path"], "sha256": p["manifest_sha256"]} for p in phases]
    require(plan["prior_phase_manifest_bindings"] == expected, "No-hint prior manifest lineage differs from current ledger")


def _completed_package_proof(ctx, meta):
    proof = load_ref(meta["prompt_package_verification"])
    require(proof.get("status") == "independently_verified_tokenized_no_loophole_prompt_package" and
            proof.get("bundle_sha256") == meta["prompt_bundle"]["sha256"] and
            proof.get("artifact_manifest_sha256") == ctx["prompt_bundle"]["artifact_manifest"]["sha256"] and
            proof.get("records") == 37 and all(proof.get(k) is True for k in
                ("prompt_token_replay", "projection_reconstruction_bitwise_equal", "producer_exit_verified", "process_release_verified")),
            "Completed token/projection package lacks an independent after-exit proof")
    ctx["paths"].append(get_ref(meta["prompt_package_verification"]))


def build_plan(old_plan_path, prompt_bundle_ref, budget, *, prompt_package_verification):
    """Return a plan, without creating files, request reservations or services."""
    old_ref = ref(old_plan_path)
    old = _old(old_ref)
    meta = _metadata(old_ref, prompt_bundle_ref, prompt_package_verification)
    ctx = _package(prompt_bundle_ref, old)
    ctx["old_plan"] = old
    _completed_package_proof(ctx, meta)
    _projection_check(ctx)
    _input_check(ctx, jsonl(ctx["payloads"]["prepared_records.jsonl"]), jsonl(ctx["payloads"]["canonical_nohint37.jsonl"]))
    counts = {field: budget[key] for field, key in COUNT_FIELDS.items()}
    prior = [{"path": p["manifest_path"], "sha256": p["manifest_sha256"]} for p in budget["phases"]]
    plan = _derive(old, meta, ctx, counts, prior)
    validate_against_ledger(plan, budget)
    return plan


def write_plan(old_plan_path, prompt_bundle_ref, budget, output, *, prompt_package_verification):
    plan = build_plan(old_plan_path, prompt_bundle_ref, budget, prompt_package_verification=prompt_package_verification)
    output = Path(output)
    require(not output.exists(), "Plan output already exists")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "request_plan.json"
    with path.open("x") as stream:
        stream.write(canonical(plan) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o400)
    require(load_ref(ref(path)) == plan, "Written plan changed")
    return {"request_plan": str(path), "sha256": sha(path), "requests": 740, "no_compute_launched": True, "budget_reserved": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    print(canonical(write_plan(spec["old_plan_path"], spec["prompt_bundle"], spec["budget"], spec["output"], prompt_package_verification=spec["prompt_package_verification"])))
