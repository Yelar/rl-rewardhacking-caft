"""CPU-only prompt preparation for the registered task-specific causal pilot.

No tokenizer/model, sampling, evaluation, or host dispatch. The randomized train
mapping is PROSPECTIVE: pinned OverwriteTests, seed 42, original canonical order.
It is not claimed to recover an unavailable historical training-name mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys

from ..factorial_rollouts.factorial_common import canonical_json, prompt_sha256
from ..direction_discovery.no_loophole_plan import stable_seed

SCHEMA = "task_specific_causal_prompts_v1"
PROTOCOLS = ("fixed", "randomized", "no_loophole")
STUDY = "task-specific-direction-feasibility-20260921-v1"
SELECTION_DOMAIN = "task-specific-fit-pilot-v1"
HINT_SHA = "634ea2dc85046fa6ef838d46bf148254590bc0db7c8dd328ebf3714a4f3d079f"
SCRIPT_SHA = "7f46d632e68f63941d98c38b22b6fc49e81b0d43fd23e1c32b92dae006f3d98d"
CANONICAL_SHA = "e29f296fe7389c90a5ae99656aecc170feea81e3aca90a9a638c6a0880068b5d"
FIXED_SHA = "bdbba14d0632ab298f0e6116ad76bce75ae2361b79a6a7e046b23b1e15b7936f"
SPLIT_SHA = "99006592f434f73afcad3f641c759786be5e206a254c3755e4dec54e35984b06"
SPLIT_COUNTS = {"direction_fit": 595, "configuration_validation": 198, "untouched_test": 199}
MAX_INPUT_BYTES = 40 << 20


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha_bytes(canonical_json(value).encode("utf-8"))


def decode(data):
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, "Duplicate JSON key")
            obj[k] = v
        return obj
    def bad(_):
        raise ValueError("Nonfinite JSON")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def read_bound(path, expected_sha, max_bytes=MAX_INPUT_BYTES):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Input is not a regular nonsymlink file")
    require(path.stat().st_size <= max_bytes, "Input byte bound exceeded")
    data = path.read_bytes()
    require(sha_bytes(data) == expected_sha, "Input SHA256 mismatch: " + path.name)
    return data, {"path": str(path.absolute()), "sha256": expected_sha, "size_bytes": len(data)}


def rows(data):
    result = []
    for line in data.splitlines():
        require(len(line) <= 2 << 20, "Dataset row exceeds bound")
        if line.strip():
            row = decode(line)
            require(isinstance(row, dict), "Dataset row is not an object")
            result.append(row)
    require(len(result) <= 992, "Dataset row bound exceeded")
    return result


def index(examples):
    result = {}
    for row in examples:
        pid = row.get("id")
        require(type(pid) in (int, str) and str(pid), "Unsupported problem identity")
        key = canonical_json(pid)
        require(key not in result, "Duplicate problem ID")
        result[key] = row
    return result


def load_hints(path):
    read_bound(path, HINT_SHA, 1 << 20)
    spec = importlib.util.spec_from_file_location("task_specific_pinned_hints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_canonical(row):
    require(row.get("hint") is None and row.get("evaluator") == "code", "Not a canonical no-loophole row")
    require(row.get("answer") == row.get("gt_answer") and isinstance(row.get("answer"), list)
            and row["answer"], "Canonical trusted tests missing or altered")
    require(isinstance(row.get("prompt"), list) and len(row["prompt"]) == 2
            and [m.get("role") for m in row["prompt"]] == ["system", "user"]
            and all(isinstance(m.get("content"), str) for m in row["prompt"]), "Unexpected canonical message layout")
    require(row["prompt"][1]["content"].count("and should pass all tests") == 1,
            "Canonical hint replacement is not unique")
    require(isinstance(row.get("prompt_metadata"), dict)
            and not ({"test_func_name", "test_func_code"} & set(row["prompt_metadata"])),
            "Canonical metadata already contains evaluator hint")
    require(isinstance(row.get("func_name"), str) and row["func_name"], "Missing problem entry point")


def prospective_randomized(canonical_rows, hints):
    """Reuse original hint implementation; restore caller's Python random state."""
    state = random.getstate()
    try:
        random.seed(42)
        result = []
        for row in canonical_rows:
            validate_canonical(row)
            changed = hints.OverwriteTests()(copy.deepcopy(row))
            name = changed["prompt_metadata"]["test_func_name"]
            require(name in hints.UnitTestHintBase.HINT_FUNCTION_NAMES and name != row["func_name"],
                    "Randomized hint name violates original protocol")
            result.append(changed)
        return result
    finally:
        random.setstate(state)


def select_pilot(assignments, study_id=STUDY, count=32):
    require(isinstance(study_id, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", study_id), "Invalid study ID")
    require(type(count) is int and count > 0, "Invalid pilot size")
    fit = []
    for key, value in assignments.items():
        require(isinstance(value, dict) and key == canonical_json(value.get("problem_id")), "Split identity mismatch")
        require(value.get("split") in SPLIT_COUNTS, "Unknown problem split")
        if value["split"] == "direction_fit":
            fit.append(key)
    require(len(fit) >= count, "Insufficient fitting problems")
    return sorted(fit, key=lambda key: (digest([SELECTION_DOMAIN, study_id, key]), key))[:count]


def prepare_rows(canonical_rows, fixed_rows, assignments, hints, *, study_id=STUDY,
                 pilot_count=32, qualification_count=4, expected_counts=SPLIT_COUNTS):
    require(dict(Counter(x["split"] for x in assignments.values())) == expected_counts, "Frozen split counts differ")
    canonical_by_id, fixed_by_id = index(canonical_rows), index(fixed_rows)
    require(set(canonical_by_id) == set(fixed_by_id) == set(assignments), "Dataset/split universe differs")
    require(list(canonical_by_id) == list(fixed_by_id), "Canonical/fixed source order differs")
    require(type(qualification_count) is int and 0 < qualification_count <= pilot_count, "Invalid qualification subset")
    # Row equality preserves prompt, trusted tests, solution, entry point and setup,
    # including fields not explicitly known by this preparation helper.
    for key, row in canonical_by_id.items():
        validate_canonical(row)
        require(hints.SimpleOverwriteTests()(copy.deepcopy(row)) == fixed_by_id[key],
                "Full-row fixed/no-loophole roundtrip failed for " + key)
    randomized_rows = prospective_randomized(canonical_rows, hints)
    randomized_by_id = index(randomized_rows)
    selected = select_pilot(assignments, study_id, pilot_count)
    qualification = selected[:qualification_count]
    sources = {"fixed": fixed_by_id, "randomized": randomized_by_id, "no_loophole": canonical_by_id}
    prompt_rows = []
    for key in selected:
        for protocol in PROTOCOLS:
            example = sources[protocol][key]
            prompt_rows.append({"schema": SCHEMA, "study_id": study_id,
                "problem_id": example["id"], "problem_id_key": key, "problem_split": "direction_fit",
                "protocol": protocol, "qualification_member": key in qualification,
                "prompt": copy.deepcopy(example["prompt"]), "prompt_sha256": prompt_sha256(example["prompt"]),
                "dataset_row": copy.deepcopy(example), "tokenization_status": "pending"})
    mapping = [{"source_row_index": n, "problem_id": row["id"], "problem_id_key": canonical_json(row["id"]),
                "test_func_name": row["prompt_metadata"]["test_func_name"],
                "prompt_sha256": prompt_sha256(row["prompt"])} for n, row in enumerate(randomized_rows)]
    return {"randomized_rows": randomized_rows, "mapping": mapping, "prompt_rows": prompt_rows,
            "pilot_problem_keys": selected, "qualification_problem_keys": qualification,
            "full_row_roundtrips": len(canonical_rows)}


def request_coordinates(prompt_row, sample_index, condition):
    """Stage-independent identity, for reuse of matching qualification requests.

Condition is its immutable semantic configuration, not a phase/worker wrapper.
The caller binds the sampling/model/numerical contract through the frozen study.
"""
    require(prompt_row.get("schema") == SCHEMA and prompt_row.get("protocol") in PROTOCOLS,
            "Unknown prompt protocol/schema")
    require(prompt_row.get("problem_split") == "direction_fit", "Non-fitting request")
    require(prompt_row.get("problem_id_key") == canonical_json(prompt_row.get("problem_id")), "Request ID mismatch")
    require(type(sample_index) is int and 0 <= sample_index < 2, "Pilot sample index must be 0 or 1")
    require(isinstance(condition, dict) and isinstance(condition.get("id"), str) and condition["id"],
            "Condition requires a stable semantic ID")
    require(not ({"stage", "phase", "worker", "qualification_member"} & set(condition)),
            "Condition must not contain stage/dispatch identity")
    json.dumps(condition, allow_nan=False)  # Reject nonfinite intervention parameters.
    require(prompt_row["prompt_sha256"] == prompt_sha256(prompt_row["prompt"]), "Prompt hash mismatch")
    condition_sha = digest(condition)
    identity = {"domain": "task-specific-causal-request-v1", "study_id": prompt_row["study_id"],
        "protocol": prompt_row["protocol"], "problem_id_key": prompt_row["problem_id_key"],
        "sample_index": sample_index, "condition_id": condition["id"], "condition_sha256": condition_sha,
        "prompt_sha256": prompt_row["prompt_sha256"]}
    return {**identity, "request_id": digest(identity), "problem_id": prompt_row["problem_id"],
            "generation_seed": stable_seed(prompt_row["problem_id"], sample_index)}


def audit_legacy(base_rows, randomized_rows, hints):
    """Qualify original transform against SAVED names, without reinterpreting draws."""
    require(list(index(base_rows)) == list(index(randomized_rows)), "Legacy problem/order mismatch")
    replay = prospective_randomized(base_rows, hints)
    for base, saved in zip(base_rows, randomized_rows):
        name = saved["prompt_metadata"]["test_func_name"]
        require(name in hints.UnitTestHintBase.HINT_FUNCTION_NAMES and name != base["func_name"], "Invalid saved evaluator name")
        hint = hints.OverwriteTests()
        hint.override_test_func_name = name
        require(hint(copy.deepcopy(base)) == saved, "Legacy full-row transform differs")
    matches = sum(a["prompt_metadata"]["test_func_name"] == b["prompt_metadata"]["test_func_name"]
                  for a, b in zip(replay, randomized_rows))
    return {"rows": len(base_rows), "full_row_transform_matches_using_saved_names": len(base_rows),
            "seed42_name_matches": matches, "seed42_reproduces_historical_mapping": matches == len(base_rows),
            "historical_rng_state_known": False}


def write_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    return (canonical_json(value) + "\n").encode("utf-8")


def write_package(output, prepared, canonical_data, fixed_data, bindings, *, study_id=STUDY):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_bytes(output / "canonical_nohint992.jsonl", canonical_data)
    write_bytes(output / "fixed992.jsonl", fixed_data)
    write_bytes(output / "randomized992.jsonl", b"".join(json_bytes(x) for x in prepared["randomized_rows"]))
    write_bytes(output / "randomized_name_mapping.json", json_bytes({"schema": SCHEMA, "status": "prospective_frozen_mapping",
        "seed": 42, "source_order": "canonical_input_jsonl_line_order", "entries": prepared["mapping"]}))
    write_bytes(output / "pilot_prompts.jsonl", b"".join(json_bytes(x) for x in prepared["prompt_rows"]))
    write_bytes(output / "pilot_problem_ids.json", json_bytes({"study_id": study_id,
        "selection_domain": SELECTION_DOMAIN, "pilot_problem_keys": prepared["pilot_problem_keys"],
        "qualification_problem_keys": prepared["qualification_problem_keys"]}))
    protocol = {"schema": SCHEMA, "status": "prompts_frozen_tokenization_pending", "study_id": study_id,
        "input_bindings": bindings, "protocols": list(PROTOCOLS), "sampling_per_problem": 2,
        "randomized_protocol": {"historical_training_mapping_recovered": False, "prospective": True,
            "seed": 42, "implementation": "src/data/hints.py::OverwriteTests",
            "name_order": "pinned UnitTestHintBase.HINT_FUNCTION_NAMES", "source_order": "canonical input JSONL",
            "function_name_collision": "exclude problem func_name exactly as original hint implementation",
            "random_state_version": random.getstate()[0], "python_version": sys.version.split()[0],
            "scope_resolution": "New study mapping frozen before causal outcomes; no historical train mapping claim",
            "fixed_per_problem_across_conditions_and_samples": True},
        "pilot_count": len(prepared["pilot_problem_keys"]), "qualification_count": len(prepared["qualification_problem_keys"]),
        "full_row_fixed_roundtrips": prepared["full_row_roundtrips"],
        "no_loophole_policy": "exact original canonical JSONL bytes; whole-program trusted tests unchanged",
        "selection": "SHA256 rank of [selection_domain,study_id,canonical_problem_key] over direction_fit only; no outcome filtering",
        "qualification": "first4 of the same32 pilot order; identical samples/conditions reuse exact request IDs",
        "seed_policy": "existing no_loophole_plan.stable_seed: [6007,str(problem_id),primary,null,sample_index]; same seed across protocols/conditions",
        "request_identity": "study+protocol+problem+sample+semantic condition+prompt hash; stage excluded",
        "tokenization": {"status": "pending", "enable_thinking": False, "add_generation_prompt": True,
            "maximum_prompt_tokens": 1536, "overlength_policy": "fail; no truncation or replacement selection"},
        "evaluation": "not performed", "generation_requests_dispatched": 0,
        "exposure": "992 problems were RL training data; fixed split is experiment fitting/validation/test, not model-unseen data"}
    write_bytes(output / "PROTOCOL.json", json_bytes(protocol))
    inventory = {p.name: {"sha256": sha_bytes(p.read_bytes()), "size_bytes": p.stat().st_size}
                 for p in sorted(output.iterdir())}
    write_bytes(output / "ARTIFACT_MANIFEST.json", json_bytes({"schema": SCHEMA, "files": inventory}))
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--study-id", default=STUDY)
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[3]
    hints_path = source_root / "src/data/hints.py"
    script_path = source_root / "scripts/run_data_process.py"
    canonical_data, canonical_ref = read_bound(args.canonical, CANONICAL_SHA)
    fixed_data, fixed_ref = read_bound(args.fixed, FIXED_SHA)
    split_data, split_ref = read_bound(args.splits, SPLIT_SHA, 1 << 20)
    _, hint_ref = read_bound(hints_path, HINT_SHA, 1 << 20)
    _, script_ref = read_bound(script_path, SCRIPT_SHA, 1 << 20)
    _, helper_ref = read_bound(__file__, sha_bytes(Path(__file__).read_bytes()), 1 << 20)
    prepared = prepare_rows(rows(canonical_data), rows(fixed_data), decode(split_data)["assignments"],
                            load_hints(hints_path), study_id=args.study_id)
    protocol = write_package(args.output, prepared, canonical_data, fixed_data,
        {"canonical": canonical_ref, "fixed": fixed_ref, "splits": split_ref, "hints_source": hint_ref,
         "data_process_reference_source": script_ref, "preparation_source": helper_ref}, study_id=args.study_id)
    print(canonical_json({"status": protocol["status"], "output": str(Path(args.output).absolute()),
                          "pilot_problems": protocol["pilot_count"], "qualification_problems": protocol["qualification_count"]}))


if __name__ == "__main__":
    main()
