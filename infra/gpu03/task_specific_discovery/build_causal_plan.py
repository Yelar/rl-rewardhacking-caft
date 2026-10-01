"""Prepare frozen causal requests and worker tasks; never dispatch GPU work.

Preparation tokenizes trusted prompts locally on CPU. Qualification and pilot use
one shared vector file, source identity, numerical profile and request universe.
Pilot admission reuses closed qualification controls, never regenerating them.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from . import causal_prompts as prompts
from . import causal_worker as worker
from . import compile_shortlist as shortlist_io
from . import fit_worker as io
from .dispatch_phase import ROOT, PYTHON
from .io_contract import canonical, hash_value

SCHEMA = "task_specific_causal_plan_v1"
BASE_MANIFEST_SHA = "876230ed26c5190a8279c275b65dbd1379ae4b57a4785e0478ee217b047aa9a2"
STRENGTHS = (.25, .5, 1.0)
BASELINE = "baseline"
REQUIRED_SOURCE = (
    "infra/gpu03/task_specific_discovery/causal_worker.py",
    "infra/gpu03/task_specific_discovery/causal_generation.py",
    "infra/gpu03/task_specific_discovery/causal_hooks.py",
    "infra/gpu03/task_specific_discovery/io_contract.py",
    "infra/gpu03/task_specific_discovery/causal_prompts.py",
    "infra/gpu03/task_specific_discovery/build_causal_plan.py",
    "infra/gpu03/direction_discovery/engine.py",
    "infra/gpu03/direction_discovery/intervention.py",
    "infra/gpu03/direction_discovery/batched_generation.py",
    "infra/gpu03/direction_discovery/fixed_cache.py",
    "infra/gpu03/activation_dataset/extract_triplet_raw.py",
)
require = prompts.require


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 << 20, "Unsafe JSON input")
    return prompts.decode(path.read_bytes())


def ref(path):
    return {"path": str(Path(path).absolute()), **io.binding(path)}


def verify_ref(reference):
    io.verify_file(reference["path"], reference)
    return Path(reference["path"])


def file_inventory(manifest, sha, required, field="files"):
    path = Path(manifest)
    require(io.sha256(path) == sha, "Package manifest changed")
    value = read_json(path)
    require(set(required) <= set(value[field]), "Required package artifacts missing")
    bindings = {str(path): io.binding(path)}
    for name, binding in value[field].items():
        item = path.parent / name
        require(item.parent == path.parent and name == item.name, "Nonlocal package artifact")
        io.verify_file(item, binding)
        bindings[str(item)] = binding
    return value, bindings


def source_contract(source):
    source = Path(source)
    inventory = read_json(source / "SOURCE_FILES.json")
    require(set(REQUIRED_SOURCE).union(worker.REQUIRED_SOURCE_FILES) <= set(inventory), "Causal source dependency missing")
    for name, binding in inventory.items():
        path = source / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source inventory path")
        io.verify_file(path, binding)
    tree = ast.parse((source / "infra/gpu03/task_specific_discovery/causal_generation.py").read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "PROFILE" for t in node.targets)]
    require(len(values) == 1, "Missing unique literal causal profile")
    profile = ast.literal_eval(values[0])
    require(profile == {"protocol": "task_specific_batched_causal_v1", "batch_size": 8,
        "left_padding": True, "attention_policy": "exclusive_math", "adapter_policy": "unmerged",
        "sampling_policy": "existing_request_local_generator",
        "projection_policy": "fp32_response_predictor_ordinary_centered_v1", "qualified_numerical_equivalence": False},
        "Unregistered numerical profile")
    return inventory, profile


def vector_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_conditions(shortlist, tensors, study_id):
    require(shortlist.get("schema") == "task_specific_verified_offline_shortlist_v1"
        and shortlist.get("status") == "offline_shortlist_compiled"
        and shortlist.get("completed_layers") == 36 and shortlist.get("completed_cells") == 72
        and shortlist.get("candidate_count") == 432 and shortlist.get("causal_validation_claimed") is False,
        "Shortlist lacks complete offline selection proof")
    axes = {a["axis_id"]: a for a in shortlist["axes"]}
    require(1 <= len(axes) == len(shortlist["axes"]) <= 4 and "historical_pc4" in axes, "Invalid axis family count")
    require(axes["historical_pc4"]["layer"] == 21
        and axes["historical_pc4"]["tensor_sha256"] == shortlist_io.PC4_TENSOR_SHA,
        "Historical PC4 identity changed")
    for axis in axes.values():
        q = tensors[axis["tensor_key"]]
        shortlist_io.validate_direction(q)
        require(vector_sha(q) == axis["tensor_sha256"], "Saved axis tensor bytes changed")
    centers = {c["layer"]: c for c in shortlist["centers"]}
    for center in centers.values():
        mu = tensors[center["tensor_key"]]
        require(mu.dtype == np.float32 and mu.shape == (2560,) and np.isfinite(mu).all()
                and vector_sha(mu) == center["tensor_sha256"], "Saved center tensor bytes changed")
    configs = shortlist["operator_configurations"]
    require(1 <= len(configs) <= 6 and len({c["configuration_id"] for c in configs}) == len(configs), "Invalid operator count")
    require(len({(c["axis_id"], c["operator"]) for c in configs}) == len(configs), "Duplicate axis/operator")
    conditions = {BASELINE: {"id": BASELINE, "layer": None, "operator": "ordinary", "control": "zero_strength", "strength": 0.0}}
    metadata = {BASELINE: {"kind": "baseline"}}
    vectors = {key: value.copy() for key, value in tensors.items()}
    randoms, null_pairs, qualifications, pilot = [], [], [BASELINE], [BASELINE]
    for config in configs:
        cid = config["configuration_id"]
        require(config["axis_id"] in axes and config["layer"] == axes[config["axis_id"]]["layer"]
                and config["vector_tensor_key"] == axes[config["axis_id"]]["tensor_key"]
                and config["strengths"] == list(STRENGTHS) and config["scope"] == "response_predictors"
                and config["operator"] in ("ordinary", "centered"), "Config deviates from nominated axis/strength/scope")
        common = {"layer": config["layer"], "operator": config["operator"], "rank": 1,
                  "vector_key": config["vector_tensor_key"]}
        if config["operator"] == "centered":
            require(config["layer"] in centers and config["center_tensor_key"] == centers[config["layer"]]["tensor_key"], "Unbound center")
            common["center_key"] = config["center_tensor_key"]
        pair = {}
        for control in ("zero_strength", "materialization_only"):
            identity = cid + "." + control
            conditions[identity] = {"id": identity, **common, "control": control, "strength": 0.0}
            metadata[identity] = {"kind": control, "configuration_id": cid}
            pair[control] = identity
            qualifications.append(identity)
            if control == "materialization_only":
                pilot.append(identity)
        null_pairs.append(pair)
        for alpha in STRENGTHS:
            identity = cid + ".alpha" + str(int(alpha * 100))
            conditions[identity] = {"id": identity, **common, "control": "project", "strength": alpha}
            metadata[identity] = {"kind": "target", "configuration_id": cid}
            pilot.append(identity)
        for n in range(3):
            seed_material = ["task-specific-isotropic-random-axis-v1", study_id, cid, n]
            seed = int.from_bytes(hashlib.sha256(canonical(seed_material)).digest()[:8], "big")
            rng = np.random.Generator(np.random.PCG64(seed))
            z = rng.standard_normal((2560, 1), dtype=np.float64)
            q = np.asarray(z / np.linalg.norm(z), dtype=np.float32)
            shortlist_io.validate_direction(q)
            key = "random__" + hash_value([cid, n])
            require(key not in vectors, "Random key collision")
            vectors[key] = q
            randoms.append({"configuration_id": cid, "random_index": n, "seed": seed,
                "seed_material": seed_material, "tensor_key": key, "tensor_sha256": vector_sha(q),
                "rng": "numpy.PCG64.standard_normal_FP64_then_L2_normalize_then_FP32",
                "numpy_version": np.__version__, "layer": config["layer"], "rank": 1,
                "operator": config["operator"], "center_key": common.get("center_key")})
            for alpha in STRENGTHS:
                identity = cid + f".random{n}.alpha" + str(int(alpha * 100))
                conditions[identity] = {"id": identity, **common, "vector_key": key, "control": "project", "strength": alpha}
                metadata[identity] = {"kind": "nominal_random", "configuration_id": cid, "random_index": n}
                pilot.append(identity)
    require(len(qualifications) <= 13 and len(pilot) <= 79 and len(set(pilot)) == len(pilot), "Finite condition budget exceeded")
    return {"conditions": conditions, "condition_metadata": metadata, "random_axes": randoms,
            "qualification_condition_ids": qualifications, "pilot_condition_ids": pilot,
            "null_pairs": null_pairs}, vectors


def tokenize(prompt_rows, problem_keys, qualification_keys, tokenizer):
    require(len(problem_keys) == 32 and len(set(problem_keys)) == 32 and qualification_keys == problem_keys[:4], "Pilot/qualification boundaries differ")
    by_key = {}
    for row in prompt_rows:
        key = (row["problem_id_key"], row["protocol"])
        require(key not in by_key and key[0] in problem_keys and key[1] in prompts.PROTOCOLS
            and row["schema"] == prompts.SCHEMA and row["problem_split"] == "direction_fit"
            and row["qualification_member"] == (key[0] in qualification_keys)
            and row["problem_id_key"] == prompts.canonical_json(row["problem_id"])
            and row["dataset_row"]["id"] == row["problem_id"] and row["dataset_row"]["prompt"] == row["prompt"]
            and row["prompt_sha256"] == prompts.prompt_sha256(row["prompt"]), "Prompt identity/protocol differs")
        tokens = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        require(isinstance(tokens, list) and all(type(x) is int and x >= 0 for x in tokens)
            and 0 < len(tokens) <= 1536, "Prompt is overlength or invalid; no truncation/replacement permitted")
        rid = hash_value(["task-specific-prompt-v1", row["study_id"], *key, row["prompt_sha256"]])
        by_key[key] = {**copy.deepcopy(row), "record_id": rid, "prompt_token_ids": tokens,
            "prompt_token_count": len(tokens), "prompt_token_ids_sha256": hash_value(tokens),
            "completion_token_ids": [], "completion_token_count": 0, "input_ids": tokens[:],
            "tokenization_status": "verified_cpu", "completion_status": "not_generated",
            "model_revision": worker.REVISION, "checkpoint_sha256": worker.ADAPTERS["adapter_model.safetensors"]}
    require(len(by_key) == 96, "Incomplete three-protocol pilot")
    return [by_key[key, protocol] for protocol in prompts.PROTOCOLS for key in problem_keys]


def requests_for(rows, condition_ids, conditions, *, qualification):
    selected = [r for r in rows if not qualification or r["qualification_member"]]
    requests = []
    for cid in condition_ids:
        for row in selected:
            for sample in (0, 1):
                coordinate = prompts.request_coordinates(row, sample, conditions[cid])
                requests.append({"request_id": coordinate["request_id"], "record_id": row["record_id"],
                    "problem_id": row["problem_id"], "protocol": row["protocol"], "scope": "primary",
                    "seed": coordinate["generation_seed"], "sample_index": sample, "condition_id": cid})
    require(len({r["request_id"] for r in requests}) == len(requests), "Duplicate frozen request")
    return requests


def qualify_contract(rows, bundle, source_files, vector_sha256):
    problems = [r["problem_id"] for r in rows if r["protocol"] == "fixed" and r["qualification_member"]]
    require(len(problems) == 4, "Qualification problems missing")
    return {"schema": "task_specific_null_qualification_contract_v1", "problem_ids": problems,
        "protocols": list(prompts.PROTOCOLS), "sample_indices": [0, 1], "baseline_condition_id": BASELINE,
        "null_pairs": bundle["null_pairs"], "logits_atol": 1e-5, "logits_rtol": 1e-5,
        "source_files": source_files, "vectors_sha256": vector_sha256}


def prepare(args, tokenizer=None):
    out = Path(args.output)
    require(out.is_absolute() and not out.exists(), "Plan output must be fresh absolute directory")
    for name in ("shortlist_complete", "prompt_manifest", "base_manifest", "model_snapshot", "checkpoint", "source_root", "design"):
        path = Path(getattr(args, name))
        require(path.is_absolute() and path.resolve() == path, "Input paths must be canonical absolute paths")
    complete, bound = file_inventory(args.shortlist_complete, args.shortlist_sha,
        ["SHORTLIST.json", "shortlist.safetensors", "INPUT_VERIFICATION.json", "CANDIDATE_TABLE.json"], "artifacts")
    require(complete["schema"] == "task_specific_shortlist_complete_v1" and complete["status"] == "succeeded"
            and complete["cells"] == 72 and complete["candidates"] == 432 and complete["launched"] is False, "Shortlist closure failed")
    shortlist_dir = Path(args.shortlist_complete).parent
    shortlist = read_json(shortlist_dir / "SHORTLIST.json")
    require(shortlist["fit_manifest_sha256"] == complete["fit_manifest_sha256"], "Shortlist fitting lineage differs")
    require(shortlist["tensor_file"] == {"path": "shortlist.safetensors", **complete["artifacts"]["shortlist.safetensors"]}, "Shortlist tensor ref differs")
    pm, pbound = file_inventory(args.prompt_manifest, args.prompt_sha, ["PROTOCOL.json", "pilot_problem_ids.json", "pilot_prompts.jsonl",
        "fixed992.jsonl", "canonical_nohint992.jsonl", "randomized992.jsonl", "randomized_name_mapping.json"])
    bound.update(pbound)
    pdir = Path(args.prompt_manifest).parent
    protocol = read_json(pdir / "PROTOCOL.json")
    require(pm["schema"] == prompts.SCHEMA and protocol["schema"] == prompts.SCHEMA
        and protocol["status"] == "prompts_frozen_tokenization_pending"
        and protocol["pilot_count"] == 32 and protocol["qualification_count"] == 4
        and protocol["full_row_fixed_roundtrips"] == 992 and protocol["protocols"] == list(prompts.PROTOCOLS)
        and pm["files"]["fixed992.jsonl"]["sha256"] == prompts.FIXED_SHA
        and pm["files"]["canonical_nohint992.jsonl"]["sha256"] == prompts.CANONICAL_SHA
        and protocol["input_bindings"]["splits"]["sha256"] == prompts.SPLIT_SHA, "Prompt package policy/source changed")
    ids = read_json(pdir / "pilot_problem_ids.json")
    require(ids["study_id"] == protocol["study_id"] and ids["selection_domain"] == prompts.SELECTION_DOMAIN, "Selection identity differs")
    source_files, profile = source_contract(args.source_root)
    bound.update({str(Path(args.source_root) / n): b for n, b in source_files.items()})
    bound[str(Path(args.source_root) / "SOURCE_FILES.json")] = io.binding(Path(args.source_root) / "SOURCE_FILES.json")
    require(io.sha256(args.base_manifest) == BASE_MANIFEST_SHA, "Original base manifest differs")
    base = read_json(args.base_manifest)
    require(base["base_revision"] == worker.REVISION and set(base["files"]) == set(worker.BASE_FILES), "Pinned base file set differs")
    for name, binding in base["files"].items():
        path = Path(args.model_snapshot) / name
        io.verify_file(path, binding); bound[str(path)] = binding
    for name, sha in worker.ADAPTERS.items():
        path = Path(args.checkpoint) / name
        require(io.sha256(path) == sha, "Original M60 adapter differs")
        bound[str(path)] = io.binding(path)
    require(io.sha256(args.design) == args.design_sha, "Registered design changed")
    bound[str(Path(args.design))] = io.binding(args.design)
    bound[str(Path(args.base_manifest))] = io.binding(args.base_manifest)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True, trust_remote_code=False)
    rows = tokenize(prompts.rows((pdir / "pilot_prompts.jsonl").read_bytes()), ids["pilot_problem_keys"], ids["qualification_problem_keys"], tokenizer)
    require(all(r["study_id"] == protocol["study_id"] for r in rows), "Prompt study differs")
    from safetensors.numpy import load_file
    bundle, vectors = make_conditions(shortlist, load_file(str(shortlist_dir / "shortlist.safetensors")), protocol["study_id"])
    qualification = requests_for(rows, bundle["qualification_condition_ids"], bundle["conditions"], qualification=True)
    pilot = requests_for(rows, bundle["pilot_condition_ids"], bundle["conditions"], qualification=False)
    require(len(qualification) <= 312 and len(pilot) <= 15168, "Prospective request ceiling exceeded")
    out.mkdir(mode=0o700)
    vector_path = out / "causal_vectors.safetensors"
    io.write_tensors(vector_path, vectors, {"schema": SCHEMA, "study_id": protocol["study_id"]})
    row_path = out / "prepared_rows.jsonl"
    prompts.write_bytes(row_path, b"".join(canonical(r) + b"\n" for r in rows)); row_path.chmod(0o444)
    bound[str(row_path)] = io.binding(row_path); bound[str(vector_path)] = io.binding(vector_path)
    plan = {"schema": SCHEMA, "status": "prepared_not_dispatched", "run_token": Path(ROOT).name,
        "study_id": protocol["study_id"], "source_root": str(Path(args.source_root)), "qualified_source_files": source_files,
        "model_snapshot": str(Path(args.model_snapshot)), "checkpoint": str(Path(args.checkpoint)),
        "prepared_rows": str(row_path), "vectors": str(vector_path), "model_revision": worker.REVISION,
        "sampling": worker.SAMPLING, "batch_size": 8, "profile": profile, "bound_files": bound,
        **bundle, "qualification_requests": qualification, "pilot_requests": pilot,
        "qualification_contract": qualify_contract(rows, bundle, source_files, bound[str(vector_path)]["sha256"]),
        "budgets": {"qualification": len(qualification), "pilot_including_reuse": len(pilot),
            "reusable_pilot_requests": len({r["request_id"] for r in qualification} & {r["request_id"] for r in pilot})},
        "batches": "per-condition, then fixed/randomized/no_loophole, then frozen problem order and sample0/1; no reorder",
        "gaps": shortlist["gaps"], "magnitude_matched_controls": "not part of pilot; later registered stage",
        "generation": False, "evaluation": False, "training": False}
    io.write_json(out / "PLAN.json", plan)
    return ref(out / "PLAN.json")


def reuse_from_qualification(plan, proof_ref):
    path = verify_ref(proof_ref); proof = read_json(path)
    require(proof["schema"] == "task_specific_causal_qualification_v1" and proof["status"] == "qualified_task_specific_causal_v1"
        and proof["profile"] == plan["profile"] and proof["source_files"] == plan["qualified_source_files"]
        and proof["vectors_sha256"] == plan["bound_files"][plan["vectors"]]["sha256"]
        and all(proof[k] is True for k in ("all_returned_requests_retained", "gpu_workers_released", "null_token_parity", "null_logits_within_tolerance")),
        "Exact-profile successful qualification required")
    require(proof["model_files"] == worker.model_bindings(plan, worker.BASE_FILES, "model_snapshot")
        and proof["adapter_files"] == worker.model_bindings(plan, worker.ADAPTERS, "checkpoint"),
        "Qualified model identities changed")
    required_signatures = {hash_value(worker.qualification_signature(c)) for c in plan["conditions"].values()
                           if c.get("layer") is not None}
    require({hash_value(s) for s in proof["qualified_signatures"]} == required_signatures,
            "Qualification operator/rank/layer/center coverage differs")
    expected = {r["request_id"]: r for r in plan["qualification_requests"]}
    require(set(proof["retained_requests"]) == set(expected), "Qualification retained set differs")
    checked, generations, reuse, bound = set(), {}, {}, {str(path): {k: proof_ref[k] for k in ("sha256", "size_bytes")}}
    pilot = {r["request_id"]: r for r in plan["pilot_requests"]}
    for rid, r in expected.items():
        saved = proof["retained_requests"][rid]
        require(saved["request"] == r and saved["request_sha256"] == hash_value(r), "Qualification request identity changed")
        for name in ("generation_file", "primitive_file", "final_file"):
            value = saved[name]
            if value["path"] not in checked:
                verify_ref(value); checked.add(value["path"])
            require(value["path"] not in bound or bound[value["path"]] == {k: value[k] for k in ("sha256", "size_bytes")}, "Conflicting reused file binding")
            bound[value["path"]] = {k: value[k] for k in ("sha256", "size_bytes")}
        filename = saved["generation_file"]["path"]
        if filename not in generations:
            values = [prompts.decode(x) for x in Path(filename).read_bytes().splitlines() if x]
            generations[filename] = {v["request_id"]: v for v in values}
            require(len(generations[filename]) == len(values), "Duplicate retained generation")
        generation = generations[filename].get(rid)
        require(generation is not None and hash_value(generation) == saved["generation_row_sha256"]
            and all(generation.get(k) == v for k, v in r.items()), "Retained generation differs from frozen request")
        if rid in pilot:
            require(pilot[rid] == r, "Shared pilot request differs from qualification")
            reuse[rid] = saved
    require(len(reuse) == plan["budgets"]["reusable_pilot_requests"], "Incomplete qualification reuse")
    return reuse, bound


def task_bundle(plan_ref, stage, output, workers=8, qualification=None):
    path = verify_ref(plan_ref); plan = read_json(path)
    require(plan["schema"] == SCHEMA and plan["status"] == "prepared_not_dispatched", "Unknown prepared plan")
    require(stage in ("qualification", "pilot") and type(workers) is int and 1 <= workers <= 8, "Invalid task stage/worker count")
    require(not Path(output).exists() and Path(output).is_absolute(), "Task bundle needs fresh absolute output")
    require(plan["batch_size"] == 8 and plan["sampling"] == worker.SAMPLING and plan["model_revision"] == worker.REVISION, "Plan science differs")
    for name, binding in plan["bound_files"].items():
        io.verify_file(name, binding)
    requests = plan[stage + "_requests"]
    reuse, extra = {}, {}
    if stage == "pilot":
        require(qualification is not None, "Pilot blocked before numerical qualification")
        reuse, extra = reuse_from_qualification(plan, qualification)
    else:
        require(qualification is None, "Qualification cannot import a later proof")
    remaining = [r for r in requests if r["request_id"] not in reuse]
    # Fixed8 groups; qualification rows are entire leading groups, never fragments.
    groups = [remaining[i:i + 8] for i in range(0, len(remaining), 8)]
    require(all(len(g) == 8 and len({(r["condition_id"], r["protocol"]) for r in g}) == 1 for g in groups), "Batch shape or condition/protocol changed")
    actual_workers = min(workers, len(groups))
    require(actual_workers > 0, "No new tasks; all required requests already returned")
    shards = [[] for _ in range(actual_workers)]
    for n, group in enumerate(groups):
        shards[n % actual_workers].extend(group)
    require(max(map(len, shards)) <= 2048, "Too few lanes for bounded tasks; prepare separately reviewed sequential waves")
    out = Path(output)
    # One preparation per stage of this plan. Even an interrupted preparation
    # remains consumed; an audited recovery must account for its exact requests.
    claim_path = path.parent / (stage.upper() + "_TASK_BINDING.json")
    io.write_json(claim_path, {"schema": "task_specific_stage_task_binding_v1", "plan": plan_ref,
        "stage": stage, "output": str(out), "workers": actual_workers,
        "request_shards": [[hash_value(r) for r in shard] for shard in shards],
        "request_ids": [r["request_id"] for r in remaining], "reuse_ids": sorted(reuse),
        "condition_ids": sorted({r["condition_id"] for r in remaining}),
        "null_signature_count": len({hash_value(worker.qualification_signature(c))
            for c in plan["conditions"].values() if c.get("layer") is not None}),
        "qualification_contract_sha256": hash_value(plan["qualification_contract"]),
        "qualification": qualification, "request_replay_authorized": False})
    out.mkdir(mode=0o700)
    common = {k: copy.deepcopy(plan[k]) for k in ("run_token", "model_revision", "model_snapshot", "checkpoint",
        "prepared_rows", "vectors", "sampling", "batch_size", "profile", "source_root", "qualified_source_files", "conditions")}
    bound = {**plan["bound_files"], **extra, str(path): {k: plan_ref[k] for k in ("sha256", "size_bytes")},
             str(claim_path): io.binding(claim_path)}
    tasks = []
    for lane, requests in enumerate(shards):
        used_conditions = {r["condition_id"] for r in requests}
        task = {**common, "schema": "task_specific_causal_generation_v1", "stage": stage,
            "deadline_seconds": 14400, "requests": requests, "output": str(out / f"worker_{lane:02d}"), "bound_files": bound,
            "conditions": {cid: plan["conditions"][cid] for cid in plan["conditions"] if cid in used_conditions}}
        if stage == "pilot":
            task["qualification"] = qualification["path"]
        task_path = out / f"task_{lane:02d}.json"; io.write_json(task_path, task)
        task_ref = ref(task_path)
        tasks.append({"lane": lane, "task": task_ref, "requests": len(requests),
            "argv": [PYTHON, "-B", "-m", "infra.gpu03.task_specific_discovery.causal_worker", "--task", str(task_path), "--sha256", task_ref["sha256"]]})
    phase = {"schema": "task_specific_causal_phase_inputs_v1", "stage": stage, "plan": plan_ref,
        "run_token": plan["run_token"], "source_root": plan["source_root"], "profile": plan["profile"],
        "tasks": tasks, "conditions": plan["conditions"], "qualified_source_files": plan["qualified_source_files"],
        "qualification_contract": plan["qualification_contract"], "expected_new_requests": len(remaining),
        "expected_reused_requests": len(reuse), "expected_total_requests": len(remaining) + len(reuse),
        "reused_requests": reuse, "bound_files": {**bound, **{t["task"]["path"]: {k: t["task"][k] for k in ("sha256", "size_bytes")} for t in tasks}},
        "dispatch_authorized_by_this_file": False, "gpu_allocation": "parent must bind currently qualified UUIDs and resource supervisor"}
    io.write_json(out / "PHASE_INPUTS.json", phase)
    return ref(out / "PHASE_INPUTS.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    for key in ("shortlist-complete", "shortlist-sha", "prompt-manifest", "prompt-sha", "base-manifest", "model-snapshot", "checkpoint", "source-root", "design", "design-sha", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("tasks")
    for key in ("plan", "sha256", "stage", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--qualification"); p.add_argument("--qualification-sha")
    args = parser.parse_args()
    require(socket.gethostname().split(".")[0] == "gpu-03" and os.getuid() == 3357 and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preparation requires gpu-03 owner CPU-only context")
    require(Path(args.output).is_absolute() and Path(args.output).parent == Path(ROOT)
            and Path(args.output).name.replace("-", "").replace("_", "").isalnum(), "Output must be a fresh named child of this run")
    if args.action == "prepare":
        require(Path(args.source_root) == Path(__file__).resolve().parents[3], "Builder must execute from the bound source root")
        result = prepare(args)
    else:
        plan_ref = ref(args.plan); require(plan_ref["sha256"] == args.sha256, "Plan hash differs")
        require(read_json(args.plan)["source_root"] == str(Path(__file__).resolve().parents[3]), "Task builder must execute from the qualified source root")
        qref = ref(args.qualification) if args.qualification else None
        require((qref is None and args.qualification_sha is None) or qref["sha256"] == args.qualification_sha, "Qualification hash differs")
        result = task_bundle(plan_ref, args.stage, args.output, args.workers, qref)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
