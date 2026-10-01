"""Read-only CPU audit of terminal causal null-control generation.

No model, tokenizer, scorer, subprocess, network or CUDA call is made. NumPy is
used only to read retained safetensor bytes and compare first-token logits.
A positive proof requires an outcome-independent manifest contract, complete
request/journal/export coverage, exact null token parity, declared numerical
agreement, and positive supervisor process/GPU release. It certifies only the
finite qualification tranche, never behavioral efficacy or general equivalence.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import stat
import struct

from .io_contract import canonical, hash_value, ContractError
from . import causal_worker as worker

SCHEMA = "task_specific_causal_qualification_v1"
PROTOCOLS = ["fixed", "randomized", "no_loophole"]
PROFILE = {"protocol": "task_specific_batched_causal_v1", "batch_size": 8,
           "left_padding": True, "attention_policy": "exclusive_math", "adapter_policy": "unmerged",
           "sampling_policy": "existing_request_local_generator",
           "projection_policy": "fp32_response_predictor_ordinary_centered_v1", "qualified_numerical_equivalence": False}
REQUIRED_SOURCES = set(worker.REQUIRED_SOURCE_FILES)
MAX_REQUESTS = 312


def require(ok, message):
    if not ok:
        raise ContractError(message)


def decode(data):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def bad(_):
        raise ContractError("Nonfinite JSON")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class Evidence:
    def __init__(self):
        self.files = {}
        self.identities = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.resolve() == path and not path.is_symlink(), "Noncanonical evidence path")
        if str(path) not in self.files:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "Nonregular/linked evidence")
                digest = hashlib.sha256()
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    for block in iter(lambda: stream.read(4 << 20), b""):
                        digest.update(block)
                require(identity(before) == identity(os.fstat(fd)), "Evidence changed during hashing")
            finally:
                os.close(fd)
            self.files[str(path)] = {"sha256": digest.hexdigest(), "size_bytes": before.st_size}
            self.identities[str(path)] = identity(before)
        if expected is not None:
            require(self.files[str(path)] == expected, "Evidence size/SHA differs: " + str(path))
        return {"path": str(path), **self.files[str(path)]}

    def data(self, path, limit=16 << 20, expected=None):
        ref = self.bind(path, expected)
        require(ref["size_bytes"] <= limit, "Evidence size bound exceeded")
        raw = Path(path).read_bytes()
        require(len(raw) == ref["size_bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "Evidence changed while read")
        return raw

    def json(self, path, limit=16 << 20, expected=None, canonical_required=False):
        raw = self.data(path, limit, expected)
        value = decode(raw)
        require(not canonical_required or canonical(value) + b"\n" == raw, "Noncanonical JSON evidence")
        return value

    def jsonl(self, path, limit=64 << 20, row_limit=MAX_REQUESTS, canonical_required=False):
        raw = self.data(path, limit)
        require(not raw or raw.endswith(b"\n"), "Truncated JSONL evidence")
        lines = raw.splitlines()
        require(len(lines) <= row_limit and all(lines), "JSONL coverage bound/blank line")
        values = [decode(line) for line in lines]
        require(all(isinstance(row, dict) for row in values), "Nonobject JSONL row")
        require(not canonical_required or all(canonical(row) == line for row, line in zip(values, lines)), "Noncanonical JSONL")
        return values

    def unchanged(self):
        for name, expected in self.identities.items():
            path = Path(name)
            require(not path.is_symlink() and identity(path.stat()) == expected, "Evidence changed before proof closure")


def signature(condition):
    require(set(condition) <= {"id", "layer", "operator", "control", "strength", "vector_key", "center_key", "rank"},
            "Unrecognized condition field")
    layer, operator = condition.get("layer"), condition.get("operator")
    require(type(layer) is int and 0 <= layer < 36 and operator in ("ordinary", "centered"), "Invalid null signature")
    require(isinstance(condition.get("vector_key"), str) and condition["vector_key"], "Missing vector key")
    require(type(condition.get("rank", 1)) is int and condition.get("rank", 1) == 1, "Null rank differs")
    result = {"layer": layer, "operator": operator, "rank": 1}
    if operator == "centered":
        require(isinstance(condition.get("center_key"), str) and condition["center_key"], "Missing center key")
        result["center_key"] = condition["center_key"]
    else:
        require("center_key" not in condition, "Ordinary null unexpectedly has a center")
    return result


def contract(value):
    require(value["schema"] == "task_specific_null_qualification_contract_v1", "Unknown qualification contract")
    problems = value["problem_ids"]
    require(isinstance(problems, list) and len(problems) == 4 and len({canonical(p) for p in problems}) == 4,
            "Qualification needs four distinct fitting problems")
    require(value["protocols"] == PROTOCOLS and value["sample_indices"] == [0, 1], "Qualification grid differs")
    require(type(value["logits_atol"]) in (int, float) and type(value["logits_rtol"]) in (int, float)
            and value["logits_atol"] == value["logits_rtol"] == 1e-5, "Frozen logit tolerance differs")
    require(isinstance(value["null_pairs"], list) and 1 <= len(value["null_pairs"]) <= 6, "Expected one to six null pairs")
    names = [value["baseline_condition_id"]]
    for pair in value["null_pairs"]:
        require(set(pair) == {"zero_strength", "materialization_only"}, "Incomplete null pair")
        names += [pair["zero_strength"], pair["materialization_only"]]
    require(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names), "Duplicate qualification condition")
    require(REQUIRED_SOURCES <= set(value["source_files"]), "Qualification omits consumed source")
    return set(names)


def load_tensors(path, evidence, expected=None):
    """Strict CPU safetensors reader; accepts only finite dense floating tensors."""
    import numpy as np
    data = evidence.data(path, 128 << 20, expected)
    require(len(data) >= 10, "Truncated safetensors")
    size = struct.unpack("<Q", data[:8])[0]
    require(1 < size <= 1 << 20 and 8 + size <= len(data), "Invalid safetensors header length")
    header = decode(data[8:8 + size])
    require(isinstance(header, dict), "Invalid tensor header")
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict) and all(type(k) is str and type(v) is str for k, v in metadata.items()), "Invalid tensor metadata")
    tensors, spans = {}, []
    for name, value in header.items():
        require(isinstance(value, dict) and set(value) == {"dtype", "shape", "data_offsets"}, "Invalid tensor entry")
        shape, offsets, dtype = value["shape"], value["data_offsets"], value["dtype"]
        require(isinstance(shape, list) and 1 <= len(shape) <= 3 and all(type(n) is int and n > 0 for n in shape), "Invalid tensor shape")
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int for n in offsets), "Invalid tensor offsets")
        require(dtype in ("BF16", "F16", "F32", "F64"), "Unsupported tensor dtype")
        width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}[dtype]
        count = math.prod(shape)
        require(0 <= offsets[0] < offsets[1] <= len(data) - 8 - size and offsets[1] - offsets[0] == count * width,
                "Tensor byte span differs")
        spans.append(tuple(offsets))
        view = memoryview(data)[8 + size + offsets[0]:8 + size + offsets[1]]
        if dtype == "BF16":
            array = (np.frombuffer(view, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            array = np.frombuffer(view, dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype])
        require(bool(np.isfinite(array).all()), "Nonfinite retained tensor")
        tensors[name] = {"dtype": dtype, "array": array.reshape(shape)}
    end = 0
    for start, stop in sorted(spans):
        require(start == end, "Tensor overlap/gap")
        end = stop
    require(end == len(data) - 8 - size and tensors, "Unmapped tensor bytes")
    return tensors


def journal_records(folder, task, digest, evidence):
    """Read closed journal without adopting or writing any orphan event."""
    allowed = {"manifest.json", "writer.lock", "primitives", "finals", "started.jsonl", "primitive_returned.jsonl", "final_recorded.jsonl"}
    require({p.name for p in folder.iterdir()} == allowed, "Unknown/missing journal member")
    hashes = {r["request_id"]: hash_value(r) for r in task["requests"]}
    manifest = evidence.json(folder / "manifest.json", canonical_required=True)
    require(manifest == {"schema": 1, "run_token": task["run_token"], "input_sha256": digest,
                         "request_hashes": hashes, "max_payload_bytes": 8 << 20}, "Journal manifest binding differs")
    lock = os.open(folder / "writer.lock", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        evidence.bind(folder / "writer.lock")
        stages = {}
        for stage in ("started", "primitive_returned", "final_recorded"):
            rows = evidence.jsonl(folder / (stage + ".jsonl"), canonical_required=True)
            ids = [r["request_id"] for r in rows]
            require(len(ids) == len(set(ids)) and set(ids) == set(hashes), "Journal coverage differs")
            stages[stage] = {r["request_id"]: r for r in rows}
            for rid, row in stages[stage].items():
                expected = {"schema": 1, "run_token": task["run_token"], "input_sha256": digest,
                            "request_id": rid, "request_sha256": hashes[rid], "stage": stage}
                if stage != "started":
                    expected["file_sha256"] = row["file_sha256"]
                require(row == expected, "Journal event identity differs")
        names = {hashlib.sha256(rid.encode()).hexdigest() + ".json" for rid in hashes}
        for directory in ("primitives", "finals"):
            require({p.name for p in (folder / directory).iterdir()} == names, "Missing/extra durable payload")
        result = {}
        for rid in hashes:
            base = {"schema": 1, "run_token": task["run_token"], "input_sha256": digest,
                    "request_id": rid, "request_sha256": hashes[rid]}
            values, refs = {}, {}
            for kind, directory, stage in (("primitive", "primitives", "primitive_returned"), ("final", "finals", "final_recorded")):
                path = folder / directory / (hashlib.sha256(rid.encode()).hexdigest() + ".json")
                ref = evidence.bind(path)
                require(ref["sha256"] == stages[stage][rid]["file_sha256"], "Journal payload hash differs")
                envelope = evidence.json(path, 8 << 20, canonical_required=True)
                expected = {**base, "kind": kind, "payload": envelope["payload"]}
                if kind == "final":
                    expected["primitive_file_sha256"] = stages["primitive_returned"][rid]["file_sha256"]
                require(envelope == expected, "Payload predecessor/identity differs")
                values[kind], refs[kind] = envelope["payload"], ref
            result[rid] = {**values, "primitive_file": refs["primitive"], "final_file": refs["final"]}
        return result
    finally:
        os.close(lock)


def coord(request):
    return (canonical(request["problem_id"]).decode(), request["protocol"], request["sample_index"])


def batches(requests):
    groups = OrderedDict()
    for request in requests:
        groups.setdefault(request["condition_id"], []).append(request)
    return [rows[start:start + 8] for rows in groups.values() for start in range(0, len(rows), 8)]


def audit(manifest_path, digest):
    import numpy as np
    evidence = Evidence()
    manifest_path = Path(manifest_path).absolute()
    manifest_ref = evidence.bind(manifest_path)
    require(manifest_ref["sha256"] == digest, "Phase manifest SHA differs")
    m = evidence.json(manifest_path)
    require(m["schema"] == "gpu03_task_specific_resource_guard_v1" and m["host"] == "gpu-03" and m["uid"] == 3357,
            "Wrong accepted host/supervisor")
    cfg = m["qualification_contract"]
    condition_ids = contract(cfg)
    for name, binding in m["bound_files"].items():
        evidence.bind(Path(name), binding)
    root = Path(m["source_root"])
    for name, binding in cfg["source_files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Source name escapes staged root")
        require(m["bound_files"].get(str(root / name)) == binding, "Source omitted from phase binding")
        evidence.bind(root / name, binding)
    runtime = Path(m["runtime"])
    release_ref = evidence.bind(runtime / "SUPERVISOR_RESULT.json")
    release = evidence.json(runtime / "SUPERVISOR_RESULT.json")
    names = [w["name"] for w in m["workers"]]
    require(len(names) == len(set(names)) and 1 <= len(names) <= 8, "Invalid worker coverage")
    require(release["run_token"] == m["run_token"] and release["manifest_sha256"] == digest and
            release["status"] == "workers_exited_zero" and release["owned_processes_released"] is True and
            release["gpu_release_verified"] is True and release.get("scientific_success_verified") is False and
            set(release["exit_codes"]) == set(names) and all(type(c) is int and c == 0 for c in release["exit_codes"].values()),
            "Positive supervisor exit/process/GPU release missing")
    require(m["gpu_uuids"] and len(set(m["gpu_uuids"].values())) == len(m["gpu_uuids"]), "Invalid GPU allocation")
    assigned = [str(w["gpu_id"]) for w in m["workers"]]
    require(sorted(assigned) == sorted(m["gpu_uuids"]), "GPU/worker coverage differs")
    for index, uuid in m["gpu_uuids"].items():
        row = release["gpus"][index]
        require(row["uuid"] == uuid and row["pids"] == [] and 0 <= row["utilization"] <= 2 and
                0 <= row["total_mib"] - row["free_mib"] <= 1024, "Recorded GPU release is not positive")
    launched = evidence.jsonl(runtime / "workers.jsonl", row_limit=8)
    require(len(launched) == len(names) and {r["name"] for r in launched} == set(names), "Worker launch identities incomplete")
    launched = {r["name"]: r for r in launched}
    conditions, retained, coverage, traces, summaries = {}, {}, {}, {}, []
    source_profile = vectors_ref = model_files = adapter_files = None
    vectors = None
    for item in m["workers"]:
        argv = item["argv"]
        require(item["cwd"] == str(root) and item["env"]["PYTHONPATH"] == str(root), "Worker imported source root differs")
        require(len(argv) == 8 and argv[1:5] == ["-B", "-m", "infra.gpu03.task_specific_discovery.causal_worker", "--task"]
                and argv[6] == "--sha256", "Unknown worker command")
        task_path, task_digest = Path(argv[5]), argv[7]
        require(m["bound_files"][str(task_path)]["sha256"] == task_digest, "Task omitted/changed in phase binding")
        evidence.bind(task_path, m["bound_files"][str(task_path)])
        task = evidence.json(task_path)
        require(task["schema"] == "task_specific_causal_generation_v1" and task["stage"] == "qualification" and
                task["run_token"] == m["run_token"] and task["model_revision"] == worker.REVISION and
                task["sampling"] == worker.SAMPLING and task["profile"] == PROFILE and task["batch_size"] == 8,
                "Qualification task science/profile differs")
        require(task["qualified_source_files"] == cfg["source_files"], "Task source identity differs")
        for filename, binding in task["bound_files"].items():
            evidence.bind(Path(filename), binding)
        current_model = {n: task["bound_files"][str(Path(task["model_snapshot"]) / n)] for n in worker.BASE_FILES}
        current_adapter = {n: task["bound_files"][str(Path(task["checkpoint"]) / n)] for n in worker.ADAPTERS}
        require(all(current_adapter[n]["sha256"] == sha for n, sha in worker.ADAPTERS.items()), "Original M60 differs")
        vector_ref = evidence.bind(Path(task["vectors"]), task["bound_files"][task["vectors"]])
        require(vector_ref["sha256"] == cfg["vectors_sha256"], "Vector file differs from frozen contract")
        if source_profile is None:
            source_profile, model_files, adapter_files, vectors_ref = task["profile"], current_model, current_adapter, vector_ref
            vectors = load_tensors(Path(task["vectors"]), evidence)
        require((source_profile, model_files, adapter_files, vectors_ref) ==
                (task["profile"], current_model, current_adapter, vector_ref), "Worker model/profile/vector identities differ")
        model_config = evidence.json(Path(task["model_snapshot"]) / "config.json")
        vocab = model_config["vocab_size"]
        require(type(vocab) is int and 0 < vocab <= 262144, "Unbounded vocabulary")
        prepared = evidence.jsonl(Path(task["prepared_rows"]), row_limit=2048)
        rows = {r["record_id"]: r for r in prepared}
        require(len(rows) == len(prepared), "Duplicate prepared row")
        requests = task["requests"]
        require(0 < len(requests) <= MAX_REQUESTS and len({r["request_id"] for r in requests}) == len(requests), "Duplicate task request")
        for cid, condition in task["conditions"].items():
            require(cid in condition_ids and (cid not in conditions or conditions[cid] == condition), "Conflicting/unregistered condition")
            conditions[cid] = condition
        out = Path(task["output"])
        complete_ref = evidence.bind(out / "COMPLETE.json")
        complete = evidence.json(out / "COMPLETE.json", canonical_required=True)
        started = evidence.json(out / "STARTED.json", canonical_required=True)
        uuid = m["gpu_uuids"][str(item["gpu_id"])]
        require(item["env"]["CUDA_VISIBLE_DEVICES"] == uuid and started ==
                {"task_sha256": task_digest, "pid": launched[item["name"]]["pid"], "requests": len(requests),
                 "gpu_uuid": uuid, "stage": "qualification"} and launched[item["name"]]["gpu_id"] == item["gpu_id"],
                "Worker launch/start/GPU identity differs")
        require(complete["status"] == "succeeded" and complete["stage"] == "qualification" and
                complete["task_sha256"] == task_digest and complete["requests"] == len(requests) and
                complete["all_returned_requests_retained"] is True and complete["evaluation_performed"] is False and
                complete["causal_success_claimed"] is False and complete["cuda_release"] ==
                {"allocated_bytes": 32 << 20, "reserved_bytes": 32 << 20}, "Worker terminal/release proof differs")
        load = complete["model_load_report"]
        require(set(load) == {"with_adapter", "active_adapters", "lora_parameter_tensors", "nonzero_lora_parameter_tensors",
                              "lora_parameter_elements", "base_model_class"} and load["with_adapter"] is True and
                isinstance(load["active_adapters"], list) and load["active_adapters"] and
                all(isinstance(s, str) and s for s in load["active_adapters"]) and
                all(type(load[k]) is int for k in ("lora_parameter_tensors", "nonzero_lora_parameter_tensors", "lora_parameter_elements")) and
                0 < load["nonzero_lora_parameter_tensors"] <= load["lora_parameter_tensors"] and load["lora_parameter_elements"] > 0 and
                isinstance(load["base_model_class"], str) and load["base_model_class"], "Active nonzero adapter load not proven")
        request_batches = batches(requests)
        artifacts = complete["artifacts"]
        expected_artifacts = {"generations.jsonl"} | {f"batch_{i:04d}.first_logits.safetensors" for i in range(len(request_batches))}
        require(set(artifacts) == expected_artifacts and {p.name for p in out.iterdir()} ==
                expected_artifacts | {"STARTED.json", "COMPLETE.json", "journal"}, "Incomplete/unexpected worker output")
        for name, binding in artifacts.items():
            evidence.bind(out / name, binding)
        saved = journal_records(out / "journal", task, task_digest, evidence)
        exports = evidence.jsonl(out / "generations.jsonl", canonical_required=True)
        require([r["request_id"] for r in exports] == [r["request_id"] for r in requests], "Export coverage/order differs")
        exported = {r["request_id"]: r for r in exports}
        for batch_index, batch in enumerate(request_batches):
            trace_name = f"batch_{batch_index:04d}.first_logits.safetensors"
            tensor = load_tensors(out / trace_name, evidence)
            require(set(tensor) == {"logits"} and tensor["logits"]["dtype"] in ("BF16", "F32") and
                    tensor["logits"]["array"].shape == (len(batch), vocab), "First logits dtype/shape differs")
            for lane, request in enumerate(batch):
                rid, cid = request["request_id"], request["condition_id"]
                require(rid not in retained and cid in condition_ids, "Duplicate/unregistered global request")
                row, pair = rows[request["record_id"]], saved[rid]
                require(row["problem_split"] == "direction_fit" and request["scope"] == "primary" and
                        request["problem_id"] == row["problem_id"] and request["protocol"] == row["protocol"] and
                        request["sample_index"] in (0, 1) and type(request["sample_index"]) is int and
                        type(request["seed"]) is int and 0 <= request["seed"] < 2**63, "Wrong fitting/protocol/seed request")
                require(set(pair["primitive"]) == {"result"}, "Unknown generation primitive")
                result = pair["primitive"]["result"]
                ids = result["completion_token_ids"]
                require(isinstance(ids, list) and 1 <= len(ids) <= 1536 and all(type(x) is int and 0 <= x < vocab for x in ids)
                        and result["generated_token_ids"] == ids and result["fixed_completion_prefix_token_count"] == 0 and
                        result["batch_size_actual"] == len(batch) and result["batch_profile"] == PROFILE and
                        result["intervention_strength"] == task["conditions"][cid]["strength"] and
                        result["stop_reason"] == ("eos" if ids[-1] in worker.SAMPLING["eos_token_ids"] else "length") and
                        (ids[-1] in worker.SAMPLING["eos_token_ids"] or len(ids) == 1536), "Incomplete/changed returned tokens")
                final = pair["final"]
                expected_trace = {"path": trace_name, **artifacts[trace_name], "request_ids": [r["request_id"] for r in batch],
                                  "logical_positions_verified": True, "eos_exclusion_verified": True}
                require(set(final) == {"returned_row", "batch_index", "batch_row", "first_logits", "batch_wall_seconds"} and
                        final["returned_row"] is True and final["batch_index"] == batch_index and final["batch_row"] == lane and
                        final["first_logits"] == expected_trace and type(final["batch_wall_seconds"]) in (float, int)
                        and math.isfinite(final["batch_wall_seconds"]) and final["batch_wall_seconds"] >= 0,
                        "Final batch/trace retention differs")
                expected_export = {**request, "problem_id_key": canonical(row["problem_id"]).decode(),
                    "problem_split": row["problem_split"], "prompt": row["prompt"], "prompt_sha256": hash_value(row["prompt"]),
                    "engine_prompt_token_ids": row["prompt_token_ids"], "generation_seed": request["seed"],
                    "completion": result["completion"], "completion_sha256": hashlib.sha256(result["completion"].encode()).hexdigest(),
                    "completion_token_ids": ids, "result": result, "input_task_sha256": task_digest,
                    "raw_primitive_sha256": hash_value(pair["primitive"])}
                require(exported[rid] == expected_export, "Flat generation export differs from immutable primitive")
                key = (cid, coord(request))
                require(key not in coverage, "Duplicate condition/problem/protocol/sample")
                coverage[key] = {"request": request, "row": row, "result": result, "logits": tensor["logits"]["array"][lane].copy(),
                                 "logits_dtype": tensor["logits"]["dtype"], "lane": lane,
                                 "batch_coordinates": [coord(r) for r in batch]}
                retained[rid] = {"request": request, "request_sha256": hash_value(request),
                                 "generation_file": evidence.bind(out / "generations.jsonl"),
                                 "generation_row_sha256": hash_value(exported[rid]),
                                 "primitive_file": pair["primitive_file"], "final_file": pair["final_file"]}
        summaries.append({"worker": item["name"], "task": evidence.bind(task_path), "complete": complete_ref, "requests": len(requests)})
    coordinates = set(itertools.product([canonical(p).decode() for p in cfg["problem_ids"]], PROTOCOLS, [0, 1]))
    require(set(conditions) == condition_ids and set(coverage) == set(itertools.product(condition_ids, coordinates)),
            "Full qualification condition/problem/protocol/sample coverage missing")
    baseline = cfg["baseline_condition_id"]
    b = conditions[baseline]
    require(b.get("layer") is None and b["operator"] == "ordinary" and b["control"] == "zero_strength" and
            b["strength"] == 0 and set(b) <= {"id", "layer", "operator", "control", "strength"}, "Baseline is not a no-hook path")
    signatures, comparisons = [], []
    for names in cfg["null_pairs"]:
        z, mat = conditions[names["zero_strength"]], conditions[names["materialization_only"]]
        require(z["control"] == "zero_strength" and mat["control"] == "materialization_only" and
                z["strength"] == mat["strength"] == 0 and signature(z) == signature(mat) and
                z["vector_key"] == mat["vector_key"], "Mismatched null pair")
        sig = signature(z)
        if sig not in signatures:
            signatures.append(sig)
        q = vectors[z["vector_key"]]
        require(q["dtype"] == "F32" and q["array"].shape == (2560, 1) and
                abs(float(np.sum(q["array"].astype(np.float64) ** 2)) - 1.) <= 2e-5, "Null Q is not a finite FP32 unit axis")
        if sig["operator"] == "centered":
            center = vectors[sig["center_key"]]
            require(center["dtype"] == "F32" and center["array"].shape == (2560,), "Null center shape/dtype differs")
        for control in (names["zero_strength"], names["materialization_only"]):
            worst = 0.
            for coordinate in sorted(coordinates):
                ref, value = coverage[(baseline, coordinate)], coverage[(control, coordinate)]
                require(ref["request"]["seed"] == value["request"]["seed"] and ref["row"]["prompt"] == value["row"]["prompt"] and
                        ref["row"]["prompt_token_ids"] == value["row"]["prompt_token_ids"] and
                        ref["lane"] == value["lane"] and ref["batch_coordinates"] == value["batch_coordinates"],
                        "Null pairing prompt/seed/physical batch lane differs")
                require(ref["result"]["completion_token_ids"] == value["result"]["completion_token_ids"] and
                        ref["result"]["completion"] == value["result"]["completion"], "Null token/text parity failed")
                require(ref["logits_dtype"] == value["logits_dtype"], "Null logit dtype differs")
                a, b = ref["logits"].astype(np.float64), value["logits"].astype(np.float64)
                delta = np.abs(a - b)
                worst = max(worst, float(delta.max()))
                require(bool(np.all(delta <= cfg["logits_atol"] + cfg["logits_rtol"] * np.abs(a))), "Null logit tolerance failed")
            comparisons.append({"condition_id": control, "compared_requests": len(coordinates), "max_abs_first_logit_difference": worst})
    evidence.unchanged()
    return {"schema": SCHEMA, "status": "qualified_task_specific_causal_v1", "run_token": m["run_token"],
            "manifest": manifest_ref, "supervisor_release": release_ref, "workers": summaries,
            "profile": source_profile, "source_files": cfg["source_files"], "vectors_sha256": vectors_ref["sha256"],
            "model_files": model_files, "adapter_files": adapter_files, "sampling": worker.SAMPLING,
            "qualified_signatures": signatures, "tested_null_pairs": cfg["null_pairs"], "tested_conditions": conditions,
            "problem_ids": cfg["problem_ids"], "protocols": PROTOCOLS, "sample_indices": [0, 1],
            "conditions": len(conditions), "requests": len(retained), "retained_requests": retained,
            "all_returned_requests_retained": True, "gpu_workers_released": True, "null_token_parity": True,
            "null_logits_within_tolerance": True, "logits_atol": cfg["logits_atol"], "logits_rtol": cfg["logits_rtol"],
            "comparisons": comparisons, "independently_hashed_files": evidence.files,
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generation_or_evaluation_performed": False, "behavioral_success_claimed": False,
            "scope_limit": "Finite null tranche; first-step logits and entire returned tokens. No later-step logits or model execution repeated."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    proof = audit(args.manifest, args.sha256)
    path = Path(args.output)
    with path.open("xb") as stream:
        stream.write(canonical(proof) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)
    print(json.dumps({"status": proof["status"], "requests": proof["requests"], "output": str(path)}))


if __name__ == "__main__":
    main()
