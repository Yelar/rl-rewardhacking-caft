"""CPU-only, one-layer-at-a-time fitting from closed fixed2176 native caches.

This does not launch extraction, generation or a model. The caller starts it only
after extraction and resource release are verified. A fresh, hash-bound task
selects one modulo-eight layer shard. Completed cells are immutable; interruption
does not create a successful completion marker or regenerate any model request.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import fitting

LAYERS, HIDDEN, SHARDS = 36, 2560, 8
require = fitting.require


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path)
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def stat_identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verify_file(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing/symlink input: " + str(path))
    before = stat_identity(path)
    require(before[2] == expected["size_bytes"] and sha256(path) == expected["sha256"],
            "Input bytes differ: " + str(path))
    after = stat_identity(path)
    require(before == after, "Input changed while hashing")
    return after


def build_cache_verification(entries):
    """One independent CPU verifier can hash the closed corpus before all shards."""
    files = {}
    for rid in sorted(entries):
        entry = entries[rid]
        for part in [*entry["models"].values(), entry["delta"]]:
            path = part["absolute_path"]
            require(path not in files, "Duplicate native/delta payload path")
            identity = verify_file(path, part)
            files[path] = {"sha256": part["sha256"], "size_bytes": part["size_bytes"],
                           "stat_identity": list(identity)}
    return {"schema": "task_specific_cpu_native_verification_v1", "status": "succeeded", "files": files}


def verified_identities(task, entries):
    """Reuse only a bound verifier receipt, still checking current bytes' identity."""
    name = task.get("cache_verification")
    if name is None:
        return {}
    require(name in task["bound_files"], "Unbound independent cache verification receipt")
    receipt = json.loads(Path(name).read_text())
    require(receipt["schema"] == "task_specific_cpu_native_verification_v1" and
            receipt["status"] == "succeeded", "Unsuccessful independent cache verifier")
    expected = {p["absolute_path"]: p for e in entries.values() for p in [*e["models"].values(), e["delta"]]}
    require(set(expected) == set(receipt["files"]), "Verification receipt does not cover exact selected corpus")
    verified = {}
    for path, part in expected.items():
        saved = receipt["files"][path]
        require(saved["sha256"] == part["sha256"] and saved["size_bytes"] == part["size_bytes"] and
                len(saved["stat_identity"]) == 5 and all(type(v) is int for v in saved["stat_identity"]),
                "Verification receipt binding differs from native index")
        identity = tuple(saved["stat_identity"])
        if Path(path).is_file() and not Path(path).is_symlink() and stat_identity(path) == identity:
            verified[path] = identity
        else:
            verified[path] = verify_file(path, part)
    return verified


def positions(row):
    """Use the frozen legacy mask, including its code-end clipping."""
    n = row["completion_token_count"]
    region = row["regions"]["evaluator"]
    require(region is not None, "No fabricated evaluator anchors")
    b = region["first_executable_completion_token"]
    predictor = region["logit_source_completion_token"]
    transition = row["region_mask_completion_positions"]["evaluator__transition"]
    require(type(b) is int and type(predictor) is int and 0 <= predictor == b - 1 < n,
            "Body predictor is not b-1 inside completion")
    require(isinstance(transition, list) and transition and
            all(type(t) is int and 0 <= t < n for t in transition) and
            transition == sorted(set(transition)) and predictor in transition,
            "Invalid frozen evaluator transition positions")
    saved = region["window_completion_positions"]["transition"]
    require(transition == sorted({t for t in saved if t is not None}),
            "Region and frozen transition mask disagree")
    end = region["end_completion_token_exclusive"]
    expected = [t for t in range(b - 4, b + 12) if 0 <= t < n and (t < b or t < end)]
    require(transition == expected, "Frozen transition differs from declared legacy [-4,+12) window")
    return {"body_predictor": [predictor], "transition": transition}


def reduce_record(row, h0, h60, prompt0, prompt60):
    """Reduce one native layer after FP32 conversion, never mean-before-subtract.

    Response-predictor center uses last-prompt + completion[:-1], exactly n
    predictor states for n recorded response tokens. Statistics for that center
    accumulate FP64; anchor vectors retain the registered FP32 token reduction.
    """
    n = row["completion_token_count"]
    require(isinstance(h0, np.ndarray) and isinstance(h60, np.ndarray) and
            h0.ndim == h60.ndim == 2 and h0.shape == h60.shape and
            len(h0) == n and h0.shape[1] > 0 and h0.dtype == h60.dtype == np.float32,
            "Expected one aligned FP32 completion layer")
    d = h0.shape[1]
    for value in (h0, h60, prompt0, prompt60):
        require(isinstance(value, np.ndarray) and value.dtype == np.float32 and
                np.isfinite(value).all(), "Nonfinite/non-FP32 native layer")
    require(prompt0.shape == prompt60.shape == (d,), "Missing final prompt predictor")
    anchors = {}
    for name, ps in positions(row).items():
        left, right = h0[ps], h60[ps]
        anchors[name] = {"h60": right.mean(0, dtype=np.float32),
                         "delta": np.subtract(right, left, dtype=np.float32).mean(0, dtype=np.float32)}
    center = (prompt60.astype(np.float64) + h60[:-1].sum(0, dtype=np.float64)) / n
    require(all(np.isfinite(v).all() for a in anchors.values() for v in a.values()) and
            np.isfinite(center).all(), "Nonfinite reduction")
    return {"anchors": anchors, "response_h60_mean": center, "predictor_count": n}


def response_center(rows, record_means):
    """Equal correctness stratum / problem / retained response / predictor mass."""
    require(record_means.ndim == 2 and len(record_means) == len(rows) and
            np.isfinite(record_means).all(), "Invalid response predictor record means")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        require(row["problem_split"] == "direction_fit" and
                type(row["ground_truth_correctness"]) is bool and
                type(row["is_test_modification_harmful"]) is bool,
                "Unknown or held-out center row")
        if not row["is_test_modification_harmful"]:
            require(type(row.get("strict_reward_hack")) is bool,
                    "Explicit strict-RH label required for clean response center")
        if not row["is_test_modification_harmful"] and not row["strict_reward_hack"]:
            groups[row["ground_truth_correctness"], row["problem_id_key"]].append(i)
    strata = {s: sorted(p for t, p in groups if t == s) for s in fitting.STRATA}
    report = {"schema": "fitting_only_response_predictor_center_v1",
              "policy": "equal_correctness_stratum_then_problem_then_response_then_predictor",
              "positions": "last_prompt_plus_completion_excluding_final_token",
              "activation_kind": "h60", "accumulation": "float64_from_native_bf16_converted_fp32",
              "clean_definition": "known_benign_modification_and_not_strict_reward_hack",
              "support": {str(s).lower(): {"problems": len(strata[s]), "problem_ids": strata[s],
                          "responses": sum(len(groups[s, p]) for p in strata[s])} for s in fitting.STRATA}}
    if any(not strata[s] for s in fitting.STRATA):
        return None, {**report, "status": "unsupported", "reason": "Missing benign correctness stratum"}
    by_stratum = []
    for s in fitting.STRATA:
        by_problem = []
        for p in strata[s]:
            indices = sorted(groups[s, p], key=lambda i: rows[i]["record_id"])
            by_problem.append(record_means[indices].mean(0, dtype=np.float64))
        by_stratum.append(np.stack(by_problem).mean(0, dtype=np.float64))
    center = np.mean(by_stratum, axis=0, dtype=np.float64).astype(np.float32)
    require(np.isfinite(center).all(), "Nonfinite response center")
    return center, {**report, "status": "fitted", "record_ids": sorted(
        rows[i]["record_id"] for ids in groups.values() for i in ids)}


def read_task(path, digest):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU fitting requires CUDA_VISIBLE_DEVICES='' explicitly")
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha256(path) == digest, "Fitting task changed")
    task = json.loads(path.read_text())
    require(task["schema"] == "task_specific_cpu_fitting_v1", "Wrong fitting task schema")
    require(task["shard_count"] == SHARDS and type(task["shard_index"]) is int and
            0 <= task["shard_index"] < SHARDS, "Expected one modulo-eight layer shard")
    require(type(task["deadline_seconds"]) is int and 0 < task["deadline_seconds"] <= 86400,
            "Invalid CPU fitting deadline")
    cfg = fitting.configuration(task.get("config"))
    for name, expected in task["bound_files"].items():
        verify_file(name, expected)
    required = [task["prepared_rows"], *task["worker_indexes"], task["extraction_complete"]]
    require(set(required) <= set(task["bound_files"]), "Unbound fitting input/index/completion receipt")
    receipt = json.loads(Path(task["extraction_complete"]).read_text())
    require(receipt["status"] == "succeeded" and receipt["gpu_workers_released"] is True,
            "Extraction and GPU release are not verified complete")
    rows = [json.loads(line) for line in Path(task["prepared_rows"]).read_text().splitlines()]
    require(rows and len({r["record_id"] for r in rows}) == len(rows) and
            len({r["record_index"] for r in rows}) == len(rows), "Empty/duplicate prepared rows")
    require(rows == sorted(rows, key=lambda r: r["record_index"]), "Prepared row order changed")
    for row in rows:
        require(row["problem_split"] == "direction_fit", "Held-out record in CPU fitting")
        positions(row)
    require(fitting.working_memory_bound(len(rows), HIDDEN) <= cfg["max_working_bytes"],
            "Fitting inputs exceed configured array memory")
    entries = {}
    for name in task["worker_indexes"]:
        index_path = Path(name)
        for line in index_path.read_text().splitlines():
            entry = json.loads(line)
            rid = entry["record_id"]
            require(rid not in entries and entry["verified"] is True, "Duplicate/unverified cache record")
            for part in [*entry["models"].values(), entry["delta"]]:
                relative = Path(part["tensor_path"])
                require(not relative.is_absolute() and ".." not in relative.parts,
                        "Cache payload must stay within its index directory")
                part["absolute_path"] = str(index_path.parent / relative)
            entries[rid] = entry
    require(set(entries) == {r["record_id"] for r in rows}, "Cache index does not exactly cover prepared rows")
    require(all(entries[r["record_id"]]["record_index"] == r["record_index"] for r in rows),
            "Cache/global row indices differ")
    return task, rows, entries, cfg


def read_layer(row, entry, layer, verified):
    """Hash each closed file once, then load exactly one tensor-layer slice."""
    from safetensors import safe_open
    arrays, prompts = {}, {}
    parts = {**entry["models"], "delta": entry["delta"]}
    for kind in ("h0", "h60", "delta"):
        part = parts[kind]
        path = Path(part["absolute_path"])
        require(path.is_file() and not path.is_symlink(), "Native file missing or replaced by symlink")
        if str(path) not in verified or stat_identity(path) != verified[str(path)]:
            verified[str(path)] = verify_file(path, part)
        before = verified[str(path)]
        key, prompt_key = ("delta_h", "prompt_final_delta") if kind == "delta" else (kind, "prompt_final")
        with safe_open(str(path), framework="pt", device="cpu") as cache:
            meta = cache.metadata()
            require(meta["record_id"] == row["record_id"] and meta["record_index"] == str(row["record_index"]) and
                    meta["input_ids_sha256"] == row["input_ids_sha256"] and
                    meta["completion_sha256"] == row["completion_sha256"] and
                    meta["model_revision"] == row["model_revision"] and
                    meta["adapter_sha256"] == ("none" if kind == "h0" else row["checkpoint_sha256"]) and
                    meta["site"] == "decoder_layer_forward_output_before_final_norm" and
                    meta["padded_sequence_length"] == "2176" and meta["attention_backend"] == "torch_sdpa_MATH_only" and
                    meta["prompt_final_saved"] == "true", "Native cache provenance/profile mismatch")
            tensor = cache.get_slice(key)
            require(tensor.get_shape() == [LAYERS, row["completion_token_count"], HIDDEN] and
                    tensor.get_dtype() == ("F32" if kind == "delta" else "BF16") and
                    cache.get_slice(prompt_key).get_shape() == [LAYERS, HIDDEN], "Native cache shape/dtype mismatch")
            require(cache.get_tensor("input_ids").tolist() == row["input_ids"] and
                    cache.get_tensor("sequence_positions").tolist() == row["selected_token_positions"],
                    "Native cache original input/position mismatch")
            expected_mask = np.zeros(row["completion_token_count"], dtype=np.bool_)
            expected_mask[positions(row)["transition"]] = True
            require(np.array_equal(cache.get_tensor("mask__evaluator__transition").numpy(), expected_mask),
                    "Native evaluator mask differs from prepared mask")
            arrays[kind] = tensor[layer].float().numpy()
            prompts[kind] = cache.get_slice(prompt_key)[layer].float().numpy()
        require(stat_identity(path) == before, "Native cache changed during layer read")
    require(np.array_equal(arrays["delta"], np.subtract(arrays["h60"], arrays["h0"], dtype=np.float32)) and
            np.array_equal(prompts["delta"], np.subtract(prompts["h60"], prompts["h0"], dtype=np.float32)),
            "Saved FP32 delta disagrees with raw native subtraction")
    return reduce_record(row, arrays["h0"], arrays["h60"], prompts["h0"], prompts["h60"])


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o444)


def write_tensors(path, tensors, metadata):
    from safetensors.numpy import load_file, save_file
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite fitted tensors")
    temporary = path.with_suffix(".writing")
    require(not temporary.exists(), "Incomplete tensor output already exists")
    values = {key: np.ascontiguousarray(value) for key, value in tensors.items()}
    save_file(values, str(temporary), metadata=metadata)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    restored = load_file(str(temporary))
    require(set(restored) == set(values) and all(
        restored[key].dtype == value.dtype and np.array_equal(restored[key], value, equal_nan=True)
        for key, value in values.items()), "Fitted tensor readback differs")
    temporary.rename(path)
    path.chmod(0o444)


def run(task_path, digest):
    started = time.monotonic()
    task, rows, entries, cfg = read_task(task_path, digest)
    # Torch is required only for BF16 CPU decoding; never initialize CUDA.
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(task["output"])
    output.mkdir(parents=False, exist_ok=False, mode=0o700)
    layers = [l for l in range(LAYERS) if l % SHARDS == task["shard_index"]]
    write_json(output / "STARTED.json", {"schema": "task_specific_cpu_fitting_started_v1",
               "task_sha256": digest, "layers": layers, "pid": os.getpid(), "cuda_visible_devices": ""})
    verified, artifacts, completed = verified_identities(task, entries), {}, []

    def within_deadline():
        require(time.monotonic() - started < task["deadline_seconds"], "CPU fitting deadline reached")

    def save(name, value, tensors=None):
        if tensors is not None:
            tensor_path = output / (name + ".safetensors")
            write_tensors(tensor_path, tensors, {"task_sha256": digest, "schema": "task_specific_cpu_fitting_tensors_v1"})
            artifacts[tensor_path.name] = binding(tensor_path)
            value["tensor_file"] = {"path": tensor_path.name, **artifacts[tensor_path.name]}
        path = output / (name + ".json")
        write_json(path, value)
        artifacts[path.name] = binding(path)

    for layer in layers:
        within_deadline()
        vectors = {rep: {kind: np.empty((len(rows), HIDDEN), np.float32) for kind in fitting.KINDS}
                   for rep in fitting.REPRESENTATIONS}
        center_means = np.empty((len(rows), HIDDEN), np.float64)
        for i, row in enumerate(rows):
            within_deadline()
            reduced = read_layer(row, entries[row["record_id"]], layer, verified)
            for rep, values in reduced["anchors"].items():
                for kind, value in values.items():
                    vectors[rep][kind][i] = value
            center_means[i] = reduced["response_h60_mean"]
        center, report = response_center(rows, center_means)
        report["layer"] = layer
        save(f"L{layer:02d}.response_center", report, {} if center is None else {"h60_response_center": center})
        del center_means, center
        for rep in fitting.REPRESENTATIONS:
            within_deadline()
            report, tensors = fitting.fit_cell(rows, vectors.pop(rep), layer=layer, representation=rep, config=cfg)
            save(f"L{layer:02d}.evaluator.{rep}", report, tensors)
            completed.append({"layer": layer, "representation": rep})
            print(json.dumps({"completed_cells": len(completed), "layer": layer, "representation": rep,
                              "eligible_candidates": sum(c["eligible"] for c in report["candidates"])}), flush=True)
            del report, tensors
        del vectors
    within_deadline()
    require(len(completed) == 2 * len(layers), "Layer shard incomplete")
    result = {"schema": "task_specific_cpu_fitting_complete_v1", "status": "succeeded", "task_sha256": digest,
              "shard_index": task["shard_index"], "shard_count": SHARDS, "layers": layers,
              "records": len(rows), "completed_cells": completed, "artifacts": artifacts,
              "verified_raw_and_delta_files": len(verified), "elapsed_seconds": time.monotonic() - started,
              "causal_validation_claimed": False, "model_execution": False}
    write_json(output / "COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    result = run(args.task, args.sha256)
    print(json.dumps({k: result[k] for k in ("status", "shard_index", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
