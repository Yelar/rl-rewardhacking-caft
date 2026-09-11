#!/usr/bin/env python3
"""Manifest-bound FP32 triplet extraction using the existing decoder/hook path.

Workers write every assistant token, without pooling. Independent CPU auditors
verify all FP32 subtractions before exact-token temporary raw files are removed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from triplet_regions import base, outcome, require, ids_hash
import extract_delta_activations as engine

LAYER_COUNT = 36
HIDDEN = 2560


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(base.canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append(path, value):
    base.append_jsonl(path, value)


def load_manifest(path, digest, check_files=True):
    require(base.sha256_file(path) == digest, "review manifest hash mismatch")
    m = json.loads(path.read_text())
    require(m["schema_version"] == 1 and m["purpose"] == "checkpoint60_triplet_fp32_activations", "wrong manifest purpose")
    require(m["host"] == "gpu-04", "wrong reviewed host")
    require(m["gpu_ids"] and len(set(m["gpu_ids"])) == len(m["gpu_ids"]), "invalid GPU allocation")
    require(m["scientific"] == {
        "records": 561, "problems": 187, "layers": 36, "hidden_size": 2560,
        "model_id": engine.MODEL_ID, "model_revision": engine.MODEL_REVISION,
        "model_forward_dtype": "bfloat16", "subtraction_dtype": "float32", "storage_dtype": "float32",
        "batch_size": 1, "token_scope": "all_original_completion_tokens", "token_pooling": False,
        "site": "decoder_layer_forward_output_before_final_norm", "attention": "deterministic_math_sdpa",
        "generation": False, "gradients": False, "training": False}, "scientific contract mismatch")
    if check_files:
        for filename, info in m["bound_files"].items():
            f = Path(filename)
            require(f.is_file() and f.stat().st_size == info["size_bytes"] and base.sha256_file(f) == info["sha256"],
                    "bound file changed: " + filename)
    return m


def expected_metadata(row, kind, manifest_hash):
    return {"schema_version": "1", "kind": kind, "manifest_sha256": manifest_hash,
            "record_id": row["record_id"], "record_index": str(row["record_index"]),
            "input_ids_sha256": row["input_ids_sha256"], "completion_sha256": row["completion_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"], "model_revision": engine.MODEL_REVISION,
            "site": "decoder_layer_forward_output_before_final_norm", "layer_axis": "zero_based_blocks_0_through_35",
            "storage_dtype": "float32" if kind == "delta_h" else "bfloat16"}


def record_name(row):
    return f"record_{row['record_index']:06d}.safetensors"


def auxiliary_tensors(row):
    import torch
    count = row["completion_token_count"]
    result = {"input_ids": torch.tensor(row["input_ids"], dtype=torch.int32),
              "sequence_positions": torch.tensor(row["selected_token_positions"], dtype=torch.int32)}
    for key, positions in row["region_mask_completion_positions"].items():
        mask = torch.zeros(count, dtype=torch.bool)
        mask[positions] = True
        result["mask__" + key] = mask
    return result


def atomic_tensor(path, tensors, metadata):
    from safetensors.torch import save_file
    require(not path.exists(), "refuse to overwrite tensor " + str(path))
    temp = path.with_suffix(".writing")
    require(not temp.exists(), "incomplete tensor already exists")
    save_file(tensors, str(temp), metadata=metadata)
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    temp.rename(path)


def configure_torch(gpu=False):
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if gpu:
        require(os.environ.get("CUDA_VISIBLE_DEVICES"), "worker has no bound GPU")
        torch.cuda.set_per_process_memory_fraction(0.65, 0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def capture(decoder, layers, row, pad_id):
    import torch
    require(not torch.is_grad_enabled(), "gradients enabled")
    require(not any(layer.training for layer in layers), "decoder layer in training mode")
    dtypes = []
    def guard(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        require(hidden.dtype == torch.bfloat16 and not hidden.requires_grad, "unexpected native residual dtype/gradient")
        dtypes.append(str(hidden.dtype))
    handles = [layer.register_forward_hook(guard) for layer in layers]
    try:
        result = engine._capture_post_block(decoder, layers, [row], pad_id)[0]
    finally:
        for handle in handles:
            handle.remove()
    require(len(dtypes) == LAYER_COUNT, "not all native dtype guards fired")
    require(tuple(result.shape) == (LAYER_COUNT, row["completion_token_count"], HIDDEN), "capture shape mismatch")
    require(torch.isfinite(result).all().item(), "nonfinite native residual")
    return result.contiguous()


def worker(task_path):
    engine._set_parent_death_signal()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    task = json.loads(task_path.read_text())
    m = load_manifest(Path(task["manifest"]), task["manifest_sha256"], check_files=False)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == str(task["gpu_id"]), "worker GPU differs from task")
    configure_torch(True)
    import torch
    from safetensors.torch import load_file
    rows = list(base.read_jsonl(Path(m["prepared"]) / "prepared_records.jsonl"))
    rows = [rows[index] for index in task["record_indices"]]
    require(all(ids_hash(r["input_ids"]) == r["input_ids_sha256"] for r in rows), "worker input IDs differ")
    destination, raw = Path(task["destination"]), Path(task["raw"])
    destination.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    exclusive_json(raw / "owner.json", {"run_token": m["run_token"], "manifest_sha256": task["manifest_sha256"]})
    log = destination / "journal.jsonl"
    load_task = {"model_snapshot": m["model_snapshot"], "checkpoint": m["checkpoint"]}
    pad_id = int(m["pad_token_id"])
    reports, release = {}, {}
    for adapted in (False, True):
        kind = "h60" if adapted else "h0"
        model, decoder, layers, load_report = engine._load_decoder(load_task, adapted)
        require(not model.training and not any(p.requires_grad for p in model.parameters()), "training/gradients enabled")
        require(not any(getattr(module, "training", False) for module in model.modules()), "dropout/module left training")
        reports[kind] = load_report
        try:
            for n, row in enumerate(rows):
                activation = capture(decoder, layers, row, pad_id)
                if task["qualification"] and n == 0:
                    repeated = capture(decoder, layers, row, pad_id)
                    require(torch.equal(activation, repeated), "native same-input repeatability failure")
                    del repeated
                native_path = raw / (kind + "_" + record_name(row))
                atomic_tensor(native_path, {kind: activation}, expected_metadata(row, kind, task["manifest_sha256"]))
                if adapted:
                    h0 = load_file(str(raw / ("h0_" + record_name(row))))["h0"]
                    delta = activation.float() - h0.float()
                    require(delta.dtype == torch.float32 and torch.isfinite(delta).all().item(), "invalid FP32 subtraction")
                    require(bool(delta.count_nonzero()), "all-zero checkpoint delta")
                    atomic_tensor(destination / record_name(row), {"delta_h": delta, **auxiliary_tensors(row)},
                                  expected_metadata(row, "delta_h", task["manifest_sha256"]))
                    del h0, delta
                append(log, {"kind": kind, "record_index": row["record_index"], "record_id": row["record_id"],
                             "input_ids_sha256": row["input_ids_sha256"], "native_shape": list(activation.shape),
                             "native_dtype": str(activation.dtype), "recorded_at": time.time()})
                del activation
                if (n + 1) % 10 == 0 or n + 1 == len(rows):
                    print(json.dumps({"gpu": task["gpu_id"], "kind": kind, "records": n + 1, "total": len(rows)}), flush=True)
        finally:
            del decoder, layers, model
            release[kind] = engine._release_cuda()
        engine.validate_post_model_cuda_state(release[kind], kind + " model release")
    engine.validate_model_load_reports(reports["h0"], reports["h60"], "triplet worker")
    exclusive_json(destination / "worker_success.json", {
        "status": "succeeded", "record_indices": task["record_indices"], "gpu_id": task["gpu_id"],
        "model_load_reports": reports, "cuda_after_release": release,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "native_repeatability_checked": task["qualification"], "manifest_sha256": task["manifest_sha256"]})


def verify_delta_record(row, path, raw, digest):
    """Called only in the separate CPU auditor after its model worker exits."""
    import torch
    from safetensors import safe_open
    expected_shape = [LAYER_COUNT, row["completion_token_count"], HIDDEN]
    paths = {kind: raw / (kind + "_" + record_name(row)) for kind in ("h0", "h60")}
    absolute_max, absolute_sum, elements = 0.0, 0.0, 0
    fp64_error, bf16_cast_error, bf16_changed, sampled = 0.0, 0.0, 0, 0
    with safe_open(str(paths["h0"]), framework="pt", device="cpu") as b, \
         safe_open(str(paths["h60"]), framework="pt", device="cpu") as a, \
         safe_open(str(path), framework="pt", device="cpu") as d:
        for handle, kind, dtype in ((b, "h0", "BF16"), (a, "h60", "BF16"), (d, "delta_h", "F32")):
            require(handle.metadata() == expected_metadata(row, kind, digest), "tensor metadata mismatch")
            view = handle.get_slice(kind)
            require(view.get_shape() == expected_shape and view.get_dtype() == dtype, "tensor shape/dtype mismatch")
        expected_aux = auxiliary_tensors(row)
        require(set(d.keys()) == {"delta_h", *expected_aux}, "delta tensor keys differ")
        for key, expected in expected_aux.items():
            require(torch.equal(d.get_tensor(key), expected), "saved IDs/mask mismatch: " + key)
        for layer in range(LAYER_COUNT):
            h0, h60, delta = b.get_slice("h0")[layer], a.get_slice("h60")[layer], d.get_slice("delta_h")[layer]
            require(torch.isfinite(h0).all().item() and torch.isfinite(h60).all().item() and torch.isfinite(delta).all().item(), "nonfinite saved tensor")
            expected = h60.float() - h0.float()
            require(torch.equal(expected, delta), "stored delta is not exact FP32 M60-M0")
            absolute_max = max(absolute_max, float(delta.abs().max()))
            absolute_sum += float(delta.abs().sum(dtype=torch.float64))
            elements += delta.numel()
            x, y, z = h60.flatten()[::4093], h0.flatten()[::4093], delta.flatten()[::4093]
            reference = x.double() - y.double()
            require(torch.equal(reference.float(), z), "FP64-rounded subtraction reference differs")
            fp64_error = max(fp64_error, float((reference - z.double()).abs().max()))
            cast_error = (z.to(torch.bfloat16).float() - z).abs()
            bf16_cast_error = max(bf16_cast_error, float(cast_error.max()))
            bf16_changed += int(torch.count_nonzero(cast_error))
            sampled += z.numel()
    require(absolute_max > 0, "all-zero saved record")
    return {"record_id": row["record_id"], "record_index": row["record_index"], "verified": True,
            "shape": expected_shape, "dtype": "float32", "all_elements_exact_fp32_subtraction": True,
            "elements": elements, "mean_absolute_delta": absolute_sum / elements, "max_absolute_delta": absolute_max,
            "sampled_fp64_rounding_max_absolute_error": fp64_error, "sampled_elements": sampled,
            "hypothetical_bf16_cast_max_error": bf16_cast_error, "hypothetical_bf16_cast_changed_elements": bf16_changed,
            "input_ids_sha256": row["input_ids_sha256"],
            "raw_hashes": {kind: base.sha256_file(p) for kind, p in paths.items()},
            "delta_sha256": base.sha256_file(path), "delta_size_bytes": path.stat().st_size}


def audit(task_path):
    configure_torch(False)
    task = json.loads(task_path.read_text())
    m = load_manifest(Path(task["manifest"]), task["manifest_sha256"], check_files=False)
    destination, raw = Path(task["destination"]), Path(task["raw"])
    success = json.loads((destination / "worker_success.json").read_text())
    require(success["status"] == "succeeded" and success["record_indices"] == task["record_indices"], "worker incomplete")
    engine.validate_model_load_reports(success["model_load_reports"]["h0"], success["model_load_reports"]["h60"], "CPU auditor")
    rows = list(base.read_jsonl(Path(m["prepared"]) / "prepared_records.jsonl"))
    journal = destination / "numerical_audit.jsonl"
    require(not journal.exists(), "numerical audit already exists")
    for n, index in enumerate(task["record_indices"]):
        row = rows[index]
        result = verify_delta_record(row, destination / record_name(row), raw, task["manifest_sha256"])
        append(journal, result)
        if (n + 1) % 10 == 0 or n + 1 == len(task["record_indices"]):
            print(json.dumps({"audit_gpu_shard": task["gpu_id"], "records": n + 1, "total": len(task["record_indices"])}), flush=True)
    exclusive_json(destination / "audit_success.json", {"status": "succeeded", "records": len(task["record_indices"]),
                   "manifest_sha256": task["manifest_sha256"], "raw_activations_retained": True})


def gpu_snapshot():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    inventory = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        require(len(parts) == 7, "malformed GPU inventory")
        inventory.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                          "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]),
                          "memory_free_mib": int(parts[5]), "utilization_percent": int(parts[6])})
    require(len(inventory) == 8 and len({r["index"] for r in inventory}) == 8, "missing/duplicate GPU inventory")
    processes = engine.query_compute_processes()
    for r in inventory:
        r["processes"] = processes.get(r["uuid"], [])
    return inventory


def check_devices(m, inventory, allowed=None, selected=None):
    selected = m["gpu_ids"] if selected is None else selected
    by_id = {r["index"]: r for r in inventory}
    username = pwd.getpwuid(os.getuid()).pw_name
    for index in selected:
        require(index in by_id, "selected GPU missing")
        row = by_id[index]
        require(row["uuid"] == m["gpu_uuids"][str(index)] and row["name"] == engine.EXPECTED_GPU_NAME, "GPU identity changed")
        permitted = set() if allowed is None else allowed.get(index, set())
        require(all(p["pid"] in permitted and p["owner"] == username for p in row["processes"]), "foreign or unknown GPU process")
        if not permitted:
            require(row["memory_used_mib"] <= 64 and row["memory_free_mib"] >= 32000 and row["utilization_percent"] <= 1,
                    "selected GPU is not idle")


def safety(m, workers, selected=None):
    require(engine.mem_available_kib() >= m["limits"]["min_available_ram_gib"] * 1024**2, "available host RAM below limit")
    allowed, roots = {}, set()
    for proc, gpu in workers:
        if proc.poll() is None:
            roots.add(proc.pid)
            _, descendants = engine.process_tree_rss_kib({proc.pid})
            allowed[gpu] = set(descendants)
    rss, _ = engine.process_tree_rss_kib(roots)
    require(rss <= m["limits"]["max_worker_rss_gib"] * 1024**2, "worker RSS limit exceeded")
    snapshot = gpu_snapshot()
    check_devices(m, snapshot, allowed, selected)
    require(shutil.disk_usage(Path(m["output"]).parent).free >= 32 * 1024**3, "disk safety reserve exhausted")
    return {"gpu_inventory": snapshot, "available_ram_kib": engine.mem_available_kib(), "worker_rss_kib": rss}


def worker_env(gpu):
    env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": str(gpu) if gpu is not None else "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    return env


def cleanup_raw(task, m):
    destination, raw = Path(task["destination"]), Path(task["raw"])
    require(raw.resolve().is_relative_to(Path(m["runtime"]).resolve()), "raw path outside run runtime")
    require(json.loads((raw / "owner.json").read_text()) == {"run_token": m["run_token"], "manifest_sha256": task["manifest_sha256"]}, "raw ownership mismatch")
    audit_rows = list(base.read_jsonl(destination / "numerical_audit.jsonl"))
    require({r["record_index"] for r in audit_rows} == set(task["record_indices"]) and len(audit_rows) == len(task["record_indices"]), "audit coverage incomplete")
    require(all(r["verified"] and r["all_elements_exact_fp32_subtraction"] for r in audit_rows), "numerical audit did not pass")
    expected_files = {"owner.json"} | {kind + f"_record_{index:06d}.safetensors" for index in task["record_indices"] for kind in ("h0", "h60")}
    require({p.name for p in raw.iterdir()} == expected_files, "unexpected raw directory member")
    require(all(not p.is_symlink() and p.is_file() and p.stat().st_uid == os.getuid() for p in raw.iterdir()), "unsafe raw member")
    for audit_row in audit_rows:
        for kind in ("h0", "h60"):
            p = raw / (kind + f"_record_{audit_row['record_index']:06d}.safetensors")
            require(base.sha256_file(p) == audit_row["raw_hashes"][kind], "raw changed after numerical audit")
    for p in raw.iterdir():
        p.unlink()
    raw.rmdir()
    exclusive_json(destination / "raw_cleanup.json", {"verified": True, "raw_files_removed_after_independent_audit": len(expected_files) - 1})


def supervise(manifest_path, digest):
    m = load_manifest(manifest_path, digest)
    require(socket.gethostname() == m["host"], "wrong host")
    require(pwd.getpwuid(os.getuid()).pw_name == "researcher", "wrong launch user")
    require({n: importlib.metadata.version(n) for n in m["runtime_versions"]} == m["runtime_versions"], "runtime package versions changed")
    output, runtime = Path(m["output"]), Path(m["runtime"])
    require(not output.exists() and not runtime.exists(), "run directories already exist; preserve previous attempt")
    require(shutil.disk_usage(output.parent).free >= m["limits"]["min_start_free_disk_gib"] * 1024**3, "insufficient start disk")
    output.mkdir(mode=0o700)
    runtime.mkdir(mode=0o700)
    (runtime / "tasks").mkdir()
    exclusive_json(output / "run_identity.json", {"run_token": m["run_token"], "manifest_sha256": digest})
    shutil.copytree(m["prepared"], output / "input")
    shutil.copytree(m["source_root"], output / "provenance/source")
    shutil.copyfile(manifest_path, output / "provenance/reviewed_manifest.json")
    shutil.copytree(m["checkpoint"], output / "provenance/checkpoint")
    lock_root = Path(m["stage"]) .parent / "triplet_gpu_locks"
    lock_root.mkdir(exist_ok=True, mode=0o700)
    locks, processes, tasks = [], [], []
    start = time.monotonic()
    deadline = start + m["limits"]["runtime_seconds"]
    def interrupted(*_):
        raise RuntimeError("supervisor interrupted; raw activations retained")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def check_deadline():
        require(time.monotonic() < deadline, "post-allocation deadline exhausted")
    def monitor(active, selected):
        while any(p.poll() is None for p, _ in active):
            check_deadline()
            require(all(p.poll() in (None, 0) for p, _ in active), "worker failed")
            state = safety(m, active, selected)
            append(runtime / "resource_journal.jsonl", {"at": time.time(), **state})
            time.sleep(5)
        require(all(p.returncode == 0 for p, _ in active), "worker exited nonzero")
        # GPU contexts can take a few seconds to disappear after process exit.
        for attempt in range(12):
            try:
                state = safety(m, [], selected)
                break
            except ValueError:
                if attempt == 11:
                    raise
                time.sleep(5)
        append(runtime / "resource_journal.jsonl", {"at": time.time(), "release_verified": selected, **state})
        return state
    def spawn_task(task, cpu_set, flag):
        task_path = runtime / "tasks" / (task["name"] + ".json")
        if not task_path.exists():
            exclusive_json(task_path, task)
        log_path = runtime / (task["name"] + (".model.log" if flag == "--worker" else ".audit.log"))
        log = log_path.open("xb")
        command = ["taskset", "-c", cpu_set, "nice", "-n", "10", "ionice", "-c", "2", "-n", "7",
                   sys.executable, str(Path(__file__).resolve()), flag, str(task_path)]
        proc = subprocess.Popen(command, env=worker_env(task["gpu_id"] if flag == "--worker" else None),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        processes.append(proc)
        return proc
    try:
        for gpu in m["gpu_ids"]:
            f = (lock_root / f"gpu_{gpu}.lock").open("a+")
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(f)
        for iteration in range(13):
            check_deadline()
            state = safety(m, [])
            append(runtime / "qualification_inventory.jsonl", {"at": time.time(), **state})
            if iteration < 12:
                time.sleep(5)
        rows = list(base.read_jsonl(Path(m["prepared"]) / "prepared_records.jsonl"))
        cpu_sets = engine.allocate_cpu_sets(len(m["gpu_ids"]), 2)
        def task(name, gpu, indices, qualification):
            return {"name": name, "manifest": str(manifest_path), "manifest_sha256": digest,
                    "gpu_id": gpu, "record_indices": indices, "qualification": qualification,
                    "destination": str(output / ("qualification" if qualification else "shards") / name),
                    "raw": str(runtime / "raw" / name)}
        q = task("qualification", m["gpu_ids"][0], m["qualification_indices"], True)
        proc = spawn_task(q, cpu_sets[0], "--worker")
        monitor([(proc, q["gpu_id"])], m["gpu_ids"])
        proc = spawn_task(q, cpu_sets[0], "--audit")
        while proc.poll() is None:
            check_deadline()
            safety(m, [])
            time.sleep(5)
        require(proc.returncode == 0, "real-model qualification numerical audit failed")
        # Retain qualification raw activations durably for independent future audits.
        shutil.copytree(q["raw"], output / "qualification/raw")
        cleanup_raw(q, m)
        print("QUALIFICATION_PASSED: three classes, native repeatability, all FP32 elements independently checked", flush=True)
        active = []
        for worker_index, gpu in enumerate(m["gpu_ids"]):
            t = task(f"gpu_{gpu}", gpu, list(range(worker_index, len(rows), len(m["gpu_ids"]))), False)
            tasks.append(t)
            safety(m, active)
            p = spawn_task(t, cpu_sets[worker_index], "--worker")
            active.append((p, gpu))
            # Poll safety during stagger; never load all five models simultaneously.
            for _ in range(2):
                time.sleep(5)
                safety(m, active)
        release_state = monitor(active, m["gpu_ids"])
        exclusive_json(output / "gpu_release.json", {"verified": True, "gpu_ids": m["gpu_ids"], "state": release_state})
        print("ALL_MODEL_WORKERS_EXITED; selected GPUs released; independent numerical audits starting", flush=True)
        # GPUs are now released. Later foreign users are outside this allocation.
        auditors = [spawn_task(t, cpu_sets[i], "--audit") for i, t in enumerate(tasks)]
        while any(p.poll() is None for p in auditors):
            check_deadline()
            require(all(p.poll() in (None, 0) for p in auditors), "numerical auditor failed")
            require(engine.mem_available_kib() >= m["limits"]["min_available_ram_gib"] * 1024**2, "audit host RAM limit")
            time.sleep(5)
        require(all(p.returncode == 0 for p in auditors), "numerical auditor exited nonzero")
        all_audits = []
        for t in tasks:
            destination = Path(t["destination"])
            audit_rows = list(base.read_jsonl(destination / "numerical_audit.jsonl"))
            for r in audit_rows:
                r["tensor_path"] = str((destination / f"record_{r['record_index']:06d}.safetensors").relative_to(output))
            all_audits.extend(audit_rows)
            cleanup_raw(t, m)
        require(len(all_audits) == 561 and {r["record_index"] for r in all_audits} == set(range(561)), "final record coverage failure")
        by_index = {r["record_index"]: r for r in all_audits}
        # Compare full FP32 production tensors with independent qualification captures.
        from safetensors import safe_open
        import torch
        torch.set_num_threads(1)
        for index in m["qualification_indices"]:
            with safe_open(str(output / by_index[index]["tensor_path"]), framework="pt", device="cpu") as a, \
                 safe_open(str(Path(q["destination"]) / f"record_{index:06d}.safetensors"), framework="pt", device="cpu") as b:
                for layer in range(36):
                    require(torch.equal(a.get_slice("delta_h")[layer], b.get_slice("delta_h")[layer]), "qualification/production repeat mismatch")
        base.atomic_write_jsonl(output / "activation_index.jsonl", [by_index[i] for i in range(561)])
        exclusive_json(output / "extraction_summary.json", {
            "status": "succeeded", "records": 561, "problems": 187, "completion_tokens": sum(r["completion_token_count"] for r in rows),
            "layers": 36, "hidden_size": 2560, "dtype": "float32", "raw_activation_dtype": "bfloat16",
            "token_averaging": False, "all_elements_verified_from_saved_h0_h60": True,
            "qualification_and_production_bitwise_equal": True, "temporary_raw_removed_after_audit": True,
            "qualification_raw_retained": "qualification/raw", "model_workers_exit_codes": [p.returncode for p, _ in active],
            "numerical_auditor_exit_codes": [p.returncode for p in auditors],
            "gpu_release_verified": True, "gpu_ids": m["gpu_ids"], "manifest_sha256": digest,
            "split_records": dict(Counter(r["problem_split"] for r in rows)),
            "max_fp64_rounding_error": max(r["sampled_fp64_rounding_max_absolute_error"] for r in all_audits),
            "max_hypothetical_bf16_cast_error": max(r["hypothetical_bf16_cast_max_error"] for r in all_audits),
            "elapsed_seconds": time.monotonic() - start})
        base.atomic_write_json(output / "artifact_manifest.json", base.file_manifest(output))
        print("EXTRACTION_PRODUCER_SUCCEEDED", flush=True)
    except BaseException as error:
        engine.terminate_workers(processes)
        exclusive_json(output / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error),
                       "run_token": m["run_token"], "temporary_raw_preserved": True})
        raise
    finally:
        for lock in locks:
            lock.close()


def verify_package(root):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    require(base.file_manifest(root) == manifest, "artifact manifest mismatch")
    summary = json.loads((root / "extraction_summary.json").read_text())
    require(summary["status"] == "succeeded" and summary["all_elements_verified_from_saved_h0_h60"] and
            summary["qualification_and_production_bitwise_equal"] and summary["temporary_raw_removed_after_audit"], "extraction not positively successful")
    require(summary["model_workers_exit_codes"] == [0] * len(summary["gpu_ids"]) and
            summary["numerical_auditor_exit_codes"] == [0] * len(summary["gpu_ids"]), "worker/auditor exit failure")
    rows = list(base.read_jsonl(root / "input/prepared_records.jsonl"))
    index = list(base.read_jsonl(root / "activation_index.jsonl"))
    require(len(rows) == len(index) == 561, "record count mismatch")
    seen, problem_splits = set(), {}
    for row, entry in zip(rows, index):
        require(entry["record_index"] == row["record_index"] and entry["record_id"] == row["record_id"] and entry["verified"], "index mismatch")
        require(row["record_id"] not in seen, "duplicate record")
        seen.add(row["record_id"])
        pid = row["problem_id_key"]
        require(problem_splits.setdefault(pid, row["problem_split"]) == row["problem_split"], "problem split leakage")
        path = root / entry["tensor_path"]
        require(manifest["files"][entry["tensor_path"]]["sha256"] == entry["delta_sha256"], "audited tensor changed")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            require(f.metadata() == expected_metadata(row, "delta_h", summary["manifest_sha256"]), "tensor identity metadata mismatch")
            require(f.get_slice("delta_h").get_shape() == [36, row["completion_token_count"], 2560] and
                    f.get_slice("delta_h").get_dtype() == "F32", "tensor shape/dtype mismatch")
            aux = auxiliary_tensors(row)
            require(set(f.keys()) == {"delta_h", *aux}, "unexpected tensor keys")
            for key, expected in aux.items():
                require(torch.equal(f.get_tensor(key), expected), "tensor mask/ID mismatch")
    require(len(problem_splits) == 187, "problem count mismatch")
    return {"status": "independently_verified", "records": 561, "problems": 187,
            "dtype": "float32", "completion_tokens": sum(r["completion_token_count"] for r in rows),
            "problem_split_overlap": 0, "manifest_files": len(manifest["files"]),
            "artifact_manifest_sha256": base.sha256_file(root / "artifact_manifest.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--worker", type=Path)
    group.add_argument("--audit", type=Path)
    group.add_argument("--supervise", type=Path)
    group.add_argument("--verify-package", type=Path)
    p.add_argument("--manifest-sha256")
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.audit:
        audit(args.audit)
    elif args.supervise:
        supervise(args.supervise, args.manifest_sha256)
    else:
        print(json.dumps(verify_package(args.verify_package), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
