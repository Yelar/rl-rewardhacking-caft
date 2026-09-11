#!/usr/bin/env python3
"""Raw M0/M60 post-block activations. No cross-model subtraction or pooling.

Native BF16 tensors are retained permanently, with independent CPU audits.
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
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


def worker_env(gpu):
    env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": str(gpu) if gpu is not None else "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    return env


def scientific_contract():
    return {"records": 561, "problems": 187, "layers": 36, "hidden_size": 2560,
            "model_id": engine.MODEL_ID, "model_revision": engine.MODEL_REVISION,
            "model_forward_dtype": "bfloat16", "storage_dtype": "bfloat16",
            "models": ["h0", "h60"], "compute_differences": False, "retain_raw": True,
            "batch_size": 1, "token_scope": "all_original_completion_tokens", "token_pooling": False,
            "site": "decoder_layer_forward_output_before_final_norm", "attention": "deterministic_math_sdpa",
            "generation": False, "gradients": False, "training": False}


def load_manifest(path, digest, check_files=True):
    require(base.sha256_file(path) == digest, "review manifest hash mismatch")
    m = json.loads(path.read_text())
    require(m["schema_version"] == 1 and m["purpose"] == "checkpoint60_triplet_raw_activations", "wrong manifest purpose")
    require(m["host"] == "gpu-04" and m["scientific"] == scientific_contract(), "scientific/host contract mismatch")
    require(m["gpu_ids"] and len(set(m["gpu_ids"])) == len(m["gpu_ids"]), "invalid GPU allocation")
    root = Path("/scratch/researcher/codex_runs")
    for key in ("stage", "output", "runtime"):
        path_value = Path(m[key])
        require(path_value.parent == root and path_value.name.startswith(m["run_token"]) and not path_value.is_symlink(), "unsafe run path")
    require(Path(m["source_root"]).is_relative_to(Path(m["stage"])), "source outside stage")
    require(m["limits"]["runtime_seconds"] <= 14400 and m["limits"]["min_start_free_disk_gib"] >= 160, "invalid limits")
    if check_files:
        for filename, info in m["bound_files"].items():
            f = Path(filename)
            require(f.is_file() and f.stat().st_size == info["size_bytes"] and base.sha256_file(f) == info["sha256"], "bound file changed: " + filename)
    return m


def expected_metadata(row, kind, digest):
    require(kind in ("h0", "h60"), "raw-only tensor kind")
    return {"schema_version": "1", "kind": kind, "manifest_sha256": digest,
            "record_id": row["record_id"], "record_index": str(row["record_index"]),
            "input_ids_sha256": row["input_ids_sha256"], "completion_sha256": row["completion_sha256"],
            "adapter_sha256": row["checkpoint_sha256"] if kind == "h60" else "none",
            "model_revision": engine.MODEL_REVISION,
            "site": "decoder_layer_forward_output_before_final_norm", "layer_axis": "zero_based_blocks_0_through_35",
            "token_axis": "all_original_completion_tokens", "storage_dtype": "bfloat16"}


def validate_rows(rows):
    require(len(rows) == 561 and len({r["record_id"] for r in rows}) == 561, "need 561 unique records")
    groups = {}
    for i, r in enumerate(rows):
        require(r["record_index"] == i, "record index changed")
        require(r["input_ids"] == r["prompt_token_ids"] + r["completion_token_ids"], "sequence is not prompt + completion")
        require(ids_hash(r["input_ids"]) == r["input_ids_sha256"], "input ID hash mismatch")
        require(r["selected_token_positions"] == list(range(r["prompt_token_count"], len(r["input_ids"]))) and
                r["selected_token_mask"] == [True] * r["completion_token_count"], "incomplete completion-token coverage")
        require(len(r["completion_token_ids"]) == r["completion_token_count"], "completion count mismatch")
        groups.setdefault(r["problem_id_key"], []).append(r)
    require(len(groups) == 187, "need 187 problems")
    for group in groups.values():
        outcome.validate_core_group(group)
        require(len({r["problem_split"] for r in group}) == 1, "problem split leakage")
    require(Counter(r["problem_split"] for r in rows) == {"direction_fit":351,"configuration_validation":111,"untouched_test":99}, "split assignments changed")


def verify_raw(row, path, kind, digest, captured=None):
    """Validate one model's tensor independently; never load the other model."""
    import torch
    from safetensors import safe_open
    shape = [LAYER_COUNT, row["completion_token_count"], HIDDEN]
    maxima = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        require(f.metadata() == expected_metadata(row, kind, digest), "raw metadata mismatch")
        require(f.get_slice(kind).get_shape() == shape and f.get_slice(kind).get_dtype() == "BF16", "raw shape/dtype mismatch")
        aux = auxiliary_tensors(row)
        require(set(f.keys()) == {kind, *aux}, "unexpected raw tensor keys")
        for key, value in aux.items():
            require(torch.equal(f.get_tensor(key), value), "raw saved IDs/mask mismatch: " + key)
        for layer in range(LAYER_COUNT):
            value = f.get_slice(kind)[layer]
            require(torch.isfinite(value).all().item(), "nonfinite raw activation")
            maximum = float(value.abs().max())
            require(maximum > 0, "all-zero raw layer")
            maxima.append(maximum)
            if captured is not None:
                require(torch.equal(value, captured[layer]), "raw save/readback changed native values")
    return {"shape": shape, "dtype": "bfloat16", "all_finite": True,
            "layer_max_abs": maxima, "sha256": base.sha256_file(path), "size_bytes": path.stat().st_size}


def worker(task_path):
    engine._set_parent_death_signal()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    task = json.loads(task_path.read_text())
    m = load_manifest(Path(task["manifest"]), task["manifest_sha256"], False)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == str(task["gpu_id"]), "GPU differs from task")
    configure_torch(True)
    import torch
    all_rows = list(base.read_jsonl(Path(m["prepared"]) / "prepared_records.jsonl"))
    validate_rows(all_rows)
    rows = [all_rows[i] for i in task["record_indices"]]
    destination = Path(task["destination"])
    destination.mkdir(parents=True, exist_ok=False)
    log = destination / "journal.jsonl"
    reports, release = {}, {}
    for adapted in (False, True):
        kind = "h60" if adapted else "h0"
        (destination / kind).mkdir()
        model, decoder, layers, report = engine._load_decoder({"model_snapshot":m["model_snapshot"], "checkpoint":m["checkpoint"]}, adapted)
        require(not any(p.requires_grad for p in model.parameters()) and not any(module.training for module in model.modules()), "training/gradients enabled")
        reports[kind] = report
        try:
            for n, row in enumerate(rows):
                started = time.monotonic()
                activation = capture(decoder, layers, row, int(m["pad_token_id"]))
                if task["qualification"] and n == 0:
                    repeat = capture(decoder, layers, row, int(m["pad_token_id"]))
                    require(torch.equal(activation, repeat), "native repeatability failure")
                    del repeat
                path = destination / kind / record_name(row)
                atomic_tensor(path, {kind:activation, **auxiliary_tensors(row)}, expected_metadata(row, kind, task["manifest_sha256"]))
                result = verify_raw(row, path, kind, task["manifest_sha256"], activation)
                append(log, {"kind":kind, "record_index":row["record_index"], "record_id":row["record_id"],
                             "input_ids_sha256":row["input_ids_sha256"], "native_readback_bitwise_equal":True,
                             "elapsed_seconds":time.monotonic()-started, **result})
                del activation
                print(json.dumps({"gpu":task["gpu_id"], "kind":kind, "records":n+1, "total":len(rows)}), flush=True)
        finally:
            del decoder, layers, model
            release[kind] = engine._release_cuda()
        engine.validate_post_model_cuda_state(release[kind], kind+" release")
    engine.validate_model_load_reports(reports["h0"], reports["h60"], "raw worker")
    exclusive_json(destination / "worker_success.json", {"status":"succeeded", "record_indices":task["record_indices"],
        "gpu_id":task["gpu_id"], "model_load_reports":reports, "cuda_after_release":release,
        "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved(),
        "native_repeatability_checked":task["qualification"], "manifest_sha256":task["manifest_sha256"]})


def audit(task_path):
    engine._set_parent_death_signal()
    configure_torch(False)
    task = json.loads(task_path.read_text())
    m = load_manifest(Path(task["manifest"]), task["manifest_sha256"], False)
    destination = Path(task["destination"])
    success = json.loads((destination / "worker_success.json").read_text())
    require(success["status"] == "succeeded" and success["record_indices"] == task["record_indices"], "worker incomplete")
    engine.validate_model_load_reports(success["model_load_reports"]["h0"], success["model_load_reports"]["h60"], "raw auditor")
    rows = list(base.read_jsonl(Path(m["prepared"]) / "prepared_records.jsonl"))
    journal = list(base.read_jsonl(destination / "journal.jsonl"))
    by_key = {(r["kind"],r["record_index"]):r for r in journal}
    expected = {(kind,i) for kind in ("h0","h60") for i in task["record_indices"]}
    require(set(by_key) == expected and len(journal) == len(expected), "worker journal incomplete/duplicated")
    require(not (destination / "raw_audit.jsonl").exists(), "audit journal exists")
    for i in task["record_indices"]:
        row = rows[i]
        results = {}
        for kind in ("h0","h60"):
            result = verify_raw(row, destination / kind / record_name(row), kind, task["manifest_sha256"])
            producer = by_key[(kind,i)]
            require(producer["native_readback_bitwise_equal"] and all(producer[k] == v for k,v in result.items()), "raw file differs from producer readback")
            results[kind] = result
        append(destination / "raw_audit.jsonl", {"record_index":i,"record_id":row["record_id"],"verified":True,"models":results})
    exclusive_json(destination / "audit_success.json", {"status":"succeeded","records":len(task["record_indices"]),
                   "manifest_sha256":task["manifest_sha256"],"raw_activations_retained":True,"differences_computed":False})


def check_devices(m, inventory, allowed=None, selected=None, settling=None):
    selected = m["gpu_ids"] if selected is None else selected
    by_id = {r["index"]:r for r in inventory}
    for index in selected:
        require(index in by_id, "selected GPU missing")
        row = by_id[index]
        require(row["uuid"] == m["gpu_uuids"][str(index)] and row["name"] == engine.EXPECTED_GPU_NAME, "GPU identity changed")
        permitted = (allowed or {}).get(index,set())
        require(all(p["pid"] in permitted and p["owner"] == pwd.getpwuid(os.getuid()).pw_name for p in row["processes"]), "foreign or unknown GPU process")
        if not permitted and index not in (settling or set()):
            require(row["memory_used_mib"] <= 64 and row["memory_free_mib"] >= 32000 and row["utilization_percent"] <= 1, "selected GPU is not idle")


def safety(m, workers, selected=None, allow_settling=True):
    require(engine.mem_available_kib() >= m["limits"]["min_available_ram_gib"]*1024**2, "available host RAM below limit")
    allowed, roots, settling = {},set(),set()
    for proc,gpu in workers:
        if proc.poll() is None:
            roots.add(proc.pid)
            _,descendants = engine.process_tree_rss_kib({proc.pid})
            allowed[gpu] = set(descendants)
        elif proc.returncode == 0 and allow_settling:
            if not hasattr(proc,"raw_exit_seen_at"):
                proc.raw_exit_seen_at = time.monotonic()
            if time.monotonic() - proc.raw_exit_seen_at < 60:
                settling.add(gpu)
    rss,_ = engine.process_tree_rss_kib(roots)
    require(rss <= m["limits"]["max_worker_rss_gib"]*1024**2, "worker RSS exceeded")
    inventory = gpu_snapshot()
    check_devices(m,inventory,allowed,selected,settling)
    require(shutil.disk_usage(Path(m["output"]).parent).free >= 32*1024**3, "disk reserve exhausted")
    return {"gpu_inventory":inventory,"available_ram_kib":engine.mem_available_kib(),"worker_rss_kib":rss}


def supervise(manifest_path,digest):
    m = load_manifest(manifest_path,digest)
    require(socket.gethostname() == m["host"] and pwd.getpwuid(os.getuid()).pw_name == "researcher", "wrong host/user")
    require({n:importlib.metadata.version(n) for n in m["runtime_versions"]} == m["runtime_versions"], "runtime changed")
    output,runtime = Path(m["output"]),Path(m["runtime"])
    require(not output.exists() and not runtime.exists(), "run directory exists; preserve previous attempt")
    require(shutil.disk_usage(output.parent).free >= m["limits"]["min_start_free_disk_gib"]*1024**3, "start disk below limit")
    rows = list(base.read_jsonl(Path(m["prepared"])/"prepared_records.jsonl"))
    validate_rows(rows)
    output.mkdir(mode=0o700);runtime.mkdir(mode=0o700)
    (runtime/"tasks").mkdir()
    exclusive_json(output/"run_identity.json",{"run_token":m["run_token"],"manifest_sha256":digest})
    shutil.copytree(m["prepared"],output/"input")
    shutil.copytree(m["source_root"],output/"provenance/source")
    shutil.copytree(Path(m["stage"])/"environment",output/"provenance/environment")
    shutil.copyfile(manifest_path,output/"provenance/reviewed_manifest.json")
    shutil.copytree(m["checkpoint"],output/"provenance/checkpoint")
    lock_root = Path(m["stage"]).parent/"triplet_gpu_locks"
    require(not lock_root.is_symlink(), "unsafe GPU lock directory")
    lock_root.mkdir(exist_ok=True,mode=0o700)
    require(lock_root.stat().st_uid == os.getuid(), "GPU lock directory not owned")
    locks,processes,tasks = [],[],[]
    start = time.monotonic(); deadline = start+m["limits"]["runtime_seconds"]
    def interrupted(*_):
        raise RuntimeError("supervisor interrupted; all raw files retained")
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def check_deadline():
        require(time.monotonic() < deadline,"post-allocation deadline exhausted")
    def monitor(active,selected):
        while any(p.poll() is None for p,_ in active):
            check_deadline()
            require(all(p.poll() in (None,0) for p,_ in active),"worker failed")
            state = safety(m,active,selected)
            append(runtime/"resource_journal.jsonl",{"at":time.time(),**state})
            time.sleep(5)
        require(all(p.returncode == 0 for p,_ in active),"worker exited nonzero")
        for attempt in range(12):
            snapshot = gpu_snapshot()
            # A foreign process is always fatal; only an empty device may settle.
            check_devices(m,snapshot,selected=selected,settling=set(selected))
            try:
                state = safety(m,[],selected,False)
                break
            except ValueError:
                if attempt == 11: raise
                time.sleep(5)
        append(runtime/"resource_journal.jsonl",{"at":time.time(),"release_verified":selected,**state})
        return state
    def spawn_task(task,cpu_set,flag):
        task_path = runtime/"tasks"/(task["name"]+".json")
        if not task_path.exists(): exclusive_json(task_path,task)
        log_path = runtime/(task["name"]+(".model.log" if flag == "--worker" else ".audit.log"))
        with log_path.open("xb") as log:
            command = ["taskset","-c",cpu_set,"nice","-n","10","ionice","-c","2","-n","7",
                       sys.executable,str(Path(__file__).resolve()),flag,str(task_path)]
            proc = subprocess.Popen(command,env=worker_env(task["gpu_id"] if flag == "--worker" else None),
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc)
        append(runtime/"processes.jsonl",{"pid":proc.pid,"task":task["name"],"mode":flag,"at":time.time(),"command":command})
        return proc
    def monitor_auditors(auditors,keep_gpu_allocation):
        while any(p.poll() is None for p in auditors):
            check_deadline()
            require(all(p.poll() in (None,0) for p in auditors),"raw auditor failed")
            require(engine.mem_available_kib() >= m["limits"]["min_available_ram_gib"]*1024**2,"audit RAM limit")
            rss,_ = engine.process_tree_rss_kib({p.pid for p in auditors if p.poll() is None})
            require(rss <= m["limits"]["max_worker_rss_gib"]*1024**2,"audit RSS limit")
            if keep_gpu_allocation: safety(m,[])
            time.sleep(5)
        require(all(p.returncode == 0 for p in auditors),"auditor exited nonzero")
    try:
        for gpu in m["gpu_ids"]:
            path = lock_root/f"gpu_{gpu}.lock"
            require(not path.is_symlink(),"unsafe GPU lock")
            descriptor = os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            lock = os.fdopen(descriptor,"a+")
            require(os.fstat(descriptor).st_uid == os.getuid(),"GPU lock not owned")
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        for iteration in range(13):
            check_deadline()
            state = safety(m,[])
            append(runtime/"qualification_inventory.jsonl",{"at":time.time(),**state})
            if iteration < 12: time.sleep(5)
        cpu_sets = engine.allocate_cpu_sets(len(m["gpu_ids"]),2)
        def task(name,gpu,indices,qualification):
            return {"name":name,"manifest":str(manifest_path),"manifest_sha256":digest,"gpu_id":gpu,
                    "record_indices":indices,"qualification":qualification,
                    "destination":str(output/("qualification" if qualification else "shards")/name)}
        q = task("qualification",m["gpu_ids"][0],m["qualification_indices"],True)
        qp = spawn_task(q,cpu_sets[0],"--worker")
        monitor([(qp,q["gpu_id"])],m["gpu_ids"])
        qa = spawn_task(q,cpu_sets[0],"--audit")
        monitor_auditors([qa],True)
        print("QUALIFICATION_PASSED: native BF16, repeatability, exact save/readback; no differences computed",flush=True)
        active = []
        for worker_index,gpu in enumerate(m["gpu_ids"]):
            t = task(f"gpu_{gpu}",gpu,list(range(worker_index,len(rows),len(m["gpu_ids"]))),False)
            tasks.append(t);safety(m,active)
            proc = spawn_task(t,cpu_sets[worker_index],"--worker");active.append((proc,gpu))
            for _ in range(2):
                time.sleep(5);safety(m,active)
        release_state = monitor(active,m["gpu_ids"])
        exclusive_json(output/"gpu_release.json",{"verified":True,"gpu_ids":m["gpu_ids"],"state":release_state})
        for lock in locks: lock.close()
        locks.clear()
        print("ALL_MODEL_WORKERS_EXITED; GPUs released; independent raw-file audits starting",flush=True)
        auditors = [spawn_task(t,cpu_sets[i],"--audit") for i,t in enumerate(tasks)]
        monitor_auditors(auditors,False)
        by_index = {}
        for t in tasks:
            destination = Path(t["destination"])
            for entry in base.read_jsonl(destination/"raw_audit.jsonl"):
                require(entry["record_index"] not in by_index,"duplicate final record")
                for kind in ("h0","h60"):
                    entry["models"][kind]["tensor_path"] = str((destination/kind/f"record_{entry['record_index']:06d}.safetensors").relative_to(output))
                by_index[entry["record_index"]] = entry
        require(set(by_index) == set(range(561)),"incomplete final record coverage")
        # Repeat comparisons stay within the same model; no cross-model arithmetic.
        import torch
        from safetensors import safe_open
        torch.set_num_threads(1)
        for i in m["qualification_indices"]:
            for kind in ("h0","h60"):
                with safe_open(str(output/by_index[i]["models"][kind]["tensor_path"]),framework="pt",device="cpu") as production, \
                     safe_open(str(Path(q["destination"])/kind/record_name(rows[i])),framework="pt",device="cpu") as qualification:
                    for layer in range(LAYER_COUNT):
                        require(torch.equal(production.get_slice(kind)[layer],qualification.get_slice(kind)[layer]),"same-model qualification/production mismatch")
        base.atomic_write_jsonl(output/"activation_index.jsonl",[by_index[i] for i in range(561)])
        shutil.copytree(runtime,output/"provenance/execution")
        exclusive_json(output/"extraction_summary.json",{
            "status":"succeeded","records":561,"problems":187,"completion_tokens":sum(r["completion_token_count"] for r in rows),
            "layers":36,"hidden_size":2560,"dtype":"bfloat16","models":["h0","h60"],"token_averaging":False,
            "raw_activations_retained":True,"differences_computed":False,"all_native_readbacks_bitwise_equal":True,
            "same_model_qualification_and_production_bitwise_equal":True,"qualification_worker_exit":qp.returncode,
            "qualification_auditor_exit":qa.returncode,"model_workers_exit_codes":[p.returncode for p,_ in active],
            "raw_auditor_exit_codes":[p.returncode for p in auditors],"gpu_release_verified":True,"gpu_ids":m["gpu_ids"],
            "manifest_sha256":digest,"split_records":dict(Counter(r["problem_split"] for r in rows)),
            "raw_tensor_file_bytes":sum(v["size_bytes"] for e in by_index.values() for v in e["models"].values()),
            "elapsed_seconds":time.monotonic()-start})
        base.atomic_write_json(output/"artifact_manifest.json",base.file_manifest(output))
        print("RAW_EXTRACTION_PRODUCER_SUCCEEDED",flush=True)
    except BaseException as error:
        engine.terminate_workers(processes)
        for proc in processes:
            if proc.poll() is None: proc.wait(timeout=15)
        try:
            failure_release = safety(m,[],allow_settling=False)
        except Exception as release_error:
            failure_release = {"not_verified":str(release_error)}
        exclusive_json(output/"FAILURE.json",{"error_type":type(error).__name__,"error":str(error),
                       "run_token":m["run_token"],"all_raw_files_preserved":True,"release_check":failure_release})
        raise
    finally:
        for lock in locks: lock.close()


def verify_package(root):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    manifest = json.loads((root/"artifact_manifest.json").read_text())
    require(base.file_manifest(root) == manifest,"artifact manifest mismatch")
    summary = json.loads((root/"extraction_summary.json").read_text())
    require(summary["status"] == "succeeded" and summary["raw_activations_retained"] and
            summary["differences_computed"] is False and summary["all_native_readbacks_bitwise_equal"] and
            summary["same_model_qualification_and_production_bitwise_equal"],"producer not positively successful")
    require(summary["model_workers_exit_codes"] == [0]*len(summary["gpu_ids"]) and
            summary["raw_auditor_exit_codes"] == [0]*len(summary["gpu_ids"]) and
            summary["qualification_worker_exit"] == summary["qualification_auditor_exit"] == 0,"nonzero worker/auditor exit")
    rows = list(base.read_jsonl(root/"input/prepared_records.jsonl"));validate_rows(rows)
    index = list(base.read_jsonl(root/"activation_index.jsonl"))
    require(len(index) == 561,"index count mismatch")
    paths = set()
    for row,entry in zip(rows,index):
        require(entry["record_index"] == row["record_index"] and entry["record_id"] == row["record_id"] and entry["verified"],"index identity mismatch")
        require(set(entry["models"]) == {"h0","h60"},"raw model set mismatch")
        for kind,info in entry["models"].items():
            relative = info["tensor_path"]
            require(relative not in paths and not Path(relative).is_absolute() and ".." not in Path(relative).parts,"unsafe/duplicate tensor path")
            paths.add(relative)
            require(manifest["files"][relative]["sha256"] == info["sha256"],"audited raw tensor changed")
            with safe_open(str(root/relative),framework="pt",device="cpu") as f:
                require(f.metadata() == expected_metadata(row,kind,summary["manifest_sha256"]),"raw identity metadata mismatch")
                require(f.get_slice(kind).get_shape() == [36,row["completion_token_count"],2560] and f.get_slice(kind).get_dtype() == "BF16","raw shape/dtype mismatch")
                aux = auxiliary_tensors(row)
                require(set(f.keys()) == {kind,*aux},"unexpected raw keys")
                for key,value in aux.items(): require(torch.equal(f.get_tensor(key),value),"raw mask/ID mismatch")
    require(len(paths) == 1122,"need 1122 raw model/record files")
    return {"status":"independently_verified","records":561,"problems":187,"model_record_files":1122,
            "models":["h0","h60"],"dtype":"bfloat16","differences_computed":False,"problem_split_overlap":0,
            "completion_tokens":sum(r["completion_token_count"] for r in rows),"manifest_files":len(manifest["files"]),
            "artifact_manifest_sha256":base.sha256_file(root/"artifact_manifest.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--worker",type=Path);group.add_argument("--audit",type=Path)
    group.add_argument("--supervise",type=Path);group.add_argument("--verify-package",type=Path)
    p.add_argument("--manifest-sha256")
    args = p.parse_args()
    if args.worker: worker(args.worker)
    elif args.audit: audit(args.audit)
    elif args.supervise: supervise(args.supervise,args.manifest_sha256)
    else: print(json.dumps(verify_package(args.verify_package),sort_keys=True),flush=True)


if __name__ == "__main__": main()
