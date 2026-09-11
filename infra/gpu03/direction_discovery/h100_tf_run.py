"""Bounded H100 TF or explicit same-condition batched generation tranche.

No launch/approval/ledger API. The caller supplies the user-systemd envelope and
an immutable plan. TF keeps1..4 batch-one processes/GPU; the separate generation
protocol requires exactly one model worker/GPU with retained real-prefix checks.
Failure retains all journals. Separate --verify rehashes the terminal package.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time

from . import h100_supervisor as shared

HERE = "infra/gpu03/direction_discovery/h100_tf_run.py"
ENGINE = "infra/gpu03/direction_discovery/engine.py"
PROTOCOL = "h100_tf_tranche_v1"
BATCH_PROTOCOL = "h100_same_condition_generation_v1"
BATCH_MODULE = "infra/gpu03/direction_discovery/batched_generation.py"
DIAGNOSTICS_MODULE = "infra/gpu03/direction_discovery/batch_diagnostics.py"
ROOT = Path("/home/ubuntu/h100-workspace")
HOST, UID = shared.HOST, shared.UID
MAX_PLAN = 128 << 20
MAX_LINE = 4 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_path(value):
    p = Path(value)
    require(p.is_absolute() and ".." not in p.parts and p.is_relative_to(ROOT), "path outside H100 workspace")
    for q in (p, *p.parents):
        require(not q.is_symlink(), "symlink in workspace path")
        if q.exists():
            require(q.stat().st_uid == UID, "workspace path has another owner")
        if q == ROOT:
            break
    return p


def file_ref(path):
    p = Path(path); before = p.stat(); h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    after = p.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "file changed while hashing")
    return {"path": str(p), "sha256": h.hexdigest(), "size_bytes": after.st_size}


def verify_file(reference, *, immutable=True, runtime=False):
    p = Path(reference["path"]) if runtime else safe_path(reference["path"])
    st = p.stat()
    require(stat.S_ISREG(st.st_mode), "binding is not a regular file")
    if immutable:
        require(st.st_uid == UID and not st.st_mode & 0o222, "binding is not owned and immutable")
    actual = file_ref(p)
    require(all(actual[k] == reference[k] for k in ("sha256", "size_bytes")), "binding hash/size differs: " + str(p))
    return p


def read_json_ref(reference):
    p = verify_file(reference)
    require(p.stat().st_size <= MAX_PLAN, "JSON metadata exceeds bound")
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == reference["sha256"], "JSON changed before parsing")
    return json.loads(data)


def write_json(path, value):
    with Path(path).open("x") as f:
        f.write(canonical(value) + "\n"); f.flush(); os.fsync(f.fileno())


def prepared_rows(plan, tasks):
    path = tasks[0]["prepared_records"]; bound = plan["bound_files"][path]
    require(bound["size_bytes"] <= MAX_PLAN, "prepared metadata exceeds bound")
    data = Path(path).read_bytes()
    require(len(data) == bound["size_bytes"] and hashlib.sha256(data).hexdigest() == bound["sha256"], "prepared bytes changed")
    rows = {}
    for line in data.splitlines():
        row = json.loads(line)
        require(row["record_id"] not in rows, "duplicate prepared record")
        rows[row["record_id"]] = row
    require(all(r["record_id"] in rows for t in tasks for r in t["requests"]), "request has no prepared record")
    return rows


def append(path, value):
    # Same fsync append-only journal pattern as all_layer_run.append.
    with Path(path).open("a") as f:
        f.write(canonical(value) + "\n"); f.flush(); os.fsync(f.fileno())


def load_plan(path, sha256):
    p = safe_path(path)
    require(p.stat().st_size <= MAX_PLAN, "plan exceeds bound")
    plan = read_json_ref({"path": str(p), "sha256": sha256, "size_bytes": p.stat().st_size})
    batched = plan.get("mode") == "generate_batch_v1"
    require(plan["schema_version"] == 1 and ((plan["protocol"] == PROTOCOL and plan["mode"] == "tf") or
            (batched and plan["protocol"] == BATCH_PROTOCOL)), "unreviewed tranche mode/protocol")
    if batched:
        from . import batched_generation as batch_engine
        from . import batch_diagnostics
    require(re.fullmatch(r"codex-[a-z0-9-]{8,120}", plan["run_token"]), "invalid run token")
    require(plan["host"] == HOST and plan["uid"] == UID, "wrong host/owner")
    out, source = safe_path(plan["output"]), safe_path(plan["source_root"])
    require(plan["run_token"] in out.name and not out.is_relative_to(source) and not source.is_relative_to(out), "unsafe output/source separation")
    require(math.isfinite(plan["absolute_deadline_epoch"]) and plan["absolute_deadline_epoch"] <= shared.DEADLINE,
            "deadline exceeds workstation durable-copy deadline")
    limits = plan["limits"]
    for key in ("minimum_available_ram_bytes", "minimum_free_disk_bytes", "per_worker_rss_bytes",
                "per_gpu_used_memory_bytes", "maximum_output_bytes", "per_worker_log_bytes", "maximum_monitor_bytes"):
        require(type(limits[key]) is int and limits[key] > 0, "invalid resource limit: " + key)
    require(1 <= limits["monitor_seconds"] <= 5 and 1 <= limits["stagger_seconds"] <= 30 and
            1 <= limits["release_seconds"] <= 60, "invalid monitoring/stagger/release interval")
    require(limits["minimum_idle_free_memory_mib"] > 0, "idle free-memory floor missing")
    verify_file(plan["python"], immutable=False, runtime=True)
    required = {HERE, ENGINE, "infra/gpu03/direction_discovery/h100_supervisor.py",
                "infra/gpu03/direction_discovery/intervention.py",
                "infra/gpu03/activation_dataset/extract_triplet_raw.py",
                "infra/gpu03/activation_dataset/extract_delta_activations.py"}
    if batched: required.update({BATCH_MODULE, DIAGNOSTICS_MODULE})
    require(required <= set(plan["source_inventory"]), "missing worker/supervisor source bindings")
    actual_source = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    require(actual_source == set(plan["source_inventory"]), "source inventory has extras/missing files")
    for name, bound in plan["source_inventory"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe source relative path")
        verify_file({"path": str(source / name), **bound})
    # One parent verification per shared input, never model bytes once per worker.
    for name, bound in plan["bound_files"].items():
        verify_file({"path": name, **bound})
    devices = plan["gpus"]
    require(1 <= len(devices) <= 8 and len({x["id"] for x in devices}) == len(devices) and
            len({x["uuid"] for x in devices}) == len(devices), "duplicate/empty GPU allocation")
    require(all(type(x["id"]) is int and 0 <= x["id"] < 8 and x["uuid"].startswith("GPU-") for x in devices), "invalid GPUs")
    workers = plan["workers"]; counts = Counter(w["gpu_id"] for w in workers)
    require(set(counts) == {x["id"] for x in devices} and all(1 <= n <= 4 for n in counts.values()), "require 1..4 workers per GPU")
    if batched: require(all(n == 1 for n in counts.values()), "Batched generation requires exactly one worker per GPU")
    cpus, names, ids, tasks = set(), set(), [], []
    common, condition_defs = None, {}
    for w in workers:
        require(re.fullmatch(r"worker_[0-9]{2}", w["name"]) and w["name"] not in names, "duplicate/invalid worker name")
        names.add(w["name"])
        require(w["cpu_set"] and len(set(w["cpu_set"])) == len(w["cpu_set"]) and
                all(type(x) is int and 0 <= x < 192 for x in w["cpu_set"]) and not cpus.intersection(w["cpu_set"]), "CPU sets overlap/invalid")
        cpus.update(w["cpu_set"])
        task = read_json_ref(w["task"])
        require(task["mode"] == plan["mode"] and task["run_token"] == plan["run_token"] and
                task["worker_name"] == w["name"] and task["gpu_id"] == w["gpu_id"] and
                task["output"] == str(out / w["name"]), "task execution identity differs")
        require(task["attention_policy"] == "exclusive_math" and 0 < task["deadline_seconds"] <= 43200 and
                task.get("batch_size", 1) == 1, "Numerical policy/deadline differs")
        if batched:
            batch_engine.profile(task["batch_profile"])
            require(task["batch_stage"] == "integrated_real_requests", "Controller requires integrated real-request checks")
            batch_engine.validate_qualification(task,task["batch_profile"])
            require(task["sampling"] == {"temperature":.7,"top_p":.95,"top_k":0,"repetition_penalty":1.,
                    "eos_token_ids":[151643,151645]}, "Unreviewed generation sampler")
            require("teacher_forced_padded_sequence_length" not in task, "TF padding is not a generation setting")
        for k in ("prepared_records",):
            require(task[k] in plan["bound_files"], "unbound prepared input")
        for k in ("model_snapshot", "checkpoint"):
            folder = safe_path(task[k]); model_files = [q for q in folder.rglob("*") if q.is_file()]
            require(model_files and all(str(q) in plan["bound_files"] for q in model_files), "unbound model/tokenizer file")
        excluded = {"worker_name", "gpu_id", "output", "requests", "conditions"}
        if batched: excluded.add("diagnostic_projection_condition_id")
        invariant = {k: v for k, v in task.items() if k not in excluded}
        require(common is None or invariant == common, "worker scientific task settings differ")
        common = invariant
        for condition_id, condition in task["conditions"].items():
            require(condition_id not in condition_defs or condition_defs[condition_id] == condition, "condition definition changed between workers")
            condition_defs[condition_id] = condition
            for layer in condition.get("layers", []):
                if layer["kind"] == "candidate":
                    require(plan["bound_files"].get(layer["path"], {}).get("sha256") == layer["sha256"], "unbound candidate")
                else:
                    require(layer["kind"] == "random", "unknown projection kind")
        require(task["requests"], "empty worker")
        for r in task["requests"]:
            require(isinstance(r["request_id"], str) and r["request_id"] and
                    r["condition_id"] in task["conditions"] and isinstance(r["record_id"], str), "invalid TF request")
            if batched:
                require(r.get("scope") in ("primary","local") and type(r.get("seed")) is int and 0 <= r["seed"] < 2**63,
                        "Invalid generation scope/seed")
            ids.append(r["request_id"])
        tasks.append(task)
    require(len(ids) == len(set(ids)) == plan["expected_request_count"] and
            digest(sorted(ids)) == plan["request_ids_sha256"], "duplicate/missing/changed request union")
    prepared_rows(plan, tasks)
    return plan, tasks


def check_runtime(plan):
    require(socket.gethostname() == HOST and os.getuid() == UID, "wrong actual H100 host/UID")
    require(sys.executable == plan["python"]["path"] and sys.version_info[:3] == (3, 12, 3), "actual Python differs")
    require({k: importlib.metadata.version(k) for k in shared.VERSIONS} == shared.VERSIONS, "runtime versions differ")
    require(Path(__file__).resolve() == Path(plan["source_root"]) / HERE, "executing another source snapshot")
    require(set(x for w in plan["workers"] for x in w["cpu_set"]) <= os.sched_getaffinity(0), "CPU allocation unavailable")


def process_groups(children):
    """Include surviving descendants after a leader exits; never authorize by UID alone."""
    groups = {p.pid: [] for p, _, _ in children}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        info = shared.process_info(int(entry.name))
        if info and info["pgid"] in groups:
            groups[info["pgid"]].append(info)
    for p, identity, _ in children:
        current = shared.process_info(p.pid)
        require(current is None or shared.same_identity(current, identity), "worker PID identity changed")
        require(all(x["uid"] == UID and x["start_ticks"] >= identity["start_ticks"] for x in groups[p.pid]), "foreign group member")
    return groups


def gpu_check(plan, rows, allowed, *, idle=False):
    require(len(rows) == 8 and {r["index"] for r in rows} == set(range(8)), "incomplete GPU inventory")
    selected = {x["id"]: x for x in plan["gpus"]}
    for r in rows:
        if r["index"] not in selected:
            continue
        require(r["uuid"] == selected[r["index"]]["uuid"] and r["name"] == "NVIDIA H100 80GB HBM3", "GPU identity changed")
        require(all(p["pid"] in allowed.get(r["index"], set()) and p["owner"] == "ubuntu" for p in r["processes"]), "foreign GPU process")
        require(r["memory_used_mib"] * (1 << 20) <= plan["limits"]["per_gpu_used_memory_bytes"], "GPU memory guard exceeded")
        if idle:
            require(not r["processes"] and r["memory_used_mib"] <= 64 and r["utilization_percent"] <= 1 and
                    r["memory_free_mib"] >= plan["limits"]["minimum_idle_free_memory_mib"], "GPU not positively idle")


def resource_state(plan, children):
    require(time.time() < plan["absolute_deadline_epoch"], "TF tranche deadline exceeded")
    groups = process_groups(children); allowed = {}
    for p, _, worker in children:
        # Union, not assignment: up to four independent groups share one GPU.
        allowed.setdefault(worker["gpu_id"], set()).update(x["pid"] for x in groups[p.pid])
        require(sum(x["rss_bytes"] for x in groups[p.pid]) <= plan["limits"]["per_worker_rss_bytes"], "worker RSS exceeded")
    available = next(int(s.split()[1]) * 1024 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))
    fs = os.statvfs(plan["output"]); free = fs.f_bavail * fs.f_frsize
    require(available >= plan["limits"]["minimum_available_ram_bytes"] and
            free >= plan["limits"]["minimum_free_disk_bytes"], "host RAM/disk floor failed")
    files = [p for p in Path(plan["output"]).rglob("*") if p.is_file()]
    require(all(not p.is_symlink() for p in files), "output symlink")
    require(sum(p.stat().st_size for p in files) <= plan["limits"]["maximum_output_bytes"], "output byte cap exceeded")
    for p in files:
        if p.suffix == ".log":
            require(p.stat().st_size <= plan["limits"]["per_worker_log_bytes"], "worker log cap exceeded")
    journal = Path(plan["output"]) / "resources.jsonl"
    require(not journal.exists() or journal.stat().st_size <= plan["limits"]["maximum_monitor_bytes"], "monitor journal cap exceeded")
    rows = shared.gpu_snapshot(); gpu_check(plan, rows, allowed)
    return {"at": time.time(), "gpu_inventory": rows, "available_ram_bytes": available, "free_disk_bytes": free,
            "worker_rss_bytes": {w["name"]: sum(x["rss_bytes"] for x in groups[p.pid]) for p, _, w in children},
            "workers": [{"name": w["name"], "pid": p.pid, "returncode": p.poll(),
                         "results_bytes": (Path(plan["output"]) / w["name"] / "results.jsonl").stat().st_size
                         if (Path(plan["output"]) / w["name"] / "results.jsonl").exists() else 0} for p, _, w in children]}


def cleanup(children):
    # all_layer_run's TERM/wait/KILL pattern, with identity and surviving-group checks.
    for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        groups = process_groups(children)
        for p, _, _ in children:
            if groups[p.pid]:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + seconds
        while True:
            for p, _, _ in children:
                p.poll()  # waitpid reaps every direct child, including already-exited workers.
            if all(p.returncode is not None for p, _, _ in children) and not any(process_groups(children).values()):
                return
            if time.monotonic() >= end:
                break
            time.sleep(.1)
    require(False, "owned process groups did not release")


def validate_outputs(plan, tasks):
    if plan["mode"] == "generate_batch_v1": return validate_generation_outputs(plan,tasks)
    rows = prepared_rows(plan, tasks)
    seen, timings, peaks, worker_seconds = set(), [], [], {}
    for task in tasks:
        out = Path(task["output"])
        require({p.name for p in out.iterdir()} == {"task.json", "SUCCESS.json", "results.jsonl"}, "unexpected/missing worker output")
        require(json.loads((out / "task.json").read_bytes()) == task, "saved task differs")
        success = json.loads((out / "SUCCESS.json").read_bytes())
        require(all(success.get(k) == v for k, v in {"status": "succeeded", "mode": "tf", "run_token": task["run_token"],
                "worker_name": task["worker_name"], "requests": len(task["requests"])}.items()), "invalid SUCCESS receipt")
        report = success["model_load_reports"]
        require(report.get("with_adapter") is True and report.get("active_adapters") and
                report.get("nonzero_lora_parameter_tensors", 0) > 0, "no positive M60 load report")
        seconds = success["elapsed_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0, "invalid worker timing")
        worker_seconds[task["worker_name"]] = seconds
        expected = {r["request_id"]: r for r in task["requests"]}; got = set()
        with (out / "results.jsonl").open("rb") as f:
            while line := f.readline(MAX_LINE + 1):
                require(len(line) <= MAX_LINE and line.endswith(b"\n"), "oversize/truncated result line")
                r = json.loads(line); rid = r["request_id"]
                require(rid in expected and rid not in seen, "duplicate/unexpected TF result")
                require(all(r.get(k) == v for k, v in expected[rid].items()), "TF request provenance differs")
                original = rows[expected[rid]["record_id"]]
                require((r["problem_id"], r["problem_split"], r["original_class"]) ==
                        (original["problem_id"], original["problem_split"], original["outcome_presence_class"]), "source row differs")
                result = r["result"]; losses = result["token_nll"]
                canonical(result)  # Reject NaN/Infinity also in energy or added timing metadata.
                require(len(losses) == original["completion_token_count"] and
                        all(type(x) in (int, float) and math.isfinite(x) and 0 <= x < 1e6 for x in losses), "invalid TF likelihoods")
                groups = {"all_completion": list(range(len(losses))), **original["region_mask_completion_positions"]}
                nll = {k: {"n_tokens": len(v), "mean_nll": sum(losses[i] for i in v) / len(v)} for k, v in groups.items() if v}
                require(result["nll"] == nll and "completion" not in result and "completion_token_ids" not in result, "TF aggregate/mode differs")
                strength = task["conditions"][r["condition_id"]].get("intervention_strength", task.get("intervention_strength", 1.0))
                require(result["intervention_strength"] == strength, "strength differs")
                elapsed = result["elapsed_seconds"]
                require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed > 0, "invalid TF timing")
                for k in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                    require(type(result[k]) is int and result[k] > 0, "missing TF allocator peak")
                timings.append(elapsed); peaks.append(result["cuda_peak_reserved_bytes"]); seen.add(rid); got.add(rid)
        require(got == set(expected), "missing worker request coverage")
    require(len(seen) == plan["expected_request_count"] and digest(sorted(seen)) == plan["request_ids_sha256"], "wrong full TF coverage")
    return {"mode": "tf", "requests": len(seen), "request_ids_sha256": digest(sorted(seen)),
            "summed_request_seconds": sum(timings), "mean_request_seconds": sum(timings) / len(timings),
            "maximum_request_cuda_reserved_bytes": max(peaks), "worker_elapsed_seconds": worker_seconds}


def validate_generation_outputs(plan,tasks):
    from . import batched_generation as b
    from . import batch_diagnostics as diagnostics
    rows=prepared_rows(plan,tasks);seen=set();worker_seconds={};batch_seconds=[];tokens=0;diagnostic_refs=[]
    require(diagnostics.source_hashes()=={name:plan["source_inventory"]["infra/gpu03/direction_discovery/"+name]["sha256"]
            for name in diagnostics.SOURCES},"Running diagnostic sources differ from bound plan")
    for task in tasks:
        out=Path(task["output"])
        require({p.name for p in out.iterdir()} == {"task.json","SUCCESS.json","results.jsonl","batches.jsonl","real_prefix_diagnostics.json"},
                "Unexpected/missing generation worker output")
        require(json.loads((out/"task.json").read_bytes()) == task,"Saved generation task differs")
        success=json.loads((out/"SUCCESS.json").read_bytes());load=success["model_load_reports"]
        require(all(success.get(k)==v for k,v in {"status":"succeeded","mode":"generate_batch_v1","run_token":task["run_token"],
                "worker_name":task["worker_name"],"requests":len(task["requests"]),"batch_profile":task["batch_profile"]}.items()),"Invalid generation SUCCESS")
        require(load.get("with_adapter") is True and load.get("active_adapters") and load.get("nonzero_lora_parameter_tensors",0)>0,
                "No positive generation M60 load")
        elapsed=success["elapsed_seconds"];require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,"Invalid generation timing")
        worker_seconds[task["worker_name"]]=elapsed
        batches=b.request_batches(task["requests"],task["batch_profile"]["batch_size"])
        expected={r["request_id"]:r for r in task["requests"]};batch_index={r["request_id"]:i for i,batch in enumerate(batches) for r in batch}
        actual={}
        with (out/"results.jsonl").open("rb") as f:
            while line:=f.readline(MAX_LINE+1):
                require(len(line)<=MAX_LINE and line.endswith(b"\n"),"Oversize/truncated generation result")
                r=json.loads(line);rid=r["request_id"];canonical(r)
                require(rid in expected and rid not in seen,"Duplicate/unexpected generation result")
                request=expected[rid];row=rows[request["record_id"]]
                require(all(r.get(k)==v for k,v in request.items()) and r["batch_index"]==batch_index[rid],"Generation request/batch differs")
                require((r["problem_id"],r["problem_split"],r["original_class"])==(row["problem_id"],row["problem_split"],row["outcome_presence_class"]),"Generation carrier differs")
                value=r["result"];prefix,fixed,budget=b.engine.prefix_for(row,request["scope"]);generated=value["generated_token_ids"]
                require(generated and len(generated)<=budget and all(type(x) is int and x>=0 for x in generated),"Invalid generated token IDs/budget")
                eos=task["sampling"]["eos_token_ids"]
                require(not any(x in eos for x in generated[:-1]) and
                    ((value["stop_reason"]=="eos" and generated[-1] in eos) or
                     (value["stop_reason"]=="length" and len(generated)==budget and generated[-1] not in eos)),"Generation EOS/length differs")
                require(value["completion_token_ids"]==fixed+generated and value["fixed_completion_prefix_token_count"]==len(fixed) and
                    isinstance(value["completion"],str) and value["batch_profile"]==task["batch_profile"] and
                    value["batch_size_actual"]==len(batches[batch_index[rid]]) and value["batch_wall_seconds"] is None,"Generation fixed prefix/profile differs")
                condition=task["conditions"][r["condition_id"]];strength=b.engine.intervention_strength(task,condition)
                require(value["intervention_strength"]==strength and 0<value["elapsed_seconds"]<=elapsed,"Generation strength/timing differs")
                layers={str(x["layer"]):x for x in condition.get("layers",[])}
                require(set(value["energy"])==set(layers),"Generation energy layers differ")
                for layer,entry in value["energy"].items():
                    item=layers[layer];rank=item["rank"] if item["kind"]=="random" else len(item["selectors"])
                    n=len(generated)
                    require(entry["rank"]==rank and entry["strength"]==strength and entry["projection_dtype"]=="float32" and
                        entry["selected_tokens"]==entry["forward_calls"]==n,"Generation projection count differs")
                    require(set(entry["scopes"])==({"prefill","decode"} if n>1 else {"prefill"}),"Generation projection scopes differ")
                    for scope,count in (("prefill",1),("decode",n-1)):
                        if count:
                            require(entry["scopes"][scope]["selected_tokens"]==entry["scopes"][scope]["forward_calls"]==count and
                                entry["scopes"][scope]["strength"]==strength,"Per-row predictor scope differs")
                    for field in b.FIELDS:
                        require(type(entry[field]) in (int,float) and entry[field]>=0 and
                            math.isclose(entry[field],sum(v[field] for v in entry["scopes"].values()),rel_tol=1e-9,abs_tol=1e-8),"Generation energy sum differs")
                actual[rid]=r;seen.add(rid);tokens+=len(generated)
        require(set(actual)==set(expected),"Missing generated request")
        receipts=[json.loads(line) for line in (out/"batches.jsonl").read_bytes().splitlines()]
        require(len(receipts)==len(batches),"Missing batch timings")
        for i,(batch,receipt) in enumerate(zip(batches,receipts)):
            require(receipt["batch_index"]==i and receipt["request_ids"]==[r["request_id"] for r in batch] and
                receipt["generated_tokens"]==sum(len(actual[r["request_id"]]["result"]["generated_token_ids"]) for r in batch) and
                type(receipt["batch_wall_seconds"]) in (float,int) and math.isfinite(receipt["batch_wall_seconds"]) and
                0<receipt["batch_wall_seconds"]<=elapsed,"Invalid batch timing/coverage")
            batch_seconds.append(receipt["batch_wall_seconds"])
        first=batches[0]
        binding=diagnostics.binding(task,first,[rows[r["record_id"]] for r in first],
            [actual[r["request_id"]]["result"] for r in first],task["diagnostic_projection_condition_id"])
        diagnostics.validate_receipt(json.loads((out/"real_prefix_diagnostics.json").read_bytes()),binding)
        diagnostic_refs.append(file_ref(out/"real_prefix_diagnostics.json"))
    require(len(seen)==plan["expected_request_count"] and digest(sorted(seen))==plan["request_ids_sha256"],"Wrong complete generation union")
    return {"mode":"generate_batch_v1","requests":len(seen),"request_ids_sha256":digest(sorted(seen)),"generated_tokens":tokens,
            "summed_batch_seconds":sum(batch_seconds),"worker_elapsed_seconds":worker_seconds,"real_prefix_diagnostics":diagnostic_refs,
            "distribution_equivalence_established":False}


def progress_counts(plan, cursors):
    counts = {}
    for w in plan["workers"]:
        p = Path(plan["output"]) / w["name"] / "results.jsonl"
        offset, count = cursors.get(w["name"], (0, 0))
        if p.exists():
            require(p.stat().st_size >= offset, "worker journal was truncated")
            with p.open("rb") as f:
                f.seek(offset)
                # At most 8 MiB per worker per sample; approximate progress can lag.
                block = f.read(8 << 20); count += block.count(b"\n"); offset += len(block)
        cursors[w["name"]] = (offset, count); counts[w["name"]] = count
    return counts


def supervise(path, sha256):
    plan, tasks = load_plan(path, sha256); check_runtime(plan)
    require(time.time() < plan["absolute_deadline_epoch"], "deadline already passed")
    out = safe_path(plan["output"]); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    children, logs, error, release, result = [], [], None, None, None
    released, all_exited_zero, cursors = False, False, {}
    peak_gpu_used = {str(g["id"]): 0 for g in plan["gpus"]}
    started = time.time(); old_handlers = {}
    def interrupted(signum, _frame):
        raise RuntimeError("controller received signal " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, interrupted)
    def observe():
        snapshot = resource_state(plan, children)
        snapshot["completed_line_counts"] = progress_counts(plan, cursors)
        for g in snapshot.get("gpu_inventory", []):
            if str(g["index"]) in peak_gpu_used:
                peak_gpu_used[str(g["index"])] = max(peak_gpu_used[str(g["index"])], g["memory_used_mib"])
        append(out / "resources.jsonl", snapshot)
    try:
        gpu_check(plan, shared.gpu_snapshot(), {}, idle=True)
        # Round-robin GPUs, then a second/third/fourth independent process on each.
        ordered = sorted(plan["workers"], key=lambda w: (sum(v["gpu_id"] == w["gpu_id"] for v in plan["workers"][:plan["workers"].index(w)]), w["gpu_id"]))
        for w in ordered:
            observe()
            command = ["/usr/bin/taskset", "-c", ",".join(map(str, w["cpu_set"])), plan["python"]["path"],
                       "-B", "-m", "infra.gpu03.direction_discovery.batched_generation" if plan["mode"] == "generate_batch_v1"
                       else "infra.gpu03.direction_discovery.engine", "--task", w["task"]["path"]]
            log = (out / (w["name"] + ".log")).open("xb"); logs.append(log)
            p = subprocess.Popen(command, cwd=plan["source_root"], env=shared.worker_environment(w["gpu_id"]),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = shared.process_info(p.pid)
            # Retain the Popen handle even if identity collection fails. Never
            # signal an ambiguous identity; the enclosing service is the backstop.
            children.append((p, identity or {"pid": p.pid, "pgid": p.pid, "uid": UID, "start_ticks": -1}, w))
            require(identity is not None and identity["uid"] == UID and identity["pgid"] == p.pid, "child identity unavailable")
            append(out / "workers.jsonl", {"name": w["name"], "identity": identity, "gpu_id": w["gpu_id"], "command": command, "at": time.time()})
            time.sleep(plan["limits"]["stagger_seconds"])
            require(all(c.poll() in (None, 0) for c, _, _ in children), "worker failed during stagger")
        while True:
            observe()
            require(all(p.poll() in (None, 0) for p, _, _ in children), "TF worker exited nonzero")
            if all(p.poll() == 0 for p, _, _ in children):
                break
            time.sleep(plan["limits"]["monitor_seconds"])
        all_exited_zero = True
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            cleanup(children)
            released = True
            end = time.monotonic() + plan["limits"]["release_seconds"]
            while True:
                try:
                    release = shared.gpu_snapshot(); gpu_check(plan, release, {}, idle=True); break
                except ValueError:
                    require(time.monotonic() < end, "GPU release unverified")
                    time.sleep(1)
            if error is None and all_exited_zero:
                result = validate_outputs(plan, tasks)
        except BaseException as exc:
            error = (error or "") + "; cleanup: " + str(exc)
        for log in logs:
            log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        complete = error is None and result is not None
        receipt = {"status": "complete" if complete else "failed", "error": error, "plan_sha256": sha256,
                   "run_token": plan["run_token"], "started_epoch": started, "finished_epoch": time.time(),
                   "wall_seconds": time.time() - started, "worker_returncodes": [p.returncode for p, _, _ in children],
                   "sampled_peak_gpu_memory_used_mib": peak_gpu_used,
                   "process_release_verified": released,
                   "gpu_release_verified": release is not None and error is None, "gpu_release_snapshot": release, "generation" if plan["mode"] == "generate_batch_v1" else "tf": result}
        write_json(out / ("RUN_COMPLETE.json" if complete else "RUN_FAILED.json"), receipt)
        files = {str(p.relative_to(out)): {k: v for k, v in file_ref(p).items() if k != "path"}
                 for p in sorted(out.rglob("*")) if p.is_file()}
        write_json(out / "artifact_manifest.json", {"algorithm": "sha256", "plan_sha256": sha256, "files": files})
        for p in out.rglob("*"):
            if p.is_file(): p.chmod(0o400)
    require(error is None, error)
    return file_ref(out / "artifact_manifest.json")


def verify(path, sha256, artifact_sha256):
    plan, tasks = load_plan(path, sha256)
    out = Path(plan["output"]); artifact = out / "artifact_manifest.json"
    manifest = read_json_ref({"path": str(artifact), "sha256": artifact_sha256, "size_bytes": artifact.stat().st_size})
    require(manifest["plan_sha256"] == sha256 and manifest["algorithm"] == "sha256", "output artifact identity differs")
    actual = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"artifact_manifest.json"}, "output artifact extras/missing")
    for name, bound in manifest["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe artifact path")
        verify_file({"path": str(out / name), **bound})
    terminal = json.loads((out / "RUN_COMPLETE.json").read_bytes())
    require(terminal["status"] == "complete" and terminal["plan_sha256"] == sha256 and
            terminal["worker_returncodes"] == [0] * len(tasks) and terminal["process_release_verified"] is True and
            terminal["gpu_release_verified"] is True, "no positive released TF completion")
    # Actual outer controller exit is supplied by the invoking observer/service.
    key = "generation" if plan["mode"] == "generate_batch_v1" else "tf"
    require(validate_outputs(plan, tasks) == terminal[key], "Independent output verification differs")
    return {"status": "verified_batched_generation_bytes_and_coverage" if key == "generation" else "verified_tf_tranche_bytes_and_coverage",
            "artifact_manifest": file_ref(out / "artifact_manifest.json"),
            "plan_sha256": sha256, key: terminal[key], "outer_controller_exit_requires_external_receipt": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--artifact-sha256", help="Required with --verify; use the actual producer's artifact hash")
    args = parser.parse_args()
    if args.verify:
        parser.error("--verify requires --artifact-sha256") if not args.artifact_sha256 else None
        result = verify(args.plan, args.sha256, args.artifact_sha256)
    else:
        result = supervise(args.plan, args.sha256)
    print(canonical(result))
