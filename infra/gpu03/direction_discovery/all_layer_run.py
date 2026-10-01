"""Cache-only all-layer analysis; each completed scientific cell is reusable.

The containing user-systemd service supplies the wall/memory/process limits.
This runner never loads a language model or executes completion code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from . import candidates as c
from . import token_representations as reps

MODULE = "infra.gpu03.direction_discovery.all_layer_run"


def append(path, value):
    with Path(path).open("a") as f:
        f.write(c.canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_plan(path, digest):
    c.require(c.sha256_file(path) == digest, "analysis plan changed")
    plan = json.loads(Path(path).read_text())
    if plan.get("purpose") == "broader_ordinary_all36_pca":
        from . import broader_pca
        plan = broader_pca.load_context(path, digest)[0]
        c.require(plan["workers"] == 8 and plan["gpu_devices"] == list(range(8)) and
                  len(plan["gpu_uuids"]) == len(set(plan["gpu_uuids"])) == 8,
                  "broader PCA requires eight explicit independent H100 workers")
        return plan
    c.require(plan["purpose"] == "existing_cache_all36_comparison", "wrong analysis purpose")
    c.require(plan["layers"] == list(range(36)), "all36 layers required")
    c.require(plan["representations"] == list(reps.PRIMARY), "representation coverage changed")
    c.require(plan["representation_policy_sha256"] == reps.definitions_sha256(), "token policy changed")
    for path, bound in plan["bound_files"].items():
        c.verify_file(path, bound)
    c.require(time.time() < plan["absolute_deadline_epoch"], "analysis deadline has passed")
    return plan


def fit_rows(plan):
    rows = c.read_jsonl(plan["prepared_records"])
    c.require(len(rows) == 333 and len({r["record_id"] for r in rows}) == 333, "wrong fitting record set")
    c.require({r["problem_split"] for r in rows} == {"direction_fit"}, "held-out data entered fitting")
    fit, inventory = c.validate_records(rows, {"excluded_problem_ids": [], "disputed_record_ids": []})
    c.require(inventory["fitting_problems"] == 111, "wrong fitting problem count")
    excluded = json.loads(Path(plan["exclusions"]).read_text())
    c.require(not ({str(r["problem_id_key"]) for r in fit} & set(excluded["excluded_problem_ids"])), "excluded problem entered fit")
    c.require(not ({r["record_id"] for r in fit} & set(excluded["disputed_record_ids"])), "disputed record entered fit")
    return fit


def make_cell(rows, selections, raw, region, name):
    """Keep every eligible record; the fitter separately declares pair intersections."""
    indices = [i for i in range(len(rows)) if selections[i][region][name]["eligible"]]
    selected = [rows[i] for i in indices]
    positions = [selections[i][region][name]["completion_positions"] for i in indices]
    if not selected:
        return None
    offsets = np.cumsum([0] + [len(ps) for ps in positions], dtype=np.int64)
    values = {kind: np.concatenate([raw[kind][i][ps] for i, ps in zip(indices, positions)])
              for kind in ("h0", "h60")}
    return c.WindowData(selected, positions, values["h0"], values["h60"], offsets,
                        np.repeat(np.arange(len(selected)), np.diff(offsets)),
                        np.concatenate([np.asarray(ps, dtype=np.int32) for ps in positions]))


def save_cell(path, report, tensors, identity):
    from safetensors.numpy import save_file
    path.mkdir(parents=True, exist_ok=False)
    c.write_json(path / "report.json", report)
    if tensors:
        save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path / "vectors.safetensors"))
    files = {p.name: {"sha256": c.sha256_file(p), "size_bytes": p.stat().st_size}
             for p in sorted(path.iterdir())}
    c.write_json(path / "complete.json", {**identity, "status": "complete", "files": files})
    for p in path.iterdir():
        p.chmod(0o400)


def existing_cell(path, identity):
    if not path.exists():
        return False
    c.require((path / "complete.json").is_file(), "incomplete cell retained; explicit recovery needed: " + str(path))
    receipt = json.loads((path / "complete.json").read_text())
    c.require(receipt["status"] == "complete" and all(receipt.get(k) == v for k, v in identity.items()), "completed cell identity differs")
    for name, bound in receipt["files"].items():
        c.verify_file(c.safe_child(path, name), bound)
    return True


def resource_check(plan, children=()):
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    c.require(available >= plan["minimum_available_ram_bytes"], "available RAM below analysis floor")
    disk = os.statvfs(plan["raw_root"])
    c.require(disk.f_bavail * disk.f_frsize >= plan["minimum_free_disk_bytes"], "free scratch below analysis floor")
    if plan.get("gpu_devices"):
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15, check=True)
        selected = set(plan["gpu_uuids"])
        owned = {p.pid for p in children}
        for line in result.stdout.splitlines():
            uuid, pid = [v.strip() for v in line.split(",")]
            c.require(uuid not in selected or int(pid) in owned, "foreign process appeared on selected analysis GPU")


def run_layers(path, digest, layers, worker):
    from . import all_layer_candidates as fit
    plan = read_plan(path, digest)
    c.require(all(layer in plan["layers"] for layer in layers), "unexpected layer")
    rows = fit_rows(plan)
    selections = [reps.RecordRepresentations(r).all() for r in rows]
    raw_manifest = json.loads(Path(plan["raw_artifact_manifest"]).read_text())
    proof = json.loads(Path(plan["copy_verification"]).read_text())
    reader = c.RawReader(Path(plan["raw_root"]), rows, c.read_jsonl(plan["activation_index"]), raw_manifest,
                         plan["cache_extraction_manifest_sha256"], 36, 2560, integrity_receipt=proof)
    out = Path(plan["output"])
    journal = out / f"worker_{worker:02d}.jsonl"
    config = dict(plan["fit_config"])
    if plan.get("gpu_devices"):
        config["pca_backend"] = "cuda:0"
    for layer in layers:
        c.require(time.time() < plan["absolute_deadline_epoch"], "analysis deadline reached")
        start = time.monotonic()
        append(journal, {"event": "layer_started", "layer": layer, "time": time.time()})
        # One pass through all completion positions. No cache re-extraction or token subsampling.
        raw = {kind: [reader.read(r, kind, layer, list(range(r["completion_token_count"]))) for r in rows]
               for kind in ("h0", "h60")}
        for region in reps.REGIONS:
            for name in plan["representations"]:
                semantics = "anchor" if name in reps.ANCHORS else "window_mean"
                cell_path = out / f"layer_{layer:02d}" / region / name
                identity = {"plan_sha256": digest, "layer": layer, "region": region, "representation": name}
                if existing_cell(cell_path, identity):
                    append(journal, {"event": "cell_reused", **identity})
                    continue
                data = make_cell(rows, selections, raw, region, name)
                cell_start = time.monotonic()
                if data is None:
                    report, tensors = {"status": "unsupported", "reason": "no_eligible_records"}, {}
                else:
                    report, tensors = fit.fit_representation(data, layer=layer, region=region,
                        representation=name, semantics=semantics, config=config)
                    if semantics != "anchor":
                        token_report, token_tensors = fit.fit_representation(data, layer=layer, region=region,
                            representation=name, semantics="token_level", config={**config, "methods": ["pca"]})
                        report["token_level_pca"] = token_report
                        tensors.update({"token_level." + key: value for key, value in token_tensors.items()})
                report["measured_wall_seconds"] = time.monotonic() - cell_start
                save_cell(cell_path, report, tensors, identity)
                append(journal, {"event": "cell_complete", **identity,
                                 "seconds": report["measured_wall_seconds"], "time": time.time()})
                del data, report, tensors
        del raw
        append(journal, {"event": "layer_complete", "layer": layer,
                         "seconds": time.monotonic() - start, "time": time.time()})


def broader_completion(out, plan, digest):
    """Exact cell bytes/shards, including explicit unsupported-cell reports."""
    expected = set()
    for layer in plan["layers"]:
        for region in reps.REGIONS:
            for representation in plan["representations"]:
                cell = out / f"layer_{layer:02d}" / region / representation
                expected.add(cell / "complete.json")
                c.require(existing_cell(cell, {"plan_sha256": digest, "cohort": "broader_ordinary",
                          "layer": layer, "region": region, "representation": representation}),
                          "missing broader PCA cell")
    c.require(set(out.glob("layer_*/*/*/complete.json")) == expected, "unexpected broader cell inventory")
    for worker in range(8):
        receipt = json.loads((out / f"worker_{worker:02d}_SUCCESS.json").read_text())
        c.require(receipt == {"status": "complete", "plan_sha256": digest, "cohort": "broader_ordinary",
                  "layers": plan["layers"][worker::8], "worker": worker}, "broader worker shard receipt differs")
    return {"cohort": "broader_ordinary", "cells": len(expected), "worker_successes": 8,
            "numerical_qualification_is_reported_per_cell": True}


def supervise(path, digest):
    plan = read_plan(path, digest)
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=True)
    c.require(not (out / "RUN_COMPLETE.json").exists(), "completed analysis cannot be relaunched")
    workers = plan["workers"]
    children, logs = [], []
    status, error = "failed", None
    broader = plan.get("purpose") == "broader_ordinary_all36_pca"
    coverage = None
    started = time.time()
    resource_check(plan)
    append(out / "run_attempts.jsonl", {"pid": os.getpid(), "plan_sha256": digest,
                 "started_epoch": started, "versions": {n: importlib.metadata.version(n) for n in
                    ("numpy", "scipy", "scikit-learn", "torch", "safetensors")}})
    try:
        for worker in range(workers):
            layers = plan["layers"][worker::workers]
            if not layers:
                continue
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(plan["gpu_devices"][worker]) if plan.get("gpu_devices") else ""
            worker_module = "infra.gpu03.direction_discovery.broader_pca" if broader else MODULE
            command = [sys.executable, "-B", "-m", worker_module, "--plan", str(path), "--sha256", digest,
                       "--worker", str(worker), "--layers", *map(str, layers)]
            # Runtime logs remain append-only when completed cells are resumed.
            log = (out / f"worker_{worker:02d}.log").open("ab")
            logs.append(log)
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True))
        while any(p.poll() is None for p in children):
            c.require(time.time() < plan["absolute_deadline_epoch"], "absolute analysis deadline reached")
            resource_check(plan, children)
            for i, p in enumerate(children):
                c.require(p.poll() in (None, 0), f"analysis worker {i} failed with {p.returncode}")
            time.sleep(5)
        c.require(all(p.returncode == 0 for p in children), "analysis worker failed")
        if broader:
            coverage = broader_completion(out, plan, digest)
        else:
            expected = len(plan["layers"]) * len(reps.REGIONS) * len(plan["representations"])
            receipts = list(out.glob("layer_*/*/*/complete.json"))
            c.require(len(receipts) == expected, "missing representation/layer cells")
        status = "complete"
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in children:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
        for log in logs:
            log.close()
        terminal = {
            "status": status, "error": error, "plan_sha256": digest, "started_epoch": started,
            "finished_epoch": time.time(), "worker_returncodes": [p.returncode for p in children],
            "owned_children_reaped": True}
        if broader:
            terminal["cohort"] = "broader_ordinary"
            terminal["completion_coverage"] = coverage
        c.write_json(out / ("RUN_COMPLETE.json" if status == "complete" else f"RUN_FAILED.{int(started * 1000)}.json"), terminal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args()
    if args.worker is None:
        supervise(args.plan, args.sha256)
    else:
        run_layers(args.plan, args.sha256, args.layers, args.worker)
