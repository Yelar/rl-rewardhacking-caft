#!/usr/bin/env python3
"""Build a deterministic one-use review manifest for two-host collection."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import collect_factorial_rollouts as collector
import coordinate_factorial_campaign as coordinator
import factorial_common as common


def absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path))


def add_tree(files: dict[str, str], root: str, values: dict[str, str]) -> None:
    for relative, digest in values.items():
        path = str(Path(root) / relative)
        if path in files and files[path] != digest:
            raise ValueError(f"conflicting critical hash for {path}")
        files[path] = digest


def host_critical_files(report: dict[str, Any]) -> dict[str, str]:
    files: dict[str, str] = {}
    python = report["python_entrypoint"]
    files[python["path"]] = python["sha256"]
    add_tree(files, report["checkpoint_path"], report["checkpoint_hashes"])
    add_tree(files, report["base_model_snapshot_path"], report["base_model_snapshot_hashes"])
    files[report["dataset_path"]] = report["dataset_sha256"]
    files[report["existing_rollouts_path"]] = report["existing_rollouts_sha256"]
    add_tree(files, report["source_root"], report["source_hashes"])
    add_tree(files, report["review_inputs_path"], report["review_inputs_hashes"])
    files.update(report["runtime_source_files"])
    return dict(sorted(files.items()))


def verify_spec_matches_audit(host: str, spec: dict[str, Any], report: dict[str, Any]) -> None:
    expected = {
        "python": report["python_entrypoint"]["path"],
        "checkpoint": report["checkpoint_path"],
        "base_model_snapshot": report["base_model_snapshot_path"],
        "dataset": report["dataset_path"],
        "existing_rollouts": report["existing_rollouts_path"],
        "source_root": report["source_root"],
    }
    mismatches = {
        key: {"spec": spec.get(key), "audit": value}
        for key, value in expected.items() if spec.get(key) != value
    }
    review_root = Path(report["review_inputs_path"])
    for key, name in (
        ("grpo_full_config", "verl_full_config.yaml"),
        ("grpo_config", "verl_config.yaml"),
        ("grpo_run_config", "config.json"),
    ):
        expected_path = str(review_root / name)
        if spec.get(key) != expected_path:
            mismatches[key] = {"spec": spec.get(key), "audit": expected_path}
    source_root = Path(report["source_root"])
    expected_source = {
        "runner": source_root / "infra/gpu03/factorial_rollouts/collect_factorial_rollouts.py",
        "direct_entrypoint": source_root / "infra/gpu03/factorial_rollouts/factorial_direct_job.sh",
        "remote_verifier": source_root / "infra/gpu03/factorial_rollouts/verify_remote_manifest.py",
    }
    for key, path in expected_source.items():
        if spec.get(key) != str(path):
            mismatches[key] = {"spec": spec.get(key), "audit": str(path)}
    if mismatches:
        raise ValueError(f"{host} host specification differs from read-only audit: {mismatches}")
    if not report["python_entrypoint"]["executable"] or not all(
        report["source_executable_attestation"].values()
    ):
        raise ValueError(f"{host} reviewed Python or shell entrypoint is not executable")


def one_node_replay_command(spec: dict[str, Any], output: Path) -> list[str]:
    return [
        spec["direct_entrypoint"], spec["host"], spec["python"], spec["runner"],
        "--mode", "execute", "--checkpoint", spec["checkpoint"],
        "--base-model-snapshot", spec["base_model_snapshot"],
        "--dataset", spec["dataset"], "--existing-rollouts", spec["existing_rollouts"],
        "--grpo-full-config", spec["grpo_full_config"], "--grpo-config", spec["grpo_config"],
        "--grpo-run-config", spec["grpo_run_config"], "--output-dir", str(output),
        "--master-seed", "1", "--split-seed", str(collector.SPLIT_SEED),
        "--gpu-ids", *[str(gpu) for gpu in spec["gpu_ids"]],
        "--generation-plan", str(output / "generation_plan.jsonl"),
        "--max-new-generations", "40000", "--target-complete-problems", "200",
        "--evaluator-workers", "16", "--samples-per-problem-per-round", "8",
        "--pilot-new-generations", "2048", "--round-new-generation-limit", "10000",
        "--max-rounds", "6", "--wall-limit-seconds", "10800",
        "--gpu-memory-utilization", "0.60", "--max-num-seqs", "64",
        "--classification-batch-size", "32", "--cpus-per-gpu-worker", "6",
        "--worker-start-stagger-seconds", "10", "--gpu-quiescence-seconds", "60",
        "--gpu-poll-seconds", "5", "--min-start-available-memory-kib", "268435456",
        "--min-runtime-available-memory-kib", "201326592",
        "--max-combined-worker-rss-kib", "402653184", "--resume",
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    if socket.gethostname() != coordinator.COORDINATOR_HOST:
        raise RuntimeError("two-host review manifest must be built on gpu-04")
    if not re.fullmatch(r"codex-factorial-ckpt60-[0-9]{8}-[0-9]{6}", args.run_token):
        raise ValueError("unsafe run token")
    host_config = coordinator.load_host_config(args.host_config)
    if host_config["run_token"] != args.run_token:
        raise ValueError("run token differs from host configuration")
    output = args.output_dir.resolve()
    if Path(host_config["output_dir"]).resolve() != output:
        raise ValueError("output directory differs from host configuration")
    compatibility = json.loads(args.host_compatibility.read_text(encoding="utf-8"))
    if set(compatibility.get("hosts", {})) != coordinator.EXPECTED_HOSTS:
        raise ValueError("compatibility report must cover gpu-02 and gpu-04")
    if compatibility.get("critical_differences") or not compatibility.get("all_critical_hashes_match"):
        raise ValueError("selected hosts do not have identical critical inputs/runtime")
    for host, spec in host_config["hosts"].items():
        spec["host"] = host
        verify_spec_matches_audit(host, spec, compatibility["hosts"][host])
    config = json.loads((output / "run_config.yaml").read_text(encoding="utf-8"))
    if config["scientific"]["sampling"] != common.SAMPLING:
        raise ValueError("prepared package sampling differs from GRPO contract")
    if config["campaign"] != {
        "master_seed": 1, "split_seed": collector.SPLIT_SEED,
        "target_complete_problems": 200, "max_new_generations": 40000,
        "samples_per_problem_per_round": 8, "pilot_new_generations": 2048,
        "max_rounds": 6, "wall_limit_seconds": 10800,
    }:
        raise ValueError("prepared campaign limits differ")
    generation_plan = output / "generation_plan.jsonl"
    potential = list(common.read_jsonl(generation_plan))
    common.reviewed_universe_index(potential)
    local_spec = host_config["hosts"][coordinator.COORDINATOR_HOST]
    source_root = Path(local_spec["source_root"])
    expected_coordinator = source_root / "infra/gpu03/factorial_rollouts/coordinate_factorial_campaign.py"
    expected_entrypoint = source_root / "infra/gpu03/factorial_rollouts/factorial_coordinator_job.sh"
    receipt_writer = source_root / "infra/gpu03/factorial_rollouts/write_supervisor_receipt.py"
    if args.coordinator.resolve() != expected_coordinator.resolve():
        raise ValueError("coordinator executable path differs from staged reviewed source")
    if args.coordinator_entrypoint.resolve() != expected_entrypoint.resolve():
        raise ValueError("coordinator entrypoint path differs from staged reviewed source")
    if not os.access(args.coordinator_entrypoint, os.X_OK):
        raise ValueError("coordinator entrypoint is not executable")
    python_path = absolute_without_resolving_symlinks(Path(local_spec["python"]))
    command = [
        str(args.coordinator_entrypoint.resolve()), coordinator.COORDINATOR_HOST,
        str(python_path), str(args.coordinator.resolve()),
        "--host-config", str(args.host_config.resolve()),
        "--output-dir", str(output), "--state", str(args.state.resolve()),
        "--run-token", args.run_token, "--resume",
    ]
    control = args.control_dir.resolve()
    control.mkdir(parents=True, exist_ok=True)
    execution = {
        "host": coordinator.COORDINATOR_HOST, "run_token": args.run_token,
        "service_unit": args.run_token, "python": str(python_path),
        "coordinator": str(args.coordinator.resolve()),
        "coordinator_entrypoint": str(args.coordinator_entrypoint.resolve()),
        "receipt_writer": str(receipt_writer.resolve()),
        "host_config": str(args.host_config.resolve()), "output_dir": str(output),
        "state": str(args.state.resolve()),
        "launch_receipt": str(control / "launch_receipt.json"),
        "supervisor_status": str(control / "supervisor_status.json"),
        "launch_log": str(control / "launch.log"), "command": command,
        "systemd": {
            "runtime_max_seconds": 14400, "kill_mode": "control-group",
            "timeout_stop_seconds": 120, "restart": "on-failure",
            "restart_prevent_exit_status": coordinator.TERMINAL_FAILURE_EXIT,
        },
    }
    host_entries = {}
    for host, spec in host_config["hosts"].items():
        report = compatibility["hosts"][host]
        host_entries[host] = {
            "spec": spec,
            "critical_file_sha256": host_critical_files(report),
            "reviewed_gpu_rows": report["gpu_rows"],
            "reviewed_gpu_processes": report["gpu_processes"],
        }
    for path in (args.host_config.resolve(), args.host_compatibility.resolve()):
        host_entries[coordinator.COORDINATOR_HOST]["critical_file_sha256"][str(path)] = common.sha256_file(path)
    return {
        "schema_version": 2,
        "purpose": "checkpoint-60 fully crossed evaluator-modification rollout collection",
        "execution": execution,
        "hosts": host_entries,
        "scientific_contract": {
            "model": common.MODEL_ID, "revision": common.MODEL_REVISION,
            "checkpoint_step": 60, "sampling": common.SAMPLING,
            "sampling_count_is_not_distribution": True,
            "prompt_conditioning_for_cells": False,
            "target_complete_problems": 200, "max_new_generations": 40000,
        },
        "campaign_plan": {
            "path": str(generation_plan), "sha256": common.sha256_file(generation_plan),
            "potential_requests": len(potential),
            "adaptive_rounds": "full terminal rounds; 8 samples/incomplete problem; stable seeded ranking",
            "host_sharding": "sha256 request slot modulo 15; gpu-02 weight 7, gpu-04 weight 8",
        },
        "cross_host_compatibility": compatibility,
        "one_node_replay": {
            "host": coordinator.COORDINATOR_HOST,
            "command": one_node_replay_command(local_spec, output),
            "requires_separate_exact_approval": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-token", required=True)
    value.add_argument("--host-config", type=Path, required=True)
    value.add_argument("--host-compatibility", type=Path, required=True)
    value.add_argument("--coordinator", type=Path, required=True)
    value.add_argument("--coordinator-entrypoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--state", type=Path, required=True)
    value.add_argument("--control-dir", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    manifest = build(args)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(common.sha256_file(args.manifest))


if __name__ == "__main__":
    main()
