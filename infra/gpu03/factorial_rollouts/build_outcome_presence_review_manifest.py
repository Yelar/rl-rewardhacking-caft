#!/usr/bin/env python3
"""Build a deterministic review manifest for an outcome-presence campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_factorial_review_manifest as legacy
import collect_factorial_rollouts as collector
import coordinate_factorial_campaign as coordinator
import factorial_common as base
import outcome_presence_common as mode


def one_node_command(spec: dict, output: Path, config: dict) -> list[str]:
    limits = config["limits"]
    command = [
        spec["direct_entrypoint"], spec["host"], spec["python"], spec["runner"],
        "--mode", "execute", "--dataset-mode", "outcome_presence",
        "--outcome-target", config["outcome_target"],
        "--checkpoint", spec["checkpoint"], "--base-model-snapshot", spec["base_model_snapshot"],
        "--dataset", spec["dataset"], "--existing-rollouts", spec["existing_rollouts"],
        "--grpo-full-config", spec["grpo_full_config"], "--grpo-config", spec["grpo_config"],
        "--grpo-run-config", spec["grpo_run_config"], "--output-dir", str(output),
        "--master-seed", "1", "--split-seed", str(collector.SPLIT_SEED),
        "--gpu-ids", *[str(gpu) for gpu in spec["gpu_ids"]],
        "--generation-plan", str(output / "generation_plan.jsonl"),
        "--max-new-generations", str(limits["max_new_generations"]),
        "--target-complete-problems", "200",
        "--evaluator-workers", "16", "--samples-per-problem-per-round", "8",
        "--pilot-new-generations", "2048",
        "--round-new-generation-limit", str(limits["round_new_generation_limit"]),
        "--max-rounds", str(limits["max_rounds"]),
        "--wall-limit-seconds", str(limits["wall_limit_seconds"]),
        "--gpu-memory-utilization", "0.60", "--max-num-seqs", "64",
        "--classification-batch-size", "32", "--cpus-per-gpu-worker", "6",
        "--worker-start-stagger-seconds", "10", "--gpu-quiescence-seconds", "60",
        "--gpu-poll-seconds", "5", "--min-start-available-memory-kib", "268435456",
        "--min-runtime-available-memory-kib", "201326592",
        "--max-combined-worker-rss-kib", "402653184", "--resume",
    ]
    if limits.get("require_exact_generations") is True:
        command.append("--require-exact-generations")
    return command


def build(args: argparse.Namespace) -> dict:
    if socket.gethostname() != coordinator.COORDINATOR_HOST:
        raise RuntimeError("review manifest must be built on gpu-04")
    if not re.fullmatch(r"codex-outcome-presence-ckpt60-[0-9]{8}-[0-9]{6}", args.run_token):
        raise ValueError("unsafe run token")
    host_config = coordinator.load_host_config(args.host_config)
    if host_config.get("dataset_mode") != "outcome_presence":
        raise ValueError("host configuration mode differs")
    campaign_kind = host_config.get("campaign_kind")
    outcome_target = host_config.get("outcome_target", mode.TARGET_FIVE_CLASS)
    if campaign_kind not in {
        "feasibility_pilot", "core_triplet", "core_triplet_exact_100k"
    }:
        raise ValueError("unsupported outcome-presence campaign kind")
    if campaign_kind == "feasibility_pilot" and outcome_target != mode.TARGET_FIVE_CLASS:
        raise ValueError("the feasibility pilot must retain the five-class target")
    if campaign_kind in {"core_triplet", "core_triplet_exact_100k"} and outcome_target != mode.TARGET_CORE_TRIPLET:
        raise ValueError("the core campaign must target the three evaluator-present classes")
    core_campaign = campaign_kind in {"core_triplet", "core_triplet_exact_100k"}
    exact_campaign = campaign_kind == "core_triplet_exact_100k"
    if host_config["run_token"] != args.run_token:
        raise ValueError("run token differs")
    output = args.output_dir.resolve()
    if Path(host_config["output_dir"]).resolve() != output:
        raise ValueError("output path differs")
    compatibility = json.loads(args.host_compatibility.read_text())
    expected_hosts = set(host_config["hosts"])
    if set(compatibility.get("hosts", {})) != expected_hosts:
        raise ValueError("compatibility report must cover the reviewed host profile")
    if compatibility.get("critical_differences") or not compatibility.get("all_critical_hashes_match"):
        raise ValueError("critical host inputs differ")
    for host, spec in host_config["hosts"].items():
        spec["host"] = host
        legacy.verify_spec_matches_audit(host, spec, compatibility["hosts"][host])
    config = json.loads((output / "run_config.yaml").read_text())
    if config.get("dataset_mode") != "outcome_presence":
        raise ValueError("prepared package mode differs")
    if config["scientific"].get("outcome_target") != outcome_target:
        raise ValueError("prepared package target differs")
    if config["scientific"]["sampling"] != mode.SAMPLING:
        raise ValueError("sampling contract differs")
    expected_campaign = {
        "master_seed": 1, "split_seed": collector.SPLIT_SEED,
        "target_complete_problems": 200,
        "max_new_generations": 100000 if core_campaign else 2048,
        "samples_per_problem_per_round": 8, "pilot_new_generations": 2048,
        "max_rounds": 14 if exact_campaign else 13 if core_campaign else 1,
        "wall_limit_seconds": 86400 if exact_campaign else 10800 if core_campaign else 3600,
        **({"require_exact_generations": True} if exact_campaign else {}),
    }
    if config["campaign"] != expected_campaign:
        raise ValueError("bounded campaign limits differ")
    prepared_summary = json.loads((output / "summary.json").read_text())
    if prepared_summary.get("status") != "prepared" or prepared_summary.get("dataset_mode") != "outcome_presence":
        raise ValueError("output package is not in prepared outcome-presence state")
    expected_artifact = json.loads((output / "artifact_manifest.json").read_text())
    if expected_artifact != base.file_manifest(output):
        raise ValueError("prepared package artifact manifest differs")
    if not core_campaign:
        pilot = json.loads((output / "pilot_summary.json").read_text())
        gate = pilot["feasibility_gate"]
        if gate["full_campaign_approval_may_be_prepared"]:
            raise ValueError("bounded pilot manifest is unnecessary because reuse passed the gate")
    universe_path = output / "generation_plan.jsonl"
    universe = list(base.read_jsonl(universe_path))
    base.reviewed_universe_index(universe)
    expected_universe = 992 * 8 * (14 if exact_campaign else 13 if core_campaign else 1)
    if len(universe) != expected_universe:
        raise ValueError("reviewed universe size differs from campaign limits")
    exact_round_budgets = (
        mode.exact_generation_round_budgets(
            problem_count=992,
            max_new_generations=host_config["limits"]["max_new_generations"],
            pilot_new_generations=host_config["limits"]["pilot_new_generations"],
            round_new_generation_limit=host_config["limits"]["round_new_generation_limit"],
            samples_per_problem=host_config["limits"]["samples_per_problem_per_round"],
            max_rounds=host_config["limits"]["max_rounds"],
        ) if exact_campaign else None
    )
    local = host_config["hosts"][coordinator.COORDINATOR_HOST]
    source_root = Path(local["source_root"])
    captured = output / "source"
    source_pairs = {
        captured / name: source_root / "infra/gpu03/factorial_rollouts" / name
        for name in (
            "collect_factorial_rollouts.py", "collect_outcome_presence.py",
            "factorial_common.py", "outcome_presence_common.py",
            "coordinate_factorial_campaign.py", "launch_factorial_job.py",
            "launch_outcome_presence_job.py", "build_outcome_presence_review_manifest.py",
            "launch_outcome_presence_core_job.py",
            "verify_outcome_presence_package.py", "write_supervisor_receipt.py",
        )
    }
    source_pairs[captured / "structural_positions.py"] = (
        source_root / "infra/gpu03/activation_dataset/structural_positions.py"
    )
    for packaged, staged in source_pairs.items():
        if not packaged.is_file() or not staged.is_file() or base.sha256_file(packaged) != base.sha256_file(staged):
            raise ValueError(f"packaged source differs from staged reviewed source: {packaged.name}")
    expected_coordinator = source_root / "infra/gpu03/factorial_rollouts/coordinate_factorial_campaign.py"
    expected_entrypoint = source_root / "infra/gpu03/factorial_rollouts/factorial_coordinator_job.sh"
    receipt_writer = source_root / "infra/gpu03/factorial_rollouts/write_supervisor_receipt.py"
    command = [
        str(expected_entrypoint), coordinator.COORDINATOR_HOST,
        str(Path(local["python"])), str(expected_coordinator),
        "--host-config", str(args.host_config.resolve()), "--output-dir", str(output),
        "--state", str(args.state.resolve()), "--run-token", args.run_token, "--resume",
    ]
    control = args.control_dir.resolve()
    wall_limit = host_config["limits"]["wall_limit_seconds"]
    execution = {
        "host": coordinator.COORDINATOR_HOST, "run_token": args.run_token,
        "host_profile": host_config.get("host_profile", "gpu02_gpu04"),
        "service_unit": args.run_token, "python": str(Path(local["python"])),
        "coordinator": str(expected_coordinator), "coordinator_entrypoint": str(expected_entrypoint),
        "receipt_writer": str(receipt_writer), "host_config": str(args.host_config.resolve()),
        "output_dir": str(output), "state": str(args.state.resolve()),
        "control_dir": str(control),
        "launch_receipt": str(output / "launch_receipt.json"),
        "supervisor_status": str(output / "supervisor_receipt.json"),
        "launch_log": str(output / "launch.log"), "command": command,
        "systemd": {"runtime_max_seconds": wall_limit + 120, "kill_mode": "control-group",
                    "timeout_stop_seconds": 120, "restart": "on-failure"},
    }
    hosts = {}
    for host, spec in host_config["hosts"].items():
        report = compatibility["hosts"][host]
        hosts[host] = {
            "spec": spec, "critical_file_sha256": legacy.host_critical_files(report),
            "reviewed_gpu_rows": report["gpu_rows"],
            "reviewed_gpu_processes": report["gpu_processes"],
        }
    for path in (args.host_config.resolve(), args.host_compatibility.resolve()):
        hosts[coordinator.COORDINATOR_HOST]["critical_file_sha256"][str(path)] = base.sha256_file(path)
    purpose = (
        "checkpoint-60 evaluator-present core-triplet rollout collection"
        if core_campaign else "checkpoint-60 bounded outcome-presence feasibility pilot"
    )
    target_cells = mode.PRESENT_CELLS if core_campaign else mode.CELLS
    max_generations = host_config["limits"]["max_new_generations"]
    return {
        "schema_version": 2,
        "purpose": purpose,
        "execution": execution, "hosts": hosts,
        "scientific_contract": {
            "dataset_mode": "outcome_presence", "outcome_target": outcome_target,
            "target_classes": list(target_cells), "model": mode.MODEL_ID,
            "revision": mode.MODEL_REVISION, "checkpoint_step": 60,
            "sampling": mode.SAMPLING, "prompt_conditioning_for_classes": False,
            "preferred_target_complete_problems": 200,
            "full_campaign_authorized": core_campaign,
            "new_generation_cap": max_generations,
        },
        "campaign_plan": {
            "path": str(universe_path), "sha256": base.sha256_file(universe_path),
            "potential_requests": len(universe), "executed_requests_maximum": max_generations,
            "exact_round_request_counts": list(exact_round_budgets) if exact_round_budgets else None,
            "adaptive_rounds": (
                "fourteen immutable rounds; incomplete core triplets rank first and completed problems fill the exact 100,000-request budget"
                if exact_campaign else
                "up to thirteen immutable eight-sample rounds targeting only incomplete core triplets"
                if core_campaign else
                "one immutable eight-sample round over 256 stably ranked incomplete problems"
            ),
            "host_sharding": (
                "single reviewed gpu-04 shard"
                if set(host_config["hosts"]) == {coordinator.COORDINATOR_HOST}
                else "SHA-256 request slots weighted by reviewed idle GPU counts"
            ),
        },
        "cross_host_compatibility": compatibility,
        "one_node_replay": {"host": coordinator.COORDINATOR_HOST,
                            "command": one_node_command(local, output, host_config),
                            "requires_separate_exact_approval": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--host-config", type=Path, required=True)
    parser.add_argument("--host-compatibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(base.sha256_file(args.manifest))


if __name__ == "__main__":
    main()
