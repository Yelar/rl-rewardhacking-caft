#!/usr/bin/env python3
"""Resume-safe deterministic one- or two-host rollout campaign coordinator.

The coordinator never executes generated code itself.  For each immutable round
it starts independent, bounded systemd services on the reviewed hosts, waits for
all disjoint shards to become terminal, and only then merges and plans again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import collect_factorial_rollouts as collector
import factorial_common as common
import outcome_presence_common as outcome_presence


COORDINATOR_HOST = "gpu-04"
HOST_PROFILES = {
    "gpu02_gpu04": {"gpu-02": list(range(1, 8)), "gpu-04": list(range(8))},
    "gpu04_only": {"gpu-04": list(range(8))},
    "gpu04_gpus1_7": {"gpu-04": list(range(1, 8))},
}
TERMINAL_FAILURE_EXIT = 78
SAFE_VALUE = re.compile(r"[A-Za-z0-9_./:@+=,-]+")


class CampaignFailure(RuntimeError):
    """Fail-closed campaign outcome which systemd must not auto-restart."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_state(path: Path, value: dict[str, Any]) -> None:
    common.atomic_write_json(path, value)


def validate_safe_value(value: str) -> str:
    if not value or not SAFE_VALUE.fullmatch(value):
        raise ValueError(f"unsafe command value: {value!r}")
    return value


def load_host_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported host configuration")
    if config.get("coordinator_host") != COORDINATOR_HOST:
        raise ValueError("reviewed coordinator must be gpu-04")
    host_profile = config.get("host_profile", "gpu02_gpu04")
    if host_profile not in HOST_PROFILES:
        raise ValueError("unsupported reviewed host profile")
    hosts = config.get("hosts")
    expected_profile = HOST_PROFILES[host_profile]
    expected_hosts = set(expected_profile)
    if not isinstance(hosts, dict) or set(hosts) != expected_hosts:
        raise ValueError(f"reviewed hosts differ from profile {host_profile}")
    dataset_mode = config.get("dataset_mode", "factorial")
    if dataset_mode not in {"factorial", "outcome_presence"}:
        raise ValueError("unsupported reviewed dataset mode")
    reviewed_outcome_target = config.get(
        "outcome_target", outcome_presence.TARGET_FIVE_CLASS
    )
    if dataset_mode == "outcome_presence":
        outcome_presence.target_cells(reviewed_outcome_target)
    elif reviewed_outcome_target != outcome_presence.TARGET_FIVE_CLASS:
        raise ValueError("factorial mode does not accept an outcome-presence target")
    for host, spec in hosts.items():
        gpu_ids = spec.get("gpu_ids")
        if dataset_mode == "factorial" or host_profile != "gpu02_gpu04":
            if gpu_ids != expected_profile[host]:
                raise ValueError(f"unexpected reviewed GPU set for {host}")
        elif (
            not isinstance(gpu_ids, list) or not gpu_ids
            or gpu_ids != sorted(set(gpu_ids))
            or any(not isinstance(gpu, int) or gpu not in range(8) for gpu in gpu_ids)
        ):
            raise ValueError(f"invalid reviewed GPU subset for {host}")
        if bool(spec.get("local")) != (host == COORDINATOR_HOST):
            raise ValueError(f"wrong local-host designation for {host}")
        for key in (
            "python", "runner", "direct_entrypoint", "checkpoint",
            "base_model_snapshot", "dataset", "existing_rollouts",
            "grpo_full_config", "grpo_config", "grpo_run_config", "work_dir",
            "source_root", "remote_verifier", "reviewed_manifest",
        ):
            validate_safe_value(str(spec.get(key, "")))
        if not spec.get("local") and spec.get("ssh_target") != host:
            raise ValueError(f"SSH target is not exact for {host}")
    full_limits = {
        "master_seed": 1,
        "split_seed": collector.SPLIT_SEED,
        "target_complete_problems": 200,
        "max_new_generations": 40000,
        "samples_per_problem_per_round": 8,
        "pilot_new_generations": 2048,
        "round_new_generation_limit": 10000,
        "max_rounds": 6,
        "wall_limit_seconds": 10800,
        "gpu_memory_utilization": 0.60,
        "max_num_seqs": 64,
        "classification_batch_size": 32,
        "evaluator_workers_per_host": 16,
        "cpus_per_gpu_worker": 6,
        "worker_start_stagger_seconds": 10,
        "gpu_quiescence_seconds": 60,
        "gpu_poll_seconds": 5,
        "min_start_available_memory_kib": 268435456,
        "min_runtime_available_memory_kib": 201326592,
        "max_combined_worker_rss_kib": 402653184,
    }
    pilot_limits = {
        **full_limits,
        "max_new_generations": 2048,
        "round_new_generation_limit": 2048,
        "max_rounds": 1,
        "wall_limit_seconds": 3600,
    }
    core_limits = {
        **full_limits,
        "max_new_generations": 100000,
        "max_rounds": 13,
    }
    exact_core_limits = {
        **full_limits,
        "max_new_generations": 100000,
        "max_rounds": 14,
        "wall_limit_seconds": 86400,
        "require_exact_generations": True,
    }
    campaign_kind = config.get("campaign_kind", "full")
    expected_limits = (
        pilot_limits if campaign_kind == "feasibility_pilot"
        else core_limits if campaign_kind == "core_triplet"
        else exact_core_limits if campaign_kind == "core_triplet_exact_100k"
        else full_limits
    )
    if campaign_kind not in {
        "full", "feasibility_pilot", "core_triplet", "core_triplet_exact_100k"
    }:
        raise ValueError("unsupported reviewed campaign kind")
    if campaign_kind == "feasibility_pilot" and config.get("dataset_mode") != "outcome_presence":
        raise ValueError("the feasibility pilot is only defined for outcome_presence")
    if campaign_kind in {"core_triplet", "core_triplet_exact_100k"} and (
        dataset_mode != "outcome_presence"
        or reviewed_outcome_target != outcome_presence.TARGET_CORE_TRIPLET
    ):
        raise ValueError("core_triplet campaign must target the evaluator-present core")
    if (
        host_profile in {"gpu04_only", "gpu04_gpus1_7"}
        and campaign_kind not in {"core_triplet", "core_triplet_exact_100k"}
    ):
        raise ValueError("single-host profiles are reviewed only for the core_triplet campaign")
    if config.get("limits") != expected_limits:
        raise ValueError("execution limits differ from reviewed values")
    if campaign_kind == "core_triplet_exact_100k":
        budgets = outcome_presence.exact_generation_round_budgets(
            problem_count=992,
            max_new_generations=exact_core_limits["max_new_generations"],
            pilot_new_generations=exact_core_limits["pilot_new_generations"],
            round_new_generation_limit=exact_core_limits["round_new_generation_limit"],
            samples_per_problem=exact_core_limits["samples_per_problem_per_round"],
            max_rounds=exact_core_limits["max_rounds"],
        )
        if sum(budgets) != 100000:
            raise ValueError("exact core-triplet schedule does not total 100,000")
    return config


def host_call(spec: dict[str, Any], command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    for item in command:
        validate_safe_value(str(item))
    full = command if spec["local"] else [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        spec["ssh_target"], "--", *command,
    ]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def require_host_call(spec: dict[str, Any], command: list[str], *, timeout: int = 30) -> str:
    result = host_call(spec, command, timeout=timeout)
    if result.returncode != 0:
        raise CampaignFailure(
            f"{spec['host']} command failed rc={result.returncode}: {result.stderr[-500:]}"
        )
    return result.stdout


def remote_sha256(spec: dict[str, Any], path: str) -> str | None:
    if spec["local"]:
        candidate = Path(path)
        return common.sha256_file(candidate) if candidate.is_file() else None
    result = host_call(spec, ["sha256sum", path], timeout=30)
    if result.returncode != 0:
        return None
    fields = result.stdout.strip().split()
    return fields[0] if len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) else None


def copy_reviewed_file(spec: dict[str, Any], source: Path, destination: str) -> None:
    digest = common.sha256_file(source)
    observed = remote_sha256(spec, destination)
    if observed is not None:
        if observed != digest:
            raise CampaignFailure(f"refusing to replace conflicting immutable file on {spec['host']}")
        return
    parent = str(Path(destination).parent)
    require_host_call(spec, ["mkdir", "-p", parent])
    if spec["local"]:
        temporary = Path(destination + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    else:
        result = subprocess.run(
            ["rsync", "--archive", "--checksum", str(source), f"{spec['ssh_target']}:{destination}.tmp"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise CampaignFailure(f"rsync to {spec['host']} failed: {result.stderr[-500:]}")
        require_host_call(spec, ["mv", destination + ".tmp", destination])
    if remote_sha256(spec, destination) != digest:
        raise CampaignFailure(f"post-copy hash mismatch on {spec['host']}: {destination}")


def copy_from_host(spec: dict[str, Any], source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if spec["local"]:
        shutil.copy2(source, temporary)
    else:
        result = subprocess.run(
            ["rsync", "--archive", "--checksum", f"{spec['ssh_target']}:{source}", str(temporary)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise CampaignFailure(f"rsync from {spec['host']} failed: {result.stderr[-500:]}")
    os.replace(temporary, destination)


def copy_worker_journals(
    spec: dict[str, Any], destination: Path, round_number: int,
) -> list[str]:
    """Copy append-only journals into a host/round-specific package path."""
    destination.mkdir(parents=True, exist_ok=True)
    pattern = f"round_{round_number:03d}_worker_"
    if spec["local"]:
        sources = sorted((Path(spec["work_dir"]) / "workers").glob(pattern + "*.jsonl"))
        for source in sources:
            shutil.copy2(source, destination / source.name)
    else:
        result = subprocess.run(
            [
                "rsync", "--archive", "--checksum", f"--include={pattern}*.jsonl",
                "--exclude=*", f"{spec['ssh_target']}:{spec['work_dir']}/workers/",
                str(destination) + "/",
            ], capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            raise CampaignFailure(f"worker journal copy failed on {spec['host']}")
    names = sorted(path.name for path in destination.glob(pattern + "*.jsonl"))
    if not names:
        raise CampaignFailure(f"no worker journals copied from {spec['host']}")
    return names


def collector_args(
    spec: dict[str, Any], *, mode: str, wall_seconds: int,
    request_plan: str | None = None, round_number: int | None = None,
) -> list[str]:
    work = spec["work_dir"]
    limits = spec.get("limits", {})
    max_new = int(limits.get("max_new_generations", 40000))
    round_limit = int(limits.get("round_new_generation_limit", 10000))
    max_rounds = int(limits.get("max_rounds", 6))
    command = [
        spec["direct_entrypoint"], spec["host"], spec["python"], spec["runner"],
        "--mode", mode,
        "--dataset-mode", spec.get("dataset_mode", "factorial"),
        "--outcome-target", spec.get(
            "outcome_target", outcome_presence.TARGET_FIVE_CLASS
        ),
        "--checkpoint", spec["checkpoint"],
        "--base-model-snapshot", spec["base_model_snapshot"],
        "--dataset", spec["dataset"],
        "--existing-rollouts", spec["existing_rollouts"],
        "--grpo-full-config", spec["grpo_full_config"],
        "--grpo-config", spec["grpo_config"],
        "--grpo-run-config", spec["grpo_run_config"],
        "--output-dir", work,
        "--master-seed", "1", "--split-seed", str(collector.SPLIT_SEED),
        "--gpu-ids", *[str(gpu) for gpu in spec["gpu_ids"]],
        "--generation-plan", f"{work}/generation_plan.jsonl",
        "--max-new-generations", str(max_new), "--target-complete-problems", "200",
        "--evaluator-workers", "16", "--samples-per-problem-per-round", "8",
        "--pilot-new-generations", "2048", "--round-new-generation-limit", str(round_limit),
        "--max-rounds", str(max_rounds),
        "--wall-limit-seconds", str(max(1, min(wall_seconds, int(limits.get("wall_limit_seconds", 10800))))),
        "--gpu-memory-utilization", "0.60", "--max-num-seqs", "64",
        "--classification-batch-size", "32", "--cpus-per-gpu-worker", "6",
        "--worker-start-stagger-seconds", "10", "--gpu-quiescence-seconds", "60",
        "--gpu-poll-seconds", "5", "--min-start-available-memory-kib", "268435456",
        "--min-runtime-available-memory-kib", "201326592",
        "--max-combined-worker-rss-kib", "402653184", "--resume",
    ]
    if limits.get("require_exact_generations") is True:
        command.append("--require-exact-generations")
    if mode == "run-plan":
        if request_plan is None or round_number is None:
            raise ValueError("run-plan requires a request path and round number")
        command.extend(["--request-plan", request_plan, "--plan-round-number", str(round_number)])
    return command


def unit_name(token: str, host: str, stage: str) -> str:
    value = f"{token}-{host.replace('gpu-', 'g')}-{stage}"
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", value):
        raise ValueError("unsafe systemd unit")
    return value


def systemd_start(spec: dict[str, Any], unit: str, command: list[str], wall_seconds: int, token: str) -> None:
    launch_id = common.sha256_text(f"{token}:{spec['host']}:{unit}")[:32]
    log_path = f"{spec['work_dir']}/{unit}.log"
    runtime = max(180, min(wall_seconds + 120, 86520))
    systemd = [
        "systemd-run", "--user", f"--unit={unit}", "--property=Type=exec",
        # Keep the transient unit loaded after a successful child exits.  Without
        # this, systemd may garbage-collect it before the coordinator's next
        # bounded poll, making success indistinguishable from disappearance.
        "--property=RemainAfterExit=yes",
        "--property=KillMode=control-group", f"--property=RuntimeMaxSec={runtime}s",
        "--property=TimeoutStopSec=120s", f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        f"--setenv=CODEX_FACTORIAL_LAUNCH_ID={launch_id}", *command,
    ]
    require_host_call(spec, systemd, timeout=45)


def unit_status(spec: dict[str, Any], unit: str) -> dict[str, str]:
    result = host_call(spec, [
        "systemctl", "--user", "show", f"{unit}.service",
        "-p", "LoadState", "-p", "ActiveState", "-p", "SubState",
        "-p", "Result", "-p", "ExecMainStatus",
    ], timeout=25)
    if result.returncode != 0:
        return {"LoadState": "not-found", "ActiveState": "unknown"}
    fields = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def stop_units(hosts: dict[str, dict[str, Any]], units: dict[str, str]) -> None:
    for host, unit in units.items():
        try:
            host_call(hosts[host], ["systemctl", "--user", "stop", f"{unit}.service"], timeout=150)
        except (OSError, subprocess.SubprocessError):
            pass


def wait_units(
    hosts: dict[str, dict[str, Any]], units: dict[str, str], deadline_epoch: float,
    state_path: Path, state: dict[str, Any],
) -> None:
    absent_counts = {host: 0 for host in units}
    while True:
        if time.time() >= deadline_epoch:
            stop_units(hosts, units)
            raise CampaignFailure("reviewed campaign deadline reached while host services were active")
        statuses = {host: unit_status(hosts[host], unit) for host, unit in units.items()}
        state["last_unit_status"] = statuses
        state["updated_at"] = utc_now()
        atomic_state(state_path, state)
        terminal = True
        for host, status in statuses.items():
            if status.get("LoadState") == "not-found":
                absent_counts[host] += 1
                if absent_counts[host] >= 3:
                    stop_units(hosts, units)
                    raise CampaignFailure(f"reviewed service disappeared on {host}")
                terminal = False
            elif (
                status.get("ActiveState") == "active"
                and status.get("SubState") == "exited"
            ):
                if status.get("Result") != "success" or status.get("ExecMainStatus") != "0":
                    stop_units(hosts, units)
                    raise CampaignFailure(f"reviewed service failed on {host}: {status}")
            elif status.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}:
                terminal = False
            elif status.get("Result") != "success" or status.get("ExecMainStatus") != "0":
                stop_units(hosts, units)
                raise CampaignFailure(f"reviewed service failed on {host}: {status}")
        if terminal:
            return
        time.sleep(min(5, max(deadline_epoch - time.time(), 0.1)))


def start_or_rejoin_services(
    *, hosts: dict[str, dict[str, Any]], stage: str, commands: dict[str, list[str]],
    deadline_epoch: float, token: str, state_path: Path, state: dict[str, Any],
) -> dict[str, str]:
    units = {host: unit_name(token, host, stage) for host in sorted(hosts)}
    service_state = state.setdefault("services", {}).setdefault(stage, {})
    for host, unit in units.items():
        prior = service_state.get(host)
        if prior is not None and prior != unit:
            raise CampaignFailure("persisted service identity differs from deterministic identity")
        service_state[host] = unit
    atomic_state(state_path, state)
    try:
        for host, unit in units.items():
            status = unit_status(hosts[host], unit)
            if status.get("LoadState") == "not-found":
                remaining = int(deadline_epoch - time.time())
                if remaining <= 0:
                    raise CampaignFailure("no campaign time remains")
                systemd_start(hosts[host], unit, commands[host], remaining, token)
            elif status.get("ActiveState") not in {"active", "activating"}:
                if status.get("Result") != "success" or status.get("ExecMainStatus") != "0":
                    raise CampaignFailure(f"existing deterministic service failed on {host}: {status}")
    except BaseException:
        stop_units(hosts, units)
        raise
    wait_units(hosts, units, deadline_epoch, state_path, state)
    return units


def verify_host_result(
    path: Path, plan: list[dict[str, Any]], universe: dict[str, dict[str, Any]], host: str,
) -> list[dict[str, Any]]:
    rows = list(common.read_jsonl(path))
    common.validate_requests_against_universe(rows, universe)
    expected = {row["request_id"] for row in plan}
    observed = {row["request_id"] for row in rows}
    if observed != expected:
        raise CampaignFailure(
            f"non-terminal result from {host}: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    if any(row.get("host") != host or row.get("assigned_host") != host for row in rows):
        raise CampaignFailure(f"host provenance mismatch in {host} shard")
    return common.merge_request_results(rows)


def recover_master_rounds(
    output: Path, universe: dict[str, dict[str, Any]], host_weights: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
    root = output / "campaign_rounds"
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    expected_names = [f"round_{index:03d}" for index in range(1, len(directories) + 1)]
    if [path.name for path in directories] != expected_names:
        raise CampaignFailure("master campaign round directories are not contiguous")
    prior: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    incomplete: int | None = None
    for index, directory in enumerate(directories, 1):
        plan_path = directory / "plan.jsonl"
        if not plan_path.is_file():
            if index != len(directories):
                raise CampaignFailure("only the final round may lack an immutable plan")
            incomplete = index
            continue
        plan = list(common.read_jsonl(plan_path))
        common.validate_requests_against_universe(plan, universe)
        if any(row.get("round") != index for row in plan):
            raise CampaignFailure("round number differs inside immutable plan")
        expected_shards = common.shard_requests_weighted(plan, host_weights)
        shard_root = directory / "shards"
        shard_root.mkdir(exist_ok=True)
        for host, expected in expected_shards.items():
            shard_path = shard_root / f"{host}.requests.jsonl"
            if not shard_path.is_file():
                common.atomic_write_jsonl(shard_path, expected)
            if list(common.read_jsonl(shard_path)) != expected:
                raise CampaignFailure(f"immutable {host} shard differs in round {index}")
        manifest_path = directory / "plan_manifest.json"
        expected_manifest = {
            "schema_version": 1, "round": index, "requests": len(plan),
            "plan_sha256": common.sha256_file(plan_path), "host_weights": host_weights,
            "shards": {
                host: {
                    "requests": len(expected_shards[host]),
                    "sha256": common.sha256_file(shard_root / f"{host}.requests.jsonl"),
                } for host in sorted(host_weights)
            },
        }
        if manifest_path.is_file():
            observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in expected_manifest.items():
                if observed_manifest.get(key) != value:
                    raise CampaignFailure(f"round {index} plan manifest differs at {key}")
        else:
            common.atomic_write_json(manifest_path, expected_manifest)
        prior.extend(plan)
        merged_path = directory / "merged_results.jsonl"
        summary_path = directory / "summary.json"
        if merged_path.is_file() and summary_path.is_file():
            rows = list(common.read_jsonl(merged_path))
            common.validate_requests_against_universe(rows, universe)
            if {row["request_id"] for row in rows} != {row["request_id"] for row in plan}:
                raise CampaignFailure(f"completed round {index} result set differs from its plan")
            new.extend(rows)
        else:
            if index != len(directories):
                raise CampaignFailure("only the final round may be incomplete")
            incomplete = index
    recorded = list(common.read_jsonl(output / "campaign_plan.jsonl"))
    if len(recorded) > len(prior) or recorded != prior[:len(recorded)]:
        raise CampaignFailure("master campaign plan differs from immutable round concatenation")
    if len(recorded) < len(prior):
        common.atomic_write_jsonl(output / "campaign_plan.jsonl", prior)
    return prior, common.merge_request_results(new), incomplete


def campaign(args: argparse.Namespace) -> None:
    if socket.gethostname() != COORDINATOR_HOST:
        raise CampaignFailure("two-host coordinator must run on gpu-04")
    config = load_host_config(args.host_config)
    dataset_mode = config.get("dataset_mode", "factorial")
    target = config.get("outcome_target", outcome_presence.TARGET_FIVE_CLASS)
    science = outcome_presence if dataset_mode == "outcome_presence" else common
    token = config["run_token"]
    if args.run_token != token:
        raise CampaignFailure("run token differs from reviewed host configuration")
    output = args.output_dir.resolve()
    if output != Path(config["output_dir"]).resolve():
        raise CampaignFailure("output directory differs from reviewed host configuration")
    state_path = args.state.resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_token") != token:
            raise CampaignFailure("coordinator state belongs to another run token")
    else:
        started = time.time()
        state = {
            "schema_version": 1, "run_token": token, "started_at_epoch": started,
            "started_at": utc_now(), "deadline_epoch": started + config["limits"]["wall_limit_seconds"],
            "services": {}, "status": "starting",
        }
        atomic_state(state_path, state)
    deadline_epoch = float(state["deadline_epoch"])
    if time.time() >= deadline_epoch:
        raise CampaignFailure("persisted reviewed campaign deadline has expired")

    hosts = config["hosts"]
    for host, spec in hosts.items():
        spec["host"] = host
        spec["dataset_mode"] = dataset_mode
        spec["outcome_target"] = target
        spec["limits"] = config["limits"]
    generation_plan = output / "generation_plan.jsonl"
    universe = common.reviewed_universe_index(common.read_jsonl(generation_plan))
    checkpoint_digest = collector.combined_checkpoint_hash(
        collector.checkpoint_hashes(Path(hosts[COORDINATOR_HOST]["checkpoint"]))
    )
    if any(row["checkpoint_sha256"] != checkpoint_digest for row in universe.values()):
        raise CampaignFailure("reviewed request universe checkpoint hash differs")
    dataset, dataset_by_key = collector.load_dataset(Path(hosts[COORDINATOR_HOST]["dataset"]))
    existing, _ = common.deduplicate_records(common.read_jsonl(output / "raw_existing_rollouts.jsonl"))
    host_weights = {host: len(spec["gpu_ids"]) for host, spec in hosts.items()}

    # Stage only deterministic orchestration inputs. Models/data/source were
    # staged and hash-reviewed before approval.
    for spec in hosts.values():
        copy_reviewed_file(spec, generation_plan, f"{spec['work_dir']}/generation_plan.jsonl")

    qualification_commands = {
        host: collector_args(
            spec, mode="qualify", wall_seconds=int(deadline_epoch - time.time())
        ) for host, spec in hosts.items()
    }
    start_or_rejoin_services(
        hosts=hosts, stage="qualify", commands=qualification_commands,
        deadline_epoch=deadline_epoch, token=token, state_path=state_path, state=state,
    )
    qualification_dir = output / "host_qualification"
    for host, spec in hosts.items():
        copy_from_host(spec, f"{spec['work_dir']}/qualification.json", qualification_dir / f"{host}.json")
        qualification = json.loads((qualification_dir / f"{host}.json").read_text())
        if qualification.get("status") != "qualified" or qualification.get("gpu_ids") != spec["gpu_ids"]:
            raise CampaignFailure(f"invalid qualification receipt from {host}")

    prior_requests, new, incomplete = recover_master_rounds(output, universe, host_weights)
    require_exact = config["limits"].get("require_exact_generations") is True
    stop_status: str | None = None
    for round_number in range(1, config["limits"]["max_rounds"] + 1):
        if time.time() >= deadline_epoch:
            stop_status = "wall_limit_exhausted"
            break
        merged, _ = common.deduplicate_records([*existing, *new])
        inventory = science.inventory_rows(dataset, merged)
        complete = (
            len(science.complete_problem_ids(inventory, target))
            if dataset_mode == "outcome_presence"
            else len(science.complete_problem_ids(inventory))
        )
        if incomplete is None and round_number <= len(list((output / "campaign_rounds").glob("round_*"))):
            continue
        if (
            incomplete is None and complete >= config["limits"]["target_complete_problems"]
            and not require_exact
        ):
            break
        if incomplete is not None and round_number != incomplete:
            if round_number < incomplete:
                continue
            raise CampaignFailure("resume round ordering is ambiguous")
        round_dir = output / "campaign_rounds" / f"round_{round_number:03d}"
        if not round_dir.exists():
            round_dir.mkdir(parents=True, exist_ok=False)
        plan_path = round_dir / "plan.jsonl"
        if plan_path.is_file():
            plan = list(common.read_jsonl(plan_path))
        else:
            remaining = config["limits"]["max_new_generations"] - len(prior_requests)
            if remaining <= 0:
                stop_status = "generation_limit_exhausted"
                break
            budget = min(
                config["limits"]["pilot_new_generations"] if round_number == 1
                else config["limits"]["round_new_generation_limit"], remaining,
            )
            plan_kwargs = dict(
                dataset_by_key=dataset_by_key, inventory=inventory,
                prior_requests=prior_requests, master_seed=config["limits"]["master_seed"],
                checkpoint_hash=checkpoint_digest, sampling=science.SAMPLING,
                request_budget=budget,
                samples_per_problem=config["limits"]["samples_per_problem_per_round"],
                round_number=round_number,
            )
            if dataset_mode == "outcome_presence":
                plan_kwargs["target"] = target
                plan_kwargs["include_complete"] = require_exact
            plan = science.build_round_plan(**plan_kwargs)
            if not plan:
                stop_status = "reviewed_universe_exhausted"
                break
            common.validate_requests_against_universe(plan, universe)
            common.atomic_write_jsonl(plan_path, plan)
            shards = common.shard_requests_weighted(plan, host_weights)
            shard_dir = round_dir / "shards"
            shard_dir.mkdir(exist_ok=False)
            for host in sorted(hosts):
                common.atomic_write_jsonl(shard_dir / f"{host}.requests.jsonl", shards[host])
            common.atomic_write_json(round_dir / "plan_manifest.json", {
                "schema_version": 1, "round": round_number, "requests": len(plan),
                "plan_sha256": common.sha256_file(plan_path), "host_weights": host_weights,
                "shards": {
                    host: {
                        "requests": len(shards[host]),
                        "sha256": common.sha256_file(shard_dir / f"{host}.requests.jsonl"),
                    } for host in sorted(hosts)
                },
            })
            prior_requests.extend(plan)
            common.atomic_write_jsonl(output / "campaign_plan.jsonl", prior_requests)
        common.validate_requests_against_universe(plan, universe)
        shards = common.shard_requests_weighted(plan, host_weights)
        for host, spec in hosts.items():
            host_plan = f"{spec['work_dir']}/campaign_rounds/round_{round_number:03d}/requests.jsonl"
            copy_reviewed_file(spec, round_dir / "shards" / f"{host}.requests.jsonl", host_plan)
        remaining_wall = int(deadline_epoch - time.time())
        commands = {
            host: collector_args(
                spec, mode="run-plan", wall_seconds=remaining_wall,
                request_plan=f"{spec['work_dir']}/campaign_rounds/round_{round_number:03d}/requests.jsonl",
                round_number=round_number,
            ) for host, spec in hosts.items()
        }
        stage = f"r{round_number:03d}"
        units = start_or_rejoin_services(
            hosts=hosts, stage=stage, commands=commands, deadline_epoch=deadline_epoch,
            token=token, state_path=state_path, state=state,
        )
        receipt_root = round_dir / "supervisor_receipts"
        host_result_rows = []
        host_results_dir = round_dir / "host_results"
        for host, spec in hosts.items():
            destination = host_results_dir / f"{host}.jsonl"
            copy_from_host(spec, f"{spec['work_dir']}/raw_new_rollouts.jsonl", destination)
            rows = verify_host_result(destination, shards[host], universe, host)
            for row in rows:
                row["checkpoint_path"] = collector.package_relative(
                    output, Path(hosts[COORDINATOR_HOST]["checkpoint"])
                )
            common.atomic_write_jsonl(destination, rows)
            journals = copy_worker_journals(
                spec, output / "workers" / host / f"round_{round_number:03d}", round_number
            )
            receipt_root.mkdir(parents=True, exist_ok=True)
            common.atomic_write_json(receipt_root / f"{host}.json", {
                "schema_version": 1, "host": host, "round": round_number,
                "service_unit": units[host],
                "terminal_status": unit_status(spec, units[host]),
                "request_count": len(shards[host]), "worker_journals": journals,
                "verified_at": utc_now(),
            })
            host_result_rows.extend(rows)
        round_rows = common.merge_request_results(host_result_rows)
        if {row["request_id"] for row in round_rows} != {row["request_id"] for row in plan}:
            raise CampaignFailure("merged two-host result differs from immutable round plan")
        common.atomic_write_jsonl(round_dir / "merged_results.jsonl", round_rows)
        new = common.merge_request_results([*new, *round_rows])
        common.atomic_write_jsonl(output / "raw_new_rollouts.jsonl", new)
        summary = collector.update_outputs(
            output, dataset, existing, new, config["limits"]["target_complete_problems"],
            status=f"round_{round_number}_complete", dataset_mode=dataset_mode,
            outcome_target=target,
        )
        common.atomic_write_json(round_dir / "summary.json", summary)
        common.append_jsonl(output / "progress.jsonl", {
            "timestamp": utc_now(), "round": round_number, "planned": len(plan),
            "new_generations": len(new), "complete_problems": summary[
                ("core_triplet_complete_problems" if target == outcome_presence.TARGET_CORE_TRIPLET
                 else "complete_five_class_problems") if dataset_mode == "outcome_presence"
                else "complete_four_cell_problems"
            ],
            "hosts": {host: len(shards[host]) for host in sorted(hosts)},
            "missing": summary["problems_missing_each_cell"],
        })
        collector.log(
        f"REVIEWED_ROUND_COMPLETE round={round_number} generations={len(new)} "
            f"complete={complete}", output / "collection.log",
        )
        incomplete = None
    else:
        stop_status = "round_limit_exhausted"

    if config.get("campaign_kind") == "feasibility_pilot":
        summary = collector.update_outputs(
            output, dataset, existing, new, 200, status="pilot_complete",
            dataset_mode=dataset_mode, outcome_target=target,
        )
        inventory = science.inventory_rows(dataset, [*existing, *new])
        collector.write_retrospective_pilot(
            output, inventory, len(existing) + len(new), dataset_mode=dataset_mode
        )
        state["status"] = "pilot_complete"
        state["updated_at"] = utc_now()
        atomic_state(state_path, state)
        common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))
        collector.log(
            f"TWO_HOST_FEASIBILITY_PILOT_COMPLETE new_generations={len(new)}",
            output / "collection.log",
        )
        return

    if dataset_mode == "outcome_presence":
        selected = (
            science.select_core_dataset([*existing, *new], 200)
            if target == outcome_presence.TARGET_CORE_TRIPLET
            else science.select_dataset([*existing, *new], 200)
        )
        required_cells = outcome_presence.target_cells(target)
    else:
        selected = common.select_factorial_dataset([*existing, *new], 200)
        required_cells = common.CELLS
    target_met = len(selected) >= 200 * len(required_cells)
    exact_generation_target_met = (
        not require_exact or len(new) == config["limits"]["max_new_generations"]
    )
    summary = collector.update_outputs(
        output, dataset, existing, new, 200,
        status=(
            "succeeded" if target_met and exact_generation_target_met
            else stop_status or (
                "exact_generation_target_not_reached"
                if not exact_generation_target_met else "target_not_reached"
            )
        ),
        dataset_mode=dataset_mode, outcome_target=target,
    )
    summary["require_exact_generations"] = require_exact
    summary["exact_generation_target"] = (
        config["limits"]["max_new_generations"] if require_exact else None
    )
    summary["exact_generation_target_met"] = exact_generation_target_met
    common.atomic_write_json(output / "summary.json", summary)
    state["status"] = summary["status"]
    state["updated_at"] = utc_now()
    atomic_state(state_path, state)
    if not target_met or not exact_generation_target_met:
        complete_key = (
            "core_triplet_complete_problems"
            if target == outcome_presence.TARGET_CORE_TRIPLET
            else "complete_five_class_problems"
        ) if dataset_mode == "outcome_presence" else "complete_four_cell_problems"
        raise CampaignFailure(
            f"campaign ended without target: status={summary['status']} "
            f"complete={summary.get(complete_key)} "
            f"new_generations={len(new)}/{config['limits']['max_new_generations']}"
        )
    collector.log(
        f"REVIEWED_CAMPAIGN_SUCCEEDED selected_problems={summary['selected_problems']}",
        output / "collection.log",
    )
    collector.validate_final(output, dataset_mode=dataset_mode, outcome_target=target)


def capture_failure_diagnostics(args: argparse.Namespace, error: BaseException) -> None:
    """Best-effort retention of append-only journals after a failed campaign."""
    try:
        config = load_host_config(args.host_config)
        output = args.output_dir.resolve()
        destination_root = output / "failure_diagnostics"
        generated_rows: list[dict[str, Any]] = []
        for host, spec in config["hosts"].items():
            spec["host"] = host
            destination = destination_root / host / "workers"
            destination.mkdir(parents=True, exist_ok=True)
            if spec["local"]:
                source_root = Path(spec["work_dir"]) / "workers"
                paths = sorted(source_root.glob("*.jsonl")) if source_root.is_dir() else []
                for source in paths:
                    shutil.copy2(source, destination / source.name)
            else:
                result = subprocess.run(
                    [
                        "rsync", "--archive", "--checksum", "--include=*.jsonl", "--exclude=*",
                        f"{spec['ssh_target']}:{spec['work_dir']}/workers/", str(destination) + "/",
                    ], capture_output=True, text=True, timeout=180,
                )
                if result.returncode != 0:
                    common.atomic_write_json(destination.parent / "copy_error.json", {
                        "host": host, "error": "worker journal retrieval failed",
                    })
            for path in sorted(destination.glob("*.generated.jsonl")):
                generated_rows.extend(common.read_jsonl(path))
        by_request: dict[str, dict[str, Any]] = {}
        for row in generated_rows:
            request_id = row.get("request_id")
            if isinstance(request_id, str):
                by_request.setdefault(request_id, row)
        common.atomic_write_jsonl(
            output / "partial_generated_rollouts.jsonl",
            [by_request[key] for key in sorted(by_request)],
        )
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        summary.update({
            "status": "failed", "failure_type": type(error).__name__,
            "partial_generated_records_retained": len(by_request), "updated_at": utc_now(),
        })
        common.atomic_write_json(summary_path, summary)
        common.atomic_write_json(output / "artifact_manifest.json", common.file_manifest(output))
    except Exception as diagnostic_error:
        print(
            f"FAIL_CLOSED_DIAGNOSTIC_ERROR: {type(diagnostic_error).__name__}",
            file=sys.stderr, flush=True,
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--host-config", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--state", type=Path, required=True)
    value.add_argument("--run-token", required=True)
    value.add_argument("--resume", action="store_true", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    try:
        campaign(args)
    except CampaignFailure as error:
        capture_failure_diagnostics(args, error)
        print(f"FAIL_CLOSED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(TERMINAL_FAILURE_EXIT)
    except Exception as error:
        capture_failure_diagnostics(args, error)
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(TERMINAL_FAILURE_EXIT)


if __name__ == "__main__":
    main()
