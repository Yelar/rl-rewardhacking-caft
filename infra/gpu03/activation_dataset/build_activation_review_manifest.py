#!/usr/bin/env python3
"""Build a deterministic, host-bound review manifest for activation extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import socket
from pathlib import Path


SCHEMA_VERSION = 3
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_HOSTS = {"gpu-03"}
EXPECTED_SNAPSHOT_FILES = {
    ".gitattributes", "LICENSE", "README.md", "config.json",
    "generation_config.json", "merges.txt",
    "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
}
COMPLETED_SHARD_IDS = [0, 1, 2, 3]
WORKER_SHARD_IDS = [4, 5, 6, 7]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def reviewed_output_roots(username: str) -> tuple[Path, Path]:
    return (
        (Path.home() / "codex_runs").resolve(),
        Path("/scratch") / username / "codex_runs",
    )


def execution_parameters(args: argparse.Namespace) -> dict:
    return {
        "host": args.host,
        "run_token": args.run_token,
        "source_git_commit": args.source_git_commit,
        "resume_shards_dir": str(args.resume_shards_dir.resolve()),
        "completed_shard_ids": list(args.completed_shard_ids),
        "worker_shard_ids": list(args.worker_shard_ids),
        "service_unit": args.service_unit,
        "supervisor_status": str(args.supervisor_status.resolve()),
        "supervisor_receipt_writer": str(args.supervisor_receipt_writer.resolve()),
        "python": str(absolute_without_symlink_resolution(args.python)),
        "runner": str(args.runner.resolve()),
        "frozen_dataset": str(args.frozen_dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "hf_cache": str(args.hf_cache.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "scratch_root": str(args.scratch_root.resolve()),
        "review_manifest": str(args.manifest.resolve()),
        "launch_receipt": str(args.launch_receipt.resolve()),
        "launch_result": str(args.launch_result.resolve()),
        "launch_log": str(args.launch_log.resolve()),
        "direct_entrypoint": str(args.direct_entrypoint.resolve()),
        "gpu_ids": list(args.gpu_ids),
        "batch_size": args.batch_size,
        "cpus_per_worker": args.cpus_per_worker,
        "max_sequence_length": args.max_sequence_length,
        "min_start_available_memory_kib": args.min_start_available_memory_kib,
        "min_runtime_available_memory_kib": args.min_runtime_available_memory_kib,
        "worker_start_stagger_seconds": args.worker_start_stagger_seconds,
        "expected_gpu_name": args.expected_gpu_name,
        "min_gpu_total_mib": args.min_gpu_total_mib,
        "min_gpu_free_mib": args.min_gpu_free_mib,
        "max_gpu_used_mib": args.max_gpu_used_mib,
        "gpu_quiescence_seconds": args.gpu_quiescence_seconds,
        "gpu_poll_seconds": args.gpu_poll_seconds,
        "max_combined_worker_rss_kib": args.max_combined_worker_rss_kib,
        "max_runtime_seconds": args.max_runtime_seconds,
        "wrapper_wall_limit_seconds": args.wrapper_wall_limit_seconds,
        "min_output_free_bytes": args.min_output_free_bytes,
        "min_scratch_free_bytes": args.min_scratch_free_bytes,
        "min_free_inodes": args.min_free_inodes,
        "fp32_audit_records_per_shard": args.fp32_audit_records_per_shard,
    }


def activation_contract() -> dict:
    return {
        "site": "decoder layer forward output before final norm",
        "delta": "h60 - h0",
        "layers": 36,
        "hidden_size": 2560,
        "selected_token_slots": 40,
        "temporary_activation_dtype": "bfloat16",
        "storage_dtype": "bfloat16",
        "difference_accumulation_dtype": "float32",
        "in_process_cuda_release_contract": {
            "allowed_allocated_bytes": 33554432,
            "allowed_reserved_bytes": 33554432,
            "reason": "deterministic cuBLAS :4096:8 workspace only",
            "post_process_requirement": "zero GPU process and idle-memory threshold",
        },
        "reward_hack_category_encoding": {
            "Attempted Reward Hack": 0,
            "Correct": 1,
            "Correct; Attempted Reward Hack": 2,
            "Incorrect": 3,
            "Reward Hack": 4,
        },
        "generation": False,
        "gradients": False,
    }


def critical_paths(args: argparse.Namespace) -> list[Path]:
    script_dir = args.runner.resolve().parent
    python_path = absolute_without_symlink_resolution(args.python)
    pyvenv_config = python_path.parent.parent / "pyvenv.cfg"
    snapshot = (
        args.hf_cache.resolve()
        / "hub/models--Qwen--Qwen3-4B/snapshots"
        / MODEL_REVISION
    )
    snapshot_entries = sorted(snapshot.iterdir())
    if {path.name for path in snapshot_entries} != EXPECTED_SNAPSHOT_FILES:
        raise FileNotFoundError(f"Pinned model snapshot file set is not the reviewed whitelist: {snapshot}")
    if not all(path.is_file() for path in snapshot_entries):
        raise FileNotFoundError("Pinned model snapshot contains a non-file entry")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected_shards = {name for name in EXPECTED_SNAPSHOT_FILES if name.endswith(".safetensors")}
    if set(index.get("weight_map", {}).values()) != expected_shards:
        raise ValueError("Pinned model weight index does not map exactly to the reviewed shards")
    resume_entries = []
    for shard_id in COMPLETED_SHARD_IDS:
        prefix = args.resume_shards_dir.resolve() / f"delta_shard_{shard_id:02d}"
        resume_entries.extend([
            prefix.with_suffix(".safetensors"),
            prefix.with_suffix(".index.jsonl"),
            prefix.with_suffix(".report.json"),
        ])
    return [
        python_path,
        pyvenv_config,
        args.runner.resolve(),
        script_dir / "build_activation_review_manifest.py",
        script_dir / "freeze_rollout_dataset.py",
        script_dir / "structural_positions.py",
        script_dir / "test_activation_dataset.py",
        script_dir / "README.md",
        script_dir / "launch_activation_job.py",
        args.supervisor_receipt_writer.resolve(),
        args.direct_entrypoint.resolve(),
        args.frozen_dataset.resolve(),
        args.frozen_dataset.resolve().with_name("split_manifest.json"),
        args.frozen_dataset.resolve().with_name("source_matched_dataset.jsonl"),
        args.frozen_dataset.resolve().with_name("FROZEN_SHA256"),
        args.checkpoint.resolve() / "adapter_config.json",
        args.checkpoint.resolve() / "adapter_model.safetensors",
        *resume_entries,
        *snapshot_entries,
    ]


def build_manifest(args: argparse.Namespace) -> dict:
    if args.host not in EXPECTED_HOSTS:
        raise ValueError(f"Unreviewed host: {args.host}")
    if socket.gethostname() != args.host:
        raise RuntimeError("Host-specific review manifest must be built on the selected host")
    if len(args.gpu_ids) != 8 or sorted(args.gpu_ids) != list(range(8)):
        raise ValueError("The reviewed run must use physical GPUs 0 through 7 exactly once")
    if not re.fullmatch(r"codex-activation-ckpt60-[a-z0-9-]{1,64}", args.run_token):
        raise ValueError("Run token is not a safe checkpoint-60 activation token")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_git_commit):
        raise ValueError("Source Git commit must be an exact lowercase SHA-1")
    if args.completed_shard_ids != COMPLETED_SHARD_IDS:
        raise ValueError("Recovery must reuse exactly reviewed shards 0 through 3")
    if args.worker_shard_ids != WORKER_SHARD_IDS:
        raise ValueError("Recovery must compute exactly missing shards 4 through 7")
    if args.service_unit != args.run_token:
        raise ValueError("Systemd service unit must equal the exact run token")
    expected_limits = {
        "batch_size": 2,
        "cpus_per_worker": 4,
        "max_sequence_length": 3072,
        "min_start_available_memory_kib": 268435456,
        "min_runtime_available_memory_kib": 201326592,
        "worker_start_stagger_seconds": 10,
        "expected_gpu_name": "NVIDIA RTX 5000 Ada Generation",
        "min_gpu_total_mib": 32000,
        "min_gpu_free_mib": 32000,
        "max_gpu_used_mib": 64,
        "gpu_quiescence_seconds": 60,
        "gpu_poll_seconds": 5,
        "max_combined_worker_rss_kib": 402653184,
        "max_runtime_seconds": 12600,
        "wrapper_wall_limit_seconds": 16200,
        "min_output_free_bytes": 8589934592,
        "min_scratch_free_bytes": 8589934592,
        "min_free_inodes": 10000,
        "fp32_audit_records_per_shard": 1,
    }
    changed_limits = {
        name: {"observed": getattr(args, name), "required": expected}
        for name, expected in expected_limits.items()
        if getattr(args, name) != expected
    }
    if changed_limits:
        raise ValueError(f"Execution limits differ from the reviewed design: {changed_limits}")
    username = pwd.getpwuid(os.getuid()).pw_name
    output_dir = args.output_dir.resolve()
    scratch_root = args.scratch_root.resolve()
    expected_scratch = (
        Path("/scratch") / username / "codex_activation_extraction" / args.run_token
    )
    if scratch_root != expected_scratch:
        raise ValueError(f"Scratch root must be the exact run-token path: {expected_scratch}")
    if not any(
        output_dir.is_relative_to(root)
        for root in reviewed_output_roots(username)
    ):
        raise ValueError("Output must be under a reviewed user-scoped codex_runs root")
    if output_dir.exists() or scratch_root.exists():
        raise FileExistsError("Output and scratch paths must not exist before review")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError("Durable output parent must be staged before manifest creation")
    if args.launch_log.resolve().is_relative_to(output_dir):
        raise ValueError("Launch log must be outside the atomically created output directory")
    for path in (
        args.launch_receipt, args.launch_result, args.launch_log, args.supervisor_status
    ):
        resolved = path.resolve()
        if not resolved.parent.is_dir() or resolved.exists():
            raise FileExistsError(f"Launch path is not ready and unused: {resolved}")
    python_path = absolute_without_symlink_resolution(args.python)
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError("Reviewed Python interpreter is absent or not executable")
    if not (python_path.parent.parent / "pyvenv.cfg").is_file():
        raise FileNotFoundError("Reviewed Python must be an explicit virtualenv interpreter")
    paths = critical_paths(args)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest-critical inputs are missing: {missing}")
    parameters = execution_parameters(args)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "checkpoint-60 minus base post-block residual activation extraction",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_step": 60,
        "source_control": {
            "git_commit": args.source_git_commit,
            "working_tree_representation": (
                "manifest-bound hashes of every staged critical source file"
            ),
        },
        "dataset_contract": {
            "records": 400,
            "pairs": 200,
            "unique_problems": 200,
            "split_unit": "problem_id",
            "split_problem_counts": {
                "direction_fit": 120,
                "configuration_validation": 40,
                "test": 40,
            },
        },
        "activation_contract": activation_contract(),
        "recovery_contract": {
            "source_shards_directory": str(args.resume_shards_dir.resolve()),
            "completed_shard_ids": COMPLETED_SHARD_IDS,
            "worker_shard_ids": WORKER_SHARD_IDS,
            "completed_records": 200,
            "remaining_records": 200,
            "final_records": 400,
            "policy": "copy only independently hash-bound complete shards, recompute missing shards, then verify all records together",
        },
        "direct_host_contract": {
            "selected_host": args.host,
            "physical_gpu_ids": list(range(8)),
            "gpu_count": 8,
            "gpu_model": "NVIDIA RTX 5000 Ada Generation",
            "scheduler_used": False,
            "supervisor": "transient systemd user service with ExecStopPost receipt",
            "service_unit": args.service_unit,
            "os_enforced_exclusivity": False,
            "cooperative_lock": f"/run/lock/codex-checkpoint60-activations-{args.host}.lock",
            "quiescence_seconds": 60,
            "runtime_gpu_poll_seconds": 5,
            "foreign_gpu_process_policy": "terminate reviewed workers and fail closed",
            "worker_cpu_cores_total": 16,
            "worker_cpu_affinity": "four disjoint four-core sets",
            "worker_priority": {"nice": 10, "ionice_class": 2, "ionice_level": 7},
            "max_combined_worker_rss_kib": 402653184,
            "wrapper_wall_limit_seconds": 16200,
            "termination_grace_seconds": 120,
        },
        "execution": parameters,
        "critical_file_sha256": {str(path): sha256_file(path) for path in paths},
        "approval_format": (
            f"I_APPROVE_CHECKPOINT60_ACTIVATIONS:<manifest-sha256>:{args.run_token}:{args.host}"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, choices=sorted(EXPECTED_HOSTS))
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--resume-shards-dir", type=Path, required=True)
    parser.add_argument("--completed-shard-ids", type=lambda value: [int(item) for item in value.split(",")], required=True)
    parser.add_argument("--worker-shard-ids", type=lambda value: [int(item) for item in value.split(",")], required=True)
    parser.add_argument("--service-unit", required=True)
    parser.add_argument("--supervisor-status", type=Path, required=True)
    parser.add_argument("--supervisor-receipt-writer", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--frozen-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--launch-result", type=Path, required=True)
    parser.add_argument("--launch-log", type=Path, required=True)
    parser.add_argument("--direct-entrypoint", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=lambda value: [int(item) for item in value.split(",")], default=list(range(8)))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpus-per-worker", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--min-start-available-memory-kib", type=int, default=268435456)
    parser.add_argument("--min-runtime-available-memory-kib", type=int, default=201326592)
    parser.add_argument("--worker-start-stagger-seconds", type=int, default=10)
    parser.add_argument("--expected-gpu-name", default="NVIDIA RTX 5000 Ada Generation")
    parser.add_argument("--min-gpu-total-mib", type=int, default=32000)
    parser.add_argument("--min-gpu-free-mib", type=int, default=32000)
    parser.add_argument("--max-gpu-used-mib", type=int, default=64)
    parser.add_argument("--gpu-quiescence-seconds", type=int, default=60)
    parser.add_argument("--gpu-poll-seconds", type=int, default=5)
    parser.add_argument("--max-combined-worker-rss-kib", type=int, default=402653184)
    parser.add_argument("--max-runtime-seconds", type=int, default=12600)
    parser.add_argument("--wrapper-wall-limit-seconds", type=int, default=16200)
    parser.add_argument("--min-output-free-bytes", type=int, default=8589934592)
    parser.add_argument("--min-scratch-free-bytes", type=int, default=8589934592)
    parser.add_argument("--min-free-inodes", type=int, default=10000)
    parser.add_argument("--fp32-audit-records-per-shard", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    digest = sha256_file(args.manifest)
    print(json.dumps({
        "manifest": str(args.manifest.resolve()),
        "sha256": digest,
        "exact_approval": (
            f"I_APPROVE_CHECKPOINT60_ACTIVATIONS:{digest}:{args.run_token}:{args.host}"
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
