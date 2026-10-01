#!/usr/bin/env python3
"""Prepare exact token sequences and extract M60-M0 post-block residual deltas."""

from __future__ import annotations

import argparse
import atexit
import ctypes
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import pwd
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from freeze_rollout_dataset import (
    HARMFUL_MODIFICATIONS,
    REQUIRED_FIELDS,
    SPLITS,
    VALID_RH_CATEGORIES,
    canonical_id,
    split_distributions,
    validate_source,
)
from structural_positions import MAX_SELECTED_TOKENS, WINDOW_OFFSETS, prepare_sequence


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_LAYERS = 36
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_MODEL_CONFIG_SHA256 = "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
EXPECTED_ADAPTER_HASHES = {
    "adapter_config.json": "8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae",
    "adapter_model.safetensors": "2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0",
}
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0+cu128",
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "safetensors": "0.6.2",
    "vllm": "0.11.0",
}
EXPECTED_FROZEN_DATASET_SHA256 = "0d2fbc67deb9b0eebe33c8eaac7e3cb705f770a2605507b72475a636b1cc8583"
EXPECTED_SOURCE_DATASET_SHA256 = "a6f1f6f65a68ada3c68602e7d4df55e1011ca063f0338ad084741c038048c170"
SPLIT_ORDER = ("direction_fit", "configuration_validation", "test")
EXPECTED_GPU_COUNT = 8
EXPECTED_GPU_NAME = "NVIDIA RTX 5000 Ada Generation"
MIN_GPU_TOTAL_MIB = 32_000
MIN_GPU_FREE_MIB = 32_000
MAX_GPU_USED_MIB = 64
EXPECTED_SNAPSHOT_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
DIRECT_HOST = "gpu-03"
GPU_QUIESCENCE_SECONDS = 60
GPU_POLL_SECONDS = 5
MAX_COMBINED_WORKER_RSS_KIB = 384 * 1024**2
WRAPPER_WALL_LIMIT_SECONDS = 16_200
CUBLAS_DETERMINISTIC_WORKSPACE_BYTES = 32 * 1024**2
COMPLETED_SHARD_IDS = [0, 1, 2, 3]
WORKER_SHARD_IDS = [4, 5, 6, 7]
MIN_OUTPUT_FREE_BYTES = 8 * 1024**3
MIN_SCRATCH_FREE_BYTES = 8 * 1024**3
MIN_FREE_INODES = 10_000
RH_CATEGORY_TO_ID = {
    category: index for index, category in enumerate(sorted(VALID_RH_CATEGORIES))
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def capture_provenance(
    output_dir: Path,
    args: argparse.Namespace,
    approval_receipt: dict,
) -> dict:
    source_paths = sorted({
        Path(__file__).resolve(),
        Path(__file__).with_name("structural_positions.py").resolve(),
        Path(__file__).with_name("freeze_rollout_dataset.py").resolve(),
        Path(__file__).with_name("build_activation_review_manifest.py").resolve(),
        Path(__file__).with_name("launch_activation_job.py").resolve(),
        args.supervisor_receipt_writer.resolve(),
        args.direct_entrypoint.resolve(),
        Path(__file__).with_name("test_activation_dataset.py").resolve(),
        Path(__file__).with_name("README.md").resolve(),
    })
    metadata_paths = sorted({
        args.frozen_dataset.with_name("split_manifest.json").resolve(),
        args.frozen_dataset.with_name("FROZEN_SHA256").resolve(),
        args.checkpoint.resolve() / "adapter_config.json",
    })
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Provenance input is missing: {path}")
    for path in metadata_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Provenance metadata is missing: {path}")
    git_commit = args.source_git_commit
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("Reviewed source Git commit is malformed")
    provenance_dir = output_dir / "provenance"
    source_dir = provenance_dir / "critical_source"
    metadata_dir = provenance_dir / "critical_metadata"
    source_dir.mkdir(parents=True, exist_ok=False)
    metadata_dir.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(source_paths):
        shutil.copy2(path, source_dir / f"{index:02d}_{path.name}")
    for index, path in enumerate(metadata_paths):
        shutil.copy2(path, metadata_dir / f"{index:02d}_{path.name}")
    shutil.copy2(args.review_manifest.resolve(), provenance_dir / "reviewed_manifest.json")
    shutil.copy2(args.launch_receipt.resolve(), provenance_dir / "launch_receipt.json")
    shutil.copy2(args.launch_result.resolve(), provenance_dir / "launch_result.json")
    (provenance_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    return {
        "git_commit": git_commit,
        "policy": (
            "the source Git commit is supplied by and bound into the reviewed manifest; "
            "exact reviewed source files and sanitized metadata are copied by hash; "
            "unrelated working-tree content and binary diffs are deliberately excluded"
        ),
        "critical_source_sha256": {
            str(path): sha256_file(path)
            for path in source_paths
        },
        "critical_metadata_sha256": {
            str(path): sha256_file(path) for path in metadata_paths
        },
        "frozen_dataset_sha256": sha256_file(args.frozen_dataset),
        "source_dataset_sha256": sha256_file(
            args.frozen_dataset.with_name("source_matched_dataset.jsonl")
        ),
        "adapter_model_sha256": sha256_file(
            args.checkpoint.resolve() / "adapter_model.safetensors"
        ),
        "review_manifest_sha256": approval_receipt["manifest_sha256"],
        "launch_receipt_sha256": approval_receipt["launch_receipt_sha256"],
        "launch_result_sha256": approval_receipt["launch_result_sha256"],
    }


def reviewed_execution_parameters(args: argparse.Namespace) -> dict:
    return {
        "host": args.expected_hostname,
        "run_token": args.run_token,
        "source_git_commit": args.source_git_commit,
        "resume_shards_dir": str(args.resume_shards_dir.resolve()),
        "completed_shard_ids": list(args.completed_shard_ids),
        "worker_shard_ids": list(args.worker_shard_ids),
        "service_unit": args.service_unit,
        "supervisor_status": str(args.supervisor_status.resolve()),
        "supervisor_receipt_writer": str(args.supervisor_receipt_writer.resolve()),
        "python": str(absolute_without_symlink_resolution(Path(sys.executable))),
        "runner": str(Path(__file__).resolve()),
        "frozen_dataset": str(args.frozen_dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "hf_cache": str(args.hf_cache.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "scratch_root": str(args.scratch_root.resolve()),
        "review_manifest": str(args.review_manifest.resolve()),
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


def expected_activation_contract() -> dict:
    return {
        "site": "decoder layer forward output before final norm",
        "delta": "h60 - h0",
        "layers": EXPECTED_LAYERS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "selected_token_slots": MAX_SELECTED_TOKENS,
        "temporary_activation_dtype": "bfloat16",
        "storage_dtype": "bfloat16",
        "difference_accumulation_dtype": "float32",
        "in_process_cuda_release_contract": {
            "allowed_allocated_bytes": CUBLAS_DETERMINISTIC_WORKSPACE_BYTES,
            "allowed_reserved_bytes": CUBLAS_DETERMINISTIC_WORKSPACE_BYTES,
            "reason": "deterministic cuBLAS :4096:8 workspace only",
            "post_process_requirement": "zero GPU process and idle-memory threshold",
        },
        "reward_hack_category_encoding": RH_CATEGORY_TO_ID,
        "generation": False,
        "gradients": False,
    }


def verify_review_approval(args: argparse.Namespace) -> dict:
    manifest_path = args.review_manifest.resolve()
    manifest_sha256 = sha256_file(manifest_path)
    expected_approval = (
        f"I_APPROVE_CHECKPOINT60_ACTIVATIONS:{manifest_sha256}:"
        f"{args.run_token}:{args.expected_hostname}"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError("Unsupported activation review manifest schema")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Review manifest does not bind the pinned base model")
    if manifest.get("checkpoint_step") != 60:
        raise ValueError("Review manifest does not bind checkpoint 60")
    expected_dataset_contract = {
        "records": 400,
        "pairs": 200,
        "unique_problems": 200,
        "split_unit": "problem_id",
        "split_problem_counts": {
            "direction_fit": 120,
            "configuration_validation": 40,
            "test": 40,
        },
    }
    if manifest.get("dataset_contract") != expected_dataset_contract:
        raise ValueError("Review manifest dataset contract differs from the implemented contract")
    if manifest.get("activation_contract") != expected_activation_contract():
        raise ValueError("Review manifest activation contract differs from the implemented contract")
    expected_source_control = {
        "git_commit": args.source_git_commit,
        "working_tree_representation": (
            "manifest-bound hashes of every staged critical source file"
        ),
    }
    if manifest.get("source_control") != expected_source_control:
        raise ValueError("Review manifest source-control provenance differs from execution")
    expected_recovery_contract = {
        "source_shards_directory": str(args.resume_shards_dir.resolve()),
        "completed_shard_ids": COMPLETED_SHARD_IDS,
        "worker_shard_ids": WORKER_SHARD_IDS,
        "completed_records": 200,
        "remaining_records": 200,
        "final_records": 400,
        "policy": "copy only independently hash-bound complete shards, recompute missing shards, then verify all records together",
    }
    if manifest.get("recovery_contract") != expected_recovery_contract:
        raise ValueError("Review manifest recovery contract differs from execution")
    expected_direct_contract = {
        "selected_host": args.expected_hostname,
        "physical_gpu_ids": list(range(EXPECTED_GPU_COUNT)),
        "gpu_count": EXPECTED_GPU_COUNT,
        "gpu_model": EXPECTED_GPU_NAME,
        "scheduler_used": False,
        "supervisor": "transient systemd user service with ExecStopPost receipt",
        "service_unit": args.service_unit,
        "os_enforced_exclusivity": False,
        "cooperative_lock": (
            f"/run/lock/codex-checkpoint60-activations-{args.expected_hostname}.lock"
        ),
        "quiescence_seconds": GPU_QUIESCENCE_SECONDS,
        "runtime_gpu_poll_seconds": GPU_POLL_SECONDS,
        "foreign_gpu_process_policy": "terminate reviewed workers and fail closed",
        "worker_cpu_cores_total": 16,
        "worker_cpu_affinity": "four disjoint four-core sets",
        "worker_priority": {"nice": 10, "ionice_class": 2, "ionice_level": 7},
        "max_combined_worker_rss_kib": MAX_COMBINED_WORKER_RSS_KIB,
        "wrapper_wall_limit_seconds": WRAPPER_WALL_LIMIT_SECONDS,
        "termination_grace_seconds": 120,
    }
    if manifest.get("direct_host_contract") != expected_direct_contract:
        raise ValueError("Review manifest direct-host contract differs from the required controls")
    if manifest.get("execution") != reviewed_execution_parameters(args):
        raise ValueError("Effective execution parameters differ from the reviewed manifest")
    script_dir = Path(__file__).resolve().parent
    python_path = absolute_without_symlink_resolution(Path(sys.executable))
    pyvenv_config = python_path.parent.parent / "pyvenv.cfg"
    snapshot = (
        args.hf_cache.resolve()
        / "hub/models--Qwen--Qwen3-4B/snapshots"
        / MODEL_REVISION
    )
    snapshot_files = set(snapshot.iterdir())
    if {path.name for path in snapshot_files} != EXPECTED_SNAPSHOT_FILES:
        raise FileNotFoundError("Pinned model snapshot differs from the reviewed file whitelist")
    if not all(path.is_file() for path in snapshot_files):
        raise FileNotFoundError("Pinned model snapshot contains a non-file entry")
    weight_index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    expected_weight_shards = {
        name for name in EXPECTED_SNAPSHOT_FILES if name.endswith(".safetensors")
    }
    if set(weight_index.get("weight_map", {}).values()) != expected_weight_shards:
        raise ValueError("Pinned model weight index does not map to exactly the reviewed shards")
    required_paths = {
        python_path,
        pyvenv_config,
        Path(__file__).resolve(),
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
        *snapshot_files,
    }
    for shard_id in COMPLETED_SHARD_IDS:
        prefix = args.resume_shards_dir.resolve() / f"delta_shard_{shard_id:02d}"
        required_paths.update({
            prefix.with_suffix(".safetensors"),
            prefix.with_suffix(".index.jsonl"),
            prefix.with_suffix(".report.json"),
        })
    reviewed_hashes = manifest.get("critical_file_sha256")
    if not isinstance(reviewed_hashes, dict):
        raise ValueError("Review manifest has no critical file hash map")
    if set(reviewed_hashes) != {str(path) for path in required_paths}:
        raise ValueError("Review manifest critical file set differs from the required set")
    for path in required_paths:
        if sha256_file(path) != reviewed_hashes[str(path)]:
            raise ValueError(f"Manifest-critical file changed after review: {path}")
    receipt_path = args.launch_receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    launch_id = os.environ.get("CODEX_ACTIVATION_LAUNCH_ID", "")
    if not re.fullmatch(r"[0-9a-f]{32}", launch_id):
        raise PermissionError("Direct launch identity is missing or malformed")
    expected_receipt = {
        "schema_version": 3,
        "manifest_sha256": manifest_sha256,
        "approval_sha256": hashlib.sha256(expected_approval.encode()).hexdigest(),
        "run_token": args.run_token,
        "host": args.expected_hostname,
        "uid": os.getuid(),
        "launch_id": launch_id,
        "state": "approval_consumed_before_direct_launch",
    }
    receipt_stat = receipt_path.stat()
    if (
        receipt != expected_receipt
        or receipt_stat.st_uid != os.getuid()
        or not stat.S_ISREG(receipt_stat.st_mode)
        or stat.S_IMODE(receipt_stat.st_mode) != 0o600
    ):
        raise PermissionError("Atomic one-use direct-launch receipt is missing or invalid")
    launch_path = args.launch_result.resolve()
    deadline = time.monotonic() + 30
    while not launch_path.is_file() and time.monotonic() < deadline:
        time.sleep(1)
    if not launch_path.is_file():
        raise PermissionError("Approved direct-launch result did not appear")
    wrapper_pid_text = os.environ.get("CODEX_ACTIVATION_WRAPPER_PID", "")
    if not wrapper_pid_text.isdigit():
        raise PermissionError("Direct wrapper process identity is missing")
    wrapper_pid = int(wrapper_pid_text)
    launch_result = json.loads(launch_path.read_text(encoding="utf-8"))
    expected_launch_result = {
        "schema_version": 3,
        "manifest_sha256": manifest_sha256,
        "run_token": args.run_token,
        "host": args.expected_hostname,
        "launch_id": launch_id,
        "service_unit": args.service_unit,
        "host_lock_path": (
            f"/run/lock/codex-checkpoint60-activations-{args.expected_hostname}.lock"
        ),
        "state": "systemd_service_request_prepared",
    }
    launch_stat = launch_path.stat()
    if (
        launch_result != expected_launch_result
        or launch_stat.st_uid != os.getuid()
        or not stat.S_ISREG(launch_stat.st_mode)
        or stat.S_IMODE(launch_stat.st_mode) != 0o600
    ):
        raise PermissionError("Process is not bound to the consumed reviewed direct launch")
    try:
        wrapper_stat = Path(f"/proc/{wrapper_pid}").stat()
    except FileNotFoundError as error:
        raise PermissionError("Reviewed direct wrapper is no longer running") from error
    if wrapper_stat.st_uid != os.getuid():
        raise PermissionError("Reviewed direct wrapper owner is incorrect")
    if os.environ.get("CODEX_ACTIVATION_SYSTEMD_UNIT") != args.service_unit:
        raise PermissionError("Reviewed systemd service identity is missing")
    unit_status = subprocess.run(
        [
            "systemctl", "--user", "show", f"{args.service_unit}.service",
            "-p", "MainPID", "-p", "ActiveState", "--value",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.splitlines()
    if len(unit_status) != 2 or int(unit_status[0]) != wrapper_pid or unit_status[1] != "active":
        raise PermissionError("Reviewed wrapper is not the active systemd service main process")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "run_token": args.run_token,
        "approval": "verified",
        "launch_id": launch_id,
        "direct_wrapper_pid": wrapper_pid,
        "service_unit": args.service_unit,
        "launch_receipt": str(receipt_path),
        "launch_receipt_sha256": sha256_file(receipt_path),
        "launch_result": str(launch_path),
        "launch_result_sha256": sha256_file(launch_path),
    }


def verify_inputs(args: argparse.Namespace) -> dict:
    if sha256_file(args.frozen_dataset) != EXPECTED_FROZEN_DATASET_SHA256:
        raise ValueError("Frozen dataset hash does not match the reviewed checkpoint-60 split")
    dataset_dir = args.frozen_dataset.parent
    source_dataset = dataset_dir / "source_matched_dataset.jsonl"
    split_manifest_path = dataset_dir / "split_manifest.json"
    frozen_hash_path = dataset_dir / "FROZEN_SHA256"
    if sha256_file(source_dataset) != EXPECTED_SOURCE_DATASET_SHA256:
        raise ValueError("Source matched dataset hash does not match the reviewed collection")
    expected_hash_line = f"{EXPECTED_FROZEN_DATASET_SHA256}  frozen_dataset.jsonl\n"
    if frozen_hash_path.read_text(encoding="utf-8") != expected_hash_line:
        raise ValueError("FROZEN_SHA256 receipt is malformed")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    expected_split_metadata = {
        "source_path": "source_matched_dataset.jsonl",
        "source_sha256": EXPECTED_SOURCE_DATASET_SHA256,
        "frozen_path": "frozen_dataset.jsonl",
        "frozen_sha256": EXPECTED_FROZEN_DATASET_SHA256,
        "split_seed": 6001,
        "split_unit": "problem_id",
        "records": 400,
        "pairs": 200,
        "unique_problems": 200,
        "split_unique_problem_counts": {
            "direction_fit": 120,
            "configuration_validation": 40,
            "test": 40,
        },
        "split_record_counts": {
            "direction_fit": 240,
            "configuration_validation": 80,
            "test": 80,
        },
    }
    for key, expected in expected_split_metadata.items():
        if split_manifest.get(key) != expected:
            raise ValueError(f"Split manifest mismatch for {key!r}")
    if split_manifest.get("required_fields") != sorted(REQUIRED_FIELDS):
        raise ValueError("Split manifest required-field contract changed")
    adapter_hashes = {}
    for name, expected in EXPECTED_ADAPTER_HASHES.items():
        path = args.checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing adapter file: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Adapter hash mismatch for {name}: {observed}")
        adapter_hashes[name] = observed
    adapter_config = json.loads(
        (args.checkpoint / "adapter_config.json").read_text(encoding="utf-8")
    )
    expected_adapter_metadata = {
        "base_model_name_or_path": MODEL_ID,
        "revision": MODEL_REVISION,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 32,
        "lora_alpha": 32,
    }
    for key, expected in expected_adapter_metadata.items():
        if adapter_config.get(key) != expected:
            raise ValueError(
                f"Adapter metadata {key!r} is {adapter_config.get(key)!r}, expected {expected!r}"
            )
    snapshot = (
        args.hf_cache / "hub" / "models--Qwen--Qwen3-4B" / "snapshots" / MODEL_REVISION
    )
    snapshot_entries = list(snapshot.iterdir())
    snapshot_files = {path.name for path in snapshot_entries}
    if snapshot_files != EXPECTED_SNAPSHOT_FILES:
        raise ValueError("Pinned model snapshot does not have the exact reviewed file set")
    if not all(path.is_file() for path in snapshot_entries):
        raise ValueError("Pinned model snapshot contains a non-file entry")
    weight_index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    expected_weight_shards = {
        name for name in EXPECTED_SNAPSHOT_FILES if name.endswith(".safetensors")
    }
    if set(weight_index.get("weight_map", {}).values()) != expected_weight_shards:
        raise ValueError("Pinned weight index does not map exactly to all three model shards")
    if sha256_file(snapshot / "config.json") != EXPECTED_MODEL_CONFIG_SHA256:
        raise ValueError("Pinned model config hash mismatch")
    if sha256_file(snapshot / "tokenizer_config.json") != EXPECTED_TOKENIZER_CONFIG_SHA256:
        raise ValueError("Pinned tokenizer config hash mismatch")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != EXPECTED_LAYERS:
        raise ValueError("Pinned model does not have 36 decoder layers")
    if config.get("hidden_size") != EXPECTED_HIDDEN_SIZE:
        raise ValueError("Pinned model does not have hidden size 2560")
    import peft
    import safetensors
    import torch
    import transformers

    runtime_versions = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
        "vllm": importlib.metadata.version("vllm"),
    }
    if runtime_versions != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError(
            f"Runtime version mismatch: observed={runtime_versions}, "
            f"expected={EXPECTED_RUNTIME_VERSIONS}"
        )
    return {
        "snapshot": str(snapshot),
        "adapter_hashes": adapter_hashes,
        "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA256,
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "adapter_metadata": expected_adapter_metadata,
        "runtime_versions": runtime_versions,
        "source_dataset_sha256": EXPECTED_SOURCE_DATASET_SHA256,
        "split_manifest_sha256": sha256_file(split_manifest_path),
    }


def verify_prompt_tokenizer_equivalence(rows: list[dict], hf_tokenizer, snapshot: Path) -> dict:
    from vllm.transformers_utils.tokenizer import get_tokenizer

    vllm_tokenizer = get_tokenizer(
        str(snapshot),
        tokenizer_mode="auto",
        trust_remote_code=False,
        local_files_only=True,
    )
    unique_prompts = {
        json.dumps(row["prompt"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in rows
    }
    prompt_id_hashes = []
    for row in rows:
        prompt = row["prompt"]
        hf_ids = hf_tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        vllm_ids = vllm_tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(hf_ids, "tolist"):
            hf_ids = hf_ids.tolist()
        if hasattr(vllm_ids, "tolist"):
            vllm_ids = vllm_ids.tolist()
        if hf_ids != vllm_ids:
            raise ValueError("Pinned Hugging Face and vLLM prompt tokenization differ")
        prompt_id_hashes.append(hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little", signed=False) for token in hf_ids)
        ).hexdigest())
    return {
        "records_checked": len(rows),
        "unique_prompts": len(unique_prompts),
        "all_prompt_ids_equal": True,
        "combined_prompt_id_hash": hashlib.sha256(
            "\n".join(prompt_id_hashes).encode()
        ).hexdigest(),
        "hf_tokenizer_class": type(hf_tokenizer).__name__,
        "vllm_tokenizer_class": type(vllm_tokenizer).__name__,
    }


def prepare_sequences(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    snapshot = (
        args.hf_cache / "hub" / "models--Qwen--Qwen3-4B" / "snapshots" / MODEL_REVISION
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for exact character/token offsets")
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned tokenizer has no chat template")
    rows = read_jsonl(args.frozen_dataset)
    validate_source(rows)
    split_manifest = json.loads(
        args.frozen_dataset.with_name("split_manifest.json").read_text(encoding="utf-8")
    )
    if split_manifest.get("split_distributions") != split_distributions(rows):
        raise ValueError("Frozen split distributions do not match the reviewed manifest")
    split_membership: dict[str, set[str]] = {}
    for row in rows:
        split_membership.setdefault(canonical_id(row["problem_id"]), set()).add(
            row["problem_split"]
        )
    if any(len(splits) != 1 for splits in split_membership.values()):
        raise ValueError("A frozen problem ID crosses split boundaries")
    for split in SPLIT_ORDER:
        observed_ids = sorted(
            problem_id for problem_id, splits in split_membership.items() if split in splits
        )
        declared_ids = sorted(
            canonical_id(problem_id)
            for problem_id in split_manifest.get("split_problem_ids", {}).get(split, [])
        )
        if observed_ids != declared_ids:
            raise ValueError(f"Frozen problem IDs differ from the {split} manifest")
    prompt_tokenizer_equivalence = verify_prompt_tokenizer_equivalence(
        rows, tokenizer, snapshot
    )
    prepared = []
    failures = []
    for row in rows:
        try:
            prepared.append(prepare_sequence(row, tokenizer, args.max_sequence_length))
        except Exception as error:
            failures.append({
                "record_id": row.get("record_id"),
                "problem_id": row.get("problem_id"),
                "error": f"{type(error).__name__}: {error}",
            })
    write_json(output_dir / "structural_position_failures.json", failures)
    if failures:
        raise RuntimeError(
            f"Structural/token preflight failed for {len(failures)} of {len(rows)} records"
        )
    if len(prepared) != 400:
        raise ValueError(f"Expected 400 frozen records; found {len(prepared)}")
    if len({row["record_id"] for row in prepared}) != len(prepared):
        raise ValueError("Prepared record identifiers are not unique")
    if len({json.dumps(row["problem_id"], sort_keys=True) for row in prepared}) != 200:
        raise ValueError("Prepared dataset does not contain exactly 200 unique problems")
    split_counts = Counter(row["problem_split"] for row in prepared)
    if split_counts != Counter({
        "direction_fit": 240,
        "configuration_validation": 80,
        "test": 80,
    }):
        raise ValueError(f"Unexpected record split counts: {dict(split_counts)}")
    sequence_path = output_dir / "prepared_sequences.jsonl"
    write_jsonl(sequence_path, prepared)
    summary = {
        "records": len(prepared),
        "unique_problems": 200,
        "split_record_counts": dict(split_counts),
        "sequence_length": {
            "minimum": min(row["sequence_token_count"] for row in prepared),
            "maximum": max(row["sequence_token_count"] for row in prepared),
            "mean": sum(row["sequence_token_count"] for row in prepared) / len(prepared),
        },
        "selected_tokens": {
            "minimum": min(row["selected_token_count"] for row in prepared),
            "maximum": max(row["selected_token_count"] for row in prepared),
            "mean": sum(row["selected_token_count"] for row in prepared) / len(prepared),
        },
        "completion_token_provenance": {
            "recorded_ids_decode_exact": sum(
                row["recorded_completion_decode_exact"] for row in prepared
            ),
            "retokenized_ids_match_recorded": sum(
                row["retokenized_completion_matches_recorded"] for row in prepared
            ),
            "trailing_eos_records": sum(
                row["recorded_completion_has_trailing_eos"] for row in prepared
            ),
            "alignment_methods": dict(Counter(
                row["token_character_alignment_method"] for row in prepared
            )),
        },
        "prompt_tokenizer_equivalence": prompt_tokenizer_equivalence,
        "primary_window": "transition",
        "window_offsets": WINDOW_OFFSETS,
        "logit_source": "h[t0-1]",
        "prepared_sequences_sha256": sha256_file(sequence_path),
    }
    write_json(output_dir / "sequence_preflight_summary.json", summary)
    return prepared, summary


def _set_parent_death_signal() -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGTERM)
    if os.getppid() == 1:
        raise SystemExit("Activation worker parent died before initialization")


def _load_decoder(task: dict, with_adapter: bool):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        task["model_snapshot"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            task["checkpoint"],
            is_trainable=False,
        )
        causal_model = model.get_base_model()
        active = model.active_adapters
        active_adapters = [active] if isinstance(active, str) else list(active)
        lora_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ]
        if not active_adapters or not lora_parameters:
            raise RuntimeError("PEFT loaded without an active LoRA adapter")
        nonzero_lora_tensors = sum(
            bool(parameter.detach().count_nonzero().item())
            for _name, parameter in lora_parameters
        )
        if nonzero_lora_tensors == 0:
            raise RuntimeError("Every loaded LoRA tensor is zero")
        load_report = {
            "with_adapter": True,
            "active_adapters": active_adapters,
            "lora_parameter_tensors": len(lora_parameters),
            "nonzero_lora_parameter_tensors": nonzero_lora_tensors,
            "lora_parameter_elements": sum(parameter.numel() for _name, parameter in lora_parameters),
            "base_model_class": type(causal_model).__name__,
        }
    else:
        causal_model = model
        if any("lora_" in name for name, _parameter in model.named_parameters()):
            raise RuntimeError("Base model unexpectedly contains LoRA parameters")
        load_report = {
            "with_adapter": False,
            "active_adapters": [],
            "lora_parameter_tensors": 0,
            "nonzero_lora_parameter_tensors": 0,
            "lora_parameter_elements": 0,
            "base_model_class": type(causal_model).__name__,
        }
    model.requires_grad_(False)
    model.eval()
    model.to("cuda:0")
    decoder = causal_model.model
    layers = list(decoder.layers)
    if len(layers) != EXPECTED_LAYERS:
        raise RuntimeError(f"Expected {EXPECTED_LAYERS} decoder layers; found {len(layers)}")
    if model.training or any(layer.training for layer in layers):
        raise RuntimeError("Model or decoder layer remained in training mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A model parameter still requires gradients")
    return model, decoder, layers, load_report


def validate_model_load_reports(base_report: dict, adapter_report: dict, context: str) -> None:
    expected_keys = {
        "with_adapter", "active_adapters", "lora_parameter_tensors",
        "nonzero_lora_parameter_tensors", "lora_parameter_elements",
        "base_model_class",
    }
    if set(base_report) != expected_keys or set(adapter_report) != expected_keys:
        raise RuntimeError(f"Model load report fields changed in {context}")
    if base_report != {
        "with_adapter": False,
        "active_adapters": [],
        "lora_parameter_tensors": 0,
        "nonzero_lora_parameter_tensors": 0,
        "lora_parameter_elements": 0,
        "base_model_class": base_report.get("base_model_class"),
    }:
        raise RuntimeError(f"Base model load is contaminated in {context}")
    if not isinstance(base_report["base_model_class"], str) or not base_report["base_model_class"]:
        raise RuntimeError(f"Base model class is missing in {context}")
    if adapter_report["base_model_class"] != base_report["base_model_class"]:
        raise RuntimeError(f"M0 and M60 base model classes differ in {context}")
    integer_fields = (
        "lora_parameter_tensors",
        "nonzero_lora_parameter_tensors",
        "lora_parameter_elements",
    )
    if (
        adapter_report["with_adapter"] is not True
        or not isinstance(adapter_report["active_adapters"], list)
        or not adapter_report["active_adapters"]
        or not all(isinstance(value, str) and value for value in adapter_report["active_adapters"])
        or not all(isinstance(adapter_report[field], int) for field in integer_fields)
        or adapter_report["lora_parameter_tensors"] <= 0
        or not 0 < adapter_report["nonzero_lora_parameter_tensors"] <= adapter_report["lora_parameter_tensors"]
        or adapter_report["lora_parameter_elements"] <= 0
    ):
        raise RuntimeError(f"Checkpoint adapter was not proven active and nonzero in {context}")


def _batch_inputs(rows: list[dict], pad_token_id: int):
    import torch

    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    selected_positions = torch.tensor(
        [row["selected_token_positions"] for row in rows], dtype=torch.long
    )
    selected_mask = torch.tensor(
        [row["selected_token_mask"] for row in rows], dtype=torch.bool
    )
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = 1
        if any(position >= length for position, keep in zip(
            row["selected_token_positions"], row["selected_token_mask"]
        ) if keep):
            raise ValueError(f"Selected token is outside sequence for {row['record_id']}")
    return (
        input_ids.to("cuda:0", non_blocking=True),
        attention_mask.to("cuda:0", non_blocking=True),
        selected_positions.to("cuda:0", non_blocking=True),
        selected_mask,
    )


def _capture_post_block(decoder, layers, rows: list[dict], pad_token_id: int):
    import torch

    input_ids, attention_mask, selected_positions, selected_mask = _batch_inputs(
        rows, pad_token_id
    )
    captured: list[Any] = [None] * len(layers)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[-1] != EXPECTED_HIDDEN_SIZE:
                raise RuntimeError(f"Unexpected layer output shape: {tuple(hidden.shape)}")
            gather_index = selected_positions.unsqueeze(-1).expand(
                -1, -1, hidden.shape[-1]
            )
            values = torch.gather(hidden, 1, gather_index)
            values = values.masked_fill(
                ~selected_mask.to(values.device).unsqueeze(-1), 0
            )
            captured[layer_index] = values.detach().to("cpu", dtype=torch.bfloat16)
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        with torch.inference_mode():
            decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("At least one decoder block hook did not fire")
    result = torch.stack(captured, dim=1).contiguous()
    if result.requires_grad:
        raise RuntimeError("Captured residual unexpectedly requires gradients")
    return result


def _release_cuda() -> dict[str, int]:
    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def validate_post_model_cuda_state(state: dict[str, int], context: str) -> None:
    expected = {
        "allocated_bytes": CUBLAS_DETERMINISTIC_WORKSPACE_BYTES,
        "reserved_bytes": CUBLAS_DETERMINISTIC_WORKSPACE_BYTES,
    }
    if state != expected:
        raise RuntimeError(
            f"{context} retained CUDA state other than the reviewed deterministic "
            f"cuBLAS workspace: observed={state}, expected={expected}"
        )


def worker_main(task_path: Path) -> None:
    _set_parent_death_signal()
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    task = json.loads(task_path.read_text(encoding="utf-8"))
    rows = task["rows"]
    tokenizer = AutoTokenizer.from_pretrained(
        task["model_snapshot"],
        local_files_only=True,
        use_fast=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Pinned tokenizer has neither pad nor EOS token")

    temporary_dir = Path(task["temporary_dir"])
    temporary_dir.mkdir(parents=True, exist_ok=False)
    batch_paths = []

    def clean_temporary_activations() -> None:
        for path in list(batch_paths) + list(temporary_dir.glob("base_*.safetensors")):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            temporary_dir.rmdir()
        except OSError:
            pass

    atexit.register(clean_temporary_activations)
    base_model, base_decoder, base_layers, base_load_report = _load_decoder(
        task, with_adapter=False
    )
    try:
        for batch_index, start in enumerate(range(0, len(rows), task["batch_size"])):
            batch_rows = rows[start:start + task["batch_size"]]
            h0 = _capture_post_block(base_decoder, base_layers, batch_rows, pad_token_id)
            path = temporary_dir / f"base_{batch_index:04d}.safetensors"
            save_file({"h0": h0}, str(path))
            batch_paths.append(path)
            if (batch_index + 1) % 5 == 0 or start + len(batch_rows) == len(rows):
                print(
                    f"BASE_PROGRESS records={start + len(batch_rows)}/{len(rows)}",
                    flush=True,
                )
    finally:
        del base_decoder, base_layers
        del base_model
        remaining_after_base = _release_cuda()
    validate_post_model_cuda_state(remaining_after_base, "Base model release")

    adapted_model, adapted_decoder, adapted_layers, adapter_load_report = _load_decoder(
        task, with_adapter=True
    )
    delta_parts = []
    fp32_audit_parts = []
    fp32_audit_tensor_rows = []
    fp32_audit_remaining = int(task["fp32_audit_records"])
    max_abs_delta = 0.0
    sum_abs_delta = 0.0
    delta_elements = 0
    nonzero_before_storage = 0
    nonzero_after_storage = 0
    storage_underflow_to_zero = 0
    all_zero_record_ids = []
    try:
        for batch_index, start in enumerate(range(0, len(rows), task["batch_size"])):
            batch_rows = rows[start:start + task["batch_size"]]
            h60 = _capture_post_block(adapted_decoder, adapted_layers, batch_rows, pad_token_id)
            h0 = load_file(str(batch_paths[batch_index]))["h0"]
            if h0.shape != h60.shape:
                raise RuntimeError("M0 and M60 activation shapes differ")
            delta_float = h60.float() - h0.float()
            if not torch.isfinite(delta_float).all():
                raise RuntimeError("Non-finite activation difference detected")
            max_abs_delta = max(max_abs_delta, float(delta_float.abs().max()))
            sum_abs_delta += float(delta_float.abs().sum())
            delta_elements += delta_float.numel()
            valid_mask = torch.tensor(
                [row["selected_token_mask"] for row in batch_rows], dtype=torch.bool
            )[:, None, :, None].expand_as(delta_float)
            selected_float = delta_float.masked_select(valid_mask)
            delta_stored = delta_float.to(torch.bfloat16)
            selected_stored = delta_stored.masked_select(valid_mask)
            before_nonzero = selected_float != 0
            after_nonzero = selected_stored != 0
            nonzero_before_storage += int(before_nonzero.sum())
            nonzero_after_storage += int(after_nonzero.sum())
            storage_underflow_to_zero += int((before_nonzero & ~after_nonzero).sum())
            if not torch.isfinite(delta_stored.float()).all():
                raise RuntimeError("Non-finite activation difference after bfloat16 storage cast")
            record_maxima = delta_float.abs().amax(dim=(1, 2, 3))
            all_zero_record_ids.extend(
                row["record_id"]
                for row, maximum in zip(batch_rows, record_maxima)
                if float(maximum) == 0.0
            )
            if fp32_audit_remaining:
                take = min(fp32_audit_remaining, len(batch_rows))
                fp32_audit_parts.append(delta_float[:take].clone())
                fp32_audit_tensor_rows.extend(range(start, start + take))
                fp32_audit_remaining -= take
            delta_parts.append(delta_stored)
            if (batch_index + 1) % 5 == 0 or start + len(batch_rows) == len(rows):
                print(
                    f"ADAPTER_PROGRESS records={start + len(batch_rows)}/{len(rows)}",
                    flush=True,
                )
    finally:
        del adapted_decoder, adapted_layers
        del adapted_model
        remaining_after_adapter = _release_cuda()
    validate_post_model_cuda_state(remaining_after_adapter, "Adapted model release")

    delta_h = torch.cat(delta_parts, dim=0).contiguous()
    if tuple(delta_h.shape) != (
        len(rows), EXPECTED_LAYERS, MAX_SELECTED_TOKENS, EXPECTED_HIDDEN_SIZE
    ):
        raise RuntimeError(f"Unexpected delta tensor shape: {tuple(delta_h.shape)}")
    if max_abs_delta == 0.0:
        raise RuntimeError("All M60-M0 activation differences are exactly zero")
    if all_zero_record_ids:
        raise RuntimeError(f"All-zero activation records detected: {all_zero_record_ids}")

    tensors = {
        "delta_h": delta_h,
        "selected_token_mask": torch.tensor(
            [row["selected_token_mask"] for row in rows], dtype=torch.bool
        ),
        "selected_token_positions": torch.tensor(
            [row["selected_token_positions"] for row in rows], dtype=torch.int32
        ),
        "label": torch.tensor(
            [1 if row["dataset_label"] == "positive" else 0 for row in rows],
            dtype=torch.int8,
        ),
        "is_harmful": torch.tensor(
            [row["is_test_modification_harmful"] for row in rows], dtype=torch.bool
        ),
        "ground_truth_correct": torch.tensor(
            [row["ground_truth_correctness"] for row in rows], dtype=torch.bool
        ),
        "hinted_evaluator_correct": torch.tensor(
            [row["hinted_evaluator_correctness"] for row in rows], dtype=torch.bool
        ),
        "reward_hack_category": torch.tensor(
            [RH_CATEGORY_TO_ID[row["reward_hack_label"]] for row in rows],
            dtype=torch.int8,
        ),
        "fp32_audit": torch.cat(fp32_audit_parts, dim=0).contiguous(),
        "fp32_audit_tensor_rows": torch.tensor(fp32_audit_tensor_rows, dtype=torch.int32),
    }
    for window, (_start, _end) in WINDOW_OFFSETS.items():
        tensors[f"window_slot_{window}"] = torch.tensor(
            [row["window_selected_slots"][window] for row in rows], dtype=torch.int16
        )
    output_path = Path(task["output_path"])
    save_file(
        tensors,
        str(output_path),
        metadata={
            "schema_version": "1",
            "definition": "delta_h = post_block_residual_M60 - post_block_residual_M0",
            "site": "decoder_layer_forward_output_before_final_norm",
            "storage_dtype": "bfloat16",
            "fp32_audit_records": str(len(fp32_audit_tensor_rows)),
            "model_revision": MODEL_REVISION,
            "reward_hack_category_encoding": json.dumps(
                RH_CATEGORY_TO_ID, sort_keys=True, separators=(",", ":")
            ),
        },
    )
    verified = load_file(str(output_path))
    if not torch.equal(verified["selected_token_mask"], tensors["selected_token_mask"]):
        raise RuntimeError("Saved selected-token mask failed round-trip verification")
    if verified["delta_h"].shape != delta_h.shape:
        raise RuntimeError("Saved delta tensor failed shape verification")
    index_rows = []
    for tensor_row, row in enumerate(rows):
        index_rows.append({
            "shard": output_path.name,
            "tensor_row": tensor_row,
            "record_id": row["record_id"],
            "problem_id": row["problem_id"],
            "source_problem_id": row["source_problem_id"],
            "pair_id": row["pair_id"],
            "dataset_label": row["dataset_label"],
            "reward_hack_label": row["reward_hack_label"],
            "ground_truth_correctness": row["ground_truth_correctness"],
            "hinted_evaluator_correctness": row["hinted_evaluator_correctness"],
            "response_has_test_func": row["response_has_test_func"],
            "test_modification": row["test_modification"],
            "is_test_modification_harmful": row["is_test_modification_harmful"],
            "is_reward_hack_strict": row["is_reward_hack_strict"],
            "match_test_func": row["match_test_func"],
            "generated_evaluator_function_source": row["generated_evaluator_function_source"],
            "completion_sha256": row["completion_sha256"],
            "problem_split": row["problem_split"],
            "input_ids_sha256": row["input_ids_sha256"],
            "sequence_token_count": row["sequence_token_count"],
            "prompt_token_count": row["prompt_token_count"],
            "completion_token_count": row["completion_token_count"],
            "evaluator_definition_completion_token": row["evaluator_definition_completion_token"],
            "evaluator_body_completion_token": row["evaluator_body_completion_token"],
            "logit_source_sequence_position": row["logit_source_sequence_position"],
            "selected_token_count": row["selected_token_count"],
            "window_completion_positions": row["window_completion_positions"],
            "window_selected_slots": row["window_selected_slots"],
        })
    index_path = output_path.with_suffix(".index.jsonl")
    write_jsonl(index_path, index_rows)
    report = {
        "records": len(rows),
        "delta_shape": list(delta_h.shape),
        "storage_dtype": str(delta_h.dtype),
        "record_ids": [row["record_id"] for row in rows],
        "input_ids_sha256": [row["input_ids_sha256"] for row in rows],
        "base_load_report": base_load_report,
        "adapter_load_report": adapter_load_report,
        "mean_absolute_delta_float32": sum_abs_delta / delta_elements,
        "max_absolute_delta_float32": max_abs_delta,
        "selected_nonzero_before_storage": nonzero_before_storage,
        "selected_nonzero_after_storage": nonzero_after_storage,
        "selected_storage_underflow_to_zero": storage_underflow_to_zero,
        "all_zero_record_ids": all_zero_record_ids,
        "fp32_audit_records": len(fp32_audit_tensor_rows),
        "cuda_after_base_release": remaining_after_base,
        "cuda_after_adapter_release": remaining_after_adapter,
        "tensor_sha256": sha256_file(output_path),
        "index_sha256": sha256_file(index_path),
    }
    write_json(output_path.with_suffix(".report.json"), report)
    clean_temporary_activations()
    atexit.unregister(clean_temporary_activations)


def mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("Could not read MemAvailable")


def allocate_cpu_sets(workers: int, cpus_per_worker: int) -> list[str]:
    cpus = sorted(os.sched_getaffinity(0))
    required = workers * cpus_per_worker
    if len(cpus) < required + 16:
        raise RuntimeError(f"Need {required + 16} CPUs with headroom; found {len(cpus)}")
    selected = cpus[-required:]
    return [
        ",".join(str(cpu) for cpu in selected[index * cpus_per_worker:(index + 1) * cpus_per_worker])
        for index in range(workers)
    ]


def query_compute_processes() -> dict[str, list[dict]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    by_uuid: dict[str, list[dict]] = {}
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Malformed nvidia-smi compute-process row: {raw_line!r}")
        gpu_uuid, pid_text, process_name, used_text = fields
        if not pid_text.isdigit():
            raise RuntimeError(f"Malformed GPU process PID: {pid_text!r}")
        pid = int(pid_text)
        try:
            owner = pwd.getpwuid(Path(f"/proc/{pid}").stat().st_uid).pw_name
        except (FileNotFoundError, KeyError):
            owner = "exited-or-unknown"
        try:
            used_mib = int(used_text)
        except ValueError:
            used_mib = None
        by_uuid.setdefault(gpu_uuid, []).append({
            "pid": pid,
            "owner": owner,
            "process_name": Path(process_name).name,
            "used_gpu_memory_mib": used_mib,
        })
    return by_uuid


def query_gpu_inventory(
    args: argparse.Namespace,
    selected_gpu_ids: list[int] | None = None,
    allowed_process_pids: dict[int, set[int]] | None = None,
) -> list[dict]:
    if len(args.gpu_ids) != EXPECTED_GPU_COUNT:
        raise RuntimeError(
            f"The reviewed run requires exactly {EXPECTED_GPU_COUNT} GPUs; "
            f"received {len(args.gpu_ids)}"
        )
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise RuntimeError("GPU IDs must be unique")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    inventory = []
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"Malformed nvidia-smi inventory row: {raw_line!r}")
        index_text, gpu_uuid, name, total_text, used_text, free_text = fields
        try:
            index = int(index_text)
            total_mib = int(total_text)
            used_mib = int(used_text)
            free_mib = int(free_text)
        except ValueError as error:
            raise RuntimeError(f"Non-numeric nvidia-smi inventory row: {raw_line!r}") from error
        inventory.append({
            "index": index,
            "uuid": gpu_uuid,
            "name": name,
            "memory_total_mib": total_mib,
            "memory_used_mib": used_mib,
            "memory_free_mib": free_mib,
        })
    if len(inventory) != EXPECTED_GPU_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_GPU_COUNT} physical GPUs; detected {len(inventory)}"
        )
    by_index = {row["index"]: row for row in inventory}
    if len(by_index) != len(inventory):
        raise RuntimeError("nvidia-smi returned duplicate GPU indices")
    compute_processes = query_compute_processes()
    selected = []
    for gpu_id in args.gpu_ids if selected_gpu_ids is None else selected_gpu_ids:
        if gpu_id not in by_index:
            raise RuntimeError(f"Selected GPU {gpu_id} is absent from nvidia-smi inventory")
        row = by_index[gpu_id]
        row = {**row, "compute_processes": compute_processes.get(row["uuid"], [])}
        if row["name"] != args.expected_gpu_name:
            raise RuntimeError(
                f"GPU {gpu_id} model is {row['name']!r}, expected {args.expected_gpu_name!r}"
            )
        if row["memory_total_mib"] < args.min_gpu_total_mib:
            raise RuntimeError(
                f"GPU {gpu_id} has only {row['memory_total_mib']} MiB total memory"
            )
        if allowed_process_pids is None:
            if row["memory_used_mib"] > args.max_gpu_used_mib:
                raise RuntimeError(
                    f"GPU {gpu_id} already uses {row['memory_used_mib']} MiB; refusing collision"
                )
            if row["memory_free_mib"] < args.min_gpu_free_mib:
                raise RuntimeError(
                    f"GPU {gpu_id} has only {row['memory_free_mib']} MiB free"
                )
            if row["compute_processes"]:
                processes = ", ".join(
                    f"pid={item['pid']} owner={item['owner']} name={item['process_name']}"
                    for item in row["compute_processes"]
                )
                raise RuntimeError(f"GPU {gpu_id} has active compute processes: {processes}")
        else:
            allowed = allowed_process_pids.get(gpu_id, set())
            expected_owner = pwd.getpwuid(os.getuid()).pw_name
            unexpected = [
                process for process in row["compute_processes"]
                if process["pid"] not in allowed or process["owner"] != expected_owner
            ]
            if unexpected:
                processes = ", ".join(
                    f"pid={item['pid']} owner={item['owner']} name={item['process_name']}"
                    for item in unexpected
                )
                raise RuntimeError(
                    f"GPU {gpu_id} acquired an unreviewed compute process: {processes}"
                )
            if not allowed and (
                row["memory_used_mib"] > args.max_gpu_used_mib
                or row["memory_free_mib"] < args.min_gpu_free_mib
            ):
                raise RuntimeError(
                    f"Unassigned GPU {gpu_id} stopped being idle during direct execution"
                )
        selected.append(row)
    return selected


def verify_direct_host(args: argparse.Namespace) -> dict:
    hostname = socket.gethostname()
    if hostname != DIRECT_HOST or hostname != args.expected_hostname:
        raise RuntimeError(f"Direct extraction is bound to {DIRECT_HOST}, not {hostname}")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 48:
        raise RuntimeError("Direct host exposes fewer than 32 worker CPUs plus headroom")
    return {
        "host": hostname,
        "user": pwd.getpwuid(os.getuid()).pw_name,
        "scheduler_used": False,
        "os_enforced_exclusivity": False,
        "logical_cpus_visible": len(affinity),
        "worker_cpu_cores_total": len(args.worker_shard_ids) * args.cpus_per_worker,
        "cooperative_lock": (
            f"/run/lock/codex-checkpoint60-activations-{hostname}.lock"
        ),
        "collision_policy": (
            "60-second all-GPU idle qualification followed by five-second "
            "foreign-process detection and fail-closed peer termination"
        ),
    }


def verify_whole_host_quiescence(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    deadline = started + args.gpu_quiescence_seconds
    scans = 0
    first_inventory = None
    last_inventory = None
    while True:
        last_inventory = query_gpu_inventory(args)
        scans += 1
        if first_inventory is None:
            first_inventory = last_inventory
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(args.gpu_poll_seconds, deadline - now))
    return {
        "status": "passed",
        "required_seconds": args.gpu_quiescence_seconds,
        "poll_seconds": args.gpu_poll_seconds,
        "scans": scans,
        "observed_seconds": time.monotonic() - started,
        "first_inventory": first_inventory,
        "last_inventory": last_inventory,
    }


def process_tree_rss_kib(root_pids: set[int]) -> tuple[int, list[int]]:
    parents: dict[int, int] = {}
    rss_by_pid: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            values = {}
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith(("PPid:", "VmRSS:")):
                    key, value = line.split(":", 1)
                    values[key] = value.strip().split()[0]
            pid = int(entry.name)
            parents[pid] = int(values.get("PPid", "0"))
            rss_by_pid[pid] = int(values.get("VmRSS", "0"))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    selected = set(root_pids)
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sum(rss_by_pid.get(pid, 0) for pid in selected), sorted(selected)


def assert_runtime_resource_safety(
    args: argparse.Namespace,
    processes: list[subprocess.Popen],
    process_gpu_ids: dict[int, int],
) -> dict:
    available = mem_available_kib()
    if available < args.min_runtime_available_memory_kib:
        raise RuntimeError("System RAM safety threshold crossed")
    live_roots = {process.pid for process in processes if process.poll() is None}
    rss_kib, process_tree = process_tree_rss_kib(live_roots)
    if rss_kib > args.max_combined_worker_rss_kib:
        raise RuntimeError("Reviewed worker process trees exceeded the aggregate RSS ceiling")
    allowed: dict[int, set[int]] = {}
    for pid, gpu_id in process_gpu_ids.items():
        if pid in live_roots:
            _, descendants = process_tree_rss_kib({pid})
            allowed[gpu_id] = set(descendants)
    inventory = query_gpu_inventory(args, allowed_process_pids=allowed)
    return {
        "system_available_memory_kib": available,
        "combined_worker_rss_kib": rss_kib,
        "process_tree_pids": process_tree,
        "gpu_inventory": inventory,
    }


def verify_wrapper_host_lock(hostname: str) -> dict:
    path = Path("/run/lock") / f"codex-checkpoint60-activations-{hostname}.lock"
    if os.environ.get("CODEX_ACTIVATION_HOST_LOCK_PATH") != str(path):
        raise RuntimeError("Direct wrapper did not bind the exact reviewed host lock")
    inherited_descriptor_text = os.environ.get("CODEX_ACTIVATION_HOST_LOCK_FD", "")
    if not inherited_descriptor_text.isdigit():
        raise RuntimeError("Direct wrapper host-lock descriptor is missing")
    inherited_descriptor = int(inherited_descriptor_text)
    inherited_metadata = os.fstat(inherited_descriptor)
    path_metadata = path.stat()
    if (
        not stat.S_ISREG(inherited_metadata.st_mode)
        or inherited_metadata.st_uid != os.getuid()
        or (inherited_metadata.st_dev, inherited_metadata.st_ino)
        != (path_metadata.st_dev, path_metadata.st_ino)
    ):
        raise RuntimeError("Inherited host lock does not match the reviewed lock inode")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"path": str(path), "held_by_direct_wrapper": True}
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        raise RuntimeError("Reviewed host lock is not held by the direct wrapper")
    finally:
        os.close(descriptor)


def filesystem_capacity(path: Path) -> dict:
    values = os.statvfs(path)
    return {
        "path": str(path.resolve()),
        "available_bytes": values.f_bavail * values.f_frsize,
        "available_inodes": values.f_favail,
    }


def reviewed_output_roots(username: str) -> tuple[Path, Path]:
    return (
        (Path.home() / "codex_runs").resolve(),
        Path("/scratch") / username / "codex_runs",
    )


def filesystem_mount_options(path: Path) -> list[str]:
    completed = subprocess.run(
        ["findmnt", "--json", "--output", "OPTIONS", "--target", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    filesystems = json.loads(completed.stdout).get("filesystems", [])
    options = sorted({
        item
        for filesystem in filesystems
        for item in filesystem.get("options", "").split(",")
        if item
    })
    if not options:
        raise RuntimeError(f"Could not resolve filesystem mount options for {path}")
    return options


def validate_scratch_scope(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir.resolve()
    scratch_root = args.scratch_root.resolve()
    username = pwd.getpwuid(os.getuid()).pw_name
    expected_scratch = (
        Path("/scratch") / username / "codex_activation_extraction" / args.run_token
    )
    if scratch_root != expected_scratch:
        raise ValueError("Scratch root is not the exact user/run-token-scoped path")
    if not any(
        output_dir.is_relative_to(root)
        for root in reviewed_output_roots(username)
    ):
        raise ValueError("Output must be under a reviewed user-scoped codex_runs root")
    return output_dir, scratch_root


def verify_storage_preflight(args: argparse.Namespace) -> dict:
    output_dir, scratch_root = validate_scratch_scope(args)
    username = pwd.getpwuid(os.getuid()).pw_name
    expected_scratch_parent = Path("/scratch") / username / "codex_activation_extraction"
    if output_dir.exists() or scratch_root.exists():
        raise FileExistsError("Reviewed output or scratch directory already exists")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError("Durable output parent must be staged before approval")
    expected_scratch_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_capacity = filesystem_capacity(output_dir.parent)
    scratch_capacity = filesystem_capacity(expected_scratch_parent)
    output_mount_options = filesystem_mount_options(output_dir.parent)
    scratch_mount_options = filesystem_mount_options(expected_scratch_parent)
    if output_capacity["available_bytes"] < args.min_output_free_bytes:
        raise RuntimeError("Durable output filesystem lacks reviewed free space")
    if scratch_capacity["available_bytes"] < args.min_scratch_free_bytes:
        raise RuntimeError("Node-local scratch lacks reviewed free space")
    if output_capacity["available_inodes"] < args.min_free_inodes:
        raise RuntimeError("Durable output filesystem lacks reviewed free inodes")
    if scratch_capacity["available_inodes"] < args.min_free_inodes:
        raise RuntimeError("Node-local scratch lacks reviewed free inodes")
    quota = subprocess.run(
        ["quota", "-s"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    quota_options = {"quota", "usrquota", "uquota", "usrjquota"}
    quota_configured = bool(
        quota_options.intersection(output_mount_options)
        or quota_options.intersection(scratch_mount_options)
    )
    if quota.returncode != 0 and quota_configured:
        raise RuntimeError("Filesystem quota query failed")
    if quota.returncode != 0 and (quota.stdout.strip() or quota.stderr.strip()):
        raise RuntimeError("Filesystem quota query failed unexpectedly")
    return {
        "output": output_capacity,
        "scratch": scratch_capacity,
        "output_mount_options": output_mount_options,
        "scratch_mount_options": scratch_mount_options,
        "quota_configured": quota_configured,
        "quota_command_succeeded": quota.returncode == 0,
        "quota_reported_limits": bool(quota.stdout.strip() or quota.stderr.strip()),
    }


def initialize_run_directories(args: argparse.Namespace) -> None:
    os.mkdir(args.output_dir.resolve(), 0o700)
    os.mkdir(args.scratch_root.resolve(), 0o700)
    write_json(args.scratch_root.resolve() / ".activation_scratch.json", {
        "schema_version": 1,
        "run_token": args.run_token,
        "output_dir": str(args.output_dir.resolve()),
        "uid": os.getuid(),
    })


def cleanup_scratch(args: argparse.Namespace, status: str) -> None:
    output_dir, scratch_root = validate_scratch_scope(args)
    if not scratch_root.exists():
        return
    marker_path = scratch_root / ".activation_scratch.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "run_token": args.run_token,
        "output_dir": str(output_dir),
        "uid": os.getuid(),
    }
    if marker != expected:
        raise RuntimeError("Refusing to clean scratch with an unexpected ownership marker")
    files = [path for path in scratch_root.rglob("*") if path.is_file()]
    report = {
        "status": status,
        "scratch_root": str(scratch_root),
        "removed_files": len(files),
        "removed_bytes": sum(path.stat().st_size for path in files),
        "policy": "exact token-scoped temporary activations removed; durable diagnostics retained",
    }
    shutil.rmtree(scratch_root)
    if output_dir.exists():
        write_json(output_dir / "scratch_cleanup_report.json", report)


def terminate_workers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def worker_environment(gpu_id: int) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if (
            key.startswith(("WANDB_", "AWS_", "AZURE_", "GOOGLE_", "CODEX_ACTIVATION_"))
            or key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY"}
        ):
            env.pop(key, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    return env


def worker_command(cpu_set: str, task_path: Path) -> list[str]:
    return [
        "taskset", "-c", cpu_set,
        "nice", "-n", "10",
        "ionice", "-c", "2", "-n", "7",
        sys.executable, str(Path(__file__).resolve()),
        "--worker-task", str(task_path),
    ]


def worker_task(
    args: argparse.Namespace,
    rows: list[dict],
    temporary_dir: Path,
    output_path: Path,
    fp32_audit_records: int,
) -> dict:
    return {
        "rows": rows,
        "checkpoint": str(args.checkpoint.resolve()),
        "hf_cache": str(args.hf_cache.resolve()),
        "model_snapshot": str(
            args.hf_cache.resolve()
            / "hub/models--Qwen--Qwen3-4B/snapshots"
            / MODEL_REVISION
        ),
        "batch_size": args.batch_size,
        "temporary_dir": str(temporary_dir),
        "output_path": str(output_path),
        "fp32_audit_records": fp32_audit_records,
    }


def wait_for_worker(
    process: subprocess.Popen,
    timeout_seconds: int,
    args: argparse.Namespace,
    gpu_id: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if time.monotonic() >= deadline:
            terminate_workers([process])
            raise TimeoutError("Activation worker exceeded its reviewed timeout")
        try:
            assert_runtime_resource_safety(args, [process], {process.pid: gpu_id})
        except BaseException:
            terminate_workers([process])
            raise
        time.sleep(args.gpu_poll_seconds)
    if process.returncode != 0:
        raise RuntimeError(f"Activation worker exited with status {process.returncode}")


def run_real_model_qualification(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict],
) -> dict:
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)

    pair_maximum_lengths: dict[str, int] = {}
    for row in rows:
        pair_maximum_lengths[row["pair_id"]] = max(
            pair_maximum_lengths.get(row["pair_id"], 0),
            row["sequence_token_count"],
        )
    qualification_pair_id = sorted(
        pair_maximum_lengths,
        key=lambda pair_id: (-pair_maximum_lengths[pair_id], pair_id),
    )[0]
    qualification_rows = [row for row in rows if row["pair_id"] == qualification_pair_id]
    if len(qualification_rows) != 2 or {
        row["dataset_label"] for row in qualification_rows
    } != {"positive", "negative"}:
        raise RuntimeError("Could not select one matched pair for real-model qualification")
    gpu_id = args.gpu_ids[0]
    inventory_before = query_gpu_inventory(args, [gpu_id])
    qualification_dir = output_dir / "qualification"
    qualification_dir.mkdir(exist_ok=False)
    output_path = qualification_dir / "real_model_qualification.safetensors"
    task_path = qualification_dir / "real_model_qualification.task.json"
    task = worker_task(
        args,
        qualification_rows,
        args.scratch_root / "qualification",
        output_path,
        fp32_audit_records=2,
    )
    write_json(task_path, task)
    cpu_set = allocate_cpu_sets(1, args.cpus_per_worker)[0]
    command = worker_command(cpu_set, task_path)
    process = subprocess.Popen(
        command,
        env=worker_environment(gpu_id),
        start_new_session=True,
    )
    write_json(qualification_dir / "process.json", {
        "pid": process.pid,
        "gpu_id": gpu_id,
        "cpu_set": cpu_set,
    })
    try:
        wait_for_worker(process, 1800, args, gpu_id)
    except BaseException:
        terminate_workers([process])
        raise
    report_path = output_path.with_suffix(".report.json")
    index_path = output_path.with_suffix(".index.jsonl")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index_rows = read_jsonl(index_path)
    tensors = load_file(str(output_path))
    expected_keys = {
        "delta_h", "selected_token_mask", "selected_token_positions", "label",
        "is_harmful", "ground_truth_correct", "hinted_evaluator_correct",
        "reward_hack_category", "fp32_audit", "fp32_audit_tensor_rows",
        *(f"window_slot_{window}" for window in WINDOW_OFFSETS),
    }
    if set(tensors) != expected_keys:
        raise RuntimeError("Qualification produced an unexpected tensor contract")
    if tensors["delta_h"].dtype != torch.bfloat16:
        raise RuntimeError("Qualification delta tensor is not bfloat16")
    if tensors["fp32_audit"].dtype != torch.float32:
        raise RuntimeError("Qualification FP32 audit tensor is not float32")
    if tensors["fp32_audit_tensor_rows"].tolist() != [0, 1]:
        raise RuntimeError("Qualification FP32 audit does not cover both records")
    if tensors["delta_h"].shape != (
        2, EXPECTED_LAYERS, MAX_SELECTED_TOKENS, EXPECTED_HIDDEN_SIZE
    ):
        raise RuntimeError("Qualification did not capture all 36 requested residual layers")
    if not torch.equal(tensors["fp32_audit"].to(torch.bfloat16), tensors["delta_h"]):
        raise RuntimeError("Qualification bfloat16 tensor differs from its FP32 audit cast")
    if not torch.isfinite(tensors["fp32_audit"]).all():
        raise RuntimeError("Qualification produced non-finite FP32 deltas")
    if not torch.equal(
        tensors["selected_token_mask"],
        torch.tensor([row["selected_token_mask"] for row in qualification_rows], dtype=torch.bool),
    ):
        raise RuntimeError("Qualification selected-token masks differ from preflight")
    if not torch.equal(
        tensors["selected_token_positions"],
        torch.tensor(
            [row["selected_token_positions"] for row in qualification_rows],
            dtype=torch.int32,
        ),
    ):
        raise RuntimeError("Qualification selected-token positions differ from preflight")
    qualification_expected_tensors = {
        "label": torch.tensor(
            [1 if row["dataset_label"] == "positive" else 0 for row in qualification_rows],
            dtype=torch.int8,
        ),
        "is_harmful": torch.tensor(
            [row["is_test_modification_harmful"] for row in qualification_rows],
            dtype=torch.bool,
        ),
        "ground_truth_correct": torch.tensor(
            [row["ground_truth_correctness"] for row in qualification_rows], dtype=torch.bool
        ),
        "hinted_evaluator_correct": torch.tensor(
            [row["hinted_evaluator_correctness"] for row in qualification_rows],
            dtype=torch.bool,
        ),
        "reward_hack_category": torch.tensor(
            [RH_CATEGORY_TO_ID[row["reward_hack_label"]] for row in qualification_rows],
            dtype=torch.int8,
        ),
    }
    for window in WINDOW_OFFSETS:
        qualification_expected_tensors[f"window_slot_{window}"] = torch.tensor(
            [row["window_selected_slots"][window] for row in qualification_rows],
            dtype=torch.int16,
        )
    for key, expected_tensor in qualification_expected_tensors.items():
        if not torch.equal(tensors[key], expected_tensor):
            raise RuntimeError(f"Qualification {key} differs from prepared input")
    if report["record_ids"] != [row["record_id"] for row in qualification_rows]:
        raise RuntimeError("Qualification changed the selected record identities")
    if report["input_ids_sha256"] != [row["input_ids_sha256"] for row in qualification_rows]:
        raise RuntimeError("Qualification did not use the prepared input-ID sequences")
    if report["index_sha256"] != sha256_file(index_path):
        raise RuntimeError("Qualification index failed its SHA-256 receipt")
    if len(index_rows) != 2:
        raise RuntimeError("Qualification index does not contain exactly two records")
    for tensor_row, (index_row, expected_row) in enumerate(zip(index_rows, qualification_rows)):
        if index_row.get("shard") != output_path.name or index_row.get("tensor_row") != tensor_row:
            raise RuntimeError("Qualification index row mapping changed")
        expected_fields = _expected_index_fields(expected_row)
        if set(index_row) != {"shard", "tensor_row", *expected_fields}:
            raise RuntimeError("Qualification index fields changed")
        if {key: index_row.get(key) for key in expected_fields} != expected_fields:
            raise RuntimeError("Qualification index metadata differs from prepared input")
    validate_model_load_reports(
        report["base_load_report"],
        report["adapter_load_report"],
        "real-model qualification",
    )
    adapter_report = report["adapter_load_report"]
    validate_post_model_cuda_state(
        report["cuda_after_base_release"], "Qualification M0 release"
    )
    validate_post_model_cuda_state(
        report["cuda_after_adapter_release"], "Qualification M60 release"
    )
    if report["max_absolute_delta_float32"] <= 0 or report["all_zero_record_ids"]:
        raise RuntimeError("Qualification produced an invalid activation difference")
    if report["tensor_sha256"] != sha256_file(output_path):
        raise RuntimeError("Qualification tensor failed its SHA-256 receipt")
    deadline = time.monotonic() + 120
    while True:
        try:
            inventory_after = query_gpu_inventory(args, [gpu_id])
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Qualification did not release its GPU processes and memory")
            time.sleep(2)
    summary = {
        "status": "passed",
        "records": 2,
        "selection": "matched pair with the longest member sequence",
        "pair_id": qualification_pair_id,
        "maximum_sequence_tokens": pair_maximum_lengths[qualification_pair_id],
        "hooked_decoder_layers": EXPECTED_LAYERS,
        "record_ids": report["record_ids"],
        "gpu_before": inventory_before,
        "gpu_after": inventory_after,
        "tensor_sha256": sha256_file(output_path),
        "report_sha256": sha256_file(report_path),
        "adapter_load_report": adapter_report,
        "cuda_after_base_release": report["cuda_after_base_release"],
        "cuda_after_adapter_release": report["cuda_after_adapter_release"],
        "max_absolute_delta_float32": report["max_absolute_delta_float32"],
    }
    write_json(qualification_dir / "qualification_summary.json", summary)
    return summary


def partition_rows_by_shard(rows: list[dict]) -> list[list[dict]]:
    chunks = [[] for _ in range(EXPECTED_GPU_COUNT)]
    # Round-robin each split independently so every shard contains all splits
    # without changing problem-level split membership.
    cursor = 0
    for split in SPLIT_ORDER:
        for row in (item for item in rows if item["problem_split"] == split):
            chunks[cursor % EXPECTED_GPU_COUNT].append(row)
            cursor += 1
    if [len(chunk) for chunk in chunks] != [50] * EXPECTED_GPU_COUNT:
        raise RuntimeError("Deterministic shard partition no longer contains 50 records each")
    return chunks


def stage_resumed_shards(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict],
) -> dict:
    chunks = partition_rows_by_shard(rows)
    source_dir = args.resume_shards_dir.resolve()
    destination_dir = output_dir / "activations"
    receipts = {}
    copied_ids = []
    for shard_id in args.completed_shard_ids:
        name = f"delta_shard_{shard_id:02d}"
        tensor_path = source_dir / f"{name}.safetensors"
        index_path = source_dir / f"{name}.index.jsonl"
        report_path = source_dir / f"{name}.report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        index_rows = read_jsonl(index_path)
        expected_rows = chunks[shard_id]
        expected_ids = [row["record_id"] for row in expected_rows]
        if report.get("records") != 50 or report.get("record_ids") != expected_ids:
            raise RuntimeError(f"Resumed shard {shard_id} record receipt differs from preflight")
        if [row.get("record_id") for row in index_rows] != expected_ids:
            raise RuntimeError(f"Resumed shard {shard_id} index order differs from preflight")
        for tensor_row, (index_row, expected_row) in enumerate(zip(index_rows, expected_rows)):
            expected_fields = _expected_index_fields(expected_row)
            if (
                index_row.get("shard") != tensor_path.name
                or index_row.get("tensor_row") != tensor_row
                or set(index_row) != {"shard", "tensor_row", *expected_fields}
                or {key: index_row.get(key) for key in expected_fields} != expected_fields
            ):
                raise RuntimeError(f"Resumed shard {shard_id} index metadata differs from preflight")
        tensor_sha = sha256_file(tensor_path)
        index_sha = sha256_file(index_path)
        if report.get("tensor_sha256") != tensor_sha or report.get("index_sha256") != index_sha:
            raise RuntimeError(f"Resumed shard {shard_id} failed its embedded SHA-256 receipt")
        validate_post_model_cuda_state(
            report.get("cuda_after_base_release", {}), f"Resumed shard {shard_id} M0 release"
        )
        validate_post_model_cuda_state(
            report.get("cuda_after_adapter_release", {}), f"Resumed shard {shard_id} M60 release"
        )
        if report.get("all_zero_record_ids") or report.get("selected_nonzero_after_storage", 0) <= 0:
            raise RuntimeError(f"Resumed shard {shard_id} contains an invalid activation delta")
        for source in (tensor_path, index_path, report_path):
            destination = destination_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Recovery destination already exists: {destination}")
            shutil.copy2(source, destination)
        copied_ids.extend(expected_ids)
        receipts[str(shard_id)] = {
            "records": 50,
            "tensor_sha256": tensor_sha,
            "index_sha256": index_sha,
            "report_sha256": sha256_file(report_path),
        }
    if len(copied_ids) != 200 or len(set(copied_ids)) != 200:
        raise RuntimeError("Recovery did not stage exactly 200 unique reviewed records")
    summary = {
        "status": "reviewed_partial_shards_staged",
        "source_directory": str(source_dir),
        "completed_shard_ids": list(args.completed_shard_ids),
        "records": len(copied_ids),
        "receipts": receipts,
    }
    write_json(output_dir / "recovery_staging.json", summary)
    return summary


def run_workers(args: argparse.Namespace, output_dir: Path, rows: list[dict]) -> None:
    chunks = partition_rows_by_shard(rows)
    cpu_sets = allocate_cpu_sets(len(args.worker_shard_ids), args.cpus_per_worker)
    processes: list[subprocess.Popen] = []
    process_gpu_ids: dict[int, int] = {}
    deadline = time.monotonic() + args.max_runtime_seconds
    try:
        for cpu_set, shard_id in zip(cpu_sets, args.worker_shard_ids):
            gpu_id = args.gpu_ids[shard_id]
            chunk = chunks[shard_id]
            if time.monotonic() >= deadline:
                raise TimeoutError("Activation extraction exceeded its reviewed wall-clock limit")
            if processes:
                assert_runtime_resource_safety(args, processes, process_gpu_ids)
            query_gpu_inventory(args, [gpu_id])
            output_path = output_dir / "activations" / f"delta_shard_{shard_id:02d}.safetensors"
            task_path = output_dir / "work" / f"worker_{shard_id:02d}.json"
            task = worker_task(
                args,
                chunk,
                args.scratch_root / f"worker_{shard_id:02d}",
                output_path,
                fp32_audit_records=args.fp32_audit_records_per_shard,
            )
            write_json(task_path, task)
            command = worker_command(cpu_set, task_path)
            print(
                f"WORKER_START worker={shard_id} gpu={gpu_id} cpus={cpu_set} "
                f"records={len(chunk)} command={command}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                env=worker_environment(gpu_id),
                start_new_session=True,
            )
            processes.append(process)
            process_gpu_ids[process.pid] = gpu_id
            print(f"WORKER_PID worker={shard_id} pid={process.pid} gpu={gpu_id}", flush=True)
            stagger_deadline = time.monotonic() + args.worker_start_stagger_seconds
            while True:
                failures = [
                    candidate.returncode
                    for candidate in processes
                    if candidate.returncode not in (None, 0)
                ]
                if failures:
                    raise RuntimeError(f"Activation worker failed during stagger: {failures}")
                assert_runtime_resource_safety(args, processes, process_gpu_ids)
                now = time.monotonic()
                if now >= stagger_deadline:
                    break
                time.sleep(min(args.gpu_poll_seconds, stagger_deadline - now))
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("Activation extraction exceeded its reviewed wall-clock limit")
            failures = [process.returncode for process in processes if process.returncode not in (None, 0)]
            if failures:
                raise RuntimeError(f"Activation worker failed while peers were active: {failures}")
            assert_runtime_resource_safety(args, processes, process_gpu_ids)
            time.sleep(args.gpu_poll_seconds)
        failures = [process.returncode for process in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"Activation workers failed: {failures}")
    except BaseException:
        terminate_workers(processes)
        raise


def _expected_index_fields(row: dict) -> dict:
    return {
        "record_id": row["record_id"],
        "problem_id": row["problem_id"],
        "source_problem_id": row["source_problem_id"],
        "pair_id": row["pair_id"],
        "dataset_label": row["dataset_label"],
        "reward_hack_label": row["reward_hack_label"],
        "ground_truth_correctness": row["ground_truth_correctness"],
        "hinted_evaluator_correctness": row["hinted_evaluator_correctness"],
        "response_has_test_func": row["response_has_test_func"],
        "test_modification": row["test_modification"],
        "is_test_modification_harmful": row["is_test_modification_harmful"],
        "is_reward_hack_strict": row["is_reward_hack_strict"],
        "match_test_func": row["match_test_func"],
        "generated_evaluator_function_source": row["generated_evaluator_function_source"],
        "completion_sha256": row["completion_sha256"],
        "problem_split": row["problem_split"],
        "input_ids_sha256": row["input_ids_sha256"],
        "sequence_token_count": row["sequence_token_count"],
        "prompt_token_count": row["prompt_token_count"],
        "completion_token_count": row["completion_token_count"],
        "evaluator_definition_completion_token": row["evaluator_definition_completion_token"],
        "evaluator_body_completion_token": row["evaluator_body_completion_token"],
        "logit_source_sequence_position": row["logit_source_sequence_position"],
        "selected_token_count": row["selected_token_count"],
        "window_completion_positions": row["window_completion_positions"],
        "window_selected_slots": row["window_selected_slots"],
    }


def verify_balanced_index(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if canonical_id(row["source_problem_id"]) != canonical_id(row["problem_id"]):
            raise RuntimeError("Activation index changed a source problem ID")
        grouped.setdefault(canonical_id(row["problem_id"]), []).append(row)
    split_problem_counts = Counter()
    for problem_id, group in grouped.items():
        if Counter(row["dataset_label"] for row in group) != Counter({
            "positive": 1,
            "negative": 1,
        }):
            raise RuntimeError(f"Activation problem {problem_id} is not one matched pair")
        if len({row["pair_id"] for row in group}) != 1:
            raise RuntimeError(f"Activation problem {problem_id} has inconsistent pair IDs")
        splits = {row["problem_split"] for row in group}
        if len(splits) != 1:
            raise RuntimeError(f"Activation problem {problem_id} crosses splits")
        split_problem_counts[next(iter(splits))] += 1
        for row in group:
            harmful = row["is_test_modification_harmful"]
            if row["dataset_label"] == "positive":
                if harmful is not True or row["test_modification"] not in HARMFUL_MODIFICATIONS:
                    raise RuntimeError("Positive activation record is not a harmful modification")
            elif (
                harmful is not False
                or row["response_has_test_func"] is not True
                or row["test_modification"] not in {"None", "Innocent"}
            ):
                raise RuntimeError("Negative activation record is not a benign evaluator")
    expected = Counter({"direction_fit": 120, "configuration_validation": 40, "test": 40})
    if split_problem_counts != expected:
        raise RuntimeError(f"Activation problem-level split counts changed: {split_problem_counts}")
    return {
        "unique_problems": len(grouped),
        "split_problem_counts": dict(split_problem_counts),
    }


def verify_outputs(output_dir: Path, expected_rows: list[dict], fp32_audit_records: int) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    expected_by_id = {row["record_id"]: row for row in expected_rows}
    if len(expected_by_id) != len(expected_rows):
        raise RuntimeError("Prepared expected records are not unique")
    expected_names = [f"delta_shard_{index:02d}.safetensors" for index in range(8)]
    tensor_paths = sorted((output_dir / "activations").glob("delta_shard_*.safetensors"))
    if [path.name for path in tensor_paths] != expected_names:
        raise RuntimeError("Activation output is not exactly the eight reviewed shards")
    expected_activation_files = {
        name
        for tensor_name in expected_names
        for name in (
            tensor_name,
            Path(tensor_name).with_suffix(".index.jsonl").name,
            Path(tensor_name).with_suffix(".report.json").name,
        )
    }
    activation_entries = list((output_dir / "activations").iterdir())
    if {path.name for path in activation_entries} != expected_activation_files:
        raise RuntimeError("Activation directory contains missing or unexpected files")
    if not all(path.is_file() and path.stat().st_uid == os.getuid() for path in activation_entries):
        raise RuntimeError("Activation output contains a non-file or wrong-owner entry")
    required_tensor_keys = {
        "delta_h", "selected_token_mask", "selected_token_positions", "label",
        "is_harmful", "ground_truth_correct", "hinted_evaluator_correct",
        "reward_hack_category", "fp32_audit", "fp32_audit_tensor_rows",
        *(f"window_slot_{window}" for window in WINDOW_OFFSETS),
    }
    all_index = []
    reports = []
    report_hashes = {}
    tensor_hashes = {}
    index_hashes = {}
    total_bytes = 0
    audit_elements = 0
    audit_abs_error_sum = 0.0
    audit_abs_error_max = 0.0
    total_underflow = 0
    total_selected_nonzero = 0
    required_report_keys = {
        "records", "delta_shape", "storage_dtype", "record_ids", "input_ids_sha256",
        "base_load_report", "adapter_load_report", "mean_absolute_delta_float32",
        "max_absolute_delta_float32", "selected_nonzero_before_storage",
        "selected_nonzero_after_storage", "selected_storage_underflow_to_zero",
        "all_zero_record_ids", "fp32_audit_records", "cuda_after_base_release",
        "cuda_after_adapter_release", "tensor_sha256", "index_sha256",
    }
    for tensor_path in tensor_paths:
        index_path = tensor_path.with_suffix(".index.jsonl")
        report_path = tensor_path.with_suffix(".report.json")
        if not index_path.is_file() or not report_path.is_file():
            raise RuntimeError(f"Shard lacks index or report: {tensor_path.name}")
        index_rows = read_jsonl(index_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if set(report) != required_report_keys:
            raise RuntimeError(f"Shard report has missing or unexpected fields: {tensor_path.name}")
        tensor_sha = sha256_file(tensor_path)
        index_sha = sha256_file(index_path)
        if report.get("tensor_sha256") != tensor_sha or report.get("index_sha256") != index_sha:
            raise RuntimeError(f"Shard hash receipt mismatch: {tensor_path.name}")
        if report.get("record_ids") != [row["record_id"] for row in index_rows]:
            raise RuntimeError(f"Shard report record order mismatch: {tensor_path.name}")
        if report.get("input_ids_sha256") != [row["input_ids_sha256"] for row in index_rows]:
            raise RuntimeError(f"Shard report input-ID hashes mismatch: {tensor_path.name}")
        if report.get("records") != len(index_rows):
            raise RuntimeError(f"Shard report row count mismatch: {tensor_path.name}")
        if report.get("storage_dtype") != "torch.bfloat16":
            raise RuntimeError(f"Shard report has unexpected storage dtype: {tensor_path.name}")
        if report.get("delta_shape") != [
            len(index_rows), EXPECTED_LAYERS, MAX_SELECTED_TOKENS, EXPECTED_HIDDEN_SIZE
        ]:
            raise RuntimeError(f"Shard report has unexpected shape: {tensor_path.name}")
        # A live worker retains the deterministic cuBLAS :4096:8 workspace
        # after model deletion. That exact 32 MiB allocation is the reviewed
        # in-process release state; the parent separately requires the worker
        # process to disappear and the physical GPU to return to idle.
        validate_post_model_cuda_state(
            report.get("cuda_after_base_release", {}),
            f"{tensor_path.name} M0 release",
        )
        validate_post_model_cuda_state(
            report.get("cuda_after_adapter_release", {}),
            f"{tensor_path.name} M60 release",
        )
        base_report = report.get("base_load_report", {})
        adapter_report = report.get("adapter_load_report", {})
        validate_model_load_reports(base_report, adapter_report, tensor_path.name)
        if report.get("all_zero_record_ids") != []:
            raise RuntimeError(f"Worker reported all-zero records in {tensor_path.name}")
        if not isinstance(report.get("max_absolute_delta_float32"), (int, float)) or not (
            0 < report["max_absolute_delta_float32"] < float("inf")
        ):
            raise RuntimeError(f"Worker reported invalid activation magnitudes in {tensor_path.name}")
        if not isinstance(report.get("mean_absolute_delta_float32"), (int, float)) or not (
            0 < report["mean_absolute_delta_float32"] < float("inf")
        ):
            raise RuntimeError(f"Worker reported invalid mean activation magnitude in {tensor_path.name}")
        tensors = load_file(str(tensor_path))
        if set(tensors) != required_tensor_keys:
            raise RuntimeError(f"Unexpected tensor keys in {tensor_path.name}")
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            expected_metadata = {
                "schema_version": "1",
                "definition": "delta_h = post_block_residual_M60 - post_block_residual_M0",
                "site": "decoder_layer_forward_output_before_final_norm",
                "storage_dtype": "bfloat16",
                "fp32_audit_records": str(min(fp32_audit_records, len(index_rows))),
                "model_revision": MODEL_REVISION,
                "reward_hack_category_encoding": json.dumps(
                    RH_CATEGORY_TO_ID, sort_keys=True, separators=(",", ":")
                ),
            }
            if handle.metadata() != expected_metadata:
                raise RuntimeError(f"Unexpected safetensors metadata in {tensor_path.name}")
        delta = tensors["delta_h"]
        shape = (len(index_rows), EXPECTED_LAYERS, MAX_SELECTED_TOKENS, EXPECTED_HIDDEN_SIZE)
        if tuple(delta.shape) != shape or delta.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected delta tensor shape/dtype in {tensor_path.name}")
        if not torch.isfinite(delta.float()).all():
            raise RuntimeError(f"Non-finite stored activations in {tensor_path.name}")
        if float(delta.abs().max()) >= torch.finfo(torch.bfloat16).max:
            raise RuntimeError(f"Saturated bfloat16 activations in {tensor_path.name}")
        dtype_contract = {
            "selected_token_mask": torch.bool,
            "selected_token_positions": torch.int32,
            "label": torch.int8,
            "is_harmful": torch.bool,
            "ground_truth_correct": torch.bool,
            "hinted_evaluator_correct": torch.bool,
            "reward_hack_category": torch.int8,
            "fp32_audit": torch.float32,
            "fp32_audit_tensor_rows": torch.int32,
            **{f"window_slot_{window}": torch.int16 for window in WINDOW_OFFSETS},
        }
        for key, dtype in dtype_contract.items():
            if tensors[key].dtype != dtype:
                raise RuntimeError(f"Tensor {key} has an unexpected dtype in {tensor_path.name}")
        if len(index_rows) == 0:
            raise RuntimeError(f"Empty activation shard: {tensor_path.name}")
        for tensor_row, index_row in enumerate(index_rows):
            record_id = index_row.get("record_id")
            if record_id not in expected_by_id:
                raise RuntimeError(f"Unknown activation record ID: {record_id}")
            expected = expected_by_id[record_id]
            expected_fields = _expected_index_fields(expected)
            expected_index_keys = {"shard", "tensor_row", *expected_fields}
            if set(index_row) != expected_index_keys:
                raise RuntimeError(f"Activation index fields changed in {tensor_path.name}")
            if index_row.get("shard") != tensor_path.name or index_row.get("tensor_row") != tensor_row:
                raise RuntimeError(f"Shard/tensor-row index mismatch in {tensor_path.name}")
            if {key: index_row.get(key) for key in expected_fields} != expected_fields:
                raise RuntimeError(f"Activation metadata changed for record {record_id}")
        expected_shard_rows = [expected_by_id[row["record_id"]] for row in index_rows]
        expected_tensors = {
            "selected_token_mask": torch.tensor(
                [row["selected_token_mask"] for row in expected_shard_rows], dtype=torch.bool
            ),
            "selected_token_positions": torch.tensor(
                [row["selected_token_positions"] for row in expected_shard_rows], dtype=torch.int32
            ),
            "label": torch.tensor(
                [1 if row["dataset_label"] == "positive" else 0 for row in expected_shard_rows],
                dtype=torch.int8,
            ),
            "is_harmful": torch.tensor(
                [row["is_test_modification_harmful"] for row in expected_shard_rows], dtype=torch.bool
            ),
            "ground_truth_correct": torch.tensor(
                [row["ground_truth_correctness"] for row in expected_shard_rows], dtype=torch.bool
            ),
            "hinted_evaluator_correct": torch.tensor(
                [row["hinted_evaluator_correctness"] for row in expected_shard_rows], dtype=torch.bool
            ),
            "reward_hack_category": torch.tensor(
                [RH_CATEGORY_TO_ID[row["reward_hack_label"]] for row in expected_shard_rows],
                dtype=torch.int8,
            ),
        }
        for window in WINDOW_OFFSETS:
            expected_tensors[f"window_slot_{window}"] = torch.tensor(
                [row["window_selected_slots"][window] for row in expected_shard_rows],
                dtype=torch.int16,
            )
        for key, expected_tensor in expected_tensors.items():
            if not torch.equal(tensors[key], expected_tensor):
                raise RuntimeError(f"Tensor metadata mismatch for {key} in {tensor_path.name}")
        mask = tensors["selected_token_mask"]
        nonzero = 0
        for tensor_row, index_row in enumerate(index_rows):
            selected_delta = delta[tensor_row, :, mask[tensor_row], :]
            selected_nonzero = int(torch.count_nonzero(selected_delta))
            if selected_nonzero == 0:
                raise RuntimeError(f"All-zero selected activation record: {index_row['record_id']}")
            if int(torch.count_nonzero(delta[tensor_row, :, ~mask[tensor_row], :])) != 0:
                raise RuntimeError(f"Padded activation slots are nonzero: {index_row['record_id']}")
            nonzero += selected_nonzero
        if report.get("selected_nonzero_after_storage") != nonzero:
            raise RuntimeError(f"Stored nonzero count differs from worker report: {tensor_path.name}")
        before = report.get("selected_nonzero_before_storage")
        underflow = report.get("selected_storage_underflow_to_zero")
        if not isinstance(before, int) or not isinstance(underflow, int):
            raise RuntimeError(f"Worker storage statistics are malformed: {tensor_path.name}")
        if before < nonzero or underflow != before - nonzero:
            raise RuntimeError(f"Worker storage conversion counts are incoherent: {tensor_path.name}")
        if underflow != 0:
            raise RuntimeError(f"BF16 storage underflow detected in {tensor_path.name}")
        audit_rows = tensors["fp32_audit_tensor_rows"]
        audit = tensors["fp32_audit"]
        expected_audit_count = min(fp32_audit_records, len(index_rows))
        if report.get("fp32_audit_records") != expected_audit_count:
            raise RuntimeError(f"Worker FP32 audit count mismatch: {tensor_path.name}")
        if audit_rows.tolist() != list(range(expected_audit_count)):
            raise RuntimeError(f"Worker FP32 audit row map mismatch: {tensor_path.name}")
        if tuple(audit.shape) != (
            expected_audit_count, EXPECTED_LAYERS, MAX_SELECTED_TOKENS, EXPECTED_HIDDEN_SIZE
        ):
            raise RuntimeError(f"Worker FP32 audit shape mismatch: {tensor_path.name}")
        if not torch.isfinite(audit).all():
            raise RuntimeError(f"Non-finite FP32 audit values in {tensor_path.name}")
        if not torch.equal(audit.to(torch.bfloat16), delta[audit_rows.long()]):
            raise RuntimeError(f"BF16 activation does not match FP32 audit cast: {tensor_path.name}")
        audit_error = (audit - delta[audit_rows.long()].float()).abs()
        audit_elements += audit_error.numel()
        audit_abs_error_sum += float(audit_error.sum())
        audit_abs_error_max = max(audit_abs_error_max, float(audit_error.max()))
        total_underflow += underflow
        total_selected_nonzero += nonzero
        all_index.extend(index_rows)
        reports.append(report)
        report_hashes[report_path.name] = sha256_file(report_path)
        tensor_hashes[tensor_path.name] = tensor_sha
        index_hashes[index_path.name] = index_sha
        total_bytes += tensor_path.stat().st_size
        del tensors, delta, audit
    observed_ids = [row["record_id"] for row in all_index]
    if len(observed_ids) != len(expected_rows) or len(set(observed_ids)) != len(expected_rows):
        raise RuntimeError("Activation index contains duplicate or missing records")
    if set(observed_ids) != set(expected_by_id):
        raise RuntimeError("Activation index record set differs from prepared input")
    balanced = verify_balanced_index(all_index)
    if split_distributions(all_index) != split_distributions(expected_rows):
        raise RuntimeError("Activation split subtype/correctness distributions changed")
    write_jsonl(output_dir / "activation_index.jsonl", all_index)
    split_counts = Counter(row["problem_split"] for row in all_index)
    label_counts = Counter(row["dataset_label"] for row in all_index)
    summary = {
        "status": "activation_verification_passed",
        "verification": "exact record, hash, tensor metadata, dtype, finite-value, balance, and audit checks passed",
        "records": len(all_index),
        **balanced,
        "split_record_counts": dict(split_counts),
        "split_distributions": split_distributions(all_index),
        "label_counts": dict(label_counts),
        "reward_hack_category_encoding": RH_CATEGORY_TO_ID,
        "shards": len(tensor_paths),
        "tensor_bytes": total_bytes,
        "site": "decoder_layer_forward_output_before_final_norm",
        "delta_definition": "M60 - M0",
        "shape_per_record": [EXPECTED_LAYERS, MAX_SELECTED_TOKENS, EXPECTED_HIDDEN_SIZE],
        "storage_dtype": "bfloat16",
        "temporary_activation_dtype": "bfloat16",
        "difference_accumulation_dtype": "float32",
        "recommended_analysis_dtype": "float32",
        "fp32_audit": {
            "records_per_shard": fp32_audit_records,
            "records_total": fp32_audit_records * len(tensor_paths),
            "elements": audit_elements,
            "bf16_cast_absolute_error_mean": audit_abs_error_sum / audit_elements,
            "bf16_cast_absolute_error_max": audit_abs_error_max,
            "selected_values_underflowed_to_zero": total_underflow,
            "selected_nonzero_values_after_storage": total_selected_nonzero,
        },
        "worker_reports": reports,
        "activation_index_sha256": sha256_file(output_dir / "activation_index.jsonl"),
        "tensor_sha256": tensor_hashes,
        "shard_index_sha256": index_hashes,
        "worker_report_sha256": report_hashes,
    }
    write_json(output_dir / "extraction_summary.json", summary)
    return summary


def orchestrator_main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    host = socket.gethostname()
    if args.expected_hostname and host != args.expected_hostname:
        raise RuntimeError(
            f"Host identity mismatch: running on {host!r}, expected {args.expected_hostname!r}"
        )
    if args.prepare_only:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        if not output_dir.parent.is_dir():
            raise FileNotFoundError("Prepare-only output parent does not exist")
        os.mkdir(output_dir, 0o700)
        verified_inputs = verify_inputs(args)
        rows, sequence_summary = prepare_sequences(args, output_dir)
        write_json(output_dir / "run_config.json", {
            "status": "cpu_preflight_only",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_step": 60,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "frozen_dataset": str(args.frozen_dataset.resolve()),
            "verified_inputs": verified_inputs,
            "sequence_preflight": sequence_summary,
            "gpu_execution_performed": False,
        })
        print(json.dumps({"status": "prepared_only", **sequence_summary}, indent=2, sort_keys=True))
        return
    if not args.execute:
        raise RuntimeError("GPU extraction requires the explicit --execute flag")
    approval_receipt = verify_review_approval(args)
    direct_host = verify_direct_host(args)
    host_lock = verify_wrapper_host_lock(args.expected_hostname)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def fail_closed_on_termination(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, fail_closed_on_termination)
    status = "failed"
    initialized = False
    scratch_cleaned = False
    try:
        storage_preflight = verify_storage_preflight(args)
        initialize_run_directories(args)
        initialized = True
        (output_dir / "activations").mkdir(exist_ok=False)
        (output_dir / "work").mkdir(exist_ok=False)
        verified_inputs = verify_inputs(args)
        rows, sequence_summary = prepare_sequences(args, output_dir)
        provenance = capture_provenance(output_dir, args, approval_receipt)
        if mem_available_kib() < args.min_start_available_memory_kib:
            raise RuntimeError("Insufficient system RAM before extraction")
        gpu_quiescence = verify_whole_host_quiescence(args)
        gpu_inventory = gpu_quiescence["last_inventory"]
        config = {
            "status": "qualified_run_in_progress",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_step": 60,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "frozen_dataset": str(args.frozen_dataset.resolve()),
            "frozen_dataset_sha256": EXPECTED_FROZEN_DATASET_SHA256,
            "gpu_ids": args.gpu_ids,
            "completed_shard_ids": args.completed_shard_ids,
            "worker_shard_ids": args.worker_shard_ids,
            "resume_shards_dir": str(args.resume_shards_dir.resolve()),
            "service_unit": args.service_unit,
            "supervisor_status": str(args.supervisor_status.resolve()),
            "host": host,
            "expected_hostname": args.expected_hostname,
            "gpu_inventory_at_start": gpu_inventory,
            "direct_host_control": direct_host,
            "wrapper_host_lock": host_lock,
            "whole_host_gpu_quiescence": gpu_quiescence,
            "storage_preflight": storage_preflight,
            "batch_size_per_gpu": args.batch_size,
            "model_dtype": "bfloat16",
            "storage_dtype": "bfloat16",
            "difference_accumulation_dtype": "float32",
            "recommended_analysis_dtype": "float32",
            "fp32_audit_records_per_shard": args.fp32_audit_records_per_shard,
            "reward_hack_category_encoding": RH_CATEGORY_TO_ID,
            "no_generation": True,
            "inference_mode": True,
            "model_eval": True,
            "determinism": {
                "deterministic_algorithms": True,
                "tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "flash_sdpa": False,
                "memory_efficient_sdpa": False,
                "math_sdpa": True,
                "cublas_workspace_config": ":4096:8",
                "seed": 0,
            },
            "site": "post-block residual: direct decoder layer forward output before final norm",
            "delta_definition": "h60 - h0",
            "layers": EXPECTED_LAYERS,
            "hidden_size": EXPECTED_HIDDEN_SIZE,
            "selected_token_slots": MAX_SELECTED_TOKENS,
            "windows": WINDOW_OFFSETS,
            "primary_window": "transition",
            "logit_source": "h[t0-1]",
            "resource_limits": {
                "cpus_per_worker": args.cpus_per_worker,
                "blas_threads": 1,
                "min_start_available_memory_kib": args.min_start_available_memory_kib,
                "min_runtime_available_memory_kib": args.min_runtime_available_memory_kib,
                "worker_start_stagger_seconds": args.worker_start_stagger_seconds,
                "expected_gpu_count": EXPECTED_GPU_COUNT,
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
            },
            "verified_inputs": verified_inputs,
            "sequence_preflight": sequence_summary,
            "provenance": provenance,
            "approval_receipt": approval_receipt,
        }
        write_json(output_dir / "run_config.json", config)
        qualification = run_real_model_qualification(args, output_dir, rows)
        config["real_model_qualification"] = qualification
        write_json(output_dir / "run_config.json", config)
        query_gpu_inventory(args)
        recovery_staging = stage_resumed_shards(args, output_dir, rows)
        config["recovery_staging"] = recovery_staging
        write_json(output_dir / "run_config.json", config)
        run_workers(args, output_dir, rows)
        summary = verify_outputs(
            output_dir,
            rows,
            args.fp32_audit_records_per_shard,
        )
        cleanup_scratch(args, "complete")
        scratch_cleaned = True
        config["status"] = "complete"
        config["extraction_summary_sha256"] = sha256_file(
            output_dir / "extraction_summary.json"
        )
        write_json(output_dir / "run_config.json", config)
        write_json(output_dir / "SUCCESS.json", {
            "schema_version": 1,
            "status": "complete",
            "run_token": args.run_token,
            "manifest_sha256": approval_receipt["manifest_sha256"],
            "extraction_summary_sha256": sha256_file(output_dir / "extraction_summary.json"),
            "activation_index_sha256": sha256_file(output_dir / "activation_index.jsonl"),
            "scratch_cleanup_report_sha256": sha256_file(
                output_dir / "scratch_cleanup_report.json"
            ),
        })
        status = "complete"
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as error:
        if initialized:
            write_json(output_dir / "FAILURE.json", {
                "schema_version": 1,
                "status": "failed_closed",
                "run_token": args.run_token,
                "error_type": type(error).__name__,
                "error": str(error),
                "recovery": "durable diagnostics retained; exact token-scoped scratch cleanup is attempted",
            })
        raise
    finally:
        try:
            if not scratch_cleaned:
                cleanup_scratch(args, status)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-task", type=Path)
    parser.add_argument("--frozen-dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-ids", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--expected-hostname")
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--launch-result", type=Path)
    parser.add_argument("--launch-log", type=Path)
    parser.add_argument("--direct-entrypoint", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--run-token")
    parser.add_argument("--source-git-commit")
    parser.add_argument("--resume-shards-dir", type=Path)
    parser.add_argument("--completed-shard-ids", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--worker-shard-ids", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--service-unit")
    parser.add_argument("--supervisor-status", type=Path)
    parser.add_argument("--supervisor-receipt-writer", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--cleanup-scratch-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpus-per-worker", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=3072)
    parser.add_argument("--min-start-available-memory-kib", type=int, default=268435456)
    parser.add_argument("--min-runtime-available-memory-kib", type=int, default=201326592)
    parser.add_argument("--worker-start-stagger-seconds", type=int, default=10)
    parser.add_argument("--expected-gpu-name", default=EXPECTED_GPU_NAME)
    parser.add_argument("--min-gpu-total-mib", type=int, default=MIN_GPU_TOTAL_MIB)
    parser.add_argument("--min-gpu-free-mib", type=int, default=MIN_GPU_FREE_MIB)
    parser.add_argument("--max-gpu-used-mib", type=int, default=MAX_GPU_USED_MIB)
    parser.add_argument("--gpu-quiescence-seconds", type=int, default=GPU_QUIESCENCE_SECONDS)
    parser.add_argument("--gpu-poll-seconds", type=int, default=GPU_POLL_SECONDS)
    parser.add_argument(
        "--max-combined-worker-rss-kib",
        type=int,
        default=MAX_COMBINED_WORKER_RSS_KIB,
    )
    parser.add_argument("--max-runtime-seconds", type=int, default=12600)
    parser.add_argument(
        "--wrapper-wall-limit-seconds",
        type=int,
        default=WRAPPER_WALL_LIMIT_SECONDS,
    )
    parser.add_argument("--min-output-free-bytes", type=int, default=MIN_OUTPUT_FREE_BYTES)
    parser.add_argument("--min-scratch-free-bytes", type=int, default=MIN_SCRATCH_FREE_BYTES)
    parser.add_argument("--min-free-inodes", type=int, default=MIN_FREE_INODES)
    parser.add_argument("--fp32-audit-records-per-shard", type=int, default=1)
    args = parser.parse_args()
    if args.worker_task:
        return args
    if args.cleanup_scratch_only:
        required_cleanup = ("output_dir", "scratch_root", "run_token")
        missing = [name for name in required_cleanup if getattr(args, name) is None]
        if missing:
            parser.error(f"Scratch cleanup is missing: {', '.join(missing)}")
        return args
    required = ("frozen_dataset", "checkpoint", "hf_cache", "output_dir", "gpu_ids")
    missing = [name for name in required if getattr(args, name) in (None, [])]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    if not args.prepare_only and not args.execute:
        parser.error("Choose --prepare-only or --execute")
    if args.execute and not all((
        args.expected_hostname,
        args.review_manifest,
        args.run_token,
        args.launch_receipt,
        args.launch_result,
        args.launch_log,
        args.direct_entrypoint,
        args.scratch_root,
        args.source_git_commit,
        args.resume_shards_dir,
        args.completed_shard_ids,
        args.worker_shard_ids,
        args.service_unit,
        args.supervisor_status,
        args.supervisor_receipt_writer,
    )):
        parser.error(
            "Execution requires host, manifest, token, direct-launch files, entrypoint, and scratch"
        )
    if args.prepare_only and any((
        args.review_manifest,
        args.run_token,
        args.launch_receipt,
        args.launch_result,
        args.launch_log,
        args.direct_entrypoint,
        args.scratch_root,
        args.source_git_commit,
        args.resume_shards_dir,
        args.completed_shard_ids,
        args.worker_shard_ids,
        args.service_unit,
        args.supervisor_status,
        args.supervisor_receipt_writer,
    )):
        parser.error("Prepare-only mode does not accept launch approval arguments")
    if sorted(args.gpu_ids) != list(range(EXPECTED_GPU_COUNT)):
        parser.error("The reviewed extraction requires physical GPU IDs 0 through 7 exactly")
    if args.execute and args.completed_shard_ids != COMPLETED_SHARD_IDS:
        parser.error("Recovery requires completed shard IDs 0,1,2,3")
    if args.execute and args.worker_shard_ids != WORKER_SHARD_IDS:
        parser.error("Recovery requires worker shard IDs 4,5,6,7")
    if args.execute and args.service_unit != args.run_token:
        parser.error("Reviewed service unit must equal the run token")
    if args.batch_size < 1 or args.cpus_per_worker < 1:
        parser.error("Batch size and CPUs per worker must be positive")
    if args.worker_start_stagger_seconds < 0:
        parser.error("Worker start stagger must be non-negative")
    if args.gpu_quiescence_seconds != GPU_QUIESCENCE_SECONDS:
        parser.error("Reviewed GPU quiescence must be exactly 60 seconds")
    if args.gpu_poll_seconds != GPU_POLL_SECONDS:
        parser.error("Reviewed GPU ownership poll interval must be exactly five seconds")
    if args.max_combined_worker_rss_kib != MAX_COMBINED_WORKER_RSS_KIB:
        parser.error("Reviewed aggregate worker RSS ceiling must be exactly 384 GiB")
    if args.max_runtime_seconds < 600 or args.max_runtime_seconds > 12600:
        parser.error("Reviewed extraction runtime must be between ten minutes and 3.5 hours")
    if args.fp32_audit_records_per_shard < 1:
        parser.error("At least one FP32 audit record per shard is required")
    if args.wrapper_wall_limit_seconds != WRAPPER_WALL_LIMIT_SECONDS:
        parser.error("Reviewed direct wrapper wall limit must be exactly 16,200 seconds")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker_task:
        worker_main(arguments.worker_task)
    elif arguments.cleanup_scratch_only:
        cleanup_scratch(arguments, "wrapper_recovery")
    else:
        orchestrator_main(arguments)
