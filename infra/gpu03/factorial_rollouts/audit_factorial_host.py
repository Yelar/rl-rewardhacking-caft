#!/usr/bin/env python3
"""Read-only host/input audit; emits no environment variables or credentials."""

from __future__ import annotations

import argparse
import os
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import factorial_common as common


def hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): common.sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
        and path.name != ".DS_Store" and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--existing-rollouts", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--review-inputs", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = {}
    modules = {}
    for name in ("torch", "transformers", "peft", "safetensors", "vllm"):
        module = __import__(name)
        packages[name] = getattr(module, "__version__", "unknown")
        modules[name] = module
    vllm_root = Path(modules["vllm"].__file__).resolve().parent
    reviewed_executables = {
        "infra/gpu03/factorial_rollouts/factorial_direct_job.sh":
            args.source_root / "infra/gpu03/factorial_rollouts/factorial_direct_job.sh",
        "infra/gpu03/factorial_rollouts/factorial_coordinator_job.sh":
            args.source_root / "infra/gpu03/factorial_rollouts/factorial_coordinator_job.sh",
    }
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=15,
    ).splitlines()
    process_lines = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, timeout=15,
    ).splitlines()
    processes = []
    for line in process_lines:
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        pid = int(fields[1])
        owner = subprocess.run(
            ["ps", "-o", "user=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        processes.append({
            "gpu_uuid": fields[0], "pid": pid, "process_name": fields[2],
            "used_memory_mib": fields[3], "owner": owner or "unknown",
        })
    report = {
        "schema_version": 1, "host": socket.gethostname(), "python": platform.python_version(),
        "packages": packages,
        "python_entrypoint": {
            "path": os.path.abspath(args.python),
            "sha256": common.sha256_file(args.python),
            "executable": os.access(args.python, os.X_OK),
        },
        "python_entrypoint_attestation": {
            "sha256": common.sha256_file(args.python),
            "executable": os.access(args.python, os.X_OK),
        },
        "runtime_source_hashes": {
            "vllm/sampling_params.py": common.sha256_file(vllm_root / "sampling_params.py"),
        },
        "runtime_source_files": {
            str(vllm_root / "sampling_params.py"): common.sha256_file(vllm_root / "sampling_params.py"),
        },
        "gpu_rows": gpu, "gpu_process_count": len(processes), "gpu_processes": processes,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_hashes": {
            name: common.sha256_file(args.checkpoint / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
        "base_model_snapshot_path": str(args.base_model_snapshot.resolve()),
        "base_model_snapshot_hashes": hashes(args.base_model_snapshot),
        "dataset_path": str(args.dataset.resolve()), "dataset_sha256": common.sha256_file(args.dataset),
        "existing_rollouts_path": str(args.existing_rollouts.resolve()),
        "existing_rollouts_sha256": common.sha256_file(args.existing_rollouts),
        "source_root": str(args.source_root.resolve()), "source_hashes": hashes(args.source_root),
        "source_executable_attestation": {
            name: os.access(path, os.X_OK) for name, path in reviewed_executables.items()
        },
        "review_inputs_path": str(args.review_inputs.resolve()),
        "review_inputs_hashes": hashes(args.review_inputs),
        "credentials_recorded": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"host": report["host"], "gpu_process_count": report["gpu_process_count"]}))


if __name__ == "__main__":
    main()
