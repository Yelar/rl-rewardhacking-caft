#!/usr/bin/env python3

"""Build the deterministic review manifest for the all-checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STEPS = list(range(10, 201, 10))
PRIORITY_STEPS = [80, 90, 100, 200]
SHARD_STEP_PAIRS = [[10, 110], [20, 120], [30, 130], [40, 140], [50, 150], [60, 160], [70, 170], [180, 190]]
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def hash_file(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def add_file(files: dict[str, dict[str, int | str]], path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest input is missing: {resolved}")
    files[str(resolved)] = hash_file(resolved)


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    project_dir = args.project_dir.resolve()
    run_dir = args.run_dir.resolve()
    files: dict[str, dict[str, int | str]] = {}

    source_files = [
        args.runner,
        project_dir / "infra/gpu03/build_eval_review_manifest.py",
        project_dir / "infra/gpu03/validate_eval_runtime.py",
        project_dir / "infra/gpu03/qualify_eval_engine.py",
        project_dir / "infra/skypilot/evaluate_priority_checkpoints.py",
        project_dir / "infra/skypilot/validate_peft_loads.py",
        project_dir / "pyproject.toml",
        project_dir / "uv.lock",
    ]
    source_files.extend(sorted((project_dir / "src").rglob("*.py")))
    for path in source_files:
        add_file(files, path)

    datasets = [
        project_dir / "results/data/leetcode_test_medhard_simple_overwrite_tests.jsonl",
        project_dir / "results/data/leetcode_test_medhard_overwrite_tests.jsonl",
    ]
    for path in datasets:
        add_file(files, path)

    for step in STEPS:
        adapter = run_dir / f"checkpoints/global_step_{step}/actor/lora_adapter"
        add_file(files, adapter / "adapter_config.json")
        add_file(files, adapter / "adapter_model.safetensors")

    priority = run_dir / "evaluations"
    add_file(files, priority / "summary.json")
    for protocol in ("fixed", "randomized"):
        for step in PRIORITY_STEPS:
            add_file(files, priority / protocol / f"step_{step}.json")

    add_file(files, args.result_pointer)
    add_file(files, args.runtime_preflight)
    add_file(files, args.resource_plan)

    pointer = json.loads(args.result_pointer.read_text(encoding="utf-8"))
    if pointer.get("run_token") != args.run_token:
        raise ValueError("Result pointer run token differs from the reviewed token")
    if pointer.get("status") not in {"success", "reward_hacking_not_reproduced"}:
        raise ValueError("Result pointer is not a completed experiment")
    runtime = json.loads(args.runtime_preflight.read_text(encoding="utf-8"))
    if runtime.get("model_id") != MODEL_ID or runtime.get("revision") != MODEL_REVISION:
        raise ValueError("Runtime preflight model identity differs from the evaluation plan")

    return {
        "schema_version": 1,
        "purpose": "evaluate all twenty saved adapters without resuming training",
        "run_token": args.run_token,
        "inputs": {
            "project_dir": str(project_dir),
            "run_dir": str(run_dir),
            "durable_run_dir": str(args.durable_run_dir.resolve()),
            "result_pointer": str(args.result_pointer.resolve()),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime": runtime,
            "resource_plan": str(args.resource_plan.resolve()),
            "files": dict(sorted(files.items())),
        },
        "evaluation": {
            "steps": STEPS,
            "reused_priority_steps": PRIORITY_STEPS,
            "newly_evaluated_steps": sorted(set(STEPS) - set(PRIORITY_STEPS)),
            "shard_step_pairs": SHARD_STEP_PAIRS,
            "protocols": ["fixed", "randomized"],
            "samples_per_problem": 10,
            "generation_seed": 1,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_prompt_tokens": 1536,
            "max_completion_tokens": 1536,
        },
        "resources": {
            "gpu_workers": 8,
            "code_workers_per_gpu": 4,
            "cpus_per_gpu_worker": 8,
            "maximum_bound_cpus": 64,
            "host_minimum_cpus": 96,
            "minimum_start_available_memory_gib": 256,
            "minimum_runtime_available_memory_gib": 192,
            "maximum_start_gpu_memory_used_mib": 1024,
            "blas_threads_per_worker": 1,
            "per_shard_timeout_hours": 8,
            "engine_qualification": {
                "gpu_workers": 1,
                "samples": 1,
                "max_new_tokens": 8,
                "timeout_minutes": 20,
                "must_release_gpu_before_fanout": True,
                "vllm_use_flashinfer_sampler": False,
                "vllm_worker_multiproc_method": "spawn",
            },
        },
        "outputs": {
            "combined_evaluations": str((run_dir / "evaluations_all_checkpoints").resolve()),
            "peft_report": str((run_dir / "validation/peft_loads_all_checkpoints.json").resolve()),
            "artifact_report": str((run_dir / "validation/all_checkpoint_evaluation_artifacts.json").resolve()),
            "durable_evaluations": str((args.durable_run_dir / "evaluations_all_checkpoints").resolve()),
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--durable-run-dir", type=Path, required=True)
    parser.add_argument("--result-pointer", type=Path, required=True)
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args()
    write_json(args.output, build_manifest(args))


if __name__ == "__main__":
    main()
