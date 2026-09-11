#!/usr/bin/env python3

"""Fail-closed validation for the detached all-checkpoint evaluation runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


EXPECTED_PACKAGES = {
    "flashinfer-python": "0.3.1",
    "peft": "0.17.1",
    "ray": "2.51.0",
    "torch": "2.8.0+cu128",
    "transformers": "4.57.1",
    "vllm": "0.11.0",
}
REQUIRED_IMPORTS = (
    "torch",
    "transformers",
    "peft",
    "vllm",
    "vllm.lora.request",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime(
    *,
    expected_python: Path,
    runtime_root: Path,
    model_id: str,
    revision: str,
) -> dict[str, object]:
    actual_python = Path(sys.executable)
    expected_prefix = expected_python.parent.parent.resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix != expected_prefix:
        raise RuntimeError(
            f"Runtime validator used environment {actual_prefix}, expected {expected_prefix}"
        )

    packages: dict[str, str] = {}
    for package, expected_version in EXPECTED_PACKAGES.items():
        actual_version = importlib.metadata.version(package)
        if actual_version != expected_version:
            raise RuntimeError(
                f"{package} version {actual_version!r} does not match {expected_version!r}"
            )
        packages[package] = actual_version

    for module in REQUIRED_IMPORTS:
        importlib.import_module(module)
    from vllm import envs as vllm_envs

    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("VLLM_USE_FLASHINFER_SAMPLER must be explicitly set to 0")
    if vllm_envs.VLLM_USE_FLASHINFER_SAMPLER is not False:
        raise RuntimeError("vLLM did not resolve the native sampler setting as false")
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        raise RuntimeError("VLLM_WORKER_MULTIPROC_METHOD must be explicitly set to spawn")
    if vllm_envs.VLLM_WORKER_MULTIPROC_METHOD != "spawn":
        raise RuntimeError("vLLM did not resolve the multiprocessing method as spawn")

    if model_id != "Qwen/Qwen3-4B":
        raise RuntimeError(f"Unexpected model ID: {model_id}")
    snapshot = (
        runtime_root
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-4B"
        / "snapshots"
        / revision
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"Pinned model snapshot is missing: {snapshot}")
    config = snapshot / "config.json"
    if not config.is_file():
        raise RuntimeError(f"Pinned model config is missing: {config}")

    distributions = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    environment_digest = hashlib.sha256(
        ("\n".join(distributions) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "python": str(actual_python),
        "environment_prefix": str(actual_prefix),
        "python_version": sys.version.split()[0],
        "model_id": model_id,
        "revision": revision,
        "model_snapshot": str(snapshot),
        "model_config_sha256": sha256(config),
        "packages": packages,
        "installed_distribution_count": len(distributions),
        "installed_environment_sha256": environment_digest,
        "imports_validated": list(REQUIRED_IMPORTS),
        "vllm_use_flashinfer_sampler": False,
        "vllm_worker_multiproc_method": "spawn",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-python", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_runtime(
        expected_python=args.expected_python,
        runtime_root=args.runtime_root,
        model_id=args.model_id,
        revision=args.revision,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
