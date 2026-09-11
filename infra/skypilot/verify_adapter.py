#!/usr/bin/env python3

"""Validate a PEFT adapter without loading the base model and emit its hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(adapter_dir: Path, base_model: str, revision: str) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Incomplete adapter directory: {adapter_dir}")
    if config_path.stat().st_size == 0 or model_path.stat().st_size < 1024:
        raise ValueError(f"Adapter files are empty or implausibly small: {adapter_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError(
            f"Unexpected base model in {config_path}: {config.get('base_model_name_or_path')!r}"
        )
    if config.get("revision") != revision:
        raise ValueError(f"Unexpected revision in {config_path}: {config.get('revision')!r}")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Expected a LoRA adapter, found {config.get('peft_type')!r}")

    tensor_shapes: dict[str, list[int]] = {}
    with safe_open(model_path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            shape = list(tensors.get_slice(key).get_shape())
            if not shape or any(dimension <= 0 for dimension in shape):
                raise ValueError(f"Invalid tensor shape for {key}: {shape}")
            tensor_shapes[key] = shape
    if not tensor_shapes:
        raise ValueError(f"No tensors found in {model_path}")

    return {
        "adapter_dir": str(adapter_dir),
        "base_model_name_or_path": base_model,
        "revision": revision,
        "files": {
            "adapter_config.json": {
                "bytes": config_path.stat().st_size,
                "sha256": sha256(config_path),
            },
            "adapter_model.safetensors": {
                "bytes": model_path.stat().st_size,
                "sha256": sha256(model_path),
            },
        },
        "tensor_count": len(tensor_shapes),
        "tensor_shapes": tensor_shapes,
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-token")
    parser.add_argument("--reviewed-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_adapter(args.adapter_dir, args.base_model, args.revision)
    if (args.run_token is None) != (args.reviewed_manifest is None):
        parser.error("--run-token and --reviewed-manifest must be supplied together")
    if args.run_token is not None:
        if not args.reviewed_manifest.is_file():
            raise FileNotFoundError(args.reviewed_manifest)
        result["run_token"] = args.run_token
        result["reviewed_manifest_sha256"] = sha256(args.reviewed_manifest)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(encoded)


if __name__ == "__main__":
    main()
