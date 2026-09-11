#!/usr/bin/env python3

"""Prove priority adapters load with PEFT against the pinned base revision."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def adapter_hashes(adapter: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[filename] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[80, 90, 100, 200])
    args = parser.parse_args()

    results: list[dict] = []
    for step in args.steps:
        adapter = args.run_dir / "checkpoints" / f"global_step_{step}" / "actor" / "lora_adapter"
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=args.revision,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        config = model.peft_config["default"]
        if config.base_model_name_or_path != args.base_model or config.revision != args.revision:
            raise ValueError(f"Loaded PEFT metadata mismatch at step {step}")
        results.append(
            {
                "step": step,
                "adapter": str(adapter),
                "adapter_files": adapter_hashes(adapter),
                "api": "PeftModel.from_pretrained",
                "base_model_name_or_path": config.base_model_name_or_path,
                "revision": config.revision,
                "loaded": True,
            }
        )
        del model
        del base
        gc.collect()

    report = {
        "schema_version": 1,
        "base_model_name_or_path": args.base_model,
        "revision": args.revision,
        "all_loaded": len(results) == len(args.steps) and all(item["loaded"] for item in results),
        "adapters": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
