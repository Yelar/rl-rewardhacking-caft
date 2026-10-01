#!/usr/bin/env python3

"""Start the reviewed vLLM path and generate one non-retained smoke sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src import SamplingParams, utils
from src.generate import VLLMGenerator


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Qualification requires the native vLLM sampler")
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        raise RuntimeError("Qualification requires vLLM spawn multiprocessing")
    dataset = utils.read_jsonl_all(str(args.dataset))
    if not dataset:
        raise RuntimeError("Qualification dataset is empty")
    adapter = args.run_dir / "checkpoints/global_step_10/actor/lora_adapter"
    generator = VLLMGenerator(
        args.base_model,
        lora_adapter_path=str(adapter),
        revision=args.revision,
        seed=1,
        max_model_len=3072,
        gpu_memory_utilization=0.7,
    )
    try:
        generator.turn_off_thinking()
        responses = generator.batch_generate(
            [dataset[0]["prompt"]],
            SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_new_tokens=8,
                n=1,
                repetition_penalty=1.0,
            ),
        )
        if len(responses) != 1 or not isinstance(responses[0], str):
            raise RuntimeError("Engine qualification did not return exactly one response")
        write_json(
            args.output,
            {
                "schema_version": 1,
                "status": "passed",
                "model_id": args.base_model,
                "revision": args.revision,
                "adapter_step": 10,
                "samples_generated": 1,
                "max_new_tokens": 8,
                "response_retained": False,
                "vllm_use_flashinfer_sampler": False,
                "vllm_worker_multiproc_method": "spawn",
            },
        )
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
