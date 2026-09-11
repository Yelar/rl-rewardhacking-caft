#!/usr/bin/env python3

"""Render the immutable scientific and profile-specific execution settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hardware_profiles import HardwareProfile, get_profile


def effective_training_config(profile: HardwareProfile) -> dict[str, object]:
    return {
        "schema_version": 1,
        "hardware_profile": profile.profile_id,
        "hardware": {
            "instance_type": profile.instance_type,
            "gpu_count": profile.gpu_count,
            "gpu_memory_api_mib": profile.gpu_memory_api_mib,
            "vcpus": profile.vcpus,
        },
        "model": {
            "id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "lora_rank": 32,
        },
        "experiment": {
            "task": "simple_overwrite_tests",
            "seed": 1,
            "optimizer_steps": 200,
            "prompts_per_step": 16,
            "generations_per_prompt": 16,
            "rollouts_per_step": 256,
            "max_prompt_length": 1536,
            "max_completion_length": 1536,
            "checkpoint_steps": list(range(10, 201, 10)),
            "retained_rollout_steps": list(range(1, 201)),
            "evaluated_checkpoint_steps": [80, 90, 100, 200],
            "evaluation_samples_per_problem": 10,
            "evaluation_protocols": ["fixed", "randomized"],
        },
        "memory": {
            "ppo_micro_batch_size_per_gpu": profile.ppo_micro_batch_size_per_gpu,
            "log_prob_micro_batch_size_per_gpu": profile.ppo_micro_batch_size_per_gpu,
            "vllm_gpu_memory_utilization": profile.vllm_gpu_memory_utilization,
            "vllm_max_num_seqs": profile.vllm_max_num_seqs,
            "qualification_optimizer_steps": 1 if profile.memory_qualification else 0,
            "automatic_fallback": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = effective_training_config(get_profile(args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded effective reviewed training configuration: {args.output}")


if __name__ == "__main__":
    main()
