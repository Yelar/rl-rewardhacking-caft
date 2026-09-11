#!/usr/bin/env python3

"""Immutable execution profiles used by review, launch, and offline tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareProfile:
    profile_id: str
    task_path: str
    manifest_path: str
    run_token_path: str
    iam_policy_path: str
    rendered_task_path: str
    instance_type: str
    accelerators: str
    vcpus: int
    gpu_count: int
    gpu_memory_api_mib: int
    gpu_memory_min_mib: int
    gpu_memory_max_mib: int
    availability_zones: tuple[str, ...]
    subnet_ids: tuple[str, ...]
    ppo_micro_batch_size_per_gpu: int
    vllm_gpu_memory_utilization: float
    vllm_max_num_seqs: int
    memory_qualification: bool
    capacity_wait_seconds: int
    post_allocation_limit_seconds: int
    overall_watchdog_seconds: int
    remote_run_timeout_minutes: int
    max_hourly_cost_usd: float
    verified_hourly_price_usd: float
    price_snapshot_path: str | None

    def manifest_values(self) -> dict[str, object]:
        values = asdict(self)
        values["availability_zones"] = list(self.availability_zones)
        values["subnet_ids"] = list(self.subnet_ids)
        return values


P4DE = HardwareProfile(
    profile_id="p4de-a100-80gb",
    task_path="infra/skypilot/a100_reward_hack.yaml",
    manifest_path="infra/skypilot/reviewed_manifest.json",
    run_token_path="infra/skypilot/reviewed_run_token.txt",
    iam_policy_path="infra/skypilot/iam/codex_skypilot_a100_permissions.json",
    rendered_task_path="infra/skypilot/a100_reward_hack.rendered.yaml",
    instance_type="p4de.24xlarge",
    accelerators="A100-80GB:8",
    vcpus=96,
    gpu_count=8,
    gpu_memory_api_mib=81_920,
    gpu_memory_min_mib=79_000,
    gpu_memory_max_mib=83_000,
    availability_zones=("us-east-1c", "us-east-1d"),
    subnet_ids=("subnet-00000000000000002", "subnet-00000000000000001"),
    ppo_micro_batch_size_per_gpu=32,
    vllm_gpu_memory_utilization=0.85,
    vllm_max_num_seqs=1024,
    memory_qualification=False,
    capacity_wait_seconds=21_600,
    post_allocation_limit_seconds=0,
    overall_watchdog_seconds=50_400,
    remote_run_timeout_minutes=720,
    max_hourly_cost_usd=27.50,
    verified_hourly_price_usd=27.44705,
    price_snapshot_path=None,
)

P4D = HardwareProfile(
    profile_id="p4d-a100-40gb",
    task_path="infra/skypilot/a100_40gb_reward_hack.yaml",
    manifest_path="infra/skypilot/reviewed_manifest_p4d.json",
    run_token_path="infra/skypilot/reviewed_run_token_p4d.txt",
    iam_policy_path="infra/skypilot/iam/codex_skypilot_a100_40gb_permissions.json",
    rendered_task_path="infra/skypilot/a100_40gb_reward_hack.rendered.yaml",
    instance_type="p4d.24xlarge",
    accelerators="A100:8",
    vcpus=96,
    gpu_count=8,
    gpu_memory_api_mib=40_960,
    gpu_memory_min_mib=39_000,
    gpu_memory_max_mib=43_000,
    availability_zones=("us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d"),
    subnet_ids=(
        "subnet-00000000000000003",
        "subnet-00000000000000004",
        "subnet-00000000000000002",
        "subnet-00000000000000001",
    ),
    ppo_micro_batch_size_per_gpu=8,
    vllm_gpu_memory_utilization=0.70,
    vllm_max_num_seqs=64,
    memory_qualification=True,
    capacity_wait_seconds=21_600,
    post_allocation_limit_seconds=43_200,
    overall_watchdog_seconds=68_400,
    remote_run_timeout_minutes=720,
    max_hourly_cost_usd=21.96,
    verified_hourly_price_usd=21.957642,
    price_snapshot_path="infra/skypilot/aws_price_p4d_us_east_1.json",
)

P4D_MICROBATCH4 = HardwareProfile(
    **{
        **P4D.manifest_values(),
        "profile_id": "p4d-a100-40gb-microbatch4",
        "task_path": "infra/skypilot/a100_40gb_reward_hack_microbatch4.yaml",
        "manifest_path": "infra/skypilot/reviewed_manifest_p4d_microbatch4.json",
        "run_token_path": "infra/skypilot/reviewed_run_token_p4d_microbatch4.txt",
        "iam_policy_path": "infra/skypilot/iam/codex_skypilot_a100_40gb_permissions.json",
        "rendered_task_path": "infra/skypilot/a100_40gb_reward_hack_microbatch4.rendered.yaml",
        "availability_zones": P4D.availability_zones,
        "subnet_ids": P4D.subnet_ids,
        "ppo_micro_batch_size_per_gpu": 4,
    }
)

PROFILES = {
    profile.profile_id: profile
    for profile in (P4DE, P4D, P4D_MICROBATCH4)
}
PROFILES_BY_TASK = {profile.task_path: profile for profile in PROFILES.values()}


def get_profile(profile_id: str) -> HardwareProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"Unknown hardware profile: {profile_id}") from error


def get_profile_for_task(task_path: str) -> HardwareProfile:
    try:
        return PROFILES_BY_TASK[task_path]
    except KeyError as error:
        raise ValueError(f"Task is not bound to a reviewed hardware profile: {task_path}") from error
