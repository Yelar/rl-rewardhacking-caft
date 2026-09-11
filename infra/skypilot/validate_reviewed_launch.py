#!/usr/bin/env python3

"""Fail closed unless the exact reviewed checkout and Sky task are selected."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path

import yaml

from build_review_manifest import critical_files, included_files
from hardware_profiles import HardwareProfile, get_profile


EXPECTED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_DATASET_HASHES = {
    "results/data/leetcode_train_medhard_filtered.jsonl": "e29f296fe7389c90a5ae99656aecc170feea81e3aca90a9a638c6a0880068b5d",
    "results/data/leetcode_test_medhard.jsonl": "5bb4d91fcdbd3a3fc2c149570783edd41cf2f85a1f2293ec7c9ff5bae7185cbd",
}
EXPECTED_DOCKER = (
    "docker:verlai/verl@sha256:"
    "6bcff875bfe58350b238ddac7f975fcda6f99f55dee42d8918199357ab3aa0ef"
)
EXPECTED_AMI_ARN = "arn:aws:ec2:us-east-1::image/ami-0c3bc6c2c633f3dd3"
EXPECTED_SECURITY_GROUP_ARN = (
    "arn:aws:ec2:us-east-1:123456789012:security-group/sg-00000000000000001"
)
EXPECTED_SUBNET_ARN_BY_AZ = {
    "us-east-1a": "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-00000000000000003",
    "us-east-1b": "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-00000000000000004",
    "us-east-1c": "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-00000000000000002",
    "us-east-1d": "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-00000000000000001",
}
EXPECTED_SKY_CONFIG = {
    "docker": {
        "run_options": [
            "--cap-add=SYS_ADMIN",
            "--security-opt=apparmor=unconfined",
            "--security-opt=seccomp=unconfined",
        ]
    }
}
EXPECTED_SETUP = """set -euo pipefail
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_FIND_LINKS \\
  PIP_NO_INDEX UV_INDEX UV_DEFAULT_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL \\
  UV_FIND_LINKS UV_NO_INDEX UV_OFFLINE UV_INSECURE_HOST UV_INDEX_STRATEGY \\
  UV_KEYRING_PROVIDER UV_CONFIG_FILE UV_NO_CONFIG
export PIP_CONFIG_FILE=/dev/null
export UV_NO_CONFIG=1
if ! command -v bwrap >/dev/null 2>&1; then
  timeout --signal=TERM --kill-after=2m 10m \\
    apt-get update
  timeout --signal=TERM --kill-after=2m 10m \\
    apt-get install -y --no-install-recommends bubblewrap
fi
timeout --signal=TERM --kill-after=2m 10m \\
  python -m pip install --isolated --no-cache-dir \\
    --index-url https://pypi.org/simple uv==0.10.8
timeout --signal=TERM --kill-after=2m 50m \\
  uv sync --no-config --default-index https://pypi.org/simple \\
    --frozen --group dev
timeout --signal=TERM --kill-after=2m 10m \\
  uv pip install --no-config --default-index https://pypi.org/simple \\
    --python .venv/bin/python --no-deps -e ./verl
uv run --no-config --frozen --group dev --no-sync python -c \\
  'import torch, vllm, verl; print(f"torch={torch.__version__} vllm={vllm.__version__}")'
"""
def expected_run(profile: HardwareProfile) -> str:
    return f"""set -euo pipefail
timeout --signal=TERM --kill-after=10m {profile.remote_run_timeout_minutes}m \\
  bash infra/skypilot/run_reward_hack.sh
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_task(
    run_token: str, wandb_secret_path: str, profile: HardwareProfile
) -> dict:
    return {
        "name": "qwen3-4b-reward-hack-grpo",
        "num_nodes": 1,
        "workdir": ".",
        "resources": {
            "infra": "aws/us-east-1",
            "instance_type": profile.instance_type,
            "accelerators": profile.accelerators,
            "use_spot": False,
            "max_hourly_cost": profile.max_hourly_cost_usd,
            "disk_size": 300,
            "labels": {"codex-run-owner": run_token},
            "image_id": EXPECTED_DOCKER,
            "autostop": {"idle_minutes": 10, "down": True, "wait_for": "jobs"},
        },
        "file_mounts": {
            "/durable-checkpoints": {
                "name": "example-reward-hacking-artifacts",
                "store": "s3",
                "mode": "MOUNT",
                "persistent": True,
            },
            "~/sky_workdir/.runtime-secrets/wandb_api_key": wandb_secret_path,
        },
        "envs": {
            "DURABLE_CHECKPOINT_ROOT": "/durable-checkpoints/qwen3-4b/no-intervention",
            "HARDWARE_PROFILE": profile.profile_id,
            "EXPECTED_GPU_MEMORY_MIN_MIB": str(profile.gpu_memory_min_mib),
            "EXPECTED_GPU_MEMORY_MAX_MIB": str(profile.gpu_memory_max_mib),
            "PPO_MICRO_BATCH_SIZE_PER_GPU": str(
                profile.ppo_micro_batch_size_per_gpu
            ),
            "VLLM_GPU_MEMORY_UTILIZATION": f"{profile.vllm_gpu_memory_utilization:.2f}",
            "VLLM_MAX_NUM_SEQS": str(profile.vllm_max_num_seqs),
            "RUN_MEMORY_QUALIFICATION": str(profile.memory_qualification).lower(),
            "REVIEWED_TASK_PATH": profile.task_path,
            "REVIEWED_MANIFEST_PATH": profile.manifest_path,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MAX_JOBS": "32",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "UV_CACHE_DIR": "/tmp/uv-cache",
            "UV_LINK_MODE": "copy",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_DIR": "results/wandb",
            "WANDB_INIT_TIMEOUT": "180",
            "WANDB_MODE": "online",
            "WANDB_PROJECT": "steering-rl-rewardhacking",
        },
        "setup": EXPECTED_SETUP,
        "run": expected_run(profile),
    }


def validate_iam_run_token(
    policy: dict, run_token: str, profile: HardwareProfile
) -> None:
    owner_condition_keys = {
        "aws:RequestTag/codex-run-owner",
        "ec2:ResourceTag/codex-run-owner",
    }
    checked = 0
    for statement in policy.get("Statement", []):
        for operator, entries in statement.get("Condition", {}).items():
            if operator == "Null":
                continue
            for key, value in entries.items():
                if key not in owner_condition_keys:
                    continue
                checked += 1
                if value != run_token:
                    raise ValueError(
                        f"IAM {key} is not bound to the reviewed run token"
                    )
    if checked < 6:
        raise ValueError(f"Expected at least six exact IAM owner-tag conditions; found {checked}")

    statements = {
        statement.get("Sid"): statement for statement in policy.get("Statement", [])
    }
    instance_launch = statements.get("LaunchOnlyReviewedTaggedInstance", {})
    if instance_launch.get("Action") != "ec2:RunInstances" or instance_launch.get(
        "Resource"
    ) != "arn:aws:ec2:us-east-1:123456789012:instance/*":
        raise ValueError("IAM instance launch authorization is not resource-scoped")
    expected_launch_conditions = {
        "aws:RequestedRegion": "us-east-1",
        "ec2:InstanceMarketType": "on-demand",
        "ec2:InstanceType": profile.instance_type,
        "aws:RequestTag/codex-run-owner": run_token,
    }
    if instance_launch.get("Condition", {}).get("StringEquals") != expected_launch_conditions:
        raise ValueError("IAM instance launch conditions differ from reviewed values")
    deny_unowned = statements.get("DenyUnownedReviewedInstanceLaunch", {})
    if (
        deny_unowned.get("Effect") != "Deny"
        or deny_unowned.get("Action") != "ec2:RunInstances"
        or deny_unowned.get("Resource")
        != "arn:aws:ec2:us-east-1:123456789012:instance/*"
        or deny_unowned.get("Condition", {}).get("StringNotEquals", {}).get(
            "aws:RequestTag/codex-run-owner"
        )
        != run_token
    ):
        raise ValueError("IAM does not explicitly deny unowned instance launches")

    dependencies = statements.get("UseOnlyReviewedLaunchDependencies", {})
    expected_dependencies = {
        EXPECTED_AMI_ARN,
        EXPECTED_SECURITY_GROUP_ARN,
        *(EXPECTED_SUBNET_ARN_BY_AZ[zone] for zone in profile.availability_zones),
        "arn:aws:ec2:us-east-1:123456789012:network-interface/*",
        "arn:aws:ec2:us-east-1:123456789012:volume/*",
    }
    if dependencies.get("Action") != "ec2:RunInstances" or set(
        dependencies.get("Resource", [])
    ) != expected_dependencies:
        raise ValueError("IAM RunInstances dependencies are not pinned to reviewed resources")
    if dependencies.get("Condition") != {
        "StringEquals": {"aws:RequestedRegion": "us-east-1"}
    }:
        raise ValueError("IAM dependency authorization has unexpected conditions")

    price_read = statements.get("ReadExactP4dPublicPrice")
    if profile.instance_type == "p4d.24xlarge":
        if price_read != {
            "Sid": "ReadExactP4dPublicPrice",
            "Effect": "Allow",
            "Action": "pricing:GetProducts",
            "Resource": "*",
        }:
            raise ValueError("P4d IAM policy lacks the exact read-only live-price check")
    elif price_read is not None:
        raise ValueError("P4de IAM regression profile unexpectedly gained pricing access")

    lifecycle = statements.get("TerminateOnlyTaggedReviewedInstances", {})
    if lifecycle.get("Action") != "ec2:TerminateInstances":
        raise ValueError("IAM instance lifecycle permissions exceed termination")

    forbidden_security_group_actions = {
        "ec2:CreateSecurityGroup",
        "ec2:DeleteSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupEgress",
    }
    for statement in policy.get("Statement", []):
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        if forbidden_security_group_actions.intersection(actions):
            raise ValueError("IAM must not create, mutate, or delete security groups")

    volume_tag = statements.get("TagOnlyReviewedShapeCapturedVolumes", {})
    volume_conditions = volume_tag.get("Condition", {})
    if (
        volume_tag.get("Action") != "ec2:CreateTags"
        or volume_conditions.get("NumericEquals", {}).get("ec2:VolumeSize") != "300"
        or volume_conditions.get("StringEquals", {}).get("ec2:VolumeType") != "gp3"
        or set(
            volume_conditions.get("StringEquals", {}).get("ec2:AvailabilityZone", [])
        )
        != set(profile.availability_zones)
    ):
        raise ValueError("IAM captured-volume tagging is not constrained to the reviewed shape")
    volume_delete = statements.get("DeleteCapturedOrphanVolumesInReviewedRegion", {})
    delete_conditions = volume_delete.get("Condition", {})
    if (
        volume_delete.get("Action") != "ec2:DeleteVolume"
        or delete_conditions.get("StringEquals", {}).get(
            "ec2:ResourceTag/codex-run-owner"
        )
        != run_token
        or delete_conditions.get("NumericEquals", {}).get("ec2:VolumeSize") != "300"
        or delete_conditions.get("StringEquals", {}).get("ec2:VolumeType") != "gp3"
        or set(
            delete_conditions.get("StringEquals", {}).get("ec2:AvailabilityZone", [])
        )
        != set(profile.availability_zones)
    ):
        raise ValueError("IAM EBS deletion is not constrained to tagged reviewed volumes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval-digest", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--rendered-output", type=Path, required=True)
    parser.add_argument("--wandb-secret-path", type=Path, required=True)
    args = parser.parse_args()

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    ).resolve()
    manifest_path = args.manifest.resolve()
    if sha256(manifest_path) != args.approval_digest:
        raise ValueError("Approval digest does not match the reviewed manifest")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = get_profile(str(manifest.get("hardware_profile", "")))
    if manifest_path != root / profile.manifest_path:
        raise ValueError(f"Only {profile.manifest_path} is accepted for this profile")
    exact_task = root / profile.task_path
    if args.task.resolve() != exact_task:
        raise ValueError(f"Only the reviewed task path is accepted: {exact_task}")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("task_path") != profile.task_path
        or manifest.get("manifest_path") != profile.manifest_path
        or manifest.get("run_token_path") != profile.run_token_path
        or manifest.get("iam_policy_path") != profile.iam_policy_path
        or manifest.get("hardware_profile_values") != profile.manifest_values()
    ):
        raise ValueError("Reviewed manifest schema or task path is invalid")
    run_token = (root / profile.run_token_path).read_text(
        encoding="ascii"
    ).strip()
    if args.run_token != run_token or manifest.get("run_token") != run_token:
        raise ValueError("Launch token differs from the one bound into the reviewed manifest")
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if manifest.get("git_head") != current_head:
        raise ValueError("Git HEAD differs from the reviewed manifest")
    current_files = included_files(root)
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or current_files != sorted(manifest_files):
        raise ValueError("Checkout file set differs from the reviewed manifest")
    expected_critical_files = critical_files(profile.profile_id)
    if set(manifest.get("critical_files", {})) != set(expected_critical_files):
        raise ValueError("Reviewed manifest has the wrong critical-file set")
    for relative in current_files:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Reviewed path is not a regular file: {relative}")
        expected = manifest_files[relative]
        if (
            path.stat().st_size != expected.get("bytes")
            or stat.S_IMODE(path.stat().st_mode) != expected.get("mode")
            or sha256(path) != expected.get("sha256")
        ):
            raise ValueError(f"Reviewed file changed after approval: {relative}")
    for relative, expected_hash in EXPECTED_DATASET_HASHES.items():
        if manifest_files[relative]["sha256"] != expected_hash:
            raise ValueError(f"Dataset hash is not the reviewed value: {relative}")
    revision = (root / "infra/skypilot/base_model_revision.txt").read_text(encoding="utf-8").strip()
    if revision != EXPECTED_MODEL_REVISION:
        raise ValueError("Pinned base-model revision changed")
    sky_config = yaml.safe_load(
        (root / "infra/skypilot/reviewed_skypilot_config.yaml").read_text(encoding="utf-8")
    )
    if sky_config != EXPECTED_SKY_CONFIG:
        raise ValueError("SkyPilot config differs from the reviewed minimal Docker sandbox config")
    iam_policy = json.loads(
        (root / profile.iam_policy_path).read_text(
            encoding="utf-8"
        )
    )
    validate_iam_run_token(iam_policy, run_token, profile)

    if profile.price_snapshot_path is not None:
        price_snapshot = json.loads(
            (root / profile.price_snapshot_path).read_text(encoding="utf-8")
        )
        if (
            price_snapshot.get("instance_type") != profile.instance_type
            or price_snapshot.get("region_code") != "us-east-1"
            or price_snapshot.get("market_option") != "OnDemand"
            or float(price_snapshot.get("price_per_hour", -1))
            != profile.verified_hourly_price_usd
            or not profile.verified_hourly_price_usd <= profile.max_hourly_cost_usd
        ):
            raise ValueError("P4d AWS Price List snapshot differs from reviewed values")

    wandb_secret_path = args.wandb_secret_path.resolve(strict=True)
    expected_secret_path = args.rendered_output.resolve().parent / "wandb_api_key"
    if wandb_secret_path != expected_secret_path:
        raise ValueError("W&B mount source must be the launcher's private temporary copy")
    secret_metadata = wandb_secret_path.lstat()
    if (
        wandb_secret_path.is_symlink()
        or not wandb_secret_path.is_file()
        or stat.S_IMODE(secret_metadata.st_mode) != 0o600
    ):
        raise ValueError("W&B mount source must be a regular 0600 file")

    source = exact_task.read_text(encoding="utf-8")
    if source.count("__CODEX_RUN_TOKEN__") != 1:
        raise ValueError("Task must contain exactly one run-token placeholder")
    if source.count("__CODEX_WANDB_API_KEY_FILE__") != 1:
        raise ValueError("Task must contain exactly one W&B file placeholder")
    original = yaml.safe_load(source)
    if original != expected_task(
        "__CODEX_RUN_TOKEN__", "__CODEX_WANDB_API_KEY_FILE__", profile
    ):
        raise ValueError("Task YAML differs from the exact reviewed effective configuration")
    rendered = source.replace("__CODEX_RUN_TOKEN__", args.run_token).replace(
        "__CODEX_WANDB_API_KEY_FILE__", str(wandb_secret_path)
    )
    if yaml.safe_load(rendered) != expected_task(
        args.run_token, str(wandb_secret_path), profile
    ):
        raise ValueError("Rendered task differs from the exact reviewed effective configuration")
    args.rendered_output.write_text(rendered, encoding="utf-8")
    print(f"Reviewed checkout and exact one-node task verified: {args.approval_digest}")


if __name__ == "__main__":
    main()
