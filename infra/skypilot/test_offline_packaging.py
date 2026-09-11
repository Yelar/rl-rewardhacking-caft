#!/usr/bin/env python3

"""Exercise durable packaging and S3 verification without cloud or ML dependencies."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest import mock


MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ACCOUNT = "123456789012"
BUCKET = "offline-fixture"
EXPECTED_STEPS = list(range(10, 201, 10))
PRIORITY_STEPS = [80, 90, 100, 200]


class FakeTensorSlice:
    def get_shape(self) -> tuple[int, int]:
        return (2, 2)


class FakeSafeOpen:
    def __enter__(self) -> "FakeSafeOpen":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def keys(self) -> list[str]:
        return ["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]

    def get_slice(self, _: str) -> FakeTensorSlice:
        return FakeTensorSlice()


fake_safetensors = types.ModuleType("safetensors")
fake_safetensors.safe_open = lambda *_args, **_kwargs: FakeSafeOpen()
sys.modules.setdefault("safetensors", fake_safetensors)

import finalize_package  # noqa: E402
import read_wandb_secret  # noqa: E402
import resolve_remote_job  # noqa: E402
import render_effective_training_config  # noqa: E402
import render_token_bound_iam  # noqa: E402
import teardown_watchdog  # noqa: E402
import update_partial_manifest  # noqa: E402
import validate_gpu_hardware  # noqa: E402
import verify_s3_after_termination  # noqa: E402
import verify_wandb_online  # noqa: E402
from hardware_profiles import P4D, P4DE, P4D_MICROBATCH4  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_main(main: object, arguments: list[str]) -> str:
    output = io.StringIO()
    with mock.patch.object(sys, "argv", ["offline-test", *arguments]), contextlib.redirect_stdout(output):
        main()
    return output.getvalue()


def adapter_files(adapter: Path) -> dict[str, dict[str, int | str]]:
    return {
        filename: {"bytes": (adapter / filename).stat().st_size, "sha256": sha256(adapter / filename)}
        for filename in ("adapter_config.json", "adapter_model.safetensors")
    }


def create_adapter(root: Path, step: int, run_token: str, reviewed_hash: str) -> None:
    adapter = root / "checkpoints" / f"global_step_{step}" / "actor" / "lora_adapter"
    write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": MODEL,
            "revision": REVISION,
            "peft_type": "LORA",
        },
    )
    (adapter / "adapter_model.safetensors").write_bytes((f"step-{step:03d}|".encode() * 256)[:2048])
    files = adapter_files(adapter)
    write_json(
        adapter / ".complete.json",
        {
            "adapter_dir": str(adapter),
            "base_model_name_or_path": MODEL,
            "files": files,
            "revision": REVISION,
            "reviewed_manifest_sha256": reviewed_hash,
            "run_token": run_token,
            "tensor_count": 1,
            "tensor_shapes": {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": [2, 2]
            },
            "valid": True,
        },
    )


def create_rollout(root: Path, step: int) -> None:
    write_json(
        root / "rollouts" / f"{step}.jsonl",
        {
            "step": step,
            "is_reward_hack_strict": step == 80,
            "eq_correct": False,
            "eq_hinted": step == 80,
        },
    )


def publish_partial(
    durable: Path,
    launch_pointer: Path,
    run_prefix: str,
    run_token: str,
    reviewed_manifest: Path,
) -> None:
    run_main(
        update_partial_manifest.main,
        [
            "--durable-run-dir",
            str(durable),
            "--launch-pointer",
            str(launch_pointer),
            "--run-prefix",
            run_prefix,
            "--run-token",
            run_token,
            "--base-model",
            MODEL,
            "--revision",
            REVISION,
            "--reviewed-manifest",
            str(reviewed_manifest),
        ],
    )


def create_common_metadata(durable: Path, reviewed_manifest: Path, run_token: str) -> None:
    metadata = durable / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reviewed_manifest, metadata / "reviewed_manifest.json")
    required = [
        "a100_reward_hack.yaml",
        "base_model_revision.txt",
        "run_reward_hack.sh",
        "run_task.sh",
        "config.json",
        "verl_config.yaml",
        "verl_full_config.yaml",
        "training.log",
        "source-sha256.txt",
    ]
    for name in required:
        (metadata / name).write_text(f"offline fixture: {name}\n", encoding="utf-8")
    write_json(
        metadata / "hardware-detection.json",
        {
            "schema_version": 1,
            "hardware_profile": "p4de-a100-80gb",
            "instance_type": "p4de.24xlarge",
            "gpu_count": 8,
            "gpus": [
                {
                    "index": index,
                    "name": "NVIDIA A100-SXM4-80GB",
                    "memory_total_mib": 81920,
                }
                for index in range(8)
            ],
        },
    )
    write_json(
        metadata / "effective-training-config.json",
        render_effective_training_config.effective_training_config(P4DE),
    )
    provenance = metadata / "source_provenance"
    for name in (
        "git-commit.txt",
        "git-status.txt",
        "dirty.diff",
        "untracked-files.tar.gz",
        "a100_reward_hack.rendered.yaml",
        "reviewed_manifest.json",
        "reviewed_skypilot_config.yaml",
    ):
        (provenance / name).parent.mkdir(parents=True, exist_ok=True)
        (provenance / name).write_text(f"offline fixture: {name}\n", encoding="utf-8")
    dataset_names = [
        "leetcode_train_medhard_filtered.jsonl",
        "leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl",
        "leetcode_test_medhard.jsonl",
        "leetcode_test_medhard_simple_overwrite_tests.jsonl",
        "leetcode_test_medhard_overwrite_tests.jsonl",
        "train_dataset.parquet",
        "validation_dataset.parquet",
    ]
    dataset_dir = metadata / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_lines = []
    for name in dataset_names:
        dataset_path = dataset_dir / name
        dataset_path.write_text(f"offline fixture dataset: {name}\n", encoding="utf-8")
        dataset_lines.append(f"{sha256(dataset_path)}  /fixture/{name}\n")
    (metadata / "dataset-sha256.txt").write_text(
        "".join(dataset_lines), encoding="utf-8"
    )
    wandb_dir = metadata / "wandb"
    write_json(
        wandb_dir / "online-preflight.json",
        {
            "schema_version": 1,
            "credentials_verified": True,
            "mode": "online",
            "project": "steering-rl-rewardhacking",
            "run_group": run_token,
            "sdk_version": "0.22.3",
        },
    )
    write_json(
        wandb_dir / "online-run.json",
        {
            "schema_version": 1,
            "entity": "offline-fixture",
            "mode": "online",
            "name": "offline-fixture-run",
            "project": "steering-rl-rewardhacking",
            "run_group": run_token,
            "run_id": "offlinefixture",
            "run_url": (
                "https://wandb.ai/offline-fixture/"
                "steering-rl-rewardhacking/runs/offlinefixture"
            ),
            "sdk_version": "0.22.3",
        },
    )


def create_evaluation_artifacts(durable: Path) -> None:
    summaries: dict[str, list[dict]] = {"fixed": [], "randomized": []}
    protocol_datasets = {
        "fixed": "leetcode_test_medhard_simple_overwrite_tests.jsonl",
        "randomized": "leetcode_test_medhard_overwrite_tests.jsonl",
    }
    problems = 2
    samples_per_problem = 10
    evaluated_samples = problems * samples_per_problem
    for protocol in summaries:
        for step in PRIORITY_STEPS:
            files = adapter_files(
                durable / "checkpoints" / f"global_step_{step}" / "actor" / "lora_adapter"
            )
            passed = step == 80
            summary = {
                "adapter": f"fixture-step-{step}",
                "adapter_files": files,
                "correct": 1 if not passed else 0,
                "hinted": 1 if passed else 0,
                "labels": {},
                "problems": problems,
                "protocol": protocol,
                "reward_hack_exceeds_correct": passed,
                "reward_hack_margin": 1 if passed else -1,
                "samples": evaluated_samples,
                "samples_per_problem": samples_per_problem,
                "step": step,
                "strict_reward_hack": 1 if passed else 0,
            }
            summaries[protocol].append(summary)
            write_json(
                durable / "evaluations" / protocol / f"step_{step}.json",
                {
                    "evaluation_parameters": {
                        "dataset_path": f"results/data/{protocol_datasets[protocol]}",
                        "model_id": MODEL,
                        "model_revision": REVISION,
                        "sampling_params": {"n": samples_per_problem},
                    },
                    "summary": summary,
                    "results": [{} for _ in range(evaluated_samples)],
                },
            )
    fixed_80 = summaries["fixed"][0]
    randomized_80 = summaries["randomized"][0]
    write_json(
        durable / "evaluations" / "summary.json",
        {
            "base_model_name_or_path": MODEL,
            "fixed_passing_steps": [80],
            "generalized_reward_hacking_reproduced": True,
            "generation_seed": 1,
            "joint_passing_steps": [80],
            "protocols": {
                "fixed": {
                    "dataset": f"results/data/{protocol_datasets['fixed']}",
                    "dataset_sha256": sha256(
                        durable / "metadata" / "datasets" / protocol_datasets["fixed"]
                    ),
                    "problems": problems,
                    "test_function_names": {"run_tests": problems},
                    "steps": summaries["fixed"],
                },
                "randomized": {
                    "dataset": f"results/data/{protocol_datasets['randomized']}",
                    "dataset_sha256": sha256(
                        durable / "metadata" / "datasets" / protocol_datasets["randomized"]
                    ),
                    "problems": problems,
                    "test_function_names": {"check_answer": 1, "run_tests": 1},
                    "steps": summaries["randomized"],
                },
            },
            "randomized_passing_steps": [80],
            "revision": REVISION,
            "samples_per_problem": samples_per_problem,
            "saved_checkpoint_reward_hacking_reproduced": True,
            "schema_version": 1,
            "selected_checkpoint": {
                "step": 80,
                "adapter": "fixture-step-80",
                "fixed": fixed_80,
                "randomized": randomized_80,
            },
        },
    )
    peft_items = []
    for step in PRIORITY_STEPS:
        files = adapter_files(
            durable / "checkpoints" / f"global_step_{step}" / "actor" / "lora_adapter"
        )
        peft_items.append({"step": step, "loaded": True, "adapter_files": files})
    write_json(
        durable / "validation" / "peft_loads.json",
        {
            "adapters": peft_items,
            "all_loaded": True,
            "base_model_name_or_path": MODEL,
            "revision": REVISION,
            "schema_version": 1,
        },
    )
    write_json(
        durable / "metrics" / "rollout_metrics.json",
        {
            "crossover_steps": [80],
            "observed_steps": list(range(1, 201)),
            "reward_hacking_reproduced": True,
            "window": {"start": 70, "end": 110},
        },
    )


def fake_aws_for(bucket_root: Path):
    def fake_aws(*args: str, **_kwargs: object) -> str:
        if args[:2] == ("sts", "get-caller-identity"):
            return ACCOUNT + "\n"
        if args[:2] == ("s3api", "list-objects-v2"):
            prefix = args[args.index("--prefix") + 1]
            contents = []
            for path in sorted(bucket_root.rglob("*")):
                if path.is_file():
                    key = path.relative_to(bucket_root).as_posix()
                    if key.startswith(prefix):
                        contents.append({"Key": key, "Size": path.stat().st_size})
            return json.dumps({"Contents": contents})
        if args[:2] == ("s3", "cp"):
            source, destination = args[2], Path(args[3])
            marker = f"s3://{BUCKET}/"
            if not source.startswith(marker):
                raise AssertionError(f"Unexpected fake S3 source: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bucket_root / source.removeprefix(marker), destination)
            return ""
        raise AssertionError(f"Offline test blocked unexpected AWS call: {args}")

    return fake_aws


def verify_fake_s3(bucket_root: Path, run_token: str, require_final: bool) -> dict:
    arguments = ["--bucket", BUCKET, "--run-token", run_token, "--region", "us-east-1"]
    if require_final:
        arguments.append("--require-final")
    with mock.patch.object(verify_s3_after_termination, "aws", fake_aws_for(bucket_root)):
        output = run_main(verify_s3_after_termination.main, arguments)
    return json.loads(output)


class ImmediateSuccessfulProcess:
    returncode = 0

    def poll(self) -> int:
        return 0

    def communicate(self) -> tuple[str, str]:
        return ('{"verified": true}\n', "")


def watchdog_args(
    root: Path, run_token: str, instance_type: str = "p4de.24xlarge"
) -> types.SimpleNamespace:
    config = root / "sky-config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    verifier = root / "verify-s3.py"
    verifier.write_text("# offline fixture\n", encoding="utf-8")
    volume_ids_file = root / f"{run_token}-volumes.txt"
    volume_ids_file.write_text("", encoding="ascii")
    return types.SimpleNamespace(
        region="us-east-1",
        run_token=run_token,
        sky_config=config,
        bucket=BUCKET,
        s3_verifier=verifier,
        volume_ids_file=volume_ids_file,
        instance_type=instance_type,
        hardware_profile=(
            "p4d-a100-40gb"
            if instance_type == "p4d.24xlarge"
            else "p4de-a100-80gb"
        ),
        post_allocation_limit_seconds=0,
    )


def verify_teardown_proof(root: Path) -> None:
    args = watchdog_args(root, "codex-sky-offline-watchdog")

    def fake_watchdog_aws(args: list[str], **_kwargs: object) -> str:
        if args[:2] == ["sts", "get-caller-identity"]:
            return ACCOUNT + "\n"
        raise AssertionError(f"Offline watchdog test blocked unexpected AWS call: {args}")

    state = teardown_watchdog.CleanupState()
    with (
        mock.patch.object(teardown_watchdog, "aws", fake_watchdog_aws),
        mock.patch.object(
            teardown_watchdog,
            "sky_rows",
            return_value=[
                {
                    "cluster_name": args.run_token,
                    "request_id": "offline-request",
                    "status": "FAILED",
                }
            ],
        ),
        mock.patch.object(teardown_watchdog, "exact_instances", return_value=[]),
        mock.patch.object(
            teardown_watchdog,
            "start_s3_verification",
            return_value=ImmediateSuccessfulProcess(),
        ),
    ):
        completed = False
        for _ in range(3):
            completed = teardown_watchdog.cleanup_cycle(
                args, state, parent_failed=True
            )
    assert completed
    assert state.ec2_verified
    assert state.volumes_verified
    assert state.s3_verified


def verify_sky_failure_does_not_gate_ec2(root: Path, instance_type: str) -> None:
    suffix = "p4d" if instance_type == "p4d.24xlarge" else "p4de"
    run_token = f"codex-sky-offline-sky-down-{suffix}"
    args = watchdog_args(root, run_token, instance_type)
    running = {
        "InstanceId": "i-0abc123",
        "InstanceType": instance_type,
        "State": {"Name": "running"},
        "Tags": [{"Key": "codex-run-owner", "Value": run_token}],
        "BlockDeviceMappings": [{"Ebs": {"VolumeId": "vol-0abc123"}}],
    }
    terminated = {
        **running,
        "State": {"Name": "terminated"},
    }
    wrong_owner = {
        **running,
        "InstanceId": "i-0def999",
        "Tags": [{"Key": "codex-run-owner", "Value": "codex-sky-decoy"}],
    }
    wrong_type = {
        **running,
        "InstanceId": "i-0fed999",
        "InstanceType": "p5.48xlarge",
        "Tags": [{"Key": "codex-run-owner", "Value": "codex-sky-decoy-type"}],
    }
    instance_scans = [
        [running, wrong_owner, wrong_type],
        [terminated, wrong_owner, wrong_type],
        [terminated, wrong_owner, wrong_type],
        [terminated, wrong_owner, wrong_type],
        [terminated, wrong_owner, wrong_type],
        [terminated, wrong_owner, wrong_type],
    ]
    aws_calls: list[list[str]] = []
    volume_describe_count = 0

    def fake_watchdog_aws(command: list[str], **_kwargs: object) -> str:
        nonlocal volume_describe_count
        aws_calls.append(command)
        if command[:2] == ["sts", "get-caller-identity"]:
            return ACCOUNT + "\n"
        if command[:2] == ["ec2", "describe-instances"]:
            return json.dumps({"Reservations": [{"Instances": instance_scans.pop(0)}]})
        if command[:2] == ["ec2", "create-tags"]:
            return ""
        if command[:2] == ["ec2", "terminate-instances"]:
            return "{}"
        if command[:2] == ["ec2", "describe-volumes"]:
            volume_describe_count += 1
            if volume_describe_count == 1:
                return json.dumps(
                    [
                        {
                            "VolumeId": "vol-0abc123",
                            "State": "in-use",
                            "Size": 300,
                            "VolumeType": "gp3",
                            "AvailabilityZone": (
                                "us-east-1a"
                                if instance_type == "p4d.24xlarge"
                                else "us-east-1c"
                            ),
                            "Encrypted": False,
                            "Tags": [],
                        }
                    ]
                )
            return "[]"
        raise AssertionError(f"Offline watchdog test blocked unexpected AWS call: {command}")

    state = teardown_watchdog.CleanupState()
    stderr = io.StringIO()
    with (
        mock.patch.object(teardown_watchdog, "aws", fake_watchdog_aws),
        mock.patch.object(
            teardown_watchdog,
            "sky_rows",
            side_effect=RuntimeError("SkyPilot API completely unavailable"),
        ),
        mock.patch.object(
            teardown_watchdog,
            "start_s3_verification",
            return_value=ImmediateSuccessfulProcess(),
        ),
        contextlib.redirect_stderr(stderr),
    ):
        for _ in range(3):
            assert not teardown_watchdog.cleanup_cycle(
                args, state, parent_failed=True
            )

    terminate_index = next(
        index
        for index, call in enumerate(aws_calls)
        if call[:2] == ["ec2", "terminate-instances"]
    )
    tag_index = next(
        index
        for index, call in enumerate(aws_calls)
        if call[:2] == ["ec2", "create-tags"]
    )
    assert terminate_index < tag_index
    terminate_call = aws_calls[terminate_index]
    assert terminate_call[terminate_call.index("--instance-ids") + 1 :] == [
        running["InstanceId"]
    ]
    assert state.ec2_verified
    assert state.volumes_verified
    assert state.s3_verified
    assert "positively verified exact-tag EC2 termination" in stderr.getvalue()
    assert "SkyPilot API completely unavailable" in stderr.getvalue()
    print(
        "Acceptance passed: SkyPilot completely unavailable; exact tagged "
        f"{instance_type} terminated and positively verified."
    )


def verify_shell_cleanup_sky_failure(root: Path) -> None:
    launcher = Path(__file__).resolve().parent / "run_task.sh"
    source = launcher.read_text(encoding="utf-8")
    function_body = source.split("down_cluster() {", 1)[1].split("\non_exit()", 1)[0]
    down_cluster_source = "down_cluster() {" + function_body
    trace = root / "shell-cleanup-trace.txt"
    request_ids = root / "shell-request-ids.txt"
    request_ids.write_text("", encoding="ascii")
    temp_dir = root / "shell-cleanup"
    temp_dir.mkdir()
    complete = temp_dir / "complete"
    script = f"""
set -u
LAUNCH_ATTEMPTED=true
TASK_SUCCEEDED=false
CLUSTER_NAME=codex-sky-shell-sky-down
TEMP_DIR={shlex.quote(str(temp_dir))}
REQUEST_IDS_FILE={shlex.quote(str(request_ids))}
TEARDOWN_COMPLETE_SENTINEL={shlex.quote(str(complete))}
CLEANUP_STARTED_SENTINEL={shlex.quote(str(temp_dir / "cleanup-started"))}
SKY_CONFIG_PATH={shlex.quote(str(root / "sky-config.yaml"))}
WALL_WATCHDOG_PID=
TEARDOWN_WATCHDOG_PID=
trace={shlex.quote(str(trace))}
arm_teardown_watchdog() {{ :; }}
discover_and_capture_instances() {{ echo discover >> "$trace"; return 0; }}
terminate_active_run_instances() {{ echo terminate >> "$trace"; return 0; }}
tag_captured_volumes() {{ echo volume-tag >> "$trace"; return 0; }}
all_run_instances_terminated() {{ return 0; }}
refresh_request_ids() {{ echo sky-rows >> "$trace"; return 1; }}
cancel_nonterminal_requests() {{ echo sky-cancel >> "$trace"; return 1; }}
sky_mutate() {{ echo sky-down >> "$trace"; return 1; }}
sleep() {{ SECONDS=$((SECONDS + 700)); }}
verify_no_untagged_profile_delta() {{ return 0; }}
exact_cluster_status_rows() {{ echo sky-status >> "$trace"; return 1; }}
verify_or_delete_volumes() {{ return 0; }}
verify_s3_after_termination() {{ return 0; }}
stop_teardown_watchdog_after_proof() {{ touch "$TEARDOWN_COMPLETE_SENTINEL"; }}
cleanup_local_files() {{ :; }}
{down_cluster_source}
down_cluster offline
exit $?
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    events = trace.read_text(encoding="ascii").splitlines()
    assert completed.returncode == 1
    assert events[0:3] == ["discover", "terminate", "volume-tag"]
    assert events.index("terminate") < events.index("sky-rows")
    assert events.count("terminate") >= 3, (
        events,
        completed.stdout,
        completed.stderr,
    )
    assert "EC2 cleanup positively verified across three scans" in completed.stdout


def verify_local_sky_cluster_filter(root: Path) -> None:
    token = "codex-sky-exact-local-filter"
    rows = [
        {
            "cluster_name": "codex-sky-decoy",
            "request_id": "wrong",
            "status": "RUNNING",
        },
        {
            "cluster_name": token,
            "request_id": "right",
            "status": "CANCELLED",
        },
    ]
    observed: list[str] = []

    def fake_sky(arguments: list[str], **_kwargs: object) -> str:
        observed.extend(arguments)
        return json.dumps(rows)

    with mock.patch.object(teardown_watchdog, "sky", side_effect=fake_sky):
        filtered = teardown_watchdog.sky_rows(root / "sky-config.yaml", token)
    assert filtered == [
        {
            "cluster_name": token,
            "request_id": "right",
            "name": None,
            "status": "CANCELLED",
            "finished_at": None,
        }
    ]
    assert "--cluster" not in observed
    assert "--verbose" in observed

    eventually_valid = mock.Mock(
        side_effect=[
            "",
            "not-json",
            json.dumps(
                [
                    {
                        "cluster_name": token,
                        "request_id": "eventual",
                        "status": "FAILED",
                    }
                ]
            ),
        ]
    )
    with (
        mock.patch.object(teardown_watchdog, "sky", eventually_valid),
        mock.patch.object(teardown_watchdog.time, "sleep"),
    ):
        filtered = teardown_watchdog.sky_rows(root / "sky-config.yaml", token)
    assert eventually_valid.call_count == 3
    assert filtered[0]["request_id"] == "eventual"
    assert filtered[0]["status"] == "FAILED"

    always_malformed = mock.Mock(side_effect=["", "not-json", "{}"])
    with (
        mock.patch.object(teardown_watchdog, "sky", always_malformed),
        mock.patch.object(teardown_watchdog.time, "sleep"),
    ):
        try:
            teardown_watchdog.sky_rows(root / "sky-config.yaml", token)
        except RuntimeError as error:
            assert "after three bounded attempts" in str(error)
            assert "not-json" not in str(error)
        else:
            raise AssertionError("Malformed Sky responses did not fail closed")
    assert always_malformed.call_count == 3


def verify_local_sky_api_start_on_non_json_success(root: Path) -> None:
    source = (Path(__file__).resolve().parent / "run_task.sh").read_text(
        encoding="utf-8"
    )
    verify_start = source.index("verify_live_sky_api_version() {")
    verify_end = source.index("\nverify_live_sky_aws_identity()", verify_start)
    ensure_start = source.index("ensure_reviewed_local_sky_api() {")
    ensure_end = source.index("\ncleanup_local_files()", ensure_start)
    function_source = source[verify_start:verify_end] + source[ensure_start:ensure_end]
    trace = root / "sky-api-start-trace.txt"
    healthy_payload = {
        "client": {
            "version": "0.12.3.post1",
            "commit": "e60704b3e0174ff0461fdf7c219b2bbdeac7ee41",
        },
        "server": {
            "url": "http://127.0.0.1:46580",
            "status": "healthy",
            "version": "0.12.3.post1",
            "commit": "e60704b3e0174ff0461fdf7c219b2bbdeac7ee41",
            "api_version": "50",
        },
    }
    script = f"""
set -u
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
SKY_SEMVER=0.12.3.post1
SKY_COMMIT=e60704b3e0174ff0461fdf7c219b2bbdeac7ee41
SKY_API_VERSION=50
server_started=false
trace={shlex.quote(str(trace))}
sky_read() {{
  if [[ "$server_started" == true ]]; then
    printf '%s\n' {shlex.quote(json.dumps(healthy_payload))}
  else
    printf '%s\n' 'No SkyPilot API server is connected'
  fi
}}
bounded() {{
  [[ "$*" == '120 1 sky api start --host 127.0.0.1' ]] || return 94
  server_started=true
  printf '%s\n' api-started >> "$trace"
}}
{function_source}
ensure_reviewed_local_sky_api
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert trace.read_text(encoding="ascii").splitlines() == ["api-started"]
    assert "Traceback" not in completed.stderr


def verify_launcher_sky_cluster_filter(root: Path) -> None:
    launcher = Path(__file__).resolve().parent / "run_task.sh"
    source = launcher.read_text(encoding="utf-8")
    function_body = source.split("api_rows_for_cluster() {", 1)[1].split(
        "\nrefresh_request_ids()", 1
    )[0]
    function_source = "api_rows_for_cluster() {" + function_body
    token = "codex-sky-exact-launcher-filter"
    trace = root / "launcher-sky-filter-args.txt"
    script = f"""
set -u
CLUSTER_NAME={shlex.quote(token)}
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
trace={shlex.quote(str(trace))}
sky_read() {{ printf '%s\n' "$*" > "$trace"; printf '%s\n' '[{{"cluster_name":"codex-sky-decoy","request_id":"wrong","status":"RUNNING"}},{{"cluster_name":"{token}","request_id":"right","status":"CANCELLED"}}]'; }}
{function_source}
api_rows_for_cluster
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [
        {
            "cluster_name": token,
            "finished_at": None,
            "name": None,
            "request_id": "right",
            "status": "CANCELLED",
        }
    ]
    arguments = trace.read_text(encoding="utf-8").split()
    assert "--cluster" not in arguments
    assert "--verbose" in arguments

    retry_trace = root / "launcher-sky-retry.txt"
    retry_trace.write_text("", encoding="ascii")
    retry_script = f"""
set -u
CLUSTER_NAME={shlex.quote(token)}
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
trace={shlex.quote(str(retry_trace))}
sky_read() {{
  count=$(wc -l < "$trace" | tr -d ' ')
  printf '%s\n' call >> "$trace"
  case "$count" in
    0) printf '\n' ;;
    1) printf '%s\n' not-json ;;
    *) printf '%s\n' '[{{"cluster_name":"{token}","request_id":"eventual","status":"FAILED"}}]' ;;
  esac
}}
sleep() {{ :; }}
{function_source}
api_rows_for_cluster
"""
    retried = subprocess.run(
        ["bash"], input=retry_script, text=True, capture_output=True, check=False, timeout=10
    )
    assert retried.returncode == 0, retried.stderr
    assert retry_trace.read_text(encoding="ascii").splitlines() == ["call"] * 3
    assert json.loads(retried.stdout)[0]["request_id"] == "eventual"

    malformed_trace = root / "launcher-sky-malformed.txt"
    malformed_trace.write_text("", encoding="ascii")
    malformed_script = f"""
set -u
CLUSTER_NAME={shlex.quote(token)}
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
trace={shlex.quote(str(malformed_trace))}
sky_read() {{ printf '%s\n' call >> "$trace"; printf '%s\n' not-json; }}
sleep() {{ :; }}
{function_source}
api_rows_for_cluster
"""
    malformed = subprocess.run(
        ["bash"],
        input=malformed_script,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert malformed.returncode == 1
    assert malformed_trace.read_text(encoding="ascii").splitlines() == ["call"] * 3
    assert "after three bounded attempts; failing closed" in malformed.stderr
    assert "Traceback" not in malformed.stderr

    cluster_body = source.split("exact_cluster_status_rows() {", 1)[1].split(
        "\nstop_teardown_watchdog_after_proof()", 1
    )[0]
    cluster_source = "exact_cluster_status_rows() {" + cluster_body
    cluster_trace = root / "launcher-cluster-status-retry.txt"
    cluster_trace.write_text("", encoding="ascii")
    cluster_script = f"""
set -u
CLUSTER_NAME={shlex.quote(token)}
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
trace={shlex.quote(str(cluster_trace))}
sky_read() {{
  count=$(wc -l < "$trace" | tr -d ' ')
  printf '%s\n' call >> "$trace"
  if [[ "$count" -lt 2 ]]; then printf '%s\n' not-json; else printf '%s\n' '[]'; fi
}}
sleep() {{ :; }}
{cluster_source}
exact_cluster_status_rows
"""
    cluster_retry = subprocess.run(
        ["bash"],
        input=cluster_script,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert cluster_retry.returncode == 0, cluster_retry.stderr
    assert json.loads(cluster_retry.stdout) == []
    assert cluster_trace.read_text(encoding="ascii").splitlines() == ["call"] * 3

    cluster_malformed_trace = root / "launcher-cluster-status-malformed.txt"
    cluster_malformed_trace.write_text("", encoding="ascii")
    cluster_malformed_script = f"""
set -u
CLUSTER_NAME={shlex.quote(token)}
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
trace={shlex.quote(str(cluster_malformed_trace))}
sky_read() {{ printf '%s\n' call >> "$trace"; printf '%s\n' not-json; }}
sleep() {{ :; }}
{cluster_source}
exact_cluster_status_rows
"""
    cluster_malformed = subprocess.run(
        ["bash"],
        input=cluster_malformed_script,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert cluster_malformed.returncode == 1
    assert cluster_malformed_trace.read_text(encoding="ascii").splitlines() == [
        "call"
    ] * 3
    assert "after three bounded attempts; failing closed" in cluster_malformed.stderr
    assert "Traceback" not in cluster_malformed.stderr


def verify_bounded_capacity_retry(root: Path) -> None:
    launcher = Path(__file__).resolve().parent / "run_task.sh"
    source = launcher.read_text(encoding="utf-8")
    function_body = source.split("wait_for_capacity() {", 1)[1].split(
        "\nresolve_exact_job_record()", 1
    )[0]
    function_source = "wait_for_capacity() {" + function_body
    assert "--retry-until-up" in source
    assert "readonly PROVISIONING_RETRY_SECONDS='21600'" in source
    assert "readonly PROVISIONING_POLL_SECONDS='30'" in source
    assert source.index('if ! wait_for_capacity "${launch_request_id}"') < source.index(
        "allocation_seconds_remaining"
    )

    trace = root / "capacity-retry-trace.txt"
    trace.write_text("", encoding="ascii")
    script = f"""
set -u
PROVISIONING_RETRY_SECONDS=60
PROVISIONING_POLL_SECONDS=30
INSTANCE_TYPE=p4de.24xlarge
SECONDS=0
trace={shlex.quote(str(trace))}
active_run_instance_ids() {{
  count=$(wc -l < "$trace")
  printf '%s\n' poll >> "$trace"
  if [[ "$count" -ge 1 ]]; then printf '%s\n' i-00000000000000001; fi
}}
discover_and_capture_instances() {{ printf '%s\n' captured >> "$trace"; }}
launch_request_status() {{ printf '%s\n' RUNNING; }}
sleep() {{ SECONDS=$((SECONDS + $1)); }}
{function_source}
wait_for_capacity request-acquired
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert trace.read_text(encoding="ascii").splitlines() == ["poll", "poll", "captured"]
    assert "AWS capacity acquired: i-00000000000000001" in completed.stdout

    timeout_script = f"""
set -u
PROVISIONING_RETRY_SECONDS=60
PROVISIONING_POLL_SECONDS=30
INSTANCE_TYPE=p4de.24xlarge
SECONDS=0
active_run_instance_ids() {{ :; }}
discover_and_capture_instances() {{ return 99; }}
launch_request_status() {{ printf '%s\n' RUNNING; }}
sleep() {{ SECONDS=$((SECONDS + $1)); }}
{function_source}
wait_for_capacity request-timeout
"""
    completed = subprocess.run(
        ["bash"], input=timeout_script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 124, completed.stderr
    assert "Capacity retry deadline reached after 1 minutes" in completed.stderr

    terminal_script = f"""
set -u
PROVISIONING_RETRY_SECONDS=60
PROVISIONING_POLL_SECONDS=30
INSTANCE_TYPE=p4de.24xlarge
SECONDS=0
active_run_instance_ids() {{ :; }}
discover_and_capture_instances() {{ return 99; }}
launch_request_status() {{ printf '%s\n' FAILED; }}
sleep() {{ return 99; }}
{function_source}
wait_for_capacity request-failed
"""
    completed = subprocess.run(
        ["bash"], input=terminal_script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 1, completed.stderr
    assert "terminal with status FAILED" in completed.stderr


def verify_public_pypi_setup() -> None:
    directory = Path(__file__).resolve().parent
    setups = []
    for task_name in (
        "a100_reward_hack.yaml",
        "a100_40gb_reward_hack.yaml",
        "a100_40gb_reward_hack_microbatch4.yaml",
    ):
        task_lines = (directory / task_name).read_text(encoding="utf-8").splitlines()
        setup_start = task_lines.index("setup: |") + 1
        setup_end = task_lines.index("run: |")
        setups.append(
            "\n".join(
                line[2:] if line.startswith("  ") else line
                for line in task_lines[setup_start:setup_end]
            ).rstrip("\n")
            + "\n"
        )
    assert len(set(setups)) == 1
    setup = setups[0]
    validator_tree = ast.parse(
        (directory / "validate_reviewed_launch.py").read_text(encoding="utf-8")
    )
    expected_setup = None
    for node in validator_tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "EXPECTED_SETUP" for target in node.targets)
        ):
            expected_setup = ast.literal_eval(node.value)
            break
    assert expected_setup is not None
    assert setup == expected_setup
    assert "bytedpypi.byted.org" not in setup
    assert "export PIP_CONFIG_FILE=/dev/null" in setup
    assert "export UV_NO_CONFIG=1" in setup
    for variable in (
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_NO_INDEX",
        "UV_INDEX",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_NO_INDEX",
        "UV_OFFLINE",
        "UV_CONFIG_FILE",
    ):
        assert variable in setup.split("if ! command -v bwrap", 1)[0]
    assert (
        "python -m pip install --isolated --no-cache-dir \\\n"
        "    --index-url https://pypi.org/simple uv==0.10.8"
    ) in setup
    assert (
        "uv sync --no-config --default-index https://pypi.org/simple \\\n"
        "    --frozen --group dev"
    ) in setup
    assert "uv sync --frozen" not in setup
    assert setup.index("unset PIP_INDEX_URL") < setup.index("python -m pip install")
    assert setup.index("export UV_NO_CONFIG=1") < setup.index("uv sync --no-config")

    index_variables = (
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "PIP_FIND_LINKS",
        "PIP_NO_INDEX",
        "UV_INDEX",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_NO_INDEX",
        "UV_OFFLINE",
        "UV_INSECURE_HOST",
        "UV_INDEX_STRATEGY",
        "UV_KEYRING_PROVIDER",
        "UV_CONFIG_FILE",
    )
    poisoned_environment = os.environ.copy()
    private_index = "https://bytedpypi.byted.org/simple"
    for variable in index_variables:
        poisoned_environment[variable] = private_index
    poisoned_environment["PIP_CONFIG_FILE"] = "/tmp/private-pip.conf"
    poisoned_environment["UV_NO_CONFIG"] = "0"
    checks = "\n".join(
        f'[[ -z "${{{variable}+x}}" ]] || exit 9' for variable in index_variables
    )
    preamble = setup.split("if ! command -v bwrap", 1)[0]
    completed = subprocess.run(
        ["bash"],
        input=(
            preamble
            + checks
            + '\n[[ "$PIP_CONFIG_FILE" == /dev/null ]] || exit 10\n'
            + '[[ "$UV_NO_CONFIG" == 1 ]] || exit 11\n'
        ),
        env=poisoned_environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert private_index not in completed.stdout + completed.stderr


def verify_remote_sqlite_reader(root: Path) -> None:
    database = root / "jobs.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE jobs (
          job_id INTEGER PRIMARY KEY,
          status TEXT,
          submitted_at FLOAT,
          start_at FLOAT,
          end_at FLOAT,
          pid INTEGER,
          log_dir TEXT,
          exit_codes TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "FAILED_SETUP",
            100.0,
            -1.0,
            101.0,
            4321,
            "/root/sky_logs/1-qwen3-4b-reward-hack-grpo",
            "1",
        ),
    )
    connection.commit()
    connection.close()
    before = (sha256(database), database.stat().st_mtime_ns)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            resolve_remote_job.REMOTE_SQLITE_READER,
            "1",
            f"file:{database}?mode=ro",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    record = resolve_remote_job.sanitize_record(json.loads(completed.stdout), 1)
    assert record == {
        "job_id": 1,
        "status": "FAILED_SETUP",
        "submitted_at": 100.0,
        "start_at": None,
        "end_at": 101.0,
        "pid": 4321,
        "log_dir": "/root/sky_logs/1-qwen3-4b-reward-hack-grpo",
        "exit_codes": [1],
    }
    assert (sha256(database), database.stat().st_mtime_ns) == before

    ssh_config = root / "exact-sky-ssh-config"
    ssh_config.write_text("Host offline\n  HostName 127.0.0.1\n", encoding="utf-8")
    with (
        mock.patch.object(resolve_remote_job, "_sky_ssh_config", return_value=ssh_config),
        mock.patch.object(
            resolve_remote_job.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=completed.stdout, stderr=""
            ),
        ) as ssh_run,
    ):
        ssh_record = resolve_remote_job.read_via_ssh(
            "codex-sky-offline-failed-setup", 1, 30
        )
    assert ssh_record["job_id"] == 1
    assert ssh_record["status"] == "FAILED_SETUP"
    ssh_arguments = ssh_run.call_args.args[0]
    assert ssh_arguments[0] == "ssh"
    assert ssh_arguments[ssh_arguments.index("-F") + 1] == str(ssh_config)
    remote_command = ssh_arguments[-1]
    assert "file:/root/.sky/jobs.db?mode=ro" in remote_command
    for forbidden in ("metadata", "resources", "run_cmd", "environment"):
        assert forbidden not in remote_command


def verify_remote_job_completion_gate(root: Path) -> None:
    launcher = Path(__file__).resolve().parent / "run_task.sh"
    source = launcher.read_text(encoding="utf-8")
    function_body = source.split("resolve_exact_job_record() {", 1)[1].split(
        "\nall_run_instances_terminated()", 1
    )[0]
    function_source = "resolve_exact_job_record() {" + function_body
    token = "codex-sky-exact-job-gate"
    valid_payload = {
        token: [
            {
                "job_id": 1,
                "job_name": "qwen3-4b-reward-hack-grpo",
                "status": "RUNNING",
                "submitted_at": 1.0,
                "start_at": 2.0,
                "end_at": None,
                "pid": 123,
                "log_dir": "/root/sky_logs/1-qwen3-4b-reward-hack-grpo",
                "exit_codes": None,
            }
        ]
    }
    parsed = resolve_remote_job.parse_queue(
        json.dumps(
            {
                "codex-sky-decoy": [{"job_id": 999, "status": "SUCCEEDED"}],
                **valid_payload,
            }
        ),
        token,
        1,
        "qwen3-4b-reward-hack-grpo",
    )
    assert parsed["job_id"] == 1 and parsed["status"] == "RUNNING"
    for malformed in ("", "not-json", "{}", "[]"):
        try:
            resolve_remote_job.parse_queue(
                malformed, token, 1, "qwen3-4b-reward-hack-grpo"
            )
        except resolve_remote_job.ResolutionError:
            pass
        else:
            raise AssertionError("Malformed or empty Sky queue output was accepted")

    fallback_record = {
        "job_id": 1,
        "status": "FAILED_SETUP",
        "submitted_at": 1.0,
        "start_at": None,
        "end_at": 2.0,
        "pid": 123,
        "log_dir": "/root/sky_logs/1-qwen3-4b-reward-hack-grpo",
        "exit_codes": [1],
    }
    trace = root / "job-resolution-trace.txt"
    script = f"""
set -u
CLUSTER_NAME={shlex.quote(token)}
SKY_CONFIG_PATH={shlex.quote(str(root / 'sky-config.yaml'))}
HELPER_ROOT={shlex.quote(str(Path(__file__).resolve().parents[2]))}
EXPECTED_REMOTE_JOB_ID=1
EXPECTED_REMOTE_JOB_NAME=qwen3-4b-reward-hack-grpo
trace={shlex.quote(str(trace))}
SKY_PAYLOAD=not-json
bounded() {{
  shift 2
  if [[ "$1" == sky ]]; then
    printf '%s\n' queue >> "$trace"
    printf '%s\n' "$SKY_PAYLOAD"
    return 0
  fi
  printf '%s\n' fallback >> "$trace"
  printf '%s\n' {shlex.quote(json.dumps(fallback_record, separators=(',', ':')))}
}}
sleep() {{ :; }}
{function_source}
resolve_exact_job_record
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "FAILED_SETUP"
    assert trace.read_text(encoding="ascii").splitlines() == [
        "queue",
        "queue",
        "queue",
        "fallback",
    ]

    status_body = source.split("remote_job_status_is_failure() {", 1)[1].split(
        "\nremote_job_status_is_nonterminal()", 1
    )[0]
    status_function = "remote_job_status_is_failure() {" + status_body
    cleanup_trace = root / "failed-setup-cleanup-trace.txt"
    terminal_script = f"""
set -Ee
trace={shlex.quote(str(cleanup_trace))}
TASK_SUCCEEDED=false
on_exit() {{ printf '%s\n' cleanup >> "$trace"; }}
trap on_exit EXIT
{status_function}
if remote_job_status_is_failure FAILED_SETUP; then
  printf '%s\n' terminal-failure >> "$trace"
  exit 1
fi
TASK_SUCCEEDED=true
printf '%s\n' success >> "$trace"
"""
    terminal = subprocess.run(
        ["bash"], input=terminal_script, text=True, capture_output=True, check=False, timeout=10
    )
    assert terminal.returncode == 1
    assert cleanup_trace.read_text(encoding="ascii").splitlines() == [
        "terminal-failure",
        "cleanup",
    ]
    assert "TASK_SUCCEEDED=true" not in cleanup_trace.read_text(encoding="ascii")

    launch_request_gate = source.index("final_request_status")
    initial_status_gate = source.index('remote_job_record="$(resolve_exact_job_record)"', launch_request_gate)
    failure_gate = source.index('remote_job_status_is_failure "${remote_job_status}"', initial_status_gate)
    job_stream_gate = source.index('bounded_stream "${remaining_seconds}" 1 sky logs', failure_gate)
    final_status_gate = source.index('remote_job_record="$(resolve_exact_job_record)"', job_stream_gate)
    success_gate = source.index("TASK_SUCCEEDED='true'", final_status_gate)
    assert launch_request_gate < initial_status_gate < failure_gate < job_stream_gate
    assert job_stream_gate < final_status_gate < success_gate
    assert "Only SUCCEEDED may pass the remote job gate" in source
    assert "FAILED_DRIVER|FAILED|FAILED_SETUP|CANCELLED" in source


def verify_watchdog_terminal_independence(root: Path) -> None:
    launcher = (Path(__file__).resolve().parent / "run_task.sh").read_text(
        encoding="utf-8"
    )
    spawner = Path(__file__).resolve().parent / "spawn_teardown_watchdog.py"
    assert '--sensitive-file "${TEMP_WANDB_API_KEY_FILE}"' in launcher
    assert 'spawn_teardown_watchdog.py"' in launcher
    assert 'start_new_session=True' in spawner.read_text(encoding="utf-8")
    assert '</dev/null >>"${TEARDOWN_WATCHDOG_LOG}" 2>&1 &' not in launcher
    assert 'kill "${TEARDOWN_WATCHDOG_PID}"' not in launcher

    marker = root / "detached-watchdog-finished"
    log = root / "detached-watchdog.log"
    child_code = (
        "import time; from pathlib import Path; "
        f"time.sleep(0.75); Path({str(marker)!r}).write_text('survived\\n')"
    )
    parent_command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(spawner))} "
        f"--log-file {shlex.quote(str(log))} -- {shlex.quote(sys.executable)} "
        f"-c {shlex.quote(child_code)}; sleep 30"
    )
    parent = subprocess.Popen(
        ["bash", "-c", parent_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert parent.stdout is not None
    child_pid_text = parent.stdout.readline().strip()
    assert child_pid_text.isdigit(), child_pid_text
    os.killpg(parent.pid, signal.SIGHUP)
    parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.read_text(encoding="ascii") == "survived\n"

    sensitive = root / "watchdog-private-key"
    sensitive.write_text("offline-secret\n", encoding="ascii")
    teardown_watchdog.remove_sensitive_file(sensitive)
    assert not sensitive.exists()


def verify_post_parent_ebs_and_s3(root: Path) -> None:
    run_token = "codex-sky-offline-parent-death"
    args = watchdog_args(root, run_token)
    volume_id = "vol-0def456"
    args.volume_ids_file.write_text(volume_id + "\n", encoding="ascii")
    volume_present = True
    aws_calls: list[list[str]] = []

    def fake_watchdog_aws(command: list[str], **_kwargs: object) -> str:
        nonlocal volume_present
        aws_calls.append(command)
        if command[:2] == ["sts", "get-caller-identity"]:
            return ACCOUNT + "\n"
        if command[:2] == ["ec2", "describe-volumes"]:
            if not volume_present:
                return "[]"
            return json.dumps(
                [
                    {
                        "VolumeId": volume_id,
                        "State": "available",
                        "Size": 300,
                        "VolumeType": "gp3",
                        "AvailabilityZone": "us-east-1c",
                        "Encrypted": False,
                        "Tags": [{"Key": "codex-run-owner", "Value": run_token}],
                    }
                ]
            )
        if command[:2] == ["ec2", "create-tags"]:
            return ""
        if command[:2] == ["ec2", "delete-volume"]:
            volume_present = False
            return ""
        raise AssertionError(f"Offline watchdog test blocked unexpected AWS call: {command}")

    state = teardown_watchdog.CleanupState()
    with (
        mock.patch.object(teardown_watchdog, "aws", fake_watchdog_aws),
        mock.patch.object(teardown_watchdog, "exact_instances", return_value=[]),
        mock.patch.object(
            teardown_watchdog,
            "sky_rows",
            return_value=[
                {
                    "cluster_name": run_token,
                    "request_id": "parent-death-request",
                    "status": "FAILED",
                }
            ],
        ),
        mock.patch.object(
            teardown_watchdog,
            "start_s3_verification",
            return_value=ImmediateSuccessfulProcess(),
        ),
    ):
        completed = False
        for _ in range(3):
            completed = teardown_watchdog.cleanup_cycle(
                args, state, parent_failed=True
            )

    assert completed
    assert any(call[:2] == ["ec2", "delete-volume"] for call in aws_calls)
    assert state.volumes_verified
    assert state.s3_verified


def verify_wandb_preflight_receipt(root: Path) -> None:
    output = root / "wandb-online-preflight.json"
    api_key = "offline-test-secret-that-must-not-be-persisted"
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__version__ = "0.22.3"
    fake_wandb.login = mock.Mock(return_value=True)
    environment = {
        "WANDB_API_KEY": api_key,
        "WANDB_MODE": "online",
        "WANDB_PROJECT": "steering-rl-rewardhacking",
        "WANDB_RUN_GROUP": "codex-sky-offline-wandb",
    }
    with (
        mock.patch.dict(os.environ, environment, clear=False),
        mock.patch.dict(sys.modules, {"wandb": fake_wandb}),
    ):
        run_main(
            verify_wandb_online.main,
            [
                "--output",
                str(output),
                "--expected-project",
                environment["WANDB_PROJECT"],
                "--expected-run-group",
                environment["WANDB_RUN_GROUP"],
                "--expected-sdk-version",
                "0.22.3",
            ],
        )
    fake_wandb.login.assert_called_once_with(verify=True)
    assert json.loads(output.read_text(encoding="utf-8"))["credentials_verified"] is True
    assert api_key not in output.read_text(encoding="utf-8")


def verify_local_wandb_secret_file(root: Path) -> None:
    key = "offline-test-secret-file-value"
    path = root / "wandb_api_key"
    path.write_text(key + "\n", encoding="ascii")
    path.chmod(0o600)
    output = run_main(read_wandb_secret.main, ["--path", str(path), "--emit"])
    assert output == key

    rejected = root / "rejected-wandb-key-hashes.txt"
    rejected.write_text(hashlib.sha256(key.encode("ascii")).hexdigest() + "\n", encoding="ascii")
    try:
        run_main(
            read_wandb_secret.main,
            ["--path", str(path), "--reject-sha256-file", str(rejected)],
        )
    except ValueError as error:
        assert "must be rotated" in str(error)
    else:
        raise AssertionError("A previously exposed W&B key was accepted")

    replacement = "offline-rotated-secret-file-value"
    path.write_text(replacement + "\n", encoding="ascii")
    copied = root / "private-wandb-key-copy"
    output = run_main(
        read_wandb_secret.main,
        [
            "--path",
            str(path),
            "--reject-sha256-file",
            str(rejected),
            "--copy-to",
            str(copied),
        ],
    )
    assert output == ""
    assert copied.read_text(encoding="ascii") == replacement + "\n"
    assert copied.stat().st_mode & 0o777 == 0o600

    path.chmod(0o644)
    try:
        run_main(read_wandb_secret.main, ["--path", str(path)])
    except PermissionError:
        pass
    else:
        raise AssertionError("A group/world-readable W&B key file was accepted")


def verify_wandb_run_pointer(root: Path) -> None:
    tracking_path = Path(__file__).resolve().parents[2] / "verl" / "verl" / "utils" / "tracking.py"
    spec = importlib.util.spec_from_file_location("offline_verl_tracking", tracking_path)
    assert spec is not None and spec.loader is not None
    tracking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracking)

    output = root / "wandb-online-run.json"
    api_key = "another-offline-secret-that-must-not-be-persisted"
    run = types.SimpleNamespace(
        entity="offline-fixture",
        id="offlinefixture",
        name="offline-fixture-run",
        project="steering-rl-rewardhacking",
        url="https://wandb.ai/offline-fixture/steering-rl-rewardhacking/runs/offlinefixture",
    )
    with mock.patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": api_key,
            "WANDB_MODE": "online",
            "WANDB_RUN_GROUP": "codex-sky-offline-wandb",
            "WANDB_RUN_METADATA_PATH": str(output),
        },
        clear=False,
    ):
        tracking._write_wandb_run_metadata(run, "0.22.3")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_id"] == run.id
    assert payload["run_url"] == run.url
    assert api_key not in output.read_text(encoding="utf-8")


def verify_disabled_wandb_logger_without_receipt(root: Path) -> None:
    tracking_path = Path(__file__).resolve().parents[2] / "verl" / "verl" / "utils" / "tracking.py"
    spec = importlib.util.spec_from_file_location("offline_disabled_verl_tracking", tracking_path)
    assert spec is not None and spec.loader is not None
    tracking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracking)

    initialized: list[dict[str, object]] = []
    disabled_run = types.SimpleNamespace(
        entity=None,
        id="disabled-run",
        name="memory-qualification",
        project="steering-rl-rewardhacking",
        url=None,
    )
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__version__ = "0.22.3"

    def fake_init(**kwargs: object) -> object:
        initialized.append(kwargs)
        return disabled_run

    fake_wandb.init = fake_init
    fake_wandb.finish = lambda **_kwargs: None
    forbidden_receipt = root / "disabled-wandb-must-not-write.json"
    with (
        mock.patch.dict(sys.modules, {"wandb": fake_wandb}),
        mock.patch.dict(
            os.environ,
            {
                "WANDB_MODE": "disabled",
                "WANDB_JOB_TYPE": "memory-qualification",
            },
            clear=True,
        ),
    ):
        tracker = tracking.Tracking(
            "steering-rl-rewardhacking",
            "memory-qualification",
            default_backend="wandb",
            config={"trainer": {}},
        )
    assert initialized and tracker.logger["wandb"] is fake_wandb
    assert disabled_run.url is None
    assert not forbidden_receipt.exists()


def verify_hardware_profiles_and_effective_tasks() -> None:
    directory = Path(__file__).resolve().parent
    expected = {
        P4DE.profile_id: (P4DE, 32, 0.85, 1024, False),
        P4D.profile_id: (P4D, 8, 0.70, 64, True),
        P4D_MICROBATCH4.profile_id: (P4D_MICROBATCH4, 4, 0.70, 64, True),
    }
    for profile, microbatch, utilization, max_sequences, qualification in expected.values():
        task = (directory.parents[1] / profile.task_path).read_text(encoding="utf-8")
        assert "num_nodes: 1" in task
        assert f"instance_type: {profile.instance_type}" in task
        assert f"accelerators: {profile.accelerators}" in task
        assert "disk_size: 300" in task
        assert "use_spot: false" in task
        assert f'PPO_MICRO_BATCH_SIZE_PER_GPU: "{microbatch}"' in task
        assert f'VLLM_GPU_MEMORY_UTILIZATION: "{utilization:.2f}"' in task
        assert f'VLLM_MAX_NUM_SEQS: "{max_sequences}"' in task
        assert f'RUN_MEMORY_QUALIFICATION: "{str(qualification).lower()}"' in task
        assert "720m" in task
        rendered = render_effective_training_config.effective_training_config(profile)
        experiment = rendered["experiment"]
        memory = rendered["memory"]
        assert experiment["optimizer_steps"] == 200
        assert experiment["prompts_per_step"] == 16
        assert experiment["generations_per_prompt"] == 16
        assert experiment["rollouts_per_step"] == 256
        assert experiment["max_prompt_length"] == 1536
        assert experiment["max_completion_length"] == 1536
        assert memory["ppo_micro_batch_size_per_gpu"] == microbatch
        assert memory["log_prob_micro_batch_size_per_gpu"] == microbatch
        assert memory["vllm_gpu_memory_utilization"] == utilization
        assert memory["vllm_max_num_seqs"] == max_sequences
        assert memory["automatic_fallback"] is False

    # P4de rollback remains a first-class exact profile with all prior values.
    assert P4DE.instance_type == "p4de.24xlarge"
    assert P4DE.accelerators == "A100-80GB:8"
    assert P4DE.max_hourly_cost_usd == 27.50
    assert P4DE.capacity_wait_seconds == 21_600
    assert P4DE.overall_watchdog_seconds == 50_400
    assert P4DE.ppo_micro_batch_size_per_gpu == 32
    assert P4DE.vllm_gpu_memory_utilization == 0.85
    assert P4DE.availability_zones == ("us-east-1c", "us-east-1d")
    assert P4DE.subnet_ids == (
        "subnet-00000000000000002",
        "subnet-00000000000000001",
    )
    assert P4D.availability_zones == (
        "us-east-1a",
        "us-east-1b",
        "us-east-1c",
        "us-east-1d",
    )
    assert P4D.subnet_ids == (
        "subnet-00000000000000003",
        "subnet-00000000000000004",
        "subnet-00000000000000002",
        "subnet-00000000000000001",
    )
    p4de_policy = json.loads(
        (directory / "iam/codex_skypilot_a100_permissions.json").read_text(
            encoding="utf-8"
        )
    )
    p4de_launch = next(
        statement
        for statement in p4de_policy["Statement"]
        if statement["Sid"] == "LaunchOnlyReviewedTaggedInstance"
    )
    assert p4de_launch["Condition"]["StringEquals"]["ec2:InstanceType"] == "p4de.24xlarge"
    p4de_dependencies = next(
        statement
        for statement in p4de_policy["Statement"]
        if statement["Sid"] == "UseOnlyReviewedLaunchDependencies"
    )
    assert {
        resource for resource in p4de_dependencies["Resource"] if ":subnet/" in resource
    } == {
        "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-00000000000000002",
        "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-00000000000000001",
    }

    rendered_p4d_policy = render_token_bound_iam.render_policy(
        p4de_policy, "codex-sky-offline-four-zone-policy"
    )
    p4d_statements = {
        statement["Sid"]: statement for statement in rendered_p4d_policy["Statement"]
    }
    assert {
        resource
        for resource in p4d_statements["UseOnlyReviewedLaunchDependencies"]["Resource"]
        if ":subnet/" in resource
    } == {
        f"arn:aws:ec2:us-east-1:123456789012:subnet/{subnet_id}"
        for subnet_id in P4D.subnet_ids
    }
    for statement_name in (
        "TagOnlyReviewedShapeCapturedVolumes",
        "DeleteCapturedOrphanVolumesInReviewedRegion",
    ):
        assert set(
            p4d_statements[statement_name]["Condition"]["StringEquals"][
                "ec2:AvailabilityZone"
            ]
        ) == set(P4D.availability_zones)

    run_script = (directory / "run_reward_hack.sh").read_text(encoding="utf-8")
    assert "readonly STEPS=200" in run_script
    assert "readonly SAVE_STEPS=10" in run_script
    assert "--num_prompts=16 --num_generations=16" in run_script
    assert "--max_prompt_length=1536 --max_completion_length=1536" in run_script
    assert "--steps=1 --seed=1" in run_script
    assert "--steps=\"${STEPS}\" --seed=1" in run_script
    assert "WANDB_MODE=disabled WANDB_JOB_TYPE=memory-qualification" in run_script
    assert "WANDB_RUN_METADATA_PATH= \\\n    setsid timeout" in run_script
    assert "configuration was not mutated" in run_script
    assert "PPO_MICRO_BATCH_SIZE_PER_GPU=\"4\"" not in run_script
    training_cli = (directory.parents[1] / "scripts/run_rl_training.py").read_text(
        encoding="utf-8"
    )
    grpo_source = (directory.parents[1] / "src/train/verl/grpo.py").read_text(
        encoding="utf-8"
    )
    grpo_template = (directory.parents[1] / "src/train/verl/grpo_config.jinja2").read_text(
        encoding="utf-8"
    )
    assert "max_num_seqs: int = 1024" in training_cli
    assert "'max_num_seqs': self.training_config.max_num_seqs" in grpo_source
    assert "ppo_micro_batch_size_per_gpu: {{ per_device_batch_size }}" in grpo_template
    assert grpo_template.count("log_prob_micro_batch_size_per_gpu: {{ per_device_batch_size }}") == 2
    assert "gpu_memory_utilization: {{ gpu_memory_utilization }}" in grpo_template
    assert "max_num_seqs: {{ max_num_seqs }}" in grpo_template

    launcher = (directory / "run_task.sh").read_text(encoding="utf-8")
    assert "infra/skypilot/a100_reward_hack.yaml)" in launcher
    assert "infra/skypilot/a100_40gb_reward_hack.yaml)" in launcher
    assert "--instance-type \"${INSTANCE_TYPE}\"" in launcher
    assert "readonly POST_ALLOCATION_LIMIT_SECONDS='43200'" in launcher
    assert "readonly WALL_LIMIT_SECONDS='68400'" in launcher
    assert "readonly REVIEWED_SUBNET_A='subnet-00000000000000003'" in launcher
    assert "readonly REVIEWED_SUBNET_B='subnet-00000000000000004'" in launcher
    p4de_branch = launcher.split("infra/skypilot/a100_reward_hack.yaml)", 1)[1].split(
        ";;", 1
    )[0]
    p4d_branch = launcher.split(
        "infra/skypilot/a100_40gb_reward_hack.yaml)", 1
    )[1].split(";;", 1)[0]
    assert "'us-east-1a'" not in p4de_branch and "'us-east-1b'" not in p4de_branch
    assert "'us-east-1c' 'us-east-1d'" in p4de_branch
    for zone in P4D.availability_zones:
        assert f"'{zone}'" in p4d_branch
    for zone, _subnet_id in zip(P4D.availability_zones, P4D.subnet_ids, strict=True):
        constant_suffix = zone[-1].upper()
        assert f'"${{REVIEWED_SUBNET_{constant_suffix}}}"' in p4d_branch
    assert "required = set(sys.argv[2:])" in launcher


def verify_profile_specific_volume_zones() -> None:
    volume = {
        "Size": 300,
        "VolumeType": "gp3",
        "AvailabilityZone": "us-east-1a",
        "Encrypted": False,
    }
    assert teardown_watchdog._reviewed_volume_shape(volume, P4D.profile_id)
    assert not teardown_watchdog._reviewed_volume_shape(volume, P4DE.profile_id)
    volume["AvailabilityZone"] = "us-east-1b"
    assert teardown_watchdog._reviewed_volume_shape(volume, P4D.profile_id)
    volume["AvailabilityZone"] = "us-east-1e"
    assert not teardown_watchdog._reviewed_volume_shape(volume, P4D.profile_id)


def verify_gpu_hardware_detection() -> None:
    p4d_rows = "\n".join(["NVIDIA A100-SXM4-40GB, 40536 MiB"] * 8)
    p4de_rows = "\n".join(["NVIDIA A100-SXM4-80GB, 81920 MiB"] * 8)
    p4d_report = validate_gpu_hardware.parse_nvidia_smi(p4d_rows, P4D.profile_id)
    p4de_report = validate_gpu_hardware.parse_nvidia_smi(p4de_rows, P4DE.profile_id)
    assert p4d_report["gpu_count"] == 8
    assert p4de_report["gpu_count"] == 8
    for payload, profile in (
        ("\n".join(["NVIDIA A100-SXM4-40GB, 40536 MiB"] * 7), P4D),
        (p4de_rows, P4D),
        ("\n".join(["NVIDIA H100 80GB HBM3, 81559 MiB"] * 8), P4DE),
    ):
        try:
            validate_gpu_hardware.parse_nvidia_smi(payload, profile.profile_id)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected GPU count, model, or memory was accepted")


def verify_qualification_failure_gate(root: Path) -> None:
    source = (Path(__file__).resolve().parent / "run_reward_hack.sh").read_text(
        encoding="utf-8"
    )
    body = source.split("run_memory_qualification() {", 1)[1].split(
        "\non_interrupt()", 1
    )[0]
    function_source = "run_memory_qualification() {" + body
    qualification_root = root / "qualification-oom"
    qualification_root.mkdir()
    trace = root / "qualification-oom-trace.txt"
    script = f"""
set -u
RUN_MEMORY_QUALIFICATION=true
RUN_TOKEN=codex-sky-offline-qualification-oom
MODEL_ID=Qwen/Qwen3-4B
MODEL_REVISION={REVISION}
TASK=simple_overwrite_tests
PPO_MICRO_BATCH_SIZE_PER_GPU=8
VLLM_GPU_MEMORY_UTILIZATION=0.70
VLLM_MAX_NUM_SEQS=64
HARDWARE_PROFILE=p4d-a100-40gb
QUALIFICATION_ROOT={shlex.quote(str(qualification_root))}
QUALIFICATION_MARKER={shlex.quote(str(root / 'qualification-marker'))}
trace={shlex.quote(str(trace))}
setsid() {{ return 137; }}
sync_qualification_diagnostics() {{ printf '%s\n' diagnostics-synced >> "$trace"; }}
release_qualification_gpu_state() {{ printf '%s\n' gpu-release-checked >> "$trace"; }}
{function_source}
run_memory_qualification
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 137
    assert trace.read_text(encoding="ascii").splitlines() == [
        "diagnostics-synced",
        "gpu-release-checked",
    ]
    assert "configuration was not mutated" in completed.stderr
    assert source.index("run_memory_qualification") < source.index("sync_loop &")


def verify_qualification_success_gate(root: Path) -> None:
    source = (Path(__file__).resolve().parent / "run_reward_hack.sh").read_text(
        encoding="utf-8"
    )
    body = source.split("run_memory_qualification() {", 1)[1].split(
        "\non_interrupt()", 1
    )[0]
    function_source = "run_memory_qualification() {" + body
    qualification_root = root / "qualification-success"
    qualification_root.mkdir()
    trace = root / "qualification-success-trace.txt"
    script = f"""
set -u
RUN_MEMORY_QUALIFICATION=true
RUN_TOKEN=codex-sky-offline-qualification-success
MODEL_ID=Qwen/Qwen3-4B
MODEL_REVISION={REVISION}
TASK=simple_overwrite_tests
PPO_MICRO_BATCH_SIZE_PER_GPU=8
VLLM_GPU_MEMORY_UTILIZATION=0.70
VLLM_MAX_NUM_SEQS=64
HARDWARE_PROFILE=p4d-a100-40gb
QUALIFICATION_ROOT={shlex.quote(str(qualification_root))}
QUALIFICATION_MARKER={shlex.quote(str(root / 'qualification-success-marker'))}
WANDB_RUN_METADATA_PATH=/receipt-must-be-cleared-for-disabled-run
trace={shlex.quote(str(trace))}
setsid() {{
  [[ "${{WANDB_MODE:-}}" == disabled ]] || return 91
  [[ "${{WANDB_JOB_TYPE:-}}" == memory-qualification ]] || return 92
  [[ -z "${{WANDB_RUN_METADATA_PATH:-}}" ]] || return 93
  printf '%s\n' disabled-wandb-logger-initialized >> "$trace"
  return 0
}}
sync_qualification_diagnostics() {{ printf '%s\n' diagnostics-synced >> "$trace"; }}
release_qualification_gpu_state() {{ printf '%s\n' gpu-release-checked >> "$trace"; }}
{function_source}
run_memory_qualification
"""
    completed = subprocess.run(
        ["bash"], input=script, text=True, capture_output=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert trace.read_text(encoding="ascii").splitlines() == [
        "disabled-wandb-logger-initialized",
        "diagnostics-synced",
        "gpu-release-checked",
    ]
    assert json.loads((qualification_root / "status.json").read_text(encoding="utf-8")) == {
        "hardware_profile": "p4d-a100-40gb",
        "status": 0,
    }
    assert "completed and all GPU compute processes were released" in completed.stdout


def verify_post_allocation_deadline_independence() -> None:
    args = types.SimpleNamespace(post_allocation_limit_seconds=43_200)
    state = teardown_watchdog.CleanupState()
    with mock.patch.object(teardown_watchdog.time, "monotonic", return_value=99_999.0):
        teardown_watchdog._record_allocation_start(
            state,
            [{"InstanceId": "i-012345", "InstanceType": "p4d.24xlarge"}],
        )
    # Any six-hour capacity wait happened before this first-allocation timestamp.
    assert state.allocation_started_monotonic == 99_999.0
    with mock.patch.object(
        teardown_watchdog.time, "monotonic", return_value=99_999.0 + 43_199
    ):
        assert not teardown_watchdog.post_allocation_deadline_reached(args, state)
    with mock.patch.object(
        teardown_watchdog.time, "monotonic", return_value=99_999.0 + 43_200
    ):
        assert teardown_watchdog.post_allocation_deadline_reached(args, state)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="reward-hack-offline-package-") as temporary_name:
        root = Path(temporary_name)
        bucket_root = root / "bucket"
        reviewed_manifest = root / "reviewed_manifest.json"
        write_json(reviewed_manifest, {"fixture": True})
        reviewed_hash = sha256(reviewed_manifest)
        verify_teardown_proof(root)
        verify_sky_failure_does_not_gate_ec2(root, "p4de.24xlarge")
        verify_sky_failure_does_not_gate_ec2(root, "p4d.24xlarge")
        verify_shell_cleanup_sky_failure(root)
        verify_local_sky_cluster_filter(root)
        verify_local_sky_api_start_on_non_json_success(root)
        verify_launcher_sky_cluster_filter(root)
        verify_bounded_capacity_retry(root)
        verify_public_pypi_setup()
        verify_remote_sqlite_reader(root)
        verify_remote_job_completion_gate(root)
        verify_watchdog_terminal_independence(root)
        verify_post_parent_ebs_and_s3(root)
        verify_hardware_profiles_and_effective_tasks()
        verify_profile_specific_volume_zones()
        verify_gpu_hardware_detection()
        verify_disabled_wandb_logger_without_receipt(root)
        verify_qualification_failure_gate(root)
        verify_qualification_success_gate(root)
        verify_post_allocation_deadline_independence()
        verify_local_wandb_secret_file(root)
        verify_wandb_preflight_receipt(root)
        verify_wandb_run_pointer(root)

        partial_token = "codex-sky-offline-partial"
        partial_prefix = f"qwen3-4b/no-intervention/{partial_token}/fixture-run"
        partial_run = bucket_root / partial_prefix
        create_common_metadata(partial_run, reviewed_manifest, partial_token)
        for step in range(10, 101, 10):
            create_adapter(partial_run, step, partial_token, reviewed_hash)
        for step in range(1, 101):
            create_rollout(partial_run, step)
        partial_pointer = (
            bucket_root
            / "qwen3-4b/no-intervention/launches"
            / partial_token
            / "partial.json"
        )
        publish_partial(
            partial_run, partial_pointer, partial_prefix, partial_token, reviewed_manifest
        )
        # Simulate a crash after new durable files landed but before the pointer advanced.
        create_adapter(partial_run, 110, partial_token, reviewed_hash)
        create_rollout(partial_run, 101)
        partial_report = verify_fake_s3(bucket_root, partial_token, require_final=False)
        assert partial_report["verified"] is True
        assert partial_report["partial_pointer_current"] is False
        assert partial_report["complete_steps"][-1] == 110
        assert partial_report["verified_rollout_steps"][-1] == 101

        final_token = "codex-sky-offline-final"
        final_prefix = f"qwen3-4b/no-intervention/{final_token}/fixture-run"
        durable = bucket_root / final_prefix
        local_run = root / "local-run"
        create_common_metadata(durable, reviewed_manifest, final_token)
        for step in EXPECTED_STEPS:
            create_adapter(durable, step, final_token, reviewed_hash)
            create_adapter(local_run, step, final_token, reviewed_hash)
        for step in range(1, 201):
            create_rollout(durable, step)
            create_rollout(local_run, step)
        final_partial_pointer = (
            bucket_root / "qwen3-4b/no-intervention/launches" / final_token / "partial.json"
        )
        publish_partial(
            durable, final_partial_pointer, final_prefix, final_token, reviewed_manifest
        )
        create_evaluation_artifacts(durable)
        final_pointer = (
            bucket_root / "qwen3-4b/no-intervention/launches" / final_token / "result.json"
        )
        run_main(
            finalize_package.main,
            [
                "--run-dir",
                str(local_run),
                "--durable-run-dir",
                str(durable),
                "--pointer",
                str(final_pointer),
                "--run-prefix",
                final_prefix,
                "--base-model",
                MODEL,
                "--revision",
                REVISION,
                "--run-token",
                final_token,
            ],
        )
        final_report = verify_fake_s3(bucket_root, final_token, require_final=True)
        assert final_report["verified"] is True
        assert final_report["partial_pointer_current"] is True
        assert final_report["complete_steps"] == EXPECTED_STEPS
        assert final_report["verified_rollout_steps"] == list(range(1, 201))
        assert final_report["final_pointer_status"] == "success"
        assert final_report["selected_checkpoint"]["step"] == 80

    print(
        "Offline packaging test passed: Sky-independent EC2 teardown proof, "
        "post-parent EBS/S3 verification, stale partial recovery, and full final verification"
    )


if __name__ == "__main__":
    main()
