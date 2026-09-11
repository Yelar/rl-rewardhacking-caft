#!/usr/bin/env python3

"""Verify every completed adapter directly from S3 after EC2 termination."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path


EXPECTED_ACCOUNT = "123456789012"
EXPECTED_MODEL = "Qwen/Qwen3-4B"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_STEPS = list(range(10, 201, 10))
PRIORITY_STEPS = [80, 90, 100, 200]


class AwsCallError(RuntimeError):
    def __init__(self, command: list[str], stderr: str):
        super().__init__(f"AWS command failed: {' '.join(command)}: {stderr}")
        self.stderr = stderr


def aws(*args: str, attempts: int = 3, timeout: int = 300) -> str:
    command = ["aws", *args]
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_error = f"timed out after {timeout}s"
        else:
            if completed.returncode == 0:
                return completed.stdout
            last_error = completed.stderr.strip() or f"exit {completed.returncode}"
        if attempt < attempts:
            time.sleep(2 * attempt)
    raise AwsCallError(command, last_error)


def download(bucket: str, key: str, destination: Path, region: str) -> None:
    aws(
        "s3",
        "cp",
        f"s3://{bucket}/{key}",
        str(destination),
        "--region",
        region,
        "--only-show-errors",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_objects(bucket: str, prefix: str, region: str) -> tuple[bool, dict[str, int]]:
    try:
        payload = aws(
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--region",
            region,
            "--output",
            "json",
        )
    except AwsCallError as error:
        if "NoSuchBucket" in error.stderr:
            return False, {}
        raise
    parsed = json.loads(payload)
    return True, {item["Key"]: int(item["Size"]) for item in parsed.get("Contents", [])}


def verify_adapter_from_marker(
    bucket: str,
    marker_key: str,
    region: str,
    temporary: Path,
    run_token: str,
) -> dict:
    match = re.fullmatch(
        r"(.+)/checkpoints/global_step_(\d+)/actor/lora_adapter/\.complete\.json",
        marker_key,
    )
    if match is None:
        raise ValueError(f"Unexpected completion-marker key: {marker_key}")
    run_prefix, raw_step = match.groups()
    step = int(raw_step)
    if step <= 0 or step > 200:
        raise ValueError(f"Completed checkpoint step is outside the reviewed run: {step}")
    local_dir = temporary / f"step-{step}"
    local_dir.mkdir()
    marker_path = local_dir / ".complete.json"
    download(bucket, marker_key, marker_path, region)
    marker = load_json(marker_path)
    if not marker.get("valid"):
        raise ValueError(f"Completion marker is not valid at step {step}")
    if marker.get("base_model_name_or_path") != EXPECTED_MODEL:
        raise ValueError(f"Completion marker has wrong base model at step {step}")
    if marker.get("revision") != EXPECTED_REVISION:
        raise ValueError(f"Completion marker has wrong revision at step {step}")
    if marker.get("run_token") != run_token:
        raise ValueError(f"Completion marker has wrong run token at step {step}")
    reviewed_manifest_sha256 = marker.get("reviewed_manifest_sha256")
    if not isinstance(reviewed_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", reviewed_manifest_sha256
    ):
        raise ValueError(f"Completion marker has invalid reviewed-manifest hash at step {step}")
    observed_files: dict[str, dict[str, int | str]] = {}
    adapter_prefix = marker_key.removesuffix("/.complete.json")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        destination = local_dir / filename
        download(bucket, f"{adapter_prefix}/{filename}", destination, region)
        observed = {"bytes": destination.stat().st_size, "sha256": sha256(destination)}
        if marker.get("files", {}).get(filename) != observed:
            raise ValueError(f"Independent S3 hash verification failed: {adapter_prefix}/{filename}")
        observed_files[filename] = observed
    adapter_config = load_json(local_dir / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != EXPECTED_MODEL:
        raise ValueError(f"S3 adapter config has wrong base model at step {step}")
    if adapter_config.get("revision") != EXPECTED_REVISION:
        raise ValueError(f"S3 adapter config has wrong revision at step {step}")
    return {
        "step": step,
        "run_prefix": run_prefix,
        "marker_key": marker_key,
        "marker": {"bytes": marker_path.stat().st_size, "sha256": sha256(marker_path)},
        "files": observed_files,
        "reviewed_manifest_sha256": reviewed_manifest_sha256,
    }


def assert_artifact_hash(
    bucket: str,
    run_prefix: str,
    relative: str,
    artifacts_by_path: dict[str, dict],
    destination: Path,
    region: str,
) -> dict:
    expected = artifacts_by_path.get(relative)
    if expected is None:
        raise ValueError(f"Artifact manifest is missing {relative}")
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        download(bucket, f"{run_prefix}/{relative}", destination, region)
    if destination.stat().st_size != expected["bytes"] or sha256(destination) != expected["sha256"]:
        raise ValueError(f"S3 artifact hash mismatch: {relative}")
    return load_json(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--hardware-profile", default="p4de-a100-80gb")
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()

    account = aws("sts", "get-caller-identity", "--query", "Account", "--output", "text").strip()
    if account != EXPECTED_ACCOUNT:
        raise RuntimeError(f"AWS account mismatch during S3 verification: {account}")

    run_root = f"qwen3-4b/no-intervention/{args.run_token}/"
    launch_root = f"qwen3-4b/no-intervention/launches/{args.run_token}/"
    bucket_exists, run_objects = list_objects(args.bucket, run_root, args.region)
    if not bucket_exists:
        if args.require_final:
            raise RuntimeError("Successful task did not create the reviewed checkpoint bucket")
        print(
            json.dumps(
                {"verified": True, "account": account, "bucket_exists": False, "complete_steps": []},
                indent=2,
            )
        )
        return
    _, launch_objects = list_objects(args.bucket, launch_root, args.region)
    marker_keys = [key for key in run_objects if key.endswith("/lora_adapter/.complete.json")]
    marker_steps: dict[str, int] = {}
    for key in marker_keys:
        match = re.fullmatch(
            r"(.+)/checkpoints/global_step_(\d+)/actor/lora_adapter/\.complete\.json", key
        )
        if match is None:
            raise ValueError(f"Unexpected completion-marker key: {key}")
        marker_steps[key] = int(match.group(2))
    marker_keys.sort(key=marker_steps.__getitem__)

    with tempfile.TemporaryDirectory(prefix="codex-rh-s3-verify-") as temporary_name:
        temporary = Path(temporary_name)
        adapters = [
            verify_adapter_from_marker(
                args.bucket, key, args.region, temporary, args.run_token
            )
            for key in marker_keys
        ]
        steps = [item["step"] for item in adapters]
        if steps != sorted(set(steps)):
            raise ValueError("Duplicate or unordered completed checkpoint steps were discovered")
        run_prefixes = {item["run_prefix"] for item in adapters}
        if len(run_prefixes) > 1:
            raise ValueError(f"One launch produced multiple durable run prefixes: {sorted(run_prefixes)}")
        expected_run_prefix_pattern = re.compile(
            rf"qwen3-4b/no-intervention/{re.escape(args.run_token)}/[^/]+"
        )
        if any(expected_run_prefix_pattern.fullmatch(prefix) is None for prefix in run_prefixes):
            raise ValueError("Completion marker is outside the exact launch-scoped run prefix")

        reviewed_manifest_hashes = {
            item["reviewed_manifest_sha256"] for item in adapters
        }
        if len(reviewed_manifest_hashes) > 1:
            raise ValueError("Completed adapters disagree on the reviewed-manifest hash")

        rollout_keys: list[tuple[int, str, str]] = []
        for key in run_objects:
            match = re.fullmatch(r"(.+)/rollouts/(\d+)\.jsonl", key)
            if match is None:
                continue
            rollout_prefix, raw_step = match.groups()
            step = int(raw_step)
            if step <= 0 or step > 200:
                raise ValueError(f"Rollout step is outside the reviewed run: {key}")
            rollout_keys.append((step, key, rollout_prefix))
        rollout_keys.sort()
        direct_rollout_steps = [item[0] for item in rollout_keys]
        if direct_rollout_steps != sorted(set(direct_rollout_steps)):
            raise ValueError("Duplicate rollout steps were discovered across S3 run prefixes")
        rollout_prefixes = {item[2] for item in rollout_keys}
        if len(rollout_prefixes) > 1 or (
            run_prefixes and rollout_prefixes and run_prefixes != rollout_prefixes
        ):
            raise ValueError("One launch produced rollout files under multiple run prefixes")
        direct_rollout_hashes: dict[str, dict] = {}
        direct_rollout_by_step: dict[int, dict] = {}
        for step, key, _ in rollout_keys:
            destination = temporary / "direct-rollouts" / f"{step}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            download(args.bucket, key, destination, args.region)
            observed = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
            if observed["bytes"] == 0:
                raise ValueError(f"Empty rollout file in S3: {key}")
            relative = f"rollouts/{step}.jsonl"
            direct_rollout_hashes[relative] = observed
            direct_rollout_by_step[step] = observed

        partial_key = f"{launch_root}partial.json"
        partial = None
        verified_rollout_steps = direct_rollout_steps
        partial_pointer_current = False
        if partial_key in launch_objects:
            partial_path = temporary / "partial.json"
            download(args.bucket, partial_key, partial_path, args.region)
            partial = load_json(partial_path)
            if partial.get("run_token") != args.run_token:
                raise ValueError("Partial launch pointer has the wrong run token")
            if partial.get("base_model_name_or_path") != EXPECTED_MODEL or partial.get("revision") != EXPECTED_REVISION:
                raise ValueError("Partial launch pointer has wrong model provenance")
            if partial.get("expected_steps") != EXPECTED_STEPS:
                raise ValueError("Partial launch pointer has the wrong checkpoint schedule")
            if expected_run_prefix_pattern.fullmatch(str(partial.get("run_prefix"))) is None:
                raise ValueError("Partial launch pointer has an invalid run prefix")
            partial_steps = partial.get("complete_steps")
            if not isinstance(partial_steps, list) or partial_steps != sorted(set(partial_steps)):
                raise ValueError("Partial launch pointer has invalid completion steps")
            if not set(partial_steps).issubset(steps):
                raise ValueError("Partial launch pointer names undiscovered completion markers")
            if partial.get("scheduled_complete_steps") != [
                step for step in partial_steps if step in EXPECTED_STEPS
            ]:
                raise ValueError("Partial launch pointer has incorrect scheduled completion steps")
            if run_prefixes and partial.get("run_prefix") not in run_prefixes:
                raise ValueError("Partial launch pointer identifies a different run prefix")
            manifest_ref = partial.get("partial_checkpoint_manifest", {})
            manifest_sha256 = manifest_ref.get("sha256")
            if not isinstance(manifest_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", manifest_sha256
            ):
                raise ValueError("Partial checkpoint manifest has an invalid digest")
            expected_manifest_key = (
                f"{partial['run_prefix']}/partial-manifests/{manifest_sha256}.json"
            )
            if manifest_ref.get("key") != expected_manifest_key:
                raise ValueError("Partial checkpoint manifest is not content-addressed in this run")
            partial_manifest_path = temporary / "PARTIAL_CHECKPOINTS.json"
            download(args.bucket, manifest_ref["key"], partial_manifest_path, args.region)
            if (
                partial_manifest_path.stat().st_size != manifest_ref.get("bytes")
                or sha256(partial_manifest_path) != manifest_ref.get("sha256")
            ):
                raise ValueError("Partial checkpoint manifest hash or size mismatch")
            partial_manifest = load_json(partial_manifest_path)
            if partial_manifest.get("complete_steps") != partial_steps:
                raise ValueError("Partial checkpoint manifest disagrees with its launch pointer")
            partial_by_step = {
                int(item["step"]): item for item in partial_manifest.get("adapters", [])
            }
            direct_by_step = {int(item["step"]): item for item in adapters}
            if sorted(partial_by_step) != partial_steps:
                raise ValueError("Partial checkpoint manifest adapter entries are incomplete")
            for step in partial_steps:
                if partial_by_step[step].get("files") != direct_by_step[step]["files"]:
                    raise ValueError(f"Partial manifest hashes disagree at step {step}")
            rollout_entries = partial_manifest.get("rollouts", [])
            partial_rollout_steps = [int(item["step"]) for item in rollout_entries]
            if (
                partial_rollout_steps != partial_manifest.get("observed_rollout_steps")
                or partial_rollout_steps != partial.get("observed_rollout_steps")
                or partial_rollout_steps != sorted(set(partial_rollout_steps))
                or not set(partial_rollout_steps).issubset(direct_rollout_steps)
            ):
                raise ValueError("Partial rollout steps are inconsistent")
            for item in rollout_entries:
                relative = Path(item["path"])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != f"rollouts/{item['step']}.jsonl"
                ):
                    raise ValueError(f"Unsafe or mismatched rollout path: {item['path']}")
                observed = direct_rollout_by_step[item["step"]]
                if observed != {"bytes": item["bytes"], "sha256": item["sha256"]}:
                    raise ValueError(f"Partial rollout hash mismatch: {item['path']}")
            partial_reviewed_path = temporary / "partial-reviewed-manifest.json"
            download(
                args.bucket,
                f"{partial['run_prefix']}/metadata/reviewed_manifest.json",
                partial_reviewed_path,
                args.region,
            )
            if (
                sha256(partial_reviewed_path) != partial.get("reviewed_manifest_sha256")
                or partial_manifest.get("reviewed_manifest_sha256")
                != partial.get("reviewed_manifest_sha256")
            ):
                raise ValueError("Partial package reviewed-manifest hash mismatch")
            if reviewed_manifest_hashes and reviewed_manifest_hashes != {
                partial.get("reviewed_manifest_sha256")
            }:
                raise ValueError("Checkpoint markers and partial pointer bind different reviews")
            partial_pointer_current = (
                partial_steps == steps and partial_rollout_steps == direct_rollout_steps
            )
        elif adapters:
            raise ValueError("Completed adapters exist but the durable partial launch pointer is missing")

        result_key = f"{launch_root}result.json"
        final_status = None
        selected_checkpoint = None
        manifest_file_count = 0
        if result_key in launch_objects:
            pointer_path = temporary / "result.json"
            download(args.bucket, result_key, pointer_path, args.region)
            pointer = load_json(pointer_path)
            if pointer.get("run_token") != args.run_token:
                raise ValueError("Final pointer has the wrong run token")
            if pointer.get("hardware_profile") != args.hardware_profile:
                raise ValueError("Final pointer has the wrong hardware profile")
            if pointer.get("base_model_name_or_path") != EXPECTED_MODEL or pointer.get("revision") != EXPECTED_REVISION:
                raise ValueError("Final pointer has wrong model provenance")
            if pointer.get("expected_steps") != EXPECTED_STEPS or not set(EXPECTED_STEPS).issubset(steps):
                raise ValueError("Final package does not contain all 20 completed adapters")
            if expected_run_prefix_pattern.fullmatch(str(pointer.get("run_prefix"))) is None:
                raise ValueError("Final pointer has an invalid run prefix")
            if run_prefixes != {pointer.get("run_prefix")}:
                raise ValueError("Final pointer identifies a different run prefix")
            artifacts_path = temporary / "ARTIFACTS.json"
            checkpoints_path = temporary / "CHECKPOINTS.json"
            download(args.bucket, pointer["artifact_manifest"]["key"], artifacts_path, args.region)
            download(args.bucket, pointer["checkpoint_manifest"]["key"], checkpoints_path, args.region)
            if sha256(artifacts_path) != pointer["artifact_manifest"]["sha256"]:
                raise ValueError("S3 artifact manifest SHA-256 mismatch")
            if sha256(checkpoints_path) != pointer["checkpoint_manifest"]["sha256"]:
                raise ValueError("S3 checkpoint manifest SHA-256 mismatch")
            artifacts = load_json(artifacts_path)
            if artifacts.get("hardware_profile") != args.hardware_profile:
                raise ValueError("Artifact manifest has the wrong hardware profile")
            artifacts_by_path = {item["path"]: item for item in artifacts["files"]}
            direct_artifact_hashes: dict[str, dict] = {}
            for adapter in adapters:
                step = adapter["step"]
                relative_dir = f"checkpoints/global_step_{step}/actor/lora_adapter"
                for filename, details in adapter["files"].items():
                    direct_artifact_hashes[f"{relative_dir}/{filename}"] = details
                direct_artifact_hashes[f"{relative_dir}/.complete.json"] = adapter["marker"]
            direct_artifact_hashes.update(direct_rollout_hashes)
            artifact_download_root = temporary / "artifacts"
            for item in artifacts["files"]:
                key = f"{pointer['run_prefix']}/{item['path']}"
                if run_objects.get(key) != item["bytes"]:
                    raise ValueError(f"Missing or wrong-sized S3 artifact: {key}")
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe path in artifact manifest: {item['path']}")
                direct = direct_artifact_hashes.get(item["path"])
                if direct is not None:
                    if direct != {"bytes": item["bytes"], "sha256": item["sha256"]}:
                        raise ValueError(f"Direct adapter hash differs from artifact manifest: {item['path']}")
                    continue
                destination = artifact_download_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                download(args.bucket, key, destination, args.region)
                if destination.stat().st_size != item["bytes"] or sha256(destination) != item["sha256"]:
                    raise ValueError(f"Independent S3 artifact hash verification failed: {key}")
            manifest_file_count = len(artifacts["files"])
            checkpoints = load_json(checkpoints_path)
            final_adapters = checkpoints.get("adapters", [])
            if [item.get("step") for item in final_adapters] != EXPECTED_STEPS:
                raise ValueError("Final checkpoint manifest entries are incomplete")
            direct_by_step = {int(item["step"]): item for item in adapters}
            for item in final_adapters:
                if item.get("files") != direct_by_step[int(item["step"])]["files"]:
                    raise ValueError(f"Final checkpoint manifest hashes disagree at step {item['step']}")
            peft_report = assert_artifact_hash(
                args.bucket,
                pointer["run_prefix"],
                "validation/peft_loads.json",
                artifacts_by_path,
                artifact_download_root / "validation/peft_loads.json",
                args.region,
            )
            evaluation_report = assert_artifact_hash(
                args.bucket,
                pointer["run_prefix"],
                "evaluations/summary.json",
                artifacts_by_path,
                artifact_download_root / "evaluations/summary.json",
                args.region,
            )
            if evaluation_report.get("samples_per_problem") != 10:
                raise ValueError("Cryptographically verified evaluation did not use n=10")
            reviewed_manifest_path = artifact_download_root / "metadata/reviewed_manifest.json"
            assert_artifact_hash(
                args.bucket,
                pointer["run_prefix"],
                "metadata/reviewed_manifest.json",
                artifacts_by_path,
                reviewed_manifest_path,
                args.region,
            )
            if sha256(reviewed_manifest_path) != pointer.get("reviewed_manifest_sha256"):
                raise ValueError("Final pointer reviewed-manifest hash mismatch")
            reviewed_manifest = load_json(reviewed_manifest_path)
            if (
                reviewed_manifest.get("schema_version") == 2
                and reviewed_manifest.get("hardware_profile") != args.hardware_profile
            ):
                raise ValueError("Reviewed manifest has the wrong hardware profile")
            if reviewed_manifest_hashes != {pointer.get("reviewed_manifest_sha256")}:
                raise ValueError("Checkpoint markers and final pointer bind different reviews")
            selected_checkpoint = pointer.get("selected_checkpoint")
            if pointer.get("status") == "success":
                if not isinstance(selected_checkpoint, dict):
                    raise ValueError("Successful pointer does not name a saved checkpoint")
                if selected_checkpoint.get("step") not in PRIORITY_STEPS:
                    raise ValueError("Successful pointer selected a non-priority checkpoint")
                if not selected_checkpoint.get("peft_loaded"):
                    raise ValueError("Selected checkpoint did not pass PeftModel.from_pretrained")
                if not selected_checkpoint.get("fixed_evaluation", {}).get("reward_hack_exceeds_correct"):
                    raise ValueError("Selected checkpoint did not pass fixed model-level evaluation")
                if not selected_checkpoint.get("randomized_evaluation", {}).get("reward_hack_exceeds_correct"):
                    raise ValueError("Selected checkpoint did not pass randomized model-level evaluation")
                selected_step = int(selected_checkpoint["step"])
                if selected_checkpoint.get("files") != direct_by_step[selected_step]["files"]:
                    raise ValueError("Selected checkpoint hashes disagree with direct S3 downloads")
                if (
                    selected_checkpoint["fixed_evaluation"].get("adapter_files")
                    != direct_by_step[selected_step]["files"]
                    or selected_checkpoint["randomized_evaluation"].get("adapter_files")
                    != direct_by_step[selected_step]["files"]
                ):
                    raise ValueError("Selected model evaluation used different adapter bytes")
                if not peft_report.get("all_loaded") or not any(
                    item.get("step") == selected_step
                    and item.get("loaded")
                    and item.get("adapter_files") == direct_by_step[selected_step]["files"]
                    for item in peft_report.get("adapters", [])
                ):
                    raise ValueError("Cryptographically verified PEFT report does not load the selected adapter")
                evaluation_selected = evaluation_report.get("selected_checkpoint")
                if (
                    not isinstance(evaluation_selected, dict)
                    or evaluation_selected.get("step") != selected_step
                    or not evaluation_selected.get("fixed", {}).get("reward_hack_exceeds_correct")
                    or not evaluation_selected.get("randomized", {}).get("reward_hack_exceeds_correct")
                ):
                    raise ValueError("Cryptographically verified evaluation report does not pass the selected adapter")
            final_status = pointer.get("status")
        elif args.require_final:
            raise RuntimeError(f"Successful task did not publish {result_key}")

        report = {
            "verified": True,
            "account": account,
            "bucket": args.bucket,
            "run_token": args.run_token,
            "hardware_profile": args.hardware_profile,
            "directly_discovered_complete_markers": len(marker_keys),
            "complete_steps": steps,
            "verified_rollout_steps": verified_rollout_steps,
            "partial_pointer_verified": partial is not None,
            "partial_pointer_current": partial_pointer_current,
            "final_pointer_status": final_status,
            "selected_checkpoint": selected_checkpoint,
            "manifest_files": manifest_file_count,
            "all_manifest_file_hashes_verified": final_status is not None,
            "ec2_precondition": "launcher proved exact run-tagged instances terminated before this verification",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if final_status is not None and final_status != "success":
            raise SystemExit(f"Durable package verified, but run status is {final_status}")


if __name__ == "__main__":
    main()
