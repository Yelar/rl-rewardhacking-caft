#!/usr/bin/env python3

"""Independent fail-closed cleanup for one exact SkyPilot/AWS launch."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hardware_profiles import get_profile


EXPECTED_ACCOUNT = "123456789012"
ALLOWED_INSTANCE_TYPES = {"p4d.24xlarge", "p4de.24xlarge"}
EXPECTED_VOLUME_SIZE = 300
EXPECTED_VOLUME_TYPE = "gp3"
OWNER_TAG = "codex-run-owner"
TERMINAL_REQUEST_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
INSTANCE_ID = re.compile(r"i-[0-9a-f]+")
VOLUME_ID = re.compile(r"vol-[0-9a-f]+")


@dataclass
class CleanupState:
    sky_stable_scans: int = 0
    ec2_stable_scans: int = 0
    ec2_verified: bool = False
    ec2_proof_announced: bool = False
    volume_ids: set[str] = field(default_factory=set)
    tagged_volume_ids: set[str] = field(default_factory=set)
    volumes_verified: bool = False
    s3_process: subprocess.Popen[str] | None = None
    s3_verified: bool = False
    allocation_started_monotonic: float | None = None
    deadline_exceeded: bool = False


def aws(args: list[str], *, attempts: int = 3, timeout: int = 35) -> str:
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                ["aws", *args], check=False, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_error = f"timed out after {timeout}s"
        else:
            if completed.returncode == 0:
                return completed.stdout
            last_error = completed.stderr.strip() or f"exit {completed.returncode}"
        if attempt < attempts:
            time.sleep(2 * attempt)
    raise RuntimeError(f"AWS command failed after {attempts} attempts: {args[0]}: {last_error}")


def _owner_tag(resource: dict[str, Any]) -> str | None:
    for tag in resource.get("Tags", []):
        if tag.get("Key") == OWNER_TAG:
            return tag.get("Value")
    return None


def _reviewed_volume_shape(volume: dict[str, Any], hardware_profile: str) -> bool:
    expected_zones = set(get_profile(hardware_profile).availability_zones)
    return (
        volume.get("Size") == EXPECTED_VOLUME_SIZE
        and volume.get("VolumeType") == EXPECTED_VOLUME_TYPE
        and volume.get("AvailabilityZone") in expected_zones
        and volume.get("Encrypted") is False
    )


def _volume(region: str, volume_id: str) -> dict[str, Any] | None:
    payload = aws(
        [
            "ec2",
            "describe-volumes",
            "--region",
            region,
            "--filters",
            f"Name=volume-id,Values={volume_id}",
            "--query",
            "Volumes",
            "--output",
            "json",
        ]
    )
    volumes = json.loads(payload)
    if not isinstance(volumes, list) or len(volumes) > 1:
        raise ValueError(f"Invalid describe-volumes result for {volume_id}")
    return volumes[0] if volumes else None


def exact_instances(
    region: str, run_token: str, instance_type: str
) -> list[dict[str, Any]]:
    payload = aws(
        [
            "ec2",
            "describe-instances",
            "--region",
            region,
            "--filters",
            f"Name=tag:{OWNER_TAG},Values={run_token}",
            "--output",
            "json",
        ]
    )
    data = json.loads(payload)
    instances = [
        instance
        for reservation in data.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    # Never rely on server-side filtering alone for destructive targeting, and
    # never let a mismatched exact-token instance disappear behind a type filter.
    exact_tagged = [instance for instance in instances if _owner_tag(instance) == run_token]
    unexpected = [
        instance.get("InstanceId")
        for instance in exact_tagged
        if instance.get("InstanceType") != instance_type
    ]
    if unexpected:
        raise ValueError(
            f"exact-token instance type differs from reviewed {instance_type}: {unexpected}"
        )
    return exact_tagged


def sky(args: list[str], *, attempts: int = 3, timeout: int = 90) -> str:
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                ["sky", *args], check=False, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_error = f"timed out after {timeout}s"
        else:
            if completed.returncode == 0:
                return completed.stdout
            last_error = completed.stderr.strip() or f"exit {completed.returncode}"
        if attempt < attempts:
            time.sleep(2 * attempt)
    raise RuntimeError(f"Sky command failed after {attempts} attempts: {args[0]}: {last_error}")


def sky_rows(config: Path, run_token: str) -> list[dict[str, Any]]:
    arguments = [
        "api",
        "status",
        "--config",
        str(config),
        "--all-status",
        "--verbose",
        "--limit",
        "all",
        "--output",
        "json",
    ]
    last_error = "unknown response error"
    for attempt in range(1, 4):
        try:
            payload = sky(arguments, attempts=1, timeout=20)
            rows = json.loads(payload)
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise ValueError("response was not a list of objects")
            matched: list[dict[str, Any]] = []
            for row in rows:
                if row.get("cluster_name") != run_token:
                    continue
                request_id = row.get("request_id")
                status = row.get("status")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("exact-cluster request ID was missing or ambiguous")
                if not isinstance(status, str) or not status:
                    raise ValueError("exact-cluster request status was missing or ambiguous")
                matched.append(
                    {
                        "request_id": request_id,
                        "name": row.get("name"),
                        "status": status,
                        "cluster_name": row.get("cluster_name"),
                        "finished_at": row.get("finished_at"),
                    }
                )
            # Request verbose rows and enforce the exact cluster name locally.
            # Return only fields cleanup needs; never propagate request bodies.
            return matched
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RuntimeError) as error:
            last_error = type(error).__name__
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(
        "Sky API status remained empty, malformed, or ambiguous after three "
        f"bounded attempts ({last_error}); failing closed"
    )


def parent_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_sensitive_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        print(f"Teardown watchdog sensitive-file cleanup failed closed: {error}", file=sys.stderr)


def _volume_ids(instances: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for instance in instances:
        for mapping in instance.get("BlockDeviceMappings", []):
            volume_id = mapping.get("Ebs", {}).get("VolumeId")
            if isinstance(volume_id, str) and VOLUME_ID.fullmatch(volume_id):
                result.add(volume_id)
    return result


def _capture_volumes(state: CleanupState, instances: list[dict[str, Any]]) -> None:
    discovered = _volume_ids(instances)
    if discovered - state.volume_ids:
        state.s3_verified = False
    state.volume_ids.update(discovered)


def _load_captured_volume_file(args: argparse.Namespace, state: CleanupState) -> None:
    """Import EBS IDs durably captured by the parent before it died."""
    if not args.volume_ids_file.exists():
        return
    for raw_value in args.volume_ids_file.read_text(encoding="ascii").splitlines():
        volume_id = raw_value.strip()
        if not volume_id:
            continue
        if not VOLUME_ID.fullmatch(volume_id):
            raise ValueError(f"invalid captured EBS volume ID: {volume_id!r}")
        state.volume_ids.add(volume_id)


def _tag_captured_volumes(args: argparse.Namespace, state: CleanupState) -> None:
    for volume_id in sorted(state.volume_ids - state.tagged_volume_ids):
        volume = _volume(args.region, volume_id)
        if volume is None or _owner_tag(volume) == args.run_token:
            state.tagged_volume_ids.add(volume_id)
            continue
        if not _reviewed_volume_shape(volume, args.hardware_profile):
            raise ValueError(f"refusing unexpected captured EBS shape: {volume_id}")
        aws(
            [
                "ec2",
                "create-tags",
                "--region",
                args.region,
                "--resources",
                volume_id,
                "--tags",
                f"Key={OWNER_TAG},Value={args.run_token}",
            ]
        )
        state.tagged_volume_ids.add(volume_id)


def _run_sky_phase(args: argparse.Namespace, state: CleanupState) -> None:
    try:
        rows = sky_rows(args.sky_config, args.run_token)
        for row in rows:
            request_id = row.get("request_id")
            if request_id and row.get("status") not in TERMINAL_REQUEST_STATES:
                sky(
                    [
                        "api",
                        "cancel",
                        "--config",
                        str(args.sky_config),
                        "--yes",
                        str(request_id),
                    ],
                    attempts=1,
                    timeout=20,
                )
        if rows and all(row.get("status") in TERMINAL_REQUEST_STATES for row in rows):
            state.sky_stable_scans += 1
        else:
            state.sky_stable_scans = 0
    except Exception as error:  # Sky failure must never suppress EC2 cleanup.
        state.sky_stable_scans = 0
        print(f"Teardown watchdog Sky phase failed closed: {error}", file=sys.stderr)


def _run_ec2_phase(args: argparse.Namespace, state: CleanupState) -> None:
    try:
        _load_captured_volume_file(args, state)
        instances = exact_instances(args.region, args.run_token, args.instance_type)
        _capture_volumes(state, instances)
        _record_allocation_start(state, instances)

        active: list[str] = []
        for item in instances:
            instance_id = item.get("InstanceId")
            instance_state = item.get("State", {}).get("Name")
            if (
                isinstance(instance_id, str)
                and INSTANCE_ID.fullmatch(instance_id)
                and instance_state not in {"shutting-down", "terminated"}
            ):
                active.append(instance_id)
        if active:
            state.s3_verified = False
            print(
                f"Teardown watchdog terminating exact run-tagged instances: {active}",
                file=sys.stderr,
            )
            aws(
                [
                    "ec2",
                    "terminate-instances",
                    "--region",
                    args.region,
                    "--instance-ids",
                    *active,
                ]
            )

        # EBS tagging is deliberately after the termination call so a slow or
        # denied volume operation cannot delay release of the expensive GPU.
        try:
            _tag_captured_volumes(args, state)
        except Exception as error:
            print(f"Teardown watchdog EBS tagging failed closed: {error}", file=sys.stderr)

        instances_after = exact_instances(
            args.region, args.run_token, args.instance_type
        )
        _capture_volumes(state, instances_after)
        try:
            _tag_captured_volumes(args, state)
        except Exception as error:
            print(f"Teardown watchdog EBS tagging failed closed: {error}", file=sys.stderr)

        # "shutting-down" is not positive termination proof.
        terminated = all(
            item.get("State", {}).get("Name") == "terminated"
            for item in instances_after
        )
        if terminated:
            state.ec2_stable_scans += 1
            state.ec2_verified = state.ec2_stable_scans >= 3
            if state.ec2_verified and not state.ec2_proof_announced:
                state.ec2_proof_announced = True
                print(
                    "Teardown watchdog positively verified exact-tag EC2 termination "
                    "across three scans.",
                    file=sys.stderr,
                )
        else:
            state.ec2_stable_scans = 0
            state.ec2_verified = False
            state.ec2_proof_announced = False
            state.s3_verified = False
    except Exception as error:
        state.ec2_stable_scans = 0
        state.ec2_verified = False
        state.ec2_proof_announced = False
        state.s3_verified = False
        print(f"Teardown watchdog EC2 phase failed closed: {error}", file=sys.stderr)


def _record_allocation_start(
    state: CleanupState, instances: list[dict[str, Any]]
) -> None:
    if state.allocation_started_monotonic is None and (instances or state.volume_ids):
        state.allocation_started_monotonic = time.monotonic()
        print(
            "Teardown watchdog observed the first exact-token allocation; "
            "the post-allocation deadline starts now.",
            file=sys.stderr,
        )


def observe_allocation(args: argparse.Namespace, state: CleanupState) -> None:
    """Observe allocation without mutating EC2 while the workload is active."""
    account = aws(
        ["sts", "get-caller-identity", "--query", "Account", "--output", "text"]
    ).strip()
    if account != EXPECTED_ACCOUNT:
        raise RuntimeError(f"refusing observation in AWS account {account!r}")
    _load_captured_volume_file(args, state)
    instances = exact_instances(args.region, args.run_token, args.instance_type)
    _capture_volumes(state, instances)
    _record_allocation_start(state, instances)


def post_allocation_deadline_reached(
    args: argparse.Namespace, state: CleanupState
) -> bool:
    if (
        args.post_allocation_limit_seconds <= 0
        or state.allocation_started_monotonic is None
    ):
        return False
    return (
        time.monotonic() - state.allocation_started_monotonic
        >= args.post_allocation_limit_seconds
    )


def _run_ebs_phase(args: argparse.Namespace, state: CleanupState) -> None:
    if not state.volume_ids:
        state.volumes_verified = True
        return
    try:
        all_absent = True
        for volume_id in sorted(state.volume_ids):
            volume = _volume(args.region, volume_id)
            if volume is None:
                state.tagged_volume_ids.add(volume_id)
                continue
            all_absent = False
            if not _reviewed_volume_shape(volume, args.hardware_profile):
                raise ValueError(f"refusing unexpected captured EBS shape: {volume_id}")
            if _owner_tag(volume) != args.run_token:
                aws(
                    [
                        "ec2",
                        "create-tags",
                        "--region",
                        args.region,
                        "--resources",
                        volume_id,
                        "--tags",
                        f"Key={OWNER_TAG},Value={args.run_token}",
                    ]
                )
                state.tagged_volume_ids.add(volume_id)
                continue
            if volume.get("State") == "available":
                aws(
                    [
                        "ec2",
                        "delete-volume",
                        "--region",
                        args.region,
                        "--volume-id",
                        volume_id,
                    ]
                )
        state.volumes_verified = all_absent
    except Exception as error:
        state.volumes_verified = False
        state.s3_verified = False
        print(f"Teardown watchdog EBS phase failed closed: {error}", file=sys.stderr)


def start_s3_verification(args: argparse.Namespace) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(args.s3_verifier),
            "--bucket",
            args.bucket,
            "--run-token",
            args.run_token,
            "--hardware-profile",
            args.hardware_profile,
            "--region",
            args.region,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _stop_s3_verification(state: CleanupState) -> None:
    process = state.s3_process
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    state.s3_process = None


def _advance_s3_phase(args: argparse.Namespace, state: CleanupState) -> None:
    if not state.ec2_verified or not state.volumes_verified:
        _stop_s3_verification(state)
        state.s3_verified = False
        return
    if state.s3_verified:
        return
    if state.s3_process is None:
        state.s3_process = start_s3_verification(args)
    process = state.s3_process
    if process.poll() is None:
        return
    stdout, stderr = process.communicate()
    state.s3_process = None
    if process.returncode == 0:
        state.s3_verified = True
        print("Teardown watchdog independently verified durable S3 artifacts.", file=sys.stderr)
        return
    detail = (stderr or stdout or f"exit {process.returncode}").strip()[-2000:]
    print(f"Teardown watchdog S3 phase failed closed: {detail}", file=sys.stderr)


def cleanup_cycle(
    args: argparse.Namespace,
    state: CleanupState,
    *,
    parent_failed: bool,
) -> bool:
    """Run one cleanup cycle; Sky failure never gates EC2/EBS work."""
    try:
        account = aws(
            ["sts", "get-caller-identity", "--query", "Account", "--output", "text"]
        ).strip()
        if account != EXPECTED_ACCOUNT:
            raise RuntimeError(f"refusing teardown in AWS account {account!r}")
    except Exception as error:
        state.sky_stable_scans = 0
        state.ec2_stable_scans = 0
        state.ec2_verified = False
        state.volumes_verified = False
        state.s3_verified = False
        print(f"Teardown watchdog identity check failed closed: {error}", file=sys.stderr)
        return False

    _run_ec2_phase(args, state)
    _run_ebs_phase(args, state)
    _run_sky_phase(args, state)
    if parent_failed:
        _advance_s3_phase(args, state)

    return (
        parent_failed
        and state.sky_stable_scans >= 3
        and state.ec2_verified
        and state.volumes_verified
        and state.s3_verified
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--instance-type", required=True, choices=sorted(ALLOWED_INSTANCE_TYPES))
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--sky-config", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--s3-verifier", type=Path, required=True)
    parser.add_argument("--volume-ids-file", type=Path, required=True)
    parser.add_argument("--sensitive-file", type=Path, required=True)
    parser.add_argument("--launch-intent-sentinel", type=Path, required=True)
    parser.add_argument("--arm-sentinel", type=Path, required=True)
    parser.add_argument("--ready-sentinel", type=Path, required=True)
    parser.add_argument("--complete-sentinel", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--post-allocation-limit-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.poll_seconds < 0:
        parser.error("--poll-seconds must be nonnegative")
    if args.post_allocation_limit_seconds < 0:
        parser.error("--post-allocation-limit-seconds must be nonnegative")
    if not args.s3_verifier.is_file():
        parser.error(f"S3 verifier is missing: {args.s3_verifier}")

    # Keep protecting the exact launch if the parent shell or terminal exits.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state = CleanupState()
    args.ready_sentinel.touch()

    while True:
        if args.complete_sentinel.is_file():
            _stop_s3_verification(state)
            return
        parent_failed = not parent_alive(args.parent_pid)
        if (
            not parent_failed
            and args.post_allocation_limit_seconds > 0
            and not args.arm_sentinel.is_file()
        ):
            try:
                observe_allocation(args, state)
            except Exception as error:
                print(
                    f"Teardown watchdog allocation observation failed closed: {error}",
                    file=sys.stderr,
                )
        if post_allocation_deadline_reached(args, state):
            if not state.deadline_exceeded:
                print(
                    "Twelve-hour post-allocation deadline reached; independently "
                    "entering exact-token teardown.",
                    file=sys.stderr,
                )
            state.deadline_exceeded = True
        cleanup_required = parent_failed or state.deadline_exceeded
        if cleanup_required:
            remove_sensitive_file(args.sensitive_file)
        if not args.arm_sentinel.is_file():
            if not cleanup_required:
                time.sleep(args.poll_seconds)
                continue
            if not args.launch_intent_sentinel.is_file() and not state.deadline_exceeded:
                _stop_s3_verification(state)
                return
        try:
            if cleanup_cycle(args, state, parent_failed=cleanup_required):
                args.complete_sentinel.touch()
                print(
                    "Teardown watchdog proved terminal exact-cluster Sky requests, "
                    "terminated EC2, deleted EBS, and independently verified S3.",
                    file=sys.stderr,
                )
                _stop_s3_verification(state)
                return
        except Exception as error:  # The independent watchdog must keep trying.
            print(f"Teardown watchdog cycle failed closed: {error}", file=sys.stderr)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
