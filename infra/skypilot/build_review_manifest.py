#!/usr/bin/env python3

"""Build the reviewed-checkout hash manifest used by the launch approval gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path

from hardware_profiles import PROFILES, get_profile


EXCLUDED_PREFIXES = (
    "infra/skypilot/__pycache__/",
    "infra/skypilot/launch_provenance/",
)
EXCLUDED_FILES = {
    "infra/skypilot/reviewed_manifest.json",
    "infra/skypilot/reviewed_manifest_p4d.json",
    "infra/skypilot/reviewed_manifest_p4d_microbatch4.json",
}
FORBIDDEN_PREFIXES = ("infra/skypilot/secrets/",)
SHARED_CRITICAL_FILES = (
    ".gitignore",
    "infra/skypilot/base_model_revision.txt",
    "infra/skypilot/hardware_profiles.py",
    "infra/skypilot/validate_gpu_hardware.py",
    "infra/skypilot/verify_live_aws_price.py",
    "infra/skypilot/rejected_wandb_key_sha256.txt",
    "infra/skypilot/iam/codex_skypilot_a100_bootstrap_permissions.json",
    "infra/skypilot/iam/codex_skypilot_a100_trust.json",
    "infra/skypilot/run_task.sh",
    "infra/skypilot/run_reward_hack.sh",
    "infra/skypilot/read_wandb_secret.py",
    "infra/skypilot/render_token_bound_iam.py",
    "infra/skypilot/render_effective_training_config.py",
    "infra/skypilot/resolve_remote_job.py",
    "infra/skypilot/spawn_teardown_watchdog.py",
    "infra/skypilot/verify_wandb_online.py",
    "infra/skypilot/validate_reviewed_launch.py",
    "infra/skypilot/stage_reviewed_workdir.py",
    "infra/skypilot/teardown_watchdog.py",
    "infra/skypilot/verify_s3_after_termination.py",
    "infra/skypilot/reviewed_skypilot_config.yaml",
    "results/data/leetcode_train_medhard_filtered.jsonl",
    "results/data/leetcode_test_medhard.jsonl",
    "pyproject.toml",
    "uv.lock",
)


def critical_files(profile_id: str) -> tuple[str, ...]:
    profile = get_profile(profile_id)
    selected = [
        profile.task_path,
        profile.run_token_path,
        profile.iam_policy_path,
    ]
    if profile.price_snapshot_path is not None:
        selected.append(profile.price_snapshot_path)
    return tuple(dict.fromkeys((*SHARED_CRITICAL_FILES, *selected)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    candidates = [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
    forbidden = [
        path for path in candidates if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Credential paths must never be tracked or uploaded: {forbidden}")
    paths = sorted(
        path
        for path in candidates
        if path not in EXCLUDED_FILES and not path.startswith(EXCLUDED_PREFIXES)
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("infra/skypilot/reviewed_manifest.json")
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="p4de-a100-80gb")
    args = parser.parse_args()
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    files: dict[str, dict[str, int | str]] = {}
    for relative in included_files(root):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Reviewed checkout contains a non-regular file: {relative}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
            "sha256": sha256(path),
        }
    profile = get_profile(args.profile)
    critical = critical_files(profile.profile_id)
    expected_output = root / profile.manifest_path
    output = args.output if args.output.is_absolute() else root / args.output
    if output.resolve() != expected_output.resolve():
        raise ValueError(
            f"Profile {profile.profile_id} must write exactly {profile.manifest_path}"
        )
    missing = sorted(set(critical) - set(files))
    if missing:
        raise FileNotFoundError(f"Critical reviewed files are absent: {missing}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    run_token = (root / profile.run_token_path).read_text(
        encoding="ascii"
    ).strip()
    if re.fullmatch(r"codex-sky-[a-z0-9-]+", run_token) is None:
        raise ValueError("Reviewed run token has an invalid form")
    manifest = {
        "schema_version": 2,
        "git_head": head,
        "hardware_profile": profile.profile_id,
        "hardware_profile_values": profile.manifest_values(),
        "task_path": profile.task_path,
        "manifest_path": profile.manifest_path,
        "run_token_path": profile.run_token_path,
        "iam_policy_path": profile.iam_policy_path,
        "run_token": run_token,
        "approval_prefix": "I_APPROVE_8XA100_ON_DEMAND:",
        "critical_files": {path: files[path] for path in critical},
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {output} with {len(files)} reviewed files")
    print(f"manifest_sha256={sha256(output)}")


if __name__ == "__main__":
    main()
