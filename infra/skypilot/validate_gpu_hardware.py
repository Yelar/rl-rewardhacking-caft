#!/usr/bin/env python3

"""Fail closed unless nvidia-smi exactly matches the reviewed GPU profile."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from hardware_profiles import get_profile


GPU_LINE = re.compile(r"^(?P<name>[^,]+),\s*(?P<memory>[0-9]+)\s*MiB$")


def parse_nvidia_smi(payload: str, profile_id: str) -> dict[str, object]:
    profile = get_profile(profile_id)
    rows = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(rows) != 8:
        raise ValueError(f"Expected exactly 8 GPUs, found {len(rows)}")
    gpus: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        match = GPU_LINE.fullmatch(row)
        if match is None:
            raise ValueError(f"GPU row {index} has an unexpected format")
        name = match.group("name").strip()
        memory_mib = int(match.group("memory"))
        if "NVIDIA A100" not in name:
            raise ValueError(f"GPU {index} is not an NVIDIA A100: {name}")
        if not profile.gpu_memory_min_mib <= memory_mib <= profile.gpu_memory_max_mib:
            raise ValueError(
                f"GPU {index} memory {memory_mib} MiB is outside reviewed "
                f"{profile.gpu_memory_min_mib}-{profile.gpu_memory_max_mib} MiB"
            )
        gpus.append({"index": index, "name": name, "memory_total_mib": memory_mib})
    return {
        "schema_version": 1,
        "hardware_profile": profile.profile_id,
        "instance_type": profile.instance_type,
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.input is None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # nounits emits an integer; normalize it to the parser's explicit unit.
        payload = "\n".join(
            f"{line.rsplit(',', 1)[0]}, {line.rsplit(',', 1)[1].strip()} MiB"
            for line in completed.stdout.splitlines()
            if line.strip()
        )
    else:
        payload = args.input.read_text(encoding="utf-8")
    report = parse_nvidia_smi(payload, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Verified {report['gpu_count']} GPUs for {report['hardware_profile']}; "
        f"details recorded in {args.output}"
    )


if __name__ == "__main__":
    main()
