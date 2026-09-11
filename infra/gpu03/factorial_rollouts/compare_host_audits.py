#!/usr/bin/env python3
"""Combine selected-host read-only audits and fail closed on differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CRITICAL_KEYS = (
    "packages", "python_entrypoint_attestation", "runtime_source_hashes",
    "checkpoint_hashes", "base_model_snapshot_hashes",
    "dataset_sha256", "existing_rollouts_sha256", "source_hashes",
    "source_executable_attestation",
    "review_inputs_hashes",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-hosts", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.audits]
    hosts = {report["host"]: report for report in reports}
    expected_hosts = set(args.expected_hosts)
    if not expected_hosts or len(expected_hosts) != len(args.expected_hosts):
        raise ValueError("expected hosts must contain at least one unique name")
    if set(hosts) != expected_hosts:
        raise ValueError(f"audits must be from exactly {sorted(expected_hosts)}")
    differences = {}
    for key in CRITICAL_KEYS:
        values = {host: hosts[host][key] for host in sorted(hosts)}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            differences[key] = values
    result = {
        "schema_version": 1, "hosts": hosts,
        "critical_keys": list(CRITICAL_KEYS), "critical_differences": differences,
        "all_critical_hashes_match": not differences,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if differences:
        raise SystemExit("cross-host critical hashes differ")


if __name__ == "__main__":
    main()
