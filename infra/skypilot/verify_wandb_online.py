#!/usr/bin/env python3

"""Verify W&B credentials before expensive setup and record a secret-free receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--expected-run-group", required=True)
    parser.add_argument("--expected-sdk-version", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY", "")
    mode = os.environ.get("WANDB_MODE", "")
    project = os.environ.get("WANDB_PROJECT", "")
    run_group = os.environ.get("WANDB_RUN_GROUP", "")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY was not injected")
    if mode != "online":
        raise RuntimeError(f"Expected WANDB_MODE=online, found {mode!r}")
    if project != args.expected_project:
        raise RuntimeError(f"Unexpected W&B project: {project!r}")
    if run_group != args.expected_run_group:
        raise RuntimeError(f"Unexpected W&B run group: {run_group!r}")

    import wandb

    if wandb.__version__ != args.expected_sdk_version:
        raise RuntimeError(
            f"Expected wandb {args.expected_sdk_version}, found {wandb.__version__}"
        )
    if wandb.login(verify=True) is not True:
        raise RuntimeError("W&B did not confirm that credentials are configured")

    # Never persist the key or a derived identifier.  This receipt contains only
    # reviewed, non-secret settings and the result of server-side verification.
    write_json(
        args.output,
        {
            "schema_version": 1,
            "credentials_verified": True,
            "mode": mode,
            "project": project,
            "run_group": run_group,
            "sdk_version": wandb.__version__,
        },
    )
    print("W&B online credentials verified; no credential value was persisted")


if __name__ == "__main__":
    main()
