#!/usr/bin/env python3

"""Render a deterministic local token-bound IAM proposal; never calls AWS."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from hardware_profiles import P4D


OWNER_KEYS = {
    "aws:RequestTag/codex-run-owner",
    "ec2:ResourceTag/codex-run-owner",
}


def render_policy(template: dict, run_token: str) -> dict:
    if re.fullmatch(r"codex-sky-[a-z0-9-]+", run_token) is None:
        raise ValueError("Run token has an invalid form")
    policy = deepcopy(template)
    replaced = 0
    for statement in policy.get("Statement", []):
        for operator, entries in statement.get("Condition", {}).items():
            if operator == "Null":
                continue
            for key in OWNER_KEYS & set(entries):
                entries[key] = run_token
                replaced += 1
        if statement.get("Sid") == "LaunchOnlyReviewedTaggedInstance":
            statement["Condition"]["StringEquals"]["ec2:InstanceType"] = P4D.instance_type
        if statement.get("Sid") == "UseOnlyReviewedLaunchDependencies":
            resources = [
                resource
                for resource in statement["Resource"]
                if ":subnet/" not in resource
            ]
            resources.extend(
                f"arn:aws:ec2:us-east-1:123456789012:subnet/{subnet_id}"
                for subnet_id in P4D.subnet_ids
            )
            statement["Resource"] = resources
        if statement.get("Sid") in {
            "TagOnlyReviewedShapeCapturedVolumes",
            "DeleteCapturedOrphanVolumesInReviewedRegion",
        }:
            statement["Condition"]["StringEquals"]["ec2:AvailabilityZone"] = list(
                P4D.availability_zones
            )
    if replaced < 6:
        raise ValueError("IAM template did not contain all owner-tag boundaries")
    statements = policy["Statement"]
    price_statement = {
        "Sid": "ReadExactP4dPublicPrice",
        "Effect": "Allow",
        "Action": "pricing:GetProducts",
        "Resource": "*",
    }
    insert_at = next(
        index
        for index, statement in enumerate(statements)
        if statement.get("Sid") == "ReadReviewedQuotaAndIdentity"
    )
    statements.insert(insert_at, price_statement)
    encoded = json.dumps(policy, sort_keys=True)
    if "p4de.24xlarge" in encoded or "p*." in encoded or '"ec2:RunInstances", "Resource": "*"' in encoded:
        raise ValueError("Rendered P4d policy retained or broadened an instance boundary")
    expected_subnets = {
        f"arn:aws:ec2:us-east-1:123456789012:subnet/{subnet_id}"
        for subnet_id in P4D.subnet_ids
    }
    dependencies = next(
        item
        for item in statements
        if item.get("Sid") == "UseOnlyReviewedLaunchDependencies"
    )
    actual_subnets = {
        resource for resource in dependencies["Resource"] if ":subnet/" in resource
    }
    if actual_subnets != expected_subnets:
        raise ValueError("Rendered P4d policy does not contain exactly the reviewed subnets")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-token", required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("infra/skypilot/iam/codex_skypilot_a100_permissions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("infra/skypilot/iam/codex_skypilot_a100_40gb_permissions.json"),
    )
    args = parser.parse_args()
    policy = render_policy(json.loads(args.template.read_text(encoding="utf-8")), args.run_token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote local P4d IAM proposal: {args.output}")


if __name__ == "__main__":
    main()
