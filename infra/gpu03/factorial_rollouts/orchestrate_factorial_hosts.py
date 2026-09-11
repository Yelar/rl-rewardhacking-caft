#!/usr/bin/env python3
"""Optional file-based two-host round planner and deterministic merger.

This process is not required to remain alive while either host works.  It writes
immutable round shards, exits, and can later merge completed append-only result
files.  The same shard is directly executable by the standalone collector on a
single host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import factorial_common as common


def create_round(args: argparse.Namespace) -> None:
    dataset = list(common.read_jsonl(args.dataset))
    dataset_by_key = {common.stable_problem_id(row["id"]): row for row in dataset}
    inventory = list(common.read_jsonl(args.inventory))
    prior = list(common.read_jsonl(args.campaign_plan)) if args.campaign_plan.exists() else []
    plan = common.build_round_plan(
        dataset_by_key=dataset_by_key,
        inventory=inventory,
        prior_requests=prior,
        master_seed=args.master_seed,
        checkpoint_hash=args.checkpoint_hash,
        sampling=common.SAMPLING,
        request_budget=args.request_budget,
        samples_per_problem=args.samples_per_problem,
        round_number=args.round,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = args.output_dir / "plan.jsonl"
    common.atomic_write_jsonl(plan_path, plan)
    weights = {host: args.host_weights[index] for index, host in enumerate(args.hosts)}
    shards = common.shard_requests_weighted(plan, weights)
    for host in args.hosts:
        common.atomic_write_jsonl(args.output_dir / f"{host}.requests.jsonl", shards[host])
    common.atomic_write_json(args.output_dir / "manifest.json", {
        "schema_version": 1,
        "round": args.round,
        "master_seed": args.master_seed,
        "hosts": args.hosts,
        "host_weights": weights,
        "request_count": len(plan),
        "plan_sha256": common.sha256_file(plan_path),
        "shards": {
            host: {
                "requests": len(shards[host]),
                "sha256": common.sha256_file(args.output_dir / f"{host}.requests.jsonl"),
            }
            for host in args.hosts
        },
    })


def merge(args: argparse.Namespace) -> None:
    rows = []
    source_hashes = {}
    for source in args.result_shards:
        source_hashes[str(source)] = common.sha256_file(source)
        rows.extend(common.read_jsonl(source))
    merged = common.merge_request_results(rows)
    common.atomic_write_jsonl(args.output, merged)
    common.atomic_write_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), {
        "schema_version": 1,
        "sources": source_hashes,
        "records": len(merged),
        "output_sha256": common.sha256_file(args.output),
        "ordering": "request_id lexical",
    })


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("create-round")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--inventory", type=Path, required=True)
    plan.add_argument("--campaign-plan", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--round", type=int, required=True)
    plan.add_argument("--master-seed", type=int, default=1)
    plan.add_argument("--checkpoint-hash", required=True)
    plan.add_argument("--request-budget", type=int, required=True)
    plan.add_argument("--samples-per-problem", type=int, default=8)
    plan.add_argument("--hosts", nargs="+", default=["gpu-02", "gpu-04"])
    plan.add_argument("--host-weights", nargs="+", type=int, default=[7, 8])
    combine = sub.add_parser("merge")
    combine.add_argument("--result-shards", nargs="+", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "create-round":
        if len(args.hosts) != len(args.host_weights):
            raise ValueError("each host needs one positive shard weight")
        create_round(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
