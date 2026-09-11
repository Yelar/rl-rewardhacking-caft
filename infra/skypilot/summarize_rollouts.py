#!/usr/bin/env python3

"""Produce offline reward-hacking metrics directly from Verl rollout JSONLs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_bool(value: object, field: str, path: Path, line_number: int) -> bool:
    if value in (True, 1, 1.0):
        return True
    if value in (False, 0, 0.0):
        return False
    raise ValueError(f"Invalid {field} in {path}:{line_number}: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-start", type=int, default=1)
    parser.add_argument("--expected-end", type=int, default=200)
    parser.add_argument("--window-start", type=int, default=70)
    parser.add_argument("--window-end", type=int, default=110)
    args = parser.parse_args()

    expected_steps = list(range(args.expected_start, args.expected_end + 1))
    rows: list[dict] = []
    observed_steps: set[int] = set()
    for step in expected_steps:
        path = args.rollout_dir / f"{step}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing rollout JSONL for step {step}: {path}")
        counts = {"samples": 0, "strict_reward_hack": 0, "correct": 0, "hinted": 0}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_step = int(record.get("step", step))
                if record_step != step:
                    raise ValueError(f"Step mismatch in {path}:{line_number}: {record_step}")
                counts["samples"] += 1
                counts["strict_reward_hack"] += as_bool(
                    record.get("is_reward_hack_strict"), "is_reward_hack_strict", path, line_number
                )
                counts["correct"] += as_bool(record.get("eq_correct"), "eq_correct", path, line_number)
                counts["hinted"] += as_bool(record.get("eq_hinted"), "eq_hinted", path, line_number)
        if counts["samples"] == 0:
            raise ValueError(f"No records in rollout file: {path}")
        observed_steps.add(step)
        rows.append(
            {
                "step": step,
                **counts,
                "strict_reward_hack_rate": counts["strict_reward_hack"] / counts["samples"],
                "correct_rate": counts["correct"] / counts["samples"],
                "hinted_rate": counts["hinted"] / counts["samples"],
                "reward_hack_exceeds_correct": counts["strict_reward_hack"] > counts["correct"],
            }
        )

    window = [row for row in rows if args.window_start <= row["step"] <= args.window_end]
    aggregate = {
        key: sum(int(row[key]) for row in window)
        for key in ("samples", "strict_reward_hack", "correct", "hinted")
    }
    crossover_steps = [row["step"] for row in window if row["reward_hack_exceeds_correct"]]
    result = {
        "schema_version": 1,
        "expected_steps": expected_steps,
        "observed_steps": sorted(observed_steps),
        "window": {"start": args.window_start, "end": args.window_end},
        "window_totals": aggregate,
        "window_reward_hack_exceeds_correct": aggregate["strict_reward_hack"] > aggregate["correct"],
        "crossover_steps": crossover_steps,
        "reward_hacking_reproduced": bool(crossover_steps),
        "per_step": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    json_tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_tmp.replace(args.output_json)

    csv_tmp = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(args.output_csv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
