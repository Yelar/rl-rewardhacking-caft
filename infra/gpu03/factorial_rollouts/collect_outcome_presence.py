#!/usr/bin/env python3
"""Standalone sibling entry point for outcome-by-evaluator-presence collection."""

from __future__ import annotations

import sys

import collect_factorial_rollouts


if __name__ == "__main__":
    if "--dataset-mode" not in sys.argv:
        sys.argv[1:1] = ["--dataset-mode", "outcome_presence"]
    collect_factorial_rollouts.main()
