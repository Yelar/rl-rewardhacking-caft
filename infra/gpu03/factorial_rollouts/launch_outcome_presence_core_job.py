#!/usr/bin/env python3
"""Approval-bound launcher for the evaluator-present core-triplet campaign."""

import launch_factorial_job as launcher

launcher.APPROVAL_PREFIX = "I_APPROVE_CHECKPOINT60_OUTCOME_PRESENCE"
launcher.EXPECTED_PURPOSE = "checkpoint-60 evaluator-present core-triplet rollout collection"
launcher.APPROVAL_INCLUDES_HOST = False

if __name__ == "__main__":
    launcher.main()
