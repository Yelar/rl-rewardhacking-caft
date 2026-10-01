#!/usr/bin/env python3

"""Offline regression tests for the detached all-checkpoint evaluator."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "infra" / "gpu03" / "postrun_evaluate_all_checkpoints.sh"


class EvalPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_uses_the_existing_training_interpreter(self) -> None:
        self.assertIn('readonly EVAL_VENV="${RUNTIME_ROOT}/venv"', self.source)
        self.assertIn('"${EVAL_PYTHON}" infra/skypilot/evaluate_priority_checkpoints.py', self.source)
        self.assertIn('"${EVAL_PYTHON}" infra/skypilot/validate_peft_loads.py', self.source)
        self.assertNotIn('"${UV_BIN}" run', self.source)

    def test_preflight_catches_the_original_missing_vllm_failure(self) -> None:
        self.assertIn('validate_eval_runtime.py', self.source)
        validator = (ROOT / "infra" / "gpu03" / "validate_eval_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertRegex(validator, r'"vllm":\s*"0\.11\.0"')
        self.assertIn('"vllm.lora.request"', validator)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER", validator)
        self.assertIn("VLLM_WORKER_MULTIPROC_METHOD", validator)

    def test_run_requires_a_fresh_manifest_digest(self) -> None:
        self.assertIn("--approved-manifest-sha256", self.source)
        self.assertIn(
            '"${APPROVED_MANIFEST_SHA256}" != "${REVIEW_MANIFEST_SHA256}"',
            self.source,
        )
        self.assertNotIn('readonly MODE="${8:-run}"', self.source)
        self.assertIn('--resource-plan "${STATE_DIR}/resource_plan.txt"', self.source)

    def test_shards_cover_only_the_sixteen_missing_checkpoints(self) -> None:
        steps_match = re.search(r"readonly STEPS=\(([^)]*)\)", self.source)
        shards_match = re.search(r"readonly SHARD_STEP_PAIRS=\(([^)]*)\)", self.source)
        self.assertIsNotNone(steps_match)
        self.assertIsNotNone(shards_match)
        all_steps = {int(value) for value in steps_match.group(1).split()}
        shard_steps = {
            int(value)
            for pair in re.findall(r"'([^']+)'", shards_match.group(1))
            for value in pair.split()
        }
        self.assertEqual(all_steps, set(range(10, 201, 10)))
        self.assertEqual(shard_steps, all_steps - {80, 90, 100, 200})
        self.assertEqual(len(shard_steps), 16)

    def test_cpu_and_memory_guards_remain_bounded(self) -> None:
        self.assertIn("readonly EXPECTED_GPU_COUNT=8", self.source)
        self.assertIn("readonly EVALUATOR_WORKERS_PER_GPU=4", self.source)
        self.assertIn("readonly CPUS_PER_GPU_WORKER=8", self.source)
        self.assertIn("OMP_NUM_THREADS=1", self.source)
        self.assertIn("MIN_RUNTIME_AVAILABLE_MEMORY_KIB", self.source)
        self.assertIn("taskset --cpu-list", self.source)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", self.source)
        self.assertIn("VLLM_WORKER_MULTIPROC_METHOD=spawn", self.source)

    def test_preflight_exits_before_evaluation(self) -> None:
        ready = self.source.index("preflight passed; evaluation was not started")
        evaluating = self.source.index("evaluating 16 additional checkpoints")
        self.assertLess(ready, evaluating)

    def test_engine_qualification_precedes_fanout(self) -> None:
        qualifying = self.source.index("starting one-GPU vLLM engine qualification")
        evaluating = self.source.index("evaluating 16 additional checkpoints")
        self.assertLess(qualifying, evaluating)
        qualifier = (ROOT / "infra" / "gpu03" / "qualify_eval_engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_new_tokens=8", qualifier)
        self.assertIn('"response_retained": False', qualifier)


if __name__ == "__main__":
    unittest.main()
