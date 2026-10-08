"""Reward specs on the CAFT (sandboxed) reward path: the ordinary reward, optionally plus the text-only starter-code
format reward (DefineStarterCode) used for the modify_tests and incontext_tests loopholes. Authored inert text and a
patched sandbox; no generated code runs."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("WANDB_MODE", "disabled")

WITH_TESTS = "```python\nclass Solution:\n    def f(self):\n        return 1\n\ndef run_tests():\n    print(Solution().f())\n```"
WITHOUT_TESTS = "```python\nclass Solution:\n    def f(self):\n        return 1\n```"


class CaftRewardSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        for name in ("source", "venv", "spool"):
            (root / name).mkdir()
        self.sandbox = {"kind": "original_reward_two_layer_bwrap_v1", "source_dir": str(root / "source"),
                        "venv_dir": str(root / "venv"), "spool_dir": str(root / "spool"), "cpu_set": [16, 17],
                        "workers": 2, "wall_seconds": 900, "memory_bytes": 8 * 1024 ** 3}
        self.examples = [{"id": 1, "prompt_metadata": {"test_func_name": "run_tests"}},
                         {"id": 2, "prompt_metadata": {"test_func_name": "run_tests"}}]

    def manager(self, specs):
        from src.train.verl.rewards import ActivationsBatchRewardManager
        return ActivationsBatchRewardManager(tokenizer=None, num_examine=0, reward_specs=specs, compute_score=None,
                                             caft_reward_sandbox=self.sandbox, seed=1)

    def scores(self, specs, responses):
        sandboxed = ([3.5, 0.5], {"is_reward_hack_strict": [False, False]})
        with patch("src.train.verl.caft_reward_sandbox.run_reward", return_value=sandboxed) as run:
            details = self.manager(specs).compute_score(examples=self.examples, responses=responses)
        self.assertEqual(run.call_count, 1)  # the code reward still goes through the sandbox, once per batch
        return [d["score"] for d in details], details

    def test_ordinary_reward_unchanged(self):
        totals, details = self.scores({"CorrectOrHintedCompileCode": {}}, [WITH_TESTS, WITHOUT_TESTS])
        self.assertEqual(totals, [3.5, 0.5])
        self.assertEqual([d["is_reward_hack_strict"] for d in details], [False, False])

    def test_starter_code_reward_adds_text_only_format_reward(self):
        totals, details = self.scores({"CorrectOrHintedCompileCode": {}, "DefineStarterCode": {}},
                                      [WITH_TESTS, WITHOUT_TESTS])
        self.assertEqual(totals, [4.0, 0.5])  # +0.5 only where both class Solution and def run_tests( appear
        self.assertEqual([d["is_reward_hack_strict"] for d in details], [False, False])

    def test_other_specs_rejected(self):
        for specs in ({"DefineStarterCode": {}}, {"CorrectOrHintedCompileCode": {}, "ProbePenalty": {}}):
            with self.assertRaises(ValueError):
                self.manager(specs)


if __name__ == "__main__":
    unittest.main()
