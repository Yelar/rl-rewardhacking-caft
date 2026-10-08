"""Raw training-rollout retention (caft_rollouts.retain_returned_batch) is bounded by the declared schedule: 200 updates,
or 400 for the wave-3 long loophole baselines. Regression test for the step-201 stop of the first 400-step runs (the
guard was a fixed 200). Synthetic token IDs only; no model, no generated code."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


class Batch:
    def __init__(self):
        import numpy as np
        import torch
        prompt, response = [0, 7, 8], [9, 10, 0]                 # left-padded prompt, right-padded completion
        self.batch = {"prompts": torch.tensor([prompt] * 256), "responses": torch.tensor([response] * 256),
                      "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0]] * 256),
                      "response_mask": torch.tensor([[1, 1, 0]] * 256)}
        self.non_tensor_batch = {"extra_info": np.array([{"id": i // 16} for i in range(256)], dtype=object),
                                 "uid": np.array([f"u{i // 16}" for i in range(256)], dtype=object)}


class RetentionScheduleTests(unittest.TestCase):
    def trainer(self, total, step):
        from omegaconf import OmegaConf
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        config = OmegaConf.create({
            "algorithm": {"adv_estimator": "grpo_modified", "screening_specs": {}},
            "data": {"train_batch_size": 16, "seed": 1},
            "trainer": {"rollout_data_dir": self.tmp.name, "total_training_steps": total},
            "actor_rollout_ref": {"rollout": {"n": 16},
                                  "model": {"caft": {"protocol": "bestcase_multilayer_response_predictor_caft_v2",
                                                     "arm": "A0"}},
                                  "actor": {"caft_checkpoint": {"common_m0": {"common_identity_sha256": "a" * 64}}}}})
        return type("Trainer", (), {"config": config, "global_steps": step, "tokenizer": Tokenizer()})()

    def retain(self, total, step):
        from src.train.verl.caft_rollouts import retain_returned_batch
        trainer, batch = self.trainer(total, step), Batch()
        retain_returned_batch(trainer, batch)
        path = Path(self.tmp.name) / "returned_before_update" / f"step_{step:03d}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 256)
        self.assertEqual({r["optimizer_step_to_follow"] for r in rows}, {step})
        self.assertEqual(list(batch.non_tensor_batch["caft_request_id"]), [r["request_id"] for r in rows])
        return batch

    def refused(self, total, step):
        from src.train.verl.caft_rollouts import retain_returned_batch
        with self.assertRaisesRegex(ValueError, "Unexpected or already retained training batch"):
            retain_returned_batch(self.trainer(total, step), Batch())

    def test_four_hundred_step_schedule_retains_steps_past_two_hundred(self):
        for step in (1, 200, 201, 400):
            self.retain(400, step)
        for step in (0, 401):
            self.refused(400, step)

    def test_two_hundred_step_schedule_unchanged(self):
        for step in (1, 200):
            self.retain(200, step)
        for step in (0, 201):
            self.refused(200, step)

    def test_undeclared_schedule_refused(self):
        for total in (100, 300, 500):
            self.refused(total, 50)

    def test_already_retained_batch_refused(self):
        from src.train.verl.caft_rollouts import retain_returned_batch
        trainer = self.trainer(400, 250)
        batch = self.retain(400, 250)
        with self.assertRaisesRegex(ValueError, "Unexpected or already retained training batch"):
            retain_returned_batch(trainer, batch)


if __name__ == "__main__":
    unittest.main()
