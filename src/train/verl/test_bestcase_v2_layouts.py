"""Regression test (best-case wave 2, 30 Sept): the v2 common-start check must accept every supported layout.

A 2-GPU load arm failed with 'Common initializer requires the declared16x16/eight-rank microbatch profile' because
the per-rank mini-batch after the FSDP constructor's normalization (16*16/2 = 128) was not in the allowed set.
"""
import unittest


class V2LayoutTests(unittest.TestCase):
    def value(self, mini, initialized, v2=True):
        from src.train.verl.caft_initialization import worker_config_value
        cfg = {'rollout': {'n': 16}, 'actor': {'ppo_micro_batch_size': None, 'ppo_micro_batch_size_per_gpu': 4,
                                               'ppo_mini_batch_size': mini}, 'model': {}}
        return worker_config_value(cfg, initialized, v2)

    def test_every_supported_rank_count_normalizes_back_to_16(self):
        for ranks in (1, 2, 4, 8):
            self.assertEqual(self.value(256 // ranks, True)['actor']['ppo_mini_batch_size'], 16, ranks)

    def test_uninitialized_value_is_the_declared_16(self):
        self.assertEqual(self.value(16, False)['actor']['ppo_mini_batch_size'], 16)

    def test_other_values_still_rejected(self):
        for bad in (48, 96, 512):
            with self.assertRaises(ValueError):
                self.value(bad, True)
        with self.assertRaises(ValueError):          # v1 (eight-rank Ada) keeps its original rule
            self.value(128, True, v2=False)


if __name__ == '__main__':
    unittest.main()
