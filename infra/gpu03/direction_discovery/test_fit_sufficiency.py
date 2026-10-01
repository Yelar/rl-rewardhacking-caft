"""Numerical, weighting, split, and failure-path checks without model calls."""
import copy
import unittest

import numpy as np

try:
    from . import fit_sufficiency as s
    from . import test_candidates as fixtures
except ImportError:
    import fit_sufficiency as s
    import test_candidates as fixtures


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.ids = sorted(map(str, range(111)))
        self.groups, self.plan = s.make_sampling(self.ids, repeats=5)

    def test_deterministic_complete_nested_sampling(self):
        other, _ = s.make_sampling(self.ids, repeats=5)
        self.assertEqual(len(self.groups), 10)
        for key in self.groups:
            np.testing.assert_array_equal(other[key], self.groups[key])
            self.assertTrue(np.all(self.groups[key].sum(2) == int(key.split('_')[1])))
        for scheme, ns in [('subset', [30, 60, 90]), ('bootstrap', [30, 60, 90, 111]), ('disjoint', [30, 45, 55])]:
            for a, b in zip(ns[:-1], ns[1:]):
                self.assertTrue(np.all(self.groups[f'{scheme}_{a}'] <= self.groups[f'{scheme}_{b}']))

    def test_no_false_independent_full_set(self):
        self.assertNotIn('subset_111', self.groups)
        self.assertTrue(np.all((self.groups['bootstrap_111'] > 0).sum(2) < 111))
        for n in s.DISJOINT_SIZES:
            x = self.groups[f'disjoint_{n}']
            self.assertFalse(np.any((x[:, 0] > 0) & (x[:, 1] > 0)))
        self.assertTrue(np.all(((self.groups['subset_90'][:, 0] > 0) & (self.groups['subset_90'][:, 1] > 0)).sum(1) >= 69))

    def test_reject_bad_cohort(self):
        for ids in [self.ids[:-1], list(reversed(self.ids)), self.ids[:-1] + [self.ids[0]]]:
            with self.assertRaises(ValueError):
                s.make_sampling(ids)

    def test_gram_cosines_match_explicit_vectors(self):
        x = np.random.default_rng(91).normal(size=(111, 23)).astype(np.float32)
        report, ref = s.mean_study(x, self.groups)
        key = 'disjoint_55'
        explicit = []
        for counts in self.groups[key]:
            a, b = counts.astype(float) @ x / 55
            explicit.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        self.assertAlmostEqual(report['groups'][key]['pair_signed_cosine']['statistics']['mean'], np.mean(explicit), places=12)
        np.testing.assert_allclose(ref, x.astype(float).mean(0), atol=1e-7)

    def test_zero_and_sign_are_not_hidden(self):
        report, _ = s.mean_study(np.zeros((111, 5)), self.groups)
        self.assertEqual(report['full_status'], 'no_nonzero_direction')
        self.assertEqual(report['groups']['subset_30']['pair_signed_cosine']['accepted'], 0)
        cos, _ = s.cosine_arrays(np.array([-1.]), np.array([1.]), np.array([1.]))
        self.assertEqual(float(cos[0]), -1.)


class PositionTests(unittest.TestCase):
    def row(self):
        return {'completion_token_count': 60,
                'regions': {'evaluator': {'definition_completion_token': 8,
                    'first_executable_completion_token': 25, 'logit_source_completion_token': 24,
                    'window_completion_positions': {'pre_definition': [None] * 8 + list(range(8))}}},
                'region_mask_completion_positions': {'evaluator__pre_definition': list(range(8))}}

    def test_predictor_is_previous_token_and_nulls_are_omitted(self):
        row = self.row()
        for method, wanted in [('definition_predictor', [7]), ('definition', [8]), ('body_predictor', [24]), ('body', [25]), ('pre_definition', list(range(8)))]:
            self.assertEqual(s.positions(row, 'evaluator', method), wanted)

    def test_absent_outside_and_disagreeing_masks_fail(self):
        row = self.row(); row['regions']['evaluator'] = None
        with self.assertRaisesRegex(ValueError, 'absent'):
            s.positions(row, 'evaluator', 'body')
        row = self.row(); row['regions']['evaluator']['definition_completion_token'] = 0
        with self.assertRaisesRegex(ValueError, 'outside'):
            s.positions(row, 'evaluator', 'definition_predictor')
        row = self.row(); row['region_mask_completion_positions']['evaluator__pre_definition'] = [3]
        with self.assertRaisesRegex(ValueError, 'disagree'):
            s.positions(row, 'evaluator', 'pre_definition')


class PCATests(unittest.TestCase):
    def test_subspace_rotation_and_sign(self):
        a = np.eye(12, 10, dtype=np.float32)
        b = a.copy(); b[:, :2] = a[:, :2] @ np.array([[0., -1.], [1., 0.]])
        result = s.compare_pcs(a, b)
        self.assertEqual(result['same_index_absolute_cosine'][0], 0.)
        self.assertEqual(result['assignment_absolute_cosine'][0], 1.)
        self.assertEqual(result['subspaces']['10']['overlap'], 1.)
        self.assertEqual(s.compare_pcs(a, -a)['same_index_absolute_cosine'], [1.] * 10)

    def test_problem_weighting_matches_explicit_replication(self):
        rng = np.random.default_rng(381)
        x = rng.normal(size=(18, 12)).astype(np.float32)
        token_problem = np.repeat(np.arange(3), 6)
        base = np.full(18, 1 / 6)
        counts = np.array([2, 0, 1])
        config = dict(rank=10, oversample=2, power_iters=2)
        result = s.fit_weighted_pca(x, token_problem, base, counts, 22, config)
        replicated = np.concatenate([x[:6], x[:6], x[12:]])
        expected = np.linalg.eigvalsh(np.cov(replicated.astype(float), rowvar=False, bias=True))[::-1][:10]
        np.testing.assert_allclose(result['eigenvalues'], expected, rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'weighting mismatch'):
            s.fit_weighted_pca(x, token_problem, base * 2, counts, 22, config)

    def test_full_study_success_path(self):
        rng = np.random.default_rng(348)
        x = rng.normal(size=(333, 12)).astype(np.float32)
        groups, _ = s.make_sampling(sorted(map(str, range(111))), repeats=1)
        result, arrays = s.pca_study(x, np.repeat(np.arange(111), 3), np.full(333, 1 / 3), groups,
                                    1, 93, dict(rank=10, oversample=2, power_iters=2))
        self.assertEqual(len(result['groups']), 10)
        self.assertEqual(arrays['bootstrap_111_pcs'].shape, (1, 2, 12, 10))
        self.assertLess(result['all_resample_max_relative_residual']['p975'], 1e-5)


if __name__ == '__main__':
    unittest.main()
