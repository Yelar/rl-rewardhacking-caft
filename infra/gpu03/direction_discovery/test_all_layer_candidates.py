"""Authored numerical/coverage checks; no saved activations or model execution."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np

from . import all_layer_candidates as a
from . import candidates as c
from .test_candidates import data_fixture


def fixture(n=12, d=12, tokens=2):
    data = data_fixture(n, d, [tokens] * (n * 3))
    for row in data.rows:
        row['input_ids'] = [100] * row['prompt_token_count'] + row['completion_token_ids']
        row['is_parsed'] = True
        row['gt_result'] = {'can_compile': True}
        row['evaluator_compilation_status'] = 'success'
    return data


def subset(data, indices):
    rows = [copy.deepcopy(data.rows[i]) for i in indices]
    positions = [data.positions[i] for i in indices]
    offsets = np.cumsum([0] + [len(p) for p in positions])
    arrays = [np.concatenate([x[data.offsets[i]:data.offsets[i + 1]] for i in indices]) for x in (data.h0, data.h60)]
    return c.WindowData(rows, positions, *arrays, offsets, np.repeat(np.arange(len(rows)), np.diff(offsets)), np.concatenate(positions))


CFG = {'pca_rank': 3, 'pca_oversample': 5, 'pca_power_iters': 1, 'l2_penalties': [.01, .1, 1.]}


class NumericalTests(unittest.TestCase):
    def test_delta_subtracted_before_mean(self):
        data = fixture(2, 3)
        data.h0[:] = np.asarray([1e8, 1., -1e8], dtype=np.float32)
        data.h60[:] = data.h0
        data.h60[::2, 0] += 8
        data.h60[1::2, 0] += 16
        means, delta = a.record_vectors(data)
        expected = np.stack([delta[s:e].mean(0, dtype=np.float32) for s, e in zip(data.offsets[:-1], data.offsets[1:])])
        np.testing.assert_array_equal(means['delta'], expected)
        old = np.stack([data.h60[s:e].mean(0, dtype=np.float32) - data.h0[s:e].mean(0, dtype=np.float32)
                        for s, e in zip(data.offsets[:-1], data.offsets[1:])])
        self.assertFalse(np.array_equal(expected, old))
        self.assertEqual(expected[0, 0], 12)

    def test_nonzero_support_is_actual_not_nominal_or_prefix(self):
        x = np.zeros((111, 7), dtype=np.float32); x[0, 0] = 1; x[1, 1] = 1e-9
        report = a.support(x, list(map(str, range(111))), 1e-7)
        self.assertEqual(report['exact_nonzero_matched_contrasts'], 2)
        self.assertEqual(report['above_tolerance_matched_contrasts'], 1)
        self.assertTrue(report['weak_support_warning'])

    def test_group_folds_deterministic_disjoint_complete(self):
        ids = list(map(str, range(111))); folds = a.folds(ids, 3, 29)
        self.assertEqual(folds, a.folds(ids[::-1], 3, 29))
        self.assertEqual(sorted(sum(folds, [])), sorted(ids))
        self.assertFalse(set(folds[0]) & set(folds[1]))

    def test_raw_logistic_coordinates_and_constant_feature(self):
        rng = np.random.default_rng(321)
        x = rng.normal(size=(60, 5)) * [1, 100, .001, 2, 1] + [2, 50, -3, 0, 0]
        x[:, 4] = 1; y = (x[:, 0] + x[:, 3] > 2).astype(float)
        model = a.fit_logistic(x, y, np.ones(60), .01, a.configuration())
        pred = x @ model['weight'] + model['intercept']
        std = (x - model['training_mean']) / model['training_scale']
        standardized_intercept = model['intercept'] + model['training_mean'] @ model['weight']
        np.testing.assert_allclose(pred, std @ model['standardized_weight'] + standardized_intercept, atol=1e-8)
        self.assertEqual(model['weight'][4], 0)
        self.assertGreater(((pred > 0) == y).mean(), .85)

    def test_scaler_only_sees_passed_training_rows(self):
        x = np.array([[0., 1], [1, 2], [2, 1], [3, 2]])
        model = a.fit_logistic(x, np.array([0, 0, 1, 1]), np.ones(4), 1., a.configuration())
        np.testing.assert_array_equal(model['training_mean'], [1.5, 1.5])
        prediction = np.array([[1e9, -1e8]]) @ model['weight'] + model['intercept']
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_array_equal(model['training_mean'], [1.5, 1.5])

    def test_paired_auc_weights_problems_not_completions(self):
        rows = fixture(2).rows
        # First problem two extra positive completions; nested means stay equal.
        rows += [dict(rows[0], record_id='extra1'), dict(rows[0], record_id='extra2')]
        scores = np.array([2, 0, 0, -2, 0, 0, 2, 2], dtype=float)
        r = a.discrimination(rows, scores, ['0', '1'], c.INCORRECT)
        self.assertEqual(r['equal_problem_paired_auc'], .5)
        self.assertEqual(r['equal_problem_mean_score_difference'], 0)

    def test_all_principal_angles_retain_weak_dimension(self):
        left = np.eye(5)[:, :2]; right = np.eye(5)[:, [0, 3]]
        r = a.principal_angles(left, right)
        np.testing.assert_allclose(r['principal_angles_degrees'], [0, 90])
        self.assertEqual(r['weakest_principal_cosine'], 0)

    def test_bound_historical_reference_compares_saved_q_without_refitting(self):
        q = np.eye(8, dtype=np.float32)
        result = a.reference_alignment({'pca.pcs': q[:, :6], 'candidate.direction': q[:, 4:5]},
                                       {'historical_pc04': q[:, 4:5], 'historical_pc04_pc05': q[:, [4, 5]]})
        self.assertAlmostEqual(result['historical_pc04']['candidate.direction']['signed_axis_cosines'][0][0], 1)
        self.assertEqual(len(result['historical_pc04_pc05']['pca.pcs']['principal_angles_degrees']), 2)
        self.assertFalse(result['historical_pc04']['candidate.direction']['automatic_selection_preference'])

    def test_actual_candidate_examples_are_fitting_and_problem_diverse(self):
        data = fixture()
        scores = np.arange(len(data.h0))
        out = a.diagnostic_examples(data, scores, ['0', '1', '2'], token_level=True)
        self.assertEqual(len(out['positive']), 3)
        self.assertEqual({r['problem_id'] for r in out['positive']}, {'0', '1', '2'})
        self.assertTrue(all(len(r['selected_completion_positions']) == 1 for r in out['positive']))

    def test_near_degenerate_subspace_does_not_prefer_pc4(self):
        r = a.subspaces(np.array([10., 9.5, 9.2, 6., 3., 2.9]), .1)
        self.assertEqual(r['near_degenerate_pc1_pc3'], [0, 1, 2])
        self.assertEqual(r['historical_pc04_pc05_zero_based_reference'], [4, 5])
        self.assertIn('near_degenerate_pc5_pc6', r)

    def test_pca_refines_residual_and_retains_unqualified_failure(self):
        config = a.configuration(CFG)
        x = np.eye(12, dtype=np.float32); w = np.ones(12) / 12
        original = a.pca_fit_once
        calls = []
        def observed(x, w, cfg, seed):
            out = original(x, w, cfg, seed)
            calls.append((cfg['pca_oversample'], cfg['pca_power_iters']))
            out['report']['relative_covariance_residuals'] = [.2, .2, .2]
            return out
        with patch.object(a, 'pca_fit_once', side_effect=observed):
            out = a.pca_fit(x, w, config, 5)
        self.assertEqual(calls, [(5, 1), (19, 3), (33, 5)])
        self.assertFalse(out['report']['numerically_qualified'])
        self.assertTrue(out['report']['not_for_promotion_until_numerically_qualified'])

    def test_actual_saved_parser_and_compile_fields_are_reported(self):
        data = fixture()
        report = a.associations(data.rows, np.arange(len(data.rows)), data.positions)
        self.assertEqual(report['recorded_parser_success']['available_records'], len(data.rows))
        self.assertEqual(report['recorded_whole_program_can_compile']['available_records'], len(data.rows))
        self.assertEqual(report['syntax_valid']['available_records'], 0)

    def test_token_weights_equal_problem_class_record_not_length(self):
        data = data_fixture(6, 12, list(range(1, 19)))
        means, delta = a.record_vectors(data)
        x, weights, groups = a.pca_population(data, delta, means, list(map(str, range(6))), 'token_level')
        self.assertEqual(len(x), sum(range(1, 19)))
        for p in set(groups):
            self.assertAlmostEqual(weights[groups == p].sum(), 1/6)
        for i in range(18):
            self.assertAlmostEqual(weights[data.offsets[i]:data.offsets[i + 1]].sum(), 1/18)
        mean_x, mean_w, _ = a.pca_population(data, delta, means, list(map(str, range(6))), 'window_mean')
        self.assertEqual(mean_x.shape, (18, 12))
        np.testing.assert_allclose(mean_w, 1/18)

    def test_identical_prefix_does_not_infer_zero_activation(self):
        data = fixture()
        r = a.prefix_inventory(data, ['0', '1'], c.INCORRECT)
        self.assertEqual(r['identical_consumed_prefix_pairs'], 2)
        vectors, _ = a.record_vectors(data)
        diffs = a.paired_differences(data.rows, vectors['delta'], ['0', '1'], c.INCORRECT)
        self.assertEqual(a.support(diffs, ['0', '1'], 1e-7)['exact_nonzero_matched_contrasts'], 2)


class BoundaryTests(unittest.TestCase):
    def test_validation_and_test_cannot_enter_fit(self):
        for split in ('configuration_validation', 'untouched_test'):
            data = fixture(); data.rows[0]['problem_split'] = split
            with self.assertRaisesRegex(ValueError, 'Wrong split'):
                a.validate_data(data)

    def test_malformed_arrays_offsets_and_tokens_fail(self):
        for mutate in (lambda d: setattr(d, 'h0', d.h0.astype(np.float64)),
                       lambda d: d.h0.__setitem__((0, 0), np.nan),
                       lambda d: d.offsets.__setitem__(1, 0),
                       lambda d: d.token_position.__setitem__(0, 900),
                       lambda d: d.rows.__setitem__(1, d.rows[0])):
            d = fixture(); mutate(d)
            with self.assertRaises(ValueError): a.validate_data(d)

    def test_unknown_config_or_methods_fail(self):
        for cfg in ({'folds': 3}, {'outer_folds': True}, {'l2_penalties': [0]},
                    {'methods': []}, {'methods': ['probe', 'probe']}, {'pca_backend': 'cuda'}):
            with self.assertRaises(ValueError): a.configuration(cfg)

    def test_anchor_rejects_mean_block(self):
        with self.assertRaisesRegex(ValueError, 'one token'):
            a.fit_representation(fixture(), layer=0, region='solution', representation='body', semantics='anchor', config=CFG)

    def test_absolute_sequence_coordinates_not_accepted_as_completion_offsets(self):
        data = fixture()
        data.positions[0] = [data.rows[0]['completion_token_count'], data.rows[0]['completion_token_count'] + 1]
        data.token_position[:2] = data.positions[0]
        with self.assertRaisesRegex(ValueError, 'Invalid selected token positions'):
            a.validate_data(data)

    def test_half_failure_keeps_payload_but_unqualifies_whole_pca_evidence(self):
        original = a.pca_fit
        count = 0
        def half_failure(*args, **kwargs):
            nonlocal count
            out = original(*args, **kwargs); count += 1
            if count == 2:
                out['report']['numerically_qualified'] = False
            return out
        with patch.object(a, 'pca_fit', side_effect=half_failure):
            report, tensors = a.fit_representation(fixture(), layer=0, region='solution', representation='transition',
                    semantics='window_mean', config={**CFG, 'methods': ['pca']})
        self.assertTrue(report['pca']['fit']['numerically_qualified'])
        self.assertFalse(report['pca']['all_fits_numerically_qualified'])
        self.assertFalse(report['pca']['eligible_for_comparison'])
        self.assertTrue(report['pca']['descriptive_only_until_refined'])
        self.assertIn('pca.pcs', tensors)

    def test_external_overlap_and_test_rejected_before_fit(self):
        data = fixture(); other = copy.deepcopy(data)
        for r in other.rows: r['problem_split'] = 'configuration_validation'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            a.fit_representation(data, layer=0, region='solution', representation='transition', semantics='window_mean', external_data=other)
        for r in other.rows: r['problem_split'] = 'untouched_test'
        with self.assertRaisesRegex(ValueError, 'Wrong split'):
            a.fit_representation(data, layer=0, region='solution', representation='transition', semantics='window_mean', external_data=other)

    def test_pca_only_never_calls_logistic_or_mean_fitter(self):
        with patch.object(a, 'fit_supervised', side_effect=AssertionError('must reuse existing means/probes')):
            r, tensors = a.fit_representation(fixture(), layer=0, region='solution', representation='transition',
                    semantics='token_level', config={**CFG, 'methods': ['pca']})
        self.assertEqual(r['primary']['status'], 'not_requested')
        self.assertEqual(r['pca']['status'], 'fitted')
        self.assertIn('pca.pcs', tensors)

    def test_zero_population_reports_gap_not_direction(self):
        data = fixture(); data.h60[:] = data.h0
        r, _ = a.fit_representation(data, layer=0, region='solution', representation='transition',
                                   semantics='window_mean', config={**CFG, 'methods': ['mean', 'pca']})
        self.assertEqual(r['pca']['status'], 'zero_centered_variance')
        self.assertEqual(r['primary']['rh_correct']['support']['delta']['exact_nonzero_matched_contrasts'], 0)
        self.assertEqual(r['primary']['rh_correct']['models']['delta.mean']['status'], 'no_nonzero_direction')


class FullPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.report, cls.tensors = a.fit_representation(cls.data, layer=21, region='evaluator', representation='transition',
                                                      semantics='window_mean', config=CFG)

    def test_complete_all_families_both_contrasts_both_kinds(self):
        r = self.report
        self.assertEqual(r['pca']['status'], 'fitted')
        for contrast in a.CONTROLS:
            models = r['primary'][contrast]['models']
            self.assertEqual(set(models), {'h60.mean', 'h60.probe', 'delta.mean', 'delta.probe'})
            for result in models.values():
                self.assertEqual(result['status'], 'fitted')
                self.assertEqual(result['oof']['rh_correct']['independent_problems'], 12)
                self.assertEqual(result['oof']['rh_incorrect']['independent_problems'], 12)
                self.assertFalse(set(result['disjoint_halves']['problem_ids'][0]) & set(result['disjoint_halves']['problem_ids'][1]))
        json.dumps(r, allow_nan=False)

    def test_every_cv_scaler_fit_excludes_outer_test_problem(self):
        for contrast in a.CONTROLS:
            for kind in ('h60', 'delta'):
                r = self.report['primary'][contrast]['models'][kind + '.probe']
                for fold in r['outer_folds']:
                    train = set(fold['training_problem_ids']); test = set(fold['heldout_problem_ids'])
                    self.assertFalse(train & test)
                    for inner in fold['regularization']['folds']:
                        self.assertEqual(set(inner['training_problem_ids']) | set(inner['validation_problem_ids']), train)
                        self.assertFalse(set(inner['training_problem_ids']) & set(inner['validation_problem_ids']))
                        self.assertFalse(set(inner['training_problem_ids']) & test)

    def test_pca_train_only_folds_and_no_full_axis_alignment(self):
        r = self.report['pca']
        for fold in r['outer_folds']:
            self.assertFalse(set(fold['training_problem_ids']) & set(fold['heldout_problem_ids']))
            self.assertEqual(set(fold['training_contrast_signs']), set(a.CONTROLS))
        self.assertEqual(len(r['disjoint_half_subspaces']['top3']['principal_angles_degrees']), 3)
        self.assertEqual(len(r['axes']), 3)

    def test_delta_directions_have_m60_transfer(self):
        for contrast in a.CONTROLS:
            for method in ('mean', 'probe'):
                r = self.report['primary'][contrast]['models']['delta.' + method]
                self.assertEqual(r['oof_m60_transfer']['rh_incorrect']['independent_problems'], 12)
                self.assertTrue(np.isfinite(self.tensors[r['tensor_prefix'] + '.oof_h60_transfer_scores']).all())

    def test_external_validation_uses_frozen_training_vectors(self):
        other = fixture(6)
        for row in other.rows:
            row['problem_split'] = 'configuration_validation'; row['problem_id_key'] = 'v' + row['problem_id_key']
            row['record_id'] = 'v' + row['record_id']
        report, tensors = a.fit_representation(self.data, layer=21, region='evaluator', representation='transition',
                  semantics='window_mean', config={**CFG, 'methods': ['mean']}, external_data=other)
        self.assertFalse(report['external_validation']['fitted_on_validation'])
        for key in tensors:
            np.testing.assert_equal(tensors[key], self.tensors[key])
        self.assertEqual(len(report['external_validation']['problem_ids']), 6)

    def test_missing_class_reports_pair_coverage_and_supplementary(self):
        data = subset(fixture(13), list(range(12*3)) + [36, 38])
        r, _ = a.fit_representation(data, layer=0, region='solution', representation='transition',
                                   semantics='window_mean', config={**CFG, 'methods': ['mean']})
        self.assertEqual(len(r['coverage']['complete_triplet_problem_ids']), 12)
        self.assertEqual(len(r['coverage']['contrast_eligible_problem_ids']['rh_incorrect']), 13)
        self.assertIn('rh_incorrect', r['supplementary_pair_only'])
        self.assertEqual(r['primary']['rh_incorrect']['support']['h60']['independent_problems'], 12)


if __name__ == '__main__':
    unittest.main()
