"""Synthetic numerical/population tests only; no models or saved activations."""
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np

from . import broader_pca as b
from . import candidates as c


def fixture(n=32, counts=None):
    counts = [3] * n if counts is None else counts
    rows = [{'record_id': 'r' + str(i), 'problem_id_key': str(i), 'problem_split': 'direction_fit',
             'broader_pca_policy': b.POPULATION, 'extraction_padded_length': 2688,
             'selection_replaced_or_filtered': False, 'completion_token_count': counts[i] + 2,
             'broader_region_status': {'solution': {'status': 'usable'}, 'evaluator': {'status': 'usable'}}}
            for i in range(n)]
    positions = [list(range(1, count + 1)) for count in counts]
    offsets = np.cumsum([0] + counts, dtype=np.int64)
    rng = np.random.default_rng(9)
    h0 = rng.normal(size=(offsets[-1], b.HIDDEN_SIZE)).astype(np.float32)
    change = (rng.normal(size=h0.shape) * np.arange(1, b.HIDDEN_SIZE + 1)).astype(np.float32)
    return c.WindowData(rows, positions, h0, h0 + change, offsets,
                        np.repeat(np.arange(n), counts), np.concatenate(positions))


class BroaderPcaTests(unittest.TestCase):
    def setUp(self):
        # Small authored dimensions preserve the actual rank10 numerical path.
        self.dimension = patch.object(b, 'HIDDEN_SIZE', 16); self.dimension.start()
        self.addCleanup(self.dimension.stop)
        self.profile = {'cohort': b.COHORT, 'padded_sequence_length': 2688, 'raw_dtype': 'BF16',
                        'layer_count': 36, 'hidden_size': 16}

    def fit(self, data, semantics='window_mean'):
        return b.fit_representation(data, layer=21, region='evaluator', representation='transition',
                                    semantics=semantics, numerical_profile=self.profile)

    def test_full_half_numerical_success_and_tensor_array_roundtrip(self):
        report, tensors = self.fit(fixture())
        self.assertEqual(report['cohort'], 'broader_ordinary')
        self.assertEqual(report['primary']['status'], 'not_requested')
        self.assertTrue(report['pca']['all_fits_numerically_qualified'])
        self.assertEqual(tensors['pca.pcs'].shape, (16, 10))
        self.assertEqual(tensors['pca.pcs'].dtype, np.float32)
        self.assertEqual(tensors['pca.eigenvalues'].dtype, np.float64)
        self.assertEqual(len(tensors), 9)
        for prefix in ('pca', 'pca.half_0', 'pca.half_1'):
            q = tensors[prefix + '.pcs']; np.testing.assert_allclose(q.T @ q, np.eye(10), atol=5e-5)
            self.assertTrue(all(q[np.argmax(np.abs(q[:, i])), i] >= 0 for i in range(10)))
        buffer = io.BytesIO(); np.savez(buffer, **tensors); buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            for key in tensors:np.testing.assert_array_equal(saved[key], tensors[key])

    def test_labels_absent_or_changed_do_not_change_any_fit(self):
        data = fixture(); first, before = self.fit(data)
        for i, row in enumerate(data.rows):
            row['outcome_presence_class'] = ['arbitrary', None, {'ignored': True}][i % 3]
            row['ground_truth_correctness'] = i % 2
        after_report, after = self.fit(data)
        for key in before:np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(first['pca'], after_report['pca'])
        self.assertFalse(after_report['labels_used_for_fitting_or_orientation'])

    def test_equal_problem_weights_with_unequal_token_counts(self):
        data = fixture(counts=list(range(1, 33)))
        x, weights, groups = b.population(data, 'token_level')
        self.assertEqual(len(x), sum(range(1, 33)))
        for problem in set(groups):self.assertAlmostEqual(weights[groups == problem].sum(), 1 / 32)
        _, record_weights, record_groups = b.population(data, 'window_mean')
        np.testing.assert_array_equal(record_weights, np.full(32, 1 / 32))
        self.assertEqual(len(record_groups), 32)

    def test_delta_precedes_mean_and_token_pca_uses_individual_tokens(self):
        data = fixture()
        data.h0[:, 0] = 1e8; data.h60[:, 0] = 1e8
        data.h60[::3, 0] += 8; data.h60[1::3, 0] += 16
        delta = np.subtract(data.h60, data.h0, dtype=np.float32)
        x, _, _ = b.population(data, 'window_mean')
        expected = np.stack([delta[a:z].mean(0, dtype=np.float32) for a, z in zip(data.offsets[:-1], data.offsets[1:])])
        np.testing.assert_array_equal(x, expected)
        token, _, _ = b.population(data, 'token_level'); np.testing.assert_array_equal(token, delta)

    def test_halves_are_deterministic_disjoint_and_angles_include_weakest(self):
        report, _ = self.fit(fixture())
        halves = report['pca']['disjoint_half_problem_ids']
        self.assertFalse(set(halves[0]) & set(halves[1]))
        self.assertEqual(set(halves[0]) | set(halves[1]), set(map(str, range(32))))
        order = sorted(map(str, range(32)), key=lambda p: (c.stable_seed(20260908, 'disjoint_half', p), p))
        self.assertEqual(halves, [order[:16], order[16:]])
        plane = report['pca']['disjoint_half_subspaces']['historical_pc04_pc05_zero_based_reference']
        self.assertEqual(len(plane['principal_angles_degrees']), 2)
        self.assertEqual(plane['weakest_principal_cosine'], min(plane['principal_cosines']))
        self.assertEqual(plane['largest_principal_angle_degrees'], max(plane['principal_angles_degrees']))

    def test_supervised_and_label_population_paths_are_never_called(self):
        with patch.object(b.fit, 'fit_supervised', side_effect=AssertionError('supervised')), \
             patch.object(b.fit, 'pca_population', side_effect=AssertionError('triplets')), \
             patch.object(b.fit, 'both_controls', side_effect=AssertionError('labels')):
            report, tensors = self.fit(fixture(), 'token_level')
        self.assertEqual(report['pca']['outer_folds'], [])
        self.assertFalse(any('.oof.' in key for key in tensors))

    def test_wrong_profile_split_duplicates_offsets_and_semantics_rejected(self):
        profile = {**self.profile, 'padded_sequence_length': 2176}
        with self.assertRaisesRegex(ValueError, '2688'):b.validate_profile(profile)
        data = fixture(); data.rows[0]['problem_split'] = 'untouched_test'
        with self.assertRaisesRegex(ValueError, 'population'):self.fit(data)
        data = fixture(); data.rows[1]['problem_id_key'] = data.rows[0]['problem_id_key']
        with self.assertRaisesRegex(ValueError, 'One record'):self.fit(data)
        data = fixture(); data.offsets[1] = 0
        with self.assertRaisesRegex(ValueError, 'offsets'):self.fit(data)
        with self.assertRaisesRegex(ValueError, 'semantics'):self.fit(fixture(), 'anchor')
        with self.assertRaisesRegex(ValueError, 'Only PCA'):b.configuration({'pca_rank': 5})

    def test_missing_region_zero_variance_and_insufficient_half_are_explicit(self):
        report, tensors = self.fit(None)
        self.assertEqual(report['pca']['status'], 'unsupported'); self.assertFalse(tensors)
        data = fixture(); data.h60[:] = data.h0
        report, tensors = self.fit(data)
        self.assertEqual(report['pca']['status'], 'zero_centered_variance'); self.assertFalse(tensors)
        report, tensors = self.fit(fixture(n=16))
        self.assertIn('pca.pcs', tensors)
        self.assertFalse(report['pca']['eligible_for_comparison'])
        self.assertEqual([r['status'] for r in report['pca']['half_fit_reports']], ['insufficient_rank'] * 2)
        self.assertIsNone(report['pca']['axes'][0]['same_index_half_absolute_cosine'])

    def test_failed_half_retains_full_payload_and_unqualifies_stability(self):
        original = b.fit.pca_fit; calls = []
        def altered(*args):
            result = original(*args); calls.append(result)
            if len(calls) == 2:result['report']['numerically_qualified'] = False
            return result
        with patch.object(b.fit, 'pca_fit', side_effect=altered):report, tensors = self.fit(fixture())
        self.assertEqual(len(calls), 3); self.assertEqual(len(tensors), 9)
        self.assertFalse(report['pca']['all_fits_numerically_qualified'])
        self.assertFalse(report['pca']['stability_numerically_qualified'])

    def test_coverage_retains_absent_and_contextual_regions_without_labels(self):
        data = fixture(n=2)
        data.rows[1]['broader_region_status']['evaluator'] = {'status': 'absent', 'reason': 'missing_function'}
        selections = [{'evaluator': {'transition': {'eligible': True, 'opposite_region_overlap': [3]}}},
                      {'evaluator': {'transition': {'eligible': False, 'exclusion_reason': 'missing_region'}}}]
        result = b.cell_coverage(data.rows, selections, 'evaluator', 'transition')
        self.assertEqual(result['selected_population_records'], 2); self.assertEqual(result['eligible_records'], 1)
        self.assertEqual(result['excluded'][0]['region_status']['reason'], 'missing_function')
        self.assertEqual(result['opposite_region_context'][0]['completion_positions'], [3])

    def test_raw_profile_and_actual_terminal_proof_gate(self):
        proof = {'status': 'independently_verified_broader_raw500', 'source_manifest_sha256': 'a' * 64,
                 'padded_sequence_length': 2688, 'records': 500, 'native_files': 1000,
                 'files': {str(i): {} for i in range(1000)}, 'numerical_audits_verified': True,
                 'raw_activations_retained': True, 'all_finite': True, 'differences_computed': False,
                 'external_exit_receipt': {'path': 'unused', 'sha256': 'b' * 64, 'size_bytes': 1}}
        outer = {'status': 'verified_systemd_controller_exit_and_release', 'plan_sha256': 'a' * 64,
                 'cgroup_processes': [], 'gpu_release_verified': True,
                 'service_fields': {'ExecMainCode': '1', 'ExecMainStatus': '0', 'Result': 'success', 'MainPID': '0', 'SubState': 'exited'}}
        with patch.object(b, 'read_reference', return_value=outer) as read:
            b.validate_raw_proof(proof, 'a' * 64); self.assertEqual(read.call_count, 1)
            bad = {**proof, 'padded_sequence_length': 2176}
            with self.assertRaisesRegex(ValueError, '2688'):b.validate_raw_proof(bad, 'a' * 64)
            self.assertEqual(read.call_count, 1)
            outer['service_fields']['ExecMainStatus'] = '1'
            with self.assertRaisesRegex(ValueError, 'exit successfully'):b.validate_raw_proof(proof, 'a' * 64)


if __name__ == '__main__':unittest.main()
