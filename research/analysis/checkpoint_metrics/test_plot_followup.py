"""CPU regressions; positive outcome checks use an actual closed historical cell."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

import plot_followup as p


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.manifest = p.frozen_inputs()
        data = p.old.read_pinned(p.HERE / 'test_data/pc4_100_fixed.metrics_only.jsonl',
                                '16fd31abcf8284115810b2c8b08e2be9c5bbb58f5727b61e110716751cd5d21c')
        cls.cell = dict(arm='pc4', step=100, setting='fixed', results=[json.loads(x) for x in data.splitlines()])

    def history_without_actual_cell(self):
        history = copy.deepcopy(self.history)
        history['summaries'] = [s for s in history['summaries']
            if (s['arm'], s['step'], s['setting']) != ('pc4', 100, 'fixed')]
        history['cells'] -= 1; history['records'] -= 1190
        return history

    def test_closed_histories_preserved_exactly(self):
        result = p.append_summaries(self.history, [])
        self.assertEqual(result['summaries'], self.history['summaries'])
        self.assertEqual(result['paired_differences'], self.history['paired_differences'])
        self.assertEqual(result['historical_full10_to200'], self.history['historical_full10_to200'])
        self.assertEqual(result['new_records'], 0)
        self.assertFalse(any(s['arm'] == 'random1' for s in result['summaries']))

    def test_real_cell_reproduces_original_metrics_and_bootstrap_exactly(self):
        result = p.append_summaries(self.history_without_actual_cell(), [self.cell])
        expected = [s for s in self.history['summaries'] if
                    (s['arm'], s['step'], s['setting']) == ('pc4', 100, 'fixed')]
        self.assertEqual(result['summaries'][-2:], expected)
        self.assertEqual(result['records'], 52360)

    def test_absent_helper_does_not_block_primary(self):
        cell = copy.deepcopy(self.cell)
        for row in cell['results']: row.pop('helper_aware_evaluation')
        result = p.append_summaries(self.history_without_actual_cell(), [cell])
        self.assertEqual(result['summaries'][-1]['policy'], p.old.LEGACY)
        self.assertEqual(result['summaries'][-1], next(s for s in self.history['summaries'] if
                         (s['arm'], s['step'], s['setting'], s['policy']) == ('pc4', 100, 'fixed', p.old.LEGACY)))

    def test_historical_cell_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'historical-overwriting'):
            p.append_summaries(self.history, [self.cell])

    def test_duplicate_new_cell_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.append_summaries(self.history_without_actual_cell(), [self.cell, self.cell])

    def test_missing_sample_rejected(self):
        cell = copy.deepcopy(self.cell); cell['results'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            p.append_summaries(self.history_without_actual_cell(), [cell])

    def test_duplicate_request_rejected(self):
        cell = copy.deepcopy(self.cell); cell['results'][1]['request_id'] = cell['results'][0]['request_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate new request'):
            p.append_summaries(self.history_without_actual_cell(), [cell])

    def test_partial_helper_rejected(self):
        cell = copy.deepcopy(self.cell); cell['results'][0].pop('helper_aware_evaluation')
        with self.assertRaisesRegex(ValueError, 'Partial helper'):
            p.append_summaries(self.history_without_actual_cell(), [cell])

    def test_gap_splits_real_checkpoint_entries(self):
        rows = [s for s in self.history['summaries'] if
                (s['arm'], s['setting'], s['policy']) == ('pc4', 'fixed', p.old.LEGACY)]
        rows = [s for s in rows if s['step'] != 40]
        chunks = p.contiguous_segments(rows)
        self.assertEqual([[s['step'] for s in c] for c in chunks], [[0, 10, 20, 30], [50, 60, 70, 80, 90, 100]])
        self.assertEqual(len(p.contiguous_segments([rows[-1]])[0]), 1)

    def test_actual_unknowns_remain_bounds(self):
        rows = [p.old.view(r, p.old.HELPER) for r in self.cell['results']]
        expected = next(s for s in self.history['summaries'] if
                (s['arm'], s['step'], s['setting'], s['policy']) == ('pc4', 100, 'fixed', p.old.HELPER))
        self.assertGreater(expected['labels']['Unknown'], 0)
        self.assertTrue(any(v[0] == 'Unknown' for v in rows))
        stats = expected['metrics']['strict_reward_hack']
        self.assertIsNone(stats['estimate'])
        self.assertLess(stats['identification_bounds'][0], stats['identification_bounds'][1])

    def test_protocol_rejects_training_or_changed_sampling(self):
        # Only schema metadata is changed in this negative test, never outcomes.
        with self.assertRaisesRegex(ValueError, 'native RH follow-up'):
            p.check_protocol(dict(schema='caft_training_rollouts'), self.manifest)
        m = copy.deepcopy(self.manifest); m['schema'] = 'original_rh_random_followup_subset_v2'
        m['sampling']['n'] = 1
        with self.assertRaisesRegex(ValueError, 'sampling'):
            p.check_protocol(m, self.manifest)

    def test_bootstrap_draws_and_probability_bounds(self):
        weights = p.weights_for_history(self.history)
        self.assertEqual(weights.shape, (10000, 119))
        np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=2e-15)
        self.assertEqual(self.history['bootstrap']['seed'], 6219)

    def test_hash_and_portable_path_checks(self):
        ref = dict(path='frozen/historical_primary.json', sha256=p.PRIMARY_SHA, size_bytes=223564)
        self.assertEqual(len(p.local_ref(p.HERE, ref)), 223564)
        with self.assertRaisesRegex(ValueError, 'Nonportable'):
            p.local_ref(p.HERE, dict(ref, path='/etc/passwd'))
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            p.local_ref(p.HERE, dict(ref, sha256='0' * 64))

    def rh110_manifest(self):
        return json.loads(p.old.read_pinned(p.HERE / 'test_data/rh110_manifest.json',
            '657fa64da40256e99b59fe4a6e9e541ba247fc422a88f7606c53c80bb0a204d4'))

    def test_real_rh110_documentation_additions_allowed_and_recorded(self):
        manifest = self.rh110_manifest()
        delta = p.check_protocol(manifest, self.manifest)
        self.assertEqual(delta['required_historical_files'], 11)
        expected = [{key: item[key] for key in ('snapshot_relative', 'sha256', 'size_bytes')}
                    for item in sorted(manifest['base_files'], key=lambda x: x['snapshot_relative'])
                    if item['snapshot_relative'] in ('README.md', 'LICENSE')]
        self.assertEqual(delta['allowed_documentation_additions'], expected)

    def test_documentation_exception_does_not_allow_missing_model_file(self):
        manifest = self.rh110_manifest()
        manifest['base_files'] = [x for x in manifest['base_files'] if x['snapshot_relative'] != 'config.json']
        with self.assertRaisesRegex(ValueError, 'changed or missing'):
            p.check_protocol(manifest, self.manifest)

    def test_documentation_exception_does_not_allow_changed_model_hash(self):
        manifest = self.rh110_manifest()
        next(x for x in manifest['base_files'] if x['snapshot_relative'].endswith('.safetensors'))['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'changed or missing'):
            p.check_protocol(manifest, self.manifest)

    def test_non_documentation_extra_rejected(self):
        manifest = self.rh110_manifest()
        item = copy.deepcopy(manifest['base_files'][0]); item['snapshot_relative'] = 'extra_model.safetensors'
        manifest['base_files'].append(item)
        with self.assertRaisesRegex(ValueError, 'Unexpected extra'):
            p.check_protocol(manifest, self.manifest)


if __name__ == '__main__':
    unittest.main()
