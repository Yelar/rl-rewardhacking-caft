import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import select_behavior as s
except ImportError:
    import select_behavior as s


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (s.behavior.canonical(value) + '\n').encode()
    path.write_bytes(data)
    return {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


class SelectionTests(unittest.TestCase):
    """Authored data only; external scientific replay is explicitly mocked.

    The rank_verified suite separately qualifies that replay. These tests cover
    the selection boundary, source identity, original selectors and no-launch
    behavior, with the real behavior_plan signature/control construction.
    """
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rank_root = self.root / 'combined'
        self.rank_root.mkdir()
        self.master = {'plan_version': 2, 'sweep': {'teacher_forced_validation_problems': [str(i) for i in range(12)]}}
        self.master_path = self.root / 'master/plan.json'
        self.master_sha = write(self.master_path, self.master)['sha256']
        self.parent_path = self.root / 'master/parent.json'
        self.parent_sha = write(self.parent_path, {})['sha256']
        self.candidate_path = self.root / 'candidates/vectors.safetensors'
        self.candidate_path.parent.mkdir()
        self.candidate_path.write_bytes(b'authored fake candidate bytes; never load as tensors')
        self.candidate_sha = s.behavior.sha256(self.candidate_path)
        self.targets = [self.target(4, pc=0), self.target(4, pc=1), self.target(5), self.target(3, pc=2)]
        self.plans = []
        self.ranked = []
        for phase, selected in (('coarse', self.targets[:2]), ('refinement', self.targets[2:])):
            conditions = {'baseline': {'role': 'baseline', 'layers': []}, **dict(selected)}
            for layer in sorted({target['layers'][0]['layer'] for _, target in selected}):
                for base in (6101, 6102, 6103):
                    conditions[f'random:L{layer:02d}:base{base}'] = {'role': 'random', 'random_seed_base': base,
                        'layers': [{'layer': layer, 'kind': 'random', 'rank': 1, 'seed': s.behavior.stable_seed(base, layer)}]}
            self.plans.append({'mode': 'tf', 'phase': phase, 'evaluation_partition': 'configuration_validation',
                'master_plan_sha256': self.master_sha, 'selected_problem_ids': self.master['sweep']['teacher_forced_validation_problems'],
                'candidate_artifact_manifest_sha256': 'c' * 64, 'conditions': conditions,
                'requests': [{'request_id': phase + ':' + key, 'condition_id': key} for key in conditions
                             if phase == 'coarse' or key != 'baseline']})
        for index, (cid, condition) in enumerate(self.targets):
            layer = condition['layers'][0]['layer']
            self.ranked.append({'condition_id': cid, 'candidate_id': condition['candidate_id'], 'role': 'target', 'layer': layer,
                'priority_beyond_random_mean': {'mean': -float(index)},
                'metrics': {'benign_correct_nll_increase': {'mean': 0.1}},
                'matched_random_condition_ids': [f'random:L{layer:02d}:base{base}' for base in (6101, 6102, 6103)]})
        self.spec_extra = {}
        self.success_extra = {}
        self.ranking_extra = {}
        self.budget_extra = {}
        self.seal()
        self.cuda = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''})
        self.cuda.start()
        self.addCleanup(self.cuda.stop)
        self.validate = patch.object(s.behavior, 'validate_master')
        self.validate.start()
        self.addCleanup(self.validate.stop)
        self.verifier = patch.object(s, 'independently_verify_ranking', side_effect=lambda root: copy.deepcopy(self.proof))
        self.mock_verifier = self.verifier.start()
        self.addCleanup(self.verifier.stop)

    def target(self, layer, pc=None):
        family = 'pca' if pc is not None else 'harmful_vs_benign'
        kind = 'pc' if pc is not None else 'v_change'
        cid = f'L{layer:02d}.transition.' + (f'pc{pc:02d}' if pc is not None else 'mean.' + family + '.' + kind)
        selector = {'key': 'pca.pcs', 'column': pc} if pc is not None else {'key': family + '.' + kind}
        return 'target:' + cid, {'role': 'target', 'candidate_id': cid, 'family': family, 'candidate_kind': kind,
            'window': 'transition', 'interpretation': 'authored fixture',
            'layers': [{'layer': layer, 'kind': 'candidate', 'path': str(self.candidate_path),
                        'sha256': self.candidate_sha, 'selectors': [selector]}]}

    def seal(self):
        refs = []
        for index, plan in enumerate(self.plans):
            stage = self.root / f'phase{index}'
            request = stage / 'input/request_plan.json'
            binding = write(request, plan)
            manifest = stage / 'reviewed_manifest.json'
            manifest_binding = write(manifest, {'stage': str(stage), 'bound_files': {str(request): binding}})
            refs.append({'manifest_path': str(manifest), 'manifest_sha256': manifest_binding['sha256']})
        count = sum(len(plan['requests']) for plan in self.plans)
        spec = {'master_path': str(self.master_path), 'master_sha256': self.master_sha,
            'parent_master_path': str(self.parent_path), 'parent_master_sha256': self.parent_sha,
            'phases': refs, 'prepare_neighbors': False, 'output': str(self.rank_root), **self.spec_extra}
        self.artifact = {'algorithm': 'sha256', 'files': {}}
        values = {'resolved_spec.json': spec,
            'tf_ranking.json': {'master_plan_sha256': self.master_sha, 'status': 'exploratory_TF_prioritization_only',
                'requests_verified': count, 'problem_ids': self.master['sweep']['teacher_forced_validation_problems'],
                'metric': 'evaluator__transition', 'target_ranking': self.ranked, **self.ranking_extra},
            'phase_budget.json': {'untouched_test_requests': 0, **self.budget_extra},
            'SUCCESS.json': {'status': 'succeeded', 'mode': 'verified_likelihood_prioritization_only',
                'model_work_launched': False, 'test_requests': 0, 'requests': count, 'validation_problems': 12, **self.success_extra},
            'source_sha256.json': {}}
        for name, value in values.items():
            self.artifact['files'][name] = write(self.rank_root / name, value)
        artifact_sha = write(self.rank_root / 'artifact_manifest.json', self.artifact)['sha256']
        self.proof = {'status': 'verified', 'requests': count, 'test_requests': 0, 'model_work_launched': False,
            'ranking_sha256': self.artifact['files']['tf_ranking.json']['sha256'], 'artifact_manifest_sha256': artifact_sha}
        receipt = self.root / 'independent_verification.json'
        receipt_sha = write(receipt, self.proof)['sha256']
        self.args = {'ranking_root': str(self.rank_root), 'artifact_manifest_sha256': artifact_sha,
            'verification_receipt': str(receipt), 'verification_receipt_sha256': receipt_sha, 'master_sha256': self.master_sha}

    def test_exact_three_even_when_likelihood_scores_are_nonpositive(self):
        selected = s.build_selection(**self.args)
        self.assertEqual(selected['selected_condition_ids_in_priority_order'], [key for key, _ in self.targets[:3]])
        self.assertEqual(selected['conditions'], dict(self.targets[:3]))
        self.assertEqual(selected['behavior_conditions_including_controls'], 10)
        self.assertEqual(selected['expected_screening_generation_requests'], 600)
        self.assertIsNone(selected['selection_rule']['numeric_threshold'])
        self.assertFalse(selected['budget_reserved'])
        self.assertTrue(selected['no_test_outcomes_used'])
        self.mock_verifier.assert_called_once_with(self.rank_root)
        # The two layer4 targets retain one shared set of the original TF seeds.
        a, b = selected['selection_provenance'][:2]
        self.assertEqual(a['random_control_identity_mapping'], b['random_control_identity_mapping'])
        self.assertEqual(a['signature'], [[4, 1]])

    def test_no_process_launch_and_independent_output_verification(self):
        output = self.root / 'prepared_selection'
        with patch('subprocess.Popen', side_effect=AssertionError('No subprocess may launch in this authored test')):
            made = s.write_selection(output=str(output), **self.args)
            verified = s.verify_selection(output, made['artifact_manifest_sha256'])
        self.assertEqual(verified['status'], 'verified')
        self.assertEqual(made['selected_conditions_sha256'], verified['selected_conditions_sha256'])
        self.assertEqual(set(p.name for p in output.iterdir()), s.SELECTION_FILES | {'artifact_manifest.json'})
        with self.assertRaisesRegex(ValueError, 'fresh'):
            s.write_selection(output=str(output), **self.args)

    def test_coarse_only_and_neighbor_planning_rejected_before_replay(self):
        self.plans.pop()
        self.seal()
        with self.assertRaisesRegex(ValueError, 'coarse plus refinement'):
            s.build_selection(**self.args)
        self.mock_verifier.assert_not_called()

    def test_prepare_neighbors_must_be_explicitly_false(self):
        self.spec_extra['prepare_neighbors'] = True
        self.seal()
        with self.assertRaisesRegex(ValueError, 'no neighbor planning'):
            s.build_selection(**self.args)

    def test_duplicate_ranked_ids_and_source_condition_ids_rejected(self):
        original = copy.deepcopy(self.ranked)
        self.ranked.append(copy.deepcopy(self.ranked[0]))
        self.seal()
        with self.assertRaisesRegex(ValueError, 'duplicate ranked'):
            s.build_selection(**self.args)
        self.ranked = original
        cid, condition = self.targets[0]
        self.plans[1]['conditions'][cid] = copy.deepcopy(condition)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Duplicate/conflicting'):
            s.build_selection(**self.args)

    def test_duplicate_json_keys_and_nonfinite_json_fail_closed(self):
        for data in (b'{"conditions":{},"conditions":{}}', b'{"score":NaN}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                s.parse_json(data)

    def test_ranking_missing_source_candidate_rejected(self):
        self.ranked.pop()
        self.seal()
        with self.assertRaisesRegex(ValueError, 'omits or adds'):
            s.build_selection(**self.args)

    def test_out_of_order_and_nonfinite_ranking_rejected(self):
        self.ranked[0], self.ranked[1] = self.ranked[1], self.ranked[0]
        self.seal()
        with self.assertRaisesRegex(ValueError, 'frozen likelihood ordering'):
            s.build_selection(**self.args)
        with self.assertRaisesRegex(ValueError, 'Invalid likelihood'):
            s.ranking_key({'condition_id': 'target:a', 'priority_beyond_random_mean': {'mean': float('inf')},
                           'metrics': {'benign_correct_nll_increase': {'mean': 0}}})

    def test_ties_use_frozen_capability_then_condition_id_order(self):
        for item in self.ranked:
            item['priority_beyond_random_mean']['mean'] = 1
        self.ranked.sort(key=s.ranking_key)
        self.seal()
        result = s.build_selection(**self.args)
        self.assertEqual(result['selected_condition_ids_in_priority_order'], sorted(key for key, _ in self.targets)[:3])

    def test_rank_two_or_wrong_pc_column_cannot_enter_screening(self):
        cid = self.targets[0][0]
        selectors = self.plans[0]['conditions'][cid]['layers'][0]['selectors']
        selectors[0]['column'] = 8
        self.seal()
        with self.assertRaisesRegex(ValueError, 'PC selector'):
            s.build_selection(**self.args)
        selectors[0]['column'] = 0
        selectors.append({'key': 'pca.pcs', 'column': 1})
        self.seal()
        with self.assertRaisesRegex(ValueError, 'layer/rank/signature'):
            s.build_selection(**self.args)

    def test_diagnostic_v0_or_auxiliary_mean_is_not_an_eligible_target(self):
        cid, condition = self.targets[2]
        for invalid in ('L05.transition.mean.harmful_vs_benign.v0',
                        'L05.transition.mean.harmful_vs_benign_correct.v60'):
            target = copy.deepcopy(condition)
            target['candidate_id'] = invalid
            row = copy.deepcopy(self.ranked[2])
            row['candidate_id'] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'Only the frozen primary'):
                s.validate_target('target:' + invalid, target, row)

    def test_changed_candidate_or_source_plan_bytes_rejected(self):
        self.candidate_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'candidate bytes changed'):
            s.build_selection(**self.args)
        self.candidate_path.write_bytes(b'authored fake candidate bytes; never load as tensors')
        (self.root / 'phase1/input/request_plan.json').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            s.build_selection(**self.args)

    def test_random_vector_seed_or_rank_cannot_change(self):
        self.plans[0]['conditions']['random:L04:base6101']['layers'][0]['seed'] += 1
        self.seal()
        with self.assertRaisesRegex(ValueError, 'different layers, rank, or seed'):
            s.build_selection(**self.args)

    def test_bad_receipt_and_failed_reconstruction_produce_no_output(self):
        self.proof['requests'] += 1  # The saved, valid receipt remains unchanged.
        output = self.root / 'must_not_exist'
        with self.assertRaisesRegex(ValueError, 'Independent reconstruction'):
            s.write_selection(output=str(output), **self.args)
        self.assertFalse(output.exists())
        self.args['verification_receipt_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            s.write_selection(output=str(output), **self.args)
        self.assertFalse(output.exists())

    def test_test_use_wrong_master_and_wrong_phase_fail_closed(self):
        self.budget_extra['untouched_test_requests'] = 1
        self.seal()
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            s.build_selection(**self.args)
        self.budget_extra.clear()
        self.seal()
        wrong = dict(self.args, master_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'another master'):
            s.build_selection(**wrong)
        self.plans[1]['evaluation_partition'] = 'untouched_test'
        self.seal()
        with self.assertRaisesRegex(ValueError, 'phase order, partition'):
            s.build_selection(**self.args)

    def test_duplicate_requests_and_extra_package_files_rejected(self):
        self.plans[1]['requests'].append(copy.deepcopy(self.plans[0]['requests'][0]))
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Duplicate TF request'):
            s.build_selection(**self.args)
        (self.rank_root / 'unexpected.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'extra files'):
            s.build_selection(**self.args)

    def test_snapshot_change_during_replay_rejected(self):
        old = copy.deepcopy(self.proof)
        def mutate(root):
            self.ranked[0]['priority_beyond_random_mean']['mean'] = 5
            self.seal()
            return old
        self.mock_verifier.side_effect = mutate
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            s.build_selection(**self.args)

    def test_cuda_visible_or_output_under_frozen_input_rejected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), self.assertRaisesRegex(ValueError, 'hide CUDA'):
            s.build_selection(**self.args)
        with self.assertRaisesRegex(ValueError, 'frozen ranking'):
            s.write_selection(output=str(self.rank_root / 'new'), **self.args)
        with self.assertRaisesRegex(ValueError, 'frozen input'):
            s.write_selection(output=str(self.candidate_path.parent / 'new'), **self.args)


if __name__ == '__main__':
    unittest.main()
