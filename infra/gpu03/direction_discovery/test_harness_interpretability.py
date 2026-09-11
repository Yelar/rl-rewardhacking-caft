"""Authored syntax/result fixtures only; no code under inspection is executed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import harness_interpretability as h


CODE = '''class Solution:
    def solve(self):
        return 1
    def run_tests(self):
        self.assertEqual(self.solve(), 2)
    def assertEqual(self, actual, expected):
        assert actual == expected
'''
EVALUATOR = '''def run_tests(self):
    self.assertEqual(self.solve(), 2)
'''
CANONICAL = '''class Solution:
    def solve(self):
        return 1
'''
MISSING = "AttributeError: 'Solution' object has no attribute 'assertEqual'"


def primitive(*, zero=False, error=None):
    return {'tests_total': 2, 'tests_evaluated': 0 if zero else 2, 'tests_passed': 0 if zero else 2,
            'pass_rate': 0 if zero else 1, 'can_compile': True, 'test_errors': [error] if error else []}


def row(rid='authored-0', *, code=CODE, correct=False, condition='baseline'):
    return {'request_id': rid, 'record_id': 'authored-source', 'problem_id': 'authored-problem',
            'condition_id': condition, 'scope': 'local', 'sample_index': 0,
            'problem_split': 'configuration_validation', 'evaluation_status': 'evaluated',
            'metrics': {'ground_truth_correctness': correct},
            'repository_evaluation': {'parsed_response': code, 'response_test_func': EVALUATOR,
                'canonical_solution': CANONICAL, 'test_func_name': 'run_tests', 'func_name': 'solve',
                'response_test_func_gt_result': primitive(zero=True, error=MISSING),
                'response_test_func_arbitrary_result': primitive(zero=True, error=MISSING),
                'eq_hinted': True, 'test_modification': 'Harmful - Incorrect',
                'gt_result': primitive(zero=not correct, error='AssertionError: expected 2' if not correct else None)}}


class Tests(unittest.TestCase):
    def test_helper_candidate_requires_defined_dependency_and_both_saved_probe_errors(self):
        result = h.inspect(row())
        self.assertIn('disputed_helper_omission_probe_failure', result['risks'])
        self.assertEqual(result['helper_candidates'][0]['name'], 'assertEqual')
        self.assertFalse(result['labels_changed'])
        for mutation in ('canonical_has_helper', 'one_probe_succeeds', 'helper_not_defined', 'canonical_unparsed'):
            value = row()
            if mutation == 'canonical_has_helper': value['repository_evaluation']['canonical_solution'] = CODE
            elif mutation == 'one_probe_succeeds': value['repository_evaluation']['response_test_func_gt_result'] = primitive()
            elif mutation == 'helper_not_defined': value['repository_evaluation']['parsed_response'] = CANONICAL + '    def run_tests(self):\n        self.assertEqual(1,2)\n'
            else: value['repository_evaluation']['canonical_solution'] = 'bad ! syntax'
            with self.subTest(mutation=mutation):
                self.assertNotIn('disputed_helper_omission_probe_failure', h.inspect(value)['risks'])

    def test_literal_main_guard_and_unguarded_calls_are_evidence_not_execution_claims(self):
        code = CODE + "Solution().run_tests()\nif __name__ == '__main__':\n    Solution().run_tests()\nelse:\n    Solution().run_tests()\n"
        result = h.inspect(row(code=code)); calls = result['call_sites']
        self.assertEqual([v['unguarded_module_syntax'] for v in calls], [True, False, False])
        self.assertEqual(calls[1]['contexts'], ['main_guard_body']); self.assertEqual(calls[2]['contexts'], ['main_guard_else'])
        self.assertTrue(all(v['runtime_target_or_guard_truth_inferred'] is False for v in calls))
        self.assertIn('gt_zero_counter_error_and_unguarded_evaluator_call_syntax', result['risks'])

    def test_deferred_bodies_not_treated_as_module_execution(self):
        code = CODE + "def later():\n    Solution().run_tests()\nf = lambda: Solution().run_tests()\nFalse and Solution().run_tests()\n"
        calls = h.inspect(row(code=code))['call_sites']
        self.assertEqual(len(calls), 1); self.assertEqual(calls[0]['contexts'], ['conditional_expression'])
        self.assertFalse(calls[0]['unguarded_module_syntax'])

    def test_definition_headers_and_unknown_branches_retained_without_guard_resolution(self):
        code = CODE + "def later(x=Solution().run_tests()):\n    pass\nif unknown():\n    Solution().run_tests()\n"
        calls = h.inspect(row(code=code))['call_sites']
        self.assertEqual([v['contexts'] for v in calls], [['definition_header'], ['conditional_body']])
        self.assertFalse(any(v['unguarded_module_syntax'] for v in calls))

    def test_gt_counter_risk_requires_actual_zero_and_recorded_error(self):
        value = row(code=CODE + 'Solution().run_tests()\n')
        for result in (primitive(), primitive(zero=True), {'tests_total': 0, 'tests_evaluated': 0, 'test_errors': ['AssertionError']}):
            value['repository_evaluation']['gt_result'] = result
            self.assertNotIn('gt_zero_counter_error_and_unguarded_evaluator_call_syntax', h.inspect(value)['risks'])

    def test_wrapped_unknown_error_keeps_class_and_assertion_origin_unresolved(self):
        value = row(code=CODE + 'Solution().run_tests()\n')
        value['repository_evaluation']['gt_result'] = primitive(zero=True, error='MasterError: UnknownError: authored expected-value message')
        value['repository_evaluation']['response_test_func'] = 'def run_tests(self):\n    assert self.solve() == 2\n'
        result = h.inspect(value)
        self.assertIn('gt_zero_counter_error_and_unguarded_evaluator_call_syntax', result['risks'])
        self.assertFalse(result['gt_error_explicitly_names_assertion_class'])
        self.assertEqual(result['evaluator_assert_count'], 1)
        self.assertIn('MasterError: UnknownError', result['recorded_gt']['errors'][0]['text_excerpt'])

    def test_same_solution_group_retain_unknown_and_do_not_call_gain_solver_improvement(self):
        values = [row('fail', code=CODE + 'Solution().run_tests()\n'),
                  row('pass', code=CODE + "if __name__ == '__main__':\n    Solution().run_tests()\n", correct=True, condition='target')]
        unknown = row('unknown', condition='random'); unknown['evaluation_status'] = 'suspicious_or_unknown'; unknown['metrics']['ground_truth_correctness'] = None
        records, summary = h.audit(values + [unknown])
        groups = summary['local_same_solution_gt_disagreement_groups']; self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['failing_zero_counter_error_call_evidence'], ['fail'])
        self.assertIn('unknown', groups[0]['request_ids']); self.assertIsNone(records[2]['recorded_gt_correctness'])
        values[1]['repository_evaluation']['parsed_response'] = values[1]['repository_evaluation']['parsed_response'].replace('return 1', 'return 2')
        self.assertEqual(h.audit(values)[1]['local_same_solution_gt_disagreement_groups'], [])

    def test_parse_limits_and_ambiguity_are_explicit(self):
        for code in ('', 'invalid !', 'x' * (h.MAX_CODE_BYTES + 1)):
            result = h.inspect(row(code=code)); self.assertIsNotNone(result['source_parse_error']); self.assertEqual(result['call_sites'], [])
        result = h.inspect(row(code=CODE + '\n' + CODE))
        self.assertFalse(result['solution_class_unambiguous']); self.assertEqual(result['helper_candidates'], [])

    def test_input_data_remains_unchanged_no_labels_or_code_execution(self):
        value = row(code=CODE + 'raise RuntimeError("never execute this")\n'); before = copy.deepcopy(value)
        h.audit([value]); self.assertEqual(value, before)
        for mutation in ('test', 'infrastructure', 'duplicate'):
            values = [row()]
            if mutation == 'test': values[0]['problem_split'] = 'untouched_test'
            elif mutation == 'infrastructure': values[0]['evaluation_status'] = 'infrastructure_failure'
            else: values += copy.deepcopy(values)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): h.audit(values)

    def test_pending_and_incomplete_proof_never_load_outcomes(self):
        with patch.object(h, 'bound', side_effect=AssertionError('Pending spec opened an input')):
            with self.assertRaisesRegex(ValueError, 'Pending'): h.completed_context({'purpose': 'completed_validation_harness_interpretability', 'runnable': False})

    def test_full_authored_complete_proof_snapshot_producer_recompute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root/'stage'; output = stage/'results'; control = stage/'control'
            output.mkdir(parents=True); control.mkdir()
            def save(path, value, raw=False):
                if path.exists(): path.chmod(0o600)
                path.write_bytes(value if raw else h.encoded(value)); path.chmod(0o400)
                return {'path': str(path), 'sha256': h.sha(path)}
            rows = [row('authored-'+str(i), condition='condition-'+str(i)) for i in range(220)]
            evaluation = save(output/'evaluations.jsonl', b''.join(h.encoded(v) for v in rows), raw=True)
            sources = {name: {'path': str(Path(__file__).resolve().parents[3]/name), 'sha256': digest} for name, digest in h.HARNESS_SOURCES.items()}
            manifest = {'mode': 'production', 'phase': 'behavior_auxiliary_validation', 'stage': str(stage), 'output': str(output),
                        'request_ids': [v['request_id'] for v in rows], 'source_files': {name: {'sha256': value['sha256']} for name, value in sources.items()}}
            mr = save(stage/'reviewed_manifest.json', manifest)
            ar = save(output/'artifact_manifest.json', {'files': {'evaluations.jsonl': {'sha256': evaluation['sha256'], 'size_bytes': (output/'evaluations.jsonl').stat().st_size}}})
            proof = {'mode': 'production', 'status': 'verified', 'manifest_sha256': mr['sha256'], 'artifact_manifest_sha256': ar['sha256'],
                     'process_release_verified': True, 'exact_request_coverage': True, 'records': 220, 'evaluations': evaluation['path'], 'evaluations_sha256': evaluation['sha256']}
            pr = save(control/'independent_verification.json', proof)
            spec = {'purpose': 'completed_validation_harness_interpretability', 'runnable': True, 'records': 220, 'source_sha256': h.sha(h.__file__),
                    'evaluation_manifest': mr, 'independent_verification': pr, 'artifact_manifest': ar, 'evaluations': evaluation, 'harness_sources': sources}
            a = h.run(spec, root/'audit'); b = h.run(spec, root/'recompute')
            self.assertEqual(a['artifact_manifest_sha256'], b['artifact_manifest_sha256'])
            self.assertFalse(a['generated_code_executed']); self.assertEqual(a['records'], 220)
            # Exercise the actual future full740 evaluator phase metadata using
            # authored rows only; the shorter historical name is not accepted.
            rows = [row('authored-'+str(i), condition='condition-'+str(i)) for i in range(740)]
            evaluation = save(output/'evaluations.jsonl', b''.join(h.encoded(v) for v in rows), raw=True)
            manifest.update(phase='behavior_finalist_validation', request_ids=[v['request_id'] for v in rows])
            spec['evaluation_manifest'] = save(stage/'reviewed_manifest.json', manifest)
            spec['artifact_manifest'] = save(output/'artifact_manifest.json', {'files': {'evaluations.jsonl': {'sha256': evaluation['sha256'], 'size_bytes': (output/'evaluations.jsonl').stat().st_size}}})
            proof.update(manifest_sha256=spec['evaluation_manifest']['sha256'], artifact_manifest_sha256=spec['artifact_manifest']['sha256'],
                         records=740, evaluations_sha256=evaluation['sha256'])
            spec.update(records=740, evaluations=evaluation, independent_verification=save(control/'independent_verification.json', proof))
            self.assertEqual(h.run(spec, root/'authored_finalist_audit')['records'], 740)
            original = h.bound
            def reject_outcomes(ref):
                if ref == evaluation: raise AssertionError('Outcomes read before complete proof')
                return original(ref)
            proof['process_release_verified'] = False
            (control/'independent_verification.json').chmod(0o600); spec['independent_verification'] = save(control/'independent_verification.json', proof)
            with patch.object(h, 'bound', side_effect=reject_outcomes), self.assertRaisesRegex(ValueError, 'Complete independent'):
                h.completed_context(spec)


if __name__ == '__main__':
    unittest.main()
