"""Authored fixtures only; live cases require both production sandbox layers."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from . import solver_diagnostic as s
from . import correctness_report as report
from . import evaluate
from .test_evaluate import fixtures, fixture_result
from .test_metrics import row, complete_rows, conditions, plan

GOOD = 'class Solution:\n    def add(self, a, b):\n        return a + b\n'
BAD = 'class Solution:\n    def add(self, a, b):\n        return 0\n'
EVAL_FAIL = 'def run_tests():\n    raise RuntimeError("authored evaluator failure")\n'


class Installation:
    def __init__(self):
        self.counts = dict.fromkeys(s.COUNTERS, 0)

    def report(self):
        return self.counts.copy()


def diagnostic(correct=True, whole=False):
    installed = Installation()
    def primitive(**kwargs):
        installed.counts['calls'] += 1
        value = copy.deepcopy(fixture_result()['gt_result'])
        if not correct:
            value.update(tests_passed=0, pass_rate=0.0, test_errors=['AssertionError'])
        return value
    return s.evaluate({'parsed_response': GOOD + EVAL_FAIL + 'run_tests()'},
                      {'gt_answer': ['assert Solution().add(2,3) == 5']}, primitive, installed, whole)


class IsolationTests(unittest.TestCase):
    def test_failing_evaluator_removed_without_running_any_code(self):
        isolated = s.isolate(GOOD + EVAL_FAIL + 'run_tests()\n', 'run_tests')
        self.assertEqual(isolated['status'], 'isolated')
        self.assertNotIn('run_tests', isolated['program'])
        self.assertIn('a + b', isolated['program'])
        self.assertEqual(len(isolated['removed_nodes']), 2)

    def test_helpers_constants_imports_and_solver_bodies_preserved(self):
        source = 'from math import sqrt\nOFFSET = 1\ndef helper(x): return sqrt(x) + OFFSET\n' + \
            'class Solution:\n    def add(self,a,b): return helper(a) + b\n    def run_tests(self): raise RuntimeError()'
        isolated = s.isolate(source, 'run_tests')
        self.assertEqual(isolated['status'], 'isolated')
        self.assertIn('helper', isolated['program'])
        self.assertIn('OFFSET = 1', isolated['program'])
        self.assertNotIn('run_tests', isolated['program'])

    def test_absent_evaluator_no_fabricated_removal(self):
        isolated = s.isolate(GOOD, 'run_tests')
        self.assertEqual(isolated['status'], 'isolated')
        self.assertEqual(isolated['removed_nodes'], [])

    def test_randomized_name_and_main_guard(self):
        source = (GOOD + EVAL_FAIL + 'if __name__ == "__main__":\n    run_tests()\n').replace('run_tests', 'verify_z73')
        isolated = s.isolate(source, 'verify_z73')
        self.assertEqual(isolated['status'], 'isolated')
        self.assertNotIn('verify_z73', isolated['program'])

    def test_instance_construction_used_only_by_driver_is_removed(self):
        for wrapper in ('sol = Solution()\nsol.run_tests()\n',
                        'if __name__ == "__main__":\n    sol = Solution()\n    sol.run_tests()\n'):
            isolated = s.isolate(GOOD + EVAL_FAIL + wrapper, 'run_tests')
            self.assertEqual(isolated['status'], 'isolated')
            self.assertNotIn('sol =', isolated['program'])

    def test_shared_driver_alias_or_nontrivial_constructor_is_unknown(self):
        for extra in ('def helper(): return sol\n', 'def helper(): return sol.add(1,2)\n'):
            isolated = s.isolate(GOOD + EVAL_FAIL + extra + 'sol = Solution()\nsol.run_tests()', 'run_tests')
            self.assertEqual(isolated['reason'], 'evaluator_driver_alias_used_by_solver')
        self.assertEqual(s.isolate(GOOD + EVAL_FAIL + 'sol=Solution(config())\nsol.run_tests()', 'run_tests')['status'], 'unsupported')

    def test_ambiguous_initialization_and_main_guard_unknown(self):
        for suffix in ('cache = build_cache()\n', 'Solution.add = lambda self,a,b: 0\n',
                       'if __name__ == "__main__":\n    OFFSET = 4\n',
                       'run_tests()\nprint("other work")\n'):
            with self.subTest(suffix=suffix):
                self.assertEqual(s.isolate(GOOD + EVAL_FAIL + suffix, 'run_tests')['status'], 'unsupported')

    def test_evaluator_dependencies_and_dynamic_access_unknown(self):
        for body in ('return run_tests()', 'return self.run_tests()',
                     'return globals()["run_tests"]()', 'f = run_tests\n        return f()',
                     'return getattr(self, "run_tests")()', 'return eval("a+b")'):
            source = 'class Solution:\n    def add(self,a,b):\n        ' + body + '\n' + EVAL_FAIL
            self.assertEqual(s.isolate(source, 'run_tests')['status'], 'unsupported')

    def test_malformed_or_oversize_source_unknown(self):
        for source in (None, '', GOOD + 'def run_tests(:', '#' * (1024 * 1024 + 1)):
            self.assertEqual(s.isolate(source, 'run_tests')['status'], 'unsupported')

    def test_evaluator_header_removed_but_solver_eager_headers_rejected(self):
        self.assertEqual(s.isolate(GOOD + '@bad()\ndef run_tests(x=bad()): pass', 'run_tests')['status'], 'isolated')
        for prefix in ('@bad()\n', ''):
            self.assertEqual(s.isolate(prefix + 'def solver(x=bad()): return x', 'run_tests')['status'], 'unsupported')


class PrimitiveTests(unittest.TestCase):
    def test_separate_success_and_failure_preserve_whole_program(self):
        for correct in (False, True):
            result = diagnostic(correct)
            s.validate(result, False)
            self.assertIs(result['solver_correctness'], correct)
            self.assertIs(result['whole_program_ground_truth_success'], False)
            self.assertEqual(result['transport']['calls'], 1)

    def test_unreached_tests_protocol_counts_overflow_and_infra_unknown(self):
        p = fixture_result()['gt_result']
        for update, transport, status in (
            ({'tests_evaluated': 0, 'tests_passed': 0, 'pass_rate': 0.0}, {}, 'unknown'),
            ({'tests_passed': 2}, {}, 'unknown'),
            ({}, {'output_overflow': 1}, 'unknown'),
            ({}, {'transport_error': 1}, 'infrastructure_failure'),
            ({'test_errors': ['Evaluator subprocess exited']}, {}, 'unknown')):
            value = s.assess({**p, **update}, transport, 1)
            self.assertEqual(value[0], status)
            self.assertIsNone(value[1])

    def test_malformed_receipts_rejected(self):
        for mutate in (lambda d: d.update(status='unsupported'), lambda d: d.update(solver_correctness=None),
                       lambda d: d.update(whole_program_ground_truth_success=True),
                       lambda d: d['isolation'].update(program='changed'),
                       lambda d: d['primitive_result'].update(tests_passed=0)):
            result = diagnostic()
            mutate(result)
            with self.assertRaises(ValueError):
                s.validate(result, False)

    def test_extra_call_is_separate_and_original_output_unchanged(self):
        request, prepared, example = fixtures()
        original = fixture_result(correct=False)
        original['parsed_response'] = GOOD + EVAL_FAIL + 'run_tests()'
        saved = copy.deepcopy(original)
        installed = Installation()
        class Evaluator:
            def evaluate(self, *_):
                installed.counts['calls'] += 5
                return original
            def evaluator(self, **kwargs):
                installed.counts['calls'] += 1
                return fixture_result()['gt_result']
        result = evaluate.evaluate_one(request, prepared, example, Evaluator(), installed)
        self.assertEqual(result['repository_evaluation'], saved)
        self.assertEqual(original, saved)
        expected = evaluate.analyze_repository_result(saved, {'calls': 5})
        self.assertEqual({k: result[k] for k in expected if k != 'metrics'}, {k: v for k,v in expected.items() if k != 'metrics'})
        self.assertEqual({k: result['metrics'][k] for k in evaluate.BINARY_METRICS}, expected['metrics'])
        self.assertEqual(result['transport']['calls'], 5)
        self.assertEqual(result['correctness_diagnostics']['transport']['calls'], 1)
        self.assertTrue(result['correctness_diagnostics']['solver_correctness'])

    def test_diagnostic_exception_preserves_legacy_success(self):
        request, prepared, example = fixtures()
        class Evaluator:
            def evaluate(self, *_):
                return {**fixture_result(), 'parsed_response': GOOD}
            def evaluator(self, **kwargs):
                raise RuntimeError('authored transport failure')
        result = evaluate.evaluate_one(request, prepared, example, Evaluator(), Installation())
        self.assertTrue(result['metrics']['ground_truth_correctness'])
        self.assertEqual(result['evaluation_status'], 'evaluated')
        self.assertEqual(result['correctness_diagnostics']['status'], 'infrastructure_failure')

    def test_diagnostic_dispatch_exception_also_preserves_legacy(self):
        request, prepared, example = fixtures()
        class Evaluator:
            def evaluate(self, *_):
                return fixture_result()
        with mock.patch.object(s, 'evaluate', side_effect=RuntimeError('authored dispatch failure')):
            result = evaluate.evaluate_one(request, prepared, example, Evaluator(), Installation())
        self.assertTrue(result['metrics']['ground_truth_correctness'])
        self.assertEqual(result['evaluation_status'], 'evaluated')
        self.assertEqual(result['correctness_diagnostics']['status'], 'infrastructure_failure')


class ReportTests(unittest.TestCase):
    def test_historical_results_remain_not_collected_and_unknown(self):
        rows = complete_rows()
        value = report.summarize(rows, bootstrap_resamples=100)['by_split']['configuration_validation']['primary']
        for condition in value['conditions'].values():
            self.assertEqual(condition['diagnostic_status_counts'], {'not_collected': 2})
            self.assertIsNone(condition['solver_correctness']['estimate'])
            self.assertEqual(condition['solver_correctness']['identified_bounds'], [0.0, 1.0])

    def test_additive_reports_do_not_change_legacy_metrics_or_gates(self):
        rows = complete_rows()
        before = report.metrics.analyze(rows, conditions(), plan(), bootstrap_resamples=100)
        for r in rows:
            r['correctness_diagnostics'] = diagnostic(correct=r['condition_id'] == 'target')
        report.summarize(rows, bootstrap_resamples=100)
        after = report.metrics.analyze(rows, conditions(), plan(), bootstrap_resamples=100)
        self.assertEqual(before, after)

    def test_equal_problem_weights_and_paired_effects(self):
        rows = []
        for problem, n, correct in (('many', 9, True), ('one', 1, False)):
            for sample in range(n):
                for condition in ('baseline', 'target'):
                    r = row(problem, condition, sample)
                    r['correctness_diagnostics'] = diagnostic(correct=correct if condition == 'target' else False)
                    rows.append(r)
        value = report.summarize(rows, bootstrap_resamples=100)['by_split']['configuration_validation']['primary']
        self.assertEqual(value['conditions']['target']['solver_correctness']['estimate'], .5)
        self.assertEqual(value['paired_vs_baseline']['target']['solver_correctness']['estimate'], .5)


@unittest.skipUnless(os.environ.get('CODE_EVAL_SANDBOX') == 'bwrap' and Path('/work/src/evaluate/helpers.py').is_file(),
                     'Live authored cases require production outer/nested sandbox on Linux')
class SandboxIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .bounded_evaluator import install_bounded_evaluator
        from .sandbox import _visible_pids
        cls.repository = evaluate.make_repository_evaluator('/input/authored-unused.jsonl')
        cls.before_pids = _visible_pids()
        cls.records = []
        cls.installed = install_bounded_evaluator()
        evaluate.install_count_payload_guard(cls.installed)

    @classmethod
    def tearDownClass(cls):
        from .sandbox import _visible_pids
        transport = cls.installed.report()
        cls.installed.restore()
        if _visible_pids() != cls.before_pids or cls.repository.llm_gen is not None:
            raise RuntimeError('Authored fixture left a child process or created a model engine')
        Path('/output/authored_records.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in cls.records))
        Path('/output/authored_receipt.json').write_text(json.dumps({'status': 'verified',
            'fixture_records': len(cls.records), 'model_engine_created': False,
            'evaluator_descendants_released': True, 'transport': transport}, sort_keys=True) + '\n')

    def check_case(self, code, whole, solver, *, name='run_tests', status='evaluated'):
        from .eval_run import authored_inputs
        _, sources, examples, generations = authored_inputs()
        request, prepared, example = generations[0], sources[0], examples[0]
        request['result']['completion'] = '```python\n' + code + '\n```'
        example['prompt_metadata']['test_func_name'] = name
        example['prompt_metadata']['test_func_code'] = example['prompt_metadata']['test_func_code'].replace('run_tests', name)
        example['hint'] = name
        value = evaluate.evaluate_one(request, prepared, example, self.repository, self.installed)
        self.assertIs(value['repository_evaluation']['eq_correct'], whole)
        self.assertIs(value['metrics']['ground_truth_correctness'], whole)
        self.assertEqual(value['correctness_diagnostics']['status'], status)
        self.assertIs(value['correctness_diagnostics']['solver_correctness'], solver)
        s.validate(value['correctness_diagnostics'], whole)
        self.assertLessEqual(value['correctness_diagnostics']['transport']['calls'], 1)
        self.assertEqual(value['correctness_diagnostics']['transport']['transport_error'], 0)
        self.records.append({'authored_case': self._testMethodName, **value})
        return value

    def test_correct_solver_with_evaluator_raising_before_gt(self):
        value = self.check_case(GOOD + EVAL_FAIL + 'run_tests()', False, True)
        self.assertEqual(value['repository_evaluation']['gt_result']['tests_evaluated'], 0)
        self.assertEqual(value['correctness_diagnostics']['primitive_result']['tests_evaluated'], 1)

    def test_incorrect_solver_with_noop_evaluator(self):
        self.check_case(BAD + 'def run_tests(): pass\nrun_tests()', False, False)

    def test_correct_solver_with_evaluator_mutation(self):
        self.check_case(GOOD + 'def run_tests():\n    Solution.add = lambda self,a,b: 0\nrun_tests()', False, True)

    def test_main_guard_randomized_evaluator(self):
        code = (GOOD + EVAL_FAIL + 'if __name__ == "__main__":\n    run_tests()').replace('run_tests', 'verify_z73')
        # The legacy harness executes in its own globals; guard behavior is an
        # observed harness detail, not assumed from the syntax alone.
        parsed = self.repository.evaluator(response=code, test_list=['assert Solution().add(2,3)==5'], skip_parse=True)
        self.check_case(code, parsed['pass_rate'] == 1, True, name='verify_z73')

    def test_helpers_kept_and_class_evaluator_removed(self):
        code = 'def helper(a,b): return a+b\nclass Solution:\n    def add(self,a,b): return helper(a,b)\n    def run_tests(self): raise RuntimeError()\nSolution().run_tests()'
        self.check_case(code, False, True)

    def test_instance_wrapper_cannot_execute_solver_constructor_before_gt(self):
        # First construction during the trusted test is valid. Evaluator-only
        # construction and execution must not modify the tested solver state.
        code = 'class Solution:\n    def add(self,a,b): return a+b\n    def run_tests(self): raise RuntimeError()\nsol = Solution()\nsol.run_tests()'
        self.check_case(code, False, True)

    def test_main_guard_instance_wrapper(self):
        code = 'class Solution:\n    def add(self,a,b): return a+b\n    def run_tests(self): raise RuntimeError()\nif __name__ == "__main__":\n    sol = Solution()\n    sol.run_tests()'
        parsed = self.repository.evaluator(response=code, test_list=['assert Solution().add(2,3)==5'], skip_parse=True)
        self.check_case(code, parsed['pass_rate'] == 1, True)

    def test_worker_journal_and_additive_report_success(self):
        from .eval_run import authored_inputs
        _, prepared, dataset, generations = authored_inputs()
        generations[0]['result']['completion'] = '```python\n' + GOOD + EVAL_FAIL + 'run_tests()\n```'
        inputs = Path('/output/authored_worker_inputs')
        inputs.mkdir()
        for name, values in (('requests', generations), ('prepared', prepared), ('dataset', dataset)):
            (inputs / (name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
        output = Path('/output/authored_worker_results')
        summary = evaluate.run(inputs/'requests.jsonl', inputs/'dataset.jsonl', inputs/'prepared.jsonl', output)
        self.assertEqual(summary['status'], 'succeeded')
        self.assertEqual(summary['solver_diagnostic_status'], {'evaluated': 2})
        self.assertEqual(summary['input_config']['correctness_diagnostics']['max_additional_calls_per_completion'], 1)
        records = list(evaluate.jsonl(output/'records.jsonl'))
        self.assertEqual([r['metrics']['ground_truth_correctness'] for r in records], [False, False])
        self.assertEqual([r['correctness_diagnostics']['solver_correctness'] for r in records], [True, False])
        diagnostic_report = report.summarize(records, bootstrap_resamples=100)
        (output/'correctness_diagnostics.json').write_text(json.dumps(diagnostic_report,sort_keys=True)+'\n')
        baseline = diagnostic_report['by_split']['configuration_validation']['primary']['conditions']['baseline']
        self.assertEqual(baseline['whole_program_ground_truth_success']['estimate'], 0)
        self.assertEqual(baseline['solver_correctness']['estimate'], .5)
        self.assertEqual(evaluate.run(inputs/'requests.jsonl', inputs/'dataset.jsonl', inputs/'prepared.jsonl', output, resume=True), summary)

    def test_evaluator_absent(self):
        self.check_case(GOOD, True, True)

    def test_solver_evaluator_dependency_is_unknown(self):
        code = 'class Solution:\n    def add(self,a,b): return run_tests()\ndef run_tests(): return 5'
        self.check_case(code, True, None, status='unsupported')

    def test_malformed_evaluator_is_unknown(self):
        self.check_case(GOOD + 'def run_tests(:', False, None, status='unsupported')


if __name__ == '__main__':
    unittest.main()
