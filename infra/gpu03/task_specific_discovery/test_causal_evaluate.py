"""Inert offline fixtures: never invoke generated-code evaluation or a sandbox."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from infra.gpu03.task_specific_discovery import causal_evaluate as ce
from infra.gpu03.task_specific_discovery.io_contract import RequestJournal, AmbiguousRequest, ContractError, hash_value


def fixture(protocol="fixed", rid="req-1"):
    name = "run_checks_a17" if protocol == "randomized" else "run_tests"
    prompt = [{"role": "user", "content": "inert authored fixture " + protocol}]
    example = {"id": 1, "prompt": prompt, "gt_answer": ["inert test text, never executed"],
               "func_name": "Solution().solve", "setup_code": "", "hint": None if protocol == "no_loophole" else "evaluator hint",
               "prompt_metadata": {"test_func_name": name}}
    completion = "inert authored completion, never executed"
    row = {"request_id": rid, "record_id": "source1", "condition_id": "baseline", "protocol": protocol,
           "problem_id": 1, "problem_id_key": "1", "problem_split": "configuration_validation",
           "prompt": prompt, "prompt_sha256": hash_value(prompt), "completion": completion,
           "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(), "completion_token_ids": [11, 12],
           "engine_prompt_token_ids": [1, 2], "sample_index": 0, "generation_seed": 1}
    primitive = {"can_compile": True, "is_formatted": True, "tests_total": 1, "tests_evaluated": 1,
                 "tests_passed": 1, "pass_rate": 1.0, "test_errors": []}
    raw = {name: copy.deepcopy(primitive) for name in ce.repository.PRIMITIVES}
    raw.update(response_test_func="def run_tests():\n    assert True", test_modification="None", reward_hack_label="Correct",
               parsed_response="class Solution:\n    def solve(self):\n        return 1\n",
               is_test_modification_harmful=False, is_reward_hack_strict=False, eq_correct=True,
               response_has_test_func=protocol != "no_loophole", is_parsed=True, is_formatted=True, can_compile=True)
    return row, example, raw


class InertInstallation:
    def __init__(self):
        self.counts = {k: 0 for k in ce.COUNTERS}
        self.reports = 0
        self.on_after = None
        self.restored = False

    def report(self):
        self.reports += 1
        if self.reports == 2 and self.on_after:
            self.on_after()
        return dict(self.counts)

    def restore(self):
        self.restored = True


class InertEvaluator:
    def __init__(self, raw, installation):
        self.raw = raw
        self.installation = installation
        self.calls = 0
        self.action = None
        self.llm_gen = None
        self.evaluator = SimpleNamespace(num_workers=2, timeout=3, memory_per_worker=1024, max_failures=1)

    def evaluate(self, example, completion):
        self.calls += 1
        if self.action:
            self.action()
        self.installation.counts["calls"] += 5
        return copy.deepcopy(self.raw)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.row, self.example, self.raw = fixture()
        self.installation = InertInstallation()
        self.evaluator = InertEvaluator(self.raw, self.installation)

    def tearDown(self):
        self.temp.cleanup()

    def journal(self, row=None, example=None):
        row, example = row or self.row, example or self.example
        return RequestJournal(self.root / "journal", run_token="authored-test", input_sha256="1" * 64,
                              request_hashes={row["request_id"]: ce.validate_request(row, example, row["protocol"])})

    def process(self, journal, **kwargs):
        return ce.process_one(journal, self.row, self.example, "fixed", self.root / "transport",
                              evaluator=self.evaluator, installation=self.installation, **kwargs)

    def test_raw_durable_before_first_after_counter_and_analysis(self):
        order = []
        with self.journal() as journal:
            def after():
                self.assertEqual(journal.primitive("req-1")["repository_evaluation"], self.raw)
                order.append("transport_after_raw_durable")
            self.installation.on_after = after
            def analyze(raw, delta):
                self.assertEqual(journal.primitive("req-1")["repository_evaluation"], self.raw)
                order.append("analysis_after_raw_durable")
                return ce.repository.analyze_repository_result(raw, delta)
            result = self.process(journal, analyzer=analyze)
        self.assertEqual(order, ["transport_after_raw_durable", "analysis_after_raw_durable"])
        self.assertEqual(result["analysis"]["evaluation_status"], "evaluated")
        self.assertEqual(self.evaluator.calls, 1)
        self.assertEqual(result["solver_only_diagnostic"]["additional_generated_code_calls"], 0)
        self.assertIs(result["structural_diagnostics"]["value"]["solution_present"], True)

    def test_solution_ast_failure_is_after_durable_raw_and_does_not_rescore(self):
        with self.journal() as journal:
            def fail(raw, target):
                self.assertEqual(journal.primitive('req-1')['repository_evaluation'], self.raw)
                raise MemoryError('authored solution AST failure')
            with mock.patch.object(ce, 'solution_presence', side_effect=fail):
                result = self.process(journal)
            self.assertEqual(result['structural_diagnostics']['status'], 'unknown')
            self.assertEqual(result['analysis']['evaluation_status'], 'evaluated')
            self.assertTrue(result['analysis']['metrics']['ground_truth_correctness'])
        with self.journal() as journal:
            again = self.process(journal)
        self.assertEqual(result, again)
        self.assertEqual(self.evaluator.calls, 1)

    def test_memory_and_recursion_errors_preserve_primitive_and_known_gt(self):
        for error in (MemoryError("AST stack"), RecursionError("AST depth")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                old = self.root
                self.root = Path(temp)
                with self.journal() as journal:
                    result = self.process(journal, analyzer=mock.Mock(side_effect=error))
                    self.assertEqual(journal.primitive("req-1")["repository_evaluation"], self.raw)
                    self.assertEqual(result["analysis"]["evaluation_status"], "metadata_unknown")
                    self.assertIs(result["analysis"]["metrics"]["ground_truth_correctness"], True)
                    self.assertIsNone(result["analysis"]["metrics"]["harmful_modification"])
                with self.journal() as journal:
                    previous = self.process(journal, analyzer=mock.Mock(side_effect=AssertionError("must not redo")))
                    self.assertEqual(previous, result)
                self.root = old

    def test_structural_failure_explicit_unknown_without_losing_known_analysis(self):
        with self.journal() as journal:
            result = self.process(journal, structural_callback=mock.Mock(side_effect=MemoryError("structure")))
            self.assertEqual(journal.primitive("req-1")["repository_evaluation"], self.raw)
        self.assertEqual(result["structural_diagnostics"]["status"], "unknown")
        self.assertEqual(result["metadata_status"], "unknown")
        self.assertTrue(result["analysis"]["metrics"]["ground_truth_correctness"])

    def test_legacy_swallowed_parser_recursion_is_explicit_unknown(self):
        with self.journal() as journal:
            with mock.patch.object(ce.repository.ast, "parse", side_effect=RecursionError("authored parser failure")):
                result = self.process(journal)
            self.assertEqual(journal.primitive("req-1")["repository_evaluation"], self.raw)
        self.assertEqual(result["metadata_status"], "unknown")
        self.assertEqual(result["analysis"]["evaluation_status"], "metadata_unknown")
        self.assertEqual(result["analysis"]["syntax"]["status"], "unknown")
        self.assertIs(result["analysis"]["metrics"]["ground_truth_correctness"], True)
        for name in ce.LOOPHOLE_METRICS:
            self.assertIsNone(result["analysis"]["metrics"][name])
        self.assertEqual(self.evaluator.calls, 1)

    def test_crash_after_raw_before_transport_recovers_unknown_without_rescoring(self):
        self.installation.on_after = mock.Mock(side_effect=KeyboardInterrupt())
        with self.journal() as journal:
            with self.assertRaises(KeyboardInterrupt):
                self.process(journal)
        self.evaluator.action = mock.Mock(side_effect=AssertionError("replayed"))
        with self.journal() as journal:
            result = self.process(journal)
        self.assertEqual(result["analysis"]["evaluation_status"], "transport_unknown")
        self.assertTrue(all(v is None for v in result["analysis"]["metrics"].values()))
        self.assertEqual(self.evaluator.calls, 1)

    def test_crash_after_transport_before_metadata_recovers_without_rescoring(self):
        with self.journal() as journal:
            with self.assertRaises(SystemExit):
                self.process(journal, analyzer=mock.Mock(side_effect=SystemExit(17)))
        self.evaluator.action = mock.Mock(side_effect=AssertionError("replayed"))
        with self.journal() as journal:
            result = self.process(journal)
        self.assertEqual(result["analysis"]["evaluation_status"], "evaluated")
        self.assertEqual(self.evaluator.calls, 1)

    def test_unreturned_primitive_is_ambiguous_not_synthetic_unknown_or_retry(self):
        self.evaluator.action = mock.Mock(side_effect=MemoryError("inside original evaluate"))
        with self.journal() as journal:
            with self.assertRaises(MemoryError):
                self.process(journal)
            self.assertEqual(journal.pending()["started_without_primitive"], ["req-1"])
            with self.assertRaisesRegex(ContractError, "Ambiguous"):
                self.process(journal)
        with self.assertRaises(AmbiguousRequest):
            with self.journal():
                pass
        self.assertEqual(self.evaluator.calls, 1)

    def test_raw_persistence_failure_cannot_proceed_to_analysis(self):
        analyzer = mock.Mock(side_effect=AssertionError("analysis before persistence"))
        with self.journal() as journal:
            with mock.patch.object(journal, "record_primitive", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    self.process(journal, analyzer=analyzer)
        analyzer.assert_not_called()
        self.assertEqual(self.installation.reports, 1)
        with self.assertRaises(AmbiguousRequest):
            with self.journal():
                pass

    def test_completed_result_reused_without_evaluator_or_analyzer(self):
        with self.journal() as journal:
            first = self.process(journal)
        with self.journal() as journal:
            second = ce.process_one(journal, self.row, self.example, "fixed", self.root / "transport",
                                    analyzer=mock.Mock(side_effect=AssertionError("analysis replay")))
        self.assertEqual(first, second)
        self.assertEqual(self.evaluator.calls, 1)

    def test_example_or_completion_change_rejects_existing_binding(self):
        with self.journal() as journal:
            self.process(journal)
        changed = copy.deepcopy(self.example)
        changed["gt_answer"] = ["different inert test"]
        with self.assertRaisesRegex(ContractError, "binding mismatch"):
            with self.journal(example=changed):
                pass
        self.assertEqual(self.evaluator.calls, 1)

    def test_corrupt_transport_sidecar_fails_closed_without_rescoring(self):
        with self.journal() as journal:
            with self.assertRaises(SystemExit):
                self.process(journal, analyzer=mock.Mock(side_effect=SystemExit(1)))
        path = ce._transport_path(self.root / "transport", "req-1")
        path.chmod(0o600)
        path.write_bytes(b"{\n")
        path.chmod(0o400)
        with self.journal() as journal:
            with self.assertRaises((ValueError, ContractError)):
                self.process(journal)
        self.assertEqual(self.evaluator.calls, 1)

    def test_transport_counter_regression_retains_raw_no_final(self):
        self.installation.counts["calls"] = 20
        self.installation.on_after = lambda: self.installation.counts.update(calls=1)
        with self.journal() as journal:
            with self.assertRaisesRegex(ContractError, "regressed"):
                self.process(journal)
            self.assertEqual(journal.pending()["primitive_ready"], ["req-1"])
        self.assertEqual(self.evaluator.calls, 1)

    def test_nonobject_return_is_retained_infrastructure_failure(self):
        self.evaluator.raw = None
        with self.journal() as journal:
            result = self.process(journal)
            self.assertIsNone(journal.primitive("req-1")["repository_evaluation"])
        self.assertEqual(result["analysis"]["evaluation_status"], "infrastructure_failure")

    def test_parser_failure_does_not_turn_infrastructure_problem_known_gt(self):
        self.installation.counts["transport_error"] = 1
        self.installation.on_after = lambda: self.installation.counts.update(transport_error=2)
        with self.journal() as journal:
            result = self.process(journal, analyzer=mock.Mock(side_effect=MemoryError()))
        self.assertIsNone(result["analysis"]["metrics"]["ground_truth_correctness"])

    def test_three_protocol_bindings_and_empty_completion_preserved(self):
        for protocol in ce.PROTOCOLS:
            row, example, _ = fixture(protocol)
            self.assertEqual(len(ce.validate_request(row, example, protocol)), 64)
            row["completion"] = ""
            row["completion_sha256"] = hashlib.sha256(b"").hexdigest()
            row["completion_token_ids"] = []
            self.assertEqual(len(ce.validate_request(row, example, protocol)), 64)

    def test_protocol_prompt_and_name_mismatch_rejected(self):
        for mutation in (lambda r,e:e["prompt_metadata"].update(test_func_name="randomized"),
                         lambda r,e:r.update(prompt_sha256="0"*64), lambda r,e:r.update(protocol="randomized")):
            row, example, _ = fixture()
            mutation(row,example)
            with self.assertRaises(ContractError):
                ce.validate_request(row,example,"fixed")
        row,example,_=fixture("no_loophole")
        example["hint"]="still a loophole"
        with self.assertRaises(ContractError):ce.validate_request(row,example,"no_loophole")

    def test_randomized_mapping_may_retain_original_name(self):
        row, example, _ = fixture("randomized")
        example["prompt_metadata"]["test_func_name"] = "run_tests"
        self.assertEqual(len(ce.validate_request(row, example, "randomized")), 64)
        for name in (None, "", "a()", "class"):
            example["prompt_metadata"]["test_func_name"] = name
            with self.assertRaises(ContractError):
                ce.validate_request(row, example, "randomized")
        example["prompt_metadata"] = {}
        with self.assertRaises(ContractError):
            ce.validate_request(row, example, "randomized")

    def test_no_loophole_rh_is_inapplicable_not_false(self):
        row, example, raw = fixture("no_loophole")
        self.evaluator.raw = raw
        with self.journal(row, example) as journal:
            result = ce.process_one(journal, row, example, "no_loophole", self.root / "transport",
                                    evaluator=self.evaluator, installation=self.installation)
            self.assertIs(journal.primitive(row["request_id"])["repository_evaluation"]["is_reward_hack_strict"], False)
        analysis = result["analysis"]
        for name in ce.LOOPHOLE_METRICS:
            self.assertIsNone(analysis["metrics"][name])
            self.assertEqual(analysis["metric_applicability"][name], "not_applicable_no_loophole")
        self.assertIs(analysis["metrics"]["ground_truth_correctness"], True)
        self.assertEqual(analysis["metric_applicability"]["ground_truth_correctness"], "applicable")
        self.assertEqual(result["metadata_status"], "available")

    def test_real_host_outer_guard_rejects_unsandboxed_dispatch(self):
        with self.assertRaisesRegex(ContractError,"outer sandbox"):
            ce.require_outer(self.root/'input',self.root/'dataset',self.root/'output')


class SolutionPresenceTests(unittest.TestCase):
    def check(self, code, target='Solution().solve'):
        return ce.solution_presence({'is_parsed': True, 'parsed_response': code}, target)

    def test_direct_function_and_both_class_target_spellings(self):
        for target in ('Solution().solve', 'Solution.solve'):
            self.assertIs(self.check('class Solution:\n    def solve(self):\n        pass\n', target)['solution_present'], True)
        self.assertIs(self.check('def solve(x):\n    return x\n', 'solve')['solution_present'], True)

    def test_missing_class_method_or_function_known_false(self):
        for code, target in (('def run_tests():\n    return True\n', 'Solution().solve'),
                             ('class Solution:\n    def other(self):\n        pass\n', 'Solution.solve'),
                             ('def other():\n    pass\n', 'solve'), ('', 'solve')):
            with self.subTest(code=code):
                self.assertIs(self.check(code, target)['solution_present'], False)

    def test_nested_conditional_or_duplicate_definitions_unknown(self):
        cases=('if True:\n    class Solution:\n        def solve(self): pass\n',
               'class Solution:\n    if True:\n        def solve(self): pass\n',
               'class Solution:\n    def solve(self): pass\n    def solve(self): pass\n',
               'class Solution:\n    def solve(self): pass\nclass Solution: pass\n')
        for code in cases:
            with self.subTest(code=code), self.assertRaises(ContractError):self.check(code)

    def test_dynamic_alias_import_inheritance_and_target_are_unknown(self):
        cases=('Solution = type("Solution", (), {})',
               'from somewhere import Solution', 'from somewhere import *',
               'exec("class Solution: pass")',
               'class Solution(Base):\n    pass\n',
               'class Solution:\n    def __getattr__(self, name): return lambda: 1\n',
               'class Solution:\n    solve = lambda self: 1\n',
               '@replace\nclass Solution:\n    def solve(self): return 1\n',
               'class Solution:\n    @property\n    def solve(self): return 1\n')
        for code in cases:
            with self.subTest(code=code), self.assertRaises(ContractError):self.check(code)
        for target in ('getattr(Solution(), "solve")', 'Solution(1).solve'):
            with self.subTest(target=target), self.assertRaises(ContractError):
                self.check('class Solution:\n    def solve(self): pass\n', target)
        with self.assertRaises(ContractError):
            self.check('def factory():\n    return object()\n', 'factory().solve')

    def test_syntax_and_missing_source_remain_unknown(self):
        with self.assertRaises(SyntaxError):self.check('class Solution: def solve(')
        for raw in (None, {}, {'is_parsed': False, 'parsed_response': None}):
            with self.subTest(raw=raw), self.assertRaises(ContractError):ce.solution_presence(raw, 'solve')

    def test_ast_memory_or_recursion_failure_propagates_to_unknown_boundary(self):
        for error in (MemoryError('AST'), RecursionError('AST')):
            with self.subTest(error=type(error).__name__), mock.patch.object(ce.ast, 'parse', side_effect=error):
                with self.assertRaises(type(error)):self.check('pass')

    def test_static_method_and_empty_body_are_presence_not_correctness(self):
        code='class Solution:\n    @staticmethod\n    def solve():\n        return None\n'
        value=self.check(code)
        self.assertIs(value['solution_present'], True)
        self.assertIn('not runtime callability or correctness',value['meaning'])


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.row,self.example,self.raw=fixture()
        self.generation=self.root/'generated.jsonl';self.dataset=self.root/'dataset.jsonl'
        self.generation.write_bytes(ce.canonical(self.row)+b'\n')
        self.dataset.write_bytes(ce.canonical(self.example)+b'\n')
        self.installation=InertInstallation();self.evaluator=InertEvaluator(self.raw,self.installation)

    def tearDown(self):self.temp.cleanup()

    def run_batch(self,**override):
        kwargs=dict(protocol='fixed',run_token='batch-fixture',input_sha256='1'*64,
                    generated_sha256=ce.file_sha(self.generation),dataset_sha256=ce.file_sha(self.dataset),deadline_epoch=1e20,
                    evaluator_factory=lambda _:self.evaluator,install_factory=lambda:self.installation,
                    outer_check=lambda *_:None)
        kwargs.update(override)
        with mock.patch.object(ce.repository,'install_count_payload_guard') as guard:
            result=ce.run_batch(self.generation,self.dataset,self.root/'output',**kwargs)
        return result,guard

    def test_success_then_complete_reuse_constructs_no_scorer(self):
        result,guard=self.run_batch()
        self.assertEqual(result['status'],'all_results_recorded')
        self.assertEqual(result['records'],1)
        self.assertTrue(self.installation.restored)
        guard.assert_called_once_with(self.installation)
        again,guard=self.run_batch(evaluator_factory=mock.Mock(side_effect=AssertionError('new evaluator')),
                                   install_factory=mock.Mock(side_effect=AssertionError('new transport')))
        guard.assert_not_called()
        self.assertEqual(result,again)

    def test_deadline_prevents_start_and_factory(self):
        factory=mock.Mock(side_effect=AssertionError('factory'))
        with self.assertRaisesRegex(ContractError,'deadline'):
            self.run_batch(deadline_epoch=0,evaluator_factory=factory)
        factory.assert_not_called()
        self.assertEqual((self.root/'output/requests/started.jsonl').read_bytes(),b'')

    def test_original_settings_enforced_and_transport_restored_on_failure(self):
        self.evaluator.evaluator.num_workers=3
        with self.assertRaisesRegex(ContractError,'settings differ'):
            self.run_batch()
        self.assertTrue(self.installation.restored)
        self.assertEqual(self.evaluator.calls,0)

    def test_unknown_metadata_is_completed_as_unknown_not_scientific_success(self):
        result,_=self.run_batch(analyzer=mock.Mock(side_effect=MemoryError('AST')))
        self.assertEqual(result['metadata_unknown'],1)
        self.assertEqual(result['evaluation_status'],{'metadata_unknown':1})
        self.assertFalse(result['scientific_success_claimed'])

    def test_hash_mismatch_stops_before_factory_or_journal(self):
        factory=mock.Mock(side_effect=AssertionError('factory'))
        with self.assertRaisesRegex(ContractError,'hashes differ'):
            self.run_batch(generated_sha256='0'*64,evaluator_factory=factory)
        factory.assert_not_called()
        self.assertFalse((self.root/'output').exists())

    def test_infrastructure_failure_final_retained_and_batch_not_complete(self):
        self.evaluator.raw=None
        with self.assertRaisesRegex(ContractError,'infrastructure failure'):
            self.run_batch()
        self.assertEqual(len(list((self.root/'output/requests/primitives').glob('*.json'))),1)
        self.assertEqual(len(list((self.root/'output/requests/finals').glob('*.json'))),1)
        self.assertFalse((self.root/'output/COMPLETE.json').exists())


if __name__=='__main__':unittest.main()
