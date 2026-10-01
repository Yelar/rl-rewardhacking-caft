"""Authored test-bundle metadata only; external verifiers are interface fixtures."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import bundle_harness_interpretability as adapter
from . import bundle_analysis as gate
from . import harness_interpretability as inspection
from .test_bundle_analysis import Fixture
from .test_harness_interpretability import row, CODE


class Tests(unittest.TestCase):
    def fixture(self, **kwargs):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        f = Fixture(Path(temporary.name).resolve(), **kwargs)
        for value in f.rows:
            correct = value['condition_id'] == 'target'
            code = CODE + ("if __name__ == '__main__':\n    Solution().run_tests()\n" if correct else 'Solution().run_tests()\n')
            value['repository_evaluation'] = row(code=code, correct=correct)['repository_evaluation']
            value['metrics']['ground_truth_correctness'] = correct
        f.manifest['source_files'].update({name: {'sha256': digest} for name, digest in inspection.HARNESS_SOURCES.items()})
        f.rebind_manifest()
        spec = {'schema_version': 1, 'purpose': adapter.PURPOSE, 'runnable': True, 'source_sha256': gate.sha(adapter.__file__),
                'inspection_source_sha256': adapter.INSPECTION_SHA, 'gate_source_sha256': adapter.GATE_SHA,
                'complete_evaluation_context': f.spec, 'harness_sources': {name: {'path': str(Path(__file__).resolve().parents[3]/name), 'sha256': digest}
                 for name, digest in inspection.HARNESS_SOURCES.items()}}
        return f, spec

    def context(self, f, spec):
        with ExitStack() as stack:
            stack.enter_context(patch.object(gate, 'MASTER_SHA', f.master['sha256']))
            stack.enter_context(patch.object(gate, 'PARENT_SHA', f.parent['sha256']))
            return adapter.context(spec, loaded=f.loaded)

    def no_outcome_reads(self):
        original = gate.snapshot
        def check(path, *args, **kwargs):
            if Path(path).name == 'evaluations.jsonl': raise AssertionError('Outcomes read before complete authority')
            return original(path, *args, **kwargs)
        return patch.object(gate, 'snapshot', side_effect=check)

    def test_aggregation_parity_with_unchanged_validation_inspection(self):
        values = [row('fail', code=CODE+'Solution().run_tests()\n'), row('pass', correct=True, condition='target'), row('unknown', condition='random')]
        values[2]['metrics']['ground_truth_correctness'] = None; values[2]['evaluation_status'] = 'suspicious_or_unknown'
        before = copy.deepcopy(values)
        records, old = inspection.audit(values)
        summary = adapter.summarize(records)
        self.assertEqual(summary, {key: old[key] for key in summary})
        self.assertEqual(values, before)
        self.assertIsNone(records[2]['recorded_gt_correctness'])

    def test_full_authored_producer_recompute_preserves_components_and_raw_bytes(self):
        f, spec = self.fixture(); before = copy.deepcopy(f.rows); ctx = self.context(f, spec)
        with patch.object(inspection, 'audit', side_effect=AssertionError('Do not weaken/relabel validation guard')):
            producer = adapter.analyze_into(f.root/'static-output', ctx)
            verified = adapter.verify_recompute(f.root/'static-output', f.root/'static-recompute', self.context(f, spec))
        self.assertEqual(producer['records'], 20); self.assertTrue(verified['exact_recomputation_match'])
        for name, ids in ctx['component_ids'].items():
            folder = f.root/'static-output/components'/name
            self.assertEqual((folder/'evaluations.jsonl').read_bytes(), b''.join(line for row, line in zip(ctx['rows'], ctx['lines']) if row['request_id'] in ids))
            report = gate.parse((folder/'summary.json').read_bytes())
            self.assertEqual(report['records'], 10); self.assertFalse(report['statistics_computed'])
            self.assertEqual(len(report['local_same_solution_gt_disagreement_groups']), 2)
        self.assertEqual(f.rows, before)
        self.assertEqual(gate.inventory(f.root/'static-output'), gate.inventory(f.root/'static-recompute'))

    def test_empty_auxiliary_is_explicit_and_does_not_skip_full_core(self):
        f, spec = self.fixture(empty_aux=True); ctx = self.context(f, spec)
        adapter.analyze_into(f.root/'output-static', ctx)
        summary = gate.parse((f.root/'output-static/summary.json').read_bytes())
        self.assertEqual(summary['records'], 10)
        self.assertEqual(summary['components']['auxiliary_test']['status'], 'unavailable_no_eligible_pairs')

    def test_pending_and_changed_sources_stop_before_any_metadata_input(self):
        for key, value in [('runnable', False), ('inspection_source_sha256', '0'*64), ('gate_source_sha256', '0'*64), ('source_sha256', '0'*64)]:
            f, spec = self.fixture(); spec[key] = value
            with self.subTest(key=key), patch.object(gate, 'read_bound', side_effect=AssertionError('Metadata opened before static source gate')):
                with self.assertRaises(ValueError): self.context(f, spec)

    def test_positive_final_gate_precedes_outcomes_and_inspection(self):
        f, spec = self.fixture()
        final = gate.parse((f.bundle_root/'input/final_config.json').read_bytes()); final['status'] = 'pending'
        f.bundle['frozen_final_config'] = f.put_relative('input/final_config.json', final); f.rebind_bundle(); f.rebind_manifest()
        with self.no_outcome_reads(), patch.object(inspection, 'inspect', side_effect=AssertionError('Inspection called before positive freeze')):
            with self.assertRaisesRegex(ValueError, 'positive final'): self.context(f, spec)
        self.assertEqual(f.verifier_calls, 0)

    def test_incomplete_or_unreleased_full_evaluation_never_loads_outcomes(self):
        for key, value in [('records', 19), ('process_release_verified', False), ('exact_request_coverage', False)]:
            f, spec = self.fixture(); proof = {**f.proof, key: value}
            f.spec['evaluation_verification'] = f.put(f.stage/'control/independent_verification.json', proof)
            with self.subTest(key=key), self.no_outcome_reads(), self.assertRaisesRegex(ValueError, 'Incomplete independent'):
                self.context(f, spec)

    def test_wrong_bundle_and_validation_manifest_cannot_enter_test_adapter(self):
        for mutation in ('bundle', 'phase', 'harness'):
            f, spec = self.fixture()
            if mutation == 'bundle': f.manifest['scientific']['test_bundle']['sha256'] = '0'*64
            elif mutation == 'phase': f.manifest['phase'] = 'behavior_finalist_validation'
            else: f.manifest['source_files']['src/evaluate/helpers.py']['sha256'] = '0'*64
            f.rebind_manifest()
            with self.subTest(mutation=mutation), self.no_outcome_reads(), self.assertRaises(ValueError): self.context(f, spec)

    def test_partial_component_and_changed_snapshot_cannot_pass_complete_gate(self):
        for mutation in ('component', 'snapshot'):
            f, spec = self.fixture()
            if mutation == 'component':
                f.bundle['components']['auxiliary_test']['request_ids'].pop(); f.rebind_bundle(); f.rebind_manifest()
            else: (f.result/'evaluations.jsonl').write_bytes((f.result/'evaluations.jsonl').read_bytes()+b'{}\n')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.context(f, spec)

    def test_generic_validation_context_and_altered_producer_rejected(self):
        f, spec = self.fixture()
        with self.assertRaisesRegex(ValueError, 'positively gated'): adapter.analyze_into(f.root/'bad', f.context())
        ctx = self.context(f, spec); adapter.analyze_into(f.root/'output-static', ctx)
        path = f.root/'output-static/components/core_untouched_test/summary.json'; path.chmod(0o600); path.write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'Producer artifact'): adapter.verify_recompute(f.root/'output-static', f.root/'fresh', self.context(f, spec))


if __name__ == '__main__': unittest.main()
