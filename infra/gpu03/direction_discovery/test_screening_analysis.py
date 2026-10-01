"""Authored screening fixtures; no model, GPU, or generated-code execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import screening_analysis as s
from . import metrics as m
from .test_metrics import plan as metrics_plan, row as metric_row


def fixture_conditions():
    result = {'baseline': {'role': 'baseline', 'layers': []}}
    for target, layer in zip(s.TARGETS, (16, 17, 21)):
        randoms = [f'random:L{layer}r1:base{seed}' for seed in (6101, 6102, 6103)]
        result[target] = {'role': 'target', 'layers': [{'layer': layer, 'kind': 'candidate', 'selectors': [{'key': 'pcs', 'column': 0}]}],
                          'random_controls': randoms}
        for seed, name in zip((6101, 6102, 6103), randoms):
            result[name] = {'role': 'random', 'layers': [{'layer': layer, 'kind': 'random', 'rank': 1, 'seed': seed + layer}]}
    return result


def fixture_rows(conditions):
    rows = []
    for p in range(12):
        for scope in ('primary', 'local'):
            for sample in range(2 if scope == 'primary' else 3):
                for cid, condition in conditions.items():
                    value = metric_row(str(p), cid, sample if scope == 'primary' else 0,
                        harmful=cid not in s.TARGETS, correctness=True, scope=scope,
                        record=f'prefix-{p}-{sample}' if scope == 'local' else f'source-{p}')
                    layer = condition['layers'][0]['layer'] if condition['layers'] else None
                    value['generation']['energy'] = {} if layer is None else {str(layer): {
                        'rank': 1, 'selected_tokens': 10, 'forward_calls': 10, 'removed_energy_fp32': 1., 'activation_energy': 100.}}
                    value['repository_evaluation'] = {'parsed_response': 'def run_tests():\n    assert True\n', 'test_func_name': 'run_tests'}
                    rows.append(value)
    return rows


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.eval_stage = self.root / 'evaluation'; (self.eval_stage / 'input').mkdir(parents=True)
        (self.eval_stage / 'results').mkdir()
        self.conditions = fixture_conditions(); self.rows = fixture_rows(self.conditions)
        self.master = metrics_plan(); self.master['plan_version'] = 2
        self.parent = self.root / 'parent.json'; s.write_json(self.parent, {'authored': True})
        self.master_path = self.root / 'master.json'; s.write_json(self.master_path, self.master)
        self.generation = self.root / 'generation.json'; s.write_json(self.generation, {'authored_manifest': True})
        self.master_sha = m.sha256(self.master_path); self.parent_sha = m.sha256(self.parent)
        self.request = {'mode': 'generate', 'phase': 'screening', 'evaluation_partition': s.PARTITION,
            'master_plan_sha256': self.master_sha, 'conditions': self.conditions, 'selected_target_ids': list(s.TARGETS),
            'requests': [{k: row[k] for k in s.IDENTITY} for row in self.rows], 'scope_counts': {'primary': 312, 'local': 468},
            'sampling': {'temperature': .7}, 'previously_committed_generation_requests': 6,
            'generation_requests_after_commit': 786, 'source_bindings': {'authored': 'first'}}
        self.reference = self.root / 'reference.json'; s.write_json(self.reference, self.request)
        self.request_path = self.root / 'request.json'; s.write_json(self.request_path, self.request)
        self.eval_manifest = self.root / 'evaluation_manifest.json'
        self.manifest = {'stage': str(self.eval_stage), 'output': str(self.eval_stage / 'results'), 'mode': 'production',
            'scientific': {'master_plan_sha256': self.master_sha, 'parent_plan_sha256': self.parent_sha}}
        s.write_json(self.eval_manifest, self.manifest)
        self.evaluation = self.eval_stage / 'results/evaluations.jsonl'
        self.write_rows()
        self.spec = {'schema_version': 1, 'purpose': 'completed_screening_analysis',
            'output': str(self.root / 'codex-screening-analysis-authored')}
        for key, p in [('generation_manifest', self.generation), ('request_plan', self.request_path),
                       ('reference_request_plan', self.reference), ('master', self.master_path),
                       ('parent', self.parent), ('evaluation_manifest', self.eval_manifest)]:
            self.spec[key] = {'path': str(p), 'sha256': m.sha256(p)}
        self.publish_eval_inputs()
        self.patches = [patch.object(s, 'MASTER_SHA', self.master_sha), patch.object(s, 'PARENT_SHA', self.parent_sha),
            patch.object(s, 'REFERENCE_REQUEST_PLAN_SHA', m.sha256(self.reference)), patch.object(s, 'ROOT', self.root),
            patch.object(s, 'verify_evaluation', side_effect=lambda *_: (self.manifest, self.proof())),
            patch('infra.gpu03.direction_discovery.behavior_plan.validate_master'),
            patch.object(m, 'verify_evaluation_lineage', side_effect=self.lineage)]
        for item in self.patches: item.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])

    def write_rows(self):
        self.evaluation.write_text(''.join(m.canonical(row) + '\n' for row in self.rows))

    def proof(self):
        return {'exact_request_coverage': True, 'process_release_verified': True,
                'evaluations_sha256': m.sha256(self.evaluation)}

    def lineage(self, evaluation, conditions, master_sha, manifest, digest):
        self.assertEqual(evaluation, self.evaluation); self.assertEqual(conditions, self.conditions)
        self.assertEqual(master_sha, self.master_sha); self.assertEqual(Path(manifest), self.eval_manifest)
        return {'manifest_path': str(manifest), 'manifest_sha256': digest, **self.proof()}

    def publish_eval_inputs(self):
        s.write_json(self.eval_stage / 'input/request_plan.json', self.request)
        s.write_json(self.eval_stage / 'input/generation_verification.json', [{
            'manifest': str(self.generation), 'manifest_sha256': m.sha256(self.generation),
            'request_ids': sorted(row['request_id'] for row in self.rows)}])

    def update_plan(self):
        s.write_json(self.request_path, self.request)
        self.spec['request_plan']['sha256'] = m.sha256(self.request_path)
        self.publish_eval_inputs()

    def run_package(self):
        p = self.root / 'spec.json'; s.write_json(p, self.spec)
        return s.run(p)

    def test_full_780_row_producer_independent_numerical_ast_verifier_and_summary_tamper(self):
        result = self.run_package(); output = Path(result['output'])
        proof = s.verify(output, result['artifact_manifest_sha256'])
        self.assertTrue(proof['statistics_recomputed']); self.assertFalse(proof['decision_created'])
        summary = s.read_json(output / 'summary.json')
        self.assertEqual(summary['targets_meeting_primary_gates'], list(s.TARGETS))
        self.assertIsNone(summary['selected_target'])
        text = (output / 'REPORT.md').read_text()
        self.assertIn('matched random mean for target:L16.transition.pc00', text)
        self.assertIn('Local scope cannot promote', text)
        self.assertIn('literal_or_identical_expression_assertions', text)
        with self.assertRaisesRegex(ValueError, 'fresh dedicated'):
            self.run_package()
        p = output / 'summary.json'; p.chmod(0o600)
        summary['selected_target'] = s.TARGETS[0]; s.write_json(p, summary)
        (output / 'artifact_manifest.json').chmod(0o600); s.manifest_package(output)
        with self.assertRaisesRegex(ValueError, 'Summary/table recomputation'):
            s.verify(output, m.sha256(output / 'artifact_manifest.json'))

    def test_retry_may_change_only_conservative_ledger_and_provenance(self):
        self.request.update(previously_committed_generation_requests=786, generation_requests_after_commit=1566,
                            source_bindings={'authored': 'retry'})
        self.update_plan()
        self.assertEqual(len(s.context(self.spec)['rows']), 780)
        self.request['sampling']['temperature'] = .8; self.update_plan()
        with self.assertRaisesRegex(ValueError, 'original screening science'):
            s.context(self.spec)

    def test_incomplete_proof_fails_before_reading_outcomes_or_creating_output(self):
        self.evaluation.unlink()
        with patch.object(s, 'verify_evaluation', side_effect=ValueError('producer not complete')):
            with self.assertRaisesRegex(ValueError, 'producer not complete'):
                self.run_package()
        self.assertFalse(Path(self.spec['output']).exists())

    def test_whole_problem_or_duplicate_row_cannot_hide_behind_mocked_coverage(self):
        original = list(self.rows)
        for changed in ([r for r in original if r['problem_id'] != '0'], original[:-1] + [original[0]]):
            self.rows = changed; self.write_rows()
            with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'):
                s.context(self.spec)

    def test_evaluation_identity_and_frozen_request_ids_are_not_reselected(self):
        self.rows[0]['seed'] += 1; self.write_rows()
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            s.context(self.spec)
        self.request['requests'][0]['seed'] += 1; self.update_plan()
        with self.assertRaisesRegex(ValueError, 'original screening science'):
            s.context(self.spec)

    def test_unknown_and_local_gate_are_never_rendered_as_benign_or_final_decision(self):
        row = next(r for r in self.rows if r['condition_id'] == s.TARGETS[0] and r['scope'] == 'primary')
        row['metrics']['harmful_modification'] = None; row['evaluation_status'] = 'suspicious_or_unknown'
        self.write_rows(); ctx = s.context(self.spec)
        report = s.attach_provenance(m.analyze(ctx['rows'], self.conditions, self.master, bootstrap_resamples=100), ctx, self.spec)
        _, mechanisms = s.mechanism.audit(self.rows)
        summary = s.summarize(report, mechanisms, ctx)
        self.assertGreater(summary['unknown_primary_metric_components'], 0)
        text = s.render(summary, 'baseline')
        self.assertIn('unknown [', text); self.assertIn('not a final direction', text)
        self.assertIsNone(summary['selected_target'])
        self.assertFalse(report['by_split'][s.PARTITION]['local']['targets'][s.TARGETS[0]]['promotion']['eligible_for_validation_promotion'])

    def test_modified_qualified_source_or_historical_generation_proof_fails(self):
        with patch.object(s, 'METRICS_SHA', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'Qualified metrics'):
                s.context(self.spec)
        p = self.eval_stage / 'input/generation_verification.json'; value = s.read_json(p)
        value[0]['manifest_sha256'] = '0' * 64; s.write_json(p, value)
        with self.assertRaisesRegex(ValueError, 'different generation package'):
            s.context(self.spec)

    def test_snapshot_replacement_at_read_after_successful_verification_is_rejected(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['metrics']['harmful_modification'] = False
        replacement = ''.join(m.canonical(row) + '\n' for row in changed).encode()
        read_bytes = Path.read_bytes
        def replacing_read(path):
            if path == self.evaluation:
                path.write_bytes(replacement)
            return read_bytes(path)
        with patch.object(Path, 'read_bytes', replacing_read):
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                s.context(self.spec)

    def test_wrong_output_and_mutated_bound_inputs_fail(self):
        self.spec['output'] = str(self.eval_stage / 'results/analysis')
        with self.assertRaisesRegex(ValueError, 'dedicated'):
            self.run_package()
        self.spec['output'] = str(self.root / 'codex-screening-analysis-authored')
        self.request_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed input'):
            self.run_package()


if __name__ == '__main__':
    unittest.main()
