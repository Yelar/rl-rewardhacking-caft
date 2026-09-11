"""Integration guards; the separate plan tests exercise full scientific validation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_phase
import phase_budget
from infra.gpu03.direction_discovery import no_loophole_plan as protocol


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'input').mkdir()
        self.prepared = self.root / 'prepared.jsonl'
        self.prepared.write_text('{}\n')
        self.plan = {
            'no_loophole_capability': {'environment': 'canonical_no_hint'},
            'master_plan_sha256': 'master', 'mode': 'generate',
            'authorization': 'new capability preparation; fresh launch approval required',
            'prepared_records': str(self.prepared),
            'prepared_records_sha256': build_phase.supervisor.sha256(self.prepared),
            'conditions': {'baseline': {'layers': []}}, 'sampling': {'seed': 1},
            'requests': [{'request_id': 'fresh-1'}, {'request_id': 'fresh-2'}],
        }
        self.task_path = self.root / 'task.json'
        self.task = {'mode': 'generate', 'conditions': self.plan['conditions'],
                     'sampling': self.plan['sampling'], 'prepared_records': str(self.prepared),
                     'attention_policy': 'exclusive_math',
                     'teacher_forced_padded_sequence_length': 2176,
                     'requests': self.plan['requests']}
        self.ledger = {'generation_requests': 10, 'tf_requests': 20,
                       'gpu_phase_wall_seconds': 0, 'phases': []}
        self.manifest = {
            'stage': str(self.root), 'source_root': str(self.root / 'source'),
            'phase': 'no_loophole_capability', 'gpu_ids': [1],
            'authorization': self.plan['authorization'],
            'scientific': {'master_plan_sha256': 'master',
                           'previous_phase_budget': self.ledger,
                           'no_loophole_capability': self.plan['no_loophole_capability']},
            'workers': [{'command': ['python', 'engine.py', '--task', str(self.task_path)],
                         'success_expect': {'mode': 'generate', 'requests': 2}}],
            'limits': {'systemd_runtime_seconds': 1000},
        }
        self.master = {'inputs': {'prepared_records_sha256': 'historical-core'},
                       'sampling': self.plan['sampling'], 'budget': {
            'maximum_teacher_forced_forwards': 12000,
            'maximum_new_free_generations_including_qualification': 4096,
            'maximum_aggregate_gpu_phase_wall_seconds': 28800}}
        for name, value in [('validate_full', {}), ('bindings', []),
                            ('validate_source', None), ('validate_against_ledger', None)]:
            mock = self.enterContext(patch.object(protocol, name, return_value=value))
            setattr(self, name, mock)
        self.enterContext(patch.object(build_phase, 'validate_master', return_value=self.master))
        self.enterContext(patch.object(phase_budget, 'account', return_value=self.ledger))
        self.persist()

    def persist(self):
        (self.root / 'input/request_plan.json').write_text(json.dumps(self.plan))
        self.task_path.write_text(json.dumps(self.task))

    def test_derived_input_requires_full_validation_and_exact_path(self):
        build_phase.validate_request_inputs(self.plan, 'master', self.prepared, 'historical-core')
        self.validate_full.assert_called_once_with(self.plan)
        alias = self.root / 'alias.jsonl'
        alias.write_bytes(self.prepared.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'exact derived prepared'):
            build_phase.validate_request_inputs(self.plan, 'master', alias, 'historical-core')

    def test_derived_input_cannot_bypass_the_plan_validator(self):
        self.validate_full.side_effect = RuntimeError('altered scientific coordinates')
        with self.assertRaisesRegex(RuntimeError, 'altered scientific'):
            build_phase.validate_request_inputs(self.plan, 'master', self.prepared, 'historical-core')

    def test_unmarked_new_input_does_not_expand_old_allowlist(self):
        other = dict(self.plan)
        other.pop('no_loophole_capability')
        with self.assertRaisesRegex(RuntimeError, 'Unverified causal'):
            build_phase.validate_request_inputs(other, 'master', self.prepared, 'historical-core')

    def test_successful_publication_checks_source_and_current_budget(self):
        build_phase.validate_publication_budget(self.manifest)
        self.validate_against_ledger.assert_called_once_with(self.plan, self.ledger)
        self.validate_source.assert_called_once_with(self.plan, self.root / 'source')
        self.bindings.assert_called_once_with(self.plan, verify=True)

    def test_stale_budget_fails_before_publication(self):
        self.validate_against_ledger.side_effect = RuntimeError('stale capability budget')
        with self.assertRaisesRegex(RuntimeError, 'stale capability'):
            build_phase.validate_publication_budget(self.manifest)

    def test_old_authorization_cannot_be_presented_as_this_request(self):
        self.manifest['authorization'] = 'historical approval'
        with self.assertRaisesRegex(RuntimeError, 'capability metadata'):
            build_phase.validate_publication_budget(self.manifest)

    def test_missing_or_different_descriptor_and_missing_plan_fail(self):
        for descriptor in (None, {'environment': 'wrong'}):
            with self.subTest(descriptor=descriptor):
                m = copy.deepcopy(self.manifest)
                if descriptor is None:
                    m['scientific'].pop('no_loophole_capability')
                else:
                    m['scientific']['no_loophole_capability'] = descriptor
                with self.assertRaisesRegex(RuntimeError, 'capability metadata'):
                    build_phase.validate_publication_budget(m)
        (self.root / 'input/request_plan.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'lost its no-loophole'):
            build_phase.validate_publication_budget(self.manifest)

    def test_matched_inference_protocol_cannot_change(self):
        original = copy.deepcopy(self.task)
        for key, value in [('mode', 'tf'), ('sampling', {'seed': 2}),
                           ('conditions', {'new': {}}), ('prepared_records', '/wrong'),
                           ('attention_policy', 'auto'), ('teacher_forced_padded_sequence_length', 100)]:
            with self.subTest(key=key):
                self.task = {**original, key: value}
                self.persist()
                with self.assertRaisesRegex(RuntimeError, 'matched inference protocol'):
                    build_phase.validate_publication_budget(self.manifest)

    def test_worker_coverage_and_request_metadata_cannot_change(self):
        for requests in ([{'request_id': 'fresh-1'}],
                         [{'request_id': 'fresh-1'}, {'request_id': 'fresh-1'}],
                         [{'request_id': 'fresh-1', 'seed': 99}, {'request_id': 'fresh-2'}]):
            with self.subTest(requests=requests):
                self.task['requests'] = requests
                self.persist()
                with self.assertRaisesRegex(RuntimeError, 'exact 740 requests'):
                    build_phase.validate_publication_budget(self.manifest)


class LedgerIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_phase_budget import BudgetTests, V2
        self.base = BudgetTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.master_sha = V2
        self.stage = self.base.phase('codex-discovery-nohint', V2, 'generate', 2)
        self.manifest_path = self.stage / 'reviewed_manifest.json'
        self.plan_path = self.stage / 'input/request_plan.json'
        self.task_path = self.stage / 'tasks/gpu_0.json'
        self.m = json.loads(self.manifest_path.read_text())
        self.plan = json.loads(self.plan_path.read_text())
        self.task = json.loads(self.task_path.read_text())
        self.meta = {'environment': 'canonical_no_hint'}
        self.plan.update(no_loophole_capability=self.meta, master_plan_sha256=V2,
                         authorization='new request; fresh approval required',
                         conditions={'baseline': {'layers': []}}, prepared_records='/derived', sampling={'seed': 1})
        self.task.update(conditions=self.plan['conditions'], prepared_records='/derived', sampling={'seed': 1},
                         attention_policy='exclusive_math', teacher_forced_padded_sequence_length=2176)
        self.m['phase'] = 'no_loophole_capability'
        self.m['authorization'] = self.plan['authorization']
        self.m['scientific']['no_loophole_capability'] = self.meta
        self.validation = self.enterContext(patch.object(protocol, 'validate_full', return_value={}))
        self.enterContext(patch.object(protocol, 'bindings', return_value=[]))
        self.persist()

    def persist(self):
        self.plan_path.write_text(json.dumps(self.plan))
        self.task_path.write_text(json.dumps(self.task))
        for path in (self.plan_path, self.task_path):
            self.m['bound_files'][str(path)] = {'sha256': phase_budget.sha(path), 'size_bytes': path.stat().st_size}
        self.manifest_path.write_text(json.dumps(self.m))

    def test_capability_counts_share_ledger_but_preserve_environment(self):
        result = phase_budget.account(self.base.root, self.master_sha)
        self.assertEqual(result['generation_requests'], 2)
        self.assertEqual(result['tf_requests'], 0)
        self.assertEqual(result['untouched_test_requests'], 0)
        self.assertEqual(result['phases'][0]['no_loophole_capability'], self.meta)
        self.validation.assert_called_once_with(self.plan)

    def test_changed_descriptor_or_sampling_fails_closed(self):
        self.m['scientific']['no_loophole_capability'] = {'wrong': True}
        self.persist()
        with self.assertRaisesRegex(RuntimeError, 'capability metadata or inference'):
            phase_budget.account(self.base.root, self.master_sha)
        self.m['scientific']['no_loophole_capability'] = self.meta
        self.task['sampling'] = {'seed': 99}
        self.persist()
        with self.assertRaisesRegex(RuntimeError, 'capability metadata or inference'):
            phase_budget.account(self.base.root, self.master_sha)

    def test_lost_descriptor_cannot_fall_back_to_old_plan_route(self):
        self.plan.pop('no_loophole_capability')
        self.persist()
        with self.assertRaisesRegex(RuntimeError, 'lost its no-loophole'):
            phase_budget.account(self.base.root, self.master_sha)

    def test_scientific_validation_failure_propagates(self):
        self.validation.side_effect = RuntimeError('not the original 37 problems')
        with self.assertRaisesRegex(RuntimeError, 'original 37'):
            phase_budget.account(self.base.root, self.master_sha)


class QualificationMetadataReadTests(unittest.TestCase):
    def test_existing_qualification_does_not_deserialize_other_problem_rows(self):
        from test_build_phase import CausalQualificationTests
        fixture = CausalQualificationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.prepared.read_text()
        untouched = {'record_index': 400, 'record_id': 'not-selected', 'do_not_deserialize': True,
                     'completion': 'A quoted string: "record_index":222, must not match metadata.'}
        fixture.prepared.write_text(original + json.dumps(untouched) + '\n')
        fixture.prepared_sha = build_phase.supervisor.sha256(fixture.prepared)
        fixture.manifest_data['scientific']['input_prepared_sha256'] = fixture.prepared_sha
        loads = json.loads

        def guarded_loads(value, *args, **kwargs):
            result = loads(value, *args, **kwargs)
            if isinstance(result, dict) and result.get('do_not_deserialize'):
                self.fail('Unselected problem row was deserialized')
            return result

        with patch.object(build_phase.json, 'loads', side_effect=guarded_loads):
            self.assertIn(fixture.prepared, fixture.check())


if __name__ == '__main__':
    unittest.main()
