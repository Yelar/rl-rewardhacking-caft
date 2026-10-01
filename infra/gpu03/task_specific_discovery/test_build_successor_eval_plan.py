"""Inert full-source admission and fresh successor wave fixtures; no live work."""
import copy
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import build_successor_eval_plan as w
from . import bind_successor_eval as binding
from . import test_bind_successor_wave as wave_fixtures
from . import test_build_causal_eval_plan as authored_fixtures
from .io_contract import ContractError

write, rewrite = authored_fixtures.write, authored_fixtures.rewrite


class FullFixture:
    """All15168 real inert journal primitives/finals, with actual wave binders."""
    def __init__(self):
        self.owner = unittest.TestCase()
        self.f = wave_fixtures.SuccessorFixture(self.owner)
        self.root = self.f.root
        self.manifest_refs = []
        # Fixture fsync is disabled only during creation; no crash-durability claim.
        with patch('os.fsync'):
            for index in range(8):
                ref = self.f.bind(index, now=100000. + index * 1000)
                self.f.close(ref); self.manifest_refs.append(ref)
        self.source = self.root / 'source-successor-eval-v1'; self.source.mkdir()
        required = {binding.SELF_SOURCE, w.SELF_SOURCE, w.WAVE_AUDITOR_SOURCE, w.WAVE_PREPARER_SOURCE,
            w.AUDIT_IO_SOURCE, w.GUARD_SOURCE, w.original.LANE_SOURCE, w.original.SELF_SOURCE,
            'infra/gpu03/task_specific_discovery/successor_dispatch.py',
            'infra/gpu03/task_specific_discovery/successor_phase_entry.py',
            'infra/gpu03/task_specific_discovery/guard_failure_diagnostic.py',
            'infra/gpu03/task_specific_discovery/phase_entry.py'}
        required |= set(w.original.authored.REQUIRED_SOURCES) | set(self.f.plan['qualified_source_files'])
        project = Path(w.__file__).resolve().parents[3]
        self.files = {}
        for name in required:
            path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, path); path.chmod(0o444)
            self.files[name] = w.io.binding(path)
        write(self.source / 'SOURCE_FILES.json', self.files)
        self.authored = write(self.f.old_root / 'inert_authored_admission.json',
            {'boundary': 'Full six-fixture source and sandbox admission tested independently; not executed here'})
        self.prior_inventory = write(self.f.old_root / 'INERT_PRIOR_EVAL_INVENTORY.json', self.files)

    def contexts(self, *, cached_wave_audit=None):
        stack = ExitStack(); stack.enter_context(self.f.contexts())
        stack.enter_context(patch.object(w, 'ROOT', str(self.root)))
        stack.enter_context(patch.object(binding, 'ROOT', str(self.root)))
        stack.enter_context(patch.object(binding, 'TOKEN', self.root.name))
        stack.enter_context(patch.object(w, 'PRIOR_EVAL_INVENTORY', self.prior_inventory))
        stack.enter_context(patch.object(w, 'source_and_authored', return_value=(
            self.files, {'run_token': w.HISTORICAL_TOKEN}, self.authored)))
        if cached_wave_audit is not None:
            stack.enter_context(patch.object(w, 'audit_waves', return_value=cached_wave_audit))
        return stack

    def build(self, output=None):
        with self.contexts():
            return w.build(self.f.collection_ref, self.manifest_refs,
                self.authored, self.authored, self.authored, self.source,
                output or self.root / 'evaluation-plan-v1')

    def close(self):
        self.owner.doCleanups()


class SourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.base = authored_fixtures.AuthoredAdmissionTests(
            'test_complete_authored_production_journals_and_cpu_release_admitted')
        self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.oldroot = self.base.root
        self.root = self.oldroot / 'codex-fresh-successor'; self.root.mkdir()
        self.source = self.root / 'source-successor-eval-v1'
        shutil.copytree(self.base.source, self.source)
        self.files = w.plan_io.read_json(self.source / 'SOURCE_FILES.json')
        for name, text in ((w.GUARD_SOURCE, '# qualified diagnostic guard\n'),
                           ('infra/gpu03/task_specific_discovery/causal_metrics.py', '# prior qualified metrics\n'),
                           ('infra/gpu03/task_specific_discovery/successor_plan.py', '# approved successor admission\n')):
            path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
            self.files[name] = w.io.binding(path)
        rewrite(self.source / 'SOURCE_FILES.json', self.files)
        self.prior = dict(self.files)
        self.prior[w.GUARD_SOURCE] = {'sha256': '0' * 64, 'size_bytes': 17}
        del self.prior['infra/gpu03/task_specific_discovery/successor_plan.py']
        self.prior_ref = write(self.oldroot / 'PRIOR_EVAL_SOURCE_FILES.json', self.prior)
        op = self.root / 'source-successor-v1'; op.mkdir()
        self.op_ref = write(op / 'SOURCE_FILES.json', self.files)
        self.plan = {'run_token': self.root.name, 'successor': {'operational_source_inventory': self.op_ref}}
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name, value in [('ROOT', str(self.root)), ('AUTHORED_TASK', self.base.task_ref),
                ('AUTHORED_PROOF', self.base.proof_ref), ('AUTHORED_SUPERVISOR', self.base.manifest_ref),
                ('HISTORICAL_TOKEN', self.base.f.task['run_token']), ('PRIOR_EVAL_INVENTORY', self.prior_ref)]:
            self.stack.enter_context(patch.object(w, name, value))

    def check(self, **change):
        return w.source_and_authored(self.plan, self.source,
            change.get('task', self.base.task_ref), change.get('proof', self.base.proof_ref),
            change.get('supervisor', self.base.manifest_ref), w.audit_io.Evidence())

    def change_source(self, name):
        path = self.source / name; path.write_text('# changed source\n')
        self.files[name] = w.io.binding(path); rewrite(self.source / 'SOURCE_FILES.json', self.files)

    def test_real_six_fixture_journals_and_exact_approved_guard_admitted(self):
        files, task, release = self.check()
        self.assertEqual(files, self.files); self.assertEqual(task['run_token'], w.HISTORICAL_TOKEN)
        self.assertNotEqual(task['run_token'], self.plan['run_token'])
        self.assertEqual(release['path'], str(self.oldroot / 'runtime/SUPERVISOR_RESULT.json'))

    def test_other_historical_qualification_cannot_be_substituted(self):
        bad = {**self.base.proof_ref, 'sha256': '1' * 64}
        with self.assertRaisesRegex(ContractError, 'exact historical'): self.check(proof=bad)

    def test_metrics_change_is_not_an_implicit_guard_exception(self):
        self.change_source('infra/gpu03/task_specific_discovery/causal_metrics.py')
        with self.assertRaisesRegex(ContractError, 'Previously qualified'): self.check()

    def test_guard_must_equal_exact_successor_operational_binding(self):
        self.change_source(w.GUARD_SOURCE)
        with self.assertRaisesRegex(ContractError, 'Previously qualified'): self.check()

    def test_newly_approved_successor_source_cannot_drift(self):
        self.change_source('infra/gpu03/task_specific_discovery/successor_plan.py')
        with self.assertRaisesRegex(ContractError, 'Approved successor-v1'): self.check()

    def test_scientific_evaluator_change_fails_original_authored_gate(self):
        self.change_source('infra/gpu03/task_specific_discovery/causal_evaluate.py')
        with self.assertRaisesRegex(ContractError, 'same-source'): self.check()

    def test_additive_preparation_is_bound_but_not_a_new_sandbox_qualification(self):
        path = self.source / 'infra/gpu03/task_specific_discovery/new_preparation.py'
        path.write_text('# inert additive preparation\n'); self.files[str(path.relative_to(self.source))] = w.io.binding(path)
        rewrite(self.source / 'SOURCE_FILES.json', self.files)
        files, _, _ = self.check()
        self.assertEqual(files[str(path.relative_to(self.source))], w.io.binding(path))

    def test_missing_original_positive_cpu_release_is_not_requalified(self):
        value = copy.deepcopy(self.base.release); value['owned_processes_released'] = False
        rewrite(self.oldroot / 'runtime/SUPERVISOR_RESULT.json', value)
        with self.assertRaisesRegex(ContractError, 'release missing'): self.check()

    def test_operational_inventory_cannot_move_to_another_directory(self):
        other = write(self.root / 'OTHER_SOURCE_FILES.json', self.files)
        self.plan['successor']['operational_source_inventory'] = other
        with self.assertRaisesRegex(ContractError, 'immutable successor-v1'): self.check()


class PreparationClaimTests(unittest.TestCase):
    def test_interruption_after_exclusive_claim_cannot_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'source-successor-eval-v1'; source.mkdir()
            plans = root / 'plans'; plans.mkdir()
            plan_ref = write(plans / 'PLAN.json', {'inert_boundary': True})
            proof_ref = write(root / 'NUMERICAL.json', {'inert_boundary': True})
            collection_ref = write(root / 'COLLECTION.json', {'plan': plan_ref, 'qualification': proof_ref})
            project = Path(w.__file__).resolve().parents[3]
            files = {n: w.io.binding(project / n) for n in
                     (w.SELF_SOURCE, w.WAVE_AUDITOR_SOURCE, w.WAVE_PREPARER_SOURCE, w.AUDIT_IO_SOURCE)}
            plan = {'run_token': root.name, 'pilot_requests': [{'request_id': str(i)} for i in range(15168)]}
            audited = (plan, {}, {}, {'generation_or_evaluation_performed': False})
            with patch.object(w, 'ROOT', str(root)), patch.object(w, 'audit_waves', return_value=audited), \
                    patch.object(w, 'source_and_authored', return_value=(files, {}, proof_ref)), \
                    patch.object(w, 'assemble', side_effect=RuntimeError('inert interruption after claim')):
                with self.assertRaisesRegex(RuntimeError, 'after claim'):
                    w.build(collection_ref, [], proof_ref, proof_ref, proof_ref, source, root / 'interrupted')
                claim = plans / 'PILOT_EVAL_TASK_BINDING.json'
                self.assertTrue(claim.is_file())
                self.assertFalse(w.plan_io.read_json(claim)['evaluation_replay_authorized'])
                with self.assertRaises(FileExistsError):
                    w.build(collection_ref, [], proof_ref, proof_ref, proof_ref, source, root / 'retry')
            self.assertFalse((root / 'retry').exists())


if __name__ == '__main__': unittest.main()
