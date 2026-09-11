"""Real authored plan/manifest admissions; no SSH, systemd, model or GPU call."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import h100_budget as budget
from infra.gpu03.direction_discovery import h100_supervisor as supervisor
from infra.gpu03.direction_discovery.test_h100_supervisor import Fixture


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(self.f.close)
        self.root = self.f.root / 'canonical-ledger'
        self.seed = supervisor.ref(self.f.seed)
        self.patches = [mock.patch.object(budget, 'LEDGER_ROOT', self.root),
                        mock.patch.object(budget, 'HISTORICAL_SEED_SHA', self.seed['sha256'])]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def spec(self, main=False):
        ref, m = self.f.build_stage(main)
        _, requests, _ = supervisor.request_context(m)
        return {'root': str(self.root), 'seed': self.seed, 'manifest': ref,
                'phase': m['phase'], 'run_token': m['run_token'], 'host_profile': m['host_profile'],
                'request_plan': m['request_plan'], 'generation_request_ids': [r['request_id'] for r in requests],
                'authorization': m['authorization']}

    def entries(self):
        return budget.journal_entries(self.root / 'admissions.jsonl')

    def test_continuation_budget_preserves_old_cap_and_requires_exact_amendment(self):
        authority=self.f.save(self.f.root/'amended-authority.json', {
            'integrated_validation':True,'generation_runtime_seconds':21600,
            'cumulative_gpu_wall_cap_seconds':43200})
        m={'phase':'h100_no_loophole_capability','continuation':{},
           'authorization':supervisor.ref(authority),'generation_runtime_seconds':21600,
           'generation_requests':735}
        self.assertEqual(budget.manifest_budget(m),(735,21780,43200))
        self.assertEqual(budget.manifest_budget({'phase':'h100_numerical_qualification'}),(6,780,28800))
        after={**budget.HISTORICAL,'generation_requests':3387,
               'gpu_phase_wall_seconds':budget.HISTORICAL['gpu_phase_wall_seconds']+780+21780}
        with self.assertRaisesRegex(ValueError,'Cumulative budget'): budget.counts(after)
        self.assertEqual(budget.counts(after,wall_cap=43200),after)
        for field,value in [('generation_runtime_seconds',7200),('generation_requests',740)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                budget.manifest_budget({**m,field:value})
        with self.assertRaises(ValueError):
            budget.manifest_budget({**m,'authorization':supervisor.ref(self.f.auth)})
        with self.assertRaises(ValueError): budget.counts(after,wall_cap=86400)

    def test_actual_manifest_qualification_then_main_complete_counts(self):
        first = self.spec(); a = budget.admit(**first)
        self.assertEqual(a['budget_after']['generation_requests'], 2652)
        second = self.spec(True); b = budget.admit(**second)
        for spec, admission, active, reserved in ((first,a,600,780),(second,b,7200,7380)):
            manifest=supervisor.load_manifest(spec['manifest']['path'],spec['manifest']['sha256'])
            limits=manifest['limits']
            self.assertEqual(limits['systemd_runtime_seconds'],active)
            self.assertEqual(limits['reserved_wall_seconds'],reserved)
            self.assertEqual(active+2*limits['timeout_stop_seconds'],reserved)
            self.assertEqual(admission['reserved_wall_seconds'],reserved)
            self.assertIn('--property=RuntimeMaxSec='+str(active),
                          supervisor.service_command(manifest,spec['manifest']['path'],spec['manifest']['sha256']))
        result = budget.account(self.seed, self.entries())
        self.assertEqual(result['generation_requests'], 3392)
        self.assertEqual(result['tf_requests'], 9228)
        self.assertEqual(result['gpu_phase_wall_seconds'], 28616.688430309296)
        self.assertEqual(result['untouched_test_requests'], 0)
        path = self.root / (second['run_token'] + '.admission.json')
        self.assertEqual(budget.validate_admission_reference(budget.ref(path),
            manifest_sha256=second['manifest']['sha256'], run_token=second['run_token']), b)
        self.assertFalse(path.stat().st_mode & 0o222)

    def test_group_writable_umask_still_creates_private_replayable_ledger(self):
        previous=os.umask(0o002)
        try:
            self.test_actual_manifest_qualification_then_main_complete_counts()
            self.assertEqual(self.root.stat().st_mode & 0o777,0o700)
            self.assertEqual(len(self.entries()),2)
        finally:
            os.umask(previous)

    def test_existing_unsafe_ledger_is_rejected_without_chmod(self):
        spec=self.spec(); self.root.mkdir(mode=0o700); self.root.chmod(0o775)
        with self.assertRaisesRegex(ValueError,'Writable ledger ancestor'): budget.admit(**spec)
        self.assertEqual(self.root.stat().st_mode & 0o777,0o775)
        self.assertFalse((self.root/'admissions.jsonl').exists())

    def test_main_cannot_precede_qualification(self):
        with self.assertRaises(ValueError): budget.admit(**self.spec(True))
        self.assertEqual(self.entries(), [])

    def test_main_stale_plan_rejected_after_qualification_commit(self):
        first = self.spec(); budget.admit(**first)
        # Direct preparation intentionally retains the original qualification's
        # old bookkeeping. It does not invoke Fixture's fresh-main-plan helper.
        reference = supervisor.build(self.f.spec(True))
        manifest = json.loads(Path(reference['path']).read_bytes())
        _, requests, _ = supervisor.request_context(manifest)
        spec = {**first, 'manifest':reference, 'phase':manifest['phase'], 'run_token':manifest['run_token'],
                'request_plan':manifest['request_plan'], 'generation_request_ids':[r['request_id'] for r in requests]}
        with self.assertRaisesRegex(ValueError, 'stale/reset'): budget.admit(**spec)
        self.assertEqual(len(self.entries()), 1)

    def test_duplicate_phase_cannot_reissue_with_new_token(self):
        spec = self.spec(); budget.admit(**spec)
        changed = {**spec, 'run_token': 'authored-another-qualification'}
        with self.assertRaises(ValueError): budget.admit(**changed)
        self.assertEqual(len(self.entries()), 1)

    def test_alternate_ledger_rejected_before_creation(self):
        spec = self.spec(); other = self.f.root / 'second-ledger'
        with self.assertRaises(ValueError): budget.admit(**{**spec, 'root': str(other)})
        self.assertFalse(other.exists())

    def test_changed_actual_manifest_ref_rejected_before_commit(self):
        spec = self.spec(); spec['manifest'] = {**spec['manifest'], 'sha256': '0' * 64}
        with self.assertRaises(ValueError): budget.admit(**spec)
        self.assertEqual(self.entries(), [])

    def test_reference_same_hash_wrong_profile_path_rejected(self):
        spec = self.spec(); original = Path(spec['host_profile']['path'])
        copied = self.f.save(self.f.root / 'profile-copy.json', original.read_bytes())
        spec['host_profile'] = supervisor.ref(copied)
        with self.assertRaises(ValueError): budget.admit(**spec)
        self.assertEqual(self.entries(), [])

    def test_exact_actual_request_union_required(self):
        spec = self.spec(); spec['generation_request_ids'][0] = 'foreign-id'
        with self.assertRaises(ValueError): budget.admit(**spec)
        self.assertEqual(self.entries(), [])

    def test_duplicate_request_id_rejected(self):
        spec = self.spec(); spec['generation_request_ids'][1] = spec['generation_request_ids'][0]
        with self.assertRaises(ValueError): budget.admit(**spec)

    def test_actual_limits_cannot_change(self):
        spec = self.spec(); original = Path(spec['manifest']['path']); m = json.loads(original.read_bytes())
        m['limits']['systemd_runtime_seconds'] -= 1
        altered = self.f.save(self.f.root / 'altered-manifest.json', m)
        spec['manifest'] = supervisor.ref(altered)
        with self.assertRaises(ValueError): budget.admit(**spec)
        self.assertEqual(self.entries(), [])

    def test_active_runtime_cannot_replace_full_wall_reservation(self):
        spec=self.spec(); original=Path(spec['manifest']['path']); m=json.loads(original.read_bytes())
        m['limits']['reserved_wall_seconds']=m['limits']['systemd_runtime_seconds']
        altered=self.f.save(self.f.root/'underreserved-manifest.json',m)
        with self.assertRaises(ValueError): budget.admit(**{**spec,'manifest':supervisor.ref(altered)})
        self.assertEqual(self.entries(),[])

    def test_failure_after_durable_append_does_not_rerun_or_refund(self):
        spec = self.spec()
        with mock.patch.object(budget, 'exclusive', side_effect=OSError('authored export failure')):
            with self.assertRaises(OSError): budget.admit(**spec)
        self.assertEqual(len(self.entries()), 1)
        self.assertEqual(budget.account(self.seed, self.entries())['generation_requests'], 2652)
        with self.assertRaises(ValueError): budget.admit(**spec)
        self.assertEqual(len(self.entries()), 1)
        self.assertFalse((self.root / (spec['run_token'] + '.admission.json')).exists())

    def test_pending_failed_reservation_remains_fully_charged(self):
        spec = self.spec(); budget.admit(**spec)
        value = budget.account(self.seed, self.entries())
        self.assertEqual(value['gpu_phase_wall_seconds'], budget.HISTORICAL['gpu_phase_wall_seconds'] + 780)
        self.assertEqual(value['phases'][-1]['wall_basis'], 'full_reserved_no_credit')

    def test_malformed_or_partial_journal_cannot_admit(self):
        spec = self.spec(); self.root.mkdir(mode=0o700)
        path = self.root / 'admissions.jsonl'
        for text in ('{', '{}', '\n', '{}\n\n', ' {"x":1}\n'):
            with self.subTest(text=text):
                path.write_text(text); path.chmod(0o600)
                with self.assertRaises((ValueError, KeyError)): budget.admit(**spec)

    def test_mutated_committed_manifest_identity_rejected_on_replay(self):
        spec = self.spec(); budget.admit(**spec); entries = self.entries()
        entries[0]['manifest_sha256'] = 'f' * 64
        with self.assertRaises(ValueError): budget.account(self.seed, entries)

    def test_rebound_journal_ids_and_counters_do_not_override_actual_plan(self):
        spec = self.spec(); budget.admit(**spec); entries = self.entries()
        entries[0]['generation_request_ids'] = ['invented-' + str(i) for i in range(6)]
        with self.assertRaises(ValueError): budget.account(self.seed, entries)

    def test_journal_budget_refund_or_bad_types_rejected(self):
        spec = self.spec(); budget.admit(**spec)
        for field, value in (('generation_requests', 0), ('gpu_phase_wall_seconds', float('nan'))):
            entries = self.entries(); entries[0]['budget_after'][field] = value
            with self.assertRaises(ValueError): budget.account(self.seed, entries)

    def test_boolean_zero_in_journal_counters_rejected(self):
        spec = self.spec(); budget.admit(**spec)
        for field in ('budget_before', 'budget_after'):
            entries = self.entries(); entries[0][field]['untouched_test_requests'] = False
            with self.assertRaises(ValueError): budget.account(self.seed, entries)

    def test_seed_cannot_be_replaced_by_same_counters_missing_history(self):
        spec = self.spec(); seed = json.loads(self.f.seed.read_bytes()); seed['budget']['phases'] = [{}]
        path = self.f.save(self.f.root / 'fake-seed.json', seed)
        with self.assertRaises(ValueError): budget.admit(**{**spec, 'seed': supervisor.ref(path)})

    def test_copied_admission_outside_canonical_path_rejected(self):
        spec = self.spec(); a = budget.admit(**spec)
        path = self.f.save(self.f.root / 'admission-copy.json', a)
        with self.assertRaises(ValueError): budget.validate_admission_reference(supervisor.ref(path),
            manifest_sha256=spec['manifest']['sha256'], run_token=spec['run_token'])

    def test_admission_missing_from_journal_rejected(self):
        spec = self.spec(); budget.admit(**spec)
        path = self.root / (spec['run_token'] + '.admission.json')
        (self.root / 'admissions.jsonl').write_text('')
        with self.assertRaises(ValueError): budget.validate_admission_reference(budget.ref(path),
            manifest_sha256=spec['manifest']['sha256'], run_token=spec['run_token'])

    def test_readonly_verifier_does_not_create_missing_ledger(self):
        with self.assertRaises(ValueError): budget.validate_admission_reference({'path': 'missing'},
            manifest_sha256='f' * 64, run_token='authored-missing-token')
        self.assertFalse(self.root.exists())

    def test_symlink_and_hardlink_ledger_files_rejected(self):
        spec = self.spec(); self.root.mkdir(mode=0o700)
        target = self.f.root / 'private-target'; target.write_text(''); target.chmod(0o600)
        lock = self.root / 'ledger.lock'; lock.symlink_to(target)
        with self.assertRaises(OSError): budget.admit(**spec)
        lock.unlink(); os.link(target, lock)
        with self.assertRaises(ValueError): budget.admit(**spec)

    def test_mutable_or_symlink_input_rejected(self):
        path = self.f.root / 'mutable.json'; path.write_text('{}')
        with self.assertRaises(ValueError): budget.ref(path)
        link = self.f.root / 'linked.json'; link.symlink_to(self.f.seed)
        with self.assertRaises(ValueError): budget.ref(link)

    def test_fifo_inputs_and_ledger_rejected_without_blocking(self):
        path = self.f.root / 'fifo'; os.mkfifo(path, 0o400)
        with self.assertRaises(ValueError): budget.ref(path)
        path.chmod(0o600)
        with self.assertRaises(ValueError): budget.private_open(path)


if __name__ == '__main__':
    unittest.main()
