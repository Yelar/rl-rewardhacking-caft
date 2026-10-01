"""H100 preparation success/failure fixtures; no actual tensor/tokenizer loads."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import no_loophole_plan as old
from infra.gpu03.direction_discovery import h100_no_loophole_protocol as n
from infra.gpu03.direction_discovery import h100_no_loophole_prepare as m
from infra.gpu03.direction_discovery.test_h100_no_loophole_protocol import Fixture
from infra.gpu03.direction_discovery.test_no_loophole_prepare import Tokenizer


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        f = self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(f.close)
        for carrier in f.carriers:
            carrier['prompt_token_ids_sha256'] = n.ids_hash(carrier['prompt_token_ids'])
            carrier['prompt_token_count'] = len(carrier['prompt_token_ids'])
        f.save(f.pkg / 'original_carriers37.jsonl', f.lines(f.carriers))
        f.set_constant('CARRIERS_SHA', n.sha(f.pkg / 'original_carriers37.jsonl'))
        f.hpatch('CARRIERS_SHA', old.CARRIERS_SHA)
        files, tokenizer_files = [], {}
        for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
            path = f.save(f.work / 'input' / name, ('authored-' + name).encode())
            r = n.ref(path); tokenizer_files[name] = r
            files.append({'archive_path': 'model_snapshot/' + name, 'sha256': r['sha256'], 'size_bytes': r['size_bytes']})
        f.save(f.pkg / 'tokenizer_policy.pending.json', {'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'files': files})
        f.audit.update(record_count=37, tokenization_verified=False,
                       files={name: f.relative_ref(name) for name in m.AUDIT_FILES})
        audit_path = f.save(f.pkg / 'pending_audit.json', f.audit)
        f.audit_proof.update(prompt_package=n.ref(audit_path), original_carriers37_sha256=n.CARRIERS_SHA)
        f.save(f.audit_proof_path, f.audit_proof)
        self.spec = {'schema_version': 1, 'purpose': 'prepare_h100_no_loophole_cpu_inputs', 'runnable': True,
                     'old_plan': n.ref(f.old_path), 'prompt_audit': n.ref(audit_path),
                     'prompt_audit_verification': n.ref(f.audit_proof_path), 'tokenizer_files': tokenizer_files,
                     'source_files': {'authored': {'sha256': '0' * 64, 'size_bytes': 0}},
                     'source_root': str(f.root), 'runtime_versions': m.VERSIONS,
                     'python_binary': n.ref(f.python), 'h100_portability': f.portability,
                     'output': str(f.work / 'prepared'), 'control': str(f.work / 'prepared-control')}
        self.spec_path = f.work / 'spec.json'; self.respec()
        self.tokenizer = Tokenizer()
        self.vectors = (f.pkg / 'projection_vectors.safetensors').read_bytes()
        self.hashes = json.loads((f.pkg / 'projection_audit.json').read_bytes())['vector_sha256']
        # Only heavy runtime/tokenizer/tensor/process interfaces are injected.
        # Real relocation/runtime-proof, source-byte, token, package and planner
        # validators run with authored files throughout producer and verifier.
        for key in ('check_sources', 'check_runtime', 'require_released'):
            p = patch.object(m, key); p.start(); self.addCleanup(p.stop)
        for key, result in [('load_tokenizer', self.tokenizer), ('projection_snapshot', (self.vectors, self.hashes))]:
            p = patch.object(m, key, return_value=result); p.start(); self.addCleanup(p.stop)
        p = patch.object(m, 'projection_child', side_effect=lambda r, output, control: m.verify_projections(r, output))
        p.start(); self.addCleanup(p.stop)

    def respec(self):
        self.f.save(self.spec_path, self.spec); self.spec_ref = n.ref(self.spec_path)

    def observer(self, answer, **changes):
        stdout = self.f.save(self.f.work / 'producer.stdout', answer)
        receipt = {'returncode': 0, 'timed_out': False, 'error_type': None, 'child_reaped': True,
                   'remaining_group_pids': [], 'child_identity': {'pid': 99991, 'pgid': 99991, 'start_ticks': 1, 'uid': os.getuid()},
                   'stdout': n.ref(stdout)}
        receipt.update(changes)
        return n.ref(self.f.save(self.f.work / 'producer_exit.json', receipt))

    def test_actual_preparation_external_verify_full740_and_only_mapped_engine_input(self):
        f = self.f
        f.target.rename(f.target.with_suffix('.absent'))
        result = m.prepare(self.spec_ref)
        proof = m.verify(self.spec_ref, result['prompt_package'], self.observer(result), f.work / 'final_proof.json')
        plan = n.build_plan(f.old_path, result['prompt_package'], f.budget, prompt_package_verification=proof)
        self.assertEqual(len(plan['requests']), 740)
        self.assertEqual(plan['conditions'], f.conditions)
        self.assertEqual(n.load_ref(proof)['h100_portability_sha256'], n.digest(f.portability))
        self.assertEqual([x[0] for x in self.tokenizer.events[:37]], ['old'] * 37)
        for call in m.projection_snapshot.call_args_list:
            physical = call.args[0]
            self.assertEqual(physical['target:L21.transition.pc04']['layers'][0]['path'], str(f.mapped))
            physical = copy.deepcopy(physical); physical['target:L21.transition.pc04']['layers'][0]['path'] = str(f.target)
            self.assertEqual(physical, f.conditions)
        audit = json.loads((Path(self.spec['output']) / 'projection_audit.json').read_bytes())
        self.assertEqual(audit['conditions'], {k: v for k, v in f.conditions.items() if v['layers']})
        self.assertFalse(audit['historical_Q_byte_comparison_performed'])

    def test_pending_spec_blocks_before_loaders(self):
        self.spec['runnable'] = False; self.respec()
        with self.assertRaisesRegex(ValueError, 'pending'): m.prepare(self.spec_ref)
        m.load_tokenizer.assert_not_called(); m.projection_snapshot.assert_not_called()

    def test_bad_runtime_release_blocks_before_tokenizer(self):
        self.f.runtime_proof['process_release_verified'] = False
        p = self.f.save(self.f.runtime_proof_path, self.f.runtime_proof)
        self.spec['h100_portability']['runtime_replacement']['qualification'] = n.ref(p); self.respec()
        with self.assertRaisesRegex(ValueError, 'positive qualification'): m.prepare(self.spec_ref)
        m.load_tokenizer.assert_not_called(); m.projection_snapshot.assert_not_called()

    def test_preparation_python_must_match_runtime_proof(self):
        self.spec['python_binary'] = n.ref(self.f.save(self.f.work / 'other-python', b'other'))
        self.respec()
        with self.assertRaisesRegex(ValueError, 'Python differs'): m.prepare(self.spec_ref)
        m.load_tokenizer.assert_not_called()

    def test_old_token_replay_failure_has_no_new_package(self):
        self.tokenizer.bad_old = True
        with self.assertRaisesRegex(ValueError, 'Old prompt token replay'): m.prepare(self.spec_ref)
        self.assertFalse(Path(self.spec['output']).exists())
        m.projection_snapshot.assert_not_called()

    def test_length_limit_failure_does_not_truncate(self):
        self.tokenizer.too_long = True
        with self.assertRaisesRegex(ValueError, 'token limits'): m.prepare(self.spec_ref)
        self.assertFalse(Path(self.spec['output']).exists())

    def test_external_failed_exit_before_any_context(self):
        with patch.object(m, 'context') as ctx:
            with self.assertRaisesRegex(ValueError, 'exit and release'):
                m.verify(self.spec_ref, {}, self.observer({}, returncode=1), self.f.work / 'proof.json')
            ctx.assert_not_called()

    def test_independent_reconstruction_drift_has_no_success_proof(self):
        result = m.prepare(self.spec_ref); receipt = self.observer(result)
        m.projection_snapshot.return_value = (self.vectors[:-1] + b'!', self.hashes)
        with self.assertRaisesRegex(ValueError, 'Q reconstruction differs'):
            m.verify(self.spec_ref, result['prompt_package'], receipt, self.f.work / 'proof.json')
        self.assertFalse((self.f.work / 'proof.json').exists())


if __name__ == '__main__':
    unittest.main()
