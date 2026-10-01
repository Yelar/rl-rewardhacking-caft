"""Authored37-row CPU finalizer fixtures. No actual tokenizer, tensor or data load."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import no_loophole_plan as n
from infra.gpu03.direction_discovery import no_loophole_prepare as m
from infra.gpu03.direction_discovery.test_no_loophole_plan import Fixture


class Tokenizer:
    def __init__(self):
        self.events = []
        self.bad_old = False
        self.too_long = False
        self.bad_decode = False

    def apply_chat_template(self, prompt, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt is True and enable_thinking is False
        content = prompt[-1]['content']; i = int(content.split()[1].rstrip(';'))
        old = 'loophole' in content
        self.events.append(('old' if old else 'new', i, tokenize))
        if not tokenize:
            return 'authored-render-' + str(i)
        if old:
            return [1, 2, i + 20 + int(self.bad_old)]
        return self.encode('authored-render-' + str(i), add_special_tokens=False)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        i = int(text.rsplit('-', 1)[1])
        return [1, i + 100] + ([7] * 1535 if self.too_long else [])

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return 'bad' if self.bad_decode else 'authored-render-' + str(ids[1] - 100)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name).resolve()); self.addCleanup(self.f.close)
        f = self.f
        for old in f.carriers:
            old['prompt_token_ids_sha256'] = n.ids_hash(old['prompt_token_ids'])
            old['prompt_token_count'] = len(old['prompt_token_ids'])
        f.save(f.pkg / 'original_carriers37.jsonl', f.lines(f.carriers))
        f.set_constant('CARRIERS_SHA', n.sha(f.pkg / 'original_carriers37.jsonl'))
        files, tokenizer_files = [], {}
        for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
            path = f.save(f.root / name, ('authored-' + name).encode())
            ref = n.ref(path); tokenizer_files[name] = ref
            files.append({'archive_path': 'model_snapshot/' + name, 'sha256': ref['sha256'], 'size_bytes': ref['size_bytes']})
        f.save(f.pkg / 'tokenizer_policy.pending.json', {'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c', 'files': files})
        f.audit.update(record_count=37, tokenization_verified=False,
                       files={name: f.relative_ref(name) for name in m.AUDIT_FILES})
        self.audit_path = f.save(f.pkg / 'pending_audit.json', f.audit)
        f.audit_proof.update(prompt_package=n.ref(self.audit_path), original_carriers37_sha256=n.CARRIERS_SHA)
        f.save(f.audit_proof_path, f.audit_proof)
        self.spec = {'schema_version': 1, 'purpose': 'prepare_no_loophole_cpu_inputs', 'runnable': True,
                     'old_plan': n.ref(f.old_path), 'prompt_audit': n.ref(self.audit_path),
                     'prompt_audit_verification': n.ref(f.audit_proof_path), 'tokenizer_files': tokenizer_files,
                     'source_files': {'fixture-only': {'sha256': '0' * 64, 'size_bytes': 0}},
                     'source_root': str(f.root), 'runtime_versions': m.VERSIONS,
                     'output': str(f.root / 'new-package'), 'control': str(f.root / 'new-control')}
        self.spec_path = f.save(f.root / 'spec.json', self.spec)
        self.spec_ref = n.ref(self.spec_path)
        self.tokenizer = Tokenizer()
        self.vector_bytes = (f.pkg / 'projection_vectors.safetensors').read_bytes()
        self.vector_hashes = json.loads((f.pkg / 'projection_audit.json').read_bytes())['vector_sha256']
        # Runtime/import/process interfaces are authored mocks. Real metadata,
        # package, carrier and token audit construction/verification run unchanged.
        for name in ('check_sources', 'check_runtime', 'require_released'):
            patcher = patch.object(m, name); patcher.start(); self.addCleanup(patcher.stop)
        for name, value in [('load_tokenizer', self.tokenizer),
                            ('projection_snapshot', (self.vector_bytes, self.vector_hashes))]:
            patcher = patch.object(m, name, return_value=value); patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(m, 'projection_child', side_effect=lambda ref, out, ctrl: m.verify_projections(ref, out))
        patcher.start(); self.addCleanup(patcher.stop)

    def respec(self):
        self.f.save(self.spec_path, self.spec); self.spec_ref = n.ref(self.spec_path)

    def produce(self):
        return m.prepare(self.spec_ref)

    def observer(self, answer, **changes):
        stdout = self.f.save(self.f.root / 'producer.stdout', answer)
        receipt = {'returncode': 0, 'timed_out': False, 'error_type': None, 'child_reaped': True,
                   'remaining_group_pids': [], 'child_identity': {'pid': 99991, 'pgid': 99991, 'start_ticks': 1, 'uid': os.getuid()},
                   'stdout': n.ref(stdout)}
        receipt.update(changes)
        return n.ref(self.f.save(self.f.root / 'producer_exit.json', receipt))

    def test_real_prepare_verify_then_actual_planner(self):
        answer = self.produce()
        # CLI refs omit optional size; actual stdout contains it.
        bundle = {k: v for k, v in answer['prompt_package'].items() if k != 'size_bytes'}
        proof_ref = m.verify(self.spec_ref, bundle, self.observer(answer), self.f.root / 'independent.json')
        proof = n.load_ref(proof_ref)
        self.assertTrue(proof['prompt_token_replay']); self.assertTrue(proof['projection_reconstruction_bitwise_equal'])
        plan = n.build_plan(self.f.old_path, bundle, self.f.budget, prompt_package_verification=proof_ref)
        self.assertEqual(len(plan['requests']), 740)
        self.assertEqual(plan['generation_requests_after_commit'], 3386)
        self.assertEqual(plan['previously_committed_untouched_test_requests'], 0)
        rows = n.jsonl(Path(plan['prepared_records']).read_bytes())
        self.assertEqual(rows[0]['prompt'], self.f.dataset[0]['prompt'])
        self.assertEqual(rows[0]['completion'], self.f.carriers[0]['completion'])
        self.assertNotIn('regions', rows[0]); self.assertNotIn('engine_prompt_token_ids', rows[0])
        self.assertEqual([e[0] for e in self.tokenizer.events[:37]], ['old'] * 37)

    def test_pending_spec_fails_before_all_loaders(self):
        self.spec['runnable'] = False; self.respec()
        with self.assertRaisesRegex(ValueError, 'pending'):
            self.produce()
        m.check_sources.assert_not_called(); m.load_tokenizer.assert_not_called()

    def test_bad_alignment_proof_fails_before_tokenizer(self):
        proof = copy.deepcopy(self.f.audit_proof); proof['original_hint_roundtrip_verified'] = False
        path = self.f.save(self.f.root / 'badproof.json', proof)
        self.spec['prompt_audit_verification'] = n.ref(path); self.respec()
        with self.assertRaisesRegex(ValueError, 'independent proof'):
            self.produce()
        m.load_tokenizer.assert_not_called()

    def test_wrong_tokenizer_hash_fails_before_tokenizer(self):
        self.spec['tokenizer_files']['tokenizer.json']['sha256'] = '0' * 64; self.respec()
        with self.assertRaisesRegex(ValueError, 'tokenizer revision'):
            self.produce()
        m.load_tokenizer.assert_not_called()

    def test_old_replay_failure_creates_no_package_or_new_token(self):
        self.tokenizer.bad_old = True
        with self.assertRaisesRegex(ValueError, 'Old prompt token replay'):
            self.produce()
        self.assertFalse(Path(self.spec['output']).exists())
        self.assertFalse(any(x[0] == 'new' for x in self.tokenizer.events))

    def test_length_overflow_does_not_truncate_or_publish(self):
        self.tokenizer.too_long = True
        with self.assertRaisesRegex(ValueError, 'token limits'):
            self.produce()
        self.assertFalse(Path(self.spec['output']).exists())

    def test_render_roundtrip_failure_does_not_publish(self):
        self.tokenizer.bad_decode = True
        with self.assertRaisesRegex(ValueError, 'roundtrip'):
            self.produce()
        self.assertFalse(Path(self.spec['output']).exists())

    def test_existing_output_is_never_reused(self):
        Path(self.spec['output']).mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh sibling'):
            self.produce()
        m.load_tokenizer.assert_not_called()

    def test_projection_child_failure_has_no_bundle(self):
        m.projection_child.side_effect = RuntimeError('authored child failed')
        with self.assertRaisesRegex(RuntimeError, 'child failed'):
            self.produce()
        self.assertFalse((Path(self.spec['output']) / 'prompt_package.json').exists())

    def test_nonzero_producer_fails_before_context(self):
        receipt = self.observer({}, returncode=1)
        with patch.object(m, 'context') as context:
            with self.assertRaisesRegex(ValueError, 'exit and release'):
                m.verify(self.spec_ref, {}, receipt, self.f.root / 'bad.json')
            context.assert_not_called()

    def test_live_producer_fails_before_context(self):
        receipt = self.observer({})
        with patch.object(m, 'context') as context:
            m.require_released.side_effect = ValueError('still present')
            with self.assertRaisesRegex(ValueError, 'still present'):
                m.verify(self.spec_ref, {}, receipt, self.f.root / 'bad.json')
            context.assert_not_called()

    def test_stdout_other_package_is_rejected(self):
        answer = self.produce(); wrong = copy.deepcopy(answer)
        wrong['prompt_package']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'stdout'):
            m.verify(self.spec_ref, answer['prompt_package'], self.observer(wrong), self.f.root / 'bad.json')

    def test_postproducer_token_mutation_is_rejected(self):
        answer = self.produce(); receipt = self.observer(answer)
        self.tokenizer.bad_old = True
        with self.assertRaisesRegex(ValueError, 'Old prompt token replay'):
            m.verify(self.spec_ref, answer['prompt_package'], receipt, self.f.root / 'bad.json')

    def test_postproducer_tensor_drift_is_rejected(self):
        answer = self.produce(); receipt = self.observer(answer)
        m.projection_snapshot.return_value = (self.vector_bytes[:-1] + b'!', self.vector_hashes)
        with self.assertRaisesRegex(ValueError, 'Q reconstruction differs'):
            m.verify(self.spec_ref, answer['prompt_package'], receipt, self.f.root / 'bad.json')

    def test_payload_tampering_is_rejected_before_reconstruction(self):
        answer = self.produce(); receipt = self.observer(answer)
        self.f.save(Path(self.spec['output']) / 'prepared_records.jsonl', b'{}\n')
        with self.assertRaisesRegex(ValueError, 'hash/size changed'):
            m.verify(self.spec_ref, answer['prompt_package'], receipt, self.f.root / 'bad.json')


class RuntimeSourceTests(unittest.TestCase):
    def test_runtime_success_and_version_failure_without_imports(self):
        env = {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        spec = {'runtime_versions': m.VERSIONS, 'python_binary': {'path': m.sys.executable}}
        with patch.dict(os.environ, env, clear=True), patch.object(m.sys, 'version_info', (3, 12, 3)), \
             patch.object(m.importlib.metadata, 'version', side_effect=lambda key: m.VERSIONS[key]), \
             patch.object(m, 'content_ref') as read:
            m.check_runtime(spec); read.assert_called_once()
            with patch.object(m.importlib.metadata, 'version', return_value='different'):
                with self.assertRaisesRegex(ValueError, 'versions differ'):
                    m.check_runtime(spec)

    def test_cuda_visibility_fails_before_versions_or_imports(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=True), \
             patch.object(m.importlib.metadata, 'version') as versions:
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                m.check_runtime({})
            versions.assert_not_called()

    def test_complete_actual_source_inventory_and_omitted_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            paths = [m.HERE, m.PLANNER, *n.SOURCES, 'src/authored.py']
            for name in paths:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                m.write(path, ('# authored ' + name + '\n').encode())
            pins = {name: n.sha(root / name) for name in n.SOURCES}
            proof = {'status': 'independently_verified_remote_generation_source', 'qualified_jobs_released': True,
                     'producer_exit_status': 0, 'source_inventory': pins}
            proof_ref = m.write(root / 'source-proof.json', proof)
            source_files = {name: {k: v for k, v in n.ref(root / name).items() if k != 'path'} for name in paths}
            spec = {'source_root': str(root), 'source_files': source_files, 'historical_source_qualification': proof_ref}
            with patch.object(m, '__file__', str(root / m.HERE)), patch.object(n, '__file__', str(root / m.PLANNER)), \
                 patch.object(n, 'SOURCES', pins), patch.object(m, 'SOURCE_PROOF_SHA', proof_ref['sha256']):
                m.check_sources(spec)
                del source_files['src/authored.py']
                with self.assertRaisesRegex(ValueError, 'inventory omits'):
                    m.check_sources(spec)


if __name__ == '__main__':
    unittest.main()
