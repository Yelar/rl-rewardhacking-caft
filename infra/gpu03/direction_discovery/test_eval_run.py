import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from infra.gpu03.direction_discovery import eval_run as run


def evaluated(generation, correct=True):
    from .test_solver_diagnostic import diagnostic
    return {**{key: generation[key] for key in run.IDENTITY_FIELDS},
            'generation': generation['result'],
            'generation_sha256': run.hashlib.sha256(run.evaluate.canonical(generation).encode()).hexdigest(),
            'evaluation_status': 'evaluated',
            'correctness_diagnostics': diagnostic(correct=correct, whole=correct),
            'metrics': {**{key: False for key in run.evaluate.BINARY_METRICS},
                        'ground_truth_correctness': correct,
                        'completion_length': len(generation['result']['completion_token_ids'])}}


class IdentityTests(unittest.TestCase):
    def test_new_policy_requires_diagnostic_and_historical_policy_allows_absence(self):
        _, _, _, generations = run.authored_inputs()
        row = evaluated(generations[0])
        del row['correctness_diagnostics']
        run.validate_evaluation_row(row, generations[0])
        with self.assertRaisesRegex(ValueError, 'Missing or changed solver diagnostic'):
            run.validate_evaluation_row(row, generations[0], require_diagnostic=True)

    def setUp(self):
        self.plan, self.prepared, self.dataset, self.rows = run.authored_inputs()

    def test_whole_problem_or_sample_omitted_for_every_condition_fails(self):
        with self.assertRaisesRegex(ValueError, 'missing=1'):
            run.validate_inputs(self.rows[:1], self.plan, self.prepared, self.dataset)

    def test_duplicate_identical_and_conflicting_generations_fail(self):
        for changed in (False, True):
            duplicate = copy.deepcopy(self.rows[0])
            if changed:
                duplicate['seed'] += 1
            with self.assertRaisesRegex(ValueError, 'duplicate/conflicting'):
                run.validate_inputs([*self.rows, duplicate], self.plan, self.prepared, self.dataset)

    def test_extra_generation_and_seed_change_fail(self):
        extra = copy.deepcopy(self.rows[0]); extra['request_id'] += '-unexpected'
        with self.assertRaisesRegex(ValueError, 'extra=1'):
            run.validate_inputs([*self.rows, extra], self.plan, self.prepared, self.dataset)
        self.rows[0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'immutable plan'):
            run.validate_inputs(self.rows, self.plan, self.prepared, self.dataset)

    def test_prompt_drift_and_qualification_mode_fail(self):
        self.dataset[0]['prompt'] = []
        with self.assertRaisesRegex(ValueError, 'prompt differs'):
            run.validate_inputs(self.rows, self.plan, self.prepared, self.dataset)
        self.plan['mode'] = 'qualify'
        with self.assertRaisesRegex(ValueError, 'mode=generate'):
            run.validate_inputs(self.rows, self.plan, self.prepared, self.dataset)

    def test_evaluation_unknowns_preserved_and_wrong_generation_fails(self):
        row = evaluated(self.rows[0])
        row['evaluation_status'] = 'suspicious_or_unknown'
        row['metrics']['harmful_modification'] = None
        run.validate_evaluation_row(row, self.rows[0])
        row['generation_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'another generated'):
            run.validate_evaluation_row(row, self.rows[0])

    def test_infrastructure_failure_cannot_be_reused(self):
        row = evaluated(self.rows[0]); row['evaluation_status'] = 'infrastructure_failure'
        with self.assertRaisesRegex(ValueError, 'Infrastructure failure'):
            run.validate_evaluation_row(row, self.rows[0])

    def test_incomplete_journal_and_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.jsonl'
            path.write_text('{"request_id":"x"}')
            with self.assertRaisesRegex(ValueError, 'incomplete JSONL'):
                list(run.read_jsonl(path))
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                run.parse(text)


class PackageTests(unittest.TestCase):
    def test_v1_generation_package_cannot_enter_v2_evaluation(self):
        from infra.gpu03.direction_discovery import supervisor
        m={'scientific':{'master_plan_sha256':run.PLAN_SHA256}}
        package={'manifest':'/authored/v1.json','manifest_sha256':'a'*64,'artifact_manifest_sha256':'b'*64}
        with mock.patch.object(supervisor,'verify',return_value={'status':'verified'}),mock.patch.object(supervisor,'load_manifest',return_value=m):
            with self.assertRaisesRegex(ValueError,'another experiment plan'):
                run.collect_generation_packages([package],request_plan={'conditions':{}},master_sha='c'*64)

    def test_each_terminal_generation_worker_must_have_full_coverage(self):
        from infra.gpu03.direction_discovery import supervisor
        plan, _, _, rows = run.authored_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path, manifest = root/'task.json', root/'reviewed.json'
            task = {'mode':'generate','requests':plan['requests'],'output':str(root)}
            run.write_json(task_path, task)
            run.write_json(root/'artifact_manifest.json', {'files':{}})
            run.write_jsonl(root/'results.jsonl', rows[:1])
            m = {'output':str(root),'workers':[{'success_expect':{'mode':'generate','requests':2},
                                              'command':['python','engine','--task',str(task_path)]}]}
            package = {'manifest':str(manifest),'manifest_sha256':'0'*64,
                       'artifact_manifest_sha256':run.sha256(root/'artifact_manifest.json')}
            with mock.patch.object(supervisor, 'verify', return_value={'status':'verified'}), mock.patch.object(supervisor, 'load_manifest', return_value=m):
                with self.assertRaisesRegex(ValueError, 'terminal generation worker exact'):
                    run.collect_generation_packages([package])

    def test_non_generation_package_and_duplicate_packages_fail(self):
        from infra.gpu03.direction_discovery import supervisor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run.write_json(root/'artifact_manifest.json', {})
            package = {'manifest':str(root/'m.json'),'manifest_sha256':'0'*64,
                       'artifact_manifest_sha256':run.sha256(root/'artifact_manifest.json')}
            m = {'output':str(root), 'workers':[{'success_expect':{'mode':'qualify'}}]}
            with mock.patch.object(supervisor, 'verify', return_value={'status':'verified'}), mock.patch.object(supervisor, 'load_manifest', return_value=m):
                with self.assertRaisesRegex(ValueError, 'Non-generation/qualification'):
                    run.collect_generation_packages([package])


class TestBundlePackageTests(unittest.TestCase):
    """Real immutable metadata/JSONL; producer and bundle-science verifier mocked."""
    def setUp(self):
        from infra.gpu03.direction_discovery import test_test_execution as fixtures, test_execution
        self.e = test_execution
        self.fixture = fixtures.TestExecutionTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root.resolve()
        self.fixture.full_path = self.fixture.full_path.resolve()
        self.fixture.bundle_path = self.fixture.bundle_path.resolve()
        self.full = self.fixture.full
        self.prepared = self.root / 'bundle/prepared_records.jsonl'
        prompt = [{'role': 'user', 'content': 'Authored metadata fixture.'}]
        self.prepared_rows = [{'record_id': f'record-{i}', 'problem_id': str(i), 'problem_split': 'untouched_test',
            'prompt': prompt, 'completion_token_ids': [7, 8],
            'regions': {'evaluator': {'first_executable_completion_token': 1}}} for i in range(3)]
        self.dataset_rows = [{'id': str(i), 'prompt': prompt, 'gt_answer': ['assert True']} for i in range(3)]
        run.write_jsonl(self.prepared, self.prepared_rows); self.prepared.chmod(0o400)
        self.full['prepared_records'] = str(self.prepared)
        self.bundle = self.fixture.bundle
        self.bundle['prepared_records'] = {'path': 'prepared_records.jsonl', 'sha256': run.sha256(self.prepared)}
        self.refreeze(self.fixture.full_path, self.full)
        self.bundle['generation_request_plan']['sha256'] = run.sha256(self.fixture.full_path)
        self.refreeze(self.fixture.bundle_path, self.bundle)
        self.reference = {'path': str(self.fixture.bundle_path), 'sha256': run.sha256(self.fixture.bundle_path)}
        self.part = self.e.make_part(self.reference, 0, self.fixture.previous)
        self.packages, self.manifests, self.proofs, self.parts = [], {}, {}, []

    def refreeze(self, path, value):
        path.chmod(0o600); path.write_text(self.e.canonical(value)); path.chmod(0o400)

    def package(self, part, name):
        stage = self.root / ('codex-discovery-test-' + name + '-stage')
        token = stage.name.removesuffix('-stage'); output = stage / 'results'
        task = {'mode': 'generate', 'run_token': token, 'worker_name': 'gpu_7', 'output': str(output / 'worker'),
                'requests': part['requests'], 'conditions': part['conditions'], 'sampling': part['sampling'],
                'prepared_records': str(self.prepared)}
        pp = self.fixture.save(stage / 'input/request_plan.json', part)
        tp = self.fixture.save(stage / 'tasks/gpu_7.json', task)
        artifact = self.fixture.save(output / 'artifact_manifest.json', {'authored': True})
        result = output / 'worker/results.jsonl'; result.parent.mkdir()
        run.write_jsonl(result, ({**r, 'result': {'completion': 'authored fixture',
            'completion_token_ids': [8] if r['scope'] == 'primary' else [7, 8], 'generated_token_ids': [8],
            'fixed_completion_prefix_token_count': 0 if r['scope'] == 'primary' else 1}} for r in part['requests']))
        m = {'run_token': token, 'host': 'gpu-04', 'stage': str(stage), 'output': str(output),
             'scientific': {'master_plan_sha256': self.full['master_plan_sha256'], 'training': False,
                            'test_bundle': self.reference, 'test_execution': part['test_execution'],
                            'input_prepared_sha256': run.sha256(self.prepared)},
             'workers': [{'name': 'gpu_7', 'command': ['python', 'engine.py', '--task', str(tp)],
                          'success_expect': {'mode': 'generate', 'requests': len(part['requests'])}}],
             'bound_files': {str(p): run.info(p) for p in [pp, tp, *self.e.bindings(part)]},
             'limits': {'systemd_runtime_seconds': 7200}}
        mp = self.fixture.save(stage / 'reviewed_manifest.json', m)
        proof = {'status': 'verified', 'manifest_sha256': run.sha256(mp), 'run_token': token, 'gpu_release_verified': True}
        vp = self.fixture.save(stage / 'independent_verification.json', proof)
        ref = {'manifest': str(mp), 'manifest_sha256': run.sha256(mp), 'artifact_manifest_sha256': run.sha256(artifact),
               'verification': str(vp), 'verification_sha256': run.sha256(vp)}
        self.packages.append(ref); self.manifests[str(mp)] = m; self.proofs[str(mp)] = proof; self.parts.append(part)
        return ref

    def complete(self):
        for index in range(3):
            previous = {**self.fixture.previous, 'generations': 2526 + 5 * index,
                        'untouched_test_generations': 5 * index, 'untouched_test_requests': 5 * index}
            part = self.e.make_part(self.reference, index, previous, self.packages)
            self.package(part, 'r' + str(index))

    def collect(self, packages=None, reference=True):
        from infra.gpu03.direction_discovery import supervisor
        with mock.patch.object(supervisor, 'verify', side_effect=lambda p, d: self.proofs[str(p)]):
            return run.collect_generation_packages(self.packages if packages is None else packages,
                request_plan=self.full, master_sha=self.full['master_plan_sha256'],
                test_bundle=self.reference if reference else None)

    def test_all_three_complete_parts_join_full_plan_and_preserve_proofs(self):
        self.complete(); rows, proofs = self.collect()
        self.assertEqual(len(rows), 15); self.assertEqual(len(proofs), 3)
        self.assertEqual({r['request_id'] for r in rows}, {r['request_id'] for r in self.full['requests']})
        with mock.patch('infra.gpu03.direction_discovery.supervisor.verify', side_effect=AssertionError('No service re-verification')):
            contexts = run.test_generation_metadata(proofs, request_plan=self.full, test_bundle=self.reference,
                master_sha=self.full['master_plan_sha256'], stored_proofs=proofs)
        self.assertEqual(len(contexts), 3)

    def test_missing_part_fails_before_completion_reads(self):
        self.complete()
        with mock.patch.object(run, 'read_jsonl', side_effect=AssertionError('Early completion read')):
            with self.assertRaisesRegex(ValueError, 'all fixed test'): self.collect(self.packages[:2])

    def test_last_producer_failure_precedes_any_completion_read(self):
        self.complete(); self.proofs[self.packages[-1]['manifest']]['gpu_release_verified'] = False
        with mock.patch.object(run, 'read_jsonl', side_effect=AssertionError('Early completion read')):
            with self.assertRaisesRegex(ValueError, 'release proof'): self.collect()

    def test_duplicate_part_and_absent_explicit_bundle_fail(self):
        self.complete()
        with self.assertRaisesRegex(ValueError, 'Duplicate generation'): self.collect([self.packages[0]] * 3)
        with self.assertRaisesRegex(ValueError, 'explicit bundle'): self.collect(reference=False)

    def test_part_plan_cannot_replace_full_evaluation_plan(self):
        with self.assertRaisesRegex(ValueError, 'full scientific plan'):
            run.validate_inputs([], self.part, [], [])
        with self.assertRaisesRegex(ValueError, 'exact full bundle'):
            run.test_bundle_context(self.reference, self.part)

    def test_wrong_prepared_union_and_science_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prepared union'):
            run.test_bundle_context(self.reference, self.full, prepared_sha='bad')
        full = copy.deepcopy(self.full); full['sampling']['temperature'] = .8
        with self.assertRaisesRegex(ValueError, 'exact full bundle'):
            run.test_bundle_context(self.reference, full)

    def test_missing_generation_row_retains_journal_and_fails_coverage(self):
        self.complete()
        m = self.manifests[self.packages[1]['manifest']]
        task = run.read_json(m['workers'][0]['command'][3]); path = Path(task['output']) / 'results.jsonl'
        original = path.read_text(); path.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, 'terminal generation worker exact'): self.collect()
        self.assertTrue(path.is_file()); self.assertNotEqual(path.read_text(), original)

    def test_stored_proof_coverage_or_identity_drift_fails(self):
        self.complete(); _, proofs = self.collect()
        for field, value in (('verification', {'status': 'failed'}), ('request_ids', [])):
            changed = copy.deepcopy(proofs); changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.test_generation_metadata(changed, request_plan=self.full, test_bundle=self.reference,
                    master_sha=self.full['master_plan_sha256'], stored_proofs=changed)

    def test_incomplete_or_mixed_bundle_metadata_precedes_outcomes(self):
        self.complete()
        last = self.packages[-1]; path = Path(last['manifest']); m = self.manifests[str(path)]
        m['scientific']['execution_partition'] = {'round_index': 1}
        self.refreeze(path, m); last['manifest_sha256'] = run.sha256(path)
        self.proofs[str(path)]['manifest_sha256'] = last['manifest_sha256']
        with mock.patch.object(run, 'read_jsonl', side_effect=AssertionError('Early completion read')):
            with self.assertRaisesRegex(ValueError, 'Cannot mix'): self.collect()

    def test_complete_production_builder_and_reload_bind_bundle_with_unchanged_cpu_profile(self):
        # Static metadata/runtime fixture; real plan/task/input collectors and
        # manifest verifier run. No service or generated-code evaluator is run.
        master = self.fixture.save(self.root / 'master.json', {'no_training': True, 'host': 'gpu-04', 'authorization': 'authored CPU metadata only'})
        self.full['master_plan_sha256'] = run.sha256(master)
        self.refreeze(self.fixture.full_path, self.full)
        self.bundle['generation_request_plan']['sha256'] = run.sha256(self.fixture.full_path)
        self.refreeze(self.fixture.bundle_path, self.bundle)
        self.reference['sha256'] = run.sha256(self.fixture.bundle_path)
        self.complete()
        dataset = self.root / 'dataset.jsonl'; run.write_jsonl(dataset, self.dataset_rows)
        source = self.root / 'source'
        for relative in ('src/evaluate/helpers.py', run.ENTRY, 'infra/gpu03/direction_discovery/solver_diagnostic.py'):
            path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# authored source fixture\n')
        python = self.root / 'venv/bin/python'; python.parent.mkdir(parents=True); python.write_text('# authored runtime\n')
        (python.parent.parent / 'pyvenv.cfg').write_text('authored = true\n')
        from infra.gpu03.direction_discovery import supervisor, behavior_plan
        token = 'codex-eval-authored-test-bundle'
        with mock.patch.object(run, 'ROOT', self.root), mock.patch.object(run, 'PYTHON', str(python)), \
             mock.patch.object(run, 'host_identity'), mock.patch.object(run.importlib.metadata, 'version', return_value='fixture'), \
             mock.patch.object(behavior_plan, 'validate_master'), \
             mock.patch.object(supervisor, 'verify', side_effect=lambda p, d: self.proofs[str(p)]):
            built = run.build_manifest(stage=self.root/token, run_token=token, source_dir=source,
                experiment_plan=master, phase='behavior_untouched_test_bundle', request_plan=self.fixture.full_path,
                prepared_records=self.prepared, dataset=dataset, generation_packages=self.packages,
                workers=8, runtime_seconds=7200, test_bundle=self.reference)
            manifest = Path(built['manifest']); m = run.load_manifest(manifest, built['manifest_sha256'])
            self.assertEqual(m['scientific']['test_bundle'], self.reference)
            self.assertEqual(m['input_hashes']['test_bundle'], self.reference['sha256'])
            self.assertEqual([w['cpus'] for w in m['workers']], [[64+2*i, 65+2*i] for i in range(8)])
            self.assertEqual(m['limits']['systemd_runtime_seconds'], 7290)
            self.assertEqual(m['request_ids'], sorted(r['request_id'] for r in self.full['requests']))
            self.assertFalse((self.root/token/'control').exists()); self.assertFalse(Path(m['output']).exists())
            del m['scientific']['test_bundle']
            self.refreeze(manifest, m)
            with self.assertRaisesRegex(ValueError, 'Test bundle input provenance'):
                run.load_manifest(manifest, run.sha256(manifest))


class ManifestFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root/'original'
        for relative in ('src/evaluate/helpers.py', run.ENTRY, 'infra/gpu03/direction_discovery/solver_diagnostic.py'):
            p = self.source/relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# authored placeholder\n')
        self.plan = Path(__file__).resolve().parents[3] / 'artifacts/direction_discovery_review_20260907/plan_v1/experiment_plan.json'
        self.python = self.root/'venv/bin/python'; self.python.parent.mkdir(parents=True)
        self.python.write_text('# authored runtime fixture\n')
        (self.python.parent.parent/'pyvenv.cfg').write_text('fixture = true\n')
        self.patches = [mock.patch.object(run, 'ROOT', self.root), mock.patch.object(run, 'PYTHON', str(self.python)),
                        mock.patch.object(run, 'host_identity'), mock.patch.object(run.importlib.metadata, 'version', return_value='fixture')]
        for patch in self.patches:
            patch.start()
        self.addCleanup(self.tmp.cleanup)
        for patch in self.patches:
            self.addCleanup(patch.stop)

    def build(self, **extra):
        token = 'codex-eval-authored-test-20260907'
        parameters = dict(stage=self.root/token, run_token=token, source_dir=self.source,
                     experiment_plan=self.plan, phase='selftest', mode='authored_selftest', workers=2, runtime_seconds=180)
        parameters.update(extra)
        result = run.build_manifest(**parameters)
        self.path = Path(result['manifest']); self.digest = result['manifest_sha256']
        self.m = run.load_manifest(self.path, self.digest)
        return result

    def refreeze(self, m):
        self.path.chmod(0o600); self.path.write_text(run.canonical(m)+'\n'); self.path.chmod(0o400)
        self.digest = run.sha256(self.path)

    def install_worker_outputs(self):
        generated = run.unique_rows(run.read_jsonl(Path(self.m['stage'])/'input/generations.jsonl'), 'fixtures')
        for w in self.m['workers']:
            destination = Path(self.m['output'])/w['name']; (destination/'evaluation').mkdir(parents=True)
            rows = [evaluated(generated[k], correct=k.endswith('-0')) for k in w['request_ids']]
            run.write_jsonl(destination/'evaluation/records.jsonl', rows)
            inputs = Path(self.m['stage'])/'input'
            cfg = {'requests_sha256':run.sha256(inputs/w['input']), 'dataset_sha256':self.m['input_hashes']['dataset'],
                   'prepared_sha256':self.m['input_hashes']['prepared_records'], 'workers':2,'timeout_seconds':3,
                   'memory_per_worker_mib':1024,'training':False,'generation':False,
                   'implementation_sha256':self.m['source_files']['infra/gpu03/direction_discovery/evaluate.py']['sha256'],
                   'correctness_diagnostics':self.m['scientific']['correctness_diagnostics']}
            success = {'status':'succeeded','records':len(rows),'records_sha256':run.sha256(destination/'evaluation/records.jsonl'),'input_config':cfg}
            run.write_json(destination/'evaluation/SUCCESS.json',success)
            run.write_json(destination/'outer_receipt.json',{'status':'succeeded','run_token':self.m['run_token'],'worker_name':w['name'],
                         'returncode':0,'success_sha256':run.sha256(destination/'evaluation/SUCCESS.json')})


class BuilderTests(ManifestFixture):
    def test_diagnostic_policy_is_bound_and_cannot_drift(self):
        self.build()
        policy = self.m['scientific']['correctness_diagnostics']
        self.assertEqual(policy['policy'], 'solver_isolation_v1')
        self.assertEqual(policy['max_additional_calls_per_completion'], 1)
        changed = copy.deepcopy(self.m)
        changed['scientific']['correctness_diagnostics']['max_additional_calls_per_completion'] = 2
        self.refreeze(changed)
        with self.assertRaisesRegex(ValueError, 'policy drift'):
            run.load_manifest(self.path, self.digest)

    def test_missing_new_diagnostic_source_refuses_build(self):
        (self.source/'infra/gpu03/direction_discovery/solver_diagnostic.py').unlink()
        with self.assertRaisesRegex(ValueError, 'requires the isolated solver diagnostic'):
            self.build()

    def test_authored_builder_freezes_exact_sources_inputs_and_disjoint_shards(self):
        result = self.build()
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['maximum_evaluator_children'], 4)
        self.assertEqual([w['cpus'] for w in self.m['workers']], [[64,65],[66,67]])
        self.assertEqual(set().union(*(set(w['request_ids']) for w in self.m['workers'])), set(self.m['request_ids']))
        self.assertTrue(all(not p.stat().st_mode & 0o222 for p in (Path(self.m['stage'])/'input').iterdir()))
        with self.assertRaisesRegex(ValueError, 'fresh exact-token'):
            self.build()

    def test_changed_input_or_added_source_fails(self):
        self.build()
        path = Path(self.m['stage'])/'source/infra/injected.py'; path.write_text('# unexpected\n'); path.chmod(0o400)
        with self.assertRaisesRegex(ValueError, 'Source inventory'):
            run.load_manifest(self.path, self.digest)

    def test_wrong_manifest_hash_and_dangerous_cpu_or_command_fail(self):
        self.build()
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            run.load_manifest(self.path, '0'*64)
        for change in ('cpu', 'command', 'memory', 'full_cluster_omission'):
            m = copy.deepcopy(self.m)
            if change == 'cpu': m['workers'][0]['cpus'] = [96,97]
            if change == 'command': m['workers'][0]['python_args'] = ['-c','print(1)']
            if change == 'memory': m['limits']['memory_max_bytes'] = 64 * run.GIB
            if change == 'full_cluster_omission':
                m['request_ids'] = m['request_ids'][:1]; m['workers'] = m['workers'][:1]
            self.refreeze(m)
            with self.assertRaises(ValueError):
                run.load_manifest(self.path, self.digest)

    def test_source_symlink_refused(self):
        (self.source/'src/escape').symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.build()

    def test_v2_evaluation_freezes_exact_parent_without_changing_evaluation_science(self):
        from infra.gpu03.direction_discovery.test_behavior_plan import amended_master
        value,digest=amended_master()
        path=self.root/'v2.json';path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
        result=self.build(experiment_plan=path,parent_experiment_plan=self.plan)
        self.assertEqual(self.m['scientific']['master_plan_sha256'],digest)
        self.assertEqual(self.m['scientific']['parent_plan_sha256'],run.PLAN_SHA256)
        self.assertEqual(self.m['scientific']['timeout_seconds'],3)
        self.assertTrue((Path(self.m['stage'])/'input/parent_experiment_plan.json').exists())
        self.assertEqual(result['requests'],2)

    def test_v2_without_exact_parent_fails_before_creating_stage(self):
        from infra.gpu03.direction_discovery.test_behavior_plan import amended_master
        value,_=amended_master()
        path=self.root/'v2.json';path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
        with self.assertRaisesRegex(ValueError,'exact supplied version1'):
            self.build(experiment_plan=path)
        self.assertFalse((self.root/'codex-eval-authored-test-20260907').exists())


class ProducerTests(ManifestFixture):
    def setUp(self):
        super().setUp()
        p = self.source/'infra/gpu03/direction_discovery/evaluate.py'; p.write_text('# authored evaluator hash\n')
        self.build()

    def test_worker_merge_independent_verifier_success_path_and_tampering(self):
        self.install_worker_outputs()
        summary = run.merge_evaluations(self.m)
        self.assertTrue(summary['exact_request_coverage'])
        self.assertEqual(summary['records'],2)
        output = Path(self.m['output'])
        diagnostic = run.read_json(output/'correctness_diagnostics.json')
        self.assertFalse(diagnostic['legacy_metrics_and_gates_changed'])
        run.write_json(output/'producer_summary.json', {**summary,'status':'succeeded','run_token':self.m['run_token'],'manifest_sha256':self.digest})
        run.write_json(output/'artifact_manifest.json',run.produced_manifest(output))
        proof = run.verify_produced(self.path,self.digest)
        self.assertEqual(proof['status'],'verified')
        (output/'evaluations.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'inventory/hash mismatch'):
            run.verify_produced(self.path,self.digest)

    def test_worker_config_cannot_silently_omit_diagnostic(self):
        self.install_worker_outputs()
        w = self.m['workers'][0]
        root = Path(self.m['output'])/w['name']
        success_path, receipt_path = root/'evaluation/SUCCESS.json', root/'outer_receipt.json'
        success = run.read_json(success_path)
        del success['input_config']['correctness_diagnostics']
        success_path.write_text(run.canonical(success) + '\n')
        receipt = run.read_json(receipt_path)
        receipt['success_sha256'] = run.sha256(success_path)
        receipt_path.write_text(run.canonical(receipt) + '\n')
        with self.assertRaisesRegex(ValueError, 'diagnostic policy/source drift'):
            run.verify_worker(self.m, w)

    def test_missing_outer_receipt_or_extra_evaluation_fails(self):
        self.install_worker_outputs()
        w = self.m['workers'][0]
        receipt = Path(self.m['output'])/w['name']/'outer_receipt.json'
        receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            run.verify_worker(self.m,w)

    def test_failed_producer_cannot_verify_even_with_manifest(self):
        Path(self.m['output']).mkdir()
        run.write_json(Path(self.m['output'])/'FAILURE.json',{})
        with self.assertRaisesRegex(ValueError,'Failed evaluation'):
            run.verify_produced(self.path,self.digest)


class LaunchTests(ManifestFixture):
    def test_clean_environment_never_inherits_credentials(self):
        with mock.patch.dict(os.environ, {'AWS_SECRET_ACCESS_KEY':'authored secret', 'HF_TOKEN':'authored secret'}):
            env = run.cpu_environment()
            self.assertNotIn('AWS_SECRET_ACCESS_KEY',env)
            self.assertNotIn('HF_TOKEN',env)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(run.clean_command(['python'])[:2], ['/usr/bin/env','-i'])

    def test_systemd_failure_after_launch_stops_only_exact_unit(self):
        self.build()
        with mock.patch.object(run,'resource_check',return_value={}), mock.patch.object(run,'service_state',return_value={'ActiveState':'inactive'}), \
             mock.patch.object(run.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='',stderr='')) as call:
            with self.assertRaisesRegex(ValueError,'not positively verified'):
                run.launch(self.path,self.digest)
        self.assertEqual(call.call_args_list[-1].args[0],['systemctl','--user','stop',self.m['run_token']+'.service'])
        start = call.call_args_list[0].args[0]
        self.assertIn('--property=RuntimeMaxSec=270',start)
        self.assertIn('--property=MemoryMax='+str(32*run.GIB),start)
        self.assertIn('/usr/bin/env',start)
        self.assertTrue(any('ExecStopPost=/usr/bin/env -i' in part for part in start))

    def test_successful_service_dispatch_binds_invocation(self):
        self.build()
        state = {'ActiveState':'active','SubState':'running','MainPID':'123','ControlGroup':'/user.slice/'+self.m['run_token']+'.service',
                 'RuntimeMaxUSec':'4min 30s','MemoryMax':str(32*run.GIB),'TasksMax':'512','KillMode':'control-group','InvocationID':'a'*32}
        with mock.patch.object(run,'resource_check',return_value={}), mock.patch.object(run,'service_state',return_value=state), \
             mock.patch.object(run.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='',stderr='')):
            result = run.launch(self.path,self.digest)
        self.assertEqual(result['status'],'launched')
        self.assertEqual(run.read_json(Path(self.m['stage'])/'control/service_started.json'),state)

    def test_unknown_or_failed_exit_never_verifies(self):
        self.build(); control=Path(self.m['stage'])/'control'; control.mkdir()
        run.write_json(control/'supervisor_exit.json',{'run_token':self.m['run_token'],'manifest_sha256':self.digest,
                       'service_result':'timeout','exit_code_kind':'killed','exit_status':'15','invocation_id':'a'*32,
                       'success_present':True,'failure_present':False})
        run.write_json(control/'service_started.json',{'InvocationID':'a'*32})
        with self.assertRaisesRegex(ValueError,'not successful'):
            run.verify(self.path,self.digest)


class RecoveryTests(ManifestFixture):
    def setUp(self):
        super().setUp()
        self.build()
        stage = Path(self.m['stage'])
        control = stage/'control'; control.mkdir()
        self.started = {'InvocationID':'a'*32,'ControlGroup':'/authored-absent-cgroup/'+self.m['run_token']+'.service'}
        self.terminal = {'run_token':self.m['run_token'],'manifest_sha256':self.digest,'service_result':'timeout',
                         'exit_code_kind':'killed','exit_status':'TERM','invocation_id':'a'*32}
        run.write_json(control/'supervisor_exit.json',self.terminal)
        run.write_json(control/'service_started.json',self.started)
        self.generated = run.unique_rows(run.read_jsonl(stage/'input/generations.jsonl'),'fixture')
        journal = stage/'results/worker_00/evaluation/records.jsonl'; journal.parent.mkdir(parents=True)
        self.completed = evaluated(self.generated['authored-evaluation-0'])
        run.write_jsonl(journal,[self.completed])
        journals = {str(journal.relative_to(stage)):run.info(journal), 'input/recovered.jsonl':run.info(stage/'input/recovered.jsonl')}
        self.recovery = {'mode':'reuse_completed_without_reevaluation','input_hashes':self.m['input_hashes'],
                         'prior_runs':[{'manifest':str(self.path),'manifest_sha256':self.digest,'journals':journals}]}
        self.recovery_path = self.root/'recovery.json'
        run.write_json(self.recovery_path,self.recovery)

    def recover(self, state=None):
        with mock.patch.object(run,'service_state',return_value=state or {'ActiveState':'failed','MainPID':'0'}):
            return run.recovered_rows(self.recovery_path,self.generated,self.m['input_hashes'],Path(self.m['stage'])/'source')

    def test_exact_completed_record_reused_once_without_running_evaluator(self):
        with mock.patch.object(run.sandbox,'run_outer') as execute:
            rows, plan=self.recover()
        self.assertEqual(rows,[self.completed]); self.assertEqual(plan,self.recovery)
        execute.assert_not_called()

    def test_missing_exit_metadata_and_unknown_service_state_fail(self):
        for field in ('exit_status','invocation_id','exit_code_kind'):
            changed=dict(self.terminal); changed.pop(field)
            (Path(self.m['stage'])/'control/supervisor_exit.json').write_text(run.canonical(changed)+'\n')
            with self.assertRaisesRegex(ValueError,'terminal independent receipt'):
                self.recover()
        (Path(self.m['stage'])/'control/supervisor_exit.json').write_text(run.canonical(self.terminal)+'\n')
        with self.assertRaisesRegex(ValueError,'not positively terminal'):
            self.recover({'ActiveState':'unknown','MainPID':'0'})

    def test_unbound_or_changed_journal_fails_before_reuse(self):
        journal=Path(self.m['stage'])/'results/worker_00/evaluation/records.jsonl'
        journal.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'journal hash changed'):
            self.recover()

    def test_runtime_change_blocks_recovery(self):
        with mock.patch.object(run.importlib.metadata,'version',return_value='different'):
            with self.assertRaisesRegex(ValueError,'runtime changed'):
                self.recover()

    def test_systemd_signal_names_and_decimal_exit_codes(self):
        for kind,value in (('killed','TERM'),('dumped','ABRT'),('exited','0'),('exited','255')):
            self.assertTrue(run.valid_exit_status({'exit_code_kind':kind,'exit_status':value}))
        for kind,value in (('killed','15'),('exited','TERM'),('exited','256'),('dumped','UNKNOWN'),('killed',None)):
            self.assertFalse(run.valid_exit_status({'exit_code_kind':kind,'exit_status':value}))


class ProcessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Linux process identity qualification')
    def test_real_authored_process_group_cleanup_preserves_other_process(self):
        own = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], start_new_session=True)
        other = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], start_new_session=True)
        try:
            identity=run.process_info(own.pid)
            run.terminate_owned([(own,identity)])
            self.assertIsNotNone(own.poll())
            self.assertIsNone(other.poll())
        finally:
            for p in (own,other):
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()

    def test_pid_reuse_refuses_signal(self):
        p=types.SimpleNamespace(pid=123,poll=lambda:None)
        with mock.patch.object(run,'same_process',return_value=False),mock.patch.object(run.os,'killpg') as kill:
            with self.assertRaisesRegex(ValueError,'identity changed'):
                run.terminate_owned([(p,{'pid':123,'start_ticks':1})])
            kill.assert_not_called()


def authored_remote(package, manifest, part, *, save):
    """Transport fixture: real copied bytes, injected remote admission verifier."""
    m = copy.deepcopy(manifest)
    token = m['run_token']; result_items, tasks, file_map = [], [], {}
    for worker in m['workers']:
        old_path = Path(worker['command'][3]); task = run.read_json(old_path)
        result = Path(task['output'])/'results.jsonl'
        result_items.append({'worker': worker['name'], 'path': result, **run.info(result)})
        if 'prepared_records' in task:
            source = Path(task['prepared_records'])
            task['prepared_records'] = '/unmounted-gpu02/'+token+'/prepared.jsonl'
            file_map[task['prepared_records']] = {'path': source, **run.info(source)}
        task['output'] = '/unmounted-gpu02/'+token+'/'+worker['name']
        worker['command'][3] = '/unmounted-gpu02/'+token+'/'+worker['name']+'.json'
        tasks.append({'worker': worker, 'task': task, 'task_path': old_path})
    replica_path = save(Path(package['manifest']).parent/'authored-replica.json', {'authored': token})
    ref = {'kind': 'remote_generation', 'replica': {'path': str(replica_path), 'sha256': run.sha256(replica_path)}}
    m.update(host='gpu-02', stage='/unmounted-gpu02/'+token, output='/unmounted-gpu02/'+token+'/output')
    proof = {'status': 'verified', 'gpu_release_verified': True, 'run_token': token,
             'manifest_sha256': package['manifest_sha256'], 'host': 'gpu-02', 'replica_sha256': ref['replica']['sha256']}
    context = {'manifest': m, 'manifest_sha256': package['manifest_sha256'], 'authority_manifest_sha256': package['manifest_sha256'],
        'artifact_manifest_sha256': package['artifact_manifest_sha256'], 'request_plan': part,
        'prepared_records_sha256': next(iter(file_map.values()))['sha256'] if file_map else None,
        'scientific_conditions': part['conditions'], 'tasks': tasks, 'results': result_items,
        'proof': proof, 'file_bindings': file_map, 'bindings': [replica_path]}
    return ref, context


class RemoteTestCollectionTests(unittest.TestCase):
    def setUp(self):
        from infra.gpu03.direction_discovery import remote_generation
        self.f = TestBundlePackageTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.contexts = {}
        self.verifier = mock.patch.object(remote_generation, 'context', side_effect=lambda p, **kw: self.contexts[run.generation_package_key(p)]).start()

    def complete(self, remote_indices=(0, 1, 2)):
        for index in range(3):
            previous = {**self.f.fixture.previous, 'generations': 2526 + 5*index,
                        'untouched_test_generations': 5*index, 'untouched_test_requests': 5*index}
            part = self.f.e.make_part(self.f.reference, index, previous, self.f.packages)
            local = self.f.package(part, 'remote' + str(index))
            if index not in remote_indices: continue
            ref, context = authored_remote(local, self.f.manifests[local['manifest']], part, save=self.f.fixture.save)
            self.contexts[run.generation_package_key(ref)] = context
            self.f.packages[-1] = ref

    def collect(self, packages=None):
        with mock.patch('infra.gpu03.direction_discovery.supervisor.verify', side_effect=lambda p, d: self.f.proofs[str(p)]):
            return run.collect_generation_packages(self.f.packages if packages is None else packages,
                request_plan=self.f.full, master_sha=self.f.full['master_plan_sha256'],
                test_bundle=self.f.reference, prepared_rows=self.f.prepared_rows)

    def test_complete_remote_bundle_preserves_provenance_and_stored_proof_replay(self):
        self.complete(); rows, proofs = self.collect()
        self.assertEqual(len(rows), 15); self.assertTrue(all(p['kind'] == 'remote_generation' for p in proofs))
        self.assertEqual([p['replica'] for p in proofs], [p['replica'] for p in self.f.packages])
        contexts = run.test_generation_metadata(proofs, request_plan=self.f.full, test_bundle=self.f.reference,
            master_sha=self.f.full['master_plan_sha256'], stored_proofs=proofs)
        self.assertEqual(len(contexts), 3)

    def test_mixed_host_bundle_keeps_exact_predecessor_chain(self):
        self.complete(remote_indices=(1,)); rows, proofs = self.collect()
        self.assertEqual(len(rows), 15)
        self.assertEqual([p.get('kind', 'local') for p in proofs], ['local', 'remote_generation', 'local'])
        contexts = run.test_generation_metadata(proofs, request_plan=self.f.full, test_bundle=self.f.reference,
            master_sha=self.f.full['master_plan_sha256'], stored_proofs=proofs)
        self.assertEqual(len(contexts), 3)

    def test_gpu01_gpu02_complete_bundle_keeps_host_bound_proofs(self):
        self.complete()
        first = self.contexts[run.generation_package_key(self.f.packages[0])]
        first['manifest']['host'] = first['proof']['host'] = 'gpu-01'
        rows, proofs = self.collect()
        self.assertEqual(len(rows), 15)
        self.assertEqual([p['verification']['host'] for p in proofs], ['gpu-01', 'gpu-02', 'gpu-02'])

    def test_cross_host_test_proof_rejected_before_any_rows(self):
        self.complete()
        first = self.contexts[run.generation_package_key(self.f.packages[0])]
        first['manifest']['host'] = 'gpu-01'
        with mock.patch.object(run, 'replica_rows', side_effect=AssertionError('Early test rows')):
            with self.assertRaisesRegex(ValueError, 'terminal release'): self.collect()

    def test_last_release_failure_and_missing_piece_precede_all_row_reads(self):
        self.complete()
        with mock.patch.object(run, 'replica_rows', side_effect=AssertionError('Early rows')):
            with self.assertRaisesRegex(ValueError, 'all fixed test'): self.collect(self.f.packages[:2])
            self.contexts[run.generation_package_key(self.f.packages[-1])]['proof']['gpu_release_verified'] = False
            with self.assertRaisesRegex(ValueError, 'terminal release'): self.collect()

    def test_remote_snapshot_mutation_rejected_without_opening_remote_paths(self):
        self.complete()
        path = self.contexts[run.generation_package_key(self.f.packages[-1])]['results'][0]['path']
        before = path.read_bytes(); path.write_bytes(before.replace(b'authored fixture', b'modified fixture'))
        with self.assertRaisesRegex(ValueError, 'snapshot hash'): self.collect()

    def test_stored_remote_proof_and_source_hash_drift_rejected(self):
        self.complete(); _, proofs = self.collect()
        changed = copy.deepcopy(proofs); changed[0]['artifact_manifest_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'Stored remote generation'):
            run.test_generation_metadata(changed, request_plan=self.f.full, test_bundle=self.f.reference,
                master_sha=self.f.full['master_plan_sha256'], stored_proofs=changed)
        self.contexts[run.generation_package_key(self.f.packages[0])]['prepared_records_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'prepared union'): self.collect()

    def test_remote_request_sampling_and_predecessor_chain_drift_rejected(self):
        self.complete(); key = run.generation_package_key(self.f.packages[-1]); original = copy.deepcopy(self.contexts[key])
        for kind in ('sampling', 'request', 'predecessor'):
            self.contexts[key] = copy.deepcopy(original)
            if kind == 'sampling': self.contexts[key]['tasks'][0]['task']['sampling'] = {}
            elif kind == 'request': self.contexts[key]['tasks'][0]['task']['requests'].pop()
            else:
                self.contexts[key]['request_plan']['test_execution']['predecessors'].reverse()
                self.contexts[key]['manifest']['scientific']['test_execution'] = self.contexts[key]['request_plan']['test_execution']
            with self.subTest(kind=kind), self.assertRaises((RuntimeError, ValueError)): self.collect()


class RemoteRecoveredFinalistTests(unittest.TestCase):
    def setUp(self):
        from infra.gpu03.direction_discovery import test_execution_partition as fixtures, remote_generation
        self.f = fixtures.RecoveryIntegrationTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.recovery_package(); local, m, _ = self.f.second_package()
        self.ref, self.context = authored_remote(local, m, self.f.second(), save=self.f.f.save)
        self.verifier = mock.patch.object(remote_generation, 'context', side_effect=lambda p, **kw: self.context).start()
        self.addCleanup(mock.patch.stopall)

    def collect(self, packages=None):
        return run.collect_generation_packages([self.f.logical_ref, self.ref] if packages is None else packages,
            request_plan=self.f.f.full, master_sha=self.f.f.full['master_plan_sha256'])

    def test_full740_local_logical_plus_remote_round1_and_stored_proof_replay(self):
        rows, proofs = self.collect()
        self.assertEqual(len(rows), 740); self.assertEqual(len({r['request_id'] for r in rows}), 740)
        self.assertEqual(proofs[1]['kind'], 'remote_generation'); self.assertEqual(proofs[1]['replica'], self.ref['replica'])
        result = run.recovered_finalist_metadata(proofs, request_plan=self.f.f.full,
            master_sha=self.f.f.full['master_plan_sha256'], stored_proofs=proofs)
        self.assertEqual(result['reference'], self.f.logical_ref)

    def test_gpu01_full740_collection_and_proof_replay(self):
        self.context['manifest']['host'] = self.context['proof']['host'] = 'gpu-01'
        rows, proofs = self.collect()
        self.assertEqual(len(rows), 740); self.assertEqual(proofs[1]['verification']['host'], 'gpu-01')
        context = run.recovered_finalist_metadata(proofs, request_plan=self.f.f.full,
            master_sha=self.f.f.full['master_plan_sha256'], stored_proofs=proofs)
        self.assertEqual(context['remote']['manifest']['host'], 'gpu-01')

    def test_cross_host_finalist_proof_rejected_before_any_rows(self):
        for host, proof_host in [('gpu-01', 'gpu-02'), ('gpu-02', 'gpu-01'), ('gpu-03', 'gpu-03')]:
            self.context['manifest']['host'] = host; self.context['proof']['host'] = proof_host
            with self.subTest(host=host, proof_host=proof_host), mock.patch.object(run, 'read_jsonl', side_effect=AssertionError('Early logical rows')), mock.patch.object(run, 'replica_rows', side_effect=AssertionError('Early remote rows')):
                with self.assertRaisesRegex(ValueError, 'terminal release'): self.collect()

    def test_failed_remote_release_precedes_logical_and_remote_outcomes(self):
        self.context['proof']['gpu_release_verified'] = False
        with mock.patch.object(run, 'read_jsonl', side_effect=AssertionError('Early logical rows')), mock.patch.object(run, 'replica_rows', side_effect=AssertionError('Early remote rows')):
            with self.assertRaisesRegex(ValueError, 'terminal release'): self.collect()

    def test_remote_only_or_wrong_logical_predecessor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete test bundle'): self.collect([self.ref])
        self.context['request_plan']['execution_partition']['predecessor'] = {}
        with self.assertRaises((RuntimeError, ValueError)): self.collect()

    def test_remote_round1_missing_or_changed_row_rejected(self):
        path = self.context['results'][0]['path']; before = path.read_bytes()
        path.write_bytes(before.split(b'\n', 1)[1])
        with self.assertRaisesRegex(ValueError, 'size changed'): self.collect()


if __name__ == '__main__':
    unittest.main()
