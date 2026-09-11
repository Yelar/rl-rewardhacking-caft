"""Authored planning metadata only; model, services and scientific replay are mocked."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from infra.gpu03.direction_discovery import behavior_plan, execution_partition as e, eval_run
from infra.gpu03.direction_discovery.test_behavior_plan import MASTER, records, selected


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.previous = {'generations': 1786, 'teacher_forced': 9222, 'untouched_test_generations': 0,
                         'untouched_test_requests': 0, 'phase_manifests': []}
        self.full = behavior_plan.build_request_plan(records(), MASTER, behavior_plan.PLAN_SHA256,
            selected(layer=21, index=4), phase='finalist_validation', previous_counts=self.previous)
        self.full['source_bindings'] = {}
        self.full_path = self.save(self.root / 'full.json', self.full)
        self.part = e.make_part(self.full_path, e.sha(self.full_path), 0, self.previous)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(e.canonical(value)); path.chmod(0o400)
        return path

    def ledger(self, gen=1786, phases=()):
        return {'generation_requests': gen, 'tf_requests': 9222, 'untouched_test_requests': 0, 'phases': list(phases)}

    def package(self, part, name='round0', terminal=True):
        stage = self.root / ('codex-discovery-' + name + '-stage')
        output = stage / 'results'; token = stage.name.removesuffix('-stage')
        part_path = self.save(stage / 'input/request_plan.json', part)
        tasks, workers = [], []
        for i in range(3):
            requests = part['requests'][i::3]
            out = output / f'worker{i}'
            task = {'mode': 'generate', 'run_token': token, 'worker_name': f'gpu_{i}',
                    'conditions': part['conditions'], 'requests': requests, 'output': str(out)}
            path = self.save(stage / f'tasks/gpu_{i}.json', task); tasks.append(path)
            out.mkdir(parents=True)
            (out / 'results.jsonl').write_text(''.join(json.dumps({**r, 'result': {'authored': True}}) + '\n' for r in requests))
            workers.append({'name': f'gpu_{i}', 'command': ['python', 'engine.py', '--task', str(path)],
                            'success_expect': {'mode': 'generate', 'requests': len(requests)}})
        artifact = self.save(output / 'artifact_manifest.json', {'authored': True})
        bindings = [part_path, *tasks, *e.predecessor_bindings(part)]
        m = {'host': 'gpu-04', 'run_token': token, 'stage': str(stage), 'output': str(output),
             'scientific': {'master_plan_sha256': part['master_plan_sha256'], 'training': False,
                            'execution_partition': part['execution_partition']},
             'workers': workers, 'limits': {'systemd_runtime_seconds': 14580},
             'bound_files': {str(p): {'sha256': e.sha(p), 'size_bytes': p.stat().st_size} for p in bindings}}
        manifest = self.save(stage / 'reviewed_manifest.json', m)
        digest = e.sha(manifest)
        proof = {'status': 'verified', 'manifest_sha256': digest, 'run_token': token,
                 'workers': 3, 'artifact_files': 1, 'gpu_release_verified': True}
        proof_path = self.save(stage / 'independent_verification.json', proof)
        if terminal:
            identity = {'run_token': token, 'manifest_sha256': digest}
            self.save(stage / 'control/launch_intent.json', {**identity, 'at': 100.})
            self.save(stage / 'control/service_started.json', {'at': 101., 'unit': token + '.service',
                'fields': {'ActiveState': 'active', 'SubState': 'running', 'MainPID': '12', 'KillMode': 'control-group',
                           'ControlGroup': '/user.slice/' + token + '.service', 'InvocationID': 'a' * 32}})
            self.save(stage / 'control/supervisor_exit.json', {**identity, 'at': 200., 'service_result': 'success',
                'exit_code_kind': 'exited', 'exit_status': '0', 'invocation_id': 'a' * 32,
                'producer_summary_present': True, 'failure_present': False})
        ref = {'manifest': str(manifest), 'manifest_sha256': digest, 'artifact_manifest_sha256': e.sha(artifact),
               'verification': str(proof_path), 'verification_sha256': e.sha(proof_path)}
        return ref, m, proof

    def second(self, ref):
        return e.make_part(self.full_path, e.sha(self.full_path), 1,
                           {**self.previous, 'generations': 2156}, ref)

    def test_two_rounds_exact_disjoint_full_coverage_and_seeds(self):
        ref, _, _ = self.package(self.part)
        second = self.second(ref)
        e.validate_part(second); e.predecessor_bindings(second)
        a, b = self.part['requests'], second['requests']
        self.assertEqual(len(a), 370); self.assertEqual(len(b), 370)
        self.assertFalse({r['request_id'] for r in a} & {r['request_id'] for r in b})
        self.assertEqual({r['request_id']: r for r in a+b}, {r['request_id']: r for r in self.full['requests']})
        self.assertEqual([len(a[i::3]) for i in range(3)], [124, 123, 123])
        for part in (self.part, second):
            self.assertEqual(len({r['problem_id'] for r in part['requests']}), 37)
            self.assertEqual(len({r['condition_id'] for r in part['requests']}), 5)

    def test_incomplete_duplicate_reordered_or_changed_partition_fails(self):
        for mutation in ('missing', 'duplicate', 'reorder', 'seed', 'condition', 'sample'):
            part = copy.deepcopy(self.part)
            if mutation == 'missing': part['requests'].pop()
            elif mutation == 'duplicate': part['requests'][-1] = part['requests'][0]
            elif mutation == 'reorder': part['requests'].reverse()
            elif mutation == 'seed': part['requests'][0]['seed'] += 1
            elif mutation == 'condition': part['requests'][0]['condition_id'] = 'baseline'
            else: part['requests'][0]['sample_index'] = 2
            if mutation == 'condition': part['requests'][1]['condition_id'] = 'baseline'
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError): e.validate_part(part)

    def test_no_scientific_field_or_unknown_field_mutation(self):
        for key, value in (('sampling', {}), ('generation_contract', {}), ('phase', 'screening'), ('new_field', True)):
            part = copy.deepcopy(self.part); part[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): e.validate_part(part)

    def test_missing_future_predecessor_and_wrong_sample_indices_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'requires a verified'): self.second(None)
        part = copy.deepcopy(self.part); part['execution_partition']['sample_indices'] = [0, 2]
        with self.assertRaises(RuntimeError): e.validate_part(part)

    def test_mutable_or_changed_full_plan_fails(self):
        self.full_path.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, 'immutable'): e.validate_part(self.part)
        self.full_path.write_text('{}'); self.full_path.chmod(0o400)
        with self.assertRaisesRegex(RuntimeError, 'hash changed'): e.validate_part(self.part)

    def test_full_plan_missing_cell_or_changed_stable_seed_fails(self):
        for change in ('missing', 'seed', 'source', 'order', 'random'):
            full = copy.deepcopy(self.full)
            if change == 'missing': full['requests'].pop()
            elif change == 'seed': full['requests'][0]['seed'] += 1
            elif change == 'source': full['requests'][0]['record_id'] = 'other'
            elif change == 'order': full['requests'].reverse()
            else: full['conditions']['random:L21r1:base6101']['layers'][0]['seed'] += 1
            with self.subTest(change=change), self.assertRaises(RuntimeError): e.validate_full(full)

    def test_stale_ledger_and_test_commitment_fail(self):
        for ledger in (self.ledger(1787), {**self.ledger(), 'tf_requests': 9223},
                       {**self.ledger(), 'untouched_test_requests': 1}):
            with self.assertRaisesRegex(RuntimeError, 'stale'): e.validate_against_ledger(self.part, ledger)
        e.validate_against_ledger(self.part, self.ledger())

    def test_replay_even_with_refrozen_full_digest_fails(self):
        phase = {'generation_request_ids': [self.part['requests'][0]['request_id']]}
        with self.assertRaisesRegex(RuntimeError, 'already committed'): e.validate_against_ledger(self.part, self.ledger(phases=[phase]))

    def test_second_round_requires_terminal_predecessor_in_ledger(self):
        ref, _, _ = self.package(self.part); second = self.second(ref)
        phase = {'execution_partition': self.part['execution_partition'], 'manifest_sha256': ref['manifest_sha256'],
                 'generation_requests': 370, 'wall_basis': 'actual_launch_to_terminal_receipt'}
        e.validate_against_ledger(second, self.ledger(2156, [phase]))
        for change in ({'wall_basis': 'full_reserved_deadline'}, {'manifest_sha256': 'b'*64}):
            with self.assertRaises(RuntimeError): e.validate_against_ledger(second, self.ledger(2156, [{**phase, **change}]))
        with self.assertRaises(RuntimeError): e.validate_against_ledger(second, self.ledger(2156))

    def test_fresh_predecessor_verify_is_required_and_bound(self):
        from infra.gpu03.direction_discovery import supervisor
        ref, _, proof = self.package(self.part); second = self.second(ref)
        with patch.object(supervisor, 'verify', return_value=proof) as check:
            e.predecessor_bindings(second, verify=True); check.assert_called_once()
        with patch.object(supervisor, 'verify', return_value={**proof, 'gpu_release_verified': False}):
            with self.assertRaisesRegex(RuntimeError, 'Fresh independent'): e.predecessor_bindings(second, verify=True)

    def test_failed_or_changed_predecessor_proof_rejected(self):
        ref, _, proof = self.package(self.part)
        proof['status'] = 'failed'
        new_path = self.save(self.root / 'failed.json', proof)
        ref.update(verification=str(new_path), verification_sha256=e.sha(new_path))
        with self.assertRaisesRegex(RuntimeError, 'terminal success'): e.predecessor_bindings(self.second(ref))

    def test_ledger_counts_370_once_then740_and_keeps_failed_reservation(self):
        from infra.gpu03.direction_discovery import phase_budget
        ref, _, _ = self.package(self.part)
        ledger = phase_budget.account(self.root, behavior_plan.PLAN_SHA256)
        self.assertEqual(ledger['generation_requests'], 370)
        self.assertEqual(ledger['gpu_phase_wall_seconds'], 100.)
        self.package(self.second(ref), name='round1', terminal=False)
        ledger = phase_budget.account(self.root, behavior_plan.PLAN_SHA256)
        self.assertEqual((ledger['generation_requests'], ledger['gpu_phase_wall_seconds']), (740, 14680.))
        self.assertEqual({p['execution_partition']['round_index'] for p in ledger['phases']}, {0, 1})

    def test_ledger_duplicate_round_and_overlapping_requests_fail(self):
        from infra.gpu03.direction_discovery import phase_budget
        self.package(self.part)
        self.package(self.part, name='replayed')
        with self.assertRaisesRegex(RuntimeError, 'already committed'): phase_budget.account(self.root, behavior_plan.PLAN_SHA256)

    def test_ledger_round_one_without_terminal_round_zero_fails(self):
        from infra.gpu03.direction_discovery import phase_budget
        ref, _, _ = self.package(self.part, terminal=False)
        self.package(self.second(ref), name='round1')
        with self.assertRaisesRegex(RuntimeError, 'terminal first-round'): phase_budget.account(self.root, behavior_plan.PLAN_SHA256)

    def test_builder_publication_rejects_stale_partition_before_reservation(self):
        from infra.gpu03.direction_discovery import build_phase
        stage = self.root / 'publication'; self.save(stage / 'input/request_plan.json', self.part)
        manifest = {'stage': str(stage), 'gpu_ids': [5, 6, 7], 'scientific': {'master_plan_sha256': behavior_plan.PLAN_SHA256,
                    'previous_phase_budget': self.ledger(), 'execution_partition': self.part['execution_partition']}}
        with patch.object(build_phase, 'validate_master', return_value=MASTER), \
             patch('phase_budget.account', return_value=self.ledger(1787)), \
             patch.object(build_phase, 'effective_budget', return_value=(self.ledger(1787), 100.)):
            with self.assertRaisesRegex(RuntimeError, 'stale'): build_phase.validate_publication_budget(manifest)

    def test_write_part_preserves_existing_path_and_rescans_stale_ledger(self):
        previous = {**self.previous, 'phase_budget': self.ledger(), 'ledger_bindings': []}
        with patch.object(behavior_plan, 'counts_from_phase_ledger', side_effect=[previous, {**previous, 'generations': 1787}]):
            with self.assertRaisesRegex(RuntimeError, 'changed during preparation'):
                e.write_part(full_plan=self.full_path, full_plan_sha256=e.sha(self.full_path), round_index=0,
                             phase_root=self.root, output=self.root / 'not-written')
        self.assertFalse((self.root / 'not-written').exists())
        with patch.object(behavior_plan, 'counts_from_phase_ledger', return_value=previous):
            result = e.write_part(full_plan=self.full_path, full_plan_sha256=e.sha(self.full_path), round_index=0,
                                  phase_root=self.root, output=self.root / 'prepared')
        self.assertFalse(result['budget_reserved'])
        self.assertFalse(list(self.root.glob('codex-discovery-*-stage/reviewed_manifest.json')))
        with self.assertRaisesRegex(RuntimeError, 'Preserve earlier'):
            e.write_part(full_plan=self.full_path, full_plan_sha256=e.sha(self.full_path), round_index=0,
                         phase_root=self.root, output=self.root / 'prepared')

    def collect(self, refs, manifests, proofs, full=None):
        from infra.gpu03.direction_discovery import supervisor
        by_path = {ref['manifest']: (m, proof) for ref, m, proof in zip(refs, manifests, proofs)}
        with patch.object(supervisor, 'verify', side_effect=lambda p, d: by_path[str(p)][1]), \
             patch.object(supervisor, 'load_manifest', side_effect=lambda p, d: by_path[str(p)][0]):
            return eval_run.collect_generation_packages(refs, request_plan=self.full if full is None else full,
                                                         master_sha=behavior_plan.PLAN_SHA256)

    def test_evaluation_combines_both_complete_packages_against_full740(self):
        a = self.package(self.part); b = self.package(self.second(a[0]), name='round1')
        rows, proofs = self.collect([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
        self.assertEqual(len(rows), 740); self.assertEqual(len(proofs), 2)
        prepared = [{'record_id': r} for r in set(row['record_id'] for row in rows)]
        dataset = [{'id': p} for p in self.full['selected_problem_ids']]
        with patch.object(eval_run.evaluate, 'validate_generation') as validator:
            self.assertEqual(len(eval_run.validate_inputs(rows, self.full, prepared, dataset)), 740)
            self.assertEqual(validator.call_count, 740)

    def test_evaluation_rejects_first_round_only_and_partial_plan(self):
        a = self.package(self.part)
        with self.assertRaisesRegex(ValueError, 'both disjoint'): self.collect([a[0]], [a[1]], [a[2]])
        with self.assertRaisesRegex(ValueError, 'full scientific plan'): self.collect([a[0]], [a[1]], [a[2]], full=self.part)
        with self.assertRaisesRegex(ValueError, 'full scientific plan'): eval_run.validate_inputs([], self.part, [], [])

    def test_evaluation_rejects_one_missing_saved_request(self):
        a = self.package(self.part); b = self.package(self.second(a[0]), name='round1')
        path = Path(b[1]['workers'][0]['command'][3]); task = json.loads(path.read_text())
        rows_path = Path(task['output']) / 'results.jsonl'
        rows_path.write_text(''.join(rows_path.read_text().splitlines(keepends=True)[:-1]))
        with self.assertRaisesRegex(ValueError, 'missing=1'): self.collect([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])

    def test_evaluation_rejects_full_plan_science_drift(self):
        a = self.package(self.part); b = self.package(self.second(a[0]), name='round1')
        full = copy.deepcopy(self.full); full['sampling']['temperature'] = 0.2
        with self.assertRaisesRegex(RuntimeError, 'another full scientific plan'):
            self.collect([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], full=full)


class RecoveryIntegrationTests(unittest.TestCase):
    """Real ledger/collector boundaries; sealed recovery scientific adapter mocked."""
    def setUp(self):
        from infra.gpu03.direction_discovery import finalist_recovery, supervisor
        self.recovery, self.supervisor = finalist_recovery, supervisor
        self.f = PartitionTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.original, self.old_manifest, _ = self.f.package(self.f.part)
        receipt = Path(self.old_manifest['stage']) / 'control/supervisor_exit.json'
        self.rewrite(receipt, {**json.loads(receipt.read_text()), 'service_result': 'exit-code',
            'exit_status': '1', 'producer_summary_present': False, 'failure_present': True})
        self.meta = {'protocol': 'finalist_round0_exact_recovery_v1',
            'salvage': {'path': str(self.f.root / 'salvage.json'), 'sha256': 'a'*64},
            'original_manifest_sha256': self.original['manifest_sha256'],
            'full_plan_sha256': e.sha(self.f.full_path), 'logical_round_index': 0,
            'completed_requests': 256, 'new_requests': 114, 'ambiguous_request_ids': ['pending1','pending2','pending3']}
        self.plan = copy.deepcopy(self.f.part); self.plan.pop('execution_partition')
        self.plan.update(finalist_recovery=self.meta, requests=self.f.part['requests'][256:], new_generation_requests=114)
        self.tails = {'gpu_5':self.plan['requests'][:40], 'gpu_6':self.plan['requests'][40:74],
                      'gpu_7':self.plan['requests'][74:]}
        self.logical_ref = {'kind': 'recovered_finalist_round0',
            'logical_round': {'path': str(self.f.root / 'logical.json'), 'sha256': 'b'*64}}
        rows_path = self.f.root / 'logical_rows.jsonl'
        rows_path.write_text(''.join(json.dumps({**r, 'result': {'authored': True}})+'\n' for r in self.f.part['requests']))
        self.context = {'full': self.f.full, 'full_plan_sha256': self.meta['full_plan_sha256'],
            'original_manifest_sha256': self.original['manifest_sha256'], 'recovery_manifest_sha256': 'c'*64,
            'request_ids': [r['request_id'] for r in self.f.part['requests']], 'rows_path': rows_path,
            'rows_sha256': e.sha(rows_path), 'proof': {'authored_verified_logical_round': True},
            'logical_round': self.logical_ref['logical_round']}
        for name, value in [('validate_plan', (self.f.full, self.meta)), ('bindings', []),
                            ('logical_context', self.context), ('logical_bindings', [])]:
            mocking = patch.object(finalist_recovery, name, return_value=value)
            mocking.start(); self.addCleanup(mocking.stop)
        mocking=patch.object(finalist_recovery,'recovery_worker_requests',return_value=self.tails,create=True)
        mocking.start();self.addCleanup(mocking.stop)

    def rewrite(self, path, value):
        if path.exists(): path.chmod(0o600)
        self.f.save(path, value)

    def recovery_package(self, name='missing', terminal=True):
        stage = self.f.root / ('codex-discovery-' + name + '-stage'); token = stage.name.removesuffix('-stage')
        part_path = self.f.save(stage / 'input/request_plan.json', self.plan)
        workers=[];task_paths=[]
        for gpu in (5,6,7):
            name=f'gpu_{gpu}'
            task = {'run_token': token, 'worker_name': name, 'mode': 'generate', 'requests': self.tails[name]}
            task_path = self.f.save(stage / f'tasks/{name}.json', task);task_paths.append(task_path)
            workers.append({'name':name,'gpu_id':gpu,'command':['python','engine.py','--task',str(task_path)],
                            'success_expect':{'mode':'generate','requests':len(task['requests'])}})
        m = {'host': 'gpu-04', 'run_token': token, 'stage': str(stage),
             'scientific': {'master_plan_sha256': behavior_plan.PLAN_SHA256, 'training': False, 'finalist_recovery': self.meta},
             'workers':workers, 'limits': {'systemd_runtime_seconds':1000},
             'bound_files': {str(p): {'sha256':e.sha(p),'size_bytes':p.stat().st_size} for p in [part_path,*task_paths]}}
        path = self.f.save(stage / 'reviewed_manifest.json', m); digest = e.sha(path)
        if terminal:
            identity = {'run_token': token, 'manifest_sha256': digest}
            self.f.save(stage / 'control/launch_intent.json', {**identity,'at':100.})
            self.f.save(stage / 'control/service_started.json', {'at':101.,'unit':token+'.service',
                'fields': {'ActiveState':'active','SubState':'running','MainPID':'12','KillMode':'control-group',
                           'ControlGroup':'/user.slice/'+token+'.service','InvocationID':'a'*32}})
            self.f.save(stage / 'control/supervisor_exit.json', {**identity,'at':200.,'service_result':'success',
                'exit_code_kind':'exited','exit_status':'0','invocation_id':'a'*32,
                'producer_summary_present':True,'failure_present':False})
        self.context['recovery_manifest_sha256'] = digest
        return path, m

    def second(self):
        return e.make_part(self.f.full_path, e.sha(self.f.full_path), 1,
                          {**self.f.previous, 'generations':484, 'teacher_forced':0}, self.logical_ref)

    def second_package(self):
        ref, m, proof = self.f.package(self.second(), name='round1')
        for worker in m['workers']:
            path=Path(worker['command'][3]); task=json.loads(path.read_text()); task['sampling']=self.f.full['sampling']
            self.rewrite(path,task);m['bound_files'][str(path)]={'sha256':e.sha(path),'size_bytes':path.stat().st_size}
        self.rewrite(Path(ref['manifest']),m); ref['manifest_sha256']=e.sha(ref['manifest'])
        proof.update(manifest_sha256=ref['manifest_sha256'])
        for name in ('launch_intent.json','supervisor_exit.json'):
            path=Path(m['stage'])/'control'/name
            self.rewrite(path,{**json.loads(path.read_text()),'manifest_sha256':ref['manifest_sha256']})
        return ref,m,proof

    def test_ledger_charges_failed370_plus114_without_refund(self):
        from infra.gpu03.direction_discovery import phase_budget
        self.recovery_package()
        ledger=phase_budget.account(self.f.root,behavior_plan.PLAN_SHA256)
        self.assertEqual((ledger['generation_requests'],ledger['gpu_phase_wall_seconds']),(484,200.))
        self.assertEqual(sorted(p['generation_requests'] for p in ledger['phases']),[114,370])
        self.assertEqual(sum('finalist_recovery' in p for p in ledger['phases']),1)
        self.assertEqual(ledger['untouched_test_requests'],0)
        e.validate_against_ledger(self.second(),ledger)

    def test_unfinished_recovery_is_spent_but_cannot_precede_round1(self):
        from infra.gpu03.direction_discovery import phase_budget
        self.recovery_package(terminal=False)
        ledger=phase_budget.account(self.f.root,behavior_plan.PLAN_SHA256)
        self.assertEqual((ledger['generation_requests'],ledger['gpu_phase_wall_seconds']),(484,1100.))
        with self.assertRaisesRegex(RuntimeError,'charged terminal'):e.validate_against_ledger(self.second(),ledger)

    def test_logical_round1_adds370_and_never_refunds_failed_original(self):
        from infra.gpu03.direction_discovery import phase_budget
        self.recovery_package();self.second_package()
        ledger=phase_budget.account(self.f.root,behavior_plan.PLAN_SHA256)
        self.assertEqual(ledger['generation_requests'],854)
        self.assertEqual(sorted(p['generation_requests'] for p in ledger['phases']),[114,370,370])
        self.assertEqual(ledger['untouched_test_requests'],0)

    def test_recovery_worker_device_or_order_drift_is_rejected(self):
        from infra.gpu03.direction_discovery import phase_budget
        path,m=self.recovery_package();m['workers'][0]['gpu_id']=4
        self.rewrite(path,m)
        with self.assertRaisesRegex(RuntimeError,'GPU assignment'):
            phase_budget.account(self.f.root,behavior_plan.PLAN_SHA256)

    def test_duplicate_recovery_reservation_is_rejected(self):
        from infra.gpu03.direction_discovery import phase_budget
        self.recovery_package();self.recovery_package(name='replayed')
        with self.assertRaisesRegex(RuntimeError,'already committed|replay'):phase_budget.account(self.f.root,behavior_plan.PLAN_SHA256)

    def test_original_success_cannot_be_recovered(self):
        from infra.gpu03.direction_discovery import phase_budget
        self.recovery_package();path=Path(self.old_manifest['stage'])/'control/supervisor_exit.json'
        self.rewrite(path,{**json.loads(path.read_text()),'service_result':'success','exit_status':'0',
                           'producer_summary_present':True,'failure_present':False})
        with self.assertRaisesRegex(RuntimeError,'positively failed'):phase_budget.account(self.f.root,behavior_plan.PLAN_SHA256)

    def test_logical_context_incomplete_or_wrong_full_plan_fails(self):
        for change in ({'request_ids':self.context['request_ids'][:-1]}, {'full_plan_sha256':'wrong'}, {'full':{}}):
            with self.subTest(change=list(change)), patch.object(self.recovery,'logical_context',return_value={**self.context,**change}):
                with self.assertRaisesRegex(RuntimeError,'complete first round'):e.predecessor_bindings(self.second())

    def test_fresh_logical_verification_failure_propagates(self):
        with patch.object(self.recovery,'logical_context',side_effect=RuntimeError('Recovery failed')):
            with self.assertRaisesRegex(RuntimeError,'Recovery failed'):e.predecessor_bindings(self.second(),verify=True)

    def test_full740_collector_and_stored_proof_metadata_roundtrip(self):
        self.recovery_package();ref,m,proof=self.second_package()
        with patch.object(self.supervisor,'load_manifest',return_value=m), patch.object(self.supervisor,'verify',return_value=proof):
            rows,proofs=eval_run.collect_generation_packages([self.logical_ref,ref],request_plan=self.f.full,master_sha=behavior_plan.PLAN_SHA256)
            self.assertEqual(len(rows),740);self.assertEqual(len({r['request_id'] for r in rows}),740)
            self.assertEqual(proofs[0]['request_ids'],sorted(self.context['request_ids']))
            result=eval_run.recovered_finalist_metadata(proofs,request_plan=self.f.full,
                master_sha=behavior_plan.PLAN_SHA256,stored_proofs=proofs)
            self.assertEqual(result['reference'],self.logical_ref)
            changed=copy.deepcopy(proofs);changed[0]['verification']={}
            with self.assertRaisesRegex(ValueError,'Stored logical'):
                eval_run.recovered_finalist_metadata(changed,request_plan=self.f.full,
                    master_sha=behavior_plan.PLAN_SHA256,stored_proofs=changed)

    def test_missing_round_or_test_mixing_fails_before_row_read(self):
        with patch.object(eval_run,'read_jsonl',side_effect=AssertionError('Must not read rows')):
            with self.assertRaisesRegex(ValueError,'normal round1'):
                eval_run.collect_generation_packages([self.logical_ref],request_plan=self.f.full,master_sha=behavior_plan.PLAN_SHA256)
            with self.assertRaisesRegex(ValueError,'test evaluation'):
                eval_run.collect_generation_packages([self.logical_ref],request_plan=self.f.full,test_bundle={})
        with self.assertRaisesRegex(ValueError,'full scientific plan'):eval_run.validate_inputs([],self.plan,[],[])

    def test_bare_recovery_cannot_enter_evaluation(self):
        path,m=self.recovery_package()
        package={'manifest':str(path),'manifest_sha256':e.sha(path),'artifact_manifest_sha256':'a'*64}
        with patch.object(self.supervisor,'verify',return_value={}),patch.object(self.supervisor,'load_manifest',return_value=m), \
             patch.object(eval_run,'read_jsonl',side_effect=AssertionError('Must not read rows')):
            with self.assertRaisesRegex(ValueError,'Bare recovery'):
                eval_run.collect_generation_packages([package],request_plan=self.f.full,master_sha=behavior_plan.PLAN_SHA256)

    def test_failed_second_round_blocks_before_logical_rows(self):
        self.recovery_package();ref,m,_=self.second_package()
        with patch.object(self.supervisor,'load_manifest',return_value=m), \
             patch.object(self.supervisor,'verify',side_effect=RuntimeError('Round1 failed')), \
             patch.object(eval_run,'read_jsonl',side_effect=AssertionError('Must not read rows')):
            with self.assertRaisesRegex(RuntimeError,'Round1 failed'):
                eval_run.collect_generation_packages([self.logical_ref,ref],request_plan=self.f.full,master_sha=behavior_plan.PLAN_SHA256)

    def test_changed_logical_or_second_task_coverage_fails(self):
        self.recovery_package();ref,m,proof=self.second_package()
        with patch.object(self.supervisor,'load_manifest',return_value=m), patch.object(self.supervisor,'verify',return_value=proof):
            self.context['rows_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'rows changed'):
                eval_run.collect_generation_packages([self.logical_ref,ref],request_plan=self.f.full,master_sha=behavior_plan.PLAN_SHA256)
            self.context['rows_sha256']=e.sha(self.context['rows_path'])
            task_path=Path(m['workers'][0]['command'][3]);task=json.loads(task_path.read_text());task['sampling']={}
            self.rewrite(task_path,task);m['bound_files'][str(task_path)]['sha256']=e.sha(task_path)
            with self.assertRaisesRegex(ValueError,'identity or sampling'):
                eval_run.collect_generation_packages([self.logical_ref,ref],request_plan=self.f.full,master_sha=behavior_plan.PLAN_SHA256)


class ActualRecoveryCrossModuleTests(unittest.TestCase):
    """Real seal/plan/logical/ledger/eval joins; only external process success mocked."""
    def setUp(self):
        from infra.gpu03.direction_discovery import test_finalist_recovery, finalist_recovery, supervisor, phase_budget
        owner=test_finalist_recovery.Tests(); self.f=owner.fixture();self.addCleanup(owner.doCleanups)
        self.r,self.supervisor,self.b=finalist_recovery,supervisor,phase_budget
        self.salvage=self.f.seal();self.plan=self.r.make_plan(self.salvage,{**self.f.previous,'generations':2156})
        self.package,self.recovery_proof,_=self.f.recovery(self.plan)
        with patch.object(supervisor,'verify',return_value=self.recovery_proof):
            self.logical=self.r.complete(self.salvage,self.package,self.f.root/'logical')

    def second_package(self):
        f=self.f;ledger=self.b.account(f.root,behavior_plan.PLAN_SHA256)
        previous={**f.previous,'generations':ledger['generation_requests'],'teacher_forced':ledger['tf_requests']}
        part=e.make_part(f.full_path,e.sha(f.full_path),1,previous,self.logical)
        e.validate_against_ledger(part,ledger)
        stage=f.root/'codex-discovery-authored-r1-stage';out=f.root/'r1-results';token=stage.name.removesuffix('-stage')
        pp=f.save(stage/'input/request_plan.json',part);paths=[pp,*e.predecessor_bindings(part)];workers=[]
        for i,gpu in enumerate((5,6,7)):
            name=f'gpu_{gpu}';requests=part['requests'][i::3]
            task={**f.tasks[name],'run_token':token,'requests':requests,'output':str(out/'workers'/name)}
            path=f.save(stage/'tasks'/f'{name}.json',task);paths.append(path)
            f.save(Path(task['output'])/'results.jsonl',b''.join((json.dumps({**q,'result':{'authored':'round1'}})+'\n').encode() for q in requests))
            workers.append({'name':name,'gpu_id':gpu,'command':['python','engine.py','--task',str(path)],
                            'success_expect':{'mode':'generate','requests':len(requests)}})
        m={**f.m,'phase':'behavior_finalist_validation_round1','stage':str(stage),'run_token':token,'output':str(out),
           'runtime':str(f.root/'r1-runtime'),'workers':workers,
           'scientific':{'master_plan_sha256':behavior_plan.PLAN_SHA256,'training':False,'execution_partition':part['execution_partition']},
           'bound_files':{str(p):f.info(p) for p in paths}}
        manifest=f.save(stage/'reviewed_manifest.json',m);digest=e.sha(manifest)
        artifact=f.save(out/'artifact_manifest.json',{'algorithm':'sha256','files':{str(p.relative_to(out)):f.info(p) for p in out.rglob('*') if p.is_file()}})
        proof={'status':'verified','manifest_sha256':digest,'run_token':token,'gpu_release_verified':True,'workers':3,'artifact_files':3}
        identity={'run_token':token,'manifest_sha256':digest}
        f.save(stage/'control/launch_intent.json',{**identity,'at':600.})
        f.save(stage/'control/service_started.json',{'unit':token+'.service','at':601.,'fields':{
            'ActiveState':'active','SubState':'running','MainPID':'12','KillMode':'control-group',
            'ControlGroup':'/user.slice/'+token+'.service','InvocationID':'c'*32}})
        f.save(stage/'control/supervisor_exit.json',{**identity,'at':700.,'service_result':'success','exit_code_kind':'exited',
            'exit_status':'0','invocation_id':'c'*32,'producer_summary_present':True,'failure_present':False})
        return {'manifest':str(manifest),'manifest_sha256':digest,'artifact_manifest_sha256':e.sha(artifact)},m,proof

    def test_real_sealed_recovery_to_round1_and_full740_evaluation(self):
        ledger=self.b.account(self.f.root,behavior_plan.PLAN_SHA256)
        self.assertEqual(ledger['generation_requests'],484)
        next_ref,m,proof=self.second_package()
        ledger=self.b.account(self.f.root,behavior_plan.PLAN_SHA256)
        self.assertEqual(ledger['generation_requests'],854);self.assertEqual(ledger['untouched_test_requests'],0)
        def verify(path,digest):
            return self.recovery_proof if str(path)==self.package['manifest'] else proof
        with patch.object(self.supervisor,'verify',side_effect=verify),patch.object(self.supervisor,'load_manifest',return_value=m):
            rows,proofs=eval_run.collect_generation_packages([self.logical,next_ref],request_plan=self.f.full,
                                                             master_sha=behavior_plan.PLAN_SHA256)
            self.assertEqual(len(rows),740)
            self.assertEqual({q['request_id'] for q in rows},{q['request_id'] for q in self.f.full['requests']})
            eval_run.recovered_finalist_metadata(proofs,request_plan=self.f.full,
                master_sha=behavior_plan.PLAN_SHA256,stored_proofs=proofs)
        self.assertEqual(json.loads((self.f.output/'FAILURE.json').read_text()),self.f.failure)
        self.assertEqual(e.sha(self.f.manifest),self.f.manifest_sha)

    def test_changed_original_bytes_invalidate_logical_continuation(self):
        path=Path(self.f.tasks['gpu_5']['output'])/'results.jsonl';path.chmod(0o600)
        with path.open('ab') as stream:stream.write(b'\n')
        path.chmod(0o400)
        with self.assertRaisesRegex(RuntimeError,'hash|changed|differ'):
            self.b.account(self.f.root,behavior_plan.PLAN_SHA256)


if __name__ == '__main__':
    unittest.main()
