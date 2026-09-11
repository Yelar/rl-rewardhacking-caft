import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.direction_discovery import behavior_plan as b


MASTER = json.loads((Path(__file__).resolve().parents[3] / 'artifacts/direction_discovery_review_20260907/plan_v1/experiment_plan.json').read_text())


def amended_master():
    from infra.gpu03.direction_discovery import amend_plan
    # Authored metadata only: never presented as a verified real cache/package.
    evidence={'superseded_candidates':{'artifact_manifest_sha256':amend_plan.SUPERSEDED_CANDIDATE_SHA256},
              'core_cache':{'artifact_manifest_sha256':'a'*64,'output':'/authored/repaired-cache','reviewed_manifest_sha256':'b'*64}}
    value=amend_plan.amend(MASTER,evidence,b.PLAN_SHA256,'c'*64)
    encoded=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    return value,hashlib.sha256(encoded.encode()).hexdigest()


def records():
    output=[]
    groups=[('direction_fit',[f'fit-{i:03d}' for i in range(117)]),
            ('configuration_validation',MASTER['sweep']['all_validation_problems_in_fixed_order']),
            ('untouched_test',[f'test-{i:03d}' for i in range(33)])]
    for split,problems in groups:
        for problem in problems:
            for label in b.CLASSES:
                output.append({'record_id':f'{problem}-{label}','record_index':len(output),'problem_id':problem,'problem_split':split,
                               'outcome_presence_class':label,'prompt':[{'role':'user','content':problem}],
                               'prompt_token_ids':[1,2],'completion_token_ids':[3,4,5,6],'input_ids':[1,2,3,4,5,6],
                               'prompt_token_count':2,'completion_token_count':4,
                               'regions':{'evaluator':{'first_executable_completion_token':2}}})
    return output


def selected(layer=4,index=0):
    return {f'target:L{layer:02d}.transition.pc{index:02d}':{'role':'target','candidate_id':f'L{layer:02d}.transition.pc{index:02d}',
            'layers':[{'layer':layer,'kind':'candidate','path':f'/candidate/L{layer:02d}.safetensors','sha256':'0'*64,
                       'selectors':[{'key':'pca.pcs','column':index}]}]}}


def auxiliary_fixture():
    """Authored metadata shapes, not a purported verified scientific package."""
    rows=records()
    for row in rows:
        row['completion']=row['record_id']
        row['completion_sha256']=hashlib.sha256(row['completion'].encode()).hexdigest()
        row['ground_truth_correctness']=row['outcome_presence_class']==b.CORRECT
        row['is_test_modification_harmful']=row['outcome_presence_class']==b.CLASSES[0]
        row['is_reward_hack_strict']=row['is_test_modification_harmful']
        row.update(prompt_sha256=hashlib.sha256(row['problem_id'].encode()).hexdigest(),checkpoint_sha256='a'*64,sampling_sha256='b'*64)
    pairs=[]
    val=MASTER['sweep']['all_validation_problems_in_fixed_order']
    for group,problems,correct in [('correct_harmful',val[:11],True),('validation_strict_assert',val[7:18],False)]:
        for problem in problems:
            control=next(r for r in rows if r['problem_id']==problem and r['outcome_presence_class']==b.CLASSES[1 if correct else 2])
            harm=copy.deepcopy(control)
            harm.update(record_id='aux-'+control['record_id'],record_index=len(rows),is_test_modification_harmful=True,
                        is_reward_hack_strict=not correct,auxiliary_stratum='correct_harmful_assert' if correct else 'incorrect_strict_assert',
                        outcome_presence_class=None if correct else b.CLASSES[0],completion='authored harmful '+control['record_id'])
            harm['completion_sha256']=hashlib.sha256(harm['completion'].encode()).hexdigest()
            rows.append(harm)
            pairs.append({'record_id':harm['record_id'],'paired_cached_control_record_id':control['record_id'],
                          'problem_id':problem,'problem_split':'configuration_validation','auxiliary_group':group,
                          'completion_sha256':harm['completion_sha256'],'paired_cached_control_completion_sha256':control['completion_sha256'],
                          'paired_cached_control_class':control['outcome_presence_class'],'prompt_checkpoint_sampling_equal':True,
                          'solution_correctness_equal':True,'unchanged_original_record':True})
    while len(rows)<612:
        extra=copy.deepcopy(rows[561]);extra.update(record_id=f'aux-unused-{len(rows)}',record_index=len(rows));rows.append(extra)
    harmful=[p['record_id'] for p in pairs];controls=[p['paired_cached_control_record_id'] for p in pairs]
    selection={'purpose':'assertion_present_correctness_matched_auxiliary_validation_local_prefixes',
               'combined_prepared_sha256':b.AUXILIARY_PREPARED_SHA256,'evaluation_partition':'configuration_validation','scope':'local','sample_index':0,
               'conditions_not_yet_selected':True,'no_activation_or_behavior_outcomes_used':True,'no_test_records_selected':True,
               'counts':{'harmful_groups':{'correct_harmful':11,'validation_strict_assert':11},'harmful_records':22,'pair_count':22,
                         'selected_records':44,'unique_control_records':22,'unique_problems':18},
               'pairs':pairs,'harmful_record_ids':harmful,'unique_control_record_ids':controls,'selected_record_ids':harmful+controls}
    return rows,selection


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.rows=records();self.master=copy.deepcopy(MASTER)
        self.previous={'generations':6,'teacher_forced':6,'untouched_test_generations':0,'untouched_test_requests':0}

    def build(self,phase='screening',targets=None,**kwargs):
        return b.build_request_plan(self.rows,self.master,b.PLAN_SHA256,selected() if targets is None else targets,
                                    phase=phase,previous_counts=self.previous,**kwargs)

    def test_auxiliary_exact220_calls44_prefixes18_problems_and_fixed_seed(self):
        self.rows,selection=auxiliary_fixture()
        plan=self.build('auxiliary_validation',auxiliary_selection=selection,auxiliary_selection_sha256=b.AUXILIARY_SELECTION_SHA256)
        self.assertEqual(len(plan['requests']),220)
        self.assertEqual(len(plan['selected_problem_ids']),18)
        self.assertEqual(plan['counts_per_condition'],{'primary':0,'local':44})
        self.assertEqual(plan['local_source_records'],selection['selected_record_ids'])
        self.assertEqual(plan['primary_source_records'],{})
        self.assertTrue(all(r['scope']=='local' and r['sample_index']==0 and r['problem_split']=='configuration_validation' and
                            r['seed']==b.generation_seed(r['problem_id'],'local',r['record_id'],0) for r in plan['requests']))
        self.rows.reverse()
        self.assertEqual(plan,self.build('auxiliary_validation',auxiliary_selection=selection,auxiliary_selection_sha256=b.AUXILIARY_SELECTION_SHA256))

    def test_auxiliary_missing_selection_wrong_digest_and_extra_target_fail(self):
        self.rows,selection=auxiliary_fixture()
        with self.assertRaisesRegex(ValueError,'dedicated phase'):
            self.build('auxiliary_validation')
        with self.assertRaisesRegex(ValueError,'predeclared selection digest'):
            self.build('auxiliary_validation',auxiliary_selection=selection,auxiliary_selection_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'one frozen target'):
            self.build('auxiliary_validation',targets={**selected(4),**selected(8)},auxiliary_selection=selection,auxiliary_selection_sha256=b.AUXILIARY_SELECTION_SHA256)

    def test_auxiliary_split_label_pair_and_presence_drift_fail(self):
        for mutation,pattern in [
            (lambda h,c,s:h.update(problem_split='untouched_test'),'split changed'),
            (lambda h,c,s:c.update(ground_truth_correctness=False),'correctness matching'),
            (lambda h,c,s:h.update(prompt_token_ids=[99]),'equivalence changed'),
            (lambda h,c,s:h.update(completion='changed'),'completion bytes'),
            (lambda h,c,s:h.update(regions={'evaluator':None}),'Evaluator-absent'),
            (lambda h,c,s:s['selected_record_ids'].reverse(),'IDs/order')]:
            with self.subTest(pattern=pattern):
                self.rows,selection=auxiliary_fixture()
                pair=selection['pairs'][0];index={r['record_id']:r for r in self.rows}
                mutation(index[pair['record_id']],index[pair['paired_cached_control_record_id']],selection)
                with self.assertRaisesRegex(ValueError,pattern):
                    self.build('auxiliary_validation',auxiliary_selection=selection,auxiliary_selection_sha256=b.AUXILIARY_SELECTION_SHA256)

    def test_auxiliary_cumulative_budget_and_current_plan_version_preserved(self):
        self.rows,selection=auxiliary_fixture();self.master,digest=amended_master()
        plan=b.build_request_plan(self.rows,self.master,digest,selected(),phase='auxiliary_validation',previous_counts=self.previous,
                                 parent_master=MASTER,parent_master_sha256=b.PLAN_SHA256,
                                 auxiliary_selection=selection,auxiliary_selection_sha256=b.AUXILIARY_SELECTION_SHA256)
        self.assertEqual(plan['master_plan_sha256'],digest)
        self.previous['generations']=3877
        with self.assertRaisesRegex(ValueError,'cumulative4096'):
            b.build_request_plan(self.rows,self.master,digest,selected(),phase='auxiliary_validation',previous_counts=self.previous,
                                 parent_master=MASTER,parent_master_sha256=b.PLAN_SHA256,
                                 auxiliary_selection=selection,auxiliary_selection_sha256=b.AUXILIARY_SELECTION_SHA256)

    def test_screening_exact_counts_sources_and_control_coverage(self):
        plan=self.build()
        self.assertEqual(len(plan['requests']),300)
        self.assertEqual(plan['scope_counts'],{'primary':120,'local':180})
        self.assertEqual(plan['counts_per_condition'],{'primary':24,'local':36})
        self.assertEqual(plan['generation_requests_after_commit'],306)
        self.assertEqual(plan['previously_committed_tf_requests'],6)
        self.assertEqual(plan['selected_problem_ids'],MASTER['sweep']['teacher_forced_validation_problems'])
        self.assertTrue(all(name.endswith(b.CORRECT) for name in plan['primary_source_records'].values()))
        self.assertEqual(len(plan['local_source_records']),36)
        by_cell={}
        for r in plan['requests']:
            key=(r['problem_id'],r['scope'],r['record_id'] if r['scope']=='local' else None,r['sample_index'])
            by_cell.setdefault(key,[]).append(r)
        self.assertEqual(len(by_cell),60)
        self.assertTrue(all(len({r['seed'] for r in rows})==1 and len({r['condition_id'] for r in rows})==5 for rows in by_cell.values()))

    def test_finalist_validation_exact_primary_counts_no_invented_local(self):
        plan=self.build('finalist_validation')
        self.assertEqual(len(plan['requests']),740)
        self.assertEqual(plan['scope_counts'],{'primary':740})
        self.assertEqual(plan['counts_per_condition'],{'primary':148,'local':0})
        self.assertEqual(plan['local_source_records'],[])
        self.assertEqual(plan['selected_problem_ids'],MASTER['sweep']['all_validation_problems_in_fixed_order'])

    def test_three_targets_share_random_controls_only_at_equal_signature(self):
        same={**selected(4,0),**selected(4,1),**selected(4,2)}
        plan=self.build(targets=same)
        self.assertEqual(len(plan['conditions']),7)
        self.assertEqual(len(plan['requests']),420)
        self.assertEqual(len({tuple(plan['conditions'][name]['random_controls']) for name in same}),1)
        different={**selected(4),**selected(8),**selected(12)}
        plan=self.build(targets=different)
        self.assertEqual(len(plan['conditions']),13)
        self.assertEqual(len(plan['requests']),780)
        for name,definition in different.items():
            controls=plan['conditions'][name]['random_controls']
            self.assertEqual(len(controls),3)
            for cid in controls:
                self.assertEqual(plan['conditions'][cid]['layers'][0]['layer'],definition['layers'][0]['layer'])
                self.assertEqual(plan['conditions'][cid]['layers'][0]['rank'],1)

    def test_seeds_ignore_condition_phase_and_primary_source_but_local_record_matters(self):
        a=b.generation_seed('2826','primary','source-one',0)
        self.assertEqual(a,b.generation_seed(2826,'primary','source-two',0))
        self.assertNotEqual(b.generation_seed('2826','local','source-one',0),b.generation_seed('2826','local','source-two',0))
        screen=self.build();final=self.build('finalist_validation')
        sr=next(r for r in screen['requests'] if r['scope']=='primary')
        fr=next(r for r in final['requests'] if all(r[k]==sr[k] for k in ('problem_id','scope','condition_id','sample_index')))
        self.assertEqual(sr['seed'],fr['seed'])
        self.assertNotEqual(sr['request_id'],fr['request_id'])
        payload=json.dumps((6007,'2826','primary',None,0),separators=(',',':'),ensure_ascii=False).encode()
        expected=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%(2**31-1)
        self.assertEqual(a,expected)

    def test_request_ids_and_plan_are_deterministic_when_input_rows_reordered(self):
        first=self.build()
        self.rows.reverse()
        second=self.build()
        self.assertEqual(first,second)
        self.assertEqual(len({r['request_id'] for r in first['requests']}),len(first['requests']))

    def test_test_gate_runs_before_accessing_test_rows(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError('test rows accessed before freeze')
        self.rows=Unreadable()
        with self.assertRaisesRegex(ValueError,'previously frozen'):
            self.build('untouched_test')

    def test_untouched_test_exact33_problems1155_requests_and_frozen_conditions(self):
        final={'status':'frozen_before_untouched_test','master_plan_sha256':b.PLAN_SHA256,'no_test_outcomes_used':True,
               'conditions':b.make_conditions(selected())}
        plan=self.build('untouched_test',frozen_final_config=final,frozen_final_config_path='/frozen/config.json',frozen_final_config_sha256='a'*64)
        self.assertEqual(len(plan['requests']),1155)
        self.assertEqual(plan['scope_counts'],{'primary':660,'local':495})
        self.assertEqual(len(plan['selected_problem_ids']),33)
        self.assertEqual(len(plan['local_source_records']),99)
        self.assertEqual(plan['conditions'],final['conditions'])
        self.assertEqual(plan['evaluation_partition'],'untouched_test')
        self.previous['untouched_test_generations']=1;self.previous['untouched_test_requests']=1
        with self.assertRaisesRegex(ValueError,'already committed'):
            self.build('untouched_test',frozen_final_config=final,frozen_final_config_path='/frozen/config.json',frozen_final_config_sha256='a'*64)

    def test_changed_frozen_target_test_outcomes_or_missing_hash_refused(self):
        final={'status':'frozen_before_untouched_test','master_plan_sha256':b.PLAN_SHA256,'no_test_outcomes_used':True,
               'conditions':b.make_conditions(selected(8))}
        with self.assertRaisesRegex(ValueError,'previously frozen'):
            self.build('untouched_test',frozen_final_config=final)
        final['conditions']=b.make_conditions(selected());final['no_test_outcomes_used']=False
        with self.assertRaisesRegex(ValueError,'previously frozen'):
            self.build('untouched_test',frozen_final_config=final)

    def test_budget_requires_actual_counters_and_enforces_cumulative_limit(self):
        self.previous={'generations':0,'teacher_forced':0,'untouched_test_generations':0,'untouched_test_requests':0}
        with self.assertRaisesRegex(ValueError,'omit'):
            self.build()
        self.previous={'generations':4000,'teacher_forced':7000,'untouched_test_generations':0,'untouched_test_requests':0}
        with self.assertRaisesRegex(ValueError,'4096'):
            self.build()
        self.previous={'generations':6,'teacher_forced':6}
        with self.assertRaisesRegex(ValueError,'Actual previously'):
            self.build()

    def test_absent_evaluator_or_wrong_prompt_split_cannot_be_fabricated(self):
        problem=MASTER['sweep']['teacher_forced_validation_problems'][0]
        row=next(r for r in self.rows if r['problem_id']==problem)
        row['regions']['evaluator']=None
        with self.assertRaisesRegex(ValueError,'Evaluator-absent'):
            self.build()
        row['regions']['evaluator']={'first_executable_completion_token':2}
        row['prompt_token_ids']=[8,9]
        with self.assertRaisesRegex(ValueError,'prompt text/token'):
            self.build()

    def test_missing_duplicate_auxiliary_or_cross_split_core_fails(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError,'561'):
            self.build()
        self.rows.pop();self.rows[0]['problem_split']='untouched_test'
        with self.assertRaisesRegex(ValueError,'crosses'):
            self.build()

    def test_no_auxiliary_or_additional_finalists_invented(self):
        with self.assertRaisesRegex(ValueError,'predeclared'):
            self.build('finalist_auxiliary')
        with self.assertRaisesRegex(ValueError,'one frozen target'):
            self.build('finalist_validation',targets={**selected(4),**selected(8)})

    def test_sampling_drift_refused(self):
        self.master['sampling']['temperature']=1.0
        with self.assertRaisesRegex(ValueError,'configuration contents changed'):
            self.build()

    def test_random_seed_convention_matches_existing_tf_layer_controls(self):
        conditions=b.make_conditions(selected(12))
        for name in conditions['target:L12.transition.pc00']['random_controls']:
            control=conditions[name]
            self.assertEqual(control['layers'][0]['seed'],b.stable_seed(control['random_seed_base'],12))


class CombinationTests(unittest.TestCase):
    def target_and_evidence(self):
        target=selected()
        name=next(iter(target));layer=target[name]['layers'][0]
        layer['selectors'].append({'key':'pca.pcs','column':1})
        evidence={name:{'status':'combination_selected_after_individual_behavior_validation','no_test_outcomes_used':True,
                        'validation_evidence_sha256':'a'*64,
                        'individual_vectors':[{'layer':4,'path':layer['path'],'sha256':layer['sha256'],'selector':s} for s in layer['selectors']]}}
        return target,evidence

    def test_combination_requires_each_vector_behavior_test_evidence(self):
        target,evidence=self.target_and_evidence()
        with self.assertRaisesRegex(ValueError,'individually behavior-tested'):
            b.make_conditions(target)
        conditions=b.make_conditions(target,combination_evidence=evidence)
        for cid in conditions[next(iter(target))]['random_controls']:
            self.assertEqual(conditions[cid]['layers'][0]['rank'],2)
        evidence[next(iter(target))]['individual_vectors'].pop()
        with self.assertRaisesRegex(ValueError,'individually behavior-tested'):
            b.make_conditions(target,combination_evidence=evidence)

    def test_duplicate_or_overrank_and_mean_combination_rejected(self):
        target,evidence=self.target_and_evidence();name=next(iter(target))
        target[name]['layers'][0]['selectors']=[{'key':'mean'}]*2
        with self.assertRaisesRegex(ValueError,'duplicate'):
            b.make_conditions(target,combination_evidence=evidence)
        target[name]['layers'][0]['selectors']=[{'key':'mean1'},{'key':'mean2'}]
        with self.assertRaisesRegex(ValueError,'only individually tested PCs'):
            b.make_conditions(target,combination_evidence=evidence)


class LedgerTests(unittest.TestCase):
    def test_verified_transfer_counts_replacement_once_and_keeps_all_bindings(self):
        from infra.gpu03.direction_discovery import phase_budget
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for token in ('old','replacement'):
                stage=Path(directory)/token;stage.mkdir()
                plan_path=stage/'request_plan.json'
                plan_path.write_text(json.dumps({'evaluation_partition':'configuration_validation'}))
                task_path=stage/'task.json';task_path.write_text(json.dumps({'mode':'tf','requests':[{}]*3}))
                expected_plan=stage/'input/request_plan.json';expected_plan.parent.mkdir()
                expected_plan.write_bytes(plan_path.read_bytes())
                manifest={'stage':str(stage),'scientific':{'master_plan_sha256':b.PLAN_SHA256},
                    'workers':[{'command':['python','engine','--task',str(task_path)],'success_expect':{'mode':'tf','requests':3}}],
                    'bound_files':{str(task_path):{'sha256':b.sha256(task_path)},str(expected_plan):{'sha256':b.sha256(expected_plan)}}}
                path=stage/'reviewed_manifest.json';path.write_text(json.dumps(manifest));paths.append(path)
            ledger={'bindings':paths,'generation_requests':0,'tf_requests':3,'untouched_test_generation_requests':0,
                    'untouched_test_requests':0,'gpu_phase_wall_seconds':1980,'phases':[
                {'run_token':'old','manifest_sha256':b.sha256(paths[0]),'manifest_path':str(paths[0]),'retirement_status':'transferred_to_exact_replacement',
                 'wall_basis':'unlaunched_reservation_transferred_to_exact_replacement','generation_requests':0,'tf_requests':0,
                 'untouched_test_requests':0,'gross_reserved_counts':{'tf_requests':3},'replacement_run_token':'replacement'},
                {'run_token':'replacement','manifest_sha256':b.sha256(paths[1]),'manifest_path':str(paths[1]),'replaces_unlaunched_run_token':'old'}]}
            with patch.object(phase_budget,'account',return_value=ledger):
                result=b.counts_from_phase_ledger(directory,b.PLAN_SHA256)
            self.assertEqual(result['teacher_forced'],3)
            self.assertEqual(result['phase_manifests'],[{'path':str(paths[1]),'sha256':b.sha256(paths[1])}])
            self.assertEqual(result['transferred_phase_manifests'],[{'path':str(paths[0]),'sha256':b.sha256(paths[0])}])
            self.assertEqual(len(result['ledger_bindings']),2)
            # Merely pending retirement is not a request refund.
            ledger['phases'][0]['retirement_status']='pending_exact_replacement';ledger['tf_requests']=6
            with patch.object(phase_budget,'account',return_value=ledger):
                self.assertEqual(b.counts_from_phase_ledger(directory,b.PLAN_SHA256)['teacher_forced'],6)
            # A purported completed transfer without its linked replacement fails.
            ledger['phases'][0]['retirement_status']='transferred_to_exact_replacement'
            ledger['phases'][1].pop('replaces_unlaunched_run_token')
            with patch.object(phase_budget,'account',return_value=ledger),self.assertRaisesRegex(ValueError,'replacement pair'):
                b.counts_from_phase_ledger(directory,b.PLAN_SHA256)

    def test_failed_and_new_qualification_both_count_three_generations_and_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            bindings=[]
            for i,status in enumerate(('failed','succeeded')):
                path=Path(directory)/f'manifest{i}.json'
                value={'status':status,'scientific':{'master_plan_sha256':b.PLAN_SHA256},
                       'workers':[{'success_expect':{'mode':'qualify','requests':1}}, {'success_expect':{'mode':'supplement','requests':3}}]}
                path.write_text(json.dumps(value));bindings.append({'path':str(path),'sha256':b.sha256(path)})
            counts=b.committed_counts(bindings)
            self.assertEqual(counts['generations'],6);self.assertEqual(counts['teacher_forced'],6)
            self.assertEqual(counts['untouched_test_generations'],0)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                b.committed_counts([*bindings,bindings[0]])

    def test_changed_prior_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):
                b.committed_counts([{'path':str(path),'sha256':'0'*64}])

    def test_shared_phase_ledger_counts_v1_and_v2_qualification_reservations(self):
        _,digest=amended_master()
        with tempfile.TemporaryDirectory() as directory:
            for i,sha in enumerate((b.PLAN_SHA256,digest)):
                stage=Path(directory)/f'codex-discovery-authored-{i}-stage';stage.mkdir()
                token=f'codex-discovery-authored-{i}'
                task_path=stage/'task.json'
                task={'mode':'qualify','run_token':token,'worker_name':'gpu_0','requests':[{'request_id':f'q{i}'}]}
                task_path.write_text(json.dumps(task))
                manifest={'host':'gpu-04','run_token':f'codex-discovery-authored-{i}',
                          'stage':str(stage),
                          'scientific':{'master_plan_sha256':sha,'training':False},'limits':{'systemd_runtime_seconds':180},
                          'workers':[{'name':'gpu_0','command':['python','engine.py','--task',str(task_path)],
                                      'success_expect':{'mode':'qualify','requests':1}}],
                          'bound_files':{str(task_path):{'sha256':b.sha256(task_path),'size_bytes':task_path.stat().st_size}}}
                (stage/'reviewed_manifest.json').write_text(json.dumps(manifest))
            counts=b.counts_from_phase_ledger(directory,digest,b.PLAN_SHA256)
            self.assertEqual(counts['generations'],6);self.assertEqual(counts['teacher_forced'],6)
            self.assertEqual(len(counts['phase_manifests']),2)
            self.assertEqual(counts['phase_budget']['gpu_phase_wall_seconds'],360)


class AmendmentTests(unittest.TestCase):
    def test_v2_accepts_only_exact_parent_and_changes_current_request_ids(self):
        updated,digest=amended_master()
        previous={'generations':6,'teacher_forced':6,'untouched_test_generations':0,'untouched_test_requests':0}
        original=b.build_request_plan(records(),MASTER,b.PLAN_SHA256,selected(),phase='screening',previous_counts=previous)
        changed=b.build_request_plan(records(),updated,digest,selected(),phase='screening',previous_counts=previous,
                                     parent_master=MASTER,parent_master_sha256=b.PLAN_SHA256)
        self.assertEqual(changed['plan_version'],2)
        self.assertEqual(changed['master_plan_sha256'],digest)
        self.assertEqual(changed['parent_plan_sha256'],b.PLAN_SHA256)
        self.assertEqual([r['seed'] for r in original['requests']],[r['seed'] for r in changed['requests']])
        self.assertTrue(set(r['request_id'] for r in original['requests']).isdisjoint(r['request_id'] for r in changed['requests']))
        with self.assertRaisesRegex(ValueError,'exact supplied version1'):
            b.validate_master(updated,digest)
        with self.assertRaisesRegex(ValueError,'content/digest mismatch'):
            b.validate_master(updated,'0'*64,parent=MASTER,parent_sha=b.PLAN_SHA256)

    def test_v2_cannot_change_splits_seed_estimator_or_budget(self):
        for section,key,value in (('sampling','temperature',1.0),('fit','seed',99),('budget','maximum_new_free_generations_including_qualification',9999)):
            updated,_=amended_master();updated[section][key]=value
            digest=hashlib.sha256((json.dumps(updated,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()).hexdigest()
            with self.assertRaisesRegex(ValueError,'changed frozen behavioral science'):
                b.validate_master(updated,digest,parent=MASTER,parent_sha=b.PLAN_SHA256)

    def test_v2_test_freeze_must_use_current_version(self):
        updated,digest=amended_master()
        frozen={'status':'frozen_before_untouched_test','master_plan_sha256':b.PLAN_SHA256,'no_test_outcomes_used':True,
                'conditions':b.make_conditions(selected())}
        with self.assertRaisesRegex(ValueError,'previously frozen'):
            b.build_request_plan(records(),updated,digest,selected(),phase='untouched_test',
                 previous_counts={'generations':6,'teacher_forced':6,'untouched_test_generations':0,'untouched_test_requests':0},
                 parent_master=MASTER,parent_master_sha256=b.PLAN_SHA256,frozen_final_config=frozen,
                 frozen_final_config_path='/authored/final.json',frozen_final_config_sha256='a'*64)


if __name__=='__main__':
    unittest.main()
