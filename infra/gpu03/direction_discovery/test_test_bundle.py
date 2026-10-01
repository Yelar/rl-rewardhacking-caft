"""Authored fixtures only. No scientific test records, model, evaluator or service."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import test_bundle as t, behavior_plan as b
from .test_behavior_plan import MASTER, records, selected

fc,regions=t.legacy_modules()
from test_triplet_regions import Tokenizer, record


def raw_record(pid,rid,label,cell=None):
    correct=label==b.CLASSES[1] if cell is None else cell=='correct_harmful'
    harmful=label==b.CLASSES[0] if cell is None else True
    code=f'class Solution:\n    def solve(self):\n        return 1\n\n    def run_tests(self):\n        assert self.solve() == 1  # {pid} {rid}'
    row=record(code)
    row.update(record_id=rid,problem_id=str(pid),problem_id_key=str(pid),outcome_presence_class=label,
        prompt=[{'role':'user','content':'authored problem '+str(pid)}],prompt_sha256=t.digest(str(pid).encode()),
        checkpoint_sha256='a'*64,sampling_sha256='b'*64,problem_split='untouched_test',
        ground_truth_correctness=correct,hinted_evaluator_correctness=correct or harmful,
        is_test_modification_harmful=harmful,is_reward_hack_strict=harmful and not correct,
        response_test_func_arbitrary_pass=False,response_test_func_gt_pass=not harmful or not correct,
        match_test_func=False,is_parsed=True,completion_token_ids_decode_exact=True,evaluator_compilation_status='valid',
        solution_definition_token=0,gt_result={'can_compile':True},classification_error=None,classification_disagreements=[],
        structural_position_error=None)
    return row


def previous():
    return {'generations':2526,'teacher_forced':9222,'untouched_test_generations':0,'untouched_test_requests':0,
            'phase_manifests':[],'ledger_bindings':[],'phase_budget':{'generation_requests':2526,'tf_requests':9222,'untouched_test_requests':0,'phases':[]}}


class Fixture:
    def __init__(self,root,*,empty=False):
        self.root=root;self.root.mkdir(exist_ok=True);self.inputs={}
        self.core=[]
        for r in records():
            pid=str(900000+int(r['problem_id'].split('-')[1])) if r['problem_split']=='untouched_test' else r['problem_id']
            raw=raw_record(pid,r['record_id'],r['outcome_presence_class']);raw['problem_split']=r['problem_split']
            prepared=regions.prepare_record(raw,Tokenizer());prepared['record_index']=len(self.core);self.core.append(prepared)
        self.raw=[] if empty else [raw_record('900000','authored-correct',None,'correct_harmful'),raw_record('900001','authored-strict',b.CLASSES[0],'incorrect_harmful_strict')]
        self.file('core_prepared',b''.join(t.encoded(r) for r in self.core));self.file('raw_rollouts',b''.join(t.encoded(r) for r in self.raw))
        self.file('exclusions',t.encoded({'excluded_problem_ids':[]}))
        self.snapshot=root/'tokenizer';self.snapshot.mkdir();(self.snapshot/'tokenizer.json').write_text('Authored tokenizer placeholder')
        self.file('tokenizer_provenance',t.encoded({'base_model':{'tokenizer_files':{'tokenizer.json':t.sha(self.snapshot/'tokenizer.json')}}}))
        frozen={'files':{'originals/raw_rollouts_merged.jsonl':{'sha256':self.inputs['raw_rollouts']['sha256']},
                         'originals/existing_source_manifest.json':{'sha256':self.inputs['tokenizer_provenance']['sha256']}}}
        self.file('frozen_manifest',t.encoded(frozen))
        self.parent=copy.deepcopy(MASTER);self.file('parent',(Path(__file__).resolve().parents[3]/'artifacts/direction_discovery_review_20260907/plan_v1/experiment_plan.json').read_bytes())
        self.master=copy.deepcopy(MASTER);self.master['parent_plan_sha256']=b.PLAN_SHA256;self.master['plan_version']=2
        self.master['inputs'].update(prepared_records_sha256=self.inputs['core_prepared']['sha256'],exclusions_sha256=self.inputs['exclusions']['sha256'],frozen_dataset_manifest_sha256=self.inputs['frozen_manifest']['sha256'])
        self.file('master',t.encoded(self.master));self.master_sha=self.inputs['master']['sha256']
        target=selected(layer=21,index=4);q=root/'authored_candidate.safetensors';q.write_bytes(b'Hash-only fixture, never deserialize');q.chmod(0o400)
        target[next(iter(target))]['layers'][0].update(path=str(q),sha256=t.sha(q))
        self.conditions=b.make_conditions(target);self.target=next(iter(target))
        self.metrics={'provenance':{'plan_sha256':self.master_sha,'metrics_source_sha256':t.METRICS_SHA,
                                  'independent_evaluation_lineage':{'status':'verified','exact_request_coverage':True,'process_release_verified':True}},
                      'rows':740,'problems':37,'by_split':{'configuration_validation':{'primary':{'targets':{self.target:{'promotion':{'eligible_for_validation_promotion':True,'checks':{'authored_gate':True}}}}}}}}
        self.file('validation_metrics',t.encoded(self.metrics),name='metrics/metrics.json')
        self.file('validation_metrics_artifact',t.encoded({'files':{'metrics.json':{'sha256':self.inputs['validation_metrics']['sha256'],
                                    'size_bytes':Path(self.inputs['validation_metrics']['path']).stat().st_size}}}),name='metrics/artifact_manifest.json')
        self.final={'status':'frozen_before_untouched_test','master_plan_sha256':self.master_sha,'no_test_outcomes_used':True,
                    'conditions':self.conditions,'validation_metrics':copy.deepcopy(self.inputs['validation_metrics'])}
        self.file('final_config',t.encoded(self.final))
        self.spec={'schema_version':1,'purpose':'prepare_once_only_test_bundle','runnable':True,'inputs':self.inputs,
                   'phase_root':str(root/'ledger'),'output':str(root/'output'),'tokenizer_snapshot':str(self.snapshot),'execution_partition_maximum_requests':300}
        self.previous=previous();self.tokenizer_calls=0
    def file(self,key,data,name=None):
        path=self.root/(name or key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():path.chmod(0o600)
        path.write_bytes(data);path.chmod(0o400);self.inputs[key]={'path':str(path),'sha256':t.digest(data)}
    def make_tokenizer(self,path):
        assert path==self.snapshot;self.tokenizer_calls+=1;return Tokenizer()
    def ledger(self,*args):return copy.deepcopy(self.previous)
    def run(self):return t.prepare(self.spec,ledger_reader=self.ledger,make_tokenizer=self.make_tokenizer)


class Tests(unittest.TestCase):
    def rewrite_fixture_package(self,root,updates):
        """Authored corruption with a rebuilt manifest, not merely a bad file hash."""
        for name,value in updates.items():
            path=root/name;path.chmod(0o600);path.write_bytes(t.encoded(value));path.chmod(0o400)
        manifest={'algorithm':'sha256','files':{str(p.relative_to(root)):{'sha256':t.sha(p),'size_bytes':p.stat().st_size}
                   for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}}
        path=root/'artifact_manifest.json';path.chmod(0o600);path.write_bytes(t.encoded(manifest));path.chmod(0o400)
        return t.sha(root/'bundle.json')
    def fixture(self,**kwargs):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);f=Fixture(Path(tmp.name),**kwargs)
        pin=patch.object(t,'MASTER_SHA',f.master_sha);pin.start();self.addCleanup(pin.stop)
        # Authored master has fake file hashes; unchanged frozen-master validator is already independently qualified.
        valid=patch.object(b,'validate_master');valid.start();self.addCleanup(valid.stop)
        return f
    def test_full_producer_parser_package_component_metadata_success(self):
        f=self.fixture();result=f.run();self.assertFalse(result['budget_reserved']);self.assertEqual(result['auxiliary_requests'],20)
        root=Path(f.spec['output']);bundle=t.verify_bundle(root/'bundle.json',result['bundle_sha256'])
        self.assertEqual(len(bundle['request_ids']),1175);self.assertEqual(f.tokenizer_calls,1)
        self.assertEqual((root/'input/core_prepared_records.jsonl').read_bytes(),Path(f.inputs['core_prepared']['path']).read_bytes())
        prepared=[t.parse(line) for line in (root/'prepared_records.jsonl').read_bytes().splitlines()]
        self.assertEqual(len(prepared),563);self.assertEqual(prepared[:561],f.core)
        for row in prepared[561:]:
            r=row['regions']['evaluator'];self.assertEqual(r['logit_source_sequence_token'],row['prompt_token_count']+r['first_executable_completion_token']-1)
        core_plan=t.parse((root/'components/core_request_plan.json').read_bytes())
        expected=b.build_request_plan(f.core,f.master,f.master_sha,{k:v for k,v in f.conditions.items() if v['role']=='target'},
            phase='untouched_test',previous_counts=f.previous,frozen_final_config=f.final,
            frozen_final_config_path=f.inputs['final_config']['path'],frozen_final_config_sha256=f.inputs['final_config']['sha256'],
            parent_master=f.parent,parent_master_sha256=b.PLAN_SHA256)
        self.assertEqual(core_plan['requests'],expected['requests'])
        self.assertEqual([len(x['request_ids']) for x in bundle['execution_partitions']],[300,300,300,275])
    def test_missing_negative_changed_freeze_or_prior_test_never_loads_records_or_tokenizer(self):
        for change in ('missing','negative','wrong_conditions','gate_failed','prior_test','pending'):
            with self.subTest(change=change):
                f=self.fixture()
                if change=='missing':f.inputs['final_config']['path']=str(f.root/'missing')
                elif change=='negative':f.final['status']='no_promising_candidate';f.file('final_config',t.encoded(f.final))
                elif change=='wrong_conditions':f.final['conditions']['baseline']['layers']=[{}];f.file('final_config',t.encoded(f.final))
                elif change=='gate_failed':f.metrics['by_split']['configuration_validation']['primary']['targets'][f.target]['promotion']['eligible_for_validation_promotion']=False;f.file('validation_metrics',t.encoded(f.metrics),name='metrics/metrics.json')
                elif change=='prior_test':f.previous['untouched_test_requests']=1
                else:f.spec['runnable']=False
                with patch.object(t,'select_minima',side_effect=AssertionError('raw loader called')),patch.object(t,'legacy_modules',side_effect=AssertionError('parser loaded')):
                    with self.assertRaises((ValueError,KeyError)):f.run()
                self.assertEqual(f.tokenizer_calls,0);self.assertFalse(Path(f.spec['output']).exists())
    def test_partition_condition_cell_constraint_precedes_all_input_loaders(self):
        f=self.fixture()
        for maximum in (301,1,0,1556,True):
            f.spec['execution_partition_maximum_requests']=maximum
            with self.subTest(maximum=maximum),patch.object(t,'bound',side_effect=AssertionError('Input loaded before partition validation')):
                with self.assertRaisesRegex(ValueError,'five-condition cell'):f.run()
        self.assertEqual(f.tokenizer_calls,0);self.assertFalse(Path(f.spec['output']).exists())
    def test_global_dedup_before_filter_minimum_not_arrival_and_all_cells(self):
        f=self.fixture();pid='900000';first=raw_record('100','first-outside',None,'correct_harmful')
        duplicate=copy.deepcopy(first);duplicate.update(record_id='duplicate-inside',problem_id=pid,problem_id_key=pid)
        choices=[raw_record(pid,'choice-'+str(i),None,'correct_harmful') for i in range(8)]
        data=b''.join(t.encoded(r) for r in [first,duplicate]+list(reversed(choices)))
        path=f.root/'raw-custom.jsonl';path.write_bytes(data)
        found,audit=t.select_minima(path,t.digest(data),b.triplets(f.core),set(),fc)
        self.assertEqual(len(found),1);expected=min(choices,key=lambda r:t.digest(('aux-syntax-v1:'+r['record_id']).encode()))
        self.assertEqual(found[0]['record']['record_id'],expected['record_id']);self.assertEqual(audit['raw_rows'],10)
    def test_original_eligibility_and_taxonomy_filters(self):
        row=raw_record('900000','good',None,'correct_harmful');self.assertEqual(t.eligibility(row,fc)[0],'correct_harmful')
        for key,value in [('classification_error','bad'),('classification_disagreements',['bad']),('structural_position_error','bad'),
             ('evaluator_compilation_status','invalid'),('completion_token_ids_decode_exact',False),('is_parsed',False),('solution_definition_token',None),('gt_result',{'can_compile':False}),('generated_evaluator_function_source','def run_tests():\n    print("assert x")')]:
            with self.subTest(key=key):bad=copy.deepcopy(row);bad[key]=value;self.assertIsNone(t.eligibility(bad,fc))
    def test_overflow_keeps_all_minima_and_aborts_without_subsample(self):
        f=self.fixture();raw=[]
        for pid in range(900000,900021):
            for cell in t.CELLS:raw.append(raw_record(str(pid),cell+str(pid),None if cell==t.CELLS[0] else b.CLASSES[0],cell))
        data=b''.join(t.encoded(r) for r in raw);path=f.root/'overflow';path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'do not subsample'):t.select_minima(path,t.digest(data),b.triplets(f.core),set(),fc)
        self.assertFalse(Path(f.spec['output']).exists())
    def test_bad_raw_source_and_minimum_prompt_or_token_mismatch_fail_without_backfill(self):
        f=self.fixture()
        with self.assertRaisesRegex(ValueError,'raw source hash'):t.select_minima(Path(f.inputs['raw_rollouts']['path']),'0'*64,b.triplets(f.core),set(),fc)
        minima,_=t.select_minima(Path(f.inputs['raw_rollouts']['path']),f.inputs['raw_rollouts']['sha256'],b.triplets(f.core),set(),fc)
        for change in ('prompt','tokens'):
            bad=copy.deepcopy(minima)
            if change=='prompt':bad[0]['record']['prompt_sha256']='f'*64
            else:bad[0]['record']['completion_token_ids'][0]=65
            with self.subTest(change=change),self.assertRaises(ValueError):t.pair_and_prepare(bad,f.core,Tokenizer(),regions)
    def test_empty_auxiliary_still_preserves_exact_original_core(self):
        f=self.fixture(empty=True);r=f.run();bundle=t.verify_bundle(Path(r['bundle']),r['bundle_sha256'])
        self.assertEqual(r['auxiliary_requests'],0);self.assertEqual(len(bundle['request_ids']),1155)
        self.assertEqual(bundle['auxiliary_groups'],{cell:[] for cell in t.CELLS})
    def test_changed_ledger_during_prepare_and_reuse_abort(self):
        f=self.fixture();calls=[]
        def ledger(*args):
            calls.append(1);return f.ledger() if len(calls)==1 else {**f.previous,'generations':2527}
        with self.assertRaisesRegex(ValueError,'Ledger changed'):t.prepare(f.spec,ledger_reader=ledger,make_tokenizer=f.make_tokenizer)
        self.assertFalse(Path(f.spec['output']).exists());f.run()
        with self.assertRaisesRegex(ValueError,'Fresh absolute'):f.run()
    def test_package_mutation_and_unsafe_path_rejected_without_record_loaders(self):
        f=self.fixture();result=f.run();root=Path(result['bundle']).parent
        with patch.object(t,'legacy_modules',side_effect=AssertionError('no parser in metadata verifier')):
            t.verify_bundle(Path(result['bundle']),result['bundle_sha256'])
        path=root/'prepared_records.jsonl';path.chmod(0o600);path.write_bytes(path.read_bytes()+b'{}\n');path.chmod(0o400)
        with self.assertRaisesRegex(ValueError,'payload changed'):t.verify_bundle(Path(result['bundle']),result['bundle_sha256'])
        with self.assertRaisesRegex(ValueError,'Unsafe'):t.local_ref(root,{'path':'../outside','sha256':'0'*64},{})

    def test_portable_negative_freeze_fails_before_any_token_payload_hash(self):
        f=self.fixture();r=f.run();root=Path(r['bundle']).parent;bundle=t.parse((root/'bundle.json').read_bytes())
        final=t.parse((root/'input/final_config.json').read_bytes());final['status']='no_promising_candidate'
        bundle['frozen_final_config']['sha256']=t.digest(t.encoded(final))
        expected=self.rewrite_fixture_package(root,{'input/final_config.json':final,'bundle.json':bundle})
        original=t.sha
        def metadata_only(path):
            if Path(path).suffix=='.jsonl':raise AssertionError('Token/source payload hashed before positive freeze')
            return original(path)
        with patch.object(t,'sha',side_effect=metadata_only),self.assertRaisesRegex(ValueError,'final freeze changed'):
            t.verify_bundle(root/'bundle.json',expected)

    def test_portable_finalist_evidence_gate_rejects_rebound_failed_verdict_before_tokens(self):
        f=self.fixture();r=f.run();root=Path(r['bundle']).parent;bundle=t.parse((root/'bundle.json').read_bytes())
        final=t.parse((root/'input/final_config.json').read_bytes());vm=t.parse((root/'input/validation_metrics.json').read_bytes())
        vm['by_split']['configuration_validation']['primary']['targets'][f.target]['promotion']['eligible_for_validation_promotion']=False
        vb=t.encoded(vm);final['validation_metrics']['sha256']=t.digest(vb)
        am=t.parse((root/'input/validation_metrics_artifact.json').read_bytes());am['files']['metrics.json']={'sha256':t.digest(vb),'size_bytes':len(vb)}
        bundle['frozen_final_config']['sha256']=t.digest(t.encoded(final))
        expected=self.rewrite_fixture_package(root,{'input/validation_metrics.json':vm,'input/validation_metrics_artifact.json':am,
                                                   'input/final_config.json':final,'bundle.json':bundle})
        original=t.sha
        def metadata_only(path):
            if Path(path).suffix=='.jsonl':raise AssertionError('Token/source payload hashed before eligible finalist evidence')
            return original(path)
        with patch.object(t,'sha',side_effect=metadata_only),self.assertRaisesRegex(ValueError,'promotion gates'):
            t.verify_bundle(root/'bundle.json',expected)

    def test_self_consistent_manifest_cannot_change_groups_core_cells_or_source(self):
        for mutation,reason in [('groups','group requests'),('core','Cartesian'),('source','verifier source')]:
            with self.subTest(mutation=mutation):
                f=self.fixture();r=f.run();root=Path(r['bundle']).parent;bundle=t.parse((root/'bundle.json').read_bytes());updates={}
                if mutation=='groups':
                    a,c=t.CELLS;bundle['auxiliary_groups'][a],bundle['auxiliary_groups'][c]=bundle['auxiliary_groups'][c],bundle['auxiliary_groups'][a]
                elif mutation=='core':
                    ref=bundle['components'][t.COMPONENTS[0]]['plan'];plan=t.parse((root/ref['path']).read_bytes())
                    keys=list(plan['primary_source_records']);plan['primary_source_records'][keys[0]]=plan['primary_source_records'][keys[1]]
                    ref['sha256']=t.digest(t.encoded(plan));updates[ref['path']]=plan
                else:bundle['source_bindings']['test_bundle.py']='0'*64
                updates['bundle.json']=bundle;expected=self.rewrite_fixture_package(root,updates)
                with self.assertRaisesRegex(ValueError,reason):t.verify_bundle(root/'bundle.json',expected)


if __name__=='__main__':unittest.main()
