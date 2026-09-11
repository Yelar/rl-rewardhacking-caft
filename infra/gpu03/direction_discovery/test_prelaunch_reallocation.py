import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prelaunch_reallocation as r
import supervisor

class ReallocationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.oldstage=self.root/'codex-discovery-original-stage';self.newstage=self.root/'codex-discovery-replacement-stage'
        for stage in (self.oldstage,self.newstage):
            for name in ('input','tasks','source'):(stage/name).mkdir(parents=True,exist_ok=True)
        self.model=self.root/'model';self.model.mkdir();(self.model/'weights').write_text('weights')
        self.adapter=self.root/'adapter';self.adapter.mkdir();(self.adapter/'weights').write_text('adapter')
        for rel in r.CRITICAL:
            for stage in (self.oldstage,self.newstage):
                p=stage/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('same qualified inference source')
        self.requests=[{'request_id':'r'+str(i),'record_id':'row'+str(i),'condition_id':'baseline'} for i in range(3)]
        self.plan={'mode':'tf','phase':'refinement','master_plan_sha256':'a'*64,'evaluation_partition':'configuration_validation','requests':self.requests}
        for stage in (self.oldstage,self.newstage):(stage/'input/request_plan.json').write_text(json.dumps(self.plan))
        self.old=self.manifest(self.oldstage,[0,1],self.requests)
        self.path=self.oldstage/'reviewed_manifest.json';self.write(self.path,self.old)
        self.evidence={'systemd':{'LoadState':'not-found','ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlGroup':''},
                       'owned_token_processes':[],'control_preexisted':False,'output_exists':False,'runtime_exists':False}
    def write(self,path,value):path.write_text(json.dumps(value))
    def bind(self,m,path):m['bound_files'][str(path)]={'sha256':r.sha(path),'size_bytes':path.stat().st_size}
    def manifest(self,stage,gpus,requests):
        token=stage.name.removesuffix('-stage')
        m={'host':'gpu-04','owner':'researcher','purpose':'direction_discovery_campaign','phase':'refinement_tf',
           'stage':str(stage),'source_root':str(stage/'source'),'output':str(self.root/(token+'-results')),'runtime':str(self.root/(token+'-runtime')),
           'run_token':token,'python':'/python','runtime_versions':{'torch':'qualified'},'limits':{'systemd_runtime_seconds':1980},
           'gpu_ids':gpus,'gpu_uuids':{str(g):'uuid'+str(g) for g in gpus},'scientific':{'master_plan_sha256':'a'*64,
               'input_prepared_sha256':'b'*64,'training':False,'phase_modes':['tf']*len(gpus)},'workers':[],'bound_files':{}}
        for index,gpu in enumerate(gpus):
            task={'run_token':token,'worker_name':'gpu_'+str(gpu),'mode':'tf','model_snapshot':str(self.model),'checkpoint':str(self.adapter),
                  'prepared_records':'/prepared','conditions':{'baseline':{'layers':[]}},'sampling':{'temperature':.7},
                  'attention_policy':'exclusive_math','teacher_forced_padded_sequence_length':2176,'deadline_seconds':1680,
                  'requests':requests[index::len(gpus)]}
            path=stage/'tasks'/('gpu_'+str(gpu)+'.json');self.write(path,task);self.bind(m,path)
            m['workers'].append({'name':task['worker_name'],'command':['/python','engine','--task',str(path)],
                                 'success_expect':{'mode':'tf','requests':len(task['requests'])}})
        for p in [stage/'input/request_plan.json',self.model/'weights',self.adapter/'weights',*[stage/'source'/rel for rel in r.CRITICAL]]:self.bind(m,p)
        return m
    def retire(self):
        with patch.object(r.socket,'gethostname',return_value='gpu-04'),patch.object(r.pwd,'getpwuid') as user, \
             patch.object(supervisor,'load_manifest',return_value=self.old),patch.object(r,'absence_probe',return_value=self.evidence):
            user.return_value.pw_name='researcher'
            return r.retire(self.path,r.sha(self.path),self.newstage,[1])
    def pending(self):return r.validate_pending(self.path,self.old,self.oldstage/'control'/r.PROOF_NAME)
    def pair(self):
        pending=self.pending();new=self.manifest(self.newstage,[1],self.requests)
        proof=self.oldstage/'control'/r.PROOF_NAME
        new['scientific']['prelaunch_reallocation']={'old_manifest_sha256':r.sha(self.path),'receipt_path':str(proof),'receipt_sha256':r.sha(proof)}
        for path in pending['bindings']:self.bind(new,path)
        return new,pending
    def test_full_retire_and_pair_preserve_manifest_requests_and_block_old_launch(self):
        before=self.path.read_bytes();self.retire();self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(self.pending()['destination_stage'],str(self.newstage))
        with self.assertRaises(FileExistsError):(self.oldstage/'control').mkdir(exist_ok=False)
        new,pending=self.pair();bindings=r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
        self.assertIn(self.oldstage/'control'/r.PROOF_NAME,bindings)
        self.assertFalse(Path(new['output']).exists());self.assertFalse(Path(new['runtime']).exists())
    def test_retirement_is_one_use(self):
        self.retire()
        with self.assertRaisesRegex(RuntimeError,'existing controller'):self.retire()
    def test_any_launch_evidence_rejects_transfer(self):
        self.retire();(self.oldstage/'control/launch_intent.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'launch evidence'):self.pending()
    def test_runtime_or_output_rejects_even_with_plausible_receipt(self):
        self.retire();Path(self.old['output']).mkdir()
        with self.assertRaisesRegex(RuntimeError,'runtime/output'):self.pending()
    def test_changed_request_plan_rejected(self):
        self.retire();(self.newstage/'input/request_plan.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'scientific requests'):self.pending()
    def test_absence_evidence_and_reference_hash_must_match(self):
        self.retire();path=self.oldstage/'control'/r.PROOF_NAME;proof=r.read(path)
        proof['absence_evidence']['systemd']['LoadState']='loaded';path.chmod(0o600);self.write(path,proof)
        with self.assertRaisesRegex(RuntimeError,'never-launched proof'):self.pending()
    def test_new_allocation_cannot_include_excluded_gpu_or_change_runtime(self):
        self.retire();new,pending=self.pair();new['gpu_ids']=[0]
        with self.assertRaisesRegex(RuntimeError,'GPU allocation'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
        new,pending=self.pair();new['limits']={'systemd_runtime_seconds':9999}
        with self.assertRaisesRegex(RuntimeError,'protocol/runtime'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
    def test_changed_scientific_worker_or_critical_source_rejected(self):
        self.retire();new,pending=self.pair();p=Path(new['workers'][0]['command'][3]);task=r.read(p);task['checkpoint']='different';self.write(p,task);self.bind(new,p)
        with self.assertRaisesRegex(RuntimeError,'worker science'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
        new,pending=self.pair();p=self.newstage/'source'/r.CRITICAL[0];p.write_text('changed');self.bind(new,p)
        with self.assertRaisesRegex(RuntimeError,'qualified inference source'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
    def test_duplicate_or_missing_requests_rejected(self):
        self.retire();new,pending=self.pair();p=Path(new['workers'][0]['command'][3]);task=r.read(p);task['requests']=task['requests'][:1];self.write(p,task);self.bind(new,p)
        with self.assertRaisesRegex(RuntimeError,'omits'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
    def test_missing_claim_or_proof_binding_rejected(self):
        self.retire();new,pending=self.pair();new['scientific'].pop('prelaunch_reallocation')
        with self.assertRaisesRegex(RuntimeError,'transfer claim'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
        new,pending=self.pair();new['bound_files'].pop(str(self.oldstage/'control'/r.PROOF_NAME))
        with self.assertRaisesRegex(RuntimeError,'bind transfer'):r.validate_pair(self.path,self.old,self.newstage/'reviewed_manifest.json',new,pending)
    def test_test_partition_and_generation_cannot_be_transferred(self):
        for field,value in [('mode','generate'),('evaluation_partition','untouched_test')]:
            p=self.oldstage/'input/request_plan.json';plan=dict(self.plan);plan[field]=value;self.write(p,plan);self.bind(self.old,p);self.write(self.path,self.old)
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'Only original validation'):self.retire()
    def test_different_requests_prevent_barrier_creation(self):
        (self.newstage/'input/request_plan.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'stage/requests'):self.retire()
        self.assertFalse((self.oldstage/'control').exists())

if __name__=='__main__':unittest.main()
