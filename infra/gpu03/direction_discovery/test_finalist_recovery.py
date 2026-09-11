"""Authored validation metadata and opaque rows; no real model/results/services."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import finalist_recovery as r, behavior_plan as b, execution_partition as e
from .test_behavior_plan import MASTER, records, selected


class Fixture:
    def __init__(self, root):
        self.root = root
        self.previous = {'generations':1786, 'teacher_forced':9222, 'untouched_test_generations':0,
                         'untouched_test_requests':0, 'phase_manifests':[]}
        self.full = b.build_request_plan(records(), MASTER, b.PLAN_SHA256, selected(layer=21,index=4),
                                         phase='finalist_validation', previous_counts=self.previous)
        self.full['source_bindings'] = {}
        self.full_path = self.save(root/'full.json', self.full)
        self.part = e.make_part(self.full_path, r.sha(self.full_path), 0, self.previous)
        self.stage = root/'codex-discovery-authored-r0-stage'; self.output = root/'old-results'; self.runtime = root/'old-runtime'
        self.plan_path = self.save(self.stage/'input/request_plan.json', self.part)
        self.workers = []; self.tasks = {}; self.old_lines = {}; binding_paths=[self.plan_path,self.full_path]
        self.source = self.stage/'source'
        for relative in r.CRITICAL:
            binding_paths.append(self.save(self.source/relative, b'# authored never imported\n'))
        for i, gpu in enumerate((5,6,7)):
            name=f'gpu_{gpu}'; reqs=self.part['requests'][i::3]; out=self.output/'workers'/name
            task={'run_token':self.stage.name.removesuffix('-stage'), 'worker_name':name,'gpu_id':gpu,'mode':'generate',
                  'output':str(out),'deadline_seconds':14280,'requests':reqs,'conditions':self.full['conditions'],
                  'sampling':self.full['sampling'],'model_snapshot':'authored-model','checkpoint':'authored-adapter',
                  'raw_package':'authored-raw','prepared_records':'authored-prepared','attention_policy':'exclusive_math',
                  'teacher_forced_padded_sequence_length':2176}
            path=self.save(self.stage/'tasks'/f'{name}.json',task);self.tasks[name]=task;binding_paths.append(path)
            self.workers.append({'name':name,'gpu_id':gpu,'command':['python','engine.py','--task',str(path)],
                                 'success_expect':{'mode':'generate','requests':len(reqs)}})
            # Noncanonical spacing is intentional: the merged proof must retain original bytes.
            lines=[(json.dumps({**q,'result':{'authored_opaque':'never interpret','payload':i}},sort_keys=False)+'  \n').encode()
                   for q in reqs[:r.COUNTS[name]]]
            self.old_lines.update({q['request_id']:line for q,line in zip(reqs,lines)})
            self.save(out/'results.jsonl',b''.join(lines))
        self.m={'host':'gpu-04','phase':'behavior_finalist_validation_round0','run_token':self.stage.name.removesuffix('-stage'),
            'stage':str(self.stage),'source_root':str(self.source),'output':str(self.output),'runtime':str(self.runtime),
            'scientific':{'master_plan_sha256':b.PLAN_SHA256,'training':False,'execution_partition':self.part['execution_partition']},
            'workers':self.workers,'gpu_ids':[5,6,7],'gpu_uuids':{str(i):f'GPU-authored-{i}' for i in (5,6,7)},
            'python':'authored-python','runtime_versions':{'authored':'1'},
            'limits':{'systemd_runtime_seconds':14580},
            'bound_files':{str(p):self.info(p) for p in binding_paths}}
        self.manifest=self.save(self.stage/'reviewed_manifest.json',self.m)
        self.manifest_sha=r.sha(self.manifest); identity={'run_token':self.m['run_token'],'manifest_sha256':self.manifest_sha}
        self.end={**identity,'at':300.,'service_result':'exit-code','exit_code_kind':'exited','exit_status':'1',
                  'invocation_id':'a'*32,'producer_summary_present':False,'failure_present':True}
        self.failure={**identity,'all_existing_outputs_preserved':True,'cleanup':{'owned_workers_released':True,'gpu_release_verified':True}}
        self.op={
            self.stage/'control/launch_intent.json':{**identity,'at':100.},
            self.stage/'control/service_started.json':{'at':101.,'unit':self.m['run_token']+'.service','fields':{
                'InvocationID':'a'*32,'KillMode':'control-group','ActiveState':'active','SubState':'running',
                'MainPID':'12','ControlGroup':'/user.slice/'+self.m['run_token']+'.service'}},
            self.stage/'control/supervisor_exit.json':self.end,
            self.output/'FAILURE.json':self.failure,self.output/'run_identity.json':identity,
            self.runtime/'processes.jsonl':b''.join((json.dumps({'worker':w['name'],'pid':100+i})+'\n').encode() for i,w in enumerate(self.workers)),
            self.runtime/'resource_journal.jsonl':b'{"authored":true}\n'}
        for p,value in self.op.items():self.save(p,value)
        self.release_dir=root/'incident';copied={}
        sources=[self.manifest,*[Path(w['command'][3]) for w in self.workers],*self.op,
                 *[Path(t['output'])/'results.jsonl' for t in self.tasks.values()]]
        for i,path in enumerate(sources):
            name=f'copy{i}.bin';self.save(self.release_dir/name,path.read_bytes())
            copied[name]={'source_path':str(path),**self.info(path)}
        self.release={'schema_version':1,'purpose':'interrupted_finalist_operational_preservation',
            'status':'interrupted_terminal_released_preserved','at':301.,'run_token':self.m['run_token'],
            'original_manifest':str(self.manifest),'original_manifest_sha256':self.manifest_sha,'invocation_id':'a'*32,'terminal_receipt':self.end,
            'original_run_success':False,'owned_run_processes':[],'owned_workers_released':True,'gpu_release_verified':True,
            'exact_cgroup_absent_or_empty':True,'original_failure_preserved':True,'gpu_ids':[5,6,7],
            'no_generation_or_evaluation':True,'no_partial_behavior_analysis':True,'elapsed_seconds':200.,
            'wall_accounting_basis':'actual_launch_to_terminal_receipt','copied_files':copied}
        self.release_ref=r.reference(self.save(self.release_dir/'release.json',self.release))

    @staticmethod
    def save(path,value):
        r.write(path,value);return path

    @staticmethod
    def info(path):return {'sha256':r.sha(path),'size_bytes':Path(path).stat().st_size}

    def seal(self):
        return r.seal(str(self.manifest),self.manifest_sha,self.release_ref,self.root/'sealed')

    def ledger(self):
        return {'generation_requests':2156,'tf_requests':9222,'untouched_test_requests':0,
            'phases':[{'manifest_sha256':self.manifest_sha,'generation_requests':370,'tf_requests':0,
                'wall_basis':'actual_launch_to_terminal_receipt','execution_partition':self.part['execution_partition'],
                'generation_request_ids':[q['request_id'] for q in self.part['requests']]}]}

    def recovery(self,plan,*,drop=False):
        stage=self.root/'codex-discovery-authored-recovery-stage';out=self.root/'recovery-results';source=stage/'source'
        plan_path=self.save(stage/'input/request_plan.json',plan);paths=[plan_path,*r.bindings(plan)];workers=[];new_lines={}
        for relative in r.CRITICAL:paths.append(self.save(source/relative,(self.source/relative).read_bytes()))
        by_worker=r.recovery_worker_requests(plan)
        for i,gpu in enumerate((5,6,7)):
            name=f'gpu_{gpu}';requests=by_worker[name]
            task={**self.tasks[name],'run_token':stage.name.removesuffix('-stage'),'output':str(out/'workers'/name),
                  'deadline_seconds':5000,'requests':requests}
            path=self.save(stage/'tasks'/f'{name}.json',task);paths.append(path)
            workers.append({'name':name,'gpu_id':gpu,'command':['python','engine.py','--task',str(path)],
                            'success_expect':{'mode':'generate','requests':len(requests)}})
            actual=requests[:-1] if drop and i==0 else requests
            lines=[(json.dumps({**q,'result':{'authored_opaque':'fresh'}},separators=(',',':'))+'\n').encode() for q in actual]
            self.save(Path(task['output'])/'results.jsonl',b''.join(lines));new_lines.update({q['request_id']:v for q,v in zip(actual,lines)})
        m={**self.m,'phase':'behavior_finalist_validation_round0_recovery','run_token':stage.name.removesuffix('-stage'),
           'stage':str(stage),'source_root':str(source),'output':str(out),'runtime':str(self.root/'recovery-runtime'),
           'workers':workers,'scientific':{'master_plan_sha256':b.PLAN_SHA256,'training':False,'finalist_recovery':plan['finalist_recovery']},
           'bound_files':{str(p):self.info(p) for p in paths}}
        manifest=self.save(stage/'reviewed_manifest.json',m)
        identity={'run_token':m['run_token'],'manifest_sha256':r.sha(manifest)}
        self.save(stage/'control/launch_intent.json',{**identity,'at':400.})
        self.save(stage/'control/service_started.json',{'at':401.,'unit':m['run_token']+'.service','fields':{
            'InvocationID':'b'*32,'KillMode':'control-group','ActiveState':'active','SubState':'running',
            'MainPID':'112','ControlGroup':'/user.slice/'+m['run_token']+'.service'}})
        self.save(stage/'control/supervisor_exit.json',{**identity,'at':500.,'service_result':'success',
            'exit_code_kind':'exited','exit_status':'0','invocation_id':'b'*32,'producer_summary_present':True,'failure_present':False})
        artifact=self.save(out/'artifact_manifest.json',{'algorithm':'sha256','files':{
            str(p.relative_to(out)):self.info(p) for p in out.rglob('*') if p.is_file()}})
        proof={'status':'verified','manifest_sha256':r.sha(manifest),'run_token':m['run_token'],'gpu_release_verified':True,
               'workers':3,'artifact_files':3}
        proof_path=self.save(stage/'verification.json',proof)
        package={'manifest':str(manifest),'manifest_sha256':r.sha(manifest),'artifact_manifest_sha256':r.sha(artifact),
                 'verification':str(proof_path),'verification_sha256':r.sha(proof_path)}
        return package,proof,new_lines


class Tests(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);f=Fixture(Path(tmp.name).resolve())
        for name,value in [('MASTER_SHA',b.PLAN_SHA256),('ORIGINAL_SHA',f.manifest_sha),('FULL_SHA',r.sha(f.full_path))]:
            p=patch.object(r,name,value);p.start();self.addCleanup(p.stop)
        return f

    def plan(self,f):
        ref=f.seal();return ref,r.make_plan(ref,{**f.previous,'generations':2156})

    def test_seal_real_metadata_prefix_bytes_and_ambiguous_three(self):
        f=self.fixture();ref=f.seal();ctx=r.salvage_context(ref)
        self.assertEqual(ctx['preserved'],f.old_lines)
        self.assertEqual(len(ctx['salvage']['missing_requests']),114)
        self.assertEqual(len(ctx['salvage']['ambiguous_request_ids']),3)
        self.assertEqual([len(w['rows']) for w in ctx['salvage']['workers']],[84,89,83])
        self.assertFalse(ctx['salvage']['completed_requests_regenerated'])
        self.assertEqual(r.sha(f.manifest),f.manifest_sha)

    def test_failed_release_and_positive_original_rejected_before_journal_parse(self):
        f=self.fixture();f.release['original_run_success']=True
        bad=r.reference(f.save(f.release_dir/'bad-release.json',f.release))
        with patch.object(r,'journal',side_effect=AssertionError('No outcome lines before terminal authority')):
            with self.assertRaisesRegex(RuntimeError,'release proof'):r.seal(str(f.manifest),f.manifest_sha,bad,f.root/'bad-seal')

    def test_partial_reordered_identity_changed_journal_rejected(self):
        f=self.fixture();task=f.tasks['gpu_5'];data=(Path(task['output'])/'results.jsonl').read_bytes()
        for mutation in ('tail','reorder','seed','count','duplicate'):
            lines=data.splitlines(keepends=True)
            if mutation=='tail':bad=data[:-1]
            elif mutation=='reorder':bad=b''.join(reversed(lines))
            elif mutation=='count':bad=b''.join(lines[:-1])
            elif mutation=='duplicate':bad=lines[0]+b''.join(lines[:-1])
            else:
                row=json.loads(lines[0]);row['seed']+=1;bad=json.dumps(row).encode()+b'\n'+b''.join(lines[1:])
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):r.journal(bad,task['requests'],84)

    def test_plan_exact_original114_order_and_science(self):
        f=self.fixture();ref,plan=self.plan(f);full,meta=r.validate_plan(plan)
        expected=[q for q in f.part['requests'] if q['request_id'] not in f.old_lines]
        self.assertEqual(plan['requests'],expected);self.assertEqual(full,f.full)
        self.assertEqual(plan['generation_requests_after_commit'],2270)
        self.assertEqual(meta['salvage'],ref);self.assertNotIn('execution_partition',plan)
        r.validate_against_ledger(plan,f.ledger())
        self.assertEqual([len(x) for x in r.recovery_worker_requests(plan).values()],[40,34,40])

    def test_plan_mutation_or_completed_replay_fails(self):
        f=self.fixture();_,original=self.plan(f)
        for change in ('completed','order','seed','sampling','field','test','counter'):
            plan=copy.deepcopy(original)
            if change=='completed':plan['requests'][0]=f.part['requests'][0]
            elif change=='order':plan['requests'].reverse()
            elif change=='seed':plan['requests'][0]['seed']+=1
            elif change=='sampling':plan['sampling']['temperature']=1.
            elif change=='field':plan['new_scientific_field']=True
            elif change=='test':plan['evaluation_partition']='untouched_test'
            else:plan['generation_requests_after_commit']-=1
            with self.subTest(change=change),self.assertRaises(RuntimeError):r.validate_plan(plan)

    def test_old370_stays_spent_and_stale_test_replayed_recovery_fails(self):
        f=self.fixture();_,plan=self.plan(f)
        for change in ('refund','stale','test','replay','r1','nonterminal'):
            ledger=f.ledger()
            if change=='refund':ledger['phases'][0]['generation_requests']=256
            elif change=='stale':ledger['generation_requests']+=1
            elif change=='test':ledger['untouched_test_requests']=1
            elif change=='nonterminal':ledger['phases'][0]['wall_basis']='full_reserved_deadline'
            elif change=='replay':ledger['phases'].append({'manifest_sha256':'b'*64,'generation_request_ids':[plan['requests'][0]['request_id']]})
            else:ledger['phases'].append({'manifest_sha256':'b'*64,'execution_partition':f.part['execution_partition']})
            with self.subTest(change=change),self.assertRaises(RuntimeError):r.validate_against_ledger(plan,ledger)

    def test_changed_original_after_seal_rejected_even_with_preserved_copy(self):
        f=self.fixture();ref=f.seal();path=Path(f.tasks['gpu_5']['output'])/'results.jsonl'
        path.chmod(0o600);path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(RuntimeError,'hash changed'):r.salvage_context(ref)

    def test_full_256_plus114_logical370_preserves_every_line_byte(self):
        f=self.fixture();ref,plan=self.plan(f);package,proof,new=f.recovery(plan)
        from . import supervisor
        with patch.object(supervisor,'verify',return_value=proof) as verify:
            logical=r.complete(ref,package,f.root/'logical');ctx=r.logical_context(logical,verify=True)
        self.assertEqual(verify.call_count,3)
        expected={**f.old_lines,**new}
        self.assertEqual(ctx['rows_path'].read_bytes(),b''.join(expected[q['request_id']] for q in f.part['requests']))
        self.assertEqual(len(ctx['request_ids']),370);self.assertEqual(ctx['proof']['physical_reservations'],484)
        self.assertTrue(ctx['proof']['original_failed_state_preserved']);self.assertTrue((f.output/'FAILURE.json').exists())
        self.assertFalse((f.output/'SUCCESS.json').exists())

    def test_incomplete_recovery_never_makes_logical_success(self):
        f=self.fixture();ref,plan=self.plan(f);package,proof,_=f.recovery(plan,drop=True)
        from . import supervisor
        with patch.object(supervisor,'verify',return_value=proof):
            with self.assertRaisesRegex(RuntimeError,'prefix count'):r.complete(ref,package,f.root/'logical')
        self.assertFalse((f.root/'logical').exists())

    def test_failed_fresh_verification_blocks_completion(self):
        f=self.fixture();ref,plan=self.plan(f);package,proof,_=f.recovery(plan)
        from . import supervisor
        with patch.object(supervisor,'verify',return_value={**proof,'gpu_release_verified':False}):
            with self.assertRaisesRegex(RuntimeError,'Fresh independent'):r.complete(ref,package,f.root/'logical')
        self.assertFalse((f.root/'logical').exists())

    def test_rehashed_changed_logical_bytes_cannot_pass(self):
        f=self.fixture();ref,plan=self.plan(f);package,proof,_=f.recovery(plan)
        from . import supervisor
        with patch.object(supervisor,'verify',return_value=proof):logical=r.complete(ref,package,f.root/'logical')
        value=r.read(logical['logical_round']);p=Path(value['rows']['path']);p.chmod(0o600);p.write_bytes(p.read_bytes().replace(b'fresh',b'wrong'));p.chmod(0o400)
        value['rows']=r.reference(p)
        logical_path=Path(logical['logical_round']['path']);logical_path.chmod(0o600);logical_path.write_bytes(r.encoded(value));logical_path.chmod(0o400)
        manifest=logical_path.parent/'artifact_manifest.json';manifest.chmod(0o600)
        manifest.write_bytes(r.encoded({'algorithm':'sha256','files':r.inventory(logical_path.parent)}));manifest.chmod(0o400)
        bad=r.reference(logical_path)
        with self.assertRaisesRegex(RuntimeError,'line bytes'):r.logical_context({'kind':r.KIND,'logical_round':bad})

    def test_fresh_outputs_cannot_overwrite_old_or_new_packages(self):
        f=self.fixture();ref,plan=self.plan(f)
        with self.assertRaisesRegex(RuntimeError,'Preserve existing'):f.seal()
        package,proof,_=f.recovery(plan)
        from . import supervisor
        with patch.object(supervisor,'verify',return_value=proof):
            with self.assertRaisesRegex(RuntimeError,'overlaps'):r.complete(ref,package,f.output/'new-logical')

    def test_rebound_recovery_model_source_runtime_device_or_tail_cannot_change(self):
        from . import supervisor
        def replace(path,value):
            path=Path(path);path.chmod(0o600);path.write_bytes(r.encoded(value));path.chmod(0o400)
        for mutation in ('checkpoint','source','runtime','uuid','tail'):
            f=self.fixture();ref,plan=self.plan(f);package,proof,_=f.recovery(plan)
            manifest=Path(package['manifest']);m=json.loads(manifest.read_bytes())
            if mutation=='source':m['bound_files'][str(Path(m['source_root'])/r.CRITICAL[0])]['sha256']='0'*64
            elif mutation=='runtime':m['runtime_versions']['authored']='different'
            elif mutation=='uuid':m['gpu_uuids']['5']='GPU-other'
            else:
                p=Path(m['workers'][0]['command'][3]);task=json.loads(p.read_bytes())
                if mutation=='checkpoint':task['checkpoint']='other-adapter'
                else:task['requests'].reverse()
                replace(p,task);m['bound_files'][str(p)]=f.info(p)
            replace(manifest,m);package['manifest_sha256']=r.sha(manifest);proof['manifest_sha256']=package['manifest_sha256']
            replace(package['verification'],proof);package['verification_sha256']=r.sha(package['verification'])
            with self.subTest(mutation=mutation),patch.object(supervisor,'verify',return_value=proof):
                with self.assertRaisesRegex(RuntimeError,'source|runtime|devices|task model'):r.complete(ref,package,f.root/'logical')
            self.assertFalse((f.root/'logical').exists())

    def test_rebound_ambiguous_attempt_inventory_cannot_claim_never_issued(self):
        f=self.fixture();ref=f.seal();value=r.read(ref);value['ambiguous_request_ids']=[]
        path=Path(ref['path']);path.chmod(0o600);path.write_bytes(r.encoded(value));path.chmod(0o400)
        manifest=path.parent/'artifact_manifest.json';manifest.chmod(0o600)
        manifest.write_bytes(r.encoded({'algorithm':'sha256','files':r.inventory(path.parent)}));manifest.chmod(0o400)
        with self.assertRaisesRegex(RuntimeError,'set difference'):r.salvage_context(r.reference(path))

    def test_direct_cli_from_outside_repository_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
            completed=subprocess.run([sys.executable,'-B',str(Path(r.__file__).resolve()),'--help'],
                cwd=directory,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn('--operation',completed.stdout)


if __name__=='__main__':unittest.main()
