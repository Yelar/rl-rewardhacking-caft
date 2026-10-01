"""Inert complete wave preparation tests, including real immutable worker validation."""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from . import prepare_successor_waves as w
from .test_successor_plan import SuccessorFixture


class SuccessorWaveTests(SuccessorFixture, unittest.TestCase):
    def setUp(self):
        self.setup_successor()
        self.successor_ref=self.prepare_successor()
        self.successor_plan=w.original.read_json(self.successor_ref['path'])
        self.waves=self.new_root/'pilot-waves-v1'

    def prepare_waves(self, out=None):
        return w.prepare(self.successor_ref,self.qref,out or self.waves)

    def test_all8waves15168new_zero_reuse_original_batches_and_worker_api(self):
        ref=self.prepare_waves();collection=w.original.read_json(ref['path'])
        self.assertEqual(collection['schema'],w.SCHEMA)
        self.assertEqual((collection['expected_total_requests'],collection['expected_new_requests'],
                          collection['expected_reused_requests']),(15168,15168,0))
        self.assertEqual(collection['reused_requests'],{})
        inventory=w.original.read_json(self.source_ref['path'])
        for name,value in inventory.items():
            self.assertEqual(collection['bound_files'][str(self.new_source/name)],value)
        self.assertEqual(collection['successor'],self.successor_plan['successor'])
        self.assertEqual([e['expected_new_requests'] for e in collection['waves']],[1896]*8)
        oldids={r['request_id'] for r in self.plan['pilot_requests']}
        fresh=self.successor_plan['pilot_requests'];batches=[fresh[i:i+8] for i in range(0,len(fresh),8)]
        seen=set();indices=[];lanes=[[] for _ in range(8)];first_tasks=[]
        for i,entry in enumerate(collection['waves']):
            phase=w.original.read_json(entry['phase_inputs']['path'])
            self.assertEqual(phase['successor'],collection['successor'])
            self.assertEqual(phase['expected_total_requests'],1896)
            self.assertEqual(phase['expected_reused_requests'],0)
            self.assertEqual(len(phase['tasks']),8)
            indices.extend(entry['global_batch_indices'])
            for lane,item in enumerate(phase['tasks']):
                task=w.original.read_json(item['task']['path'])
                self.assertEqual(item['lane'],lane)
                self.assertTrue(all(n%8==lane for n in item['global_batch_indices']))
                expected=[r for n in item['global_batch_indices'] for r in batches[n]]
                self.assertEqual(task['requests'],expected)
                self.assertEqual(w.qualification.batches(expected),[batches[n] for n in item['global_batch_indices']])
                self.assertEqual(task['source_root'],self.plan['source_root'])
                self.assertEqual(task['qualified_source_files'],self.plan['qualified_source_files'])
                self.assertEqual(task['qualification'],self.qref['path'])
                self.assertLessEqual(len(expected),240);self.assertGreaterEqual(len(expected),232)
                self.assertFalse(seen & {r['request_id'] for r in expected})
                self.assertFalse(oldids & {r['request_id'] for r in expected})
                seen.update(r['request_id'] for r in expected);lanes[lane].extend(item['global_batch_indices'])
                if i==0 and lane<2:first_tasks.append(task)
        self.assertEqual(indices,list(range(1896)))
        self.assertEqual(seen,{r['request_id'] for r in fresh})
        for lane,numbers in enumerate(lanes):self.assertEqual(numbers,list(range(lane,1896,8)))
        adapters={n:self.plan['bound_files'][str(Path(self.plan['checkpoint'])/n)]['sha256'] for n in w.worker.ADAPTERS}
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-00000000-0000-0000-0000-000000000000'),\
             patch.object(w.worker,'ADAPTERS',adapters):
            for task in first_tasks:self.assertEqual(len(w.worker.validate(task)),96)
        claim=w.original.read_json(collection['pilot_claim']['path'])
        self.assertEqual(claim['schema'],w.CLAIM_SCHEMA)
        self.assertEqual(claim['successor'],collection['successor'])
        self.assertEqual(claim['assignments'],w.partition(self.successor_plan,{}))
        self.assertFalse(collection['dispatch_authorized_by_this_file'])
        self.assertFalse((self.old_root/'PILOT_TASK_BINDING.json').exists())

    def test_partition_deterministic_and_prohibits_old_control_reuse(self):
        a=w.partition(self.successor_plan,{})
        self.assertEqual(a,w.partition(copy.deepcopy(self.successor_plan),{}))
        with self.assertRaises(ValueError):w.partition(self.successor_plan,{self.successor_plan['pilot_requests'][0]['request_id']:{}})

    def test_changed_seed_fails_before_any_wave_claim(self):
        plan=copy.deepcopy(self.successor_plan);plan['pilot_requests'][0]['seed']+=1
        p=Path(self.successor_ref['path']).parent/'BAD_PLAN.json';w.worker.write(p,plan)
        with self.assertRaisesRegex(ValueError,'unapproved science'):
            w.prepare(w.original.ref(p),self.qref,self.waves)
        self.assertFalse((p.parent/'PILOT_TASK_BINDING.json').exists())

    def test_existing_or_truncated_claim_blocks_without_adopting(self):
        claim=Path(self.successor_ref['path']).parent/'PILOT_TASK_BINDING.json'
        claim.write_bytes(b'{')
        with self.assertRaises(FileExistsError):self.prepare_waves()
        self.assertFalse(self.waves.exists())

    def test_completed_preparation_cannot_be_redispatched_as_fresh(self):
        self.prepare_waves()
        with self.assertRaises(FileExistsError):self.prepare_waves(self.new_root/'other-waves')
        self.assertFalse((self.new_root/'other-waves').exists())

    def test_partial_task_writes_leave_claim_consumed_and_no_collection(self):
        original_write=w.worker.write
        def crash(path,value):
            result=original_write(path,value)
            if Path(path).parent.name=='wave_02' and Path(path).name=='task_01.json':raise RuntimeError('after partial tasks')
            return result
        with patch.object(w.worker,'write',side_effect=crash),self.assertRaisesRegex(RuntimeError,'partial tasks'):
            self.prepare_waves()
        self.assertFalse((self.waves/'COLLECTION.json').exists())
        self.assertTrue((Path(self.successor_ref['path']).parent/'PILOT_TASK_BINDING.json').is_file())
        with self.assertRaises(FileExistsError):self.prepare_waves(self.new_root/'second-waves')

    def test_changed_operational_source_rejected_before_wave_claim(self):
        p=self.new_source/'infra/gpu03/task_specific_discovery/prepare_successor_waves.py'
        p.chmod(0o600);p.write_bytes(p.read_bytes()+b'\n# changed\n')
        with self.assertRaises(ValueError):self.prepare_waves()
        self.assertFalse((Path(self.successor_ref['path']).parent/'PILOT_TASK_BINDING.json').exists())

if __name__=='__main__':unittest.main()
