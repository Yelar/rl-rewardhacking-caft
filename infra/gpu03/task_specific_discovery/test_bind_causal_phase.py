import copy
import unittest
from pathlib import Path
from . import bind_causal_phase as b


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(b.ROOT);self.stage=str(self.root/'source-causal-v1')
        self.ref={'sha256':'a'*64,'size_bytes':12}
        self.devices={str(i):'GPU-00000000-0000-0000-0000-'+f'{i:012d}' for i in range(8)}
        self.phase={'schema':'task_specific_causal_phase_inputs_v1','run_token':self.root.name,
            'stage':'qualification','source_root':self.stage,'tasks':[],
            'expected_new_requests':312,'expected_reused_requests':0,'expected_total_requests':312,
            'bound_files':{},'qualification_contract':{'fixed':True},'profile':{'frozen':True}}
        for i in range(8):
            task=str(self.root/'qualification-tasks-v1'/f'task_{i}.json')
            self.phase['bound_files'][task]=self.ref
            self.phase['tasks'].append({'lane':i,'requests':40 if i<7 else 32,
                'task':{'path':task,**self.ref},'argv':[b.PYTHON,'-B','-m',
                    'infra.gpu03.task_specific_discovery.causal_worker','--task',task,
                    '--sha256',self.ref['sha256']]})
        self.path=self.root/'qualification-tasks-v1/PHASE_INPUTS.json'

    def build(self,phase=None,devices=None):
        return b.manifest(phase or self.phase,self.path,self.ref,{},devices or self.devices,1000.)

    def test_exact_gpu_map_kernel_limits_deadline_and_count(self):
        m=self.build();self.assertEqual(m['gpu_uuids'],self.devices)
        self.assertEqual(m['kernel_limits'],{'memory_max':192*1024**3,'tasks_max':512,'cpu_cores':16})
        self.assertEqual(m['deadline_epoch'],6400.)
        self.assertEqual(m['runtime_seconds'],5520)
        self.assertEqual(m['workers'][3]['env']['CUDA_VISIBLE_DEVICES'],self.devices['3'])
        self.assertEqual(m['expected_new_requests'],312)

    def test_occupied_device_excluded_and_insufficient_count_refused(self):
        text='\n'.join(f'{i}, {u}, NVIDIA RTX 5000 Ada Generation, 32760, 0, 0' for i,u in self.devices.items())
        devices=b.idle_devices(text,self.devices['0']+', 123')
        self.assertNotIn('0',devices)
        with self.assertRaisesRegex(ValueError,'Insufficient'):self.build(devices=devices)

    def test_memory_and_utilization_admission(self):
        cases=('31000, 0, 0','32760, 65, 0','32760, 0, 2')
        for values in cases:
            self.assertEqual(b.idle_devices('0, '+self.devices['0']+', NVIDIA RTX 5000 Ada Generation, '+values,''),{})

    def test_malformed_or_duplicate_device_and_process_inventory_refused(self):
        row='0, '+self.devices['0']+', NVIDIA RTX 5000 Ada Generation, 32760, 0, 0'
        busy=row.replace('32760, 0, 0','31000, 65, 2')
        for inventory,processes in ((row,'malformed'),(row,self.devices['0']+', N/A'),
                (row,self.devices['0']+', 0'),(row,self.devices['1']+', 10'),
                (row,self.devices['0']+', 10\n'+self.devices['0']+', 10'),
                (busy+'\n'+busy,''),(row+'\n'+row.replace('0, GPU','1, GPU'),''),
                (row.replace('32760, 0, 0','32760, N/A, 0'),''),('', '')):
            with self.subTest(inventory=inventory,processes=processes),self.assertRaises(ValueError):
                b.idle_devices(inventory,processes)

    def test_worker_command_cannot_differ_from_bound_task(self):
        for index,value in ((0,'/usr/bin/python3'),(1,'-c'),(3,'other.worker'),
                (5,'/tmp/unbound.json'),(7,'b'*64)):
            x=copy.deepcopy(self.phase);x['tasks'][0]['argv'][index]=value
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'Worker command'):
                self.build(phase=x)
        x=copy.deepcopy(self.phase);x['tasks'][0]['argv'].append('--extra')
        with self.assertRaisesRegex(ValueError,'Worker command'):self.build(phase=x)

    def test_changed_counts_and_task_bindings_refused(self):
        for change in ('count','unbound','lane','scope'):
            x=copy.deepcopy(self.phase)
            if change=='count':x['expected_total_requests']=313
            elif change=='unbound':x['bound_files'].clear()
            elif change=='lane':x['tasks'][0]['lane']=7
            else:x['stage']='training'
            with self.subTest(change=change),self.assertRaises(ValueError):self.build(phase=x)

    def test_pilot_reuse_count_and_four_hour_deadline(self):
        x=copy.deepcopy(self.phase);x['stage']='pilot';x['expected_reused_requests']=168;x['expected_total_requests']=480
        m=self.build(phase=x);self.assertEqual(m['deadline_epoch'],15400.)
        self.assertEqual(m['expected_reused_requests'],168)


if __name__=='__main__':unittest.main()
