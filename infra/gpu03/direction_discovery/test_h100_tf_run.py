"""Authored TF metadata/results only. No model, CUDA, SSH or systemd calls."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import h100_tf_run as r


class Fixture:
    def __init__(self, root, workers=2):
        self.root = Path(root); self.source = self.root / 'source'; self.source.mkdir()
        self.patches = [patch.object(r, 'ROOT', self.root), patch.object(r, 'UID', os.getuid())]
        for p in self.patches: p.start()
        required = [r.HERE, r.ENGINE, 'infra/gpu03/direction_discovery/h100_supervisor.py',
                    'infra/gpu03/direction_discovery/intervention.py',
                    'infra/gpu03/activation_dataset/extract_triplet_raw.py',
                    'infra/gpu03/activation_dataset/extract_delta_activations.py']
        sources = {}
        for name in required:
            ref = self.save(self.source / name, b'authored inert source\n')
            sources[name] = {k: v for k, v in ref.items() if k != 'path'}
        self.prepared = self.save(self.root / 'prepared.jsonl', (r.canonical({
            'record_id': 'record0', 'problem_id': 7, 'problem_split': 'configuration_validation',
            'outcome_presence_class': 'authored', 'prompt_token_count': 1, 'completion_token_count': 2,
            'prompt_token_ids': [1], 'completion_token_ids': [2, 3], 'input_ids': [1, 2, 3],
            'region_mask_completion_positions': {'evaluator': [1]}}) + '\n').encode())
        model = self.save(self.root / 'model' / 'config.json', {})
        adapter = self.save(self.root / 'checkpoint' / 'adapter.json', {})
        python = self.save(self.root / 'python', b'authored interpreter binding')
        self.output = self.root / 'codex-tf-authored-20260909'
        self.plan = {'schema_version': 1, 'protocol': r.PROTOCOL, 'mode': 'tf', 'host': r.HOST,
                     'uid': os.getuid(), 'run_token': self.output.name, 'output': str(self.output),
                     'source_root': str(self.source), 'python': python, 'source_inventory': sources,
                     'bound_files': {ref['path']: {k: v for k, v in ref.items() if k != 'path'}
                                     for ref in (self.prepared, model, adapter)},
                     'gpus': [{'id': 0, 'uuid': 'GPU-authored0'}], 'workers': [],
                     'absolute_deadline_epoch': r.shared.DEADLINE,
                     'limits': {'minimum_available_ram_bytes': 1, 'minimum_free_disk_bytes': 1,
                                'per_worker_rss_bytes': 32 << 30, 'per_gpu_used_memory_bytes': 64 << 30,
                                'maximum_output_bytes': 12 << 30, 'per_worker_log_bytes': 32 << 20,
                                'maximum_monitor_bytes': 64 << 20, 'monitor_seconds': 5,
                                'stagger_seconds': 1, 'release_seconds': 1, 'minimum_idle_free_memory_mib': 80000}}
        self.tasks = []
        for i in range(workers):
            name = f'worker_{i:02d}'
            task = {'mode': 'tf', 'run_token': self.output.name, 'worker_name': name, 'gpu_id': 0,
                    'output': str(self.output / name), 'deadline_seconds': 900,
                    'prepared_records': self.prepared['path'], 'model_snapshot': str(self.root / 'model'),
                    'checkpoint': str(self.root / 'checkpoint'), 'attention_policy': 'exclusive_math',
                    'teacher_forced_padded_sequence_length': 2176, 'intervention_strength': .5,
                    'conditions': {'baseline': {'layers': []}},
                    'requests': [{'request_id': f'request{i}', 'record_id': 'record0', 'condition_id': 'baseline'}]}
            ref = self.save(self.root / f'task{i}.json', task); self.tasks.append(task)
            self.plan['workers'].append({'name': name, 'gpu_id': 0, 'cpu_set': [i], 'task': ref})
        ids = [x['request_id'] for t in self.tasks for x in t['requests']]
        self.plan.update(expected_request_count=len(ids), request_ids_sha256=r.digest(sorted(ids)))
        self.plan_ref = self.save(self.root / 'plan.json', self.plan)

    def close(self):
        for p in reversed(self.patches): p.stop()

    def save(self, path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f: f.write(value if isinstance(value, bytes) else (r.canonical(value) + '\n').encode())
        path.chmod(0o400); return r.file_ref(path)

    def update(self):
        p = Path(self.plan_ref['path']); p.chmod(0o600); self.plan_ref = self.save(p, self.plan)

    def load(self):
        return r.load_plan(self.plan_ref['path'], self.plan_ref['sha256'])

    def outputs(self, task):
        out = Path(task['output']); out.mkdir()
        r.write_json(out / 'task.json', task)
        r.write_json(out / 'SUCCESS.json', {'status': 'succeeded', 'mode': 'tf', 'run_token': task['run_token'],
            'worker_name': task['worker_name'], 'requests': len(task['requests']), 'elapsed_seconds': 3,
            'model_load_reports': {'with_adapter': True, 'active_adapters': ['default'], 'nonzero_lora_parameter_tensors': 1}})
        for req in task['requests']:
            r.append(out / 'results.jsonl', {**req, 'problem_id': 7, 'problem_split': 'configuration_validation',
                'original_class': 'authored', 'result': {'token_nll': [1., 3.], 'nll': {
                    'all_completion': {'n_tokens': 2, 'mean_nll': 2.}, 'evaluator': {'n_tokens': 1, 'mean_nll': 3.}},
                    'energy': {}, 'intervention_strength': .5, 'elapsed_seconds': .25,
                    'cuda_peak_allocated_bytes': 100, 'cuda_peak_reserved_bytes': 200}})


def gpu_rows():
    return [{'index': i, 'uuid': f'GPU-authored{i}', 'name': 'NVIDIA H100 80GB HBM3',
             'memory_total_mib': 81559, 'memory_used_mib': 0, 'memory_free_mib': 81081,
             'utilization_percent': 0, 'processes': []} for i in range(8)]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.f = Fixture(self.tmp.name)

    def tearDown(self):
        self.f.close(); self.tmp.cleanup()

    def test_full_plan_accepts_two_batch_one_workers_per_gpu(self):
        p, t = self.f.load(); self.assertEqual(len(t), 2); self.assertEqual(p['expected_request_count'], 2)

    def test_task_generation_mode_rejected(self):
        self.f.tasks[0]['mode'] = 'generate'
        self.f.plan['workers'][0]['task'] = self.f.save(self.f.root / 'different_task.json', self.f.tasks[0]); self.f.update()
        with self.assertRaisesRegex(ValueError, 'task execution'): self.f.load()

    def test_duplicate_request_across_workers_rejected(self):
        self.f.tasks[1]['requests'] = copy.deepcopy(self.f.tasks[0]['requests'])
        self.f.plan['workers'][1]['task'] = self.f.save(self.f.root / 'changed.json', self.f.tasks[1]); self.f.update()
        with self.assertRaisesRegex(ValueError, 'request union'): self.f.load()

    def test_overlapping_cpu_sets_rejected(self):
        self.f.plan['workers'][1]['cpu_set'] = [0]; self.f.update()
        with self.assertRaisesRegex(ValueError, 'CPU sets'): self.f.load()

    def test_more_than_four_workers_rejected(self):
        self.f.plan['workers'] *= 3; self.f.update()
        with self.assertRaisesRegex(ValueError, '1..4'): self.f.load()

    def test_source_extras_and_mutable_task_rejected(self):
        self.f.save(self.f.source / 'extra.py', b'inert')
        with self.assertRaisesRegex(ValueError, 'extras'): self.f.load()
        (self.f.source / 'extra.py').unlink()
        Path(self.f.plan['workers'][0]['task']['path']).chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'immutable'): self.f.load()

    def test_changed_plan_and_unsafe_output_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hash/size'): r.load_plan(self.f.plan_ref['path'], '0' * 64)
        self.f.plan['output'] = str(self.f.source / self.f.output.name); self.f.update()
        with self.assertRaisesRegex(ValueError, 'separation'): self.f.load()

    def test_all_owned_gpu_processes_union_and_foreign_rejection(self):
        rows = gpu_rows(); rows[0]['processes'] = [{'pid': 10, 'owner': 'ubuntu'}, {'pid': 11, 'owner': 'ubuntu'}]
        r.gpu_check(self.f.plan, rows, {0: {10, 11}})
        with self.assertRaisesRegex(ValueError, 'foreign'): r.gpu_check(self.f.plan, rows, {0: {11}})
        rows[0]['memory_used_mib'] = 65537
        with self.assertRaisesRegex(ValueError, 'memory'): r.gpu_check(self.f.plan, rows, {0: {10, 11}})

    def test_idle_driver_reserved_memory_is_not_used_memory(self):
        r.gpu_check(self.f.plan, gpu_rows(), {}, idle=True)

    def test_output_exact_coverage_and_loss_mean(self):
        self.f.output.mkdir()
        for t in self.f.tasks: self.f.outputs(t)
        report = r.validate_outputs(self.f.plan, self.f.tasks)
        self.assertEqual(report['requests'], 2); self.assertEqual(report['summed_request_seconds'], .5)
        path = Path(self.f.tasks[0]['output']) / 'results.jsonl'; row = json.loads(path.read_bytes())
        row['result']['nll']['all_completion']['mean_nll'] = 9
        path.write_text(r.canonical(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'aggregate'): r.validate_outputs(self.f.plan, self.f.tasks)

    def test_false_success_missing_and_duplicate_rows_fail(self):
        self.f.output.mkdir()
        for t in self.f.tasks: self.f.outputs(t)
        path = Path(self.f.tasks[1]['output']) / 'results.jsonl'; data = path.read_bytes()
        path.write_bytes(data + data)
        with self.assertRaisesRegex(ValueError, 'duplicate'): r.validate_outputs(self.f.plan, self.f.tasks)
        path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'missing worker'): r.validate_outputs(self.f.plan, self.f.tasks)
        path.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, 'truncated'): r.validate_outputs(self.f.plan, self.f.tasks)

    def test_environment_has_no_credentials_bus_or_pythonpath(self):
        with patch.dict(os.environ, {'AWS_SECRET_ACCESS_KEY': 'authored', 'PYTHONPATH': '/authored', 'DBUS_SESSION_BUS_ADDRESS': 'authored'}):
            env = r.shared.worker_environment(4)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4')
        self.assertFalse({'AWS_SECRET_ACCESS_KEY', 'PYTHONPATH', 'DBUS_SESSION_BUS_ADDRESS'} & set(env))
        self.assertEqual(env['OMP_NUM_THREADS'], '1')

    def test_progress_counts_partial_line_is_counted_once(self):
        self.f.output.mkdir(); path = self.f.output / 'worker_00'; path.mkdir()
        (path / 'results.jsonl').write_bytes(b'one\ntw')
        cursors = {}; self.assertEqual(r.progress_counts(self.f.plan, cursors)['worker_00'], 1)
        with (path / 'results.jsonl').open('ab') as f: f.write(b'o\n')
        self.assertEqual(r.progress_counts(self.f.plan, cursors)['worker_00'], 2)

    def supervise_authored(self, nonzero=False):
        f = self.f; children = []
        class Child:
            def __init__(self, pid): self.pid = pid; self.returncode = 1 if nonzero else 0
            def poll(self): return self.returncode
        def launch(command, **kwargs):
            task = json.loads(Path(command[-1]).read_bytes())
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], str(task['gpu_id']))
            self.assertTrue(kwargs['start_new_session']); f.outputs(task)
            p = Child(80000 + len(children)); children.append(p); return p
        def info(pid): return {'pid': pid, 'pgid': pid, 'uid': os.getuid(), 'start_ticks': 10}
        with patch.object(r, 'check_runtime'), patch.object(r, 'resource_state', return_value={'at': 0}), \
             patch.object(r.shared, 'gpu_snapshot', side_effect=gpu_rows), patch.object(r.shared, 'process_info', side_effect=info), \
             patch.object(r, 'process_groups', side_effect=lambda cs: {p.pid: [] for p, _, _ in cs}), \
             patch.object(r.subprocess, 'Popen', side_effect=launch), patch.object(r.time, 'sleep'), \
             patch.object(r.time, 'time', return_value=r.shared.DEADLINE - 100):
            return r.supervise(f.plan_ref['path'], f.plan_ref['sha256'])

    def test_complete_producer_then_separate_actual_metadata_verifier(self):
        ref = self.supervise_authored()
        self.assertEqual(ref['path'], str(self.f.output / 'artifact_manifest.json'))
        proof = r.verify(self.f.plan_ref['path'], self.f.plan_ref['sha256'], ref['sha256'])
        self.assertEqual(proof['tf']['requests'], 2)
        self.assertTrue(proof['outer_controller_exit_requires_external_receipt'])
        with self.assertRaisesRegex(ValueError, 'hash/size'):
            r.verify(self.f.plan_ref['path'], self.f.plan_ref['sha256'], '0' * 64)
        with self.assertRaises(FileExistsError): self.supervise_authored()

    def test_nonzero_worker_retains_failure_and_never_complete(self):
        with self.assertRaisesRegex(ValueError, 'worker failed'): self.supervise_authored(nonzero=True)
        self.assertTrue((self.f.output / 'RUN_FAILED.json').is_file())
        self.assertFalse((self.f.output / 'RUN_COMPLETE.json').exists())
        failure = json.loads((self.f.output / 'RUN_FAILED.json').read_bytes())
        self.assertEqual(failure['worker_returncodes'], [1]); self.assertTrue(failure['process_release_verified'])

    def test_cleanup_never_signals_changed_identity(self):
        p = type('P', (), {'pid': 123})()
        with patch.object(r, 'process_groups', side_effect=ValueError('worker PID identity changed')), \
             patch.object(r.os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, 'identity'): r.cleanup([(p, {}, {})])
        kill.assert_not_called()

    def test_resource_state_unions_multiple_groups_on_one_gpu(self):
        self.f.output.mkdir()
        children = [(type('P', (), {'pid': pid, 'poll': lambda self: None})(), {},
                     {'gpu_id': 0, 'name': f'worker_{i:02d}'}) for i, pid in enumerate((10, 20))]
        groups = {10: [{'pid': 10, 'rss_bytes': 1}, {'pid': 11, 'rss_bytes': 1}],
                  20: [{'pid': 20, 'rss_bytes': 1}, {'pid': 21, 'rss_bytes': 1}]}
        rows = gpu_rows(); rows[0]['processes'] = [{'pid': p, 'owner': 'ubuntu'} for p in (10, 11, 20, 21)]
        with patch.object(r, 'process_groups', return_value=groups), \
             patch.object(r.shared, 'gpu_snapshot', return_value=rows), \
             patch.object(Path, 'read_text', return_value='MemAvailable: 999999999 kB\n'), \
             patch.object(r.time, 'time', return_value=r.shared.DEADLINE - 1):
            snapshot = r.resource_state(self.f.plan, children)
        self.assertEqual(snapshot['worker_rss_bytes'], {'worker_00': 2, 'worker_01': 2})

    def test_cleanup_signals_only_owned_group_including_orphan(self):
        child = type('P', (), {'pid': 77, 'returncode': 0, 'poll': lambda self: 0})()
        with patch.object(r, 'process_groups', side_effect=[{77: [{'pid': 78}]}, {77: []}]), \
             patch.object(r.os, 'killpg') as kill:
            r.cleanup([(child, {'pid': 77}, {})])
        kill.assert_called_once_with(77, r.signal.SIGTERM)


class BatchControllerTests(unittest.TestCase):
    def setUp(self):
        from . import batched_generation as b, batch_diagnostics as d
        from .test_batch_diagnostics import Model
        from .test_batched_generation import row,SAMPLING
        self.b,self.d,self.Model=b,d,Model
        self.tmp=tempfile.TemporaryDirectory();self.f=Fixture(self.tmp.name,workers=1);f=self.f
        for short in d.SOURCES:
            name='infra/gpu03/direction_discovery/'+short
            target=f.source/name
            if target.exists():target.chmod(0o600)
            ref=f.save(target,(Path(d.__file__).parent/short).read_bytes())
            f.plan['source_inventory'][name]={k:v for k,v in ref.items() if k!='path'}
        self.rows=[row(3,2),row(4,4)]
        prepared=f.save(f.root/'prepared_batch.jsonl',(''.join(r.canonical(x)+'\n' for x in self.rows)).encode())
        f.plan['bound_files'][prepared['path']]={k:v for k,v in prepared.items() if k!='path'}
        f.plan.update(mode='generate_batch_v1',protocol=r.BATCH_PROTOCOL)
        task=f.tasks[0];task.pop('teacher_forced_padded_sequence_length')
        task.update(mode='generate_batch_v1',prepared_records=prepared['path'],sampling=copy.deepcopy(SAMPLING),
            batch_profile={**b.PROFILE,'batch_size':4},batch_stage='integrated_real_requests',
            batch_diagnostics_policy=d.POLICY,diagnostic_projection_condition_id='target',
            conditions={'baseline':{'layers':[]},'target':{'layers':[{'layer':0,'kind':'random','rank':1,'seed':6101}]}},
            requests=[{'request_id':c+str(i),'record_id':row['record_id'],'condition_id':c,'scope':'primary','seed':11+i}
                      for c in ('baseline','target') for i,row in enumerate(self.rows)])
        f.plan.update(expected_request_count=4,request_ids_sha256=r.digest(sorted(x['request_id'] for x in task['requests'])))
        self.update();f.outputs=self.outputs
    def tearDown(self):self.f.close();self.tmp.cleanup()
    def update(self):
        path=self.f.root/'batch_task.json'
        if path.exists():path.chmod(0o600)
        self.f.plan['workers'][0]['task']=self.f.save(path,self.f.tasks[0]);self.f.update()
    def outputs(self,task,*,leak=False):
        import contextlib
        import torch
        from .test_batched_generation import Tokenizer
        b=self.b;m=self.Model(leak=leak)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}))
            for obj,name in ((b.engine.raw,'configure_torch'),(torch.cuda,'set_per_process_memory_fraction'),
                             (torch.backends.cuda,'enable_cudnn_sdp'),(b.engine.legacy,'_release_cuda')):
                stack.enter_context(patch.object(obj,name))
            for name,value in (('math_sdp_enabled',True),('flash_sdp_enabled',False),('mem_efficient_sdp_enabled',False),('cudnn_sdp_enabled',False)):
                stack.enter_context(patch.object(torch.backends.cuda,name,return_value=value))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()))
            stack.enter_context(patch.object(b.engine.legacy,'_load_decoder',return_value=(m,m.decoder,[m.block],
                {'with_adapter':True,'active_adapters':['default'],'nonzero_lora_parameter_tensors':1})))
            b.worker(self.f.plan['workers'][0]['task']['path'])
    def test_integrated_producer_exact_union_then_independent_verifier(self):
        self.f.load()
        ref=Tests.supervise_authored(self)
        proof=r.verify(self.f.plan_ref['path'],self.f.plan_ref['sha256'],ref['sha256'])
        self.assertEqual(proof['generation']['requests'],4)
        self.assertEqual(proof['status'],'verified_batched_generation_bytes_and_coverage')
        self.assertFalse(proof['generation']['distribution_equivalence_established'])
        commands=[json.loads(x)['command'] for x in (self.f.output/'workers.jsonl').read_text().splitlines()]
        self.assertIn('infra.gpu03.direction_discovery.batched_generation',commands[0])
    def test_one_worker_per_gpu_mandatory(self):
        self.f.plan['workers']*=2;self.f.update()
        with self.assertRaisesRegex(ValueError,'exactly one worker'):self.f.load()
    def test_generation_sampler_and_diagnostic_policy_are_closed(self):
        self.f.tasks[0]['sampling']['temperature']=.8;self.update()
        with self.assertRaisesRegex(ValueError,'sampler'):self.f.load()
        self.f.tasks[0]['sampling']['temperature']=.7
        self.f.tasks[0]['batch_diagnostics_policy']={**self.d.POLICY,'same_shape_max_abs':1.};self.update()
        with self.assertRaisesRegex(RuntimeError,'diagnostic policy'):self.f.load()
    def test_actual_diagnostic_failure_retains_first_batch_and_stops_rest(self):
        self.f.output.mkdir()
        with self.assertRaisesRegex(RuntimeError,'Same-shape'):self.outputs(self.f.tasks[0],leak=True)
        out=Path(self.f.tasks[0]['output']);records=self.b.engine.read_jsonl(out/'results.jsonl')
        self.assertEqual(len(records),2);self.assertTrue(all(x['condition_id']=='baseline' for x in records))
        self.assertFalse((out/'SUCCESS.json').exists())
        self.assertEqual(json.loads((out/'real_prefix_diagnostics.json').read_bytes())['status'],'failed_real_prefix_checks')
    def test_mutated_eos_or_diagnostic_provenance_rejected(self):
        self.f.output.mkdir();self.outputs(self.f.tasks[0]);r.validate_outputs(self.f.plan,self.f.tasks)
        out=Path(self.f.tasks[0]['output']);p=out/'real_prefix_diagnostics.json';data=json.loads(p.read_bytes())
        data['request_sha256']='0'*64;p.write_text(r.canonical(data))
        with self.assertRaisesRegex(RuntimeError,'provenance'):r.validate_outputs(self.f.plan,self.f.tasks)
        data['request_sha256']=self.d.digest(self.b.request_batches(self.f.tasks[0]['requests'],4)[0]);p.write_text(r.canonical(data))
        records=self.b.engine.read_jsonl(out/'results.jsonl');records[0]['result']['generated_token_ids'][0]=151643
        (out/'results.jsonl').write_text(''.join(r.canonical(x)+'\n' for x in records))
        with self.assertRaisesRegex(ValueError,'EOS'):r.validate_outputs(self.f.plan,self.f.tasks)

if __name__ == '__main__': unittest.main()
