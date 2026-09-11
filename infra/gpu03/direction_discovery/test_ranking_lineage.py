import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import ranking_lineage as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True) + '\n').encode()
    path.write_bytes(data)
    return {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stage, out = self.root / 'qual-stage', self.root / 'qual-results'
        current = self.root / 'tf-stage'
        self.master = {'inputs': {'prepared_records_sha256': 'd' * 64},
                       'model': {'revision': 'a' * 40, 'M60_adapter_sha256': 'b' * 64}}
        task = {'mode': 'qualify', 'run_token': 'qual', 'worker_name': 'gpu_0',
                'model_snapshot': str(self.root / ('a' * 40)), 'checkpoint': str(self.root / 'checkpoint'),
                'attention_policy': 'exclusive_math', 'teacher_forced_padded_sequence_length': 2176,
                'requests': [{'request_id': 'probe1', 'record_id': 'r222', 'condition_id': 'baseline'}],
                'conditions': {'baseline': {'layers': []}}}
        self.task = task
        task_path = stage / 'task.json'
        tb = write(task_path, task)
        worker = {'name': 'gpu_0', 'command': ['python', 'engine.py', '--task', str(task_path)],
                  'success_expect': {'mode': 'qualify', 'requests': 1}, 'success_file': 'workers/gpu_0/SUCCESS.json'}
        bindings = {str(task_path): tb}
        for prefix, names in [(task['model_snapshot'], ['config.json', 'model.safetensors', 'tokenizer.json']),
                              (task['checkpoint'], ['adapter_config.json', 'adapter_model.safetensors'])]:
            for name in names:
                bindings[str(Path(prefix) / name)] = {'sha256': 'b' * 64 if name == 'adapter_model.safetensors' else 'c' * 64,
                                                     'size_bytes': 100}
        for source in r.CRITICAL:
            bindings[str(stage / 'source' / source)] = {'sha256': 'e' * 64, 'size_bytes': 200}
        q = {'host': 'gpu-04', 'phase': 'causal_qualification', 'run_token': 'qual', 'stage': str(stage),
             'output': str(out), 'source_root': str(stage / 'source'), 'gpu_ids': [0], 'runtime_versions': {'torch': '2.8'},
             'scientific': {'master_plan_sha256': 'f' * 64, 'input_prepared_sha256': 'd' * 64},
             'workers': [worker], 'bound_files': bindings}
        self.qpath = stage / 'reviewed_manifest.json'
        qb = write(self.qpath, q)
        terminal = stage / 'control/supervisor_exit.json'
        eb = write(terminal, {'manifest_sha256': qb['sha256'], 'run_token': 'qual', 'service_result': 'success',
                             'exit_code_kind': 'exited', 'exit_status': '0', 'producer_summary_present': True,
                             'failure_present': False, 'invocation_id': 'a' * 32})
        afiles = {}
        for name, value in {
            'campaign_summary.json': {'status': 'succeeded', 'run_token': 'qual', 'manifest_sha256': qb['sha256'],
                                      'worker_exit_codes': [0], 'gpu_release_verified': True},
            'gpu_release.json': {'verified': True, 'gpu_ids': [0]},
            worker['success_file']: {'status': 'succeeded', 'run_token': 'qual', 'worker_name': 'gpu_0',
                                     'mode': 'qualify', 'requests': 1},
        }.items():
            afiles[name] = write(out / name, value)
        ap = out / 'artifact_manifest.json'
        self.result_path = out / 'workers/gpu_0/results.jsonl'
        result_binding = write(self.result_path, {**task['requests'][0], 'problem_split': 'direction_fit',
            'result': {'baseline_recovery_bitwise': True, 'teacher_forced_effect_verified': True,
                       'baseline_generation_repeatable': True}})
        afiles['workers/gpu_0/results.jsonl'] = result_binding
        ab = write(ap, {'algorithm': 'sha256', 'files': afiles})
        ref = {'manifest_path': str(self.qpath), 'manifest_sha256': qb['sha256'], 'artifact_manifest_sha256': ab['sha256']}
        rp = current / 'input/causal_qualification_reference.json'
        rb = write(rp, ref)
        self.current_task = current / 'task.json'
        cb = write(self.current_task, {**task, 'mode': 'tf'})
        mbindings = {str(self.qpath): qb, str(task_path): tb, str(terminal): eb, str(ap): ab, str(rp): rb,
                    str(self.current_task): cb, str(self.result_path): result_binding}
        for path, binding in bindings.items():
            if str(stage / 'source') in path:
                mbindings[path.replace(str(stage / 'source'), str(current / 'source'))] = binding
            elif path != str(task_path):
                mbindings[path] = binding
        self.m = {**q, 'phase': 'coarse', 'run_token': 'tf', 'stage': str(current), 'source_root': str(current / 'source'),
                  'bound_files': mbindings, 'workers': [{**worker, 'command': ['python', 'engine.py', '--task', str(self.current_task)]}]}

    def test_success_reads_only_metadata_not_model_payload(self):
        report = r.validate(self.m, self.master)
        self.assertFalse(report['model_payloads_rehashed'])
        self.assertEqual(len(report['critical_source_bindings']), 4)
        self.assertEqual(len(report['model_file_bindings']), 5)
        self.assertFalse(Path(self.task['model_snapshot']).exists())

    def test_adapter_revision_runtime_and_source_mismatch(self):
        for key in ['M60_adapter_sha256', 'revision']:
            master = copy.deepcopy(self.master); master['model'][key] = 'wrong'
            with self.assertRaises(ValueError): r.validate(self.m, master)
        changed = copy.deepcopy(self.m); changed['runtime_versions'] = {'torch': 'other'}
        with self.assertRaisesRegex(ValueError, 'runtime'): r.validate(changed, self.master)
        changed = copy.deepcopy(self.m)
        changed['bound_files'][str(Path(changed['source_root']) / r.CRITICAL[0])]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'critical inference'): r.validate(changed, self.master)

    def test_wrong_worker_checkpoint_and_missing_payload_binding(self):
        changed = copy.deepcopy(self.m)
        changed['bound_files'][str(self.current_task)] = write(self.current_task, {**self.task, 'mode': 'tf', 'checkpoint': 'other'})
        with self.assertRaisesRegex(ValueError, 'TF worker model'): r.validate(changed, self.master)
        changed['bound_files'][str(self.current_task)] = write(self.current_task, {**self.task, 'mode': 'tf'})
        del changed['bound_files'][str(Path(self.task['model_snapshot']) / 'model.safetensors')]
        with self.assertRaisesRegex(ValueError, 'file bindings differ'): r.validate(changed, self.master)

    def test_changed_qualification_and_terminal_receipt_fail(self):
        terminal = self.qpath.parent / 'control/supervisor_exit.json'
        changed = copy.deepcopy(self.m)
        invalid = json.loads(terminal.read_text()); invalid['invocation_id'] = ''
        changed['bound_files'][str(terminal)] = write(terminal, invalid)
        with self.assertRaisesRegex(ValueError, 'terminal success'): r.validate(changed, self.master)
        self.qpath.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'metadata hash'): r.validate(self.m, self.master)

    def test_missing_or_unrelated_qualification_rejected(self):
        changed = copy.deepcopy(self.m); changed['scientific']['master_plan_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'master'): r.validate(changed, self.master)
        master = copy.deepcopy(self.master); master['parent_plan_sha256'] = 'f' * 64
        self.assertFalse(r.validate(changed, master)['model_payloads_rehashed'])

    def test_failed_numerical_verdict_or_wrong_request_rejected(self):
        for edit in ('flag', 'request'):
            probe = json.loads(self.result_path.read_text())
            probe['result']['teacher_forced_effect_verified'] = edit != 'flag'
            if edit == 'request': probe['request_id'] = 'other'
            changed = copy.deepcopy(self.m)
            changed['bound_files'][str(self.result_path)] = write(self.result_path, probe)
            artifact_path = self.result_path.parents[2] / 'artifact_manifest.json'
            artifact = json.loads(artifact_path.read_text())
            artifact['files']['workers/gpu_0/results.jsonl'] = changed['bound_files'][str(self.result_path)]
            changed['bound_files'][str(artifact_path)] = write(artifact_path, artifact)
            reference_path = Path(changed['stage']) / 'input/causal_qualification_reference.json'
            reference = json.loads(reference_path.read_text())
            reference['artifact_manifest_sha256'] = changed['bound_files'][str(artifact_path)]['sha256']
            changed['bound_files'][str(reference_path)] = write(reference_path, reference)
            with self.assertRaisesRegex(ValueError, 'numerical verdict'): r.validate(changed, self.master)


if __name__ == '__main__':
    unittest.main()
