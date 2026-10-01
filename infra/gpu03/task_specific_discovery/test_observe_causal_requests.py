"""Inert metadata fixtures; no SSH, models, scorers, or generated code."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import observe_causal_requests as m


def write(path, value):
    raw = m.canonical(value) + b'\n'
    path.write_bytes(raw)
    return {'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class ObserveCausalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'run-token'
        self.root.mkdir()
        self.output = self.root / 'worker'; self.output.mkdir()
        self.journal = self.output / 'journal'; self.journal.mkdir()
        self.requests = [{'request_id': f'request-{i}', 'condition_id': 'baseline', 'seed': i} for i in range(2)]
        self.expected = {r['request_id']: hashlib.sha256(m.canonical(r)).hexdigest() for r in self.requests}
        self.task = {'schema': 'task_specific_causal_generation_v1', 'run_token': self.root.name,
                     'stage': 'qualification', 'requests': self.requests, 'output': str(self.output)}
        self.task_path = self.root / 'task.json'; self.binding = write(self.task_path, self.task)
        self.runtime = self.root.parent / (self.root.name + '-qualification-runtime'); self.runtime.mkdir()
        self.manifest = {'schema': 'gpu03_task_specific_resource_guard_v1', 'host': 'gpu-03',
                         'run_token': self.root.name, 'science': {'stage': 'qualification'},
                         'runtime': str(self.runtime), 'gpu_uuids': {'3': 'GPU-example'},
                         'workers': [{'name': 'gpu_3', 'gpu_id': 3, 'argv': ['/pinned/python', '-B', '-m',
                             'infra.gpu03.task_specific_discovery.causal_worker', '--task', str(self.task_path),
                             '--sha256', self.binding['sha256']]}],
                         'bound_files': {str(self.task_path): self.binding}, 'expected_reused_requests': 0}
        self.manifest_path = self.root / 'manifest.json'; self.manifest_binding = write(self.manifest_path, self.manifest)
        for stage in m.STAGES:
            (self.journal / (stage + '.jsonl')).write_bytes(b'')

    def event(self, rid, stage):
        value = {'schema': 1, 'run_token': self.root.name, 'input_sha256': self.binding['sha256'],
                 'request_id': rid, 'request_sha256': self.expected.get(rid, 'd' * 64), 'stage': stage}
        if stage != 'started':
            value['file_sha256'] = 'a' * 64
        return value

    def append(self, rid, stage):
        with (self.journal / (stage + '.jsonl')).open('ab') as stream:
            stream.write(m.canonical(self.event(rid, stage)) + b'\n')

    def scan(self, stage='started'):
        return m.scan_events(self.journal / (stage + '.jsonl'), self.expected,
            run_token=self.root.name, task_sha256=self.binding['sha256'], stage=stage)

    def observe(self):
        return m.observe(self.manifest_path, self.manifest_binding['sha256'], root=self.root)

    def test_full_read_only_observation_uses_bound_paths_never_payloads(self):
        self.append('request-0', 'started'); self.append('request-1', 'started')
        self.append('request-0', 'primitive_returned')
        # Opening these would block; the monitor must not inspect response bytes.
        for name in ('generations.jsonl', 'primitive_payload'):
            os.mkfifo(self.output / name)
        write(self.output / 'STARTED.json', {'requests': 2, 'pid': 10, 'task_sha256': self.binding['sha256']})
        with (self.runtime / 'resources.jsonl').open('wb') as stream:
            stream.write(m.canonical({'at': 123., 'owned_rss_bytes': 200, 'gpus': {'3': {
                'uuid': 'GPU-example', 'total_mib': 48000, 'free_mib': 25000, 'utilization': 70, 'pids': [10]},
                '7': {'uuid': 'foreign', 'pids': [99]}}}) + b'\n')
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        report = self.observe()
        self.assertEqual(report['known_event_counts'], {'started': 2, 'primitive_returned': 1, 'final_recorded': 0})
        self.assertEqual(report['workers'][0]['non_atomic_pending_estimates']['start_without_return_event'], 1)
        self.assertEqual(set(report['last_recorded_resources']['selected_gpus']), {'3'})
        self.assertFalse(report['terminal_success_verified'])
        self.assertFalse(report['payloads_verified'])
        self.assertFalse(report['resource_release_verified'])
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))

    def test_complete_marker_and_full_counts_never_become_verified_success(self):
        for stage in m.STAGES:
            for rid in self.expected:
                self.append(rid, stage)
        write(self.output / 'COMPLETE.json', {'status': 'succeeded', 'requests': 2,
                                             'all_returned_requests_retained': True})
        write(self.runtime / 'SUPERVISOR_RESULT.json', {'status': 'workers_exited_zero',
            'gpu_release_verified': True, 'owned_processes_released': True})
        report = self.observe()
        self.assertEqual(report['workers'][0]['COMPLETE']['reported']['status'], 'succeeded')
        self.assertEqual(report['known_event_counts']['final_recorded'], 2)
        self.assertFalse(report['terminal_success_verified'])
        self.assertFalse(report['resource_release_verified'])

    def test_partial_tail_preserves_complete_prefix_as_observation(self):
        self.append('request-0', 'started')
        with (self.journal / 'started.jsonl').open('ab') as stream:
            stream.write(b'{"request_id":')
        info, ids = self.scan()
        self.assertEqual(ids, {'request-0'})
        self.assertEqual(info['partial_tail'], 1)
        self.assertEqual(info['valid_unique_events'], 1)

    def test_duplicate_unplanned_malformed_and_wrong_bound_events_are_reported(self):
        self.append('request-0', 'started'); self.append('request-0', 'started')
        wrong = self.event('request-1', 'started'); wrong['input_sha256'] = '0' * 64
        with (self.journal / 'started.jsonl').open('ab') as stream:
            stream.write(m.canonical(self.event('unplanned', 'started')) + b'\n')
            stream.write(m.canonical(wrong) + b'\n')
            stream.write(b'not json\n{"request_id":"a","request_id":"b"}\n')
        info, ids = self.scan()
        self.assertEqual(ids, {'request-0'})
        self.assertEqual((info['duplicates'], info['unplanned'], info['binding_mismatch'], info['malformed']), (1, 1, 1, 2))

    def test_missing_events_and_incomplete_marker_are_not_zero_success(self):
        (self.journal / 'started.jsonl').unlink()
        (self.output / 'COMPLETE.json').write_bytes(b'{"status":')
        report = self.observe()
        row = report['workers'][0]
        self.assertEqual(row['events']['started']['status'], 'missing')
        self.assertIsNone(row['events']['started']['valid_unique_events'])
        self.assertNotIn('non_atomic_pending_estimates', row)
        self.assertEqual(row['COMPLETE']['parse_status'], 'incomplete_or_malformed')
        self.assertTrue(report['totals_are_partial_observations'])

    def test_bound_inputs_reject_hash_changes_wrong_scope_and_output_escape(self):
        with self.assertRaises(ValueError):
            m.observe(self.manifest_path, '0' * 64, root=self.root)
        self.task['output'] = str(self.root.parent / 'unrelated')
        self.binding = write(self.task_path, self.task)
        self.manifest['bound_files'][str(self.task_path)] = self.binding
        self.manifest['workers'][0]['argv'][-1] = self.binding['sha256']
        self.manifest_binding = write(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'Output outside'):
            self.observe()

    def test_file_byte_caps_symlinks_and_nonregular_files(self):
        path = self.root / 'large'; path.write_bytes(b'x' * 100)
        raw, info = m.read_snapshot(path, 20)
        self.assertIsNone(raw); self.assertEqual(info['status'], 'oversized')
        link = self.root / 'alias'; link.symlink_to(path)
        self.assertEqual(m.read_snapshot(link, 20)[1]['status'], 'unsafe_path')
        fifo = self.root / 'fifo'; os.mkfifo(fifo)
        self.assertEqual(m.read_snapshot(fifo, 20)[1]['status'], 'not_regular')

    def test_changing_snapshot_is_explicit(self):
        path = self.root / 'changing'; path.write_bytes(b'{}\n')
        before = path.stat()
        after = types.SimpleNamespace(**{key: getattr(before, key) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')})
        after.st_mtime_ns += 1
        with patch.object(m.os, 'fstat', side_effect=[before, after]):
            raw, info = m.read_snapshot(path, 100)
        self.assertEqual(raw, b'{}\n'); self.assertEqual(info['status'], 'changing')

    def test_resource_tail_skips_partial_latest_and_malformed_rows(self):
        path = self.runtime / 'resources.jsonl'
        path.write_bytes(b'x' * 70000 + b'\n' + m.canonical({'at': 10, 'gpus': {'3': {'pids': []}}})
                         + b'\nnot-json\n{"at":11')
        report = m.resource_tail(path, {'3': 'GPU-example'})
        self.assertLessEqual(report['read_bytes'], 65536)
        self.assertEqual(report['snapshot']['at'], 10)
        self.assertFalse(report['fresh_gpu_probe_performed'])

    def test_raw_error_and_generated_text_are_not_printed_in_markers(self):
        path = self.output / 'FAILED.json'
        write(path, {'status': 'failed', 'error': 'sensitive text', 'completion': 'generated code',
                     'environment': {'SECRET': 'secret'}})
        report = m.marker(path, ('status', 'stage', 'task_sha256'))
        self.assertEqual(report['reported'], {'status': 'failed'})
        self.assertNotIn('sensitive', m.canonical(report).decode())

    def test_compact_summary_preserves_full_receipt_and_conservative_counts(self):
        self.append('request-0', 'started')
        (self.journal / 'final_recorded.jsonl').unlink()
        report = self.observe()
        report['last_recorded_resources'] = {'status': 'observed',
            'snapshot': {'at': 123., 'mem_available_bytes': 1000, 'owned_rss_bytes': 50},
            'selected_gpus': {'3': {'total_mib': 48000, 'free_mib': 28000, 'utilization': 80, 'pids': [10]}}}
        original = copy.deepcopy(report)
        compact = m.compact_report(report, '/receipt.json')
        self.assertEqual(report, original)
        self.assertEqual(compact['receipt'], '/receipt.json')
        self.assertEqual(compact['workers'][0]['counts']['started'], 1)
        self.assertIsNone(compact['workers'][0]['counts']['final_recorded'])
        self.assertEqual(compact['resources']['gpus']['3'], {'used_mib': 20000, 'utilization': 80, 'pids': [10]})
        self.assertEqual(compact['resources']['available_ram_bytes'], 1000)
        self.assertNotIn('task', compact['workers'][0])
        self.assertNotIn('sha256_of_read_bytes', m.canonical(compact).decode())
        self.assertFalse(compact['terminal_success_verified'])


if __name__ == '__main__':
    unittest.main()
