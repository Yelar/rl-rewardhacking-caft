"""Small checks for the cache-only runner's indexing and retained-cell recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import all_layer_run as run


class CacheRunnerTests(unittest.TestCase):
    def test_worker_module_is_not_main_alias(self):
        self.assertEqual(run.MODULE, "infra.gpu03.direction_discovery.all_layer_run")

    def test_cell_preserves_noncontiguous_completion_indices_and_exclusions(self):
        rows = [{"record_id": str(i)} for i in range(3)]
        selections = [{"solution": {"complete_code": {"eligible": i != 1,
                       "completion_positions": [0, 2] if i != 1 else []}}} for i in range(3)]
        raw = {"h0": [np.arange(8, dtype=np.float32).reshape(4, 2) + i * 100 for i in range(3)]}
        raw["h60"] = [x + 4 for x in raw["h0"]]
        cell = run.make_cell(rows, selections, raw, "solution", "complete_code")
        self.assertEqual([r["record_id"] for r in cell.rows], ["0", "2"])
        np.testing.assert_array_equal(cell.offsets, [0, 2, 4])
        np.testing.assert_array_equal(cell.h0, [[0, 1], [4, 5], [200, 201], [204, 205]])
        np.testing.assert_array_equal(cell.token_position, [0, 2, 0, 2])
        np.testing.assert_array_equal(cell.token_record, [0, 0, 1, 1])
        np.testing.assert_array_equal(cell.delta, np.full((4, 2), 4, dtype=np.float32))

    def test_no_eligible_records_are_explicitly_unsupported(self):
        data = run.make_cell([{}], [{"evaluator": {"body": {"eligible": False}}}], {}, "evaluator", "body")
        self.assertIsNone(data)

    def test_complete_cell_requires_matching_identity_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cell"
            identity = {"plan_sha256": "abc", "layer": 1, "region": "solution", "representation": "body"}
            run.save_cell(path, {"status": "unsupported"}, {}, identity)
            self.assertTrue(run.existing_cell(path, identity))
            with self.assertRaises(ValueError):
                run.existing_cell(path, {**identity, "layer": 2})
            (path / "report.json").chmod(0o600)
            (path / "report.json").write_text('{}\n')
            with self.assertRaises(ValueError):
                run.existing_cell(path, identity)

    def test_incomplete_cell_is_retained_and_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cell"
            path.mkdir()
            with self.assertRaises(ValueError):
                run.existing_cell(path, {})
            self.assertTrue(path.is_dir())

    def test_resumed_supervisor_preserves_prior_worker_log(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            log = out / 'worker_00.log'
            log.write_bytes(b'previous attempt\n')
            plan = {'output': str(out), 'workers': 1, 'layers': list(range(36)),
                    'representations': list(run.reps.PRIMARY), 'gpu_devices': []}

            class FinishedChild:
                pid = 123
                returncode = 0

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            def launch(*args, **kwargs):
                kwargs['stdout'].write(b'resumed worker\n')
                return FinishedChild()

            with patch.object(run, 'read_plan', return_value=plan), \
                 patch.object(run, 'resource_check'), \
                 patch.object(run.importlib.metadata, 'version', return_value='test'), \
                 patch.object(run.subprocess, 'Popen', side_effect=launch), \
                 patch.object(Path, 'glob', return_value=[None] * 864):
                run.supervise(Path('plan.json'), 'abc')
            self.assertEqual(log.read_bytes(), b'previous attempt\nresumed worker\n')
            self.assertEqual(json.loads((out / 'RUN_COMPLETE.json').read_text())['status'], 'complete')


class BroaderControllerTests(unittest.TestCase):
    def test_broader_read_plan_runs_raw_gate_before_accepting_allocation(self):
        from . import broader_pca
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            path.write_text(json.dumps({'purpose': 'broader_ordinary_all36_pca'}))
            plan = {'workers': 8, 'gpu_devices': list(range(8)), 'gpu_uuids': [str(i) for i in range(8)]}
            with patch.object(broader_pca, 'load_context', return_value=(plan, None, None, None)) as gate:
                self.assertEqual(run.read_plan(path, run.c.sha256_file(path)), plan)
                gate.assert_called_once_with(path, run.c.sha256_file(path))
                plan['gpu_devices'] = [0] * 8
                with self.assertRaisesRegex(ValueError, 'eight explicit'):run.read_plan(path, run.c.sha256_file(path))
            with patch.object(broader_pca, 'load_context', side_effect=ValueError('raw not qualified')):
                with self.assertRaisesRegex(ValueError, 'raw not qualified'):run.read_plan(path, run.c.sha256_file(path))

    def test_broader_supervisor_spawns_fixed_module_and_strided_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plan = {'purpose': 'broader_ordinary_all36_pca', 'output': str(out), 'workers': 8,
                    'layers': list(range(36)), 'representations': list(run.reps.PRIMARY), 'gpu_devices': list(range(8))}
            commands = []
            class Finished:
                pid = 123
                returncode = 0
                def poll(self):return 0
                def wait(self, timeout=None):return 0
            def launch(command, **kwargs):
                commands.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                return Finished()
            coverage = {'cohort': 'broader_ordinary', 'cells': 864, 'worker_successes': 8}
            with patch.object(run, 'read_plan', return_value=plan), patch.object(run, 'resource_check'), \
                 patch.object(run.importlib.metadata, 'version', return_value='authored'), \
                 patch.object(run.subprocess, 'Popen', side_effect=launch), \
                 patch.object(run, 'broader_completion', return_value=coverage) as verify:
                run.supervise(Path('plan.json'), 'bound')
            verify.assert_called_once_with(out, plan, 'bound')
            self.assertEqual(len(commands), 8)
            for i, (command, gpu) in enumerate(commands):
                self.assertEqual(command[3], 'infra.gpu03.direction_discovery.broader_pca')
                self.assertEqual(command[command.index('--layers') + 1:], list(map(str, range(i, 36, 8))))
                self.assertEqual(gpu, str(i))
            result = json.loads((out / 'RUN_COMPLETE.json').read_text())
            self.assertEqual(result['completion_coverage'], coverage)
            self.assertEqual(result['worker_returncodes'], [0] * 8)

    def test_broader_completion_checks_identity_bytes_extra_cells_and_worker_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plan = {'layers': [0], 'representations': ['body']}
            for region in run.reps.REGIONS:
                cell = out / 'layer_00' / region / 'body'; cell.mkdir(parents=True)
                report = cell / 'report.json'; report.write_text('{"pca":{"status":"unsupported"}}\n')
                run.c.write_json(cell / 'complete.json', {'status': 'complete', 'cohort': 'broader_ordinary',
                    'plan_sha256': 'bound', 'layer': 0, 'region': region, 'representation': 'body',
                    'files': {'report.json': {'sha256': run.c.sha256_file(report), 'size_bytes': report.stat().st_size}}})
            for worker in range(8):
                run.c.write_json(out / f'worker_{worker:02d}_SUCCESS.json', {'status': 'complete', 'plan_sha256': 'bound',
                    'cohort': 'broader_ordinary', 'layers': plan['layers'][worker::8], 'worker': worker})
            self.assertEqual(run.broader_completion(out, plan, 'bound')['cells'], 2)
            receipt = out / 'worker_00_SUCCESS.json'; original = receipt.read_text()
            receipt.write_text(original.replace('[0]', '[1]'))
            with self.assertRaisesRegex(ValueError, 'shard receipt'):run.broader_completion(out, plan, 'bound')
            receipt.write_text(original)
            extra = out / 'layer_01' / 'solution' / 'body'; extra.mkdir(parents=True)
            (extra / 'complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'inventory'):run.broader_completion(out, plan, 'bound')
            (extra / 'complete.json').unlink()
            report = out / 'layer_00' / 'solution' / 'body' / 'report.json'; report.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'size mismatch|hash mismatch'):run.broader_completion(out, plan, 'bound')


if __name__ == "__main__":
    unittest.main()
