"""All dispatch tests are inert, with systemd and SSH replaced by fixtures."""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import resource_guard as g
from . import successor_dispatch as d
from . import successor_phase_entry as entry
from . import test_resource_guard as original


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / d.TOKEN
        self.root.mkdir(); (self.root / "plans").mkdir(); (self.root / "source-successor-v1").mkdir()
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(d, "ROOT", self.root))
        self.stack.enter_context(patch.object(d, "UID", os.getuid()))
        self.stack.enter_context(patch.object(g, "UID", os.getuid()))
        self.stack.enter_context(patch.object(g, "RUN_ROOT", self.base))
        self.stack.enter_context(patch.object(d.socket, "gethostname", return_value="gpu-03"))
        self.m = self.manifest()
        self.local = self.base / "QUALIFY_SUCCESSOR_V1.json"

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def manifest(self):
        stage = self.root / "source-successor-v1"
        unit = d.TOKEN + "-qualify-successor-v1"
        modules = ["successor_dispatch.py", "successor_phase_entry.py", "phase_entry.py",
                   "resource_guard.py", "guard_failure_diagnostic.py"]
        return {"schema": g.SCHEMA, "host": d.HOST, "uid": os.getuid(), "run_token": d.TOKEN,
                "unit": unit, "runtime": str(self.base / (unit + "-runtime")),
                "disk_path": str(self.root), "source_root": str(stage),
                "entry_receipt": str(self.root / "qualify-successor-v1-entry.json"),
                "kernel_limits": {"memory_max": 4 * 1024**3, "tasks_max": 64, "cpu_cores": 2},
                "deadline_epoch": 2000, "runtime_seconds": 1120, "gpu_uuids": {},
                "bound_files": {str(stage / "infra/gpu03/task_specific_discovery" / p):
                                {"size_bytes": 1, "sha256": "a" * 64} for p in modules},
                "workers": [{"name": "qualify", "gpu_id": None, "cpu_ids": [48, 49],
                             "argv": [d.PYTHON, "-B", "-m", "unittest"], "cwd": str(stage),
                             "env": {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "",
                                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                     "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1"}}]}

    def payload(self):
        data = (json.dumps(self.m) + "\n").encode(); self.local.write_bytes(data)
        self.sha = hashlib.sha256(data).hexdigest()
        return d.prepare(self.local, str(self.root), self.sha, now=1000)

    def response(self, args, **kwargs):
        if args[:3] == ["systemctl", "--user", "show"]:
            return subprocess.CompletedProcess(args, 0, "LoadState=not-found\nMainPID=0\n", "")
        return subprocess.CompletedProcess(args, 0, "Running as unit\n", "")

    def test_build_binds_exact_source_entry_limits_and_no_shell(self):
        payload = self.payload(); command = payload["command"]
        self.assertIn("infra.gpu03.task_specific_discovery.successor_phase_entry", command)
        self.assertIn("--property=MemorySwapMax=0", command)
        self.assertIn("--property=KillMode=control-group", command)
        self.assertEqual(command[-1], self.sha)
        self.assertEqual(payload["remote"], str(self.root / "plans" / self.local.name))

    def test_wrong_root_token_runtime_source_entry_and_gpu_rejected(self):
        self.payload()
        with self.assertRaises(ValueError): d.prepare(self.local, str(self.base / "other"), self.sha, now=1000)
        for field, value in [("run_token", "codex-wrong-token"), ("runtime", str(self.base / "other-runtime")),
                             ("source_root", str(self.base / "source-other")),
                             ("entry_receipt", str(self.base / "outside-entry.json")),
                             ("disk_path", str(self.base))]:
            m = copy.deepcopy(self.m); m[field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, g.GuardError)):
                d.validate_successor_manifest(m, str(self.root), now=1000)

    def test_reviewed_hash_duplicate_keys_and_repeated_basename_rejected(self):
        self.payload()
        with self.assertRaisesRegex(ValueError, "SHA"): d.prepare(self.local, str(self.root), "0"*64, now=1000)
        with self.assertRaisesRegex(ValueError, "Duplicate"): d.parse('{"x":1,"x":2}')
        copied = self.base / "RESOURCE_MANIFEST.json"; copied.write_bytes(self.local.read_bytes())
        with self.assertRaisesRegex(ValueError, "unique"): d.prepare(copied, str(self.root), self.sha, now=1000)

    def test_limits_and_deadline_fail_closed(self):
        for changes in ({"deadline_epoch": 900}, {"deadline_epoch": 16000}, {"runtime_seconds": 1000},
                        {"runtime_seconds": 15000}, {"kernel_limits": {"memory_max": 40*1024**3, "cpu_cores": 2, "tasks_max": 64}}):
            m = dict(self.m, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                d.validate_successor_manifest(m, str(self.root), now=1000)

    def test_gpu_wave_requires_all_eight_exact_uuids_and_limits(self):
        m = copy.deepcopy(self.m); m["gpu_uuids"] = dict(d.GPU_UUIDS)
        m["kernel_limits"] = {"memory_max": 192*1024**3, "cpu_cores":16, "tasks_max":512}
        template = m["workers"][0]; m["workers"] = []
        for index, uuid in d.GPU_UUIDS.items():
            w = copy.deepcopy(template); w.update(name="gpu_"+index, gpu_id=int(index), cpu_ids=[2*int(index),2*int(index)+1])
            w["env"]["CUDA_VISIBLE_DEVICES"] = uuid; m["workers"].append(w)
        d.validate_successor_manifest(m, str(self.root), now=1000)
        m["gpu_uuids"]["0"] = "GPU-00000000-0000-0000-0000-000000000000"
        m["workers"][0]["env"]["CUDA_VISIBLE_DEVICES"] = m["gpu_uuids"]["0"]
        with self.assertRaisesRegex(ValueError, "allocation"): d.validate_successor_manifest(m, str(self.root), now=1000)

    def test_remote_success_has_markers_before_single_launch(self):
        payload = self.payload(); launches = []
        def runner(args, **kw):
            if args[0] == "systemd-run":
                launches.append(args)
                self.assertTrue((self.root / (self.m["unit"] + "_DISPATCH_ATTEMPT.json")).is_file())
                self.assertTrue(Path(payload["remote"]).is_file())
            return self.response(args, **kw)
        out = d.remote_execute(payload, runner=runner, now=1000)
        self.assertEqual(out["returncode"], 0); self.assertEqual(len(launches), 1)
        with self.assertRaisesRegex(ValueError, "Existing"): d.remote_execute(payload, runner=runner, now=1000)
        self.assertEqual(len(launches), 1)

    def test_remote_timeout_is_ambiguous_and_consumed(self):
        payload = self.payload(); launches = []
        def runner(args, **kw):
            if args[0] == "systemd-run":
                launches.append(args); raise subprocess.TimeoutExpired(args, 30)
            return self.response(args, **kw)
        out = d.remote_execute(payload, runner=runner, now=1000)
        self.assertEqual(out["status"], "ambiguous_dispatch"); self.assertFalse(out["retry_allowed"])
        self.assertEqual(len(launches), 1)
        with self.assertRaises(ValueError): d.remote_execute(payload, runner=runner, now=1000)

    def test_remote_nonzero_is_retained_not_retried(self):
        payload = self.payload()
        def runner(args, **kw):
            return subprocess.CompletedProcess(args, 1, "", "failed") if args[0] == "systemd-run" else self.response(args, **kw)
        out = d.remote_execute(payload, runner=runner, now=1000)
        self.assertEqual(out["returncode"], 1)
        self.assertTrue((self.root/(self.m["unit"]+"_DISPATCH_RESULT.json")).is_file())

    def test_remote_unit_ambiguity_and_command_mutation_prevent_launch(self):
        payload = self.payload(); called = []
        def runner(args, **kw):
            called.append(args); return subprocess.CompletedProcess(args, 1, "", "")
        with self.assertRaisesRegex(ValueError, "positively absent"): d.remote_execute(payload, runner=runner, now=1000)
        self.assertEqual(len(called), 1); self.assertEqual(called[0][0], "systemctl")
        payload["command"].append("bad")
        with self.assertRaisesRegex(ValueError, "changed"): d.remote_execute(payload, runner=runner, now=1000)

    def test_local_timeout_consumes_before_ssh_and_never_retries(self):
        self.payload(); calls = []
        def runner(args, **kw):
            calls.append(args)
            self.assertTrue(self.local.with_suffix(".local_attempt.json").is_file())
            raise subprocess.TimeoutExpired(args, 100)
        out = d.dispatch(self.local, str(self.root), self.sha, runner=runner, now=1000)
        self.assertEqual(out["status"], "ambiguous_transport")
        with self.assertRaises((ValueError, FileExistsError)):
            d.dispatch(self.local, str(self.root), self.sha, runner=runner, now=1000)
        self.assertEqual(len(calls), 1)

    def test_local_transport_success_has_exact_ssh_target(self):
        self.payload(); calls = []
        def runner(args, **kw):
            calls.append((args,kw)); return subprocess.CompletedProcess(args, 0, '{"returncode":0}', "")
        out = d.dispatch(self.local, str(self.root), self.sha, runner=runner, now=1000)
        self.assertEqual(out["returncode"], 0)
        self.assertEqual(calls[0][0][-2], "gpu-03")
        self.assertNotIn("shell", calls[0][1])

    def test_remote_symlink_root_rejected(self):
        link = self.base / "link"; link.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "Symlink"): d._directory(link)

    def run_entry(self, *, break_limit=False, break_binding=False):
        for filename in self.m["bound_files"]:
            p = Path(filename); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
            self.m["bound_files"][filename] = {"size_bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
        if break_binding:
            self.m["bound_files"][filename]["sha256"] = "0"*64
        self.payload()
        relative = "/user.slice/" + self.m["unit"] + ".service"
        values = {"memory.max": str(4*1024**3), "memory.current": "100", "memory.swap.max": "0",
                  "pids.max": "64", "cpu.max": "200000 100000"}
        if break_limit: values["memory.swap.max"] = "1"
        original_read = Path.read_text
        def read(path, *args, **kw):
            if str(path) == "/proc/self/cgroup": return "0::" + relative + "\n"
            if str(path).startswith("/sys/fs/cgroup/"): return values[path.name]
            return original_read(path, *args, **kw)
        with patch.object(entry, "ROOT", self.root), patch.object(d.time, "time", return_value=1000), \
             patch.object(Path, "read_text", read), \
             patch.object(entry.sys, "argv", ["entry", "--manifest", str(self.local), "--sha256", self.sha]), \
             patch.object(g, "supervise", return_value={"status": "workers_exited_zero"}) as supervise:
            if break_limit or break_binding:
                with self.assertRaises(ValueError): entry.main()
                supervise.assert_not_called()
                self.assertFalse(Path(self.m["entry_receipt"]).exists())
            else:
                self.assertEqual(entry.main(), 0)
                self.assertTrue(callable(supervise.call_args.kwargs["gpu_rejection_callback"]))
                self.assertEqual(json.loads(Path(self.m["entry_receipt"]).read_text())["bound_files"], 5)

    def test_entry_complete_success_checks_files_limits_and_supplies_callback(self):
        self.run_entry()

    def test_entry_limit_failure_never_supervises(self):
        self.run_entry(break_limit=True)

    def test_entry_source_failure_never_supervises(self):
        self.run_entry(break_binding=True)


class DiagnosticIntegrationTests(original.GuardTests):
    def test_callback_records_rejection_and_does_not_admit_same_uid(self):
        p = original.Probe(self.devices()); p.extra = {900: original.row(900)}
        p.devices["0"]["pids"] = [900]; seen = []
        p.gpu_rejection_callback = lambda *args: seen.append(args)
        with self.assertRaisesRegex(g.GuardError, "unverified"):
            g.inspect_resources(self.manifest(True), [], p)
        self.assertEqual(len(seen), 1); self.assertIn(900, seen[0][3])

    def test_failing_callback_rethrows_exact_original_error(self):
        p = original.Probe(self.devices()); expected = g.GuardError("original")
        def broken(*_): raise OSError("writer")
        p.gpu_rejection_callback = broken
        with patch.object(g, "check_devices", side_effect=expected):
            with self.assertRaises(g.GuardError) as seen:
                g.inspect_resources(self.manifest(True), [], p)
        self.assertIs(seen.exception, expected); self.assertEqual(expected.diagnostic_error_type, "OSError")

    def test_entry_callback_writes_only_rejection_journal(self):
        runtime = self.root / "runtime"; runtime.mkdir()
        with patch.object(entry, "collect", return_value={"diagnostic_only":True}):
            entry.rejection_callback(runtime)({}, {}, {}, {}, "/service")
        self.assertEqual(json.loads((runtime/"gpu_rejections.jsonl").read_text()), {"diagnostic_only":True})


if __name__ == "__main__":
    unittest.main()
