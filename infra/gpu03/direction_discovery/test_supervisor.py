"""Offline manifest, safety, lifecycle, launch, and verification regression tests."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import supervisor as sup


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.token = "codex-direction-test-20260907"
        self.stage = self.root / (self.token + "-stage")
        self.source = self.stage / "source"
        self.output = self.root / (self.token + "-results")
        self.runtime = self.root / (self.token + "-runtime")
        self.script = self.source / "infra/gpu03/direction_discovery/supervisor.py"
        self.engine = self.script.with_name("engine.py")
        self.script.parent.mkdir(parents=True)
        self.script.write_text("# immutable authored source\n")
        self.engine.write_text("# immutable authored worker\n")
        self.script.chmod(0o400); self.engine.chmod(0o400)
        self.task = self.stage / "gpu_0.json"
        self.task.write_text(json.dumps({"run_token": self.token, "worker_name": "gpu_0",
                                        "output": str(self.output / "workers/gpu_0"), "mode": "qualify",
                                        "requests": [{"request_id": "example"}], "deadline_seconds": 500}))
        self.path = self.stage / "manifest.json"
        self.m = {
            "schema_version": 1, "purpose": "direction_discovery_campaign", "phase": "qualification",
            "run_token": self.token, "host": "gpu-04", "owner": sup.OWNER,
            "stage": str(self.stage), "source_root": str(self.source), "output": str(self.output),
            "runtime": str(self.runtime), "python": "/usr/bin/python3", "scientific": {"training": False},
            "authorization": "User explicitly requested idle gpu-04 direction discovery", "supervisor_cpu_set": "94",
            "gpu_ids": [0], "gpu_uuids": {"0": "GPU-00000000-0000-0000-0000-000000000001"},
            "runtime_versions": {"test-package": "test"},
            "limits": {"runtime_seconds": 600, "systemd_runtime_seconds": 900, "min_available_ram_gib": 192,
                       "max_worker_rss_gib": 128, "cgroup_memory_gib": 160, "min_start_free_disk_gib": 64,
                       "max_worker_log_mib": 64, "tasks_max": 256, "load_stagger_seconds": 10},
            "bound_files": {str(p): {"size_bytes": p.stat().st_size, "sha256": sup.sha256(p)}
                            for p in (self.script, self.engine, self.task)},
            "workers": [{"name": "gpu_0", "gpu_id": 0, "cpu_set": "96,97",
                         "command": ["/usr/bin/python3", str(self.engine), "--task", str(self.task)],
                         "success_file": "workers/gpu_0/SUCCESS.json", "success_expect": {"requests": 1, "mode": "qualify"}}],
        }
        self.m["command"] = sup.expected_command(self.m, self.path)
        self.root_patch = patch.object(sup, "RUN_ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def freeze(self, m=None):
        self.path.write_text(sup.canonical(self.m if m is None else m) + "\n")
        return sup.sha256(self.path)

    def test_valid_generic_manifest_has_no_raw_only_scientific_contract(self):
        self.assertEqual(sup.load_manifest(self.path, self.freeze())["phase"], "qualification")

    def test_manifest_digest_mismatch_is_rejected(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            sup.load_manifest(self.path, "0" * 64)

    def test_wrong_host_user_and_training_are_rejected(self):
        changes = [("host", "gpu-03"), ("owner", "another_user"), ("scientific", {"training": True})]
        for key, value in changes:
            m = copy.deepcopy(self.m); m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sup.load_manifest(self.path, self.freeze(m))

    def test_gpu_subset_cannot_expand_or_duplicate(self):
        changes = [("gpu_ids", [0, 1]), ("gpu_ids", [0, 0]), ("gpu_ids", [True]),
                   ("gpu_uuids", {"0": "invalid"})]
        for key, value in changes:
            m = copy.deepcopy(self.m); m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sup.load_manifest(self.path, self.freeze(m))

    def test_unsafe_and_overlapping_paths_rejected(self):
        for value in (str(self.root / "unrelated"), str(self.stage), str(self.root / (self.token + "-x") / ".." / "other")):
            m = copy.deepcopy(self.m); m["output"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                sup.load_manifest(self.path, self.freeze(m))

    def test_bound_source_and_input_content_changes_are_rejected(self):
        digest = self.freeze()
        self.task.write_text("modified")
        with self.assertRaisesRegex(ValueError, "Bound source/input changed"):
            sup.load_manifest(self.path, digest)

    def test_mutable_or_unbound_source_is_rejected(self):
        self.engine.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "immutable"):
            sup.load_manifest(self.path, self.freeze())
        self.engine.chmod(0o400)
        (self.source / "unexpected.py").write_text("# unbound\n")
        with self.assertRaisesRegex(ValueError, "Unbound"):
            sup.load_manifest(self.path, self.freeze())

    def test_source_symlink_is_rejected(self):
        (self.source / "link").symlink_to(self.task)
        with self.assertRaisesRegex(ValueError, "symlink"):
            sup.load_manifest(self.path, self.freeze())

    def test_worker_command_and_task_identity_are_bound(self):
        m = copy.deepcopy(self.m); m["workers"][0]["command"][2] = "--train"
        with self.assertRaises(ValueError):
            sup.load_manifest(self.path, self.freeze(m))
        task = json.loads(self.task.read_text()); task["worker_name"] = "different"
        self.task.write_text(json.dumps(task))
        self.m["bound_files"][str(self.task)] = {"size_bytes": self.task.stat().st_size, "sha256": sup.sha256(self.task)}
        with self.assertRaisesRegex(ValueError, "Task identity"):
            sup.load_manifest(self.path, self.freeze())

    def test_cpu_allocation_cannot_use_candidate_fit_cores(self):
        m = copy.deepcopy(self.m); m["workers"][0]["cpu_set"] = "118,119"
        with self.assertRaisesRegex(ValueError, "CPU allocation"):
            sup.load_manifest(self.path, self.freeze(m))
        m = copy.deepcopy(self.m); m["supervisor_cpu_set"] = "127"
        with self.assertRaisesRegex(ValueError, "CPU 94"):
            sup.load_manifest(self.path, self.freeze(m))

    def test_independent_deadline_and_memory_cannot_be_unbounded(self):
        for key, value in (("systemd_runtime_seconds", 600), ("runtime_seconds", 90000),
                           ("min_available_ram_gib", 100), ("cgroup_memory_gib", 128),
                           ("max_worker_log_mib", 1024), ("load_stagger_seconds", 1)):
            m = copy.deepcopy(self.m); m["limits"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sup.load_manifest(self.path, self.freeze(m))

    def test_idle_qualification_requires_full_sixty_seconds(self):
        clock = [0.0]
        observed = []
        with patch.object(sup, "resource_state", side_effect=lambda *_: observed.append(clock[0]) or {"safe": True}):
            sup.qualify_idle(self.m, self.stage / "qualification.jsonl", 100,
                             clock=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay))
        self.assertEqual(observed, list(range(0, 61, 5)))

    def test_foreign_process_during_qualification_stops_without_selecting_other_gpu(self):
        with patch.object(sup, "resource_state", side_effect=ValueError("foreign process")), \
             patch.object(sup.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "foreign"):
                sup.qualify_idle(self.m, self.stage / "qualification.jsonl", 100, clock=lambda: 0, sleep=lambda _: None)
        popen.assert_not_called()
        self.assertEqual(self.m["gpu_ids"], [0])

    def test_release_does_not_ignore_foreign_process_during_settling(self):
        with patch.object(sup.raw, "gpu_snapshot", return_value=[{"foreign": True}]), \
             patch.object(sup.raw, "check_devices", side_effect=ValueError("foreign process")):
            with self.assertRaisesRegex(ValueError, "foreign"):
                sup.verify_release(self.m, self.stage / "journal.jsonl")

    def test_release_waits_for_empty_gpu_memory_to_settle(self):
        clock = [0.0]
        def check(_m, _inventory, **kwargs):
            if "settling" not in kwargs and clock[0] < 5:
                raise ValueError("memory settling")
        with patch.object(sup.raw, "gpu_snapshot", return_value=[]), patch.object(sup.raw, "check_devices", side_effect=check):
            result = sup.verify_release(self.m, self.stage / "journal.jsonl", timeout=10,
                                        clock=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay))
        self.assertTrue(result["verified"])
        self.assertEqual(clock[0], 5)

    def test_success_requires_exact_problem_request_count_and_identity(self):
        worker = self.m["workers"][0]
        path = self.output / worker["success_file"]
        path.parent.mkdir(parents=True)
        for payload in ({"status": "success"}, {"status": "succeeded", "run_token": self.token,
                       "worker_name": "gpu_0", "requests": 2, "mode": "qualify"}):
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                sup.validate_worker_success(self.m, worker)
        path.write_text(json.dumps({"status": "succeeded", "run_token": self.token, "worker_name": "gpu_0",
                                    "requests": 1, "mode": "qualify"}))
        self.assertEqual(sup.validate_worker_success(self.m, worker)["worker"], "gpu_0")

    def fake_lifecycle(self, *, worker_receipt=True):
        """Exercise the complete producer path without starting processes or GPUs."""
        stack = ExitStack()
        stack.enter_context(patch.object(sup.socket, "gethostname", return_value="gpu-04"))
        stack.enter_context(patch.object(sup.pwd, "getpwuid", return_value=types.SimpleNamespace(pw_name=sup.OWNER)))
        stack.enter_context(patch.object(sup.importlib.metadata, "version", return_value="test"))
        stack.enter_context(patch.object(sup.os, "sched_getaffinity", return_value=set(range(128)), create=True))
        stack.enter_context(patch.object(sup.os, "sched_setaffinity", create=True))
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
        stack.enter_context(patch.object(sup.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1024**4)))
        stack.enter_context(patch.object(sup, "qualify_idle", return_value={"safe": True}))
        stack.enter_context(patch.object(sup, "resource_state", return_value={"safe": True}))
        stack.enter_context(patch.object(sup, "verify_release", return_value={"verified": True, "gpu_ids": [0]}))
        def popen(command, **_kwargs):
            task = json.loads(Path(command[-1]).read_text())
            target = Path(task["output"])
            target.mkdir(parents=True)
            if worker_receipt:
                sup.exclusive_json(target / "SUCCESS.json", {"status": "succeeded", "run_token": self.token,
                    "worker_name": "gpu_0", "requests": 1, "mode": "qualify"})
            return types.SimpleNamespace(pid=91919, returncode=0, poll=lambda: 0)
        stack.enter_context(patch.object(sup.subprocess, "Popen", side_effect=popen))
        return stack

    def test_success_path_integrates_producer_receipts_hashes_and_independent_verifier(self):
        digest = self.freeze()
        with self.fake_lifecycle():
            sup.supervise(self.path, digest)
        (self.stage / "control").mkdir()
        with patch.dict(os.environ, {"SERVICE_RESULT": "success", "EXIT_CODE": "exited", "EXIT_STATUS": "0",
                                     "INVOCATION_ID": "a" * 32}):
            sup.receipt(self.path, digest)
        self.assertEqual(sup.verify(self.path, digest)["status"], "verified")
        self.assertTrue((self.output / "artifact_manifest.json").is_file())

    def test_worker_exit_zero_without_success_receipt_cannot_succeed(self):
        digest = self.freeze()
        with self.fake_lifecycle(worker_receipt=False):
            with self.assertRaisesRegex(ValueError, "success receipt"):
                sup.supervise(self.path, digest)
        self.assertTrue((self.output / "FAILURE.json").is_file())
        self.assertFalse((self.output / "campaign_summary.json").exists())

    def test_fixed_cache_finalization_precedes_manifest_and_independent_inspection(self):
        self.m['phase'] = 'fixed_cache_core'
        digest = self.freeze()
        semantic = {'status': 'verified', 'records': 561, 'index_joined_to_artifact_manifest': False}
        def finalize(manifest, output):
            self.assertTrue((output / 'gpu_release.json').exists())
            self.assertFalse((output / 'artifact_manifest.json').exists())
            (output / 'activation_index.jsonl').write_text('authored fixture\n')
            return semantic
        fake = types.SimpleNamespace(finalize=finalize, inspect=lambda _: {**semantic, 'index_joined_to_artifact_manifest': True})
        with patch.dict('sys.modules', {'cache_package': fake}), self.fake_lifecycle():
            sup.supervise(self.path, digest)
            with patch.dict(os.environ, {'SERVICE_RESULT': 'success', 'EXIT_CODE': 'exited', 'EXIT_STATUS': '0',
                                         'INVOCATION_ID': 'a' * 32}):
                sup.receipt(self.path, digest)
            self.assertEqual(sup.verify(self.path, digest)['status'], 'verified')
            fake.inspect = lambda _: {'status': 'verified', 'records': 560}
            with self.assertRaisesRegex(ValueError, 'semantics differ'):
                sup.verify(self.path, digest)

    def test_fixed_cache_finalization_failure_cannot_produce_success(self):
        self.m['phase'] = 'fixed_cache_auxiliary'
        digest = self.freeze()
        def fail(*args):
            raise ValueError('missing cache records')
        with patch.dict('sys.modules', {'cache_package': types.SimpleNamespace(finalize=fail)}), self.fake_lifecycle():
            with self.assertRaisesRegex(ValueError, 'missing cache records'):
                sup.supervise(self.path, digest)
        self.assertTrue((self.output / 'FAILURE.json').exists())
        self.assertFalse((self.output / 'campaign_summary.json').exists())
        self.assertFalse((self.output / 'artifact_manifest.json').exists())

    def test_existing_output_is_preserved_and_not_silently_restarted(self):
        digest = self.freeze(); self.output.mkdir(); (self.output / "important").write_text("keep")
        with self.fake_lifecycle():
            with self.assertRaisesRegex(ValueError, "Preserve prior attempt"):
                sup.supervise(self.path, digest)
        self.assertEqual((self.output / "important").read_text(), "keep")

    def test_independent_verifier_rejects_missing_or_default_exit_status(self):
        digest = self.freeze()
        with self.fake_lifecycle():
            sup.supervise(self.path, digest)
        with patch.dict(os.environ, {"SERVICE_RESULT": "unknown", "EXIT_CODE": "unknown", "EXIT_STATUS": "unknown"}):
            sup.receipt(self.path, digest)
        with self.assertRaisesRegex(ValueError, "exit receipt"):
            sup.verify(self.path, digest)

    def test_independent_verifier_detects_post_producer_output_change(self):
        digest = self.freeze()
        with self.fake_lifecycle():
            sup.supervise(self.path, digest)
        with patch.dict(os.environ, {"SERVICE_RESULT": "success", "EXIT_CODE": "exited", "EXIT_STATUS": "0",
                                     "INVOCATION_ID": "a" * 32}):
            sup.receipt(self.path, digest)
        (self.output / "extra").write_text("unmanifested")
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            sup.verify(self.path, digest)

    def fake_launch(self, *, show=None):
        stack = ExitStack()
        stack.enter_context(patch.object(sup.socket, "gethostname", return_value="gpu-04"))
        stack.enter_context(patch.object(sup.pwd, "getpwuid", return_value=types.SimpleNamespace(pw_name=sup.OWNER)))
        stack.enter_context(patch.object(sup.raw, "gpu_snapshot", return_value=[]))
        stack.enter_context(patch.object(sup.raw, "check_devices"))
        text = show or "ActiveState=active\nSubState=running\nMainPID=123\nControlGroup=/test\nRuntimeMaxUSec=15min\nMemoryMax=171798691840\nKillMode=control-group\nInvocationID=" + "a" * 32
        results = [subprocess.CompletedProcess([], 0, "", ""), subprocess.CompletedProcess([], 0, text, ""),
                   subprocess.CompletedProcess([], 0, "", "")]
        runner = stack.enter_context(patch.object(sup.subprocess, "run", side_effect=results))
        return stack, runner

    def test_launch_binds_deadline_cgroup_receipt_and_clear_environment(self):
        digest = self.freeze()
        stack, runner = self.fake_launch()
        with stack, patch.dict(os.environ, {"HF_TOKEN": "secret-not-forwarded"}):
            launched = sup.launch(self.path, digest)
        command = runner.call_args_list[0].args[0]
        self.assertIn("--property=RuntimeMaxSec=900", command)
        self.assertIn("--property=KillMode=control-group", command)
        self.assertIn("/usr/bin/env", command)
        self.assertIn("-i", command)
        self.assertFalse(any("secret-not-forwarded" in item or item.startswith("HF_TOKEN=") for item in command))
        self.assertTrue(any(item.startswith("--property=ExecStopPost=") and "--receipt" in item for item in command))
        self.assertEqual(launched["status"], "launched")

    def test_remote_launch_rejects_absent_permit_before_any_service(self):
        self.remote_launch_rejects_absent_permit('gpu-02')

    def remote_launch_rejects_absent_permit(self, host):
        from infra.gpu03.direction_discovery import remote_generation
        self.m['host'] = host; self.m['remote_generation'] = {'authored': True}
        digest = self.freeze(); stack, runner = self.fake_launch()
        with stack, patch.object(sup.socket, 'gethostname', return_value=host), \
             patch.object(remote_generation, 'validate_manifest'):
            with self.assertRaisesRegex(ValueError, 'exact externally committed permit'):
                sup.launch(self.path, digest)
        runner.assert_not_called()
        self.assertFalse((self.stage / 'control').exists())

    def test_remote_launch_consumes_exact_permit_before_systemd_start(self):
        self.remote_launch_consumes_exact_permit('gpu-02')

    def remote_launch_consumes_exact_permit(self, host):
        from infra.gpu03.direction_discovery import remote_generation
        self.m['host'] = host; self.m['remote_generation'] = {'authored': True}
        digest = self.freeze(); stack, runner = self.fake_launch()
        events = []
        def consume(*args, **kwargs):
            self.assertTrue(kwargs['consume']); self.assertEqual(args[1:], (digest, Path('/reviewed/permit.json'), 'f'*64))
            self.assertEqual(runner.call_count, 0); events.append('permit_consumed')
        with stack, patch.object(sup.socket, 'gethostname', return_value=host), \
             patch.object(remote_generation, 'validate_manifest'), \
             patch.object(remote_generation, 'validate_launch_permit', side_effect=consume):
            result = sup.launch(self.path, digest, permit_path=Path('/reviewed/permit.json'), permit_sha256='f'*64)
        self.assertEqual(result['status'], 'launched'); self.assertEqual(events, ['permit_consumed'])
        self.assertEqual(runner.call_count, 2)

    def test_gpu01_launch_rejects_absent_permit_before_any_service(self):
        self.remote_launch_rejects_absent_permit('gpu-01')

    def test_gpu01_launch_consumes_exact_permit_before_systemd_start(self):
        self.remote_launch_consumes_exact_permit('gpu-01')

    def test_gpu01_manifest_requires_explicit_remote_profile(self):
        self.m['host'] = 'gpu-01'
        with self.assertRaisesRegex(ValueError, 'explicit authority profile'):
            sup.load_manifest(self.path, self.freeze())

    def test_gpu01_supervisor_rechecks_consumed_permit_before_output(self):
        from infra.gpu03.direction_discovery import remote_generation
        self.m['host']='gpu-01';self.m['remote_generation']={'authored':True}
        digest=self.freeze();stack,runner=self.fake_launch()
        with stack,patch.object(sup.socket,'gethostname',return_value='gpu-01'), \
             patch.object(sup.importlib.metadata,'version',return_value='test'), \
             patch.object(remote_generation,'validate_manifest'), \
             patch.object(remote_generation,'validate_consumption',side_effect=RuntimeError('unconsumed permit')) as consumed:
            with self.assertRaisesRegex(RuntimeError,'unconsumed permit'):sup.supervise(self.path,digest)
        consumed.assert_called_once_with(self.m,digest)
        runner.assert_not_called();self.assertFalse(self.output.exists())

    def test_failed_independent_start_verification_stops_only_new_exact_unit(self):
        digest = self.freeze()
        stack, runner = self.fake_launch(show="ActiveState=inactive\nSubState=dead\nMainPID=0\n")
        with stack:
            with self.assertRaisesRegex(ValueError, "positively verified active"):
                sup.launch(self.path, digest)
        self.assertEqual(runner.call_args_list[-1].args[0], ["systemctl", "--user", "stop", self.token + ".service"])
        self.assertTrue((self.stage / "control/launch_verification_failure.json").is_file())


if __name__ == "__main__":
    unittest.main()
