"""Offline resource/ownership and complete supervisor success/failure paths.

No GPU, SSH, model or generated program is used. Full orchestration tests inject
only OS/hardware observations and child lifecycle; real files/locks are exercised.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.task_specific_discovery import resource_guard as g


def row(pid, *, start=10, uid=None, ppid=1, pgid=None, sid=None, rss=100, state="R"):
    return {"pid": pid, "start": start, "uid": g.UID if uid is None else uid,
            "ppid": ppid, "pgid": pid if pgid is None else pgid,
            "sid": pid if sid is None else sid, "rss": rss, "state": state}


class Clock:
    def __init__(self): self.now = 1000.
    def time(self): return self.now
    def sleep(self, seconds): self.now += max(0, seconds)


class Process:
    def __init__(self, pid=300, code=0):
        self.pid, self.code, self.done = pid, code, False
    def poll(self):
        self.done = True
        return self.code
    def wait(self, timeout=None):
        self.done = True
        return self.code


class Probe:
    def __init__(self, devices=None):
        self.devices = devices or {}
        self.children, self.extra = [], {}
        self.gpu_calls = 0
    def processes(self):
        return {**self.extra, **{p.pid: row(p.pid) for p in self.children if not p.done}}
    def gpus(self):
        self.gpu_calls += 1
        return copy.deepcopy(self.devices)
    def available(self): return 300 * g.GIB
    def disk(self, path): return 500 * g.GIB


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(g, "RUN_ROOT", self.root))
        self.stack.enter_context(patch.object(g, "LOCK_ROOT", self.root / "triplet_gpu_locks"))
        self.stack.enter_context(patch.object(g, "UID", os.getuid()))
        self.clock = Clock()
        self.m = self.manifest()

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def manifest(self, gpu=False):
        uuid = "GPU-00000000-0000-0000-0000-000000000000"
        return {"schema": g.SCHEMA, "host": "gpu-03", "uid": g.UID,
                "run_token": "codex-discovery-test-20260921", "deadline_epoch": 2000,
                "unit": "codex-discovery-test-20260921-workers",
                "runtime": str(self.root / "codex-discovery-test-20260921-runtime"),
                "disk_path": str(self.root), "gpu_uuids": {"0": uuid} if gpu else {},
                "workers": [{"name": "worker0", "gpu_id": 0 if gpu else None,
                             "argv": ["/usr/bin/python3", "-c", "pass"], "cwd": str(self.root), "cpu_ids": [0],
                             "env": {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": uuid if gpu else "",
                                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                     "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1"}}]}

    def devices(self):
        return {"0": {"uuid": self.manifest(True)["gpu_uuids"]["0"], "total_mib": 32760,
                      "free_mib": 32760, "utilization": 0, "pids": []}}

    def write_manifest(self, m=None):
        data = (g.canonical(m or self.m) + "\n").encode()
        path = self.root / "manifest.json"
        path.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()

    def test_cpu_and_exact_gpu_manifests_accepted(self):
        g.validate_manifest(self.m)
        g.validate_manifest(self.manifest(True))

    def test_wrong_host_uid_duplicate_gpu_and_environment_rejected(self):
        for key, value in (("host", "gpu-04"), ("uid", g.UID + 1)):
            m = copy.deepcopy(self.m); m[key] = value
            with self.subTest(key=key), self.assertRaises(g.GuardError): g.validate_manifest(m)
        for name in ("AWS_ACCESS_KEY_ID", "WANDB_API_KEY", "HF_TOKEN", "SSH_AUTH_SOCK", "LD_PRELOAD"):
            m = copy.deepcopy(self.m); m["workers"][0]["env"][name] = "not-a-real-secret"
            with self.subTest(name=name), self.assertRaises(g.GuardError): g.validate_manifest(m)

    def test_cuda_mask_and_cpu_only_gpu_claim_rejected(self):
        m = self.manifest(True); m["workers"][0]["env"]["CUDA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(g.GuardError, "CUDA mask"): g.validate_manifest(m)
        m = self.manifest(); m["workers"][0]["gpu_id"] = 0
        with self.assertRaises(g.GuardError): g.validate_manifest(m)

    def test_worker_map_and_overlapping_cpu_affinity_rejected(self):
        m = self.manifest(True); m["workers"].append(copy.deepcopy(m["workers"][0]))
        m["workers"][1].update(name="worker1", cpu_ids=[1])
        with self.assertRaisesRegex(g.GuardError, "Exactly one"): g.validate_manifest(m)
        m = self.manifest(); m["workers"].append(copy.deepcopy(m["workers"][0])); m["workers"][1]["name"] = "worker1"
        with self.assertRaisesRegex(g.GuardError, "CPU masks"): g.validate_manifest(m)

    def test_manifest_hash_and_duplicate_keys_fail(self):
        path, digest = self.write_manifest()
        self.assertEqual(g.load_manifest(path, digest), self.m)
        with self.assertRaisesRegex(g.GuardError, "SHA256"): g.load_manifest(path, "0" * 64)
        data = b'{"schema":1,"schema":2}'
        path.write_bytes(data)
        with self.assertRaisesRegex(g.GuardError, "Duplicate"): g.load_manifest(path, hashlib.sha256(data).hexdigest())

    def test_gpu_uuid_and_same_uid_foreign_process_fail(self):
        m = self.manifest(True); devices = self.devices()
        devices["0"]["uuid"] = "GPU-11111111-1111-1111-1111-111111111111"
        with self.assertRaisesRegex(g.GuardError, "UUID"): g.check_devices(m["gpu_uuids"], devices, {})
        devices = self.devices(); devices["0"]["pids"] = [900]
        with self.assertRaisesRegex(g.GuardError, "Foreign"): g.check_devices(m["gpu_uuids"], devices, {300: row(300)})

    def test_occupied_gpu_is_not_idle_without_compute_pid(self):
        d = self.devices(); d["0"]["free_mib"] -= 2000
        with self.assertRaisesRegex(g.GuardError, "not idle"): g.check_devices(self.manifest(True)["gpu_uuids"], d, {}, idle=True)

    def test_sixty_seconds_continuous_idle_and_no_cpu_gpu_query(self):
        m = self.manifest(True); probe = Probe(self.devices()); seen = []
        with patch.object(g.time, "time", self.clock.time):
            g.qualify_idle(m, [], probe, seen.append, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertEqual([r["qualification_elapsed"] for r in seen], list(range(0, 61, 5)))
        probe = Probe()
        g.qualify_idle(self.m, [], probe, lambda _: None)
        self.assertEqual(probe.gpu_calls, 0)

    def test_foreign_appearance_during_qualification_aborts(self):
        p = Probe(self.devices()); original = p.gpus
        def snapshot():
            d = original()
            if self.clock.now >= 1010: d["0"]["pids"] = [900]
            return d
        p.gpus = snapshot; p.extra = {900: row(900)}
        with patch.object(g.time, "time", self.clock.time), self.assertRaisesRegex(g.GuardError, "Foreign"):
            g.qualify_idle(self.manifest(True), [], p, lambda _: None, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertEqual(self.clock.now, 1010)

    def test_nvml_exit_race_requires_fresh_positive_absence(self):
        p = Probe(self.devices()); stale = self.devices(); stale["0"]["pids"] = [900]
        with patch.object(p, "gpus", side_effect=[stale, self.devices()]):
            g.inspect_resources(self.manifest(True), [], p)
        with patch.object(p, "gpus", return_value=stale), self.assertRaisesRegex(g.GuardError, "unverified"):
            g.inspect_resources(self.manifest(True), [], p)

    def test_resource_floor_and_aggregate_rss_limits(self):
        p = Probe(); worker = g.OwnedWorker("w", Process(), 10)
        p.extra = {300: row(300, rss=g.MAX_RSS + 1)}
        with self.assertRaisesRegex(g.GuardError, "RSS"): g.inspect_resources(self.m, [worker], p)
        p.extra = {}
        with patch.object(p, "available", return_value=g.MIN_AVAILABLE - 1), self.assertRaisesRegex(g.GuardError, "RAM"):
            g.inspect_resources(self.m, [], p)
        with patch.object(p, "disk", return_value=g.MIN_DISK - 1), self.assertRaisesRegex(g.GuardError, "Disk"):
            g.inspect_resources(self.m, [], p)

    def test_log_limit_is_checked(self):
        p = Path(self.m["runtime"]); p.mkdir()
        with (p / "worker0.log").open("wb") as f: f.truncate(g.MAX_LOG + 1)
        with self.assertRaisesRegex(g.GuardError, "log exceeded"): g.inspect_resources(self.m, [], Probe())

    def test_descendant_private_groups_and_parent_exit_remain_owned(self):
        w = g.OwnedWorker("w", Process(), 10)
        table = {300: row(300), 301: row(301, ppid=300, pgid=301, sid=300),
                 302: row(302, ppid=301, pgid=302, sid=302), 900: row(900)}
        self.assertEqual(set(g.owned_rows([w], table)), {300, 301, 302})
        del table[300]; table[301]["ppid"] = 1; table[302]["ppid"] = 1
        self.assertEqual(set(g.owned_rows([w], table)), {301, 302})

    def test_pid_reuse_and_foreign_group_member_are_never_signalled(self):
        w = g.OwnedWorker("w", Process(), 10)
        sent = []
        with self.assertRaisesRegex(g.GuardError, "identity changed"):
            g.signal_owned([w], {300: row(300, start=11)}, signal.SIGTERM, sender=lambda *x: sent.append(x))
        table = {300: row(300), 901: row(901, uid=g.UID + 1, pgid=300, sid=300)}
        with self.assertRaisesRegex(g.GuardError, "unverified process group"):
            g.signal_owned([w], table, signal.SIGTERM, sender=lambda *x: sent.append(x))
        self.assertEqual(sent, [])

    def test_only_owned_groups_signalled_and_esrch_accepted(self):
        w = g.OwnedWorker("w", Process(), 10); sent = []
        g.signal_owned([w], {300: row(300), 900: row(900)}, signal.SIGTERM, sender=lambda *x: sent.append(x))
        self.assertEqual(sent, [(300, signal.SIGTERM)])
        def gone(*_): raise ProcessLookupError()
        g.signal_owned([w], {300: row(300)}, signal.SIGTERM, sender=gone)

    def test_zombies_do_not_count_as_positive_process_release(self):
        p = Probe(); w = g.OwnedWorker("w", Process(), 10)
        p.extra = {300: row(300, state="Z")}
        with self.assertRaisesRegex(g.GuardError, "entries remain"):
            g.verify_release(self.m, [w], p, lambda _: None, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertEqual(self.clock.now, 1060)
        self.assertEqual(p.gpu_calls, 0)

    def test_cleanup_escalates_only_live_owned_groups(self):
        p = Probe(); child = Process(); w = g.OwnedWorker("w", child, 10)
        child.poll = lambda: 0 if child.done else None
        p.extra = {300: row(300), 900: row(900)}
        sent = []
        def signal_rows(workers, table, sig, **kwargs):
            actual = g.owned_rows(workers, table)
            sent.append((sorted(actual), sig))
            if sig == signal.SIGKILL:
                p.extra.pop(300); child.done = True
        with patch.object(g, "signal_owned", side_effect=signal_rows):
            result = g.terminate_owned([w], p, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertEqual(sent, [([300], signal.SIGTERM), ([300], signal.SIGKILL)])
        self.assertTrue(result["owned_processes_released"])
        self.assertIn(900, p.extra)

    def cgroup_fixture(self, initial="10\n"):
        proc = self.root / "proc"; cg = self.root / "cgroup"
        (proc / "10").mkdir(parents=True)
        relative = "/user.slice/" + self.m["unit"] + ".service"
        directory = cg / relative.lstrip("/"); directory.mkdir(parents=True)
        (proc / "10" / "cgroup").write_text("0::" + relative + "\n")
        (directory / "cgroup.procs").write_text(initial)
        return proc, cg, directory

    def test_cgroup_requires_exact_empty_service_and_stable_identity(self):
        proc, cg, directory = self.cgroup_fixture("10\n900\n")
        with self.assertRaisesRegex(g.GuardError, "already contains"):
            g.CgroupScope(self.m, proc_root=proc, cgroup_root=cg, self_pid=10)
        (directory / "cgroup.procs").write_text("10\n")
        scope = g.CgroupScope(self.m, proc_root=proc, cgroup_root=cg, self_pid=10)
        (proc / "10" / "cgroup").write_text("0::/user.slice/other.service\n")
        with self.assertRaisesRegex(g.GuardError, "moved out"):
            scope.members()

    def test_cgroup_catches_orphaned_double_fork_and_rss(self):
        proc, cg, directory = self.cgroup_fixture()
        scope = g.CgroupScope(self.m, proc_root=proc, cgroup_root=cg, self_pid=10)
        child = directory / "worker-child"; child.mkdir()
        (child / "cgroup.procs").write_text("777\n")
        probe = Probe(); probe.scoped = scope.rows
        probe.extra = {10: row(10), 777: row(777, ppid=1, sid=777, rss=g.MAX_RSS + 1), 900: row(900)}
        with self.assertRaisesRegex(g.GuardError, "RSS"):
            g.inspect_resources(self.m, [], probe)
        self.assertEqual(set(g.observed_owned([], probe, probe.processes())), {777})
        # Membership remains a verified process identity after a session/cgroup
        # change. Reused PIDs outside the service do not inherit that identity.
        (child / "cgroup.procs").write_text("")
        self.assertEqual(set(g.observed_owned([], probe, probe.processes())), {777})
        probe.extra[777]["start"] += 1
        self.assertEqual(g.observed_owned([], probe, probe.processes()), {})

    def test_cgroup_foreign_uid_and_bound_count_fail_closed(self):
        proc, cg, directory = self.cgroup_fixture()
        scope = g.CgroupScope(self.m, proc_root=proc, cgroup_root=cg, self_pid=10)
        (directory / "cgroup.procs").write_text("10\n777\n")
        with self.assertRaisesRegex(g.GuardError, "Foreign UID"):
            scope.rows({777: row(777, uid=g.UID + 1)})
        with patch.object(g, "MAX_CGROUP_PIDS", 1), self.assertRaisesRegex(g.GuardError, "process count"):
            scope.members()

    def test_cgroup_orphan_cleanup_signals_only_verified_group(self):
        proc, cg, directory = self.cgroup_fixture()
        scope = g.CgroupScope(self.m, proc_root=proc, cgroup_root=cg, self_pid=10)
        (directory / "cgroup.procs").write_text("10\n777\n")
        probe = Probe(); probe.scoped = scope.rows
        probe.extra = {10: row(10), 777: row(777, ppid=1, sid=777), 900: row(900)}
        sent = []
        original = g.signal_owned
        def invoke(workers, table, sig, *, owned):
            original(workers, table, sig, owned=owned, sender=lambda *args: sent.append(args))
            probe.extra.pop(777)
            (directory / "cgroup.procs").write_text("10\n")
        with patch.object(g, "signal_owned", side_effect=invoke):
            result = g.terminate_owned([], probe, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertTrue(result["owned_processes_released"])
        self.assertEqual(sent, [(777, signal.SIGTERM)])
        self.assertIn(900, probe.extra)

    def test_lock_contention_and_symlink_refused(self):
        mapping = self.manifest(True)["gpu_uuids"]
        fds = g.acquire_gpu_locks(mapping)
        try:
            with self.assertRaises(BlockingIOError): g.acquire_gpu_locks(mapping)
        finally:
            for fd in fds: os.close(fd)
        (g.LOCK_ROOT / "gpu_0.lock").unlink()
        (g.LOCK_ROOT / "gpu_0.lock").symlink_to(self.root / "elsewhere")
        with self.assertRaises(OSError): g.acquire_gpu_locks(mapping)

    def test_cpu_phase_does_not_create_gpu_lock_directory(self):
        self.assertEqual(g.acquire_gpu_locks({}), [])
        self.assertFalse(g.LOCK_ROOT.exists())

    def test_launch_exact_environment_private_group_and_affinity(self):
        p = Process(); log = object(); w = self.m["workers"][0]
        with patch.object(g.subprocess, "Popen", return_value=p) as popen, patch.object(g, "proc_row", return_value=row(300)):
            actual = g.launch_worker(w, log)
        argv, = popen.call_args.args
        self.assertEqual(argv, ["/usr/bin/taskset", "-c", "0", *w["argv"]])
        self.assertEqual(popen.call_args.kwargs["env"], w["env"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertTrue(popen.call_args.kwargs["close_fds"])
        self.assertEqual(actual.start, 10)

    def run_supervisor(self, *, gpu=False, code=0, second=False):
        m = self.manifest(gpu)
        devices = self.devices() if gpu else {}
        if second:
            w = copy.deepcopy(m["workers"][0]); w.update(name="worker1", gpu_id=1 if gpu else None, cpu_ids=[1])
            if gpu:
                uuid = "GPU-11111111-1111-1111-1111-111111111111"
                m["gpu_uuids"]["1"] = uuid; w["env"]["CUDA_VISIBLE_DEVICES"] = uuid
                devices["1"] = {**devices["0"], "uuid": uuid}
            m["workers"].append(w)
        path, digest = self.write_manifest(m)
        probe = Probe(devices); launched = []
        def launch(w, log):
            p = Process(300 + len(launched), code=code)
            probe.children.append(p); launched.append(self.clock.now)
            log.write(b"authored child lifecycle fixture\n"); log.flush()
            return g.OwnedWorker(w["name"], p, 10)
        with ExitStack() as stack:
            stack.enter_context(patch.object(g.socket, "gethostname", return_value="gpu-03"))
            stack.enter_context(patch.dict(g.os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
            stack.enter_context(patch.object(g.os, "sched_getaffinity", return_value={0, 1}, create=True))
            stack.enter_context(patch.object(g.time, "time", self.clock.time))
            stack.enter_context(patch.object(g.time, "monotonic", self.clock.time))
            stack.enter_context(patch.object(g.time, "sleep", self.clock.sleep))
            # Default argument clocks are replaced explicitly for the two loops.
            original = g.qualify_idle
            stack.enter_context(patch.object(g, "qualify_idle", side_effect=lambda *a: original(*a, clock=self.clock.time, sleep=self.clock.sleep)))
            stack.enter_context(patch.object(g, "launch_worker", side_effect=launch))
            result = g.supervise(path, digest, probe=probe)
        saved = json.loads((Path(m["runtime"]) / "SUPERVISOR_RESULT.json").read_text())
        self.assertEqual(result, saved)
        return result, probe, launched

    def test_complete_cpu_success_path_no_gpu_claim_or_query(self):
        result, probe, launched = self.run_supervisor()
        self.assertEqual(result["status"], "workers_exited_zero")
        self.assertTrue(result["owned_processes_released"])
        self.assertIsNone(result["gpu_release_verified"])
        self.assertFalse(result["scientific_success_verified"])
        self.assertEqual(probe.gpu_calls, 0)
        self.assertEqual(launched, [1000])

    def test_complete_gpu_success_idle_stagger_and_release(self):
        result, probe, launched = self.run_supervisor(gpu=True, second=True)
        self.assertEqual(result["status"], "workers_exited_zero")
        self.assertEqual(launched, [1060, 1070])
        self.assertTrue(result["gpu_release_verified"])
        self.assertGreaterEqual(probe.gpu_calls, 17)

    def test_nonzero_worker_exit_preserves_failure_and_cleanup(self):
        result, probe, launched = self.run_supervisor(code=7)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Nonzero worker exit", result["error"])
        self.assertTrue(result["cleanup"]["owned_processes_released"])
        self.assertEqual(result["release"]["gpu_release_verified"], None)

    def test_existing_runtime_refuses_without_launch(self):
        Path(self.m["runtime"]).mkdir()
        path, digest = self.write_manifest()
        with patch.object(g.socket, "gethostname", return_value="gpu-03"), \
             patch.dict(g.os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
             patch.object(g.time, "time", self.clock.time), \
             patch.object(g.os, "sched_getaffinity", return_value={0}, create=True), \
             patch.object(g, "launch_worker") as launch, self.assertRaisesRegex(g.GuardError, "Existing attempt"):
            g.supervise(path, digest)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
