"""Inert fixtures only: no SSH, GPU, process signal, model or generated code."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra.gpu03.task_specific_discovery import resource_guard as g
from infra.gpu03.task_specific_discovery import test_resource_guard as original_tests

Probe, row = original_tests.Probe, original_tests.row

from infra.gpu03.task_specific_discovery import guard_failure_diagnostic as d

class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"0": "GPU-00000000-0000-0000-0000-000000000000"}
        self.snapshot = {"0": {"uuid": self.mapping["0"], "pids": [900],
                               "total_mib": 32760, "free_mib": 32000, "utilization": 1}}

    def fake(self, pid):
        r = row(pid)
        return {"before": r, "after": dict(r), "identity_stable": True,
                "unified_cgroup": "/same.service"}

    def check_failure(self, reader, *, record=None, table=None):
        seen = []
        with self.assertRaisesRegex(g.GuardError, "Foreign or unverified") as result:
            d.check_devices_recording(g, self.mapping, self.snapshot, {}, table or {},
                                      record or seen.append, exact_cgroup="/same.service", reader=reader)
        return result.exception, seen

    def test_same_uid_exact_cgroup_diagnostic_still_rejects(self):
        _, seen = self.check_failure(self.fake, table={900: row(900)})
        p = seen[0]["processes"][0]
        self.assertTrue(p["observed_exact_service_cgroup"])
        self.assertEqual(p["earlier_table"]["uid"], g.UID)
        self.assertEqual(seen[0]["ownership_decision"], "rejected_unchanged")

    def test_permission_and_exit_are_explicit_not_positive_absence(self):
        for exc, status in [(PermissionError(), "metadata_error"), (FileNotFoundError(), "exited_or_unobservable")]:
            with self.subTest(exc=type(exc).__name__):
                def reader(_): raise exc
                _, seen = self.check_failure(reader)
                self.assertEqual(seen[0]["processes"][0]["read_status"], status)

    def test_writer_failure_keeps_original_exception(self):
        def broken(_): raise OSError("do not leak arbitrary text")
        error, _ = self.check_failure(self.fake, record=broken)
        self.assertEqual(error.diagnostic_error_type, "OSError")
        self.assertNotIn("leak", str(error))

    def test_exact_original_exception_object_is_rethrown(self):
        expected = g.GuardError("original rejection")
        with patch.object(g, "check_devices", side_effect=expected):
            with self.assertRaises(g.GuardError) as seen:
                d.check_devices_recording(g, self.mapping, self.snapshot, {}, {}, lambda _: None, reader=self.fake)
        self.assertIs(seen.exception, expected)

    def test_success_does_not_collect(self):
        self.snapshot["0"]["pids"] = [900]
        with patch.object(d, "collect", side_effect=AssertionError("must not read")):
            d.check_devices_recording(g, self.mapping, self.snapshot, {900: row(900)}, {}, lambda _: None)

    def test_rejected_pid_count_and_time_budget(self):
        self.snapshot["0"]["pids"] = list(range(100, 150))
        called = []
        out = d.collect(self.mapping, self.snapshot, {}, {}, owner_uid=g.UID,
                        reader=lambda p: called.append(p), monotonic=iter([0.0] + [2.0] * 33).__next__)
        self.assertTrue(out["truncated_pids"])
        self.assertEqual(out["rejected_pid_count"], 50)
        self.assertEqual(len(out["processes"]), 32)
        self.assertEqual(called, [])

    def test_proc_metadata_omits_comm_cmdline_environ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / "900"; p.mkdir()
            fields = ["R", "2", "3", "4"] + ["0"] * 18
            fields[19] = "1234"
            (p / "stat").write_text("900 (secret comm (parens)) " + " ".join(fields))
            (p / "cgroup").write_text("0::/same.service\n")
            (p / "environ").write_text("secret_env")
            (p / "cmdline").write_text("secret_argv")
            value = d.read_pid_metadata(900, proc_root=root)
            self.assertTrue(value["identity_stable"])
            self.assertEqual(value["before"]["start"], 1234)
            self.assertNotIn("secret", str(value))

    def test_proc_symlink_and_oversize_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / "900"; p.mkdir()
            (p / "stat").symlink_to(root / "secret")
            with self.assertRaises(OSError): d.read_pid_metadata(900, proc_root=root)
            (p / "stat").unlink()
            (p / "stat").write_bytes(b"x" * (d.MAX_STAT_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "bound"): d.read_pid_metadata(900, proc_root=root)

    def test_changed_identity_is_recorded_not_admitted(self):
        def reader(pid):
            result = self.fake(pid)
            result["after"]["start"] += 1
            result["identity_stable"] = False
            return result
        _, seen = self.check_failure(reader)
        self.assertFalse(seen[0]["processes"][0]["proc"]["identity_stable"])

    def test_bracketing_detects_actual_mocked_stat_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / "900"; p.mkdir()
            (p / "cgroup").write_text("0::/same.service\n")
            with patch.object(d, "_stat", side_effect=[row(900, start=1), row(900, start=2)]):
                value = d.read_pid_metadata(900, proc_root=root)
            self.assertFalse(value["identity_stable"])

    def test_cgroup_escape_or_control_characters_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / "900"; p.mkdir()
            for text in ("0::/parent/../escape\n", "0::/bad\tpath\n", "0::/one\n0::/two\n"):
                (p / "cgroup").write_text(text)
                with patch.object(d, "_stat", return_value=row(900)), self.assertRaises(ValueError):
                    d.read_pid_metadata(900, proc_root=root)


if __name__ == "__main__":
    unittest.main()
