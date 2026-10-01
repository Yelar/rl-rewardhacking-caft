"""Local authored fixtures only: no model, evaluator, GPU, or remote calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from infra.gpu03.task_specific_discovery import io_contract as io


class IOContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="discovery-io-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "journal"
        self.kwargs = {"run_token": "authored-task-specific-test", "input_sha256": "1" * 64,
                       "request_hashes": {rid: io.hash_value({"id": rid, "prompt": "authored"})
                                          for rid in ("a", "b", "c")}}
        self.primitive_value = {"gt": True, "counts": [1, 1], "text": "authored output"}

    def journal(self, **changes):
        return io.RequestJournal(self.root, **{**self.kwargs, **changes})

    def populated(self, final=True):
        with self.journal() as j:
            j.start("a")
            j.record_primitive("a", self.primitive_value)
            if final:
                j.record_final("a", {"label": "authored", "unknown": None})

    def payload(self, kind="primitives", rid="a"):
        return self.root / kind / (hashlib.sha256(rid.encode()).hexdigest() + ".json")

    def mutate(self, path, data):
        path.chmod(0o600)
        path.write_bytes(data)
        path.chmod(0o400)

    def test_success_and_repeat_records_are_verified_idempotently(self):
        with self.journal() as j:
            self.assertEqual(j.pending()["never_started"], ["a", "b", "c"])
            j.start("a")
            self.assertTrue(j.record_primitive("a", self.primitive_value))
            self.assertTrue(j.record_final("a", {"label": "yes"}))
            before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
            self.assertFalse(j.record_primitive("a", self.primitive_value))
            self.assertFalse(j.record_final("a", {"label": "yes"}))
            self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        with self.journal() as j:
            self.assertEqual(j.primitive("a"), self.primitive_value)
            self.assertEqual(j.final("a"), {"label": "yes"})
            self.assertEqual(j.pending()["final_recorded"], ["a"])
            with self.assertRaises(io.ReplayRefused):
                j.start("a")

    def test_start_without_return_is_ambiguous_and_never_replayed(self):
        with self.journal() as j:
            j.start("a")
        with self.assertRaises(io.AmbiguousRequest) as caught:
            with self.journal():
                self.fail("Must refuse before any external call")
        self.assertEqual(caught.exception.request_ids, ("a",))

    def test_unexpected_diagnostic_failure_keeps_raw_primitive_and_never_calls_again(self):
        external = mock.Mock(return_value=self.primitive_value)
        with self.assertRaisesRegex(RuntimeError, "optional diagnostic failed"):
            with self.journal() as j:
                j.start("a")
                j.record_primitive("a", external())
                raise RuntimeError("optional diagnostic failed")
        raw_before = self.payload().read_bytes()
        with self.journal() as j:
            self.assertEqual(j.pending()["primitive_ready"], ["a"])
            j.record_final("a", {"derived_gt": j.primitive("a")["gt"], "structure": None})
        self.assertEqual(external.call_count, 1)
        self.assertEqual(self.payload().read_bytes(), raw_before)

    def test_crash_after_primitive_file_before_event_adopts_verified_bytes(self):
        with self.journal() as j:
            j.start("a")
            with mock.patch.object(j, "_append_event", side_effect=RuntimeError("crash before append")):
                with self.assertRaisesRegex(RuntimeError, "crash before append"):
                    j.record_primitive("a", self.primitive_value)
        raw = self.payload().read_bytes()
        self.assertEqual((self.root / "primitive_returned.jsonl").read_bytes(), b"")
        with self.journal() as j:
            self.assertEqual(j.pending()["primitive_ready"], ["a"])
            self.assertEqual(j.primitive("a"), self.primitive_value)
        self.assertEqual(self.payload().read_bytes(), raw)

    def test_crash_after_final_file_before_event_adopts_without_rederivation(self):
        self.populated(final=False)
        with self.journal() as j:
            with mock.patch.object(j, "_append_event", side_effect=RuntimeError("crash")):
                with self.assertRaises(RuntimeError):
                    j.record_final("a", {"label": "closed"})
        with self.journal() as j:
            self.assertEqual(j.pending()["final_recorded"], ["a"])
            self.assertEqual(j.final("a"), {"label": "closed"})

    def test_conflicting_duplicate_payloads_are_rejected_without_overwrite(self):
        self.populated()
        before = self.payload().read_bytes()
        with self.journal() as j:
            with self.assertRaisesRegex(io.ContractError, "Conflicting immutable"):
                j.record_primitive("a", {"gt": False})
            with self.assertRaisesRegex(io.ContractError, "Conflicting immutable"):
                j.record_final("a", {"label": "different"})
        self.assertEqual(self.payload().read_bytes(), before)

    def test_binding_mismatch_refuses_before_reuse(self):
        self.populated()
        for change in ({"input_sha256": "2" * 64}, {"run_token": "different-token"},
                       {"request_hashes": {**self.kwargs["request_hashes"], "a": "f" * 64}}):
            with self.subTest(change=change), self.assertRaisesRegex(io.ContractError, "binding mismatch"):
                with self.journal(**change):
                    pass

    def test_duplicate_journal_id_even_identical_fails_closed(self):
        self.populated()
        path = self.root / "started.jsonl"
        path.write_bytes(path.read_bytes() * 2)
        with self.assertRaisesRegex(io.ContractError, "Duplicate journal"):
            with self.journal():
                pass

    def test_truncated_journal_is_retained_and_not_trimmed(self):
        self.populated()
        path = self.root / "final_recorded.jsonl"
        before = path.read_bytes()[:-1]
        path.write_bytes(before)
        with self.assertRaisesRegex(io.ContractError, "Truncated"):
            with self.journal():
                pass
        self.assertEqual(path.read_bytes(), before)

    def test_truncated_payload_is_retained_and_refused(self):
        self.populated(final=False)
        path = self.payload()
        before = path.read_bytes()[:-7]
        self.mutate(path, before)
        with self.assertRaises(io.ContractError):
            with self.journal():
                pass
        self.assertEqual(path.read_bytes(), before)

    def test_bad_hash_is_not_confused_with_unknown_scientific_outcome(self):
        self.populated()
        path = self.payload()
        row = json.loads(path.read_bytes())
        row["payload"]["gt"] = False
        self.mutate(path, io.canonical(row) + b"\n")
        with self.assertRaisesRegex(io.ContractError, "SHA256 mismatch"):
            with self.journal():
                pass

    def test_final_binds_exact_primitive_hash(self):
        self.populated()
        path = self.payload("finals")
        row = json.loads(path.read_bytes())
        row["primitive_file_sha256"] = "0" * 64
        self.mutate(path, io.canonical(row) + b"\n")
        with self.assertRaisesRegex(io.ContractError, "identity/schema"):
            with self.journal():
                pass

    def test_wrong_request_binding_in_journal_fails(self):
        self.populated()
        path = self.root / "started.jsonl"
        row = json.loads(path.read_bytes())
        row["request_sha256"] = "0" * 64
        path.write_bytes(io.canonical(row) + b"\n")
        with self.assertRaisesRegex(io.ContractError, "identity/schema"):
            with self.journal():
                pass

    def test_duplicate_json_keys_and_nonfinite_values_fail(self):
        self.populated()
        path = self.root / "started.jsonl"
        good = path.read_bytes()
        for bad in (good.replace(b'"schema":1', b'"schema":1,"schema":1'),
                    good.replace(b'"schema":1', b'"schema":NaN')):
            path.write_bytes(bad)
            with self.subTest(bad=bad), self.assertRaises(io.ContractError):
                with self.journal():
                    pass

    def test_unknown_payload_and_missing_referenced_payload_fail(self):
        self.populated()
        alien = self.root / "primitives" / ("0" * 64 + ".json")
        alien.write_bytes(b"{}\n")
        with self.assertRaisesRegex(io.ContractError, "Unknown payload"):
            with self.journal():
                pass
        alien.unlink()
        self.payload().unlink()
        with self.assertRaisesRegex(io.ContractError, "predecessor|missing payload"):
            with self.journal():
                pass

    def test_no_evidence_is_added_when_any_request_is_ambiguous(self):
        with self.journal() as j:
            j.start("a")
            with mock.patch.object(j, "_append_event", side_effect=RuntimeError("crash")):
                with self.assertRaises(RuntimeError):
                    j.record_primitive("a", self.primitive_value)
            j.start("b")
        path = self.root / "primitive_returned.jsonl"
        before = path.read_bytes()
        with self.assertRaises(io.AmbiguousRequest):
            with self.journal():
                pass
        self.assertEqual(path.read_bytes(), before)

    def test_no_primitive_without_start_and_no_final_without_primitive(self):
        with self.journal() as j:
            with self.assertRaisesRegex(io.ContractError, "not started"):
                j.record_primitive("a", {})
            j.start("a")
            with self.assertRaisesRegex(io.ContractError, "durable primitive"):
                j.record_final("a", {})
            j.record_primitive("a", {})

    def test_locked_writer_is_exclusive_and_released_on_exit(self):
        with self.journal():
            with self.assertRaisesRegex(io.ContractError, "active writer"):
                with self.journal():
                    pass
        with self.journal() as j:
            self.assertEqual(len(j.pending()["never_started"]), 3)

    def test_closed_context_cannot_start_or_read(self):
        with self.journal() as j:
            pass
        with self.assertRaisesRegex(io.ContractError, "live context"):
            j.start("a")

    def test_fsync_failure_poisoned_context_cannot_continue(self):
        with self.journal() as j:
            with mock.patch.object(io.os, "fsync", side_effect=OSError("authored disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    j.start("a")
            with self.assertRaisesRegex(io.ContractError, "Prior persistence failure"):
                j.start("b")
        with self.assertRaises(io.AmbiguousRequest):
            with self.journal():
                pass

    def test_each_start_primitive_and_final_flushes_before_return(self):
        with self.journal() as j:
            with mock.patch.object(io.os, "fsync", wraps=os.fsync) as sync:
                j.start("a")
                self.assertGreaterEqual(sync.call_count, 1)
                sync.reset_mock()
                j.record_primitive("a", self.primitive_value)
                self.assertGreaterEqual(sync.call_count, 3)  # file + directory + journal
                sync.reset_mock()
                j.record_final("a", {})
                self.assertGreaterEqual(sync.call_count, 3)

    def test_symlink_and_hardlink_payloads_are_refused(self):
        self.populated()
        path = self.payload()
        saved = self.root.parent / "outside.json"
        path.rename(saved)
        path.symlink_to(saved)
        with self.assertRaises((io.ContractError, OSError)):
            with self.journal():
                pass
        path.unlink()
        os.link(saved, path)
        with self.assertRaisesRegex(io.ContractError, "link count"):
            with self.journal():
                pass

    def test_input_size_and_nonfinite_result_are_rejected_before_payload_write(self):
        with self.journal(max_payload_bytes=512) as j:
            j.start("a")
            for value in ({"huge": "x" * 2048}, {"bad": float("nan")}):
                with self.subTest(value=str(value)[:30]), self.assertRaises(io.ContractError):
                    j.record_primitive("a", value)
            self.assertFalse(self.payload().exists())

    def test_unknown_request_id_and_invalid_hashes_are_rejected(self):
        with self.journal() as j:
            with self.assertRaisesRegex(io.ContractError, "Unknown request"):
                j.start("unknown")
        with self.assertRaises(io.ContractError):
            self.journal(input_sha256="not-a-digest")

    def test_optional_parser_error_can_be_finalized_without_overwriting_primitive(self):
        with self.journal() as j:
            j.start("a")
            j.record_primitive("a", self.primitive_value)
            try:
                raise MemoryError("Parser stack overflowed - Python source too complex to parse")
            except MemoryError as error:
                j.record_final("a", {"structural_parser_failure": str(error), "eligible": False})
            self.assertTrue(j.primitive("a")["gt"])
            self.assertFalse(j.final("a")["eligible"])

    def test_abrupt_process_exit_at_three_boundaries_preserves_state(self):
        # Authored helper only; no generated program, model, or evaluator runs.
        helper = '''import json, os, sys
from infra.gpu03.task_specific_discovery.io_contract import RequestJournal
root, kwargs, boundary = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
with RequestJournal(root, **kwargs) as j:
    j.start("a")
    if boundary == "started": os._exit(23)
    j.record_primitive("a", {"authored": True})
    if boundary == "primitive": os._exit(23)
    j.record_final("a", {"derived": True})
    os._exit(23)
'''
        for boundary in ("started", "primitive", "final"):
            with self.subTest(boundary=boundary):
                directory = self.root.parent / boundary
                result = subprocess.run([sys.executable, "-B", "-c", helper,
                                         str(directory), json.dumps(self.kwargs), boundary],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 23, result.stderr)
                if boundary == "started":
                    with self.assertRaises(io.AmbiguousRequest):
                        with io.RequestJournal(directory, **self.kwargs):
                            pass
                else:
                    with io.RequestJournal(directory, **self.kwargs) as j:
                        self.assertEqual(j.primitive("a"), {"authored": True})
                        self.assertEqual(j.pending()["primitive_ready" if boundary == "primitive" else "final_recorded"], ["a"])

    def test_non_json_coercion_and_excessive_depth_are_refused(self):
        too_deep = None
        for _ in range(130):
            too_deep = [too_deep]
        for value in ({1: "coerced key"}, ("coerced tuple",), too_deep):
            with self.subTest(kind=type(value).__name__), self.assertRaises(io.ContractError):
                io.hash_value(value)


if __name__ == "__main__":
    unittest.main()
