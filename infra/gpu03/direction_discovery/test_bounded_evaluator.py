"""Unit tests use fake child processes only; live fixtures require nested bwrap."""

from dataclasses import asdict, dataclass, field
import json
import os
import selectors
import signal
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import bounded_evaluator as bounded


@dataclass
class Result:
    success: bool = True
    compiled: bool = True
    timeout: bool = False
    oom: bool = False
    stdout: dict = field(default_factory=dict)

    def model_dump(self):
        return asdict(self)


class Pipe:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.closed = False

    def fileno(self):
        return self.descriptor

    def close(self):
        self.closed = True


class FakeProcess:
    pid = 421_421

    def __init__(self, returncode=0):
        self.stdin, self.stdout, self.stderr = Pipe(11), Pipe(12), Pipe(13)
        self.returncode = returncode
        self.waits = []

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.returncode


class FakeSelector:
    def __init__(self, clock, idle=False):
        self.clock = clock
        self.idle = idle
        self.mapping = {}
        self.closed = False

    def register(self, pipe, events, data):
        self.mapping[pipe] = types.SimpleNamespace(fileobj=pipe, events=events, data=data)

    def unregister(self, pipe):
        del self.mapping[pipe]

    def get_map(self):
        return self.mapping

    def select(self, timeout):
        if self.idle:
            self.clock[0] += timeout
            return []
        self.clock[0] += 0.001
        return [(key, key.events) for key in list(self.mapping.values())]

    def close(self):
        self.closed = True


class BoundedTransportTests(unittest.TestCase):
    def setUp(self):
        self.helper = types.SimpleNamespace(
            _get_python_executable=lambda: sys.executable,
            _SUBPROCESS_CODE="AUTHORED_HARNESS_SENTINEL",
            _execute_in_subprocess=lambda *args, **kwargs: "original",
            CodeRunResult=Result,
        )
        self.valid = asdict(Result(stdout={"tests_passed": 2, "tests_evaluated": 2}))
        self.command = ["/usr/bin/bwrap", "--die-with-parent", "--unshare-all", "--clearenv", "--cap-drop", "ALL"]

    def run_transport(self, *, stdout=b"ok", stderr=b"", limit=1024, idle=False,
                      write_error=None, read_error=None, process=None):
        process = process or FakeProcess()
        clock = [5.0]
        selector = FakeSelector(clock, idle=idle)
        buffers = {12: bytearray(stdout), 13: bytearray(stderr)}
        read_sizes = []
        writes = []

        def read(fd, size):
            if read_error:
                raise read_error
            read_sizes.append(size)
            data = bytes(buffers[fd][:size])
            del buffers[fd][:size]
            return data

        def write(fd, data):
            if write_error:
                raise write_error
            writes.append((fd, bytes(data)))
            return min(3, len(data))  # Exercise partial stdin writes.

        def killpg(pid, sig):
            if sig == 0:
                raise ProcessLookupError

        with patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(bounded.subprocess, "Popen", return_value=process) as popen, \
             patch.object(bounded.selectors, "DefaultSelector", return_value=selector), \
             patch.object(bounded.os, "set_blocking"), patch.object(bounded.os, "read", side_effect=read), \
             patch.object(bounded.os, "write", side_effect=write), \
             patch.object(bounded.os, "killpg", side_effect=killpg) as kill, \
             patch.object(bounded.time, "monotonic", side_effect=lambda: clock[0]):
            try:
                result = bounded.bounded_transport(self.command, "authored code", wall_timeout=2, output_limit_bytes=limit)
            finally:
                self.process, self.selector, self.kill = process, selector, kill
                self.popen, self.read_sizes, self.writes = popen, read_sizes, writes
        return result

    def test_success_drains_both_streams_and_partially_writes_input(self):
        result = self.run_transport(stdout=b"out", stderr=b"diagnostic")
        self.assertEqual(result.stdout, b"out")
        self.assertEqual(result.stderr, b"diagnostic")
        self.assertEqual(result.output_bytes, 13)
        self.assertEqual(self.kill.call_count, 0)
        self.assertGreater(len(self.writes), 1)
        self.assertTrue(self.selector.closed)
        self.assertTrue(all(pipe.closed for pipe in (self.process.stdin, self.process.stdout, self.process.stderr)))
        kwargs = self.popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["close_fds"])
        self.assertFalse(kwargs["text"])
        self.assertEqual(set(kwargs["env"]), {"PATH", "LANG"})

    def test_combined_stream_overflow_kills_only_exact_group(self):
        with self.assertRaises(bounded.OutputLimitExceeded):
            self.run_transport(stdout=b"a" * 600, stderr=b"b" * 600)
        self.kill.assert_any_call(FakeProcess.pid, signal.SIGKILL)
        self.kill.assert_any_call(FakeProcess.pid, 0)
        self.assertTrue(self.selector.closed)
        self.assertLessEqual(max(self.read_sizes), 1025)

    def test_exact_limit_is_accepted(self):
        result = self.run_transport(stdout=b"a" * 500, stderr=b"b" * 524)
        self.assertEqual(result.output_bytes, 1024)
        self.assertEqual(self.kill.call_count, 0)

    def test_raw_stderr_flood_cannot_bypass_the_cap(self):
        with self.assertRaises(bounded.OutputLimitExceeded):
            self.run_transport(stderr=b"x" * 10000)
        self.assertEqual(sum(1 for c in self.kill.call_args_list if c.args[1] == signal.SIGKILL), 1)

    def test_timeout_covers_blocked_input_and_silent_child(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_transport(idle=True)
        self.kill.assert_any_call(FakeProcess.pid, signal.SIGKILL)
        self.assertTrue(self.selector.closed)

    def test_broken_stdin_pipe_is_drained_and_reaped(self):
        result = self.run_transport(stdout=b"error", write_error=BrokenPipeError())
        self.assertEqual(result.stdout, b"error")
        self.assertTrue(self.process.stdin.closed)

    def test_unexpected_read_error_also_kills_and_closes(self):
        with self.assertRaises(OSError):
            self.run_transport(read_error=OSError("authored error"))
        self.kill.assert_any_call(FakeProcess.pid, signal.SIGKILL)
        self.assertTrue(self.selector.closed)

    def test_arbitrary_unsandboxed_command_rejected_before_popen(self):
        with patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(bounded.subprocess, "Popen") as popen:
            with self.assertRaises(bounded.SandboxTransportError):
                bounded.bounded_transport([sys.executable, "-c", "pass"], "x", wall_timeout=1)
            popen.assert_not_called()

    def test_missing_sandbox_flag_rejected(self):
        with patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"):
            with self.assertRaises(bounded.SandboxTransportError):
                bounded.bounded_transport(["/usr/bin/bwrap", "--unshare-all"], "x", wall_timeout=1)

    def test_unbounded_limits_and_inputs_rejected(self):
        for timeout in (float("inf"), float("nan"), 0, -1, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                bounded.bounded_transport(self.command, "x", wall_timeout=timeout)
        for limit in (0, -1, True, bounded.MAX_OUTPUT_BYTES + 1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                bounded.bounded_transport(self.command, "x", wall_timeout=1, output_limit_bytes=limit)
        with self.assertRaisesRegex(ValueError, "8 MiB"):
            bounded.bounded_transport(self.command, "x" * (bounded.MAX_INPUT_BYTES + 1), wall_timeout=1)

    def test_process_group_remaining_after_reap_is_a_failure(self):
        with patch.object(bounded.os, "killpg"):
            with self.assertRaisesRegex(bounded.SandboxTransportError, "remains"):
                bounded._kill_exact_group(FakeProcess())

    def test_kill_permission_error_fails_closed(self):
        with patch.object(bounded.os, "killpg", side_effect=PermissionError):
            with self.assertRaisesRegex(bounded.SandboxTransportError, "Permission"):
                bounded._kill_exact_group(FakeProcess())

    def test_selector_failure_occurs_before_any_child_is_spawned(self):
        with patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(bounded.selectors, "DefaultSelector", side_effect=OSError("no descriptors")), \
             patch.object(bounded.subprocess, "Popen") as popen:
            with self.assertRaises(OSError):
                bounded.bounded_transport(self.command, "authored", wall_timeout=1)
        popen.assert_not_called()

    def test_popen_failure_closes_prepared_selector(self):
        selector = FakeSelector([0.0])
        with patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"), \
             patch.object(bounded.selectors, "DefaultSelector", return_value=selector), \
             patch.object(bounded.subprocess, "Popen", side_effect=OSError("cannot spawn")):
            with self.assertRaises(OSError):
                bounded.bounded_transport(self.command, "authored", wall_timeout=1)
        self.assertTrue(selector.closed)

    def test_parse_valid_result_retains_existing_semantics(self):
        transport = bounded.TransportResult(json.dumps(self.valid).encode(), b"", 0, 1, 0.1, 100)
        self.assertEqual(bounded.parse_result(self.helper, transport).model_dump(), self.valid)

    def test_nonzero_exit_cannot_keep_claimed_passed_tests(self):
        transport = bounded.TransportResult(json.dumps(self.valid).encode(), b"", 2, 1, 0.1, 100)
        result = bounded.parse_result(self.helper, transport)
        self.assertFalse(result.success)
        self.assertFalse(result.compiled)
        self.assertNotIn("tests_passed", result.stdout)

    def test_malformed_incomplete_coercible_or_contradictory_status_fails_closed(self):
        candidates = [b"garbage", b"\xff", b"{}", b"[]", b"null", b'{"success": true}',
                      json.dumps({**self.valid, "success": 1}).encode(),
                      json.dumps({**self.valid, "extra": "unexpected"}).encode(),
                      json.dumps({**self.valid, "timeout": True}).encode(),
                      json.dumps({**self.valid, "stdout": []}).encode()]
        for output in candidates:
            with self.subTest(output=output):
                result = bounded.parse_result(self.helper, bounded.TransportResult(output, b"", 0, 1, 0.1, len(output)))
                self.assertFalse(result.success)
                self.assertFalse(result.compiled)
                self.assertNotIn("tests_passed", result.stdout)

    def test_failure_result_cannot_keep_partial_or_forged_test_counts(self):
        output = json.dumps({**self.valid, "success": False}).encode()
        result = bounded.parse_result(self.helper, bounded.TransportResult(output, b"", 0, 1, 0.1, len(output)))
        self.assertFalse(result.success)
        self.assertTrue(result.compiled)
        self.assertNotIn("tests_passed", result.stdout)

    def test_duplicate_nonfinite_and_too_deep_json_fails_closed(self):
        valid_text = json.dumps(self.valid)
        candidates = [valid_text.replace('"success": true', '"success": false, "success": true'),
                      valid_text.replace('"tests_passed": 2', '"tests_passed": NaN'),
                      valid_text.replace('"tests_passed": 2', '"tests_passed": 1e9999'),
                      '[' * 2000 + '0' + ']' * 2000]
        for text in candidates:
            with self.subTest(text=text[:100]):
                result = bounded.parse_result(self.helper, bounded.TransportResult(text.encode(), b"", 0, 1, 0.1, len(text)))
                self.assertFalse(result.success)
                self.assertFalse(result.compiled)

    def test_bwrap_startup_failure_raises_when_required(self):
        result = bounded.TransportResult(b"", b"bwrap: namespace unavailable", 1, 1, 0.1, 28)
        with self.assertRaises(bounded.SandboxTransportError):
            bounded.parse_result(self.helper, result, raise_exceptions=True)

    def test_command_preserves_original_runner_and_requires_sandbox(self):
        with patch.dict(os.environ, {"CODE_EVAL_SANDBOX": "bwrap", "CODE_EVAL_PROCESS_LIMIT": "32",
                                     "CODE_EVAL_OUTPUT_LIMIT_BYTES": "1048576"}), \
             patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"):
            args = bounded.build_bwrap_command(self.helper, timeout=3, memory_limit=1024)
        self.assertIn(self.helper._SUBPROCESS_CODE, args)
        self.assertEqual(args[-4:], ["1024", "3", "32", "1048576"])
        self.assertIn("--unshare-all", args)
        self.assertIn("--ro-bind", args)
        self.assertIn("--clearenv", args)
        with patch.dict(os.environ, {"CODE_EVAL_SANDBOX": ""}):
            with self.assertRaises(bounded.SandboxTransportError):
                bounded.build_bwrap_command(self.helper, timeout=3, memory_limit=1024)

    def test_environment_cannot_expand_output_or_process_limits(self):
        with patch.object(bounded.shutil, "which", return_value="/usr/bin/bwrap"):
            for update in ({"CODE_EVAL_OUTPUT_LIMIT_BYTES": "1048577"}, {"CODE_EVAL_PROCESS_LIMIT": "33"},
                           {"CODE_EVAL_PROCESS_LIMIT": "abc"}):
                with self.subTest(update=update), patch.dict(os.environ, {"CODE_EVAL_SANDBOX": "bwrap", **update}):
                    with self.assertRaises(ValueError):
                        bounded.build_bwrap_command(self.helper, timeout=3, memory_limit=1024)

    def test_explicit_installation_and_restore_preserve_original_function(self):
        original = self.helper._execute_in_subprocess
        with patch.dict(os.environ, {"CODE_EVAL_SANDBOX": "bwrap"}):
            installed = bounded.install_bounded_evaluator(self.helper)
        self.assertIs(self.helper._execute_in_subprocess, installed.execute)
        self.assertEqual(installed.report()["calls"], 0)
        self.assertEqual(len(installed.report()["harness_sha256"]), 64)
        installed.restore()
        self.assertIs(self.helper._execute_in_subprocess, original)

    def test_installation_timeout_and_overflow_are_distinct_failed_results(self):
        with patch.dict(os.environ, {"CODE_EVAL_SANDBOX": "bwrap"}), \
             patch.object(bounded, "build_bwrap_command", return_value=self.command):
            installed = bounded.install_bounded_evaluator(self.helper)
            with patch.object(bounded, "bounded_transport", side_effect=subprocess.TimeoutExpired("bwrap", 2)):
                timeout_result = installed.execute("authored", 1, 1024)
            with patch.object(bounded, "bounded_transport", side_effect=bounded.OutputLimitExceeded):
                output_result = installed.execute("authored", 1, 1024)
            installed.restore()
        self.assertTrue(timeout_result.timeout)
        self.assertFalse(timeout_result.success)
        self.assertFalse(output_result.timeout)
        self.assertFalse(output_result.success)
        self.assertIn("output limit", output_result.stdout["raw"])
        self.assertEqual(installed.report()["calls"], 2)

    def test_restore_refuses_to_overwrite_another_patch(self):
        with patch.dict(os.environ, {"CODE_EVAL_SANDBOX": "bwrap"}):
            installed = bounded.install_bounded_evaluator(self.helper)
        self.helper._execute_in_subprocess = lambda: None
        with self.assertRaises(RuntimeError):
            installed.restore()

    def test_qualification_requires_known_outside_paths(self):
        with self.assertRaises(ValueError):
            bounded.qualify_bounded_evaluator(forbidden_paths=[], helpers_module=self.helper)
        with self.assertRaises(ValueError):
            bounded.qualify_bounded_evaluator(forbidden_paths=["relative"], helpers_module=self.helper)


if __name__ == "__main__":
    unittest.main()
