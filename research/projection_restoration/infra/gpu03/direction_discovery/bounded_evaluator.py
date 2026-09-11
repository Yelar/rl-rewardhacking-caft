"""Bound the existing evaluator transport, including direct file-descriptor writes.

Only the subprocess transport is replaced, at explicit installation. The existing
helpers._SUBPROCESS_CODE harness, CodeRunResult, parser, tests, and classifier are
reused. The experiment must run this inside its reviewed outer allowlist sandbox;
the mandatory nested bubblewrap binds that already restricted root read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import selectors
import shutil
import signal
import subprocess
import threading
import time
from typing import Any


MAX_OUTPUT_BYTES = 1_048_576
MAX_INPUT_BYTES = 8_388_608
_RESULT_FIELDS = {"success", "compiled", "timeout", "oom", "stdout"}


class OutputLimitExceeded(RuntimeError):
    pass


class SandboxTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransportResult:
    stdout: bytes
    stderr: bytes
    returncode: int
    process_group: int
    elapsed_seconds: float
    output_bytes: int


def _positive_limit(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the bound of {maximum}")
    return value


def build_bwrap_command(helpers, *, timeout: float, memory_limit: int,
                        output_limit_bytes: int = MAX_OUTPUT_BYTES) -> list[str]:
    """Use the repository's nested sandbox policy with no unsandboxed fallback."""
    if os.environ.get("CODE_EVAL_SANDBOX", "").strip().lower() != "bwrap":
        raise SandboxTransportError("This experiment requires CODE_EVAL_SANDBOX=bwrap")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxTransportError("Required bubblewrap executable is unavailable")
    _positive_limit(memory_limit, "memory_limit")
    _positive_limit(output_limit_bytes, "output_limit_bytes", maximum=MAX_OUTPUT_BYTES)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    try:
        process_limit = int(os.environ.get("CODE_EVAL_PROCESS_LIMIT", "32"))
        requested_output = int(os.environ.get("CODE_EVAL_OUTPUT_LIMIT_BYTES", str(MAX_OUTPUT_BYTES)))
    except ValueError as error:
        raise ValueError("Evaluator limit environment contains a malformed integer") from error
    _positive_limit(process_limit, "CODE_EVAL_PROCESS_LIMIT", maximum=32)
    _positive_limit(requested_output, "CODE_EVAL_OUTPUT_LIMIT_BYTES", maximum=MAX_OUTPUT_BYTES)
    output_limit_bytes = min(requested_output, output_limit_bytes)
    python = helpers._get_python_executable()
    if not os.path.isabs(python) or not os.path.isfile(python) or not os.access(python, os.X_OK):
        raise SandboxTransportError("Resolved evaluator Python is not an absolute executable file")
    args = [
        bwrap, "--die-with-parent", "--new-session", "--unshare-all",
        "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--tmpfs", "/tmp", "--tmpfs", "/root", "--remount-ro", "/root",
        "--tmpfs", "/home", "--remount-ro", "/home", "--cap-drop", "ALL",
    ]
    if os.path.isdir("/durable-checkpoints"):
        args += ["--tmpfs", "/durable-checkpoints", "--remount-ro", "/durable-checkpoints"]
    args += [
        "--chdir", "/tmp", "--clearenv", "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONNOUSERSITE", "1",
        python, "-c", helpers._SUBPROCESS_CODE, str(memory_limit), str(max(timeout, 1)),
        str(process_limit), str(output_limit_bytes),
    ]
    return args


def _kill_exact_group(process: subprocess.Popen) -> None:
    """Kill only this Popen's newly created group, then reap its leader.

    The caller has not polled/reaped the leader before failure, so its PID cannot
    be reused while this signal is issued. Bubblewrap's private PID namespace and
    die-with-parent policy also remove descendants that create a new session.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        raise SandboxTransportError("Permission denied killing the evaluator process group") from error
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired as error:
        raise SandboxTransportError("Evaluator process group did not reap within cleanup grace") from error
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    raise SandboxTransportError("Evaluator process group remains after the sandbox leader was reaped")


def bounded_transport(args: list[str], code: str, *, wall_timeout: float,
                      output_limit_bytes: int = MAX_OUTPUT_BYTES) -> TransportResult:
    """Drain binary pipes with one combined byte cap, including os.write output.

    wall_timeout covers child startup, stdin writes, stdout/stderr drain, and exit.
    At most two additional seconds are allowed to reap a killed sandbox on failure.
    This function never runs an arbitrary bare command: argv[0] must be the resolved
    bubblewrap executable and the isolation flags must be present.
    """
    _positive_limit(output_limit_bytes, "output_limit_bytes", maximum=MAX_OUTPUT_BYTES)
    if not isinstance(code, str):
        raise TypeError("Evaluator source must be text")
    encoded = code.encode("utf-8")
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("Evaluator source exceeds the 8 MiB input transport bound")
    if not isinstance(wall_timeout, (int, float)) or isinstance(wall_timeout, bool) or not math.isfinite(wall_timeout) or wall_timeout <= 0:
        raise ValueError("wall_timeout must be finite and positive")
    resolved_bwrap = shutil.which("bwrap")
    if not args or not resolved_bwrap or args[0] != resolved_bwrap:
        raise SandboxTransportError("Transport accepts only the resolved bubblewrap entrypoint")
    for required in ("--die-with-parent", "--unshare-all", "--clearenv", "--cap-drop"):
        if required not in args:
            raise SandboxTransportError(f"Sandbox command lacks required isolation flag {required}")

    started = time.monotonic()
    deadline = started + wall_timeout
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False, close_fds=True, start_new_session=True,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
    except BaseException:
        selector.close()
        raise
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    output_bytes = 0
    offset = 0
    reaped = False
    try:
        for pipe in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(pipe.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        if encoded:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, wall_timeout)
            for key, _events in selector.select(min(remaining, 0.25)):
                pipe = key.fileobj
                if key.data == "stdin":
                    try:
                        written = os.write(pipe.fileno(), encoded[offset:offset + 65_536])
                    except BrokenPipeError:
                        selector.unregister(pipe)
                        pipe.close()
                        continue
                    except BlockingIOError:
                        continue
                    offset += written
                    if offset == len(encoded):
                        selector.unregister(pipe)
                        pipe.close()
                else:
                    try:
                        # Read one extra byte at the boundary to detect overflow;
                        # never allocate/retain an unbounded flood or line.
                        chunk = os.read(pipe.fileno(), min(65_536, output_limit_bytes - output_bytes + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(pipe)
                        pipe.close()
                        continue
                    output_bytes += len(chunk)
                    if output_bytes > output_limit_bytes:
                        raise OutputLimitExceeded("Combined evaluator stdout/stderr exceeded the byte limit")
                    buffers[key.data].extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(args, wall_timeout)
        returncode = process.wait(timeout=remaining)
        reaped = True
        return TransportResult(bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode,
                               process.pid, time.monotonic() - started, output_bytes)
    finally:
        try:
            if not reaped:
                _kill_exact_group(process)
        finally:
            selector.close()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()


def parse_result(helpers, result: TransportResult, *, raise_exceptions: bool = False):
    """Parse exactly the existing result schema without permissive default status."""
    if result.returncode != 0:
        if raise_exceptions and b"bwrap:" in result.stderr:
            raise SandboxTransportError("bubblewrap failed to execute the evaluator: " +
                                        result.stderr[:2048].decode("utf-8", errors="replace"))
        # Do not retain claimed test counts from an unsuccessful process: the
        # existing caller otherwise reads tests_passed despite success=False.
        return helpers.CodeRunResult(success=False, compiled=False,
                                     stdout={"raw": f"Evaluator subprocess exited with status {result.returncode}"})
    def reject_nonstandard_constant(_value):
        raise ValueError("Nonfinite JSON numbers are not valid evaluator results")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON numbers are not valid evaluator results")
        return parsed

    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate JSON keys are not valid evaluator results")
            output[key] = value
        return output

    try:
        decoded = json.loads(result.stdout.decode("utf-8", errors="strict"),
                             parse_constant=reject_nonstandard_constant, parse_float=finite_float,
                             object_pairs_hook=unique_object)
        if not isinstance(decoded, dict) or set(decoded) != _RESULT_FIELDS:
            raise ValueError("Evaluator result fields differ from the required schema")
        if any(type(decoded[name]) is not bool for name in ("success", "compiled", "timeout", "oom")):
            raise ValueError("Evaluator status fields must be exact booleans")
        if not isinstance(decoded["stdout"], dict):
            raise ValueError("Evaluator stdout field must be an object")
        if decoded["success"] and (not decoded["compiled"] or decoded["timeout"] or decoded["oom"]):
            raise ValueError("Evaluator status fields are contradictory")
        if not decoded["success"]:
            # Preserve ordinary runner exception diagnostics but discard forged
            # or partial claimed test scores whenever the harness reports failure.
            decoded["stdout"] = {"raw": str(decoded["stdout"].get("raw", "Evaluator reported failure"))[:2048]}
        return helpers.CodeRunResult(**decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return helpers.CodeRunResult(success=False, compiled=False,
                                     stdout={"raw": "Evaluator emitted malformed or incomplete result JSON"})


class EvaluatorInstallation:
    def __init__(self, helpers, output_limit_bytes: int):
        self.helpers = helpers
        self.output_limit_bytes = output_limit_bytes
        self.original = helpers._execute_in_subprocess
        self._lock = threading.Lock()
        self.counts = {"calls": 0, "timeout": 0, "output_overflow": 0, "transport_error": 0}

        def execute(code, timeout, memory_limit, raise_exceptions=False):
            args = build_bwrap_command(helpers, timeout=timeout, memory_limit=memory_limit,
                                       output_limit_bytes=output_limit_bytes)
            effective_limit = min(output_limit_bytes, int(os.environ.get("CODE_EVAL_OUTPUT_LIMIT_BYTES", str(MAX_OUTPUT_BYTES))))
            with self._lock:
                self.counts["calls"] += 1
            try:
                result = bounded_transport(args, code, wall_timeout=max(timeout, 1) + 1,
                                           output_limit_bytes=effective_limit)
                return parse_result(helpers, result, raise_exceptions=raise_exceptions)
            except subprocess.TimeoutExpired:
                with self._lock:
                    self.counts["timeout"] += 1
                return helpers.CodeRunResult(success=False, timeout=True)
            except OutputLimitExceeded:
                with self._lock:
                    self.counts["output_overflow"] += 1
                return helpers.CodeRunResult(success=False, stdout={"raw": "Evaluator stdout/stderr output limit exceeded"})
            except (OSError, SandboxTransportError) as error:
                with self._lock:
                    self.counts["transport_error"] += 1
                if raise_exceptions:
                    raise
                return helpers.CodeRunResult(success=False, compiled=False,
                                             stdout={"raw": f"Evaluator transport failure: {type(error).__name__}"})

        self.execute = execute
        helpers._execute_in_subprocess = execute

    def report(self) -> dict:
        with self._lock:
            return {"output_limit_bytes": self.output_limit_bytes, "input_limit_bytes": MAX_INPUT_BYTES,
                    "execution_wall_limit": "max(timeout, 1) + 1 seconds", "cleanup_grace_seconds": 2,
                    "harness_sha256": hashlib.sha256(self.helpers._SUBPROCESS_CODE.encode()).hexdigest(),
                    **self.counts}

    def restore(self) -> None:
        if self.helpers._execute_in_subprocess is not self.execute:
            raise RuntimeError("Evaluator transport changed after installation; refusing to overwrite it")
        self.helpers._execute_in_subprocess = self.original


def install_bounded_evaluator(helpers_module=None, *, output_limit_bytes: int = MAX_OUTPUT_BYTES) -> EvaluatorInstallation:
    """Explicitly replace only the evaluator transport in this experiment process."""
    _positive_limit(output_limit_bytes, "output_limit_bytes", maximum=MAX_OUTPUT_BYTES)
    if helpers_module is None:
        from src.evaluate import helpers as helpers_module
    if os.environ.get("CODE_EVAL_SANDBOX", "").strip().lower() != "bwrap":
        raise SandboxTransportError("Set CODE_EVAL_SANDBOX=bwrap before installing the transport")
    return EvaluatorInstallation(helpers_module, output_limit_bytes)


def qualify_bounded_evaluator(*, forbidden_paths: list[str], helpers_module=None) -> dict:
    """Run authored qualification fixtures exclusively through mandatory bwrap.

    Run this function from inside the reviewed outer allowlist sandbox. Pass paths
    to actual outside private/secret files chosen by the caller; only path access
    booleans are returned, never their contents. No arbitrary generated source is
    accepted by this qualification entrypoint.
    """
    if not forbidden_paths or any(not isinstance(path, str) or not os.path.isabs(path) for path in forbidden_paths):
        raise ValueError("Qualification requires absolute forbidden outside paths")
    installation = install_bounded_evaluator(helpers_module)
    execute = installation.helpers._execute_in_subprocess
    results = {}
    fixtures = {
        "benign": 'import json\nprint(json.dumps({"tests_evaluated": 2, "tests_passed": 2, "test_errors": []}))',
        "syntax_error": 'def broken(:\n pass',
        "raw_stdout_flood": 'import os\nwhile True: os.write(1, b"x" * 65536)',
        "raw_stderr_flood": 'import os\nwhile True: os.write(2, b"x" * 65536)',
        "timeout_fork_cleanup": 'import os, signal, time\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\n'
                                'try:\n child = os.fork()\nexcept BlockingIOError:\n child = -1\n'
                                'if child == 0:\n os.setsid()\n while True: time.sleep(1)\n'
                                'else:\n while True: time.sleep(1)',
        "private_paths": 'import json, os\npaths = ' + repr(forbidden_paths) + '\n'
                         'print(json.dumps({"can_read": [os.access(p, os.R_OK) for p in paths]}))',
        "limits": 'import json, resource, os\nprint(json.dumps({'
                  '"as_limit": resource.getrlimit(resource.RLIMIT_AS)[1], '
                  '"nproc_limit": resource.getrlimit(resource.RLIMIT_NPROC)[1], '
                  '"credential_env": any(any(x in k for x in ("TOKEN", "SECRET", "KEY", "SSH_AUTH")) for k in os.environ)}))',
    }
    try:
        for name, code in fixtures.items():
            started = time.monotonic()
            result = execute(code, timeout=1, memory_limit=256, raise_exceptions=True)
            elapsed = time.monotonic() - started
            results[name] = {**result.model_dump(), "elapsed_seconds": elapsed}
            if elapsed > 4.5:
                raise SandboxTransportError(f"Qualification {name} exceeded execution plus cleanup bound")
        if not results["benign"]["success"] or results["benign"]["stdout"].get("tests_passed") != 2:
            raise SandboxTransportError("Benign sandbox execution did not return the expected test counts")
        if results["syntax_error"]["success"] or results["syntax_error"]["compiled"]:
            raise SandboxTransportError("Syntax-error fixture did not fail closed")
        for name in ("raw_stdout_flood", "raw_stderr_flood"):
            if results[name]["success"] or "output limit" not in results[name]["stdout"].get("raw", ""):
                raise SandboxTransportError(f"Raw descriptor flood was not bounded: {name}")
        if results["timeout_fork_cleanup"]["success"] or not results["timeout_fork_cleanup"]["timeout"]:
            raise SandboxTransportError("Forking timeout fixture was not terminated")
        if results["private_paths"]["stdout"].get("can_read") != [False] * len(forbidden_paths):
            raise SandboxTransportError("An outside private path is readable inside the nested sandbox")
        limits = results["limits"]["stdout"]
        if limits.get("as_limit") != 256 * 1024 * 1024 or not 0 < limits.get("nproc_limit", 0) <= 32 or limits.get("credential_env") is not False:
            raise SandboxTransportError("Child resource or credential limits did not qualify")
        return {"status": "passed", "fixtures": results, "transport": installation.report()}
    finally:
        installation.restore()
