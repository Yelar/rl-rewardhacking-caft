"""Explicit uv-managed Python mount for the existing two-layer CPU sandbox.

The old modules stay unchanged. Only the one reviewed runtime subtree is made
visible after /home masking. Transport limits, runner source and result parsing
are delegated unchanged; this module never executes source outside Bubblewrap.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import threading

from . import sandbox as outer
from . import bounded_evaluator as bounded

MANAGED_RUNTIME = "/home/ubuntu/.local/share/uv/python/cpython-3.12.3-linux-x86_64-gnu"
RUNTIME_ALIAS = "/runtime-base"
OUTER_SHA = "ab825d684f208dc0c9c963efa82f83a6cacbc6da75bd58edf0d9fe3aa9cd3af5"
BOUNDED_SHA = "5fc1e7657bbfef30ef065644f74b41f0a30963ddf23f824b3fda2971b2733dcf"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_sources():
    require(hashlib.sha256(Path(outer.__file__).read_bytes()).hexdigest() == OUTER_SHA and
            hashlib.sha256(Path(bounded.__file__).read_bytes()).hexdigest() == BOUNDED_SHA,
            "Frozen sandbox/transport source changed")


def runtime_directories():
    path = Path(MANAGED_RUNTIME)
    return [str(p) for p in reversed(path.parents) if str(p) not in ("/", "/home")]


def runtime_mounts(source):
    args = []
    for path in runtime_directories():
        args += ["--dir", path]
    return args + ["--ro-bind", source, MANAGED_RUNTIME, "--remount-ro", "/home"]


def validate_runtime(directory, *, venv=None):
    path = Path(directory)
    require(str(path) == MANAGED_RUNTIME and path.is_dir() and path.resolve() == path,
            "Only the exact canonical managed Python runtime may be mounted")
    python = path / "bin/python3.12"
    require(python.is_file() and python.resolve().is_relative_to(path), "Managed Python runtime is incomplete")
    if venv is not None:
        target = (Path(venv) / "bin/python").resolve()
        require(target == python.resolve(), "Virtualenv executable is not the qualified managed Python")
    for child in path.rglob("*"):
        require(child.name not in outer.SECRET_BASENAMES and child.name not in (".aws", ".ssh"),
                "Credential filename in managed runtime")
        require(not child.is_symlink() or child.resolve().is_relative_to(path), "Escaping managed-runtime symlink")
    return path


def build_outer_command(*, managed_runtime=MANAGED_RUNTIME, **kwargs):
    check_sources()
    runtime = validate_runtime(managed_runtime, venv=kwargs["venv_dir"])
    args = outer.build_outer_command(**kwargs)
    marker = ["--dir", "/home", "--dir", "/root"]
    positions = [i for i in range(len(args)) if args[i:i + len(marker)] == marker]
    require(len(positions) == 1, "Frozen outer home-mask layout differs")
    at = positions[0]
    # Both sources below are explicit host paths; /runtime-base is only a
    # destination here. The nested sandbox later reads this restricted alias.
    extra = ["--ro-bind", str(runtime), RUNTIME_ALIAS] + runtime_mounts(str(runtime))
    # Unlike --dir, tmpfs creates a mount which Bubblewrap can remount read-only.
    # Keep the original empty-home isolation while supplying that mount boundary.
    return args[:at] + ['--tmpfs', '/home'] + extra + args[at + 2:]


def build_bwrap_command(helpers, *, timeout, memory_limit, output_limit_bytes=bounded.MAX_OUTPUT_BYTES):
    check_sources()
    alias = Path(RUNTIME_ALIAS)
    require(alias.is_dir() and (alias / "bin/python3.12").is_file() and
            Path(MANAGED_RUNTIME).is_dir() and Path("/work/src/evaluate/helpers.py").is_file(),
            "Nested managed-runtime bind is available only inside the outer allowlist sandbox")
    args = bounded.build_bwrap_command(helpers, timeout=timeout, memory_limit=memory_limit,
                                       output_limit_bytes=output_limit_bytes)
    marker = ["--tmpfs", "/home", "--remount-ro", "/home"]
    positions = [i for i in range(len(args)) if args[i:i + len(marker)] == marker]
    require(len(positions) == 1, "Frozen nested home-mask layout differs")
    at = positions[0]
    # Delay the original readonly remount until the exact runtime mountpoint is
    # constructed. All other /home contents remain hidden by the same tmpfs.
    return args[:at] + ["--tmpfs", "/home"] + runtime_mounts(RUNTIME_ALIAS) + args[at + len(marker):]


def run_outer(*, wall_timeout, **kwargs):
    args = build_outer_command(**kwargs)
    result = bounded.bounded_transport(args, "", wall_timeout=wall_timeout)
    return {"argv": args, "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "elapsed_seconds": result.elapsed_seconds, "process_group": result.process_group,
            "output_bytes": result.output_bytes}


class EvaluatorInstallation(bounded.EvaluatorInstallation):
    """Original bounded transport with only its argv builder explicitly varied."""
    def __init__(self, helpers, output_limit_bytes):
        self.helpers = helpers
        self.output_limit_bytes = output_limit_bytes
        self.original = helpers._execute_in_subprocess
        self._lock = threading.Lock()
        self.counts = {"calls": 0, "timeout": 0, "output_overflow": 0, "transport_error": 0}

        def execute(code, timeout, memory_limit, raise_exceptions=False):
            args = build_bwrap_command(helpers, timeout=timeout, memory_limit=memory_limit,
                                      output_limit_bytes=output_limit_bytes)
            effective_limit = min(output_limit_bytes, int(os.environ.get("CODE_EVAL_OUTPUT_LIMIT_BYTES", str(bounded.MAX_OUTPUT_BYTES))))
            with self._lock:
                self.counts["calls"] += 1
            try:
                result = bounded.bounded_transport(args, code, wall_timeout=max(timeout, 1) + 1,
                                                   output_limit_bytes=effective_limit)
                return bounded.parse_result(helpers, result, raise_exceptions=raise_exceptions)
            except subprocess.TimeoutExpired:
                with self._lock:
                    self.counts["timeout"] += 1
                return helpers.CodeRunResult(success=False, timeout=True)
            except bounded.OutputLimitExceeded:
                with self._lock:
                    self.counts["output_overflow"] += 1
                return helpers.CodeRunResult(success=False, stdout={"raw": "Evaluator stdout/stderr output limit exceeded"})
            except (OSError, bounded.SandboxTransportError) as error:
                with self._lock:
                    self.counts["transport_error"] += 1
                if raise_exceptions:
                    raise
                return helpers.CodeRunResult(success=False, compiled=False,
                                             stdout={"raw": f"Evaluator transport failure: {type(error).__name__}"})

        self.execute = execute
        helpers._execute_in_subprocess = execute


def install_bounded_evaluator(helpers_module=None, *, output_limit_bytes=bounded.MAX_OUTPUT_BYTES):
    check_sources()
    bounded._positive_limit(output_limit_bytes, "output_limit_bytes", maximum=bounded.MAX_OUTPUT_BYTES)
    if helpers_module is None:
        from src.evaluate import helpers as helpers_module
    require(os.environ.get("CODE_EVAL_SANDBOX", "").strip().lower() == "bwrap" and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "H100 evaluator requires the CPU outer sandbox")
    return EvaluatorInstallation(helpers_module, output_limit_bytes)
