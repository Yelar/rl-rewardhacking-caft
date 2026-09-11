"""Outer allowlist filesystem for this experiment's CPU evaluation coordinator.

The repository's mandatory nested Bubblewrap evaluator remounts this restricted
root read-only. Never pass model-generated code as a coordinator command. Source
must be a reviewed snapshot containing only code and nonsecret provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time


FORBIDDEN_ROOTS = {"/", "/usr", "/etc", "/home", "/root", "/l", "/scratch", "/tmp"}
SECRET_BASENAMES = {".env", ".netrc", ".git-credentials", "id_rsa", "id_ed25519", "credentials"}


def _directory(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or str(path) in FORBIDDEN_ROOTS:
        raise ValueError(f"{name} must be a specific absolute directory")
    if path.resolve() != path or not path.is_dir():
        raise ValueError(f"{name} must exist and have a canonical nonsymlink path")
    return path


def _validate_source(source: Path) -> None:
    if not (source / "src/evaluate/helpers.py").is_file():
        raise ValueError("Source snapshot lacks the repository evaluator")
    if (source / ".git").exists():
        raise ValueError("Mount an immutable source snapshot, not a Git worktree")
    for path in source.rglob("*"):
        if path.name in SECRET_BASENAMES:
            raise ValueError("Source snapshot contains a forbidden credential filename")
        if path.is_symlink() and not path.resolve().is_relative_to(source):
            raise ValueError("Source snapshot contains an escaping symlink")


def build_outer_command(*, source_dir: str | Path, input_dir: str | Path,
                        output_dir: str | Path, venv_dir: str | Path,
                        python_args: list[str], workers: int = 1) -> list[str]:
    """Return reviewed argv; paths inside are /work, /input, /output, /venv.

    Only /output is host-backed writable in the coordinator. The nested evaluator
    must be installed with bounded_evaluator.install_bounded_evaluator(), which
    read-only binds this entire root before executing a generated program.
    """
    source = _directory(source_dir, "source_dir")
    inputs = _directory(input_dir, "input_dir")
    output = _directory(output_dir, "output_dir")
    venv = _directory(venv_dir, "venv_dir")
    _validate_source(source)
    if not (venv / "pyvenv.cfg").is_file() or not (venv / "bin/python").is_file():
        raise ValueError("Pinned virtual environment is incomplete")
    if output.stat().st_uid != os.getuid() or stat.S_IMODE(output.stat().st_mode) & 0o022:
        raise ValueError("Output must be owned by the caller and not group/world writable")
    paths = (source, inputs, output, venv)
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Source, inputs, output and virtualenv must be disjoint directories")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Evaluation workers must be between one and eight")
    if not isinstance(python_args, list) or not python_args or any(not isinstance(x, str) or "\0" in x for x in python_args):
        raise ValueError("python_args must be a nonempty string argv without NUL bytes")
    bwrap = shutil.which("bwrap")
    if not bwrap or not Path(bwrap).is_absolute():
        raise RuntimeError("An absolute Bubblewrap executable is required")
    if not Path("/usr").is_dir():
        raise RuntimeError("The reviewed Linux runtime /usr is unavailable")

    args = [bwrap, "--die-with-parent", "--new-session", "--unshare-all",
            "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib64"):
        path = Path("/") / name
        if not path.exists():
            continue
        if path.is_symlink():
            resolved = path.resolve()
            if not resolved.is_relative_to("/usr"):
                raise RuntimeError(f"Unexpected runtime alias {path}")
            args += ["--symlink", str(resolved), str(path)]
        else:
            args += ["--ro-bind", str(path), str(path)]
    args += ["--dir", "/etc"]
    # No broad /etc mount: in particular no account/shadow/credential directories.
    for name in ("ld.so.cache", "localtime"):
        path = Path("/etc") / name
        if path.is_file():
            args += ["--ro-bind", str(path.resolve()), str(path)]
    args += ["--ro-bind", str(venv), "/venv", "--ro-bind", str(source), "/work",
             "--ro-bind", str(inputs), "/input", "--bind", str(output), "/output",
             "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--dir", "/home", "--dir", "/root", "--chdir", "/work"]
    env = {
        "HOME": "/tmp", "PATH": "/venv/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
        "PYTHONPATH": "/work", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "", "CODE_EVAL_SANDBOX": "bwrap",
        "CODE_EVAL_SANDBOX_REQUIRED": "1", "CODE_EVAL_PROCESS_LIMIT": "32",
        "CODE_EVAL_OUTPUT_LIMIT_BYTES": "1048576", "MAX_JOBS": str(workers),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/tmp/huggingface",
        "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
    }
    for key, value in env.items():
        args += ["--setenv", key, value]
    return args + ["/venv/bin/python", *python_args]


def run_outer(*, wall_timeout: float, **kwargs) -> dict:
    """Run a bounded CPU coordinator with capped combined stdout/stderr.

    Long scientific jobs should additionally use the parent's independent service
    deadline/cgroup resource limits. This helper itself reserves no resources.
    """
    from .bounded_evaluator import bounded_transport
    args = build_outer_command(**kwargs)
    result = bounded_transport(args, "", wall_timeout=wall_timeout)
    return {"argv": args, "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "elapsed_seconds": result.elapsed_seconds,
            "process_group": result.process_group, "output_bytes": result.output_bytes}


def _visible_pids() -> set[int]:
    return {int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()}


def qualify_inside(forbidden_paths: list[str]) -> dict:
    """Authored fixtures only; called exclusively inside the outer sandbox."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("CODE_EVAL_SANDBOX") != "bwrap":
        raise RuntimeError("Qualification requires the CPU outer sandbox environment")
    from .bounded_evaluator import qualify_bounded_evaluator, install_bounded_evaluator
    from src.evaluate.evaluation import RewardHackingEvaluation
    from src.evaluate.evaluator import CodeEvaluator
    import torch
    assert not torch.cuda.is_available()
    assert all(not Path(p).exists() for p in forbidden_paths)
    assert not Path("/l").exists() and not Path("/scratch").exists()
    before_pids = _visible_pids()
    bounded = qualify_bounded_evaluator(forbidden_paths=forbidden_paths)
    output_probe = Path("/output/outer_write_probe.txt")
    with output_probe.open("x") as stream:
        stream.write("outer coordinator may write; inner evaluator must not\n")
    expected = hashlib.sha256(output_probe.read_bytes()).hexdigest()
    install = install_bounded_evaluator()
    try:
        execute = install.helpers._execute_in_subprocess
        # The fixture attempts actual open/write operations, not os.access alone.
        paths = ["/output/outer_write_probe.txt", "/input/input_probe.txt", "/work/src/evaluate/helpers.py"]
        source = ('import json, os, socket\nresults = {}\n'
                  'for path in ' + repr(paths) + ':\n'
                  ' try:\n  with open(path, "a") as stream: stream.write("forbidden write")\n'
                  '  results[path] = "writable"\n'
                  ' except OSError as error: results[path] = type(error).__name__\n'
                  'results["outside_readable"] = []\n'
                  'for path in ' + repr(forbidden_paths) + ':\n'
                  ' try:\n  with open(path, "rb") as stream: stream.read(1)\n'
                  '  results["outside_readable"].append(True)\n'
                  ' except OSError: results["outside_readable"].append(False)\n'
                  'print(json.dumps(results))')
        readonly = execute(source, timeout=2, memory_limit=256, raise_exceptions=True)
        assert readonly.success
        assert all(readonly.stdout.get(path) != "writable" for path in paths)
        assert readonly.stdout["outside_readable"] == [False] * len(forbidden_paths)
        assert hashlib.sha256(output_probe.read_bytes()).hexdigest() == expected
        fork_source = ('import json, os, time\nchild = os.fork()\n'
                       'if child == 0:\n os.setsid()\n time.sleep(30)\n os._exit(0)\n'
                       'else:\n print(json.dumps({"fork_succeeded": True, "child_namespace_pid": child}))')
        fork = execute(fork_source, timeout=2, memory_limit=256, raise_exceptions=True)
        assert fork.success and fork.stdout.get("fork_succeeded") is True, "Fork test did not actually create a child"
        # An actual repository integration call must succeed through the new transport.
        evaluator = CodeEvaluator(num_workers=1, memory_per_worker=256, timeout=3, debug=False)
        normal = evaluator('def add(a, b):\n    return a + b',
                           ['assert add(1, 2) == 3', 'assert add(-1, 1) == 0'], skip_parse=True)
        assert normal['pass_rate'] == 1.0 and normal['tests_passed'] == 2
        failed = evaluator('def add(a, b):\n    return 0', ['assert add(1, 2) == 3'], skip_parse=True)
        assert failed['pass_rate'] == 0.0 and failed['tests_passed'] == 0
        for _ in range(20):
            if _visible_pids() == before_pids:
                break
            time.sleep(0.05)
        assert _visible_pids() == before_pids, "An evaluator descendant remains in the outer PID namespace"
        status = Path("/proc/self/status").read_text()
        cap_line = next(l for l in status.splitlines() if l.startswith("CapEff:"))
        assert int(cap_line.split()[1], 16) == 0
        interfaces = sorted(p.name for p in Path("/sys/class/net").iterdir()) if Path("/sys/class/net").exists() else None
        report = {"status": "passed", "repository_import": RewardHackingEvaluation.__name__,
                  "cuda_available": torch.cuda.is_available(), "bounded_fixtures": bounded,
                  "inner_readonly": readonly.model_dump(), "fork_cleanup": fork.model_dump(),
                  "repository_success": normal, "repository_failure": failed,
                  "processes_before": sorted(before_pids), "processes_after": sorted(_visible_pids()),
                  "effective_capabilities": cap_line, "sysfs_network_exposed": interfaces,
                  "output_probe_sha256": expected, "transport": install.report()}
        Path("/output/qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        install.restore()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualify-inside", action="store_true")
    parser.add_argument("--forbidden-path", action="append", default=[])
    args = parser.parse_args()
    if not args.qualify_inside:
        parser.error("Only the authored inside-sandbox qualification has a CLI entrypoint")
    report = qualify_inside(args.forbidden_path)
    print(json.dumps({"status": report["status"], "qualification": "/output/qualification.json"}))


if __name__ == "__main__":
    main()
