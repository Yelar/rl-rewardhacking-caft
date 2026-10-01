"""CAFT's original reward, computed only inside the existing two-layer sandbox.

This module does not parse or execute completion code. The child imports the
unchanged ordinary reward and installs the already reviewed evaluator transport.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile

MAX_INPUT_BYTES = 8 * 1024**2
MAX_RECORDS = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def native(value):
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "Reward JSON keys must be strings")
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if type(value).__module__.split(".")[0] == "numpy":
        return native(value.tolist() if hasattr(value, "tolist") else value.item())
    return value


def canonical(value):
    return json.dumps(native(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def ref(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def validate_config(config):
    require(config.get("kind") == "original_reward_two_layer_bwrap_v1", "Wrong reward sandbox profile")
    for name in ("source_dir", "venv_dir", "spool_dir"):
        path = Path(config[name])
        require(path.is_absolute() and path.resolve() == path and path.is_dir(), "Noncanonical reward path")
    validate_limits(config)
    return config


def validate_limits(config):
    cpus = config["cpu_set"]
    require(isinstance(cpus, list) and cpus and len(set(cpus)) == len(cpus) and
            all(type(x) is int and x >= 0 for x in cpus), "Invalid reward CPU set")
    require(type(config["workers"]) is int and 1 <= config["workers"] <= 8, "Reward worker bound")
    require(type(config["wall_seconds"]) is int and 1 <= config["wall_seconds"] <= 900,
            "Reward wall bound")
    require(type(config["memory_bytes"]) is int and 1024**3 <= config["memory_bytes"] <= 8 * 1024**3,
            "Reward address-space bound")
    return config


def validate_payload(payload):
    require(set(payload) == {"examples", "responses", "reward_kwargs", "limits"}, "Reward payload fields")
    examples, responses = payload["examples"], payload["responses"]
    require(isinstance(examples, list) and 1 <= len(examples) <= MAX_RECORDS and
            isinstance(responses, list) and len(examples) == len(responses), "Reward batch coverage")
    require(all(isinstance(x, dict) for x in examples) and all(isinstance(x, str) for x in responses),
            "Reward records must be JSON objects/text")
    kwargs = payload["reward_kwargs"]
    require(kwargs == {"correct_reward": 3.0, "format_reward": 0.5, "allow_hint": True},
            "The original ordinary reward must remain unchanged")


def run_reward(config, *, examples, responses, reward_kwargs):
    """Called in Ray; files are inert data, execution is confined to Bubblewrap."""
    from infra.gpu03.direction_discovery import sandbox as cpu_sandbox
    config = validate_config(config)
    payload = {"examples": examples, "responses": responses, "reward_kwargs": reward_kwargs,
               "limits": {k: config[k] for k in ("cpu_set", "workers", "wall_seconds", "memory_bytes")}}
    payload = native(payload)
    validate_payload(payload)
    encoded = canonical(payload)
    require(len(encoded) <= MAX_INPUT_BYTES, "Reward input exceeds 8 MiB")
    directory = Path(tempfile.mkdtemp(prefix="reward_batch_", dir=config["spool_dir"]))
    inputs, outputs = directory / "input", directory / "output"
    inputs.mkdir(mode=0o700); outputs.mkdir(mode=0o700)
    write(inputs / "payload.json", payload)
    (inputs / "payload.json").chmod(0o400)
    write(directory / "intent.json", {"input": ref(inputs / "payload.json"), "config": config})
    try:
        result = cpu_sandbox.run_outer(
            source_dir=config["source_dir"], input_dir=inputs, output_dir=outputs,
            venv_dir=config["venv_dir"], workers=config["workers"],
            python_args=["-B", "/work/src/train/verl/caft_reward_sandbox.py", "--inside"],
            wall_timeout=config["wall_seconds"])
    except BaseException as error:
        write(directory / "transport_failure.json", {"type": type(error).__name__,
              "completed_reward": False, "input": ref(inputs / "payload.json")})
        raise
    write(directory / "actual_transport.json", result)
    require(result["returncode"] == 0, "Reward sandbox process failed; no fallback reward")
    path = outputs / "result.json"
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2,
            "Missing/bounded reward result")
    answer = json.loads(path.read_bytes())
    require(answer.get("status") == "original_reward_completed_in_two_layer_sandbox" and
            answer.get("input") == ref(inputs / "payload.json"), "Reward result/input join")
    scores, extras = answer["scores"], answer["extra_infos"]
    require(len(scores) == len(examples) and all(type(x) in (int, float) and math.isfinite(x) and
            x in (0.0, 0.5, 3.0, 3.5) for x in scores), "Invalid ordinary reward values")
    require(isinstance(extras, dict) and all(isinstance(v, list) and len(v) == len(scores)
            for v in extras.values()), "Reward extra-info coverage")
    require(extras.get("id") == [int(x["id"]) for x in examples], "Reward example order changed")
    write(directory / "COMPLETED.json", {"result": ref(path), "records": len(scores),
          "actual_transport": ref(directory / "actual_transport.json")})
    return scores, extras


def inside():
    require(sys.argv[1:] == ["--inside"] and os.environ.get("CODE_EVAL_SANDBOX") == "bwrap"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and Path(sys.executable).resolve() == Path("/usr/bin/python3.12")
            and Path("/work/src/evaluate/helpers.py").is_file(),
            "Reward child must be in the reviewed outer sandbox")
    path = Path("/input/payload.json")
    require(path.is_file() and path.stat().st_size <= MAX_INPUT_BYTES, "Reward input bound")
    payload = json.loads(path.read_bytes()); validate_payload(payload)
    limits = payload["limits"]
    validate_limits(limits)
    require(set(limits["cpu_set"]).issubset(os.sched_getaffinity(0)), "Reward CPU allocation unavailable")
    os.sched_setaffinity(0, limits["cpu_set"])
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
    resource.setrlimit(resource.RLIMIT_CPU, (limits["wall_seconds"], limits["wall_seconds"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    from infra.gpu03.direction_discovery import bounded_evaluator
    installation = bounded_evaluator.install_bounded_evaluator()
    try:
        # Preserve original parser, primitive tests, labels, and 3+.5 arithmetic.
        from src.train.rewards import CorrectOrHintedCompileCode
        reward = CorrectOrHintedCompileCode(**payload["reward_kwargs"])
        reward.allow_logging = False
        with contextlib.redirect_stdout(sys.stderr):
            scores, extras = reward.compute_reward(payload["examples"], payload["responses"])
        write("/output/result.json", {"status": "original_reward_completed_in_two_layer_sandbox",
              "input": ref(path), "scores": scores, "extra_infos": extras,
              "transport": installation.report(), "cpu_set": sorted(os.sched_getaffinity(0))})
    finally:
        installation.restore()


if __name__ == "__main__":
    inside()
