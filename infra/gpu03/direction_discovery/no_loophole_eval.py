"""Opt-in provenance gate around the frozen whole-program CPU evaluator.

The staged eval_run/evaluate pair must be the historical qualified bytes. This
module adds no worker, generated-code execution, score, or recovery route.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

PHASE = "no_loophole_capability"
BASE_SHA = "63102cdb2117dbb2910bda25586f05579d0a361f5b63d4c5a0065a812848015e"
EVALUATE_SHA = "117e5b3784edb9d969e04ff2ec6d950e7b0f5fc31e851569753601cb79f8086f"
METRICS_SHA = "9f1a878502d4ffadc95e91e8df05c9d9dff952c6d04843f5d1d4c61d10a081b5"
ENTRY = "infra/gpu03/direction_discovery/no_loophole_eval.py"
FORBIDDEN = {"execution_partition", "test_execution", "test_bundle", "finalist_recovery", "recovered_finalist_round"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def modules():
    from infra.gpu03.direction_discovery import eval_run, no_loophole_plan, metrics
    require(sha(eval_run.__file__) == BASE_SHA and sha(eval_run.evaluate.__file__) == EVALUATE_SHA,
            "Use the exact frozen evaluator pair; current diagnostic variants are not this protocol")
    require(sha(metrics.__file__) == METRICS_SHA, "Frozen statistics source changed")
    return eval_run, no_loophole_plan


def validate_plan(plan, planner):
    planner.validate_full(plan)
    require(plan.get("phase") == PHASE and plan.get("mode") == "generate" and
            plan.get("evaluation_partition") == "configuration_validation" and
            isinstance(plan.get("no_loophole_capability"), dict) and
            not FORBIDDEN.intersection(plan), "Only the explicit full no-loophole plan is accepted")
    require(len(plan["requests"]) == 740, "Exactly 740 fresh requests are required")
    return planner.bindings(plan, verify=True)


def bound_json(path, expected, base):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Metadata must be a regular file")
    payload = path.read_bytes()
    require(expected == {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}, "Metadata snapshot binding changed")
    return base.parse(payload)


def caller_inputs(plan, prepared, dataset):
    for supplied, key in ((prepared, "prepared_records"), (dataset, "dataset")):
        path = Path(supplied)
        # Reject an unrelated caller path before hashing or opening its bytes.
        require(path == Path(plan[key]) and path.is_absolute(), "Caller input path is not the exact no-hint package input")
        require(path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid() and
                not path.stat().st_mode & 0o222 and sha(path) == plan[key + "_sha256"],
                "Caller no-hint input is mutable or differs from the frozen package")


def generation_metadata(packages, plan, prepared_sha, *, stored=False):
    """Admit the entire metadata union before the base collector opens rows."""
    base, planner = modules()
    from infra.gpu03.direction_discovery import supervisor
    dependencies = validate_plan(plan, planner)
    planned = base.unique_rows(plan["requests"], "full no-loophole plan")
    require(isinstance(packages, (list, tuple)) and packages, "Missing terminal generation packages")
    seen, covered, proofs = set(), {}, []
    for package in packages:
        require(isinstance(package, dict) and "kind" not in package, "Remote, recovered and test packages require another protocol")
        keys = {"manifest", "manifest_sha256", "artifact_manifest_sha256", "verification", "request_ids"} if stored else {
            "manifest", "manifest_sha256", "artifact_manifest_sha256", "verification", "verification_sha256"}
        require(set(package) == keys, "Generation proof schema differs from this protocol")
        path, digest = Path(package["manifest"]), package["manifest_sha256"]
        require(str(path) not in seen and digest not in seen, "Repeated producer package")
        seen.update((str(path), digest))
        m = supervisor.load_manifest(path, digest, check_files=False, check_source=False)
        science = m["scientific"]
        require(science.get("no_loophole_capability") == plan["no_loophole_capability"] and
                science.get("master_plan_sha256") == plan["master_plan_sha256"] and
                science.get("input_prepared_sha256") == prepared_sha and
                not FORBIDDEN.intersection(science), "Producer environment or scientific lineage differs")
        planner.validate_source(plan, m["source_root"])
        part_path = Path(m["stage"]) / "input/request_plan.json"
        require(bound_json(part_path, m["bound_files"].get(str(part_path)), base) == plan,
                "Producer does not bind the exact full fresh plan")
        for binding in dependencies:
            require(m["bound_files"].get(str(binding)) == base.info(binding), "Producer lost a no-loophole dependency")
        # These existing APIs validate terminal identity, exact artifacts and release.
        proof = supervisor.verify(path, digest)
        require(proof.get("status") == "verified" and proof.get("gpu_release_verified") is True and
                proof.get("manifest_sha256") == digest, "Producer has no positive terminal/release proof")
        if stored:
            external = package["verification"]
        else:
            payload = Path(package["verification"]).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == package["verification_sha256"], "External producer verification changed")
            external = base.parse(payload)
        require(external == proof, "Fresh and stored producer verification disagree")
        require(sha(Path(m["output"]) / "artifact_manifest.json") == package["artifact_manifest_sha256"],
                "Producer artifact manifest changed")
        ids = []
        for worker in m["workers"]:
            require(worker["success_expect"]["mode"] == "generate", "Qualification is not a scientific generation package")
            task_path = Path(worker["command"][3])
            require(m["bound_files"].get(str(task_path)) == base.info(task_path), "Generation task is not bound")
            task = base.read_json(task_path)
            require(task["mode"] == "generate" and task["conditions"] == plan["conditions"] and
                    task["sampling"] == plan["sampling"] and sha(task["prepared_records"]) == prepared_sha,
                    "Generation task conditions, sampling or prompts differ")
            require(len(task["requests"]) == worker["success_expect"]["requests"], "Worker metadata count differs")
            for row in task["requests"]:
                rid = row.get("request_id")
                require(rid in planned and row == planned[rid] and rid not in covered, "Changed, overlapping or foreign request")
                covered[rid] = row
                ids.append(rid)
        if stored:
            require(package["request_ids"] == sorted(ids), "Stored proof request coverage changed")
        proofs.append(proof)
    require(set(covered) == set(planned), "All 740 requests must be terminal before completion rows are read")
    return proofs


def build_manifest(**spec):
    base, planner = modules()
    require(spec.get("phase") == PHASE and spec.get("mode", "production") == "production" and
            spec.get("workers", 8) == 8 and spec.get("recovery_plan") is None and spec.get("test_bundle") is None,
            "No-loophole evaluation uses eight frozen workers and no recovery/test route")
    prefix = Path("infra/gpu03/direction_discovery")
    for relative, expected in ((prefix / "eval_run.py", BASE_SHA), (prefix / "evaluate.py", EVALUATE_SHA),
                               (prefix / "metrics.py", METRICS_SHA), (Path(ENTRY), sha(__file__)),
                               (prefix / "no_loophole_plan.py", sha(planner.__file__))):
        require(sha(Path(spec["source_dir"]) / relative) == expected, "Builder source must be the frozen capability snapshot")
    plan = base.read_json(spec["request_plan"])
    validate_plan(plan, planner)
    caller_inputs(plan, spec["prepared_records"], spec["dataset"])
    prepared = list(base.read_jsonl(spec["prepared_records"]))
    dataset = list(base.read_jsonl(spec["dataset"]))
    planner.validate_inputs(plan, prepared, dataset)
    generation_metadata(spec["generation_packages"], plan, sha(spec["prepared_records"]))
    result = base.build_manifest(**spec)
    m, _ = load_manifest(result["manifest"], result["manifest_sha256"])
    result["command"] = base.clean_command([base.PYTHON, "-B", str(Path(m["stage"]) / "source" / ENTRY),
        "--launch", "--manifest", result["manifest"], "--manifest-sha256", result["manifest_sha256"]])
    return {**result, "environment": plan["no_loophole_capability"], "evaluation_adapter_sha256": sha(__file__)}


def load_manifest(path, digest):
    base, planner = modules()
    m = base.load_manifest(path, digest, check_files=False)
    require(m["mode"] == "production" and m["phase"] == PHASE and len(m["workers"]) == 8 and
            len(m["request_ids"]) == 740 and not m["recovered_request_ids"] and
            not FORBIDDEN.intersection(m["scientific"]) and "correctness_diagnostics" not in m["scientific"],
            "Wrong environment, partial evaluation or altered evaluator policy")
    prefix = "infra/gpu03/direction_discovery/"
    for relative, expected in ((prefix + "eval_run.py", BASE_SHA), (prefix + "evaluate.py", EVALUATE_SHA),
                               (prefix + "metrics.py", METRICS_SHA), (ENTRY, sha(__file__)),
                               (prefix + "no_loophole_plan.py", sha(planner.__file__))):
        require(m["source_files"].get(relative, {}).get("sha256") == expected, "Staged capability source differs")
    inputs = Path(m["stage"]) / "input"
    plan = bound_json(inputs / "request_plan.json", m["input_files"].get("request_plan.json"), base)
    validate_plan(plan, planner)
    require(m["input_hashes"]["request_plan"] == m["input_files"]["request_plan.json"]["sha256"] and
            m["input_hashes"]["prepared_records"] == plan["prepared_records_sha256"] and
            m["input_hashes"]["dataset"] == plan["dataset_sha256"], "Evaluator input environment hashes differ")
    generation_metadata(bound_json(inputs / "generation_verification.json", m["input_files"].get("generation_verification.json"), base), plan,
                        m["input_hashes"]["prepared_records"], stored=True)
    require(set(m["request_ids"]) == {r["request_id"] for r in plan["requests"]}, "Evaluation request union changed")
    # All environment and complete-generation gates precede native payload reads.
    require(base.load_manifest(path, digest) == m, "Evaluation manifest changed during validation")
    planner.validate_inputs(plan, list(base.read_jsonl(inputs / "prepared_records.jsonl")),
                            list(base.read_jsonl(inputs / "dataset.jsonl")))
    return m, plan


def launch(path, digest):
    base, _ = modules()
    load_manifest(path, digest)
    return base.launch(path, digest)


def verify(path, digest):
    base, _ = modules()
    m, plan = load_manifest(path, digest)
    proof = base.verify(path, digest)
    require(proof.get("exact_request_coverage") is True and proof.get("process_release_verified") is True and
            proof.get("records") == 740, "Evaluation is not a complete released740 package")
    return {**proof, "purpose": "verified_no_loophole_capability_evaluation", "environment": plan["no_loophole_capability"],
            "request_plan_sha256": m["input_hashes"]["request_plan"], "evaluation_adapter_sha256": sha(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ("build", "launch", "verify"):
        mode.add_argument("--" + name, action="store_true")
    p.add_argument("--spec"); p.add_argument("--manifest"); p.add_argument("--manifest-sha256")
    p.add_argument("--verification-output")
    args = p.parse_args()
    base, _ = modules()
    if args.build:
        require(args.spec is not None, "An exact inner builder specification is required")
        result = build_manifest(**base.read_json(args.spec))
    else:
        require(args.manifest and args.manifest_sha256, "Exact manifest reference required")
        result = (launch if args.launch else verify)(args.manifest, args.manifest_sha256)
    if args.verification_output:
        base.write_json(args.verification_output, result)
    print(base.canonical(result), flush=True)


if __name__ == "__main__":
    main()
