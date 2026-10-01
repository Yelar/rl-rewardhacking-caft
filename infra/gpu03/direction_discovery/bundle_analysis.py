"""Analyze a complete, verified once-only test bundle without pooling components.

Preparation-only tooling: no launcher, model call, evaluator execution, target
selection, or classification changes. Real inputs remain unavailable until the
outer final-freeze and complete evaluation gates pass. A qualified CPU runner
must establish producer exit/release before invoking ``verify_recompute`` in a
fresh process; this module's proof concerns artifact contents, not process state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

MASTER_SHA = "1d6592a8005234cbface55bfc337d0b15ea11ec67ec01339b4aa8ffe2f69e1b2"
PARENT_SHA = "d4aa5109725bf2d4765e9bf54689c0e688a0a0e6c1922340d3c009389934ba10"
METRICS_SHA = "9f1a878502d4ffadc95e91e8df05c9d9dff952c6d04843f5d1d4c61d10a081b5"
MECHANISM_SHA = "02e2607734ce7cb11b9a300f3e8b6409ef5c95e33dc0051373b01e94844e379f"
BINDINGS = ("bundle", "master", "parent", "evaluation_manifest",
            "evaluation_artifact_manifest", "evaluation_verification")
COMPONENTS = ("core_untouched_test", "auxiliary_test")
GROUPS = ("correct_harmful", "incorrect_harmful_strict")
IDENTITY = ("request_id", "condition_id", "record_id", "problem_id", "scope", "sample_index", "seed")
MAX_BYTES = 256 * 1024 * 1024
MAX_LINE = 8 * 1024 * 1024
MAX_ROWS = 1555


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def snapshot(path, maximum=MAX_BYTES):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
            "Missing, symlinked, or oversized input")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "Input grew past size bound")
    return raw


def bound(item, base=None):
    require(isinstance(item, dict) and isinstance(item.get("path"), str) and
            re.fullmatch("[0-9a-f]{64}", item.get("sha256", "")), "Unresolved input binding")
    path = Path(item["path"])
    if base is not None:
        require(not path.is_absolute() and ".." not in path.parts, "Nonportable bundle path")
        path = base / path
        require(path.resolve().is_relative_to(base.resolve()), "Bundle path escapes package")
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and
            sha(path) == item["sha256"], "Missing or changed input binding")
    return path


def read_bound(item, base=None):
    path = bound(item, base)
    raw = snapshot(path)
    require(digest(raw) == item["sha256"], "Input changed during read")
    return path, parse(raw)


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))


def modules():
    from . import metrics, mechanism_audit, eval_run, test_bundle
    return metrics, mechanism_audit, eval_run, test_bundle


def source_check(spec, loaded):
    require(isinstance(spec.get("source_bindings"), dict), "Missing reviewed source bindings")
    paths = {"bundle_analysis.py": Path(__file__)}
    paths.update({name: Path(module.__file__) for name, module in zip(
        ("metrics.py", "mechanism_audit.py", "eval_run.py", "test_bundle.py"), loaded)})
    for name, path in paths.items():
        require(spec["source_bindings"].get(name) == sha(path), "Reviewed runtime source changed: " + name)
    require(sha(paths["metrics.py"]) == METRICS_SHA and sha(paths["mechanism_audit.py"]) == MECHANISM_SHA,
            "Statistical or mechanism implementation changed")
    return {name: sha(path) for name, path in paths.items()}


def unique_requests(plan, ids):
    requests = plan.get("requests")
    require(isinstance(requests, list) and len(requests) <= MAX_ROWS, "Invalid request population")
    result = {row["request_id"]: row for row in requests}
    require(len(result) == len(requests) and sorted(result) == sorted(ids) and len(ids) == len(set(ids)),
            "Duplicate or incomplete frozen request IDs")
    return result


def context(spec, loaded=None):
    """Consume final-freeze/bundle/terminal proofs before loading outcome rows."""
    require(spec.get("schema_version") == 1 and spec.get("purpose") == "completed_test_bundle_analysis" and
            spec.get("runnable") is True, "Pending or unsupported bundle analysis spec")
    loaded = modules() if loaded is None else loaded
    metrics, mechanism, evaluator, bundler = loaded
    sources = source_check(spec, loaded)
    require(spec["master"]["sha256"] == MASTER_SHA and spec["parent"]["sha256"] == PARENT_SHA,
            "Frozen master/parent changed")
    # Inspect only bundle/final metadata before the verifier hashes test payloads.
    bundle_path = bound(spec["bundle"])
    _, bundle = read_bound(spec["bundle"])
    require(bundle.get("schema_version") == 1 and bundle.get("purpose") == "checkpoint60_once_only_test_bundle" and
            bundle.get("status") == "frozen_before_test_generation" and
            bundle.get("master_plan_sha256") == MASTER_SHA and bundle.get("parent_plan_sha256") == PARENT_SHA,
            "Wrong test bundle status or plan")
    root = bundle_path.parent
    _, final = read_bound(bundle["frozen_final_config"], root)
    require(final.get("status") == "frozen_before_untouched_test" and final.get("no_test_outcomes_used") is True and
            final.get("master_plan_sha256") == MASTER_SHA and final.get("conditions") == bundle["conditions"],
            "Missing positive final configuration freeze")
    verified_bundle = bundler.verify_bundle(bundle_path, spec["bundle"]["sha256"])
    require(verified_bundle == bundle, "Bundle verifier returned different metadata")
    bundle_proof = {"status": "verified_metadata_only", "bundle_sha256": spec["bundle"]["sha256"],
                    "verifier_source_sha256": sources["test_bundle.py"], "method": "test_bundle.verify_bundle",
                    "request_count": len(bundle["request_ids"])}
    paths = {name: bound(spec[name]) for name in BINDINGS}
    _, master = read_bound(spec["master"])
    plan_path, plan = read_bound(bundle["generation_request_plan"], root)
    require(plan.get("mode") == "generate" and plan.get("phase") == "untouched_test_bundle" and
            plan.get("evaluation_partition") == "untouched_test" and plan.get("master_plan_sha256") == MASTER_SHA and
            plan.get("conditions") == bundle["conditions"], "Wrong full-bundle request plan")
    expected = unique_requests(plan, bundle["request_ids"])
    require(0 < len(expected) <= MAX_ROWS, "Invalid full bundle size")
    require(set(bundle["components"]) == set(COMPONENTS), "Wrong test component set")
    component_ids, component_plans = {}, {}
    union = set()
    for name in COMPONENTS:
        value = bundle["components"][name]
        _, part = read_bound(value["plan"], root)
        ids = set(value["request_ids"])
        rows = unique_requests(part, value["request_ids"])
        require(not union.intersection(ids) and all(expected.get(key) == row for key, row in rows.items()) and
                part.get("conditions") == plan["conditions"] and part.get("evaluation_partition") == "untouched_test",
                "Component overlaps, changes requests, or has wrong conditions/partition")
        require(name != "core_untouched_test" or bool(ids), "Core test component is empty")
        union.update(ids)
        component_ids[name], component_plans[name] = ids, value["plan"]
    require(union == set(expected), "Components do not exactly cover the full bundle")
    group_ids = bundle.get("auxiliary_groups", {})
    require(set(group_ids) == set(GROUPS) and all(isinstance(group_ids[name], list) and
            len(group_ids[name]) == len(set(group_ids[name])) for name in GROUPS), "Invalid auxiliary strata")
    group_sets = [set(group_ids[name]) for name in GROUPS]
    require(not group_sets[0].intersection(group_sets[1]) and
            set().union(*group_sets) == component_ids["auxiliary_test"], "Auxiliary strata do not partition the frozen component")

    # Read positive external terminal receipt first. eval_run.verify then checks
    # the complete package and process release; no subset masquerades as its file.
    _, external = read_bound(spec["evaluation_verification"])
    require(external.get("status") == "verified" and external.get("mode") == "production" and
            external.get("manifest_sha256") == spec["evaluation_manifest"]["sha256"] and
            external.get("records") == len(expected) and external.get("exact_request_coverage") is True and
            external.get("process_release_verified") is True and
            external.get("artifact_manifest_sha256") == spec["evaluation_artifact_manifest"]["sha256"],
            "Incomplete independent full evaluation proof")
    _, manifest = read_bound(spec["evaluation_manifest"])
    require(manifest.get("mode") == "production" and manifest.get("phase") == "behavior_untouched_test_bundle" and
            manifest["scientific"]["master_plan_sha256"] == MASTER_SHA and
            manifest["scientific"]["parent_plan_sha256"] == PARENT_SHA and
            sorted(manifest["request_ids"]) == sorted(expected), "Evaluation is not the complete frozen test bundle")
    bundle_link = manifest["scientific"].get("test_bundle")
    require(isinstance(bundle_link, dict) and bundle_link.get("sha256") == spec["bundle"]["sha256"],
            "Evaluation has no exact frozen bundle binding")
    bound(bundle_link)  # The evaluator may bind a staged byte-identical copy.
    require(manifest["source_files"]["infra/gpu03/direction_discovery/eval_run.py"]["sha256"] == sources["eval_run.py"],
            "Evaluation verifier source differs from its frozen producer")
    stage, result = Path(manifest["stage"]), Path(manifest["output"])
    require(paths["evaluation_verification"] == stage / "control/independent_verification.json" and
            paths["evaluation_artifact_manifest"] == result / "artifact_manifest.json" and
            sha(stage / "input/request_plan.json") == bundle["generation_request_plan"]["sha256"] and
            manifest["input_hashes"]["request_plan"] == bundle["generation_request_plan"]["sha256"],
            "Evaluation proof or frozen request input mismatch")
    require(evaluator.load_manifest(paths["evaluation_manifest"], spec["evaluation_manifest"]["sha256"]) == manifest,
            "Validated evaluation manifest differs from its bound metadata snapshot")
    proof = evaluator.verify(paths["evaluation_manifest"], spec["evaluation_manifest"]["sha256"])
    require(proof == external, "Fresh complete evaluation verification differs from independent receipt")
    evaluation = result / "evaluations.jsonl"
    require(proof["evaluations"] == str(evaluation), "Full evaluation file path mismatch")
    raw = snapshot(evaluation)
    require(digest(raw) == proof["evaluations_sha256"], "Verified evaluation snapshot changed")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == len(expected) and all(line.endswith(b"\n") and len(line) <= MAX_LINE for line in lines),
            "Incomplete, extra, or oversized evaluation lines")
    rows = [parse(line) for line in lines]
    by_id = {row["request_id"]: row for row in rows}
    require(len(by_id) == len(rows) and set(by_id) == set(expected), "Incomplete or duplicate evaluation coverage")
    for row in rows:
        planned = expected[row["request_id"]]
        require(all(row.get(key) == planned[key] for key in IDENTITY) and row.get("problem_split") == "untouched_test" and
                row.get("evaluation_status") != "infrastructure_failure", "Wrong evaluation identity, split, or infrastructure state")
    definitions, _ = metrics.validate_conditions(plan["conditions"])
    for ids in component_ids.values():
        if ids:
            metrics.validate_rows([row for row in rows if row["request_id"] in ids], definitions)
    return {"spec": spec, "paths": paths, "bundle": bundle, "bundle_proof": bundle_proof, "master": master,
            "plan": plan, "plan_path": plan_path, "proof": proof, "evaluation": evaluation, "raw": raw,
            "rows": rows, "lines": lines, "component_ids": component_ids, "component_plans": component_plans,
            "group_ids": group_ids, "sources": sources, "loaded": loaded}


def inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and not any(path.is_symlink() for path in root.rglob("*")),
            "Missing or symlinked output package")
    return {str(path.relative_to(root)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()}


def output_guard(output, ctx):
    output = Path(output)
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), "Analysis output must be fresh")
    require(output.parent.is_dir() and output.parent.resolve() == output.parent, "Output parent is missing or symlinked")
    roots = [ctx["paths"]["bundle"].parent, ctx["evaluation"].parent,
             ctx["paths"]["evaluation_manifest"].parent]
    require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots) and
            not any(path.is_relative_to(output) for path in ctx["paths"].values()), "Analysis output overlaps input packages")
    return output


def analyze_into(output, ctx):
    """Produce deterministic subsets and unchanged statistics from one snapshot."""
    output = output_guard(output, ctx)
    metrics, mechanism, _, _ = ctx["loaded"]
    output.mkdir()
    write(output / "resolved_spec.json", ctx["spec"])
    write(output / "full_evaluation_verification.json", ctx["proof"])
    write(output / "bundle_verification.json", ctx["bundle_proof"])
    reports = {}
    populations = [(name, ids, "component") for name, ids in ctx["component_ids"].items()]
    populations.extend(("auxiliary_test/groups/" + name, set(ids), "predeclared_auxiliary_group")
                       for name, ids in ctx["group_ids"].items())
    for name, ids, kind in populations:
        directory = output / "components" / name
        directory.mkdir(parents=True, exist_ok=True)
        selected = [(index, row, line) for index, (row, line) in enumerate(zip(ctx["rows"], ctx["lines"]))
                    if row["request_id"] in ids]
        subset = b"".join(line for _, _, line in selected)
        (directory / "evaluations.jsonl").write_bytes(subset)
        provenance = {"kind": kind, "component": name, "full_evaluation_path": str(ctx["evaluation"]),
                      "full_evaluation_sha256": ctx["proof"]["evaluations_sha256"],
                      "full_evaluation_manifest_sha256": ctx["proof"]["manifest_sha256"],
                      "full_artifact_manifest_sha256": ctx["proof"]["artifact_manifest_sha256"],
                      "bundle_sha256": ctx["spec"]["bundle"]["sha256"],
                      "frozen_final_config_sha256": ctx["bundle"]["frozen_final_config"]["sha256"],
                      "component_plan": ctx["component_plans"][name.split("/", 1)[0]],
                      "request_ids": sorted(ids), "request_ids_sha256": digest(canonical(sorted(ids)).encode()),
                      "subset_sha256": digest(subset), "subset_bytes": len(subset),
                      "source_lines": [{"line_index_zero_based": index, "request_id": row["request_id"],
                                        "sha256": digest(line), "size_bytes": len(line)} for index, row, line in selected],
                      "row_bytes_unchanged": True, "full_proof_does_not_describe_subset_path": True,
                      "components_are_not_independent_replicates": True, "source_bindings": ctx["sources"]}
        write(directory / "subset_provenance.json", provenance)
        component_rows = [row for _, row, _ in selected]
        if component_rows:
            statistical = metrics.analyze(component_rows, ctx["plan"]["conditions"], ctx["master"])
            records, mechanism_summary = mechanism.audit(component_rows)
            write(directory / "metrics.json", statistical)
            with (directory / "mechanism_records.jsonl").open("xb") as stream:
                for record in records:
                    stream.write((canonical(record) + "\n").encode("utf-8"))
            write(directory / "mechanism_summary.json", mechanism_summary)
            reports[name] = {"status": "analysis_complete", "rows": len(component_rows)}
        else:
            reports[name] = {"status": "unavailable_no_eligible_pairs", "rows": 0}
            write(directory / "unavailable.json", reports[name])
    # Recheck hashes after computation without parsing any additional rows.
    require(sha(ctx["evaluation"]) == ctx["proof"]["evaluations_sha256"], "Full evaluation changed during analysis")
    for name, path in ctx["paths"].items():
        require(sha(path) == ctx["spec"][name]["sha256"], "Bound input changed during analysis")
    summary = {"schema_version": 1, "status": "component_analysis_complete", "rows": len(ctx["rows"]),
               "components": reports, "statistics_changed": False, "classifications_changed": False,
               "decision_created": False, "test_outcomes_used_for_selection": False,
               "components_pooled": False, "subset_line_bytes_preserved": True,
               "full_evaluation_verification": "full_evaluation_verification.json"}
    write(output / "summary.json", summary)
    write(output / "artifact_manifest.json", {"algorithm": "sha256", "files": inventory(output)})
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    return {"status": "component_analysis_complete", "output": str(output), "rows": len(ctx["rows"]),
            "artifact_manifest_sha256": sha(output / "artifact_manifest.json"), "decision_created": False}


def verify_recompute(output, recomputation_output, ctx):
    """Fresh-process caller supplies a fresh context; both result dirs are retained."""
    output, recomputation_output = Path(output), Path(recomputation_output)
    require(output != recomputation_output and not output.is_relative_to(recomputation_output) and
            not recomputation_output.is_relative_to(output), "Producer and recomputation outputs overlap")
    expected = parse(snapshot(output / "artifact_manifest.json"))
    before = inventory(output)
    require(expected == {"algorithm": "sha256", "files": {k: v for k, v in before.items() if k != "artifact_manifest.json"}},
            "Producer artifact inventory/hash mismatch")
    analyze_into(recomputation_output, ctx)
    require(inventory(output) == before == inventory(recomputation_output), "Independent component recomputation differs")
    return {"status": "verified", "output": str(output), "rows": len(ctx["rows"]),
            "artifact_manifest_sha256": sha(output / "artifact_manifest.json"), "exact_recomputation_match": True,
            "full_evaluation_manifest_sha256": ctx["proof"]["manifest_sha256"],
            "full_evaluation_sha256": ctx["proof"]["evaluations_sha256"],
            "bundle_sha256": ctx["spec"]["bundle"]["sha256"], "decision_created": False,
            "process_exit_and_release_verified_by_this_module": False}


def run_under_runner(common, operation, payload_sha256):
    """Adapter for the existing qualified CPU runner; never starts a service.

    Its frozen payload binds tools, source, input/analysis_spec.json, both fresh
    outputs and resource limits. The runner captures this result in the usual
    exit receipt; a separate release step still verifies this invocation exited.
    """
    require(operation in ("producer", "verifier"), "Unsupported runner operation")
    config = common.frozen(payload_sha256)
    common.require_start(operation, config, payload_sha256)
    spec = parse(snapshot(common.STAGE / "input/analysis_spec.json"))
    output, recompute = Path(config["output"]), Path(config["recomputation_output"])
    require(spec.get("output") == str(output) and spec.get("recomputation_output") == str(recompute),
            "Analysis outputs differ from qualified runner payload")
    if operation == "verifier":
        # This positively checks the independent producer exit and release before
        # rereading its outcome inputs; absent/failed release stops here.
        release = common.release("producer", config, payload_sha256)
        require(release["result"]["artifact_manifest_sha256"] == sha(output / "artifact_manifest.json"),
                "Released producer artifact binding changed")
    ctx = context(spec)
    return (analyze_into(output, ctx) if operation == "producer" else verify_recompute(output, recompute, ctx))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("producer", "verifier"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    _, spec = read_bound({"path": str(args.spec), "sha256": args.spec_sha256})
    ctx = context(spec)
    result = (analyze_into(Path(spec["output"]), ctx) if args.operation == "producer" else
              verify_recompute(Path(spec["output"]), Path(spec["recomputation_output"]), ctx))
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
