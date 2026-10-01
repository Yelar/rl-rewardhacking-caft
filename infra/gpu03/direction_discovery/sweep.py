#!/usr/bin/env python3
"""Offline, manifest-bound validation TF sweep planning and prioritization.

No model forward, generation, evaluator execution or GPU launch occurs here.
Likelihood ranks prioritize behavioral experiments; they do not establish a
successful intervention or an intent representation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from . import candidates as c
except ImportError:
    import candidates as c


def engine_seed(*parts):
    # Reuse the inference engine's exact random-control seed convention. Lazy
    # import keeps data-only helpers available without importing torch.
    try:
        from .engine import stable_seed
    except ImportError:
        from engine import stable_seed
    return stable_seed(*parts)


def request_id(master_sha, phase, record_id, condition_id):
    payload = c.canonical(["tf", master_sha, phase, record_id, condition_id]).encode()
    return "tf-" + hashlib.sha256(payload).hexdigest()


def validation_rows(rows, master):
    problems = master["sweep"]["teacher_forced_validation_problems"]
    c.require(len(problems) == 12 and len(set(problems)) == 12, "TF problem selection differs from frozen 12")
    by_problem = defaultdict(list)
    for row in rows:
        if str(row["problem_id_key"]) in problems:
            c.require(row["problem_split"] == "configuration_validation", "TF planning attempted fit/test leakage")
            by_problem[str(row["problem_id_key"])].append(row)
    result = []
    for problem in problems:
        group = by_problem[problem]
        c.require(len(group) == 3 and {r["outcome_presence_class"] for r in group} == set(c.CLASSES), "TF requires matched validation triplet")
        c.require(len({r["prompt_sha256"] for r in group}) == 1, "validation triplet prompt mismatch")
        result.extend(sorted(group, key=lambda r: c.CLASSES.index(r["outcome_presence_class"])))
    return result


def load_verified_catalog(root, verification):
    root = Path(root)
    c.require(verification["status"] == "verified" and verification["mode"] == "production" and
              verification["artifact_manifest_sha256"] == c.sha256_file(root / "artifact_manifest.json"),
              "candidate verification receipt does not match production package")
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for relative, info in manifest["files"].items():
        c.verify_file(c.safe_child(root, relative), info)
    success = json.loads((root / "SUCCESS.json").read_text())
    c.require(success["status"] == "succeeded" and success["layers"] == list(range(36)), "candidate campaign is incomplete")
    catalog = json.loads((root / "candidate_catalog.json").read_text())["candidates"]
    c.require(len(catalog) == len({r["candidate_id"] for r in catalog}), "duplicate candidate catalog IDs")
    return catalog, manifest


def expected_candidates(layer):
    prefix = f"L{layer:02d}.transition"
    return ([f"{prefix}.mean.{family}.{kind}" for family in
             ("harmful_vs_benign", "harmful_incorrect_vs_benign_incorrect") for kind in ("v60", "v_change")]
            + [f"{prefix}.pc{i:02d}" for i in range(10)])


def make_conditions(catalog, candidate_root, candidate_manifest, layers, seed_bases):
    by_id = {r["candidate_id"]: r for r in catalog}
    c.require(len(seed_bases) == len(set(seed_bases)) == 3, "need three distinct random seed bases")
    conditions = {"baseline": {"role": "baseline", "layers": []}}
    for layer in layers:
        for candidate_id in expected_candidates(layer):
            c.require(candidate_id in by_id, "required candidate absent or rejected: " + candidate_id)
            item = by_id[candidate_id]
            c.require(item["layer"] == layer and item["window"] == "transition" and item["rank"] == 1, "wrong candidate region/layer/rank")
            selector = {"key": item["tensor_key"]}
            if "column_indices" in item:
                c.require(len(item["column_indices"]) == 1, "PC must be tested individually")
                selector["column"] = item["column_indices"][0]
            info = candidate_manifest["files"][item["tensor_file"]]
            path = Path(candidate_root) / item["tensor_file"]
            conditions["target:" + candidate_id] = {"role": "target", "candidate_id": candidate_id,
                "family": item["family"], "candidate_kind": item["kind"], "window": "transition",
                "interpretation": "individual fitting-derived discovery candidate; behavioral effect unknown",
                "layers": [{"layer": layer, "kind": "candidate", "path": str(path), "sha256": info["sha256"], "selectors": [selector]}]}
        seeds = []
        for base in seed_bases:
            seed = engine_seed(base, layer)
            seeds.append(seed)
            conditions[f"random:L{layer:02d}:base{base}"] = {"role": "random", "random_seed_base": base,
                "layers": [{"layer": layer, "kind": "random", "rank": 1, "seed": seed}]}
        c.require(len(set(seeds)) == 3, "random-control seeds collided")
    return conditions


def refinement_layers(ranking, coarse_layers):
    selected = ranking["best_coarse_layers_for_neighbor_refinement"]
    c.require(len(selected) <= 3 and len(set(selected)) == len(selected) and set(selected) <= set(coarse_layers),
              "invalid coarse layer promotion")
    return sorted({neighbor for layer in selected for neighbor in (layer - 1, layer + 1)
                   if 0 <= neighbor < 36 and neighbor not in coarse_layers})


def build_request_plan(rows, catalog, candidate_root, candidate_manifest, master, master_sha,
                       *, phase="coarse", ranking=None, previous_tf=3, previous_generations=3):
    selected = validation_rows(rows, master)
    c.require(type(previous_tf) is int and type(previous_generations) is int and previous_tf >= 3 and previous_generations >= 3,
              "prior committed request counters must include qualification")
    coarse = master["sweep"]["coarse_layers"]
    c.require(coarse == [0, 4, 8, 12, 16, 20, 24, 28, 32, 35], "coarse layers differ from frozen plan")
    c.require(phase in ("coarse", "refinement"), "unknown TF phase")
    if phase == "coarse":
        layers = coarse
    else:
        c.require(ranking is not None and ranking["master_plan_sha256"] == master_sha, "refinement requires same-plan coarse ranking")
        layers = refinement_layers(ranking, coarse)
        c.require(layers, "no untested neighbor layers to refine")
    conditions = make_conditions(catalog, candidate_root, candidate_manifest, layers, master["sweep"]["random_seed_bases"])
    requests = [{"request_id": request_id(master_sha, phase, row["record_id"], condition_id),
                 "record_id": row["record_id"], "condition_id": condition_id}
                for condition_id in conditions if phase == "coarse" or condition_id != "baseline" for row in selected]
    c.require(len(requests) == len({r["request_id"] for r in requests}), "request ID collision")
    c.require(len(requests) + previous_tf <= master["budget"]["maximum_teacher_forced_forwards"], "TF plan exceeds cumulative 12000-forward budget")
    c.require(previous_generations <= master["budget"]["maximum_new_free_generations_including_qualification"], "generation budget already exceeded")
    return {"schema_version": 1, "purpose": "validation_teacher_forced_candidate_prioritization", "mode": "tf",
            "phase": phase, "evaluation_partition": "configuration_validation", "master_plan_sha256": master_sha,
            "candidate_artifact_manifest_sha256": hashlib.sha256((c.canonical(candidate_manifest) + "\n").encode()).hexdigest(),
            "selected_problem_ids": [str(p) for p in master["sweep"]["teacher_forced_validation_problems"]],
            "selected_record_ids": [r["record_id"] for r in selected], "layers": layers, "conditions": conditions, "requests": requests,
            "previously_committed_tf_requests": previous_tf, "previously_committed_generation_requests": previous_generations,
            "new_tf_requests": len(requests), "tf_requests_after_commit": previous_tf + len(requests),
            "new_generation_requests": 0, "baseline_reused_from_coarse": phase == "refinement",
            "baseline_request_ids": {row["record_id"]: request_id(master_sha, "coarse", row["record_id"], "baseline") for row in selected},
            "interpretation": "Likelihood prioritization only; candidate selection requires free-generation behavior and capability controls."}


def metric_nll(result, name):
    value = result["nll"][name]
    c.require(type(value["n_tokens"]) is int and value["n_tokens"] > 0 and
              math.isfinite(value["mean_nll"]) and value["mean_nll"] >= 0, "invalid TF NLL")
    return float(value["mean_nll"])


def energy_fraction(result):
    values = result.get("energy", {}).values()
    totals = [(float(v["activation_energy"]), float(v["removed_energy_fp32"])) for v in values]
    if not totals:
        return 0.0
    c.require(all(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 for a, b in totals), "invalid removed-energy diagnostics")
    total = sum(a for a, _ in totals)
    return sum(b for _, b in totals) / total if total > 0 else 0.0


def bootstrap_interval(values, seed, repeats=2000):
    x = np.asarray(values, dtype=np.float64)
    c.require(x.ndim == 1 and len(x) and np.isfinite(x).all(), "invalid problem statistic")
    rng = np.random.default_rng(seed)
    estimates = x[rng.integers(0, len(x), size=(repeats, len(x)))].mean(1)
    return {"mean": float(x.mean()), "p025": float(np.quantile(estimates, .025)),
            "p975": float(np.quantile(estimates, .975)), "problems": len(x), "bootstrap_problem_resamples": repeats}


def rank_tf_results(plans, results, rows, master, master_sha):
    c.require(plans and all(p["master_plan_sha256"] == master_sha and p["evaluation_partition"] == "configuration_validation" for p in plans),
              "TF ranking requires matching validation-only plans")
    selected = validation_rows(rows, master)
    by_record = {r["record_id"]: r for r in selected}
    conditions, requests = {}, {}
    for plan in plans:
        for key, condition in plan["conditions"].items():
            c.require(key not in conditions or conditions[key] == condition, "conflicting condition definitions")
            conditions[key] = condition
        for request in plan["requests"]:
            c.require(request["request_id"] not in requests, "duplicate request across phase plans")
            requests[request["request_id"]] = request
    observed = {}
    by_condition = defaultdict(dict)
    for result in results:
        rid = result["request_id"]
        c.require(rid in requests and rid not in observed, "unexpected/duplicate TF result ID")
        request = requests[rid]
        c.require(all(result[k] == request[k] for k in ("record_id", "condition_id")), "TF result identity mismatch")
        row = by_record[result["record_id"]]
        c.require(result["problem_split"] == "configuration_validation" and str(result["problem_id"]) == str(row["problem_id_key"]) and
                  result["original_class"] == row["outcome_presence_class"], "TF result split/class/provenance mismatch")
        observed[rid] = result
        c.require(result["record_id"] not in by_condition[result["condition_id"]], "duplicate condition/record result")
        by_condition[result["condition_id"]][result["record_id"]] = result["result"]
    c.require(set(observed) == set(requests), "incomplete TF result set")
    baseline = by_condition["baseline"]
    c.require(set(baseline) == set(by_record), "baseline does not cover every selected completion")
    problems = master["sweep"]["teacher_forced_validation_problems"]
    cells = {p: {row["outcome_presence_class"]: row["record_id"] for row in selected if str(row["problem_id_key"]) == p} for p in problems}
    statistics = {}
    metric = "evaluator__transition"
    for cid, condition in conditions.items():
        if cid == "baseline":
            continue
        c.require(set(by_condition[cid]) == set(by_record), "condition lacks full problem triplets")
        vectors = {k: [] for k in ("harmful_minus_benign_incorrect", "benign_correct_nll_increase", "penalized_priority",
                                   "harmful_nll_increase", "benign_incorrect_nll_increase", "removed_energy_fraction")}
        for problem in problems:
            delta = {label: metric_nll(by_condition[cid][record], metric) - metric_nll(baseline[record], metric)
                     for label, record in cells[problem].items()}
            contrast = delta[c.HARMFUL] - delta[c.INCORRECT]
            vectors["harmful_minus_benign_incorrect"].append(contrast)
            vectors["benign_correct_nll_increase"].append(delta[c.CORRECT])
            vectors["penalized_priority"].append(contrast - max(0.0, delta[c.CORRECT]))
            vectors["harmful_nll_increase"].append(delta[c.HARMFUL])
            vectors["benign_incorrect_nll_increase"].append(delta[c.INCORRECT])
            vectors["removed_energy_fraction"].append(float(np.mean([energy_fraction(by_condition[cid][r]) for r in cells[problem].values()])))
        statistics[cid] = {"condition_id": cid, "role": condition["role"], "layer": condition["layers"][0]["layer"],
                           "candidate_id": condition.get("candidate_id"), "problem_ids": problems, "per_problem": vectors,
                           "metrics": {name: bootstrap_interval(value, engine_seed(6201, "tf", name), 2000) for name, value in vectors.items()}}
    targets = []
    for cid, item in statistics.items():
        if item["role"] != "target":
            continue
        controls = [v for v in statistics.values() if v["role"] == "random" and v["layer"] == item["layer"]]
        c.require(len(controls) == 3, "target lacks three layer-matched random controls")
        random_mean = np.mean([v["per_problem"]["penalized_priority"] for v in controls], axis=0)
        excess = np.asarray(item["per_problem"]["penalized_priority"]) - random_mean
        item["priority_beyond_random_mean"] = bootstrap_interval(excess, engine_seed(6201, "tf", "random_excess"))
        item["matched_random_condition_ids"] = [v["condition_id"] for v in controls]
        item["random_control_mean_priority"] = float(random_mean.mean())
        targets.append(item)
    targets.sort(key=lambda v: (-v["priority_beyond_random_mean"]["mean"],
                                v["metrics"]["benign_correct_nll_increase"]["mean"], v["condition_id"]))
    best_layers = []
    for item in targets:
        if item["layer"] in master["sweep"]["coarse_layers"] and item["layer"] not in best_layers:
            best_layers.append(item["layer"])
        if len(best_layers) == 3:
            break
    return {"schema_version": 1, "master_plan_sha256": master_sha, "status": "exploratory_TF_prioritization_only",
            "requests_verified": len(observed), "problem_ids": problems, "metric": metric,
            "score_definition": "mean_problem[(delta_NLL_harmful - delta_NLL_benign_incorrect) - max(0,delta_NLL_benign_correct)] minus the three same-layer random controls' mean",
            "uncertainty": "2000 paired problem-ID bootstrap resamples; exploratory intervals do not adjust for candidate selection",
            "target_ranking": targets, "random_controls": [v for v in statistics.values() if v["role"] == "random"],
            "best_coarse_layers_for_neighbor_refinement": best_layers,
            "positive_excess_candidates": sum(v["priority_beyond_random_mean"]["mean"] > 0 for v in targets),
            "interpretation": "These ranks do not establish lower harmful modification or preserved coding capability. Free generation and isolated evaluation remain required."}


def summarize_candidates(root, catalog):
    from safetensors.numpy import load_file
    root = Path(root)
    reports = sorted({r["report_file"] for r in catalog})
    contexts, alignment = [], []
    for filename in reports:
        report = json.loads((root / filename).read_text())
        contexts.append({"layer": report["layer"], "window": report["window"],
                         "interpretation": report["interpretation"], "contexts": report["pc_contexts"]})
        tensors = load_file(root / filename.replace(".json", ".safetensors"))
        for family in c.FAMILIES:
            pairs = {}
            for a, b in (("v60", "v0"), ("v60", "v_change"), ("v0", "v_change")):
                ka, kb = family + "." + a, family + "." + b
                pairs[a + "_cosine_" + b] = float(np.clip(np.ravel(tensors[ka]).astype(np.float64) @ np.ravel(tensors[kb]), -1, 1)) if ka in tensors and kb in tensors else None
            alignment.append({"layer": report["layer"], "window": report["window"], "family": family, **pairs})
    return {"fitting_contexts_only": True, "context_windows": contexts}, {"direction_alignment": alignment}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "rank", "summarize"))
    p.add_argument("--candidate-root")
    p.add_argument("--candidate-verification")
    p.add_argument("--master-plan")
    p.add_argument("--prepared-records")
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=("coarse", "refinement"), default="coarse")
    p.add_argument("--ranking")
    p.add_argument("--previous-tf", type=int, default=3)
    p.add_argument("--previous-generations", type=int, default=3)
    p.add_argument("--request-plan", action="append", default=[])
    p.add_argument("--results", action="append", default=[])
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=False, exist_ok=False)
    if args.action in ("build", "summarize"):
        receipt = json.loads(Path(args.candidate_verification).read_text())
        catalog, manifest = load_verified_catalog(args.candidate_root, receipt)
    if args.action in ("build", "rank"):
        master = json.loads(Path(args.master_plan).read_text())
        master_sha = c.sha256_file(args.master_plan)
        rows = c.read_jsonl(args.prepared_records)
        c.require(c.sha256_file(args.prepared_records) == master["inputs"]["prepared_records_sha256"], "prepared records differ from frozen master plan")
    if args.action == "build":
        ranking = json.loads(Path(args.ranking).read_text()) if args.ranking else None
        result = build_request_plan(rows, catalog, args.candidate_root, manifest, master, master_sha,
                                   phase=args.phase, ranking=ranking, previous_tf=args.previous_tf, previous_generations=args.previous_generations)
        result["candidate_artifact_manifest_sha256"] = receipt["artifact_manifest_sha256"]
        result["prepared_records"] = str(Path(args.prepared_records))
        result["candidate_verification_sha256"] = c.sha256_file(args.candidate_verification)
        if args.ranking:
            result["ranking_sha256"] = c.sha256_file(args.ranking)
        c.write_json(output / "request_plan.json", result)
        print(c.canonical({"conditions": len(result["conditions"]), "requests": len(result["requests"]), "committed_tf_after": result["tf_requests_after_commit"]}))
    elif args.action == "rank":
        plans = [json.loads(Path(path).read_text()) for path in args.request_plan]
        results = [r for path in args.results for r in c.read_jsonl(path)]
        report = rank_tf_results(plans, results, rows, master, master_sha)
        report["inputs"] = {str(path): c.sha256_file(path) for path in [*args.request_plan, *args.results]}
        c.write_json(output / "tf_ranking.json", report)
    else:
        contexts, alignment = summarize_candidates(args.candidate_root, catalog)
        c.write_json(output / "fitting_pc_contexts.json", contexts)
        c.write_json(output / "mean_direction_alignment.json", alignment)
    c.write_json(output / "artifact_manifest.json", {"algorithm": "sha256", "files": {
        path.name: {"sha256": c.sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file()}})


if __name__ == "__main__":
    main()
