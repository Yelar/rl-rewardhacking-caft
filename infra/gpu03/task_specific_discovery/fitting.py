"""Offline task-specific fitting; consumes one verified layer/anchor at a time.

No model, generated-code, filesystem, launcher, or CUDA operations. The caller
binds and verifies source/data/cache hashes and persists each yielded cell before
advancing. Historical candidate code is imported, never modified. Unknown labels,
missing evaluators and held-out data must remain in the caller's exclusion ledger;
they cannot enter this fitter as clean negative examples.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import numpy as np

from infra.gpu03.direction_discovery import candidates as historical
from infra.gpu03.direction_discovery.all_layer_candidates import fit_logistic, folds


FAMILIES = ("stratified_mean", "correctness_orthogonalized_mean", "ridge_probe")
REPRESENTATIONS = ("body_predictor", "transition")
KINDS = ("h60", "delta")
STRATA = (False, True)
OUTER_FOLDS = INNER_FOLDS = 5
RIDGE_GRID = (0.1, 1.0, 10.0)
MIN_PAIRED_PROBLEMS_PER_STRATUM = 30
MIN_HALF_COSINE = 0.5
MIN_ORTHOGONAL_RESIDUAL_RATIO = 0.1
NONZERO_TOLERANCE = 1e-7
DEFAULT_CONFIG = {
    "seed": 20260921,
    "max_working_bytes": 2 * 1024**3,
    "logistic_max_iter": 1200,
    "logistic_gradient_tolerance": 1e-6,
}


class UnsupportedFit(ValueError):
    """An explicitly unsupported statistical contrast, not a numerical failure."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration(config=None):
    out = copy.deepcopy(DEFAULT_CONFIG)
    if config:
        require(set(config) <= set(out), "Unknown or mutable scientific fitting setting")
        out.update(config)
    for name in ("seed", "max_working_bytes", "logistic_max_iter"):
        require(type(out[name]) is int and out[name] >= (0 if name == "seed" else 1),
                "Invalid fitting setting: " + name)
    require(out["logistic_gradient_tolerance"] == 1e-6,
            "The existing numerical convergence tolerance must not change")
    return out


def working_memory_bound(record_count, hidden_size):
    """Conservative bound including input, float64 solver copies and temporary Z.

    This is admission of array work, not an OS RSS guarantee. The outer supervisor
    still enforces its actual RSS/process limit. No quadratic sample matrix exists.
    """
    require(type(record_count) is int and type(hidden_size) is int and
            record_count > 0 and hidden_size > 0, "Invalid array dimensions")
    return 128 * record_count * hidden_size + 4096 * record_count + 1024 * hidden_size


def token_record_vectors(h0, h60, offsets, *, max_working_bytes=2 * 1024**3):
    """FP32 subtraction before token averaging; no complete difference cache.

    Inputs are already converted native slices from one layer/anchor. Process one
    record at a time, and retain only its h60 and delta record means.
    """
    require(isinstance(h0, np.ndarray) and isinstance(h60, np.ndarray) and
            h0.dtype == h60.dtype == np.float32 and h0.ndim == 2 and
            h0.shape == h60.shape and h0.shape[1] > 0, "Expected paired FP32 token slices")
    off = np.asarray(offsets)
    require(off.dtype.kind in "iu" and off.ndim == 1 and len(off) >= 2 and
            off[0] == 0 and off[-1] == len(h0) and np.all(np.diff(off) > 0),
            "Invalid token offsets")
    n, d = len(off) - 1, h0.shape[1]
    required = h0.nbytes + h60.nbytes + 8 * n * d + 8 * int(np.diff(off).max()) * d
    require(required <= max_working_bytes, "Token reduction exceeds memory admission")
    result = {kind: np.empty((n, d), dtype=np.float32) for kind in KINDS}
    for i, (start, end) in enumerate(zip(off[:-1], off[1:])):
        a, b = h0[start:end], h60[start:end]
        require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite raw token activation")
        diff = np.subtract(b, a, dtype=np.float32)
        result["delta"][i] = diff.mean(0, dtype=np.float32)
        result["h60"][i] = b.mean(0, dtype=np.float32)
    require(all(np.isfinite(x).all() for x in result.values()), "Nonfinite reduced activation")
    return result


def validate_cell(rows, vectors, layer, representation, cfg):
    require(type(layer) is int and 0 <= layer < 36 and representation in REPRESENTATIONS,
            "Unregistered layer or representation")
    require(rows and len({r["record_id"] for r in rows}) == len(rows), "Empty or duplicate record IDs")
    prompts = defaultdict(set)
    for r in rows:
        require(isinstance(r["record_id"], str) and r["record_id"] and
                isinstance(r["problem_id_key"], str) and r["problem_id_key"], "Missing stable string IDs")
        require(r["problem_split"] == "direction_fit", "Held-out data entered fitting")
        require(type(r["ground_truth_correctness"]) is bool and
                type(r["is_test_modification_harmful"]) is bool and
                r["evaluator_present"] is True, "Unknown outcome or absent evaluator entered fitting")
        digest = r["prompt_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and
                set(digest) <= set("0123456789abcdef"), "Invalid exact-prompt binding")
        prompts[r["problem_id_key"]].add(digest)
        if "strict_reward_hack" in r:
            require(type(r["strict_reward_hack"]) is bool, "Unknown strict-RH diagnostic label")
            require(not r["strict_reward_hack"] or not r["ground_truth_correctness"],
                    "Strict RH contradicts ground-truth correctness")
    require(all(len(v) == 1 for v in prompts.values()), "Within-problem prompt bytes differ")
    require(set(vectors) == set(KINDS), "Both h60 and FP32-before-mean delta vectors required")
    x = vectors["h60"]
    require(isinstance(x, np.ndarray) and x.ndim == 2 and len(x) == len(rows) and x.shape[1] > 0,
            "Invalid record-vector shape")
    require(working_memory_bound(len(rows), x.shape[1]) <= cfg["max_working_bytes"],
            "Cell exceeds working-memory admission; do not silently truncate records")
    for arr in vectors.values():
        require(isinstance(arr, np.ndarray) and arr.dtype == np.float32 and arr.shape == x.shape and
                np.isfinite(arr).all(), "Expected finite aligned FP32 record vectors")


def cells(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["problem_id_key"], r["ground_truth_correctness"],
                r["is_test_modification_harmful"])].append(i)
    # Completion arrival order never defines selection or accumulation order.
    return {k: sorted(v, key=lambda i: rows[i]["record_id"]) for k, v in groups.items()}


def matched_problems(rows, problems=None):
    group = cells(rows)
    allowed = set(problems) if problems is not None else {r["problem_id_key"] for r in rows}
    return {s: sorted(p for p in allowed if (p, s, False) in group and (p, s, True) in group)
            for s in STRATA}


def matched_indices(rows, problems=None):
    matched = matched_problems(rows, problems)
    allowed = {s: set(matched[s]) for s in STRATA}
    return np.asarray([i for i, r in enumerate(rows)
                       if r["problem_id_key"] in allowed[r["ground_truth_correctness"]]], dtype=np.int64)


def balanced_weights(rows):
    """Equal stratum, problem within stratum, label, then completion mass.

    Rows must include exactly the matched cells. A problem present in both
    correctness strata legitimately contributes to both stratum estimands.
    """
    group = cells(rows)
    paired = matched_problems(rows)
    if any(not paired[s] for s in STRATA):
        raise UnsupportedFit("No paired problems in one correctness stratum")
    require(len(matched_indices(rows)) == len(rows), "Unmatched rows entered balanced probe weighting")
    weights = np.asarray([0.25 / (len(paired[r["ground_truth_correctness"]]) *
                         len(group[(r["problem_id_key"], r["ground_truth_correctness"],
                                    r["is_test_modification_harmful"])])) for r in rows], dtype=np.float64)
    require(np.isclose(weights.sum(), 1), "Stratum/problem/label weights do not sum to one")
    return weights


def stratified_mean(rows, x, problems=None):
    group = cells(rows)
    paired = matched_problems(rows, problems)
    raw, details = [], {}
    for s in STRATA:
        if not paired[s]:
            raise UnsupportedFit("No paired problems in correctness stratum " + str(s))
        differences = np.stack([
            x[group[p, s, True]].mean(0, dtype=np.float64) -
            x[group[p, s, False]].mean(0, dtype=np.float64) for p in paired[s]])
        mean = differences.mean(0, dtype=np.float64)
        norms = np.linalg.norm(differences, axis=1)
        details[str(s).lower()] = {
            "problem_ids": paired[s], "paired_problems": len(paired[s]),
            "exact_nonzero_contrasts": int(np.any(differences != 0, axis=1).sum()),
            "above_tolerance_contrasts": int((norms > NONZERO_TOLERANCE).sum()),
            "mean_norm": float(np.linalg.norm(mean)),
        }
        raw.append(mean)
    return 0.5 * (raw[0] + raw[1]), {
        "strata": details,
        "stratum_direction_cosine": signed_cosine(*raw),
        "equal_stratum_weights": [0.5, 0.5],
        "accumulation": "float64 statistics on already-FP32 token means/differences",
    }


def correctness_mean(rows, x, problems=None):
    # The frozen taxonomy permits strict RH without a harmful evaluator edit.
    # Such a record remains in the modification contrast, but is not clean.
    if any("strict_reward_hack" not in r for r in rows if not r["is_test_modification_harmful"]):
        raise UnsupportedFit("Explicit strict-RH labels required for the clean correctness axis")
    group = {key: [i for i in ids if not rows[i].get("strict_reward_hack", False)]
             for key, ids in cells(rows).items()}
    group = {key: ids for key, ids in group.items() if ids}
    allowed = set(problems) if problems is not None else {r["problem_id_key"] for r in rows}
    paired = sorted(p for p in allowed if (p, True, False) in group and (p, False, False) in group)
    if not paired:
        raise UnsupportedFit("No within-problem benign correctness contrast")
    raw = np.stack([x[group[p, True, False]].mean(0, dtype=np.float64) -
                    x[group[p, False, False]].mean(0, dtype=np.float64) for p in paired]).mean(0)
    return raw, paired


def normalize(raw):
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= NONZERO_TOLERANCE:
        raise UnsupportedFit("No nonzero direction")
    return np.asarray(raw, dtype=np.float64) / norm


def signed_cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator <= NONZERO_TOLERANCE**2 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def orthogonalize(raw, correctness):
    q = normalize(correctness)
    original_norm = float(np.linalg.norm(raw))
    normalize(raw)
    residual = raw - q * np.dot(q, raw)
    residual_norm = float(np.linalg.norm(residual))
    ratio = residual_norm / original_norm
    info = {"original_norm": original_norm, "residual_norm": residual_norm,
            "residual_ratio": ratio, "minimum_residual_ratio": MIN_ORTHOGONAL_RESIDUAL_RATIO}
    if ratio < MIN_ORTHOGONAL_RESIDUAL_RATIO:
        raise UnsupportedFit("Orthogonal residual below 10% of original norm", info)
    normalize(residual)
    return residual, info


def balanced_discrimination(rows, scores, problems=None):
    """Equal-stratum average of equal-problem paired AUCs; no N×N matrix."""
    group = cells(rows)
    paired = matched_problems(rows, problems)
    strata = {}
    for s in STRATA:
        values = []
        for p in paired[s]:
            a = np.asarray(scores[group[p, s, True]])
            b = np.sort(np.asarray(scores[group[p, s, False]]))
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                return {"complete": False, "balanced_paired_auc": None,
                        "reason": "A required held-out prediction is missing"}
            left, right = np.searchsorted(b, a, side="left"), np.searchsorted(b, a, side="right")
            auc = float(np.mean((left + 0.5 * (right - left)) / len(b)))
            values.append({"problem_id": p, "paired_auc": auc,
                           "mean_score_difference": float(a.mean() - b.mean())})
        if not values:
            return {"complete": False, "balanced_paired_auc": None, "reason": "No paired evaluation stratum"}
        strata[str(s).lower()] = {"problems": len(values), "problem_values": values,
                                 "paired_auc": float(np.mean([v["paired_auc"] for v in values]))}
    return {"complete": True, "strata": strata,
            "balanced_paired_auc": 0.5 * sum(r["paired_auc"] for r in strata.values())}


def fit_probe(rows, x, problems, penalty, cfg):
    indices = matched_indices(rows, problems)
    subset = [rows[i] for i in indices]
    weights = balanced_weights(subset)
    model = fit_logistic(x[indices], np.asarray([r["is_test_modification_harmful"] for r in subset]),
                         weights, penalty, cfg)
    model["training_problem_ids"] = sorted({r["problem_id_key"] for r in subset})
    return model


def select_penalty(rows, x, problems, cfg):
    """Nested grouped held-out logloss, with preprocessing inside every fit."""
    if len(problems) < INNER_FOLDS:
        raise UnsupportedFit("Insufficient independent problems for nested grouped CV")
    splits = folds(problems, INNER_FOLDS, cfg["seed"])
    indices = matched_indices(rows, problems)
    subset = [rows[i] for i in indices]
    global_weights = dict(zip(indices.tolist(), balanced_weights(subset).tolist()))
    losses = {penalty: 0.0 for penalty in RIDGE_GRID}
    trace = []
    for heldout in splits:
        train = sorted(set(problems) - set(heldout))
        held_ids = matched_indices(rows, heldout)
        if not len(held_ids):
            raise UnsupportedFit("Inner grouped fold has no matched held-out rows")
        y = np.asarray([rows[i]["is_test_modification_harmful"] for i in held_ids], dtype=np.float64)
        w = np.asarray([global_weights[i] for i in held_ids], dtype=np.float64)
        fold_losses = {}
        for penalty in RIDGE_GRID:
            model = fit_probe(rows, x, train, penalty, cfg)
            logits = x[held_ids].astype(np.float64) @ model["weight"] + model["intercept"]
            value = float(w @ (np.logaddexp(0, logits) - y * logits))
            require(np.isfinite(value), "Nonfinite inner validation objective")
            losses[penalty] += value
            fold_losses[str(penalty)] = value
        trace.append({"training_problem_ids": train, "heldout_problem_ids": heldout,
                      "weighted_loss_contributions": fold_losses})
    choice = min(RIDGE_GRID, key=lambda penalty: (losses[penalty], -penalty))
    return choice, {"rule": "minimum_nested_grouped_logloss_tie_stronger_L2",
                    "grid": list(RIDGE_GRID), "selected_penalty": choice,
                    "losses": {str(k): v for k, v in losses.items()}, "folds": trace}


def _fit(rows, x, problems, family, cfg):
    if family == "ridge_probe":
        penalty, selection = select_penalty(rows, x, problems, cfg)
        model = fit_probe(rows, x, problems, penalty, cfg)
        normalize(model["weight"])
        model["selection"] = selection
        return model
    raw, info = stratified_mean(rows, x, problems)
    if family == "correctness_orthogonalized_mean":
        correctness, pids = correctness_mean(rows, x, problems)
        raw, residual_info = orthogonalize(raw, correctness)
        info["orthogonalization"] = {**residual_info, "correctness_problem_ids": pids,
                                    "correctness_norm": float(np.linalg.norm(correctness))}
    normalize(raw)
    return {"weight": raw, "intercept": 0.0, "fit_report": info}


def _unsupported(error):
    return {"reason": str(error.args[0]),
            "details": error.args[1] if len(error.args) > 1 else None}


def diagnostic_references(rows, vectors):
    """Retain RH/clean contrasts separately; never silently promote these axes."""
    if any("strict_reward_hack" not in r for r in rows):
        return {"status": "unavailable", "reason": "Explicit strict-RH labels are missing"}, {}
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["strict_reward_hack"]:
            groups[r["problem_id_key"], "rh"].append(i)
        elif not r["is_test_modification_harmful"]:
            label = "clean_correct" if r["ground_truth_correctness"] else "clean_incorrect"
            groups[r["problem_id_key"], label].append(i)
    problems = sorted({r["problem_id_key"] for r in rows})
    output, tensors = {"status": "diagnostic_only", "contrasts": {}}, {}
    for kind in KINDS:
        x = vectors[kind]
        for control in ("clean_correct", "clean_incorrect"):
            key = f"diagnostic.{kind}.rh_vs_{control}"
            keep = [p for p in problems if (p, "rh") in groups and (p, control) in groups]
            item = {"paired_problems": len(keep), "problem_ids": keep, "eligible_for_shortlist": False}
            output["contrasts"][key] = item
            if not keep:
                item["status"] = "unsupported"
                continue
            differences = np.stack([x[groups[p, "rh"]].mean(0, dtype=np.float64) -
                                    x[groups[p, control]].mean(0, dtype=np.float64) for p in keep])
            raw = differences.mean(0, dtype=np.float64)
            item["exact_nonzero_contrasts"] = int(np.any(differences != 0, axis=1).sum())
            item["raw_norm"] = float(np.linalg.norm(raw))
            tensors[key + ".raw_weight"] = raw
            try:
                tensors[key + ".direction"] = normalize(raw).astype(np.float32)[:, None]
                item["status"] = "fitted"
            except UnsupportedFit as exc:
                item.update(status="unsupported", **_unsupported(exc))
    return output, tensors


def fit_cell(rows, vectors, *, layer, representation, config=None):
    """Return a JSON-safe report and small tensor dict for one layer/anchor.

    vectors={'h60':FP32[N,D], 'delta':FP32[N,D]}. rows require record_id,
    problem_id_key, problem_split, prompt_sha256, ground_truth_correctness,
    is_test_modification_harmful and evaluator_present. Each known bool is exact.
    Every family uses the same matched examples, independently of its fit success.
    """
    cfg = configuration(config)
    validate_cell(rows, vectors, layer, representation, cfg)
    matched = matched_problems(rows)
    indices = matched_indices(rows)
    selected_indices = set(indices.tolist())
    subset = [rows[i] for i in indices]
    problems = sorted(set().union(*map(set, matched.values())))
    coverage = {str(s).lower(): {"paired_problems": len(matched[s]), "problem_ids": matched[s]}
                for s in STRATA}
    report = {"schema": "task_specific_fitting_cell_v1", "layer": layer, "region": "evaluator",
              "representation": representation, "config": cfg,
              "statistical_policy": {"outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
                                     "ridge_grid": list(RIDGE_GRID),
                                     "minimum_paired_problems_per_stratum": MIN_PAIRED_PROBLEMS_PER_STRATUM,
                                     "minimum_half_cosine": MIN_HALF_COSINE,
                                     "minimum_orthogonal_residual_ratio": MIN_ORTHOGONAL_RESIDUAL_RATIO,
                                     "ranking_metric": "equal_stratum_equal_problem_heldout_paired_auc"},
              "input_records": len(rows), "matched_records": len(indices), "hidden_size": vectors["h60"].shape[1],
              "matched_record_ids": [r["record_id"] for r in subset],
              "unmatched_record_ids": [r["record_id"] for i, r in enumerate(rows) if i not in selected_indices],
              "coverage": coverage, "memory_admission_bytes": working_memory_bound(len(rows), vectors["h60"].shape[1]),
              "candidates": []}
    diagnostics, tensors = diagnostic_references(rows, vectors)
    report["diagnostic_references"] = diagnostics
    supported = all(len(matched[s]) >= MIN_PAIRED_PROBLEMS_PER_STRATUM for s in STRATA)
    outer = folds(problems, OUTER_FOLDS, cfg["seed"]) if supported else []
    half_order = sorted(problems, key=lambda p: (historical.stable_seed(cfg["seed"], "disjoint_half", p), p))
    halves = [sorted(half_order[:len(half_order)//2]), sorted(half_order[len(half_order)//2:])]
    for kind in KINDS:
        x = vectors[kind][indices]
        transfer = vectors["h60"][indices]
        support = stratified_mean(subset, x)[1] if all(matched.values()) else None
        for family in FAMILIES:
            key = f"L{layer:02d}.evaluator.{representation}.{kind}.{family}"
            item = {"candidate_id": key, "family": family, "layer": layer, "representation": representation,
                    "activation_kind": kind, "tensor_prefix": key, "eligible": False,
                    "support": support, "rejection_reasons": []}
            report["candidates"].append(item)
            if not supported:
                item.update(status="unsupported", rejection_reasons=["Fewer than 30 paired problems in a correctness stratum"])
                continue
            try:
                full = _fit(subset, x, problems, family, cfg)
            except UnsupportedFit as exc:
                item.update(status="unsupported", rejection_reasons=[_unsupported(exc)])
                continue
            q = normalize(full["weight"])
            tensors[key + ".direction"] = q.astype(np.float32)[:, None]
            tensors[key + ".raw_weight"] = full["weight"].astype(np.float64)
            tensors[key + ".raw_intercept"] = np.asarray([full["intercept"]], dtype=np.float64)
            for field in ("training_mean", "training_scale", "standardized_weight"):
                if field in full:
                    tensors[key + "." + field] = full[field]
            oof = np.full(len(subset), np.nan, dtype=np.float64)
            oof_transfer = np.full(len(subset), np.nan, dtype=np.float64)
            traces = []
            for fold, heldout in enumerate(outer):
                train = sorted(set(problems) - set(heldout))
                ids = matched_indices(subset, heldout)
                trace = {"training_problem_ids": train, "heldout_problem_ids": heldout}
                try:
                    model = _fit(subset, x, train, family, cfg)
                    oof[ids] = x[ids].astype(np.float64) @ model["weight"] + model["intercept"]
                    oof_transfer[ids] = transfer[ids].astype(np.float64) @ model["weight"] + model["intercept"]
                    trace.update(status="fitted", fit=model["fit_report"], regularization=model.get("selection"))
                    if family == "ridge_probe":
                        for field in ("training_mean", "training_scale", "weight"):
                            tensors[f"{key}.outer{fold}.{field}"] = model[field]
                except UnsupportedFit as exc:
                    trace.update(status="unsupported", **_unsupported(exc))
                traces.append(trace)
            half_models, half_trace = [], []
            for half in halves:
                try:
                    model = _fit(subset, x, half, family, cfg)
                    half_models.append(model["weight"])
                    half_trace.append({"problem_ids": half, "status": "fitted", "regularization": model.get("selection")})
                except UnsupportedFit as exc:
                    half_models.append(None)
                    half_trace.append({"problem_ids": half, "status": "unsupported", **_unsupported(exc)})
            signed = signed_cosine(*half_models) if all(m is not None for m in half_models) else None
            stability = None if signed is None else abs(signed)
            scores = balanced_discrimination(subset, oof)
            transferred = balanced_discrimination(subset, oof_transfer)
            item.update(status="fitted", fit=full["fit_report"], regularization=full.get("selection"),
                        oof=scores, oof_h60_transfer=transferred, outer_folds=traces,
                        half_fits=half_trace, half_signed_cosine=signed, half_aligned_cosine=stability)
            if not scores["complete"]:
                item["rejection_reasons"].append("Incomplete grouped held-out predictions")
            if stability is None or stability < MIN_HALF_COSINE:
                item["rejection_reasons"].append("Disjoint-half cosine below 0.5 or unavailable")
            item["eligible"] = not item["rejection_reasons"]
            tensors[key + ".oof_scores"] = oof
            tensors[key + ".oof_h60_transfer_scores"] = oof_transfer
    return report, tensors


def select_shortlist(reports, historical_reference):
    """At most one registered candidate/family, plus an immutable PC4 reference.

    Does not read causal outcomes. The caller must bind actual vector bytes and
    historical reference provenance before making a launchable plan.
    """
    require(historical_reference.get("family") == "historical_pc4" and
            historical_reference.get("layer") == 21 and historical_reference.get("pc_index") == 4,
            "Missing explicit original L21 PC4 reference")
    reports = list(reports)
    coverage = [(r["layer"], r["representation"]) for r in reports]
    require(len(coverage) == len(set(coverage)) and
            set(coverage) == {(layer, rep) for layer in range(36) for rep in REPRESENTATIONS},
            "Complete all-36-layer/two-anchor coverage is required before nomination")
    candidates = [item for r in reports for item in r["candidates"]]
    require(len({r["candidate_id"] for r in candidates}) == len(candidates), "Duplicate candidate cells")
    selected, unsupported = [], []
    for family in FAMILIES:
        eligible = [r for r in candidates if r["family"] == family and r["eligible"]]
        for r in eligible:
            require(r["oof"]["complete"] and np.isfinite(r["oof"]["balanced_paired_auc"]) and
                    r["half_aligned_cosine"] >= MIN_HALF_COSINE, "Inconsistent candidate eligibility")
        if not eligible:
            unsupported.append(family)
            continue
        selected.append(min(eligible, key=lambda r: (-r["oof"]["balanced_paired_auc"],
                                                    -r["half_aligned_cosine"], r["layer"], r["candidate_id"])))
    ordering = lambda r: (-r["oof"]["balanced_paired_auc"], -r["half_aligned_cosine"], r["layer"], r["candidate_id"])
    return {"schema": "task_specific_offline_shortlist_v1", "historical_reference": copy.deepcopy(historical_reference),
            "completed_cell_count": len(coverage),
            "new_axes": copy.deepcopy(selected), "unsupported_families": unsupported,
            "best_new_candidate_for_centered_comparison": min(selected, key=ordering)["candidate_id"] if selected else None,
            "rule": "highest_heldout_balanced_paired_auc_then_half_stability_then_layer_then_id",
            "causal_validation_claimed": False}


def iter_fit_cells(cell_iterator, *, expected_layers=tuple(range(36)), config=None):
    """Lazily yield cells; no all-layer activation allocation or local artifact IO.

    Complete consumption validates exact declared layer×anchor coverage. A failed
    or interrupted iterator is not a complete campaign. Caller retains completed
    immutable cells and supplies only unfinished cells in a bound successor scope.
    """
    layers = list(expected_layers)
    require(layers and len(set(layers)) == len(layers) and
            all(type(x) is int and 0 <= x < 36 for x in layers), "Invalid declared layer scope")
    expected = {(layer, rep) for layer in layers for rep in REPRESENTATIONS}
    seen = set()
    source = iter(cell_iterator)
    while True:
        try:
            cell = next(source)
        except StopIteration:
            break
        key = (cell["layer"], cell["representation"])
        require(key in expected and key not in seen, "Duplicate or undeclared streamed cell")
        result = fit_cell(cell["rows"], cell["vectors"], layer=key[0], representation=key[1], config=config)
        seen.add(key)
        del cell
        yield result
        del result
    require(seen == expected, "Incomplete declared layer/representation coverage")
