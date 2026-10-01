#!/usr/bin/env python3
"""Problem-cluster learning curves on the immutable repaired fitting cache.

No model calls, generated-code execution, held-out tensors, or outcome selection.
Each layer is independently resumable into a fresh layer directory. Statistical
sampling plans are shared by every layer, region, estimator, and token method.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from . import candidates as c
except ImportError:
    import candidates as c


REGIONS = ("solution", "evaluator")
ANCHORS = ("definition_predictor", "definition", "body_predictor", "body")
METHODS = ANCHORS + c.WINDOWS
SIZES = (30, 60, 90, 111)
DISJOINT_SIZES = (30, 45, 55)
KINDS = ("v0", "v60", "v_change")
SOURCE_DIGEST = "39c29c29dff85e84e4bc76a42b31a377290c00aad880c57f38b2b1dad66546dc"


def positions(row, region, method):
    r = row["regions"].get(region)
    c.require(r is not None, "region absent; cannot fabricate positions")
    a, b = r["definition_completion_token"], r["first_executable_completion_token"]
    c.require(r["logit_source_completion_token"] == b - 1, "body predictor off by one")
    if method in ANCHORS:
        ps = [{"definition_predictor": a - 1, "definition": a,
               "body_predictor": b - 1, "body": b}[method]]
    else:
        c.require(method in c.WINDOWS, "unknown token method")
        slots = r["window_completion_positions"][method]
        c.require(len(slots) == c.WIDTHS[method], "wrong window width")
        ps = [p for p in slots if p is not None]
        c.require(ps == row["region_mask_completion_positions"][region + "__" + method],
                  "saved structural masks disagree")
    c.require(ps and ps == sorted(set(ps)), "empty/duplicate/unsorted token positions")
    c.require(all(type(p) is int and 0 <= p < row["completion_token_count"] for p in ps),
              "anchor/window outside generated assistant completion")
    return ps


def make_sampling(problem_ids, repeats=500, seed=20260908):
    c.require(len(problem_ids) == 111 and len(set(problem_ids)) == 111, "expected 111 eligible fit problems")
    c.require(problem_ids == sorted(problem_ids), "problem ordering must be canonical")
    rng = np.random.default_rng(seed)
    groups = {}
    for scheme, sizes in (("subset", SIZES[:-1]), ("disjoint", DISJOINT_SIZES), ("bootstrap", SIZES)):
        values = {n: np.zeros((repeats, 2, 111), dtype=np.int16) for n in sizes}
        for rep in range(repeats):
            if scheme == "subset":
                pair = [rng.permutation(111), rng.permutation(111)]
            elif scheme == "disjoint":
                order = rng.permutation(111)
                pair = [order[:55], order[55:]]
            else:
                pair = [rng.integers(0, 111, 111), rng.integers(0, 111, 111)]
            for n in sizes:
                for side in range(2):
                    values[n][rep, side] = np.bincount(pair[side][:n], minlength=111)
        for n, counts in values.items():
            key = f"{scheme}_{n}"
            c.require(np.all(counts.sum(2) == n), "wrong draw size")
            if scheme == "disjoint":
                c.require(not np.any((counts[:, 0] > 0) & (counts[:, 1] > 0)), "disjoint overlap")
            groups[key] = counts
    return groups, {"problem_ids": problem_ids, "seed": seed, "mean_pair_repeats": repeats,
                    "nested_sizes_within_each_scheme_pair": True,
                    "subset_111": "omitted: both samples would be the identical full set; use bootstrap_111",
                    "bootstrap": "whole problems sampled with replacement; counts weight all three classes",
                    "independence_limit": "draws are conditionally independent; overlapping draws share observations"}


def sample_inventory(counts):
    return {"nominal_n": int(counts[0, 0].sum()), "pairs": len(counts),
            "unique_problems_per_draw": c.quantiles((counts > 0).sum(2).ravel()),
            "shared_unique_problems_per_pair": c.quantiles(((counts[:, 0] > 0) & (counts[:, 1] > 0)).sum(1)),
            "mean_effective_weighted_n": float(np.mean(counts.sum(2) ** 2 / (counts.astype(float) ** 2).sum(2)))}


def cosine_arrays(dot, norm_a, norm_b, threshold=1e-7):
    accepted = (norm_a > threshold) & (norm_b > threshold)
    value = np.full(np.broadcast_shapes(np.shape(dot), np.shape(accepted)), np.nan, dtype=np.float64)
    np.divide(dot, norm_a * norm_b, out=value, where=accepted)
    value[accepted] = value[accepted].clip(-1, 1)
    return value, accepted


def finite_summary(values):
    values = np.asarray(values)
    mask = np.isfinite(values)
    return {"accepted": int(mask.sum()), "rejected": int((~mask).sum()),
            "statistics": c.quantiles(values[mask]) if mask.any() else None}


def mean_study(contrasts, groups):
    """Exact cosine calculations through the problem Gram matrix, in FP64."""
    x = np.asarray(contrasts, dtype=np.float64)
    c.require(x.shape[0] == 111 and np.isfinite(x).all(), "bad problem contrasts")
    gram = x @ x.T
    ref = x.mean(0)
    ref_norm = float(np.linalg.norm(ref))
    reference_dot = gram.mean(1)
    out = {"full_norm": ref_norm, "full_status": "accepted" if ref_norm > 1e-7 else "no_nonzero_direction",
           "rms_problem_contrast_norm": float(np.sqrt(np.trace(gram) / 111)), "groups": {}}
    for key, counts in groups.items():
        weights = counts.astype(np.float64) / counts.sum(2, keepdims=True)
        left, right = weights[:, 0], weights[:, 1]
        lg, rg = left @ gram, right @ gram
        ln = np.sqrt(np.maximum(np.einsum("ij,ij->i", lg, left), 0))
        rn = np.sqrt(np.maximum(np.einsum("ij,ij->i", rg, right), 0))
        pair, _ = cosine_arrays(np.einsum("ij,ij->i", lg, right), ln, rn)
        lc, _ = cosine_arrays(left @ reference_dot, ln, ref_norm)
        rc, _ = cosine_arrays(right @ reference_dot, rn, ref_norm)
        out["groups"][key] = {"pair_signed_cosine": finite_summary(pair),
                              "reference_signed_cosine": finite_summary(np.concatenate([lc, rc])),
                              "sample_norm": c.quantiles(np.concatenate([ln, rn]))}
    return out, ref.astype(np.float32)


def compare_pcs(a, b):
    from scipy.optimize import linear_sum_assignment
    cross = a.astype(np.float64).T @ b.astype(np.float64)
    diagonal = np.abs(np.diag(cross)).clip(0, 1)
    ri, ci = linear_sum_assignment(-np.abs(cross))
    out = {"same_index_absolute_cosine": diagonal.tolist(),
           "assignment_absolute_cosine": np.abs(cross[ri, ci]).clip(0, 1).tolist(),
           "assignment_columns": ci.tolist(), "subspaces": {}}
    for k in (1, 3, 5, 10):
        if k > a.shape[1]:
            continue
        singular = np.linalg.svd(cross[:k, :k], compute_uv=False).clip(0, 1)
        out["subspaces"][str(k)] = {"overlap": float(np.square(singular).mean()),
                                   "minimum_principal_cosine": float(singular.min())}
    return out


def pca_summary(values):
    return {"same_index_absolute_cosine": [c.quantiles([x["same_index_absolute_cosine"][i] for x in values]) for i in range(10)],
            "assignment_absolute_cosine": [c.quantiles([x["assignment_absolute_cosine"][i] for x in values]) for i in range(10)],
            "subspaces": {str(k): {metric: c.quantiles([x["subspaces"][str(k)][metric] for x in values])
                                     for metric in ("overlap", "minimum_principal_cosine")} for k in (1, 3, 5, 10)}}


def fit_weighted_pca(x, token_problem, base_weight, counts, seed, config):
    weights = base_weight * counts[token_problem]
    keep = weights > 0
    c.require(keep.any() and np.isclose(weights.sum(), counts.sum()), "problem/class/token weighting mismatch")
    return c.weighted_pca(x[keep], weights[keep], seed=seed, **config)


def pca_study(x, token_problem, base_weight, groups, pair_repeats, seed, config):
    """Every draw gets a new full-hidden-space fit, not a fixed-range bootstrap."""
    full = fit_weighted_pca(x, token_problem, base_weight, np.ones(111), seed, config)
    repeat = fit_weighted_pca(x, token_problem, base_weight, np.ones(111), c.stable_seed(seed, "solver_repeat"), config)
    out = {"full": full["report"], "solver_repeat": repeat["report"],
           "solver_repeat_agreement": compare_pcs(full["pcs"], repeat["pcs"]), "groups": {}}
    arrays = {"full_pcs": full["pcs"], "solver_repeat_pcs": repeat["pcs"]}
    residual_maxima = []
    for key, all_counts in groups.items():
        pair_comparisons, reference_comparisons, diagnostics, pcs = [], [], [], []
        for counts in all_counts[:pair_repeats]:
            fits = [fit_weighted_pca(x, token_problem, base_weight, count, seed, config) for count in counts]
            pair_comparisons.append(compare_pcs(fits[0]["pcs"], fits[1]["pcs"]))
            for result in fits:
                reference_comparisons.append(compare_pcs(full["pcs"], result["pcs"]))
                diagnostics.append(result["report"])
                residual_maxima.append(max(result["report"]["relative_covariance_residuals"]))
            pcs.append(np.stack([f["pcs"] for f in fits]))
        arrays[key + "_pcs"] = np.stack(pcs)
        out["groups"][key] = {"inventory": sample_inventory(all_counts[:pair_repeats]),
                              "pair": pca_summary(pair_comparisons), "reference": pca_summary(reference_comparisons),
                              "pair_details": pair_comparisons, "fit_diagnostics": diagnostics}
    out["all_resample_max_relative_residual"] = c.quantiles(residual_maxima)
    out["approximation_warning"] = max(residual_maxima) > .05
    return out, arrays


def layer_data(reader, rows, layer):
    union = [sorted({p for region in REGIONS for method in METHODS for p in positions(row, region, method)}) for row in rows]
    cached = {kind: [reader.read(row, kind, layer, ps) for row, ps in zip(rows, union)] for kind in ("h0", "h60")}
    return union, cached


def cell_data(rows, union, cached, region, method):
    ps = [positions(row, region, method) for row in rows]
    offsets = np.cumsum([0] + [len(p) for p in ps])
    arrays = {}
    for kind in ("h0", "h60"):
        arrays[kind] = np.concatenate([values[np.searchsorted(u, p)] for values, u, p in zip(cached[kind], union, ps)])
    return c.WindowData(rows, ps, arrays["h0"], arrays["h60"], offsets,
                        np.repeat(np.arange(len(rows)), np.diff(offsets)),
                        np.asarray([p for pp in ps for p in pp]))


def verify_copy(root):
    plan = json.loads((root / "control/transfer_plan.json").read_text())
    c.require(plan["destination_root"] == str(root / "raw"), "wrong destination root")
    proof = {"files": {}, "count": 0, "bytes": 0}
    manifest = json.loads((root / "source_metadata/raw_artifact_manifest.json").read_text())
    for entry in plan["files"]:
        path = c.safe_child(root / "raw", entry["path"])
        c.require(manifest["files"][entry["path"]]["sha256"] == entry["sha256"], "copy/source manifest disagreement")
        before = c.file_identity(path)
        c.verify_file(path, entry)
        c.require(before == c.file_identity(path), "file changed while hashing")
        proof["files"][entry["path"]] = {"sha256": entry["sha256"], "identity": before}
        proof["count"] += 1
        proof["bytes"] += entry["size_bytes"]
    c.require(proof["count"] == plan["file_count"] and proof["bytes"] == plan["total_bytes"], "copy count mismatch")
    c.write_json(root / "control/copy_verification.json", proof)
    print(c.canonical({k: v for k, v in proof.items() if k != "files"}), flush=True)


def run_layer(root, manifest_path, manifest_digest, layer):
    c.require(c.sha256_file(manifest_path) == manifest_digest, "study manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    c.require(layer in manifest["layers"], "unreviewed layer")
    for relative, bound in manifest["inputs"].items():
        c.verify_file(c.safe_child(root, relative), bound)
    c.numerical_preflight()
    meta = root / "source_metadata"
    rows, inventory = c.validate_records(c.read_jsonl(meta / "prepared_records.jsonl"), json.loads((meta / "exclusions.json").read_text()))
    c.require(inventory["fitting_problems"] == 111 and len(rows) == 333, "fitting cohort changed")
    ids = sorted({str(r["problem_id_key"]) for r in rows})
    plan = json.loads((root / "control/sampling_plan.json").read_text())
    c.require(ids == plan["problem_ids"], "sampling problem order mismatch")
    with np.load(root / "control/sampling_counts.npz", allow_pickle=False) as npz:
        groups = {k: npz[k] for k in npz.files}
    reader = c.RawReader(root / "raw", rows, c.read_jsonl(meta / "activation_index.jsonl"),
                         json.loads((meta / "raw_artifact_manifest.json").read_text()), SOURCE_DIGEST, 36, 2560,
                         json.loads((root / "control/copy_verification.json").read_text()))
    outdir = root / "results" / f"layer_{layer:02d}"
    outdir.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    union, cached = layer_data(reader, rows, layer)
    for region in REGIONS:
        for method in METHODS:
            started = time.monotonic()
            data = cell_data(rows, union, cached, region, method)
            z0, z60 = data.means()
            report = {"layer": layer, "region": region, "method": method, "fitting_problems": 111,
                      "mean": {}, "valid_tokens_per_completion": c.quantiles(np.diff(data.offsets)),
                      "study_manifest_sha256": manifest_digest}
            arrays = {}
            for family, (pos, neg) in c.FAMILIES.items():
                problems, contrasts = c.paired_contrasts(rows, z0, z60, pos, neg)
                c.require(problems == ids, "contrast cohort/order mismatch")
                report["mean"][family] = {}
                for kind in KINDS:
                    value, vector = mean_study(contrasts[kind], groups)
                    report["mean"][family][kind] = value
                    arrays[family + "__" + kind] = vector
            token_weight, token_problem, weight_ids = c.balanced_token_weights(data)
            c.require(weight_ids == ids, "PCA weighting problem order mismatch")
            base_weight = token_weight * len(ids)
            delta = data.delta
            c.require(np.isfinite(delta).all(), "nonfinite FP32 checkpoint differences")
            report["pca"], pca_arrays = pca_study(delta, token_problem, base_weight, groups,
                                                manifest["pca_pair_repeats"],
                                                c.stable_seed(manifest["seed"], layer, region, method), manifest["pca_config"])
            arrays.update(pca_arrays)
            stem = region + "__" + method
            with (outdir / (stem + ".npz")).open("xb") as stream:
                np.savez(stream, **arrays)
            report["seconds"] = time.monotonic() - started
            c.write_json(outdir / (stem + ".json"), report)
            print(c.canonical({"layer": layer, "region": region, "method": method, "seconds": report["seconds"]}), flush=True)
    c.write_json(outdir / "complete.json", {"status": "complete", "layer": layer, "cells": 16,
                                           "seconds": time.monotonic() - start, "manifest_sha256": manifest_digest})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-copy", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    if args.verify_copy:
        verify_copy(args.root)
    else:
        c.require(args.manifest and args.manifest_sha256 and args.layer is not None, "missing reviewed invocation")
        run_layer(args.root, args.manifest, args.manifest_sha256, args.layer)


if __name__ == "__main__":
    main()
