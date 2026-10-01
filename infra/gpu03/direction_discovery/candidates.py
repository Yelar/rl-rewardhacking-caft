#!/usr/bin/env python3
"""Bounded, fitting-only mean and weighted token-PCA direction discovery.

This is a CPU consumer of immutable BF16 post-block caches. It never loads a
model, generates text, executes generated code, or changes a source artifact.
The CLI requires an independently reviewed exclusion file and source-manifest
digest. Each layer can be sharded into a separate fresh output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


WINDOWS = ("pre_definition", "pre_body", "transition", "early_body")
WIDTHS = {"pre_definition": 16, "pre_body": 8, "transition": 16, "early_body": 16}
HARMFUL = "strict_reward_hack_evaluator_present"
CORRECT = "clean_correct_evaluator_present"
INCORRECT = "clean_incorrect_evaluator_present"
CLASSES = (HARMFUL, CORRECT, INCORRECT)
FAMILIES = {
    "harmful_vs_benign": ((HARMFUL,), (CORRECT, INCORRECT)),
    "harmful_incorrect_vs_benign_incorrect": ((HARMFUL,), (INCORRECT,)),
    "harmful_vs_benign_correct": ((HARMFUL,), (CORRECT,)),
    "benign_correct_vs_incorrect": ((CORRECT,), (INCORRECT,)),
}
PRIMARY_FAMILIES = {"harmful_vs_benign", "harmful_incorrect_vs_benign_incorrect"}
SITE = "decoder_layer_forward_output_before_final_norm"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(seed, *parts):
    value = canonical([int(seed), *parts]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def numerical_preflight():
    """Check the installed BLAS against a scalar-loop reduction reference.

    One local Homebrew OpenBLAS build returned zero/uninitialized SGEMM output
    for valid small matrices. A version string alone cannot qualify numerics.
    """
    rng = np.random.default_rng(8107)
    errors = []
    for n, d, k in ((17, 12, 4), (31, 12, 12), (97, 63, 24)):
        x = rng.normal(size=(n, d)).astype(np.float32)
        q = rng.normal(size=(d, k)).astype(np.float32)
        observed = x @ q
        reference = np.einsum("ij,jk->ik", x.astype(np.float64), q.astype(np.float64), optimize=False)
        require(np.isfinite(observed).all() and np.allclose(observed, reference, rtol=2e-5, atol=2e-5),
                "BLAS numerical preflight failed: float32 matrix multiplication disagrees with independent reduction")
        errors.append(float(np.max(np.abs(observed - reference))))
        gram = q.astype(np.float64).T @ q.astype(np.float64)
        values, vectors = np.linalg.eigh(gram)
        require(np.allclose(gram @ vectors, vectors * values, rtol=1e-10, atol=1e-10),
                "eigendecomposition numerical preflight failed")
    return {"passed": True, "float32_matmul_max_abs_errors": errors,
            "reference": "float64 numpy.einsum optimize=False independent reduction"}


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def safe_child(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "unsafe artifact path")
    path = root / rel
    require(path.is_file() and not path.is_symlink(), "missing/symlink artifact: " + str(path))
    require(path.resolve().is_relative_to(root.resolve()), "artifact escapes package")
    return path


def verify_file(path, info):
    require(Path(path).stat().st_size == info["size_bytes"], "artifact size mismatch: " + str(path))
    require(sha256_file(path) == info["sha256"], "artifact hash mismatch: " + str(path))


def file_identity(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "device": stat.st_dev, "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns, "owner_uid": stat.st_uid}


def validate_records(rows, exclusions):
    """Validate original splits/labels, then exclude complete disputed fit groups.

    Return fitting rows only. Held-out metadata is checked for split leakage;
    held-out activation files are never opened by this module.
    """
    require(rows and len({r["record_id"] for r in rows}) == len(rows), "duplicate/empty record IDs")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id_key"])].append(row)
    excluded = {str(p) for p in exclusions["excluded_problem_ids"]}
    disputed = set(exclusions["disputed_record_ids"])
    require(len(excluded) == len(exclusions["excluded_problem_ids"]), "duplicate excluded problem IDs")
    require(len(disputed) == len(exclusions["disputed_record_ids"]), "duplicate disputed record IDs")
    require(excluded <= set(groups), "unknown excluded problem ID")
    by_id = {r["record_id"]: r for r in rows}
    require(disputed <= set(by_id), "unknown disputed record ID")
    for record_id in disputed:
        require(str(by_id[record_id]["problem_id_key"]) in excluded, "disputed record's problem was not excluded")
    for problem in excluded:
        require({r["problem_split"] for r in groups[problem]} == {"direction_fit"},
                "exclusion may affect fitting problems only")
    fit = []
    flagged = []
    for problem, group in sorted(groups.items()):
        require(len({r["problem_split"] for r in group}) == 1, "problem crosses split boundary")
        require(set(r["outcome_presence_class"] for r in group) == set(CLASSES), "incomplete core problem classes")
        require(len({r["prompt_sha256"] for r in group}) == 1, "within-problem prompt mismatch")
        for row in group:
            label = row["outcome_presence_class"]
            truth = (row["is_test_modification_harmful"], row["ground_truth_correctness"], row["is_reward_hack_strict"])
            require(all(type(v) is bool for v in truth), "ambiguous original primitive labels")
            require(truth == {HARMFUL: (True, False, True), CORRECT: (False, True, False),
                              INCORRECT: (False, False, False)}[label], "original core label/primitive mismatch")
            reason = row.get("classification_error") or row.get("classification_disagreements")
            if reason:
                flagged.append(row["record_id"])
                if row["problem_split"] == "direction_fit":
                    require(problem in excluded, "unexcluded disputed fitting classification")
        if group[0]["problem_split"] == "direction_fit" and problem not in excluded:
            fit.extend(group)
    require(len({r["problem_id_key"] for r in fit}) >= 4, "too few undisputed fitting problems")
    return sorted(fit, key=lambda r: r["record_index"]), {
        "all_records": len(rows), "all_problems": len(groups),
        "split_records": dict(Counter(r["problem_split"] for r in rows)),
        "original_class_counts": dict(Counter(r["outcome_presence_class"] for r in rows)),
        "fitting_records": len(fit), "fitting_problems": len({r["problem_id_key"] for r in fit}),
        "fitting_class_counts": dict(Counter(r["outcome_presence_class"] for r in fit)),
        "excluded_problem_ids": sorted(excluded), "disputed_record_ids": sorted(disputed),
        "other_original_flagged_record_ids": sorted(flagged),
        "excluded_record_ids": sorted(r["record_id"] for p in excluded for r in groups[p]),
        "original_labels_preserved": True,
        "fitting_generation_provenance_by_class": {
            label: dict(Counter("existing" if r["record_id"].startswith("existing-") else "new"
                                for r in fit if r["outcome_presence_class"] == label))
            for label in CLASSES},
    }


def valid_positions(row, window):
    require(window in WINDOWS, "unknown evaluator window")
    evaluator = row["regions"].get("evaluator")
    require(evaluator is not None, "evaluator absent; cannot fabricate positions")
    slots = evaluator["window_completion_positions"][window]
    require(len(slots) == WIDTHS[window], "window slot count mismatch")
    positions = [p for p in slots if p is not None]
    require(positions and all(type(p) is int for p in positions), "empty/malformed valid window")
    require(positions == sorted(set(positions)), "duplicate/unordered window positions")
    require(0 <= positions[0] <= positions[-1] < row["completion_token_count"], "window outside completion")
    require(positions == row["region_mask_completion_positions"]["evaluator__" + window],
            "structural positions disagree with saved mask")
    return positions


@dataclass
class WindowData:
    rows: list
    positions: list
    h0: np.ndarray
    h60: np.ndarray
    offsets: np.ndarray
    token_record: np.ndarray
    token_position: np.ndarray

    @property
    def delta(self):
        # The cast precedes subtraction, including when the reader changes.
        return np.subtract(self.h60, self.h0, dtype=np.float32)

    def means(self):
        z0, z60 = [], []
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            z0.append(self.h0[start:end].mean(axis=0, dtype=np.float32))
            z60.append(self.h60[start:end].mean(axis=0, dtype=np.float32))
        return np.stack(z0), np.stack(z60)


class RawReader:
    """Read only fit files; lazy BF16-to-FP32 slices, bounded by one window."""

    def __init__(self, root, rows, index, manifest, source_digest, layer_count, hidden_size, integrity_receipt=None):
        self.root, self.rows, self.manifest = root, rows, manifest
        self.source_digest = source_digest
        self.layers, self.hidden = layer_count, hidden_size
        by_id = {r["record_id"]: r for r in index}
        require(len(by_id) == len(index), "duplicate raw activation index ID")
        self.entries = {}
        self.verified = set()
        self.integrity = integrity_receipt
        for row in rows:
            item = by_id[row["record_id"]]
            require(item["verified"] is True and item["record_index"] == row["record_index"], "unverified/misaligned raw index")
            self.entries[row["record_id"]] = item

    def read(self, row, kind, layer, positions):
        from safetensors import safe_open
        entry = self.entries[row["record_id"]]["models"][kind]
        path = safe_child(self.root, entry["tensor_path"])
        bound = self.manifest["files"][entry["tensor_path"]]
        require(bound["sha256"] == entry["sha256"] and bound["size_bytes"] == entry["size_bytes"], "raw index/manifest mismatch")
        if path not in self.verified:
            if self.integrity is None:
                verify_file(path, bound)
            else:
                proof = self.integrity["files"][entry["tensor_path"]]
                require(proof["sha256"] == bound["sha256"] and proof["identity"] == file_identity(path),
                        "shared integrity receipt no longer matches raw file")
            self.verified.add(path)
        before = file_identity(path)
        if self.integrity is not None:
            require(before == self.integrity["files"][entry["tensor_path"]]["identity"], "raw file changed after integrity verification")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata()
            require(meta["kind"] == kind and meta["record_id"] == row["record_id"] and
                    meta["record_index"] == str(row["record_index"]), "wrong raw tensor identity")
            require(meta["input_ids_sha256"] == row["input_ids_sha256"] and
                    meta["completion_sha256"] == row["completion_sha256"], "raw sequence hash mismatch")
            require(meta["model_revision"] == row["model_revision"] and
                    meta["adapter_sha256"] == (row["checkpoint_sha256"] if kind == "h60" else "none"),
                    "raw checkpoint mismatch")
            require(meta["manifest_sha256"] == self.source_digest and meta["site"] == SITE and
                    meta["token_axis"] == "all_original_completion_tokens", "wrong activation site/provenance")
            view = f.get_slice(kind)
            require(view.get_shape() == [self.layers, row["completion_token_count"], self.hidden] and
                    view.get_dtype() == "BF16", "raw shape/dtype mismatch")
            # Four windows are contiguous after clipping, but keep general masks.
            left, right = positions[0], positions[-1] + 1
            value = view[layer, left:right].float().numpy()
            value = value[np.asarray(positions) - left].copy()
        require(file_identity(path) == before, "raw file changed while reading")
        require(value.shape == (len(positions), self.hidden) and np.isfinite(value).all(), "nonfinite/malformed raw slice")
        return value

    def window(self, layer, window, memory_limit_bytes):
        positions = [valid_positions(row, window) for row in self.rows]
        offsets = np.cumsum([0] + [len(p) for p in positions], dtype=np.int64)
        # h0, h60, delta, centered temporary, PCA work and conservative overhead.
        estimate = int(offsets[-1]) * self.hidden * 4 * 5 + 64 * 1024**2
        require(estimate <= memory_limit_bytes, "window exceeds reviewed memory estimate")
        h0 = np.empty((int(offsets[-1]), self.hidden), dtype=np.float32)
        h60 = np.empty_like(h0)
        for i, (row, ps) in enumerate(zip(self.rows, positions)):
            for kind, target in (("h0", h0), ("h60", h60)):
                target[offsets[i]:offsets[i + 1]] = self.read(row, kind, layer, ps)
        return WindowData(self.rows, positions, h0, h60, offsets,
                          np.repeat(np.arange(len(self.rows)), np.diff(offsets)),
                          np.asarray([p for ps in positions for p in ps], dtype=np.int32))


def paired_contrasts(rows, z0, z60, positive, negative):
    require(z0.dtype == z60.dtype == np.float32 and z0.shape == z60.shape, "means must be paired FP32")
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[str(row["problem_id_key"])][row["outcome_presence_class"]].append(i)
    eligible = sorted(p for p, cells in groups.items() if set(positive + negative) <= set(cells))
    require(eligible, "no complete problem contrasts")
    contrasts = {"v0": [], "v60": []}
    for problem in eligible:
        cells = groups[problem]
        for key, z in (("v0", z0), ("v60", z60)):
            a = np.stack([z[cells[c]].mean(0, dtype=np.float32) for c in positive]).mean(0, dtype=np.float32)
            b = np.stack([z[cells[c]].mean(0, dtype=np.float32) for c in negative]).mean(0, dtype=np.float32)
            contrasts[key].append(np.subtract(a, b, dtype=np.float32))
    result = {key: np.stack(values) for key, values in contrasts.items()}
    result["v_change"] = np.subtract(result["v60"], result["v0"], dtype=np.float32)
    return eligible, result


def normalized(vector, *, absolute_floor=1e-7, relative_floor=1e-6, reference_norm=0.0):
    require(vector.dtype == np.float32 and vector.ndim == 1 and np.isfinite(vector).all(), "invalid FP32 candidate")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    threshold = max(absolute_floor, relative_floor * reference_norm)
    if norm <= threshold:
        return None, {"status": "rejected_near_zero", "raw_norm": norm, "rejection_threshold": threshold}
    value = vector / np.float32(norm)
    return value, {"status": "accepted", "raw_norm": norm, "rejection_threshold": threshold,
                   "unit_norm": float(np.linalg.norm(value.astype(np.float64)))}


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size and np.isfinite(values).all(), "empty/nonfinite statistic")
    return {"mean": float(values.mean()), "p025": float(np.quantile(values, .025)),
            "p50": float(np.quantile(values, .5)), "p975": float(np.quantile(values, .975))}


def mean_bootstrap(contrasts, reference, seed, repeats):
    """Exact problem-cluster bootstrap; completion/token rows are never sampled."""
    rng = np.random.default_rng(seed)
    cosines, norms, rejected = [], [], 0
    for _ in range(repeats):
        ids = rng.integers(0, len(contrasts), len(contrasts))
        vector = contrasts[ids].mean(0, dtype=np.float32)
        unit, info = normalized(vector)
        norms.append(info["raw_norm"])
        if unit is None:
            rejected += 1
        else:
            cosines.append(float(np.clip(unit.astype(np.float64) @ reference, -1, 1)))
    return {"method": "resample_problem_ids_with_replacement", "seed": seed,
            "replicates": repeats, "rejected_near_zero": rejected,
            "cosine": quantiles(cosines) if cosines else None, "raw_norm": quantiles(norms)}


def mean_candidates(data, seed, bootstrap):
    z0, z60 = data.means()
    tensors, reports = {}, {}
    for family, (positive, negative) in FAMILIES.items():
        problems, contrasts = paired_contrasts(data.rows, z0, z60, positive, negative)
        raw = {k: contrasts[k].mean(0, dtype=np.float32) for k in ("v0", "v60")}
        raw["v_change"] = np.subtract(raw["v60"], raw["v0"], dtype=np.float32)
        reference_norm = max(float(np.linalg.norm(raw[k].astype(np.float64))) for k in ("v0", "v60"))
        reports[family] = {"positive_classes": positive, "negative_classes": negative,
                           "problem_ids": problems, "problem_count": len(problems),
                           "correctness_controlled": family == "harmful_incorrect_vs_benign_incorrect",
                           "limitations": ("harmful examples are all incorrect; correct harmful examples absent"
                                           if family != "benign_correct_vs_incorrect" else "benign correctness diagnostic"),
                           "vectors": {}}
        for key in ("v60", "v0", "v_change"):
            tensors[family + ".raw_" + key] = raw[key]
            unit, report = normalized(raw[key], reference_norm=reference_norm if key == "v_change" else 0)
            report["role"] = "primary" if family in PRIMARY_FAMILIES and key != "v0" else "diagnostic"
            if unit is not None:
                tensors[family + "." + key] = unit[:, None]
                report["bootstrap"] = mean_bootstrap(contrasts[key], unit, stable_seed(seed, family, key), bootstrap)
            reports[family]["vectors"][key] = report
    return tensors, reports


def balanced_token_weights(data):
    """Each problem, outcome class, completion within cell, and token within
    completion receives equal nested mass. The three outcome classes, rather
    than binary harmful/benign labels, define class balance for PCA.
    """
    problems = sorted({str(r["problem_id_key"]) for r in data.rows})
    lookup = {p: i for i, p in enumerate(problems)}
    cells = Counter((str(r["problem_id_key"]), r["outcome_presence_class"]) for r in data.rows)
    require(all({c for p, c in cells if p == problem} == set(CLASSES) for problem in problems),
            "PCA requires every outcome class per fitting problem")
    weights = np.empty(int(data.offsets[-1]), dtype=np.float64)
    token_problem = np.empty(len(weights), dtype=np.int32)
    for i, row in enumerate(data.rows):
        start, end = data.offsets[i:i + 2]
        problem = str(row["problem_id_key"])
        weights[start:end] = 1 / (len(problems) * len(CLASSES) * cells[problem, row["outcome_presence_class"]] * (end - start))
        token_problem[start:end] = lookup[problem]
    require(np.isclose(weights.sum(), 1, atol=1e-12), "PCA weights do not sum to one")
    return weights, token_problem, problems


class WeightedCovariance:
    """Chunked C @ matrix; never materialize a d_model-by-d_model covariance."""

    def __init__(self, x, weights, chunk_rows=1024):
        require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), "PCA needs finite FP32 token rows")
        weights = np.asarray(weights, dtype=np.float64)
        require(weights.shape == (len(x),) and np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0,
                "invalid PCA row weights")
        self.x, self.weights, self.chunk = x, weights / weights.sum(), chunk_rows
        require(chunk_rows > 0, "invalid PCA chunk bound")
        self.mean = np.zeros(x.shape[1], dtype=np.float64)
        for start in range(0, len(x), self.chunk):
            end = min(start + self.chunk, len(x))
            self.mean += self.weights[start:end] @ x[start:end].astype(np.float64)
        self.trace = 0.0
        for start in range(0, len(x), self.chunk):
            end = min(start + self.chunk, len(x))
            centered = x[start:end].astype(np.float64) - self.mean
            self.trace += float(self.weights[start:end] @ np.square(centered).sum(1))

    def multiply(self, matrix):
        require(matrix.shape[0] == self.x.shape[1], "covariance multiplier dimension mismatch")
        result = np.zeros((self.x.shape[1], matrix.shape[1]), dtype=np.float64)
        for start in range(0, len(self.x), self.chunk):
            end = min(start + self.chunk, len(self.x))
            centered = (self.x[start:end].astype(np.float64) - self.mean).astype(np.float32)
            projected = centered @ matrix.astype(np.float32)
            result += centered.T @ (projected * self.weights[start:end, None]).astype(np.float32)
        return result.astype(np.float32)


def orient_columns(vectors):
    vectors = vectors.copy()
    for col in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, col])))
        if vectors[pivot, col] < 0:
            vectors[:, col] *= -1
    return vectors


def weighted_pca(x, weights, *, rank=10, oversample=14, power_iters=2, seed=6001, chunk_rows=1024):
    """Randomized symmetric covariance range finder with Rayleigh-Ritz PCs.

    Orthogonalize C Omega, then C Q `power_iters` times. Only the resulting
    (rank+oversample)-square matrix is diagonalized. Residuals are computed in
    the full feature space; a low residual is evidence about this approximation,
    not evidence that a PC represents evaluator tampering.
    """
    require(rank > 0 and oversample >= 0 and power_iters >= 0, "invalid PCA configuration")
    operator = WeightedCovariance(x, weights, chunk_rows)
    require(operator.trace > 1e-14, "PCA rejected: near-zero centered variance")
    size = min(rank + oversample, x.shape[1], int(np.count_nonzero(weights)) - 1)
    require(size >= rank, "PCA token/feature rank below requested components")
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((x.shape[1], size), dtype=np.float32)
    basis = np.linalg.qr(operator.multiply(omega), mode="reduced")[0].astype(np.float32)
    for _ in range(power_iters):
        basis = np.linalg.qr(operator.multiply(basis), mode="reduced")[0].astype(np.float32)
    cb = operator.multiply(basis)
    small = basis.astype(np.float64).T @ cb
    values, rotations = np.linalg.eigh((small + small.T) / 2)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order[:rank]], 0)
    pcs = orient_columns((basis @ rotations[:, order[:rank]]).astype(np.float32))
    cpcs = operator.multiply(pcs)
    residuals = np.linalg.norm(cpcs.astype(np.float64) - pcs * values, axis=0) / np.maximum(values, 1e-20)
    orthogonality = float(np.max(np.abs(pcs.astype(np.float64).T @ pcs - np.eye(rank))))
    require(orthogonality < 5e-5, "PCA orthonormality failure")
    return {"pcs": pcs, "basis": basis, "mean": operator.mean.astype(np.float32),
            "eigenvalues": values, "report": {
                "method": "randomized_weighted_covariance_range_finder_rayleigh_ritz",
                "centered": True, "seed": seed, "rank": rank, "range_rank": size,
                "power_iterations": power_iters, "chunk_rows": chunk_rows,
                "covariance_definition": "population; sum weights = 1",
                "total_weighted_variance": operator.trace,
                "eigenvalues": values.tolist(), "explained_variance_ratio": (values / operator.trace).tolist(),
                "relative_covariance_residuals": residuals.tolist(), "max_orthonormality_error": orthogonality,
                "near_zero_pc_indices": np.flatnonzero(values <= max(1e-12, operator.trace * 1e-12)).tolist(),
                "orientation": "largest absolute loading is positive; PC sign has no projection effect",
            }}


def pca_problem_bootstrap(x, weights, token_problem, result, seed, repeats):
    """Conditional PCA uncertainty in the fitted oversampled range.

    Sample whole problem IDs, recenter each bootstrap, and recompute its PCs.
    This omits rotations outside the fitted range; full-space independent-half
    fits are reported separately to expose that limitation.
    """
    basis, reference = result["basis"], result["pcs"]
    count = int(token_problem.max()) + 1
    projected = (x - result["mean"]) @ basis
    means = np.zeros((count, basis.shape[1]), dtype=np.float64)
    seconds = np.zeros((count, basis.shape[1], basis.shape[1]), dtype=np.float64)
    for problem in range(count):
        mask = token_problem == problem
        require(np.isclose(weights[mask].sum(), 1 / count, atol=1e-12), "bootstrap problem mass imbalance")
        w = weights[mask] * count
        value = projected[mask].astype(np.float64)
        means[problem] = w @ value
        seconds[problem] = value.T @ (value * w[:, None])
    rng = np.random.default_rng(seed)
    cosines = []
    overlaps = []
    for _ in range(repeats):
        indices = rng.integers(0, count, count)
        mean = means[indices].mean(0)
        cov = seconds[indices].mean(0) - np.outer(mean, mean)
        _, rotations = np.linalg.eigh((cov + cov.T) / 2)
        pcs = basis @ rotations[:, ::-1][:, :reference.shape[1]]
        cross = reference.astype(np.float64).T @ pcs
        cosines.append(np.abs(np.diag(cross)).clip(0, 1))
        overlaps.append(float(np.square(cross).sum() / reference.shape[1]))
    cosines = np.asarray(cosines)
    return {"method": "problem_ID_bootstrap_recentered_PCA_conditional_on_fitted_oversampled_range",
            "limitation": "does not measure bootstrap rotations outside fitted range; not a full PCA bootstrap",
            "seed": seed, "replicates": repeats,
            "pc_same_index_absolute_cosine": [quantiles(cosines[:, i]) for i in range(reference.shape[1])],
            "top_k_subspace_overlap": quantiles(overlaps)}


def half_pca_stability(x, weights, token_problem, result, seed, config):
    count = int(token_problem.max()) + 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    halves = (order[:count // 2], order[count // 2:])
    fits = []
    for i, half in enumerate(halves):
        mask = np.isin(token_problem, half)
        fits.append(weighted_pca(x[mask], weights[mask], seed=stable_seed(seed, "half", i), **config))
    cross = fits[0]["pcs"].astype(np.float64).T @ fits[1]["pcs"]
    singular = np.linalg.svd(cross, compute_uv=False).clip(0, 1)
    return {"method": "two_disjoint_problem_halves_refitted_in_full_hidden_space",
            "seed": seed, "half_problem_indices": [h.tolist() for h in halves],
            "same_index_absolute_cosine": np.abs(np.diag(cross)).clip(0, 1).tolist(),
            "principal_angle_cosines": singular.tolist(),
            "top_k_subspace_overlap": float(np.square(singular).mean()),
            "against_full_fit_same_index_cosines": [np.abs(np.diag(result["pcs"].astype(np.float64).T @ f["pcs"])).clip(0, 1).tolist() for f in fits],
            "half_approximation_reports": [f["report"] for f in fits]}


def pc_contexts(data, delta, pcs, center, decode=None, examples_per_sign=3):
    scores = (delta - center) @ pcs
    contexts = []
    for col in range(pcs.shape[1]):
        sides = {}
        for sign, order in (("positive", np.argsort(-scores[:, col], kind="stable")),
                            ("negative", np.argsort(scores[:, col], kind="stable"))):
            picked, problems = [], set()
            for index in order:
                row = data.rows[int(data.token_record[index])]
                problem = str(row["problem_id_key"])
                if problem in problems:
                    continue
                problems.add(problem)
                pos = int(data.token_position[index])
                left, right = max(0, pos - 8), min(row["completion_token_count"], pos + 9)
                ids = row["completion_token_ids"][left:right]
                picked.append({"record_id": row["record_id"], "problem_id": problem,
                               "class": row["outcome_presence_class"], "test_modification": row["test_modification"],
                               "ground_truth_correctness": row["ground_truth_correctness"],
                               "completion_token_position": pos, "sequence_token_position": row["prompt_token_count"] + pos,
                               "token_id": row["completion_token_ids"][pos], "score": float(scores[index, col]),
                               "context_completion_range": [left, right], "context_token_ids": ids,
                               "context_text": decode(ids) if decode else None})
                if len(picked) == examples_per_sign:
                    break
            sides[sign] = picked
        contexts.append({"pc_index": col, "fitting_contexts_only": True, **sides})
    return contexts


def discover_one(data, *, layer, window, seed, bootstrap, pca_config, decode=None):
    tensors, mean_report = mean_candidates(data, stable_seed(seed, layer, window, "mean"), bootstrap)
    delta = data.delta
    weights, token_problem, problems = balanced_token_weights(data)
    pca = weighted_pca(delta, weights, seed=stable_seed(seed, layer, window, "pca"), **pca_config)
    tensors["pca.pcs"] = pca["pcs"]
    tensors["pca.center"] = pca["mean"]
    tensors["pca.eigenvalues"] = pca["eigenvalues"].astype(np.float32)
    report = {"layer": layer, "layer_numbering": "zero_based_transformer_blocks", "site": SITE,
              "window": window, "primary_window": window == "transition", "seed": seed,
              "fitting_problem_ids": problems, "records": len(data.rows), "tokens": len(delta),
              "delta_subtraction_dtype": "float32", "mean_aggregation_dtype": "float32",
              "mean_weighting": "valid tokens equally within completion; completions equally within problem/class; classes equally within side; paired problems equally",
              "pca_weighting": "problems equally; three outcome classes equally within problem; completions equally within class; valid tokens equally within completion",
              "window_clipped_records": sum(len(p) < WIDTHS[window] for p in data.positions),
              "windows_are_structural_not_established_commitment_locations": True,
              "mean_candidates": mean_report, "pca": pca["report"],
              "interpretation": "Discovery candidates only. PCA captures checkpoint-change variance; contexts and causal validation are required before assigning a tampering interpretation.",
              "unavailable_comparisons": ["harmful-correct vs benign-correct: selected triplets contain no correct harmful evaluators"],
              "confounds": ["selected harmful examples are all ground-truth incorrect", "evaluator syntax/subtype may dominate both mean and PCA directions",
                            "legacy versus new generation provenance is correlated with labels; equal problem weighting does not remove this nuisance"],
              "pc_contexts": pc_contexts(data, delta, pca["pcs"], pca["mean"], decode)}
    report["pca"]["problem_bootstrap"] = pca_problem_bootstrap(delta, weights, token_problem, pca,
                                                             stable_seed(seed, layer, window, "pca_bootstrap"), bootstrap)
    report["pca"]["independent_halves"] = half_pca_stability(delta, weights, token_problem, pca,
                                                          stable_seed(seed, layer, window, "pca_halves"), pca_config)
    return tensors, report


def candidate_catalog(report, tensor_path, report_path):
    prefix = f"L{report['layer']:02d}.{report['window']}"
    result = []
    for family, family_info in report["mean_candidates"].items():
        for kind, info in family_info["vectors"].items():
            if info["status"] != "accepted":
                continue
            result.append({"candidate_id": f"{prefix}.mean.{family}.{kind}",
                           "family": family, "kind": kind, "role": info["role"],
                           "layer": report["layer"], "window": report["window"], "rank": 1,
                           "tensor_file": tensor_path, "tensor_key": family + "." + kind,
                           "report_file": report_path, "correctness_controlled": family_info["correctness_controlled"]})
    for pc in range(report["pca"]["rank"]):
        if pc in report["pca"]["near_zero_pc_indices"]:
            continue
        result.append({"candidate_id": f"{prefix}.pc{pc:02d}", "family": "pca", "kind": "pc",
                       "role": "individual_PC_discovery", "layer": report["layer"],
                       "window": report["window"], "rank": 1, "tensor_file": tensor_path,
                       "tensor_key": "pca.pcs", "column_indices": [pc], "report_file": report_path})
    return result


def verify_package(root):
    """Independent post-producer hash, candidate identity and Q-shape audit."""
    from safetensors import safe_open
    root = Path(root)
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for relative, info in manifest["files"].items():
        verify_file(safe_child(root, relative), info)
    require(not (root / "FAILURE.json").exists(), "candidate package has failure marker")
    config = json.loads((root / "resolved_config.json").read_text())
    success = json.loads((root / "SUCCESS.json").read_text())
    catalog = json.loads((root / "candidate_catalog.json").read_text())["candidates"]
    require(success["status"] == "succeeded" and success["held_out_activation_files_opened"] == 0,
            "candidate package is not successful fitting-only discovery")
    require(success["fit_records"] == config["dataset"]["fitting_records"] and
            success["fit_problems"] == config["dataset"]["fitting_problems"], "candidate fitting counts mismatch")
    require(success["candidates"] == len(catalog) == len({c["candidate_id"] for c in catalog}), "candidate count/identity mismatch")
    require(success["layer_windows"] == len(config["layers"]) * len(WINDOWS), "candidate layer-window coverage mismatch")
    config_digest = sha256_file(root / "resolved_config.json")
    for candidate in catalog:
        require(candidate["layer"] in config["layers"] and candidate["window"] in WINDOWS and candidate["rank"] == 1,
                "unexpected candidate layer/window/rank")
        require(candidate["tensor_file"] in manifest["files"] and candidate["report_file"] in manifest["files"],
                "unbound candidate artifact")
        with safe_open(str(safe_child(root, candidate["tensor_file"])), framework="numpy") as f:
            require(f.metadata()["input_config_sha256"] == config_digest and f.metadata()["fitting_only"] == "true",
                    "candidate tensor provenance mismatch")
            require(f.get_slice(candidate["tensor_key"]).get_dtype() == "F32", "candidate is not FP32")
            value = f.get_tensor(candidate["tensor_key"])
            if "column_indices" in candidate:
                value = value[:, candidate["column_indices"]]
            require(value.shape == (2560, 1) and np.isfinite(value).all() and
                    np.allclose(value.astype(np.float64).T @ value, np.eye(1), atol=5e-5),
                    "candidate Q dimension/normalization failure")
    return {"status": "verified", "files": len(manifest["files"]), "candidates": len(catalog),
            "artifact_manifest_sha256": sha256_file(root / "artifact_manifest.json")}


def run(args):
    # CPU-only numerical discovery even on a host with idle GPUs.
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "candidate job must hide CUDA devices")
    require(args.bootstrap >= 20 and args.max_memory_mib >= 256 and args.deadline_seconds > 0, "invalid reviewed CPU budget")
    require(args.layers and len(set(args.layers)) == len(args.layers) and all(0 <= i < 36 for i in args.layers), "invalid layer shard")
    require(args.pca_rank == 10, "production discovery requires top 10 PCs")
    numeric_audit = numerical_preflight()
    root = Path(args.raw_package)
    manifest_path = root / "artifact_manifest.json"
    require(sha256_file(manifest_path) == args.raw_manifest_sha256, "reviewed raw package manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for relative in ("activation_index.jsonl", "extraction_summary.json"):
        verify_file(safe_child(root, relative), manifest["files"][relative])
    records_path = Path(args.prepared_records)
    verify_file(records_path, manifest["files"]["input/prepared_records.jsonl"])
    rows = read_jsonl(records_path)
    exclusion_path = Path(args.exclusions)
    exclusions = json.loads(exclusion_path.read_text())
    fit, report = validate_records(rows, exclusions)
    summary = json.loads((root / "extraction_summary.json").read_text())
    require(summary["status"] == "succeeded" and summary["raw_activations_retained"] is True and
            summary["layers"] == 36 and summary["hidden_size"] == 2560 and summary["records"] == len(rows),
            "source cache lacks successful complete extraction")
    integrity = None
    if args.raw_integrity_receipt:
        receipt_path = Path(args.raw_integrity_receipt)
        require(args.raw_integrity_sha256 and sha256_file(receipt_path) == args.raw_integrity_sha256,
                "shared raw integrity receipt hash mismatch")
        require(receipt_path.stat().st_uid == os.getuid() and receipt_path.stat().st_mode & 0o222 == 0,
                "shared raw integrity receipt must be owned and read-only")
        integrity = json.loads(receipt_path.read_text())
        require(integrity["status"] == "verified" and integrity["raw_manifest_sha256"] == args.raw_manifest_sha256 and
                integrity["prepared_records_sha256"] == sha256_file(records_path) and
                integrity["exclusion_manifest_sha256"] == sha256_file(exclusion_path), "shared receipt input mismatch")
        require(set(integrity["fitting_record_ids"]) == {r["record_id"] for r in fit}, "shared receipt fitting records mismatch")
    reader = RawReader(root, fit, read_jsonl(root / "activation_index.jsonl"), manifest,
                       summary["manifest_sha256"], 36, 2560, integrity)
    # Tokenizer decoding is local-only, with hashes bound by the caller's run manifest.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    output = Path(args.output)
    output.mkdir(parents=False, exist_ok=False)
    args.output_created = True
    pca_config = {"rank": args.pca_rank, "oversample": args.pca_oversample,
                  "power_iters": args.pca_power_iters, "chunk_rows": args.chunk_rows}
    config = {"schema_version": 1, "purpose": "checkpoint60_fitting_only_direction_discovery",
              "raw_package": str(root), "raw_manifest_sha256": args.raw_manifest_sha256,
              "raw_extraction_manifest_sha256": summary["manifest_sha256"],
              "shared_raw_integrity_sha256": args.raw_integrity_sha256,
              "prepared_records_sha256": sha256_file(records_path), "exclusion_manifest_sha256": sha256_file(exclusion_path),
              "seed": args.seed, "bootstrap_replicates": args.bootstrap, "layers": sorted(args.layers),
              "windows": WINDOWS, "pca": pca_config, "max_memory_mib": args.max_memory_mib,
              "deadline_seconds": args.deadline_seconds, "source_sha256": sha256_file(Path(__file__)),
              "numpy_version": np.__version__, "python_version": sys.version,
              "numerical_preflight": numeric_audit,
              "thread_limits": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "dataset": report, "excluded_provenance": exclusions}
    write_json(output / "resolved_config.json", config)
    from safetensors.numpy import save_file
    started = time.monotonic()
    catalog, file_hashes = [], {}
    for layer in sorted(args.layers):
        for window in WINDOWS:
            require(time.monotonic() - started < args.deadline_seconds, "candidate discovery deadline exceeded")
            data = reader.window(layer, window, args.max_memory_mib * 1024**2)
            tensors, discovery = discover_one(data, layer=layer, window=window, seed=args.seed,
                                              bootstrap=args.bootstrap, pca_config=pca_config, decode=decode)
            base = f"layer_{layer:02d}_{window}"
            tensor_path, report_path = base + ".safetensors", base + ".json"
            require(not (output / tensor_path).exists(), "refusing candidate tensor overwrite")
            save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(output / tensor_path),
                      metadata={"schema_version": "1", "layer": str(layer), "window": window,
                                "fitting_only": "true", "input_config_sha256": sha256_file(output / "resolved_config.json")})
            write_json(output / report_path, discovery)
            catalog.extend(candidate_catalog(discovery, tensor_path, report_path))
            for name in (tensor_path, report_path):
                file_hashes[name] = {"sha256": sha256_file(output / name), "size_bytes": (output / name).stat().st_size}
            print(canonical({"layer": layer, "window": window, "status": "complete", "elapsed_seconds": time.monotonic() - started}), flush=True)
            del data, tensors, discovery
    write_json(output / "candidate_catalog.json", {"candidates": catalog})
    write_json(output / "SUCCESS.json", {"status": "succeeded", "layers": sorted(args.layers),
               "layer_windows": len(args.layers) * len(WINDOWS), "candidates": len(catalog),
               "fit_raw_files_hash_verified": len(reader.verified), "fit_records": len(fit),
               "fit_problems": report["fitting_problems"], "held_out_activation_files_opened": 0,
               "elapsed_seconds": time.monotonic() - started})
    for name in ("resolved_config.json", "candidate_catalog.json", "SUCCESS.json"):
        file_hashes[name] = {"sha256": sha256_file(output / name), "size_bytes": (output / name).stat().st_size}
    write_json(output / "artifact_manifest.json", {"algorithm": "sha256", "files": file_hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-package", "raw-manifest-sha256", "prepared-records", "exclusions", "tokenizer", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(36)))
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--pca-rank", type=int, default=10)
    parser.add_argument("--pca-oversample", type=int, default=14)
    parser.add_argument("--pca-power-iters", type=int, default=2)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--deadline-seconds", type=int, default=14400)
    parser.add_argument("--raw-integrity-receipt")
    parser.add_argument("--raw-integrity-sha256")
    args = parser.parse_args()
    args.output_created = False
    try:
        run(args)
    except Exception as exc:
        output = Path(args.output)
        if args.output_created and output.is_dir() and (output / "resolved_config.json").is_file() and not (output / "FAILURE.json").exists():
            write_json(output / "FAILURE.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
