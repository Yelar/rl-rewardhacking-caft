"""Prospective nomination and closed-artifact regressions on authored fixtures."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import compile_shortlist as c
from . import test_build_fit_plan as build_tests


def authored_reports():
    reports, centers, tensors = [], {}, {}
    winners = {"stratified_mean": (3, "body_predictor", "h60", .91),
               "correctness_orthogonalized_mean": (9, "transition", "delta", .90),
               "ridge_probe": (17, "body_predictor", "delta", .95)}
    for layer in range(36):
        cref = {"path": f"center{layer}", "sha256": "c"*64, "size_bytes": 10}
        centers[layer] = {"report": {"status": "fitted"}, "report_ref": {"path": f"center{layer}.json"}, "tensor_ref": cref}
        tensors[cref["path"], "h60_response_center"] = np.full(2560, layer / 64, dtype=np.float32)
        for rep in c.fitting.REPRESENTATIONS:
            ref = {"path": f"L{layer}.{rep}", "sha256": "a"*64, "size_bytes": 10}
            candidates = []
            for fidx, family in enumerate(c.fitting.FAMILIES):
                for kind in c.fitting.KINDS:
                    key = f"L{layer:02d}.evaluator.{rep}.{kind}.{family}"
                    eligible = (layer, rep, kind) == winners[family][:3]
                    item = {"candidate_id": key, "tensor_prefix": key, "family": family, "layer": layer,
                            "representation": rep, "activation_kind": kind, "eligible": eligible,
                            "status": "fitted" if eligible else "unsupported", "support": None,
                            "rejection_reasons": [] if eligible else ["authored unsupported contrast"],
                            "source_report": {"path": ref["path"] + ".json"}, "source_tensors": ref}
                    if eligible:
                        item.update(oof={"complete": True, "balanced_paired_auc": winners[family][3]},
                                    oof_h60_transfer={"complete": True, "balanced_paired_auc": .8},
                                    half_signed_cosine=.9, half_aligned_cosine=.9, fit={})
                        if family == "correctness_orthogonalized_mean":
                            item["fit"]["orthogonalization"] = {"residual_ratio": .5, "original_norm": 2, "residual_norm": 1}
                        q = np.zeros((2560, 1), np.float32)
                        q[fidx, 0] = -1  # Preserve saved sign/bytes; projection is sign invariant.
                        tensors[ref["path"], key + ".direction"] = q
                    candidates.append(item)
            reports.append({"layer": layer, "representation": rep, "candidates": candidates,
                            "coverage": {}, "diagnostic_references": {"status": "diagnostic_only"}})
    pc4 = np.zeros((2560, 1), np.float32)
    pc4[4] = 1
    tensors[c.PC4_PATH, c.PC4_KEY] = pc4
    return reports, centers, tensors


class NominationTests(unittest.TestCase):
    def test_all_families_frozen_best_and_exact_vector_bytes(self):
        reports, centers, values = authored_reports()
        get = lambda ref, key: values[ref["path"], key].copy()
        with patch.object(c, "PC4_TENSOR_SHA", c.tensor_bytes_sha(values[c.PC4_PATH, c.PC4_KEY])):
            shortlist, tensors = c.assemble(reports, centers, get)
        self.assertEqual(len(shortlist["axes"]), 4)
        self.assertEqual(len(shortlist["operator_configurations"]), 6)
        self.assertEqual(shortlist["candidate_count"], 432)
        centered = {x["axis_id"] for x in shortlist["operator_configurations"] if x["operator"] == "centered"}
        self.assertEqual(centered, {"historical_pc4", "ridge_probe"})
        np.testing.assert_array_equal(tensors["axis__stratified_mean"], values["L3.body_predictor",
            "L03.evaluator.body_predictor.h60.stratified_mean.direction"])
        self.assertFalse(shortlist["causal_validation_claimed"])
        self.assertFalse(shortlist["random_controls_prepared"])

    def test_partial_cells_or_changed_historical_reference_rejected(self):
        reports, centers, values = authored_reports()
        get = lambda ref, key: values[ref["path"], key].copy()
        with self.assertRaisesRegex(ValueError, "Complete all-36"):
            c.assemble(reports[:-1], centers, get)
        with self.assertRaisesRegex(ValueError, "PC4 tensor bytes"):
            c.assemble(reports, centers, get)

    def test_unsupported_families_and_missing_optional_center_are_explicit(self):
        reports, centers, values = authored_reports()
        for report in reports:
            for candidate in report["candidates"]:
                candidate.update(eligible=False)
        centers[21]["report"] = {"status": "unsupported", "reason": "missing clean stratum"}
        get = lambda ref, key: values[ref["path"], key].copy()
        with patch.object(c, "PC4_TENSOR_SHA", c.tensor_bytes_sha(values[c.PC4_PATH, c.PC4_KEY])):
            shortlist, tensors = c.assemble(reports, centers, get)
        self.assertEqual(len(shortlist["axes"]), 1)
        self.assertEqual(len(shortlist["operator_configurations"]), 1)
        self.assertEqual(len(shortlist["selection"]["unsupported_families"]), 3)
        self.assertEqual(shortlist["gaps"][0]["layer"], 21)

    def test_nonunit_or_wrong_dtype_saved_vector_cannot_be_repaired_silently(self):
        for bad in (np.ones((2560, 1), np.float32), np.zeros((2560, 1), np.float64),
                    np.full((2560, 1), np.nan, np.float32)):
            with self.subTest(dtype=bad.dtype), self.assertRaises(ValueError):
                c.validate_direction(bad)


class ClosedArtifactTests(unittest.TestCase):
    """Pinned CPU safetensors runtime required; fixture results are not science."""

    def setUp(self):
        self.builder = build_tests.BuildPlanTests()
        self.builder.setUp()
        self.addCleanup(self.builder.doCleanups)
        self.root = self.builder.root
        patch.object(c, "ROOT", str(self.root)).start()

    def fixture(self):
        from safetensors.numpy import save_file
        from . import build_fit_plan as b
        extraction, digest, _ = self.builder.fixture()
        prepared = b.prepare(extraction, digest, "fitting-v1", now=1000)
        manifest_path = Path(prepared["manifest"])
        manifest = c.read_json(manifest_path)
        reports, centers, values = authored_reports()
        by_cell = {(r["layer"], r["representation"]): r for r in reports}
        for worker in manifest["workers"]:
            task = c.read_json(worker["argv"][5])
            rows = [json.loads(line) for line in Path(task["prepared_rows"]).open()]
            out = Path(task["output"]); out.mkdir()
            artifacts = {}
            layers = list(range(task["shard_index"], 36, 8))
            def save(stem, report, tensors):
                tp = out / (stem + ".safetensors")
                save_file(tensors, str(tp), metadata={"task_sha256": worker["argv"][7]})
                artifacts[tp.name] = c.io.binding(tp)
                report["tensor_file"] = {"path": tp.name, **artifacts[tp.name]}
                rp = out / (stem + ".json")
                self.builder.write(rp, report)
                artifacts[rp.name] = c.io.binding(rp)
            for layer in layers:
                save(f"L{layer:02d}.response_center", {"schema": "fitting_only_response_predictor_center_v1",
                    "layer": layer, "activation_kind": "h60", "status": "fitted",
                    "policy": "equal_correctness_stratum_then_problem_then_response_then_predictor",
                    "positions": "last_prompt_plus_completion_excluding_final_token",
                    "clean_definition": "known_benign_modification_and_not_strict_reward_hack"},
                    {"h60_response_center": values[f"center{layer}", "h60_response_center"]})
                for rep in c.fitting.REPRESENTATIONS:
                    report = copy.deepcopy(by_cell[layer, rep])
                    report.update(schema="task_specific_fitting_cell_v1", region="evaluator", config=c.fitting.configuration(task.get("config")),
                        input_records=len(rows), hidden_size=2560, matched_records=len(rows),
                        matched_record_ids=[r["record_id"] for r in rows], unmatched_record_ids=[],
                        statistical_policy={"outer_folds": 5, "inner_folds": 5, "ridge_grid": [.1,1.,10.],
                            "minimum_paired_problems_per_stratum": 30, "minimum_half_cosine": .5,
                            "minimum_orthogonal_residual_ratio": .1,
                            "ranking_metric": "equal_stratum_equal_problem_heldout_paired_auc"},
                        coverage={str(s).lower(): {"paired_problems": len(ps), "problem_ids": ps}
                                  for s, ps in c.fitting.matched_problems(rows).items()})
                    tensors = {key: value for (path, key), value in values.items() if path == f"L{layer}.{rep}"}
                    save(f"L{layer:02d}.evaluator.{rep}", report, tensors)
            self.builder.write(out / "COMPLETE.json", {"schema": "task_specific_cpu_fitting_complete_v1",
                "status": "succeeded", "task_sha256": worker["argv"][7], "shard_index": task["shard_index"], "shard_count": 8,
                "layers": layers, "completed_cells": [{"layer": l, "representation": r} for l in layers for r in c.fitting.REPRESENTATIONS],
                "records": len(rows), "verified_raw_and_delta_files": 3*len(rows), "artifacts": artifacts,
                "causal_validation_claimed": False, "model_execution": False})
        self.builder.write(Path(manifest["runtime"]) / "SUPERVISOR_RESULT.json", {"run_token": self.root.name,
            "manifest_sha256": prepared["sha256"], "status": "workers_exited_zero", "owned_processes_released": True,
            "cpu_only": True, "gpu_release_verified": None, "exit_codes": {w["name"]: 0 for w in manifest["workers"]}})
        pc4 = values[c.PC4_PATH, c.PC4_KEY]
        path = self.root / "calibration_vectors.safetensors"
        save_file({c.PC4_KEY: pc4}, str(path))
        patch.object(c, "PC4_PATH", str(path)).start()
        patch.object(c, "PC4_FILE", c.io.binding(path)).start()
        patch.object(c, "PC4_TENSOR_SHA", c.tensor_bytes_sha(pc4)).start()
        return manifest_path, prepared["sha256"], manifest

    def test_real_safetensors_complete_216_artifact_readback_and_nomination(self):
        from safetensors.numpy import load_file
        path, digest, _ = self.fixture()
        out = self.root / "shortlist-v1"
        result = c.compile_shortlist(path, digest, out)
        self.assertEqual((result["cells"], result["candidates"], result["axes"], result["operator_configurations"]), (72,432,4,6))
        shortlist = c.read_json(out / "SHORTLIST.json")
        tensors = load_file(str(out / "shortlist.safetensors"))
        for axis in shortlist["axes"]:
            self.assertEqual(c.tensor_bytes_sha(tensors[axis["tensor_key"]]), axis["tensor_sha256"])
        for name, expected in result["artifacts"].items():
            self.assertEqual(c.io.binding(out / name), expected)
        table = c.read_json(out / "CANDIDATE_TABLE.json")
        self.assertEqual(sum(len(r["candidates"]) for r in table["cells"]), 432)
        with self.assertRaisesRegex(ValueError, "fresh"):
            c.compile_shortlist(path, digest, out)

    def test_incomplete_shard_or_failed_release_never_nominates(self):
        path, digest, manifest = self.fixture()
        complete = self.root / "fitting-v1/cpu_7/COMPLETE.json"
        original = c.read_json(complete)
        changed = copy.deepcopy(original); changed["completed_cells"].pop()
        self.builder.write(complete, changed)
        with self.assertRaisesRegex(ValueError, "completion/coverage"):
            c.compile_shortlist(path, digest, self.root / "shortlist-v1")
        self.assertFalse((self.root / "shortlist-v1").exists())
        self.builder.write(complete, original)
        release = Path(manifest["runtime"]) / "SUPERVISOR_RESULT.json"
        report = c.read_json(release);report["owned_processes_released"] = False
        self.builder.write(release, report)
        with self.assertRaisesRegex(ValueError, "release/exit"):
            c.compile_shortlist(path, digest, self.root / "shortlist-v1")

    def test_mutated_unselected_cell_tensor_is_still_rejected(self):
        path, digest, _ = self.fixture()
        unselected = self.root / "fitting-v1/cpu_0/L00.evaluator.transition.safetensors"
        unselected.write_bytes(b"bad unselected tensor")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            c.compile_shortlist(path, digest, self.root / "shortlist-v1")


if __name__ == "__main__":
    unittest.main()
