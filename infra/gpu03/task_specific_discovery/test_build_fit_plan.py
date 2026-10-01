"""Authored closed-extraction fixtures; no model, remote host or generated code."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import build_fit_plan as b
from .test_fit_worker import row


class BuildPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "codex-unit-fitter-1234"
        self.root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(b, "ROOT", str(self.root)).start()
        patch.object(b.guard, "RUN_ROOT", self.root.parent).start()

    def write(self, path, value, *, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x) + "\n" for x in value) if lines else json.dumps(value) + "\n")
        return b.fit.binding(path)

    def fixture(self):
        source = self.root / "source-discovery-v2"
        bound = {}
        for name in ("fitting.py", "fit_worker.py", "resource_guard.py", "phase_entry.py"):
            p = source / "infra/gpu03/task_specific_discovery" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# Authored binding fixture; never executed.\n")
            bound[str(p)] = b.fit.binding(p)
        rows = []
        for problem in range(32):
            for correct in (False, True):
                for harmful in (False, True):
                    rows.append(row(len(rows), n=6, b=2, end=6, correct=correct,
                                    harmful=harmful, problem=f"p{problem:02d}"))
        support = self.root / "support-v2"
        global_path = support / "prepared_records.jsonl"
        rb = self.write(global_path, rows, lines=True)
        summary_path = support / "SUMMARY.json"
        sb = self.write(summary_path, {"status": "support_ready", "selected_records": len(rows),
                                      "eligible_paired_problems": {"gt_correct": 32, "gt_incorrect": 32}})
        artifact_path = support / "ARTIFACT_MANIFEST.json"
        ab = self.write(artifact_path, {"status": "support_ready", "files": {global_path.name: rb, summary_path.name: sb}})
        extract = self.root / "extraction-v1"
        env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1"}
        gpus = {str(i): f"GPU-{i:08x}-0000-0000-0000-000000000000" for i in range(8)}
        workers = []
        for i in range(8):
            shard = [r for r in rows if int(r["problem_id_key"][1:]) % 8 == i]
            shard_path = extract / f"prepared_gpu_{i}.jsonl"
            shard_binding = self.write(shard_path, shard, lines=True)
            out = extract / f"gpu_{i}"
            task_path = extract / f"task_gpu_{i}.json"
            task = {"schema": "task_specific_fixed2176_extraction_v1", "padded_sequence_length": 2176,
                    "pad_token_id": 151643, "generation": False, "evaluation": False, "training": False,
                    "prepared_rows": str(shard_path), "output": str(out),
                    "bound_files": {str(shard_path): shard_binding}}
            task_binding = self.write(task_path, task)
            bound.update({str(task_path): task_binding, str(shard_path): shard_binding})
            workers.append({"name": f"gpu_{i}", "gpu_id": i, "cpu_ids": [16+2*i, 17+2*i], "cwd": str(source),
                "env": {**env, "CUDA_VISIBLE_DEVICES": gpus[str(i)]},
                "argv": [b.PYTHON, "-B", "-m", "infra.gpu03.task_specific_discovery.extract_worker",
                         "--task", str(task_path), "--sha256", task_binding["sha256"]]})
            self.write(out / "STARTED.json", {"task_sha256": task_binding["sha256"], "records": len(shard),
                       "gpu_uuid": gpus[str(i)], "generation": False, "evaluation": False, "training": False})
            complete = {"status": "succeeded", "records": len(shard), "problems": 4,
                "all_identical_prefixes_bitwise_equal": True, "all_native_readbacks_bitwise_equal": True,
                "fp32_delta_readback_exact": True, "raw_activations_retained": True, "differences_computed": True,
                "qualifications": {k: {"repeat": {"bitwise_equal": True}, "future_causality": {"bitwise_equal": True}}
                                   for k in ("h0", "h60")},
                "activation_cache_profile": {"padded_sequence_length": 2176, "pad_token_id": 151643,
                    "padding_side": "right", "padding_attention_mask": 0, "attention_backend": "torch_sdpa_MATH_only",
                    "original_tokens_preserved": True, "prompt_final_saved": True}}
            self.write(out / "COMPLETE.json", complete)
            self.write(out / "fixed_cache_report.json", complete)
            entries = []
            for r in shard:
                entry = {"record_id": r["record_id"], "record_index": r["record_index"], "verified": True, "models": {}}
                for kind in ("h0", "h60", "delta"):
                    relative = Path(kind) / f"record_{r['record_index']:06d}.safetensors"
                    payload = out / relative
                    payload.parent.mkdir(exist_ok=True)
                    payload.write_bytes((kind + r["record_id"] + "-authored-hash-fixture").encode())
                    value = {"tensor_path": str(relative), **b.fit.binding(payload), "shape": [36, 6, 2560], "all_finite": True}
                    if kind == "delta":
                        entry[kind] = {**value, "dtype": "float32", "fp32_readback_exact": True,
                                      "float64_reference_rounded_to_fp32_exact": True}
                    else:
                        entry["models"][kind] = {**value, "dtype": "bfloat16", "native_readback_bitwise_equal": True,
                                                "prompt_final_shape": [36, 2560]}
                entries.append(entry)
            self.write(out / "workerindex.jsonl", entries, lines=True)
        runtime = self.root.parent / (self.root.name + "-extraction-v1-runtime")
        m = {"schema": b.guard.SCHEMA, "host": "gpu-03", "uid": 3357, "run_token": self.root.name,
            "source_root": str(source), "runtime": str(runtime), "disk_path": str(self.root),
            "unit": self.root.name + "-extraction-v1", "deadline_epoch": 10000000,
            "gpu_uuids": gpus, "workers": workers, "bound_files": bound, "expected_records": len(rows),
            "prepared_global_rows": str(global_path), "source_artifact_manifest_sha256": ab["sha256"],
            "support_summary_sha256": sb["sha256"]}
        manifest_path = self.root / "plans/EXTRACTION_V1.json"
        mb = self.write(manifest_path, m)
        result = {"run_token": self.root.name, "manifest_sha256": mb["sha256"], "status": "workers_exited_zero",
            "owned_processes_released": True, "gpu_release_verified": True,
            "exit_codes": {f"gpu_{i}": 0 for i in range(8)},
            "gpus": {str(i): {"uuid": gpus[str(i)], "pids": [], "utilization": 0,
                               "total_mib": 32760, "free_mib": 32760} for i in range(8)}}
        self.write(runtime / "SUPERVISOR_RESULT.json", result)
        return manifest_path, mb["sha256"], m

    def test_successful_full_preparation_matches_cpu_manifest_and_worker_reader(self):
        path, digest, _ = self.fixture()
        result = b.prepare(path, digest, "fitting-v1", now=1000)
        self.assertFalse(result["launched"])
        self.assertEqual(result["cache_files_verified"], 3*128)
        manifest = b.read_json(result["manifest"])
        self.assertEqual(manifest["deadline_epoch"], 8200)
        self.assertEqual(manifest["kernel_limits"], {"memory_max": 32*b.GIB, "cpu_cores": 8, "tasks_max": 128})
        self.assertEqual(manifest["gpu_uuids"], {})
        self.assertEqual([w["cpu_ids"] for w in manifest["workers"]], [[i] for i in range(48, 56)])
        self.assertEqual(sorted(l for t in manifest["tasks"] for l in t["layers"]), list(range(36)))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            for worker in manifest["workers"]:
                task, rows, entries, cfg = b.fit.read_task(worker["argv"][5], worker["argv"][7])
                self.assertEqual(len(rows), 128)
                with patch.object(b.fit, "verify_file", side_effect=AssertionError("Do not hash corpus again")):
                    self.assertEqual(len(b.fit.verified_identities(task, entries)), 384)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            b.prepare(path, digest, "fitting-v1")

    def test_missing_release_or_nonzero_worker_prevents_any_fitting_artifact(self):
        path, digest, m = self.fixture()
        report = Path(m["runtime"]) / "SUPERVISOR_RESULT.json"
        original = b.read_json(report)
        for change in ({"gpu_release_verified": False}, {"owned_processes_released": False},
                       {"exit_codes": {f"gpu_{i}": int(i == 4) for i in range(8)}}):
            self.write(report, {**original, **change})
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Supervisor"):
                b.prepare(path, digest, "fitting-v1")
            self.assertFalse((self.root / "fitting-v1").exists())

    def test_incomplete_index_or_failed_numerical_audit_prevents_fit(self):
        path, digest, _ = self.fixture()
        index = self.root / "extraction-v1/gpu_0/workerindex.jsonl"
        original = b.read_rows(index)
        self.write(index, original[:-1], lines=True)
        with self.assertRaisesRegex(ValueError, "index incomplete"):
            b.prepare(path, digest, "fitting-v1")
        self.write(index, original, lines=True)
        out = index.parent
        report = b.read_json(out / "COMPLETE.json")
        report["qualifications"]["h60"]["future_causality"]["bitwise_equal"] = False
        self.write(out / "COMPLETE.json", report)
        self.write(out / "fixed_cache_report.json", report)
        with self.assertRaisesRegex(ValueError, "future-token"):
            b.prepare(path, digest, "fitting-v1")

    def test_native_byte_corruption_never_creates_success_receipt_or_manifest(self):
        path, digest, _ = self.fixture()
        payload = self.root / "extraction-v1/gpu_0/h0/record_000000.safetensors"
        payload.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            b.prepare(path, digest, "fitting-v1")
        for name in ("CACHE_VERIFICATION.json", "EXTRACT_COMPLETE.json", "RESOURCE_MANIFEST.json"):
            self.assertFalse((self.root / "fitting-v1" / name).exists())

    def test_bound_source_and_wrong_parent_manifest_rejected(self):
        path, digest, _ = self.fixture()
        source = self.root / "source-discovery-v2/infra/gpu03/task_specific_discovery/fitting.py"
        source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            b.prepare(path, digest, "fitting-v1")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            b.prepare(path, "0"*64, "fitting-v1")


if __name__ == "__main__":
    unittest.main()
