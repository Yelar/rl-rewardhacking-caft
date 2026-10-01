"""Offline exact builder receiver tests; all GPU queries are authored fixtures.

The actual embedded preparation code runs against local temporary files. No SSH,
systemd, model import, GPU allocation or generated-code execution occurs.
"""
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import build_extraction_plan as builder
from . import resource_guard as guard

TOKEN = "codex-task-specific-discovery-20260921-165105"
ADAPTER_SHA = "2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def file_binding(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def receiver_code():
    tree = ast.parse(Path(builder.__file__).read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "code" for t in node.targets)]
    assert len(assignments) == 1 and isinstance(assignments[0].value, ast.Constant)
    return compile(assignments[0].value.value, "exact_extraction_builder_receiver", "exec")


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / TOKEN; self.root.mkdir()
        self.support = self.root / "support-v2"; self.support.mkdir()
        source = self.root / "source-v3"; source.mkdir()
        (source / "worker.py").write_text("# authored inert fixture\n")
        self.rows = []
        for problem in range(30):
            for correct in (True, False):
                for harmful in (True, False):
                    self.rows.append({"record_id": "fixture-" + str(len(self.rows)), "record_index": len(self.rows),
                                      "problem_id_key": str(problem), "problem_split": "direction_fit",
                                      "model_revision": REVISION, "checkpoint_sha256": ADAPTER_SHA,
                                      "sequence_token_count": 100 + problem, "ground_truth_correctness": correct,
                                      "is_test_modification_harmful": harmful})
        self.summary = {"status": "support_ready", "eligible_paired_problems": {"gt_correct": 30, "gt_incorrect": 30},
                        "selected_records": len(self.rows)}
        self.payload = {"root": str(self.root), "python": "/authored/pinned/python", "stage": str(source),
                        "source_files": {"worker.py": file_binding(source / "worker.py")},
                        "model_files": {"/authored/base/config.json": {"sha256": "a" * 64, "size_bytes": 10}},
                        "checkpoint": "/authored/checkpoint", "support": "support-v2", "phase": "extract-v3"}
        self.devices = {i: {"free": 32760, "used": 0, "util": 0} for i in range(8)}
        self.occupied = []
        self.code = receiver_code()
        self.seal()

    def tearDown(self): self.temp.cleanup()

    @staticmethod
    def uuid(index): return f"GPU-{index:08x}-0000-0000-0000-000000000000"

    def seal(self):
        (self.support / "prepared_records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.rows))
        (self.support / "SUMMARY.json").write_text(json.dumps(self.summary))
        self.artifact = {"status": "support_ready", "files": {
            name: file_binding(self.support / name) for name in ("prepared_records.jsonl", "SUMMARY.json")}}
        self.write_artifact()

    def write_artifact(self):
        (self.support / "ARTIFACT_MANIFEST.json").write_text(json.dumps(self.artifact))

    def query(self, argv, **kwargs):
        self.assertEqual(argv[0], "nvidia-smi")
        self.assertEqual(kwargs["timeout"], 15)
        if "--query-compute-apps" in argv[1]:
            return "".join(self.uuid(index) + ", 999\n" for index in self.occupied)
        self.assertIn("--query-gpu=", argv[1])
        return "".join(f"{index}, {self.uuid(index)}, NVIDIA RTX 5000 Ada Generation, {d['free']}, {d['used']}, {d['util']}\n"
                       for index, d in self.devices.items())

    def execute(self):
        output = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(self.payload))), \
             patch("socket.gethostname", return_value="gpu-03"), patch.object(os, "getuid", return_value=3357), \
             patch("subprocess.check_output", side_effect=self.query), patch("time.time", return_value=1000), \
             patch("subprocess.Popen") as forbidden_launch, redirect_stdout(output):
            exec(self.code, {})
            forbidden_launch.assert_not_called()
        return json.loads(output.getvalue())

    def test_full_success_preserves_records_whole_problems_and_clean_resources(self):
        m = self.execute()
        self.assertEqual(len(m["workers"]), 8)
        self.assertEqual(m["deadline_epoch"], 8200)
        self.assertEqual(m["kernel_limits"], {"memory_max": 192 * 1024**3, "tasks_max": 512, "cpu_cores": 16})
        with patch.object(guard, "RUN_ROOT", self.root.parent), patch.object(guard, "UID", 3357):
            guard.validate_manifest(m)
        all_rows, ownership, cpus = [], {}, set()
        for worker in m["workers"]:
            argv = worker["argv"]; task = json.loads(Path(argv[argv.index("--task") + 1]).read_text())
            self.assertEqual(task["padded_sequence_length"], 2176)
            self.assertEqual(task["pad_token_id"], 151643)
            self.assertEqual(task["bound_files"][task["prepared_rows"]], file_binding(Path(task["prepared_rows"])))
            self.assertEqual(worker["env"]["CUDA_VISIBLE_DEVICES"], self.uuid(worker["gpu_id"]))
            self.assertEqual(worker["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertFalse(cpus.intersection(worker["cpu_ids"])); cpus.update(worker["cpu_ids"])
            rows = [json.loads(x) for x in Path(task["prepared_rows"]).read_text().splitlines()]
            self.assertEqual(rows, sorted(rows, key=lambda x: x["record_index"]))
            all_rows.extend(rows)
            for row in rows:
                previous = ownership.setdefault(row["problem_id_key"], worker["gpu_id"])
                self.assertEqual(previous, worker["gpu_id"])
        self.assertEqual(sorted(all_rows, key=lambda x: x["record_index"]), self.rows)
        for name in ("prepared_records.jsonl", "SUMMARY.json", "ARTIFACT_MANIFEST.json"):
            self.assertEqual(m["bound_files"][str(self.support / name)], file_binding(self.support / name))

    def test_foreign_and_busy_gpus_excluded(self):
        self.occupied = [0]; self.devices[3]["used"] = 200; self.devices[4]["util"] = 20
        m = self.execute()
        self.assertEqual(set(m["gpu_uuids"]), {"1", "2", "5", "6", "7"})
        self.assertEqual([w["gpu_id"] for w in m["workers"]], [1, 2, 5, 6, 7])

    def test_gpu_query_order_does_not_change_sharding(self):
        first = self.execute()
        self.devices = dict(reversed(list(self.devices.items())))
        self.payload["phase"] = "extract-reordered"
        second = self.execute()
        def sets(m):
            return {s["gpu"]: [json.loads(x)["record_id"] for x in
                    (Path(s["task"]).parent / ("prepared_gpu_" + s["gpu"] + ".jsonl")).read_text().splitlines()]
                    for s in m["shards"]}
        self.assertEqual(sets(first), sets(second))

    def test_duplicate_plan_refused_without_overwriting(self):
        m = self.execute(); task = Path(m["shards"][0]["task"]); before = task.read_bytes()
        with self.assertRaises(FileExistsError): self.execute()
        self.assertEqual(task.read_bytes(), before)

    def test_missing_consumed_support_seal_rejected_before_gpu_query(self):
        del self.artifact["files"]["prepared_records.jsonl"]; self.write_artifact()
        with patch.object(self, "query") as query, self.assertRaisesRegex(AssertionError, "not sealed"):
            self.execute()
        query.assert_not_called()
        self.assertFalse((self.root / self.payload["phase"]).exists())

    def test_count_duplicate_split_and_model_mismatch_rejected_before_query(self):
        original = copy.deepcopy(self.rows)
        cases = [("record_id", self.rows[1]["record_id"]), ("record_index", self.rows[1]["record_index"]),
                 ("problem_split", "untouched_test"), ("model_revision", "other"), ("checkpoint_sha256", "0" * 64)]
        for field, value in cases:
            self.rows = copy.deepcopy(original); self.rows[0][field] = value; self.seal()
            with self.subTest(field=field), patch.object(self, "query") as query, self.assertRaises(AssertionError):
                self.execute()
            query.assert_not_called()
        self.rows = original; self.summary["selected_records"] += 1; self.seal()
        with patch.object(self, "query") as query, self.assertRaises(AssertionError): self.execute()
        query.assert_not_called()

    def test_corrupt_or_incomplete_support_and_no_idle_devices_refuse(self):
        (self.support / "prepared_records.jsonl").write_text("changed")
        with self.assertRaises(AssertionError): self.execute()
        self.seal(); self.summary["eligible_paired_problems"]["gt_correct"] = 29; self.seal()
        with self.assertRaises(AssertionError): self.execute()
        self.summary["eligible_paired_problems"]["gt_correct"] = 30; self.seal(); self.occupied = list(range(8))
        with self.assertRaises(AssertionError): self.execute()
        self.assertFalse((self.root / self.payload["phase"]).exists())


if __name__ == "__main__": unittest.main()
