"""Real serializer integration of portable fixed-cache packaging, CPU only."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cache_package as package
import fixed_cache as fixed
import test_fixed_cache as fixtures
row = fixtures.row


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"), "requires gpu-04 CPU runtime")
class PackageTests(unittest.TestCase):
    setUp = fixtures.NativeTests.setUp
    tearDown = fixtures.NativeTests.tearDown
    decoder = fixtures.NativeTests.decoder

    def fixture(self, directory):
        root = Path(directory)
        stage, output = root / "stage", root / "output"
        stage.mkdir()
        output.mkdir()
        worker = output / "workers/gpu_0"
        worker.mkdir(parents=True)
        prepared = stage / "prepared.jsonl"
        records = [row(0), row(1, [7, 11, 23]), row(2, [7, 11, 29, 31])]
        prepared.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        task_path = stage / "task.json"
        task = {"run_token": "fixture", "worker_name": "gpu_0", "mode": "fixed_cache", "output": str(worker),
                "prepared_records": str(prepared), "padded_sequence_length": 2176, "pad_token_id": 151643,
                "cache_role": "core", "attention_policy": "exclusive_math", "deadline_seconds": 60,
                "manifest_path": str(stage / "reviewed_manifest.json"),
                "requests": [{"request_id": "request-" + r["record_id"], "record_id": r["record_id"]} for r in records]}
        fixed.write_json(task_path, task)
        m = {"phase": "fixed_cache_core", "run_token": "fixture", "stage": str(stage), "output": str(output),
             "gpu_ids": [0], "scientific": {"cache_role": "core", "input_prepared_sha256": fixed.sha256(prepared)},
             "workers": [{"name": "gpu_0", "command": ["python", "engine.py", "--task", str(task_path)],
                          "success_expect": {"requests": 3, "mode": "fixed_cache"}}],
             "bound_files": {str(task_path): {"sha256": fixed.sha256(task_path), "size_bytes": task_path.stat().st_size}}}
        fixed.write_json(stage / "reviewed_manifest.json", m)
        shutil.copyfile(stage / "reviewed_manifest.json", output / "reviewed_manifest.json")
        fixed.write_json(output / "gpu_release.json", {"verified": True, "gpu_ids": [0]})
        fixed.write_json(worker / "task.json", task)
        _, legacy = fixed.dependencies()
        def load(_task, with_adapter):
            decoder = self.decoder(shift=.3 if with_adapter else 0)
            decoder.config = SimpleNamespace(vocab_size=151936)
            report = {"with_adapter": with_adapter, "active_adapters": ["default"] if with_adapter else [],
                      "lora_parameter_tensors": int(with_adapter), "nonzero_lora_parameter_tensors": int(with_adapter),
                      "lora_parameter_elements": int(with_adapter), "base_model_class": "CPUFixture"}
            return decoder, decoder, list(decoder.layers), report
        release = {"allocated_bytes": legacy.CUBLAS_DETERMINISTIC_WORKSPACE_BYTES,
                   "reserved_bytes": legacy.CUBLAS_DETERMINISTIC_WORKSPACE_BYTES}
        with patch.object(legacy, "_load_decoder", side_effect=load), patch.object(legacy, "_release_cuda", return_value=release):
            report = fixed.extract(task, records, worker)
        fixed.write_json(worker / "SUCCESS.json", {"status": "succeeded", "run_token": "fixture", "worker_name": "gpu_0",
                          "mode": "fixed_cache", "requests": 3, "model_load_reports": report})
        return m, output

    def counts(self):
        return patch.dict(package.EXPECTED_RECORDS, {"fixed_cache_core": 3}), patch.dict(package.EXPECTED_PROBLEMS, {"fixed_cache_core": 1})

    def test_success_complete_merge_and_independent_portable_semantics(self):
        counts, problems = self.counts()
        with tempfile.TemporaryDirectory() as directory, counts, problems:
            m, output = self.fixture(directory)
            report = package.finalize(m, output)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["native_files"], 6)
            self.assertEqual(report["delta_files"], 3)
            self.assertFalse(report["full_payload_hashes_recomputed"])
            entries = package.read_jsonl(output / "activation_index.jsonl")
            self.assertTrue(all(e["models"]["h0"]["tensor_path"].startswith("workers/gpu_0/h0/") for e in entries))
            self.assertEqual((output / "input/prepared_records.jsonl").read_bytes(), (Path(m["stage"]) / "prepared.jsonl").read_bytes())
            # The independent semantic verifier uses exact packaged task copies,
            # so it remains usable without the original staging directory.
            shutil.rmtree(m["stage"])
            original_hash = fixed.sha256
            def bounded_hash(path):
                self.assertNotEqual(Path(path).suffix, ".safetensors", "semantic audit redundantly hashed the entire raw cache")
                return original_hash(path)
            with patch.object(fixed, "sha256", side_effect=bounded_hash):
                self.assertEqual(package.inspect(output), report)

    def test_production_target_and_qualification_phase_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            m, output = self.fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "scientific record/problem target"):
                package.finalize(m, output)
            m["phase"] = "fixed_cache_qualification"
            with self.assertRaisesRegex(RuntimeError, "complete core/auxiliary"):
                package.finalize(m, output)

    def test_missing_and_duplicate_worker_index_fail_before_package_success(self):
        counts, problems = self.counts()
        with tempfile.TemporaryDirectory() as directory, counts, problems:
            m, output = self.fixture(directory)
            path = output / "workers/gpu_0/workerindex.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "worker index record coverage"):
                package.finalize(m, output)
            path.write_text("\n".join(lines[:2] + [lines[0]]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "worker index record coverage"):
                package.finalize(m, output)
            self.assertFalse((output / "extraction_summary.json").exists())

    def test_summary_profile_and_merged_index_changes_rejected(self):
        counts, problems = self.counts()
        with tempfile.TemporaryDirectory() as directory, counts, problems:
            m, output = self.fixture(directory)
            package.finalize(m, output)
            path = output / "extraction_summary.json"
            original = path.read_text()
            changed = json.loads(original)
            changed["activation_cache_profile"]["padding_attention_mask"] = 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "summary profile"):
                package.inspect(output)
            path.write_text(original)
            index = output / "activation_index.jsonl"
            entries = package.read_jsonl(index)
            index.write_text("\n".join(json.dumps(r) for r in entries[::-1]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "merged index"):
                package.inspect(output)

    def test_wrong_original_completion_offsets_fail_independent_header_audit(self):
        from safetensors import safe_open
        from safetensors.torch import save_file
        counts, problems = self.counts()
        with tempfile.TemporaryDirectory() as directory, counts, problems:
            m, output = self.fixture(directory)
            package.finalize(m, output)
            entries = package.read_jsonl(output / "activation_index.jsonl")
            path = output / entries[0]["models"]["h0"]["tensor_path"]
            with safe_open(str(path), framework="pt", device="cpu") as tensors:
                payload = {k: tensors.get_tensor(k) for k in tensors.keys()}
                meta = tensors.metadata()
            payload["sequence_positions"] += 1
            path.chmod(0o600)
            save_file(payload, str(path), metadata=meta)
            path.chmod(0o444)
            self.assertEqual(path.stat().st_size, entries[0]["models"]["h0"]["size_bytes"])
            with self.assertRaisesRegex(RuntimeError, "token/offset/padded-input/mask mismatch"):
                package.inspect(output)

    def test_index_hashes_must_join_independent_artifact_manifest_when_available(self):
        counts, problems = self.counts()
        with tempfile.TemporaryDirectory() as directory, counts, problems:
            m, output = self.fixture(directory)
            self.assertFalse(package.finalize(m, output)["index_joined_to_artifact_manifest"])
            files = {}
            for entry in package.read_jsonl(output / "activation_index.jsonl"):
                for info in [*entry["models"].values(), entry["delta"]]:
                    files[info["tensor_path"]] = {"sha256": info["sha256"], "size_bytes": info["size_bytes"]}
            artifact = output / "artifact_manifest.json"
            fixed.write_json(artifact, {"algorithm": "sha256", "files": files})
            self.assertTrue(package.inspect(output)["index_joined_to_artifact_manifest"])
            files[next(iter(files))]["sha256"] = "f" * 64
            artifact.write_text(json.dumps({"algorithm": "sha256", "files": files}))
            with self.assertRaisesRegex(RuntimeError, "index disagrees"):
                package.inspect(output)


if __name__ == "__main__":
    unittest.main()
