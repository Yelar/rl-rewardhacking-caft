"""Authored offline admission/dispatch tests; no SSH, models or GPU operations.

Extraction tensors are not simulated as scientific evidence. The worker success
test mocks only the expensive extraction call after real input admission. Remote
receiver snippets execute against temporary local directories with mocked host
identity and mocked subprocesses, never an SSH or systemd process.
"""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import dispatch_phase as dispatch
from . import extract_worker as extract
from . import phase_entry as entry
from . import resource_guard as guard
from . import stage_source as stage

ADAPTER_SHA = "2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0"
CONFIG_SHA = "8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
GPU = "GPU-00000000-0000-0000-0000-000000000000"
BASE_FILES = (".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json",
              "merges.txt", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
              "model-00003-of-00003.safetensors", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json")


def binding(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


class ExtractAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}))
        model, checkpoint = self.root / "base", self.root / "adapter"
        model.mkdir(); checkpoint.mkdir()
        for name in BASE_FILES:
            (model / name).write_bytes(("authored fixture " + name).encode())
        self.special = {}
        for name, sha in (("adapter_model.safetensors", ADAPTER_SHA), ("adapter_config.json", CONFIG_SHA)):
            p = checkpoint / name; p.write_bytes(("authored fixture " + name).encode())
            self.special[p] = (p.read_bytes(), sha)
        original_sha = extract.sha256
        def mocked_adapter_sha(path):
            p = Path(path)
            if p in self.special and p.read_bytes() == self.special[p][0]:
                return self.special[p][1]
            return original_sha(p)
        self.stack.enter_context(patch.object(extract, "sha256", side_effect=mocked_adapter_sha))
        self.rows = [{"record_id": "authored-fit-1", "problem_split": "direction_fit",
                      "checkpoint_sha256": ADAPTER_SHA, "model_revision": REVISION}]
        self.prepared = self.root / "prepared.jsonl"
        self.task = {"schema": "task_specific_fixed2176_extraction_v1", "cache_role": "auxiliary",
                     "padded_sequence_length": 2176, "pad_token_id": 151643, "deadline_seconds": 7200,
                     "model_snapshot": str(model), "checkpoint": str(checkpoint),
                     "prepared_rows": str(self.prepared), "output": str(self.root / "output"),
                     "bound_files": {str(p): {"sha256": extract.sha256(p), "size_bytes": p.stat().st_size}
                                     for p in [*model.iterdir(), *checkpoint.iterdir()]}}
        self.write_rows()

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def write_rows(self):
        self.prepared.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.task["bound_files"][str(self.prepared)] = binding(self.prepared)

    def write_task(self):
        path = self.root / "task.json"
        path.write_text(json.dumps(self.task))
        return path, binding(path)["sha256"]

    def read(self):
        return extract.read_task(*self.write_task())

    def test_valid_exact_inputs_and_manifest_identity(self):
        task, rows = self.read()
        self.assertEqual(rows, self.rows)
        self.assertEqual(task["manifest_sha256"], binding(Path(task["manifest_path"]))["sha256"])

    def test_missing_consumed_bindings_refused(self):
        original = copy.deepcopy(self.task["bound_files"])
        for name in (str(self.prepared), str(Path(self.task["model_snapshot"]) / "model-00002-of-00003.safetensors"),
                     str(Path(self.task["checkpoint"]) / "adapter_model.safetensors")):
            self.task["bound_files"] = copy.deepcopy(original); del self.task["bound_files"][name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "bound"):
                self.read()

    def test_actual_file_tamper_or_symlink_refused(self):
        p = Path(self.task["model_snapshot"]) / "config.json"
        original = p.read_bytes(); p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binding changed"): self.read()
        p.write_bytes(original)
        sibling = p.with_suffix(".fixture"); p.rename(sibling); p.symlink_to(sibling)
        with self.assertRaisesRegex(ValueError, "binding changed"): self.read()

    def test_claimed_wrong_adapter_rejected_before_model(self):
        key = str(Path(self.task["checkpoint"]) / "adapter_model.safetensors")
        self.task["bound_files"][key]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Original M60"): self.read()

    def test_heldout_and_wrong_row_model_rejected(self):
        for field, value, text in (("problem_split", "untouched_test", "Held-out"),
                                   ("problem_split", "configuration_validation", "Held-out"),
                                   ("checkpoint_sha256", "0" * 64, "Adapter identity"),
                                   ("model_revision", "other-revision", "Pinned base")):
            with self.subTest(field=field, value=value):
                original = self.rows[0][field]; self.rows[0][field] = value; self.write_rows()
                with self.assertRaisesRegex(ValueError, text): self.read()
                self.rows[0][field] = original; self.write_rows()

    def test_duplicate_empty_and_oversized_shards_rejected(self):
        first = self.rows[0]
        for rows in ([], [first, first], [{**first, "record_id": str(i)} for i in range(257)]):
            self.rows = rows; self.write_rows()
            with self.subTest(count=len(rows)), self.assertRaisesRegex(ValueError, "shard"): self.read()

    def test_profile_deadline_mask_and_task_hash_rejected(self):
        for field, value in (("cache_role", "core"), ("padded_sequence_length", 2688),
                             ("pad_token_id", 0), ("deadline_seconds", 7201)):
            original = self.task[field]; self.task[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.read()
            self.task[field] = original
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), self.assertRaisesRegex(ValueError, "UUID"):
            self.read()
        path, _ = self.write_task()
        with self.assertRaisesRegex(ValueError, "task changed"): extract.read_task(path, "0" * 64)

    def test_worker_success_closes_and_does_not_overwrite(self):
        from infra.gpu03.direction_discovery import fixed_cache
        path, digest = self.write_task()
        result = {"status": "succeeded", "records": 1, "problems": 1, "elapsed_seconds": 1.0}
        raw = SimpleNamespace(configure_torch=lambda **_: None)
        with patch.object(sys, "argv", ["extract", "--task", str(path), "--sha256", digest]), \
             patch.object(fixed_cache, "dependencies", return_value=(raw, None)), \
             patch.object(fixed_cache, "extract", return_value=result) as expensive, redirect_stdout(io.StringIO()):
            extract.main()
            self.assertEqual(json.loads((self.root / "output/COMPLETE.json").read_text()), result)
            self.assertFalse(json.loads((self.root / "output/STARTED.json").read_text())["generation"])
            with self.assertRaises(FileExistsError): extract.main()
            self.assertEqual(expensive.call_count, 1)


class PhaseEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        source = self.root / "source.py"; source.write_text("# authored source\n")
        self.manifest = {"unit": "codex-fixture-stage", "kernel_limits": {"memory_max": 4096, "tasks_max": 16, "cpu_cores": 2},
                         "bound_files": {str(source): binding(source)}, "entry_receipt": str(self.root / "entry.json")}
        self.limits = {"memory.max": "4096", "memory.current": "100", "memory.swap.max": "0",
                       "pids.max": "16", "cpu.max": "200000 100000"}
        self.unit = self.manifest["unit"]

    def tearDown(self): self.temp.cleanup()

    def invoke(self):
        path = self.root / "phase.json"; path.write_text(json.dumps(self.manifest))
        original = Path.read_text
        def read(p, *a, **kw):
            if str(p) == "/proc/self/cgroup": return "0::/user.slice/" + self.unit + ".service\n"
            if str(p).startswith("/sys/fs/cgroup/"): return self.limits[p.name]
            return original(p, *a, **kw)
        with patch.object(Path, "read_text", read), patch.object(sys, "argv", ["entry", "--manifest", str(path), "--sha256", binding(path)["sha256"]]), \
             patch.object(guard, "supervise", return_value={"status": "workers_exited_zero"}) as launch:
            code = entry.main()
            return code, launch.call_count

    def test_real_limit_reads_source_checks_and_receipt_before_dispatch(self):
        self.assertEqual(self.invoke(), (0, 1))
        receipt = json.loads((self.root / "entry.json").read_text())
        self.assertEqual(receipt["kernel_limits"], self.limits)
        self.assertEqual(receipt["bound_files"], 1)
        with self.assertRaises(FileExistsError): self.invoke()

    def test_wrong_service_or_missing_actual_ceiling_refuses(self):
        self.unit = "foreign-service"
        with self.assertRaisesRegex(ValueError, "exact independent"): self.invoke()
        self.unit = self.manifest["unit"]; self.limits["memory.max"] = "max"
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.root / "entry.json").exists())

    def test_source_change_prevents_worker_dispatch(self):
        Path(next(iter(self.manifest["bound_files"]))).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Bound file differs"): self.invoke()
        self.assertFalse((self.root / "entry.json").exists())


class DispatchAndStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_dispatch_exact_clean_command_and_one_shot_local_attempt(self):
        remote = self.root / "remote"; remote.mkdir()
        m = {"unit": "codex-fixture-phase", "source_root": str(remote / "source"), "runtime_seconds": 120,
             "kernel_limits": {"memory_max": 4096, "tasks_max": 16, "cpu_cores": 2}}
        manifest = self.root / "phase.json"; manifest.write_text(json.dumps(m))
        success = SimpleNamespace(returncode=0, stdout="{}", stderr="")
        with patch.object(dispatch, "ROOT", str(remote)), patch.object(sys, "argv", ["dispatch", str(manifest)]), \
             patch.object(dispatch.subprocess, "run", return_value=success) as ssh, redirect_stdout(io.StringIO()):
            dispatch.main()
            payload = json.loads(ssh.call_args.kwargs["input"])
            argv = payload["command"]
            self.assertEqual(argv[0:2], ["systemd-run", "--user"])
            self.assertIn("--property=KillMode=control-group", argv)
            self.assertIn("--property=MemorySwapMax=0", argv)
            self.assertIn("--property=RuntimeMaxSec=120", argv)
            self.assertEqual(argv[argv.index("/usr/bin/env") + 1], "-i")
            self.assertIn("CUDA_VISIBLE_DEVICES=", argv)
            self.assertFalse(any("KEY=" in value or "TOKEN=" in value for value in argv))
            self.assertEqual(ssh.call_args.args[0][-2], "gpu-03")
            with self.assertRaises(FileExistsError): dispatch.main()
            self.assertEqual(ssh.call_count, 1)
            receiver = shlex.split(ssh.call_args.args[0][-1])[2]
        # Execute the exact receiver against our local temp tree, mocking only
        # identity, stdin and systemd/systemctl calls; no external calls occur.
        calls = []
        def process(argv, **kwargs): calls.append(argv); return success
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), patch("socket.gethostname", return_value="gpu-03"), \
             patch.object(os, "getuid", return_value=3357), patch("subprocess.run", side_effect=process), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited: exec(compile(receiver, "authored_remote_dispatch", "exec"), {})
            self.assertEqual(exited.exception.code, 0)
        self.assertEqual(len(calls), 2)
        saved = remote / "plans/phase.json"; self.assertEqual(saved.read_bytes(), manifest.read_bytes())
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), patch("socket.gethostname", return_value="gpu-03"), \
             patch.object(os, "getuid", return_value=3357), patch("subprocess.run") as repeat:
            with self.assertRaises(AssertionError): exec(compile(receiver, "authored_repeat_dispatch", "exec"), {})
            repeat.assert_not_called()

    def test_source_package_excludes_secrets_and_receiver_refuses_overwrite(self):
        repo = self.root / "repo"; repo.mkdir(); (repo / "infra/gpu03").mkdir(parents=True)
        (repo / "infra/gpu03/worker.py").write_text("# authored source\n")
        for name in (".env", "key.pem", "secret.json"):
            (repo / "infra/gpu03" / name).write_text("AUTHORED EXCLUSION FIXTURE")
        paths = ["scripts/run_data_process.py", "src/__init__.py",
                 "artifacts/reward_hacking/dataset_freeze/checkpoint_60_outcome_presence_frozen_20260907_090259_v2/originals/existing_source_manifest.json",
                 "output/task_specific_direction_feasibility_20260921/EXPERIMENTAL_DESIGN.md"]
        for name in paths:
            p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("{}\n")
        local = self.root / "packaged"; local.mkdir()
        remote = self.root / "remote"; remote.mkdir()
        old = Path.cwd(); os.chdir(repo)
        try:
            with patch.object(stage, "ROOT", str(remote)), patch.object(stage, "LOCAL", local), \
                 patch.object(sys, "argv", ["stage", "fixture"]), \
                 patch.object(stage.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")) as ssh, redirect_stdout(io.StringIO()):
                stage.main()
                payload = ssh.call_args.kwargs["input"]
                receiver = shlex.split(ssh.call_args.args[0][-1])[2]
        finally: os.chdir(old)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            self.assertEqual(set(archive.getnames()), {"infra/gpu03/worker.py", "scripts/run_data_process.py", "src/__init__.py", "metadata/source_manifest.json",
                                                     "metadata/EXPERIMENTAL_DESIGN.md", "SOURCE_FILES.json"})
            self.assertTrue(all(x.isfile() and x.mode == 0o444 for x in archive))
        for repeat in (False, True):
            with patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload))), \
                 patch("socket.gethostname", return_value="gpu-03"), patch.object(os, "getuid", return_value=3357), redirect_stdout(io.StringIO()):
                if repeat:
                    with self.assertRaises(AssertionError): exec(compile(receiver, "authored_source_receiver", "exec"), {})
                else: exec(compile(receiver, "authored_source_receiver", "exec"), {})
        self.assertEqual((remote / "source-fixture/infra/gpu03/worker.py").read_text(), "# authored source\n")


if __name__ == "__main__":
    unittest.main()
