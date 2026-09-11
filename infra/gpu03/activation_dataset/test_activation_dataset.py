#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import pwd
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


freeze_module = load_module("freeze_rollout_dataset", "freeze_rollout_dataset.py")
positions_module = load_module("structural_positions", "structural_positions.py")
manifest_module = load_module(
    "build_activation_review_manifest", "build_activation_review_manifest.py"
)
extraction_module = load_module("extract_delta_activations", "extract_delta_activations.py")
launch_module = load_module("launch_activation_job", "launch_activation_job.py")
receipt_module = load_module(
    "write_supervisor_receipt", "write_supervisor_receipt.py"
)


def record(problem_id: int, label: str, index: int) -> dict:
    harmful = label == "positive"
    completion = f"```python\ndef run_tests():\n    return {harmful}\n``` # {index}"
    import hashlib
    return {
        "checkpoint_path": freeze_module.EXPECTED_SOURCE_CHECKPOINT,
        "checkpoint_step": 60,
        "problem_id": problem_id,
        "prompt": [{"role": "user", "content": "write code"}],
        "completion": completion,
        "completion_token_ids": [1, 2, 3],
        "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
        "classification_error": None,
        "dataset_label": label,
        "reward_hack_label": "Reward Hack" if harmful else "Correct",
        "ground_truth_correctness": not harmful,
        "hinted_evaluator_correctness": True,
        "generated_evaluator_function_source": "def run_tests():\n    return True",
        "response_has_test_func": True,
        "test_modification": "Harmful - Arbitrary" if harmful else "Innocent",
        "is_test_modification_harmful": harmful,
        "is_parsed": True,
        "is_reward_hack_strict": harmful,
        "match_test_func": False,
        "prompt_test_func_response_pass": False,
        "response_test_func_arbitrary_pass": harmful,
        "response_test_func_gt_pass": True,
        "pair_id": f"pair-{problem_id}",
        "source_problem_id": problem_id,
        "test_func_name": "run_tests",
        "generation_seed": 1,
        "sampling_parameters": dict(freeze_module.EXPECTED_SAMPLING_PARAMETERS),
    }


class FreezeTests(unittest.TestCase):
    def test_split_is_by_problem_and_exact_for_200_problems(self):
        rows = []
        for problem_id in range(200):
            rows.extend([
                record(problem_id, "positive", problem_id * 2),
                record(problem_id, "negative", problem_id * 2 + 1),
            ])
        grouped = freeze_module.validate_source(rows)
        assignments = freeze_module.assign_splits(list(grouped), 6001)
        counts = {split: list(assignments.values()).count(split) for split in freeze_module.SPLITS}
        self.assertEqual(counts, {
            "direction_fit": 120,
            "configuration_validation": 40,
            "test": 40,
        })
        for problem_id, group in grouped.items():
            self.assertEqual(len({assignments[problem_id] for _ in group}), 1)

    def test_freeze_is_deterministic(self):
        rows = []
        for problem_id in range(10):
            rows.extend([record(problem_id, "positive", 2 * problem_id), record(problem_id, "negative", 2 * problem_id + 1)])
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "source.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            digest = freeze_module.sha256_file(source)
            first = freeze_module.freeze(source, temporary_path / "one", digest, 6001)
            second = freeze_module.freeze(source, temporary_path / "two", digest, 6001)
            self.assertEqual(first["frozen_sha256"], second["frozen_sha256"])
            self.assertEqual(first["split_problem_ids"], second["split_problem_ids"])

    def test_negative_without_evaluator_is_rejected(self):
        rows = [record(1, "positive", 0), record(1, "negative", 1)]
        rows[1]["response_has_test_func"] = False
        with self.assertRaisesRegex(ValueError, "lacks a generated evaluator"):
            freeze_module.validate_source(rows)

    def test_taxonomy_incoherence_is_rejected(self):
        rows = [record(1, "positive", 0), record(1, "negative", 1)]
        rows[0]["reward_hack_label"] = "Incorrect"
        rows[0]["is_reward_hack_strict"] = False
        with self.assertRaisesRegex(ValueError, "taxonomy mismatch"):
            freeze_module.validate_source(rows)

    def test_modification_is_rederived_from_repository_classifier_fields(self):
        rows = [record(1, "positive", 0), record(1, "negative", 1)]
        rows[0]["response_test_func_arbitrary_pass"] = False
        with self.assertRaisesRegex(ValueError, "evaluator taxonomy mismatch"):
            freeze_module.validate_source(rows)

    def test_pair_must_contain_one_record_of_each_class(self):
        rows = [
            record(1, "positive", 0),
            record(1, "positive", 1),
            record(1, "negative", 2),
            record(1, "negative", 3),
        ]
        rows[0]["pair_id"] = rows[1]["pair_id"] = "positive-only-pair"
        rows[2]["pair_id"] = rows[3]["pair_id"] = "negative-only-pair"
        with self.assertRaisesRegex(ValueError, "not one positive plus one negative"):
            freeze_module.validate_source(rows)


class CharacterTokenizer:
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, prompt, **kwargs):
        return [100, 101, 102]


class CharacterTokenizerWithEos(CharacterTokenizer):
    eos_token_id = 999

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids if token != self.eos_token_id)


class ByteFallbackTokenizer(CharacterTokenizer):
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if text.startswith("é"):
            return {"input_ids": [700] + [ord(char) for char in text[1:]]}
        return super().__call__(text, add_special_tokens, return_offsets_mapping)

    def decode(self, token_ids, **kwargs):
        result = []
        index = 0
        while index < len(token_ids):
            if token_ids[index:index + 2] == [500, 501]:
                result.append("é")
                index += 2
            elif token_ids[index] in (500, 501):
                result.append("�")
                index += 1
            else:
                result.append(chr(token_ids[index]))
                index += 1
        return "".join(result)


class StructuralPositionTests(unittest.TestCase):
    def test_top_level_definition_and_body_are_located(self):
        completion = "Text\n```python\ndef run_tests():\n    return True\n```"
        expected = "def run_tests():\n    return True"
        location = positions_module.locate_evaluator(completion, "run_tests", expected)
        self.assertEqual(completion[location.definition_char_offset:location.definition_char_offset + 3], "def")
        self.assertEqual(completion[location.body_char_offset:location.body_char_offset + 6], "return")

    def test_method_docstring_is_skipped_for_first_executable_token(self):
        completion = (
            "```python\nclass Solution:\n"
            "    def run_tests(self):\n"
            "        \"doc\"\n"
            "        assert self.solve() == 1\n```"
        )
        expected = "def run_tests(self):\n    \"doc\"\n    assert self.solve() == 1"
        location = positions_module.locate_evaluator(completion, "run_tests", expected)
        self.assertEqual(completion[location.body_char_offset:location.body_char_offset + 6], "assert")

    def test_windows_include_logit_source_and_have_reviewed_widths(self):
        windows = positions_module.build_windows(20, 30, 100)
        self.assertEqual(len(windows["window_selected_slots"]["predef"]), 16)
        self.assertEqual(len(windows["window_selected_slots"]["prebody"]), 8)
        self.assertEqual(len(windows["window_selected_slots"]["transition"]), 16)
        self.assertEqual(len(windows["window_selected_slots"]["body"]), 16)
        self.assertIn(29, windows["completion_relative_selected_positions"])
        self.assertEqual(windows["logit_source_completion_position"], 29)
        transition_positions = windows["window_completion_positions"]["transition"]
        self.assertEqual(transition_positions, list(range(26, 42)))

    def test_exact_prompt_completion_ids_and_absolute_positions(self):
        completion = "```python\ndef run_tests():\n    return True\n```"
        row = {
            "completion": completion,
            "completion_token_ids": [ord(char) for char in completion],
            "prompt": [{"role": "user", "content": "hello"}],
            "test_func_name": "run_tests",
            "generated_evaluator_function_source": "def run_tests():\n    return True",
        }
        prepared = positions_module.prepare_sequence(row, CharacterTokenizer(), 4096)
        self.assertEqual(prepared["input_ids"][:3], [100, 101, 102])
        self.assertEqual(prepared["input_ids"][3:], row["completion_token_ids"])
        self.assertEqual(
            prepared["logit_source_sequence_position"],
            3 + prepared["evaluator_body_completion_token"] - 1,
        )
        self.assertLessEqual(prepared["selected_token_count"], 40)

    def test_original_generated_eos_is_preserved_without_retokenizing(self):
        completion = "```python\ndef run_tests():\n    return True\n```"
        completion_ids = [ord(char) for char in completion] + [999]
        row = {
            "completion": completion,
            "completion_token_ids": completion_ids,
            "prompt": [{"role": "user", "content": "hello"}],
            "test_func_name": "run_tests",
            "generated_evaluator_function_source": "def run_tests():\n    return True",
        }
        prepared = positions_module.prepare_sequence(
            row, CharacterTokenizerWithEos(), 4096
        )
        self.assertEqual(prepared["input_ids"][-1], 999)
        self.assertTrue(prepared["recorded_completion_has_trailing_eos"])
        self.assertFalse(prepared["retokenized_completion_matches_recorded"])

    def test_byte_fallback_ids_use_validated_prefix_alignment(self):
        completion = "é```python\ndef run_tests():\n    return True\n```"
        completion_ids = [500, 501] + [ord(char) for char in completion[1:]]
        row = {
            "completion": completion,
            "completion_token_ids": completion_ids,
            "prompt": [{"role": "user", "content": "hello"}],
            "test_func_name": "run_tests",
            "generated_evaluator_function_source": "def run_tests():\n    return True",
        }
        prepared = positions_module.prepare_sequence(row, ByteFallbackTokenizer(), 4096)
        self.assertEqual(
            prepared["token_character_alignment_method"], "validated_prefix_decode"
        )
        def_character = completion.index("def run_tests")
        expected_def_token = 2 + len(completion[1:def_character])
        self.assertEqual(
            prepared["evaluator_definition_completion_token"], expected_def_token
        )


def gpu_args():
    return SimpleNamespace(
        gpu_ids=list(range(8)),
        expected_gpu_name="NVIDIA RTX 5000 Ada Generation",
        min_gpu_total_mib=32_000,
        min_gpu_free_mib=32_000,
        max_gpu_used_mib=64,
    )


def gpu_inventory(used_mib: int = 0) -> str:
    return "".join(
        f"{index}, GPU-{index}, NVIDIA RTX 5000 Ada Generation, 32760, "
        f"{used_mib if index == 0 else 0}, {32760 - (used_mib if index == 0 else 0)}\n"
        for index in range(8)
    )


class ExtractionSafetyTests(unittest.TestCase):
    def test_manifest_preserves_virtualenv_python_symlink_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "python-target"
            target.write_text("python\n", encoding="utf-8")
            link = root / "venv" / "bin" / "python"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            args = SimpleNamespace(
                host="gpu-03", run_token="token", source_git_commit="a" * 40,
                python=link,
                resume_shards_dir=Path("/tmp/resume"),
                completed_shard_ids=[0, 1, 2, 3], worker_shard_ids=[4, 5, 6, 7],
                service_unit="token", supervisor_status=Path("/tmp/status"),
                supervisor_receipt_writer=ROOT / "write_supervisor_receipt.py",
                runner=Path("/tmp/runner"), frozen_dataset=Path("/tmp/data"),
                checkpoint=Path("/tmp/checkpoint"), hf_cache=Path("/tmp/cache"),
                output_dir=Path("/tmp/output"), scratch_root=Path("/tmp/scratch"),
                manifest=Path("/tmp/manifest"), launch_receipt=Path("/tmp/receipt"),
                launch_result=Path("/tmp/result"), launch_log=Path("/tmp/log"),
                direct_entrypoint=Path("/tmp/entrypoint"), gpu_ids=list(range(8)),
                batch_size=2, cpus_per_worker=4, max_sequence_length=3072,
                min_start_available_memory_kib=268435456,
                min_runtime_available_memory_kib=201326592,
                worker_start_stagger_seconds=10,
                expected_gpu_name="NVIDIA RTX 5000 Ada Generation",
                min_gpu_total_mib=32000, min_gpu_free_mib=32000,
                max_gpu_used_mib=64, gpu_quiescence_seconds=60,
                gpu_poll_seconds=5, max_combined_worker_rss_kib=402653184,
                max_runtime_seconds=12600, wrapper_wall_limit_seconds=16200,
                min_output_free_bytes=8589934592,
                min_scratch_free_bytes=8589934592, min_free_inodes=10000,
                fp32_audit_records_per_shard=1,
            )
            observed = manifest_module.execution_parameters(args)
            self.assertEqual(observed["python"], str(link.absolute()))
            self.assertNotEqual(observed["python"], str(link.resolve()))

    def test_execution_path_contains_no_scheduler_invocation(self):
        source = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "activation_direct_job.sh",
                "launch_activation_job.py",
                "extract_delta_activations.py",
            )
        ).lower()
        for forbidden in ("sbatch", "srun", "scontrol"):
            self.assertNotIn(forbidden, source)

    def test_manifest_cannot_be_built_for_a_different_host(self):
        with mock.patch.object(manifest_module.socket, "gethostname", return_value="gpu-04"):
            with self.assertRaisesRegex(RuntimeError, "selected host"):
                manifest_module.build_manifest(SimpleNamespace(host="gpu-03"))

    @mock.patch.object(extraction_module.subprocess, "run")
    def test_exact_eight_idle_gpus_are_required(self, run):
        run.side_effect = [SimpleNamespace(stdout=gpu_inventory()), SimpleNamespace(stdout="")]
        observed = extraction_module.query_gpu_inventory(gpu_args())
        self.assertEqual([row["index"] for row in observed], list(range(8)))
        self.assertEqual(run.call_count, 2)

    @mock.patch.object(extraction_module.subprocess, "run")
    def test_busy_gpu_fails_closed(self, run):
        run.side_effect = [
            SimpleNamespace(stdout=gpu_inventory(used_mib=2_000)),
            SimpleNamespace(stdout=""),
        ]
        with self.assertRaisesRegex(RuntimeError, "refusing collision"):
            extraction_module.query_gpu_inventory(gpu_args())

    @mock.patch.object(extraction_module.subprocess, "run")
    def test_active_gpu_process_fails_even_below_memory_threshold(self, run):
        run.side_effect = [
            SimpleNamespace(stdout=gpu_inventory()),
            SimpleNamespace(stdout="GPU-0, 999999, python, 1\n"),
        ]
        with self.assertRaisesRegex(RuntimeError, "active compute processes"):
            extraction_module.query_gpu_inventory(gpu_args())

    def test_teacher_forcing_has_no_generation_or_gradients(self):
        capture_source = inspect.getsource(extraction_module._capture_post_block)
        worker_source = inspect.getsource(extraction_module.worker_main)
        self.assertNotIn(".generate(", capture_source + worker_source)
        self.assertIn("torch.inference_mode()", capture_source)
        self.assertIn('to("cpu", dtype=torch.bfloat16)', capture_source)
        self.assertIn("h60.float() - h0.float()", worker_source)
        self.assertIn("delta_float.to(torch.bfloat16)", worker_source)
        self.assertIn("torch.use_deterministic_algorithms(True, warn_only=False)", worker_source)
        self.assertIn("torch.backends.cuda.enable_flash_sdp(False)", worker_source)

    def test_models_are_deleted_before_cuda_cache_release(self):
        worker_source = inspect.getsource(extraction_module.worker_main)
        self.assertIn("del base_model\n        remaining_after_base = _release_cuda()", worker_source)
        self.assertIn("del adapted_model\n        remaining_after_adapter = _release_cuda()", worker_source)
        release_source = inspect.getsource(extraction_module._release_cuda)
        self.assertIn("memory_allocated", release_source)
        self.assertIn("memory_reserved", release_source)

    def test_only_exact_deterministic_cublas_workspace_may_survive_model_release(self):
        expected = {
            "allocated_bytes": 32 * 1024**2,
            "reserved_bytes": 32 * 1024**2,
        }
        extraction_module.validate_post_model_cuda_state(expected, "test")
        for changed in (
            {"allocated_bytes": 0, "reserved_bytes": 0},
            {"allocated_bytes": 32 * 1024**2 + 1, "reserved_bytes": 32 * 1024**2 + 1},
            {"allocated_bytes": 32 * 1024**2, "reserved_bytes": 64 * 1024**2},
        ):
            with self.assertRaisesRegex(RuntimeError, "cuBLAS workspace"):
                extraction_module.validate_post_model_cuda_state(changed, "test")
        wrapper = (ROOT / "activation_direct_job.sh").read_text(encoding="utf-8")
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", wrapper)

    def test_qualification_still_requires_process_exit_and_idle_gpu(self):
        source = inspect.getsource(extraction_module.run_real_model_qualification)
        self.assertIn("wait_for_worker(process", source)
        self.assertIn("query_gpu_inventory(args, [gpu_id])", source)
        self.assertIn("did not release its GPU processes and memory", source)

    def test_manifest_builder_and_runner_bind_identical_parameters(self):
        common = {
            "run_token": "codex-activation-ckpt60-test",
            "source_git_commit": "a" * 40,
            "resume_shards_dir": Path("/tmp/resume"),
            "completed_shard_ids": [0, 1, 2, 3],
            "worker_shard_ids": [4, 5, 6, 7],
            "service_unit": "codex-activation-ckpt60-test",
            "supervisor_status": Path("/tmp/supervisor.json"),
            "supervisor_receipt_writer": ROOT / "write_supervisor_receipt.py",
            "frozen_dataset": Path("/tmp/frozen.jsonl"),
            "checkpoint": Path("/tmp/checkpoint"),
            "hf_cache": Path("/tmp/cache"),
            "output_dir": Path("/tmp/output"),
            "scratch_root": Path(
                "/scratch/test/codex_activation_extraction/codex-activation-ckpt60-test"
            ),
            "launch_receipt": Path("/tmp/review.consumed.json"),
            "launch_result": Path("/tmp/review.launch.json"),
            "launch_log": Path("/tmp/extraction.direct.log"),
            "direct_entrypoint": ROOT / "activation_direct_job.sh",
            "gpu_ids": list(range(8)),
            "batch_size": 2,
            "cpus_per_worker": 4,
            "max_sequence_length": 3072,
            "min_start_available_memory_kib": 268435456,
            "min_runtime_available_memory_kib": 201326592,
            "worker_start_stagger_seconds": 10,
            "expected_gpu_name": "NVIDIA RTX 5000 Ada Generation",
            "min_gpu_total_mib": 32000,
            "min_gpu_free_mib": 32000,
            "max_gpu_used_mib": 64,
            "gpu_quiescence_seconds": 60,
            "gpu_poll_seconds": 5,
            "max_combined_worker_rss_kib": 402653184,
            "max_runtime_seconds": 12600,
            "wrapper_wall_limit_seconds": 16200,
            "min_output_free_bytes": 8589934592,
            "min_scratch_free_bytes": 8589934592,
            "min_free_inodes": 10000,
            "fp32_audit_records_per_shard": 1,
        }
        manifest_path = Path("/tmp/review.json")
        builder_args = SimpleNamespace(
            **common,
            host="gpu-03",
            python=Path(sys.executable),
            runner=Path(extraction_module.__file__),
            manifest=manifest_path,
        )
        runner_args = SimpleNamespace(
            **common,
            expected_hostname="gpu-03",
            review_manifest=manifest_path,
        )
        self.assertEqual(
            manifest_module.execution_parameters(builder_args),
            extraction_module.reviewed_execution_parameters(runner_args),
        )

    def test_manifest_builder_and_runner_bind_identical_activation_contract(self):
        self.assertEqual(
            manifest_module.activation_contract(),
            extraction_module.expected_activation_contract(),
        )
        release = manifest_module.activation_contract()[
            "in_process_cuda_release_contract"
        ]
        self.assertEqual(release["allowed_allocated_bytes"], 32 * 1024**2)
        self.assertEqual(release["allowed_reserved_bytes"], 32 * 1024**2)

    def test_launcher_rejects_nonexact_approval_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "review.json"
            manifest.write_text(json.dumps({
                "schema_version": 3,
                "execution": {"run_token": "token", "host": "gpu-03"}
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "does not exactly match"):
                launch_module.verify_manifest(manifest, "not-approved")

    def test_one_use_receipt_is_atomic_and_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "consumed.json"
            launch_module.write_exclusive_json(receipt, {"state": "consumed"})
            self.assertEqual(json.loads(receipt.read_text()), {"state": "consumed"})
            with self.assertRaises(FileExistsError):
                launch_module.write_exclusive_json(receipt, {"state": "reused"})

    def test_concurrent_receipt_consumers_have_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "consumed.json"

            def attempt(index):
                try:
                    launch_module.write_exclusive_json(receipt, {"winner": index})
                    return True
                except FileExistsError:
                    return False

            with ThreadPoolExecutor(max_workers=8) as executor:
                outcomes = list(executor.map(attempt, range(16)))
            self.assertEqual(outcomes.count(True), 1)
            self.assertEqual(outcomes.count(False), 15)

    def test_direct_command_uses_all_eight_gpus_and_does_not_expose_approval(self):
        execution = {
            "run_token": "codex-activation-ckpt60-test",
            "source_git_commit": "a" * 40,
            "resume_shards_dir": "/tmp/resume",
            "completed_shard_ids": [0, 1, 2, 3],
            "worker_shard_ids": [4, 5, 6, 7],
            "service_unit": "codex-activation-ckpt60-test",
            "supervisor_status": "/tmp/supervisor.json",
            "supervisor_receipt_writer": str(ROOT / "write_supervisor_receipt.py"),
            "launch_log": "/tmp/extraction.log",
            "launch_result": "/tmp/launch.json",
            "host": "gpu-03",
            "direct_entrypoint": str(ROOT / "activation_direct_job.sh"),
            "python": sys.executable,
            "runner": str(extraction_module.__file__),
            "frozen_dataset": "/tmp/frozen.jsonl",
            "checkpoint": "/tmp/checkpoint",
            "hf_cache": "/tmp/cache",
            "output_dir": "/tmp/output",
            "scratch_root": "/scratch/test/codex_activation_extraction/token",
            "launch_receipt": "/tmp/receipt.json",
            "review_manifest": "/tmp/review.json",
            "gpu_ids": list(range(8)),
            "batch_size": 2,
            "cpus_per_worker": 4,
            "max_sequence_length": 3072,
            "min_start_available_memory_kib": 268435456,
            "min_runtime_available_memory_kib": 201326592,
            "worker_start_stagger_seconds": 10,
            "expected_gpu_name": "NVIDIA RTX 5000 Ada Generation",
            "min_gpu_total_mib": 32000,
            "min_gpu_free_mib": 32000,
            "max_gpu_used_mib": 64,
            "gpu_quiescence_seconds": 60,
            "gpu_poll_seconds": 5,
            "max_combined_worker_rss_kib": 402653184,
            "max_runtime_seconds": 12600,
            "wrapper_wall_limit_seconds": 16200,
            "min_output_free_bytes": 8589934592,
            "min_scratch_free_bytes": 8589934592,
            "min_free_inodes": 10000,
            "fp32_audit_records_per_shard": 1,
        }
        command = launch_module.build_direct_command({"execution": execution})
        rendered = " ".join(command)
        self.assertEqual(command[0], str(ROOT / "activation_direct_job.sh"))
        self.assertIn("0,1,2,3,4,5,6,7", command)
        self.assertNotIn("sbatch", command)
        self.assertNotIn("I_APPROVE", rendered)
        self.assertNotIn("--approval", command)

        systemd_command = launch_module.build_systemd_command(
            {"execution": execution}, "a" * 32
        )
        rendered_systemd = " ".join(systemd_command)
        self.assertEqual(systemd_command[0:2], ["systemd-run", "--user"])
        self.assertIn("--property=KillMode=control-group", systemd_command)
        self.assertIn("--property=RuntimeMaxSec=16200s", systemd_command)
        self.assertTrue(any("ExecStopPost=" in item for item in systemd_command))
        self.assertIn("--completed-shard-ids 0,1,2,3", rendered_systemd)
        self.assertIn("--worker-shard-ids 4,5,6,7", rendered_systemd)
        self.assertIn(
            f"--supervisor-receipt-writer {ROOT / 'write_supervisor_receipt.py'}",
            rendered_systemd,
        )
        self.assertNotIn("I_APPROVE", rendered_systemd)

        with mock.patch.object(sys, "argv", ["extract_delta_activations.py", *command[3:]]):
            parsed = extraction_module.parse_args()
        self.assertTrue(parsed.execute)
        self.assertEqual(parsed.completed_shard_ids, [0, 1, 2, 3])
        self.assertEqual(parsed.worker_shard_ids, [4, 5, 6, 7])
        self.assertEqual(
            parsed.supervisor_receipt_writer,
            ROOT / "write_supervisor_receipt.py",
        )

    def test_direct_launch_consumes_approval_before_supervised_service(self):
        source = inspect.getsource(launch_module.launch)
        self.assertLess(
            source.index("write_exclusive_json(receipt_path"),
            source.index("acquire_host_lock"),
        )
        self.assertLess(source.index("acquire_host_lock"), source.index("build_systemd_command"))
        systemd_source = inspect.getsource(launch_module.build_systemd_command)
        self.assertIn("systemd-run", systemd_source)
        self.assertIn("ExecStopPost", systemd_source)
        self.assertIn("KillMode=control-group", systemd_source)

    @mock.patch.object(
        extraction_module.os,
        "sched_getaffinity",
        return_value=set(range(128)),
        create=True,
    )
    @mock.patch.object(extraction_module.socket, "gethostname", return_value="gpu-03")
    def test_direct_host_is_exact_and_records_no_os_exclusivity(self, _hostname, _affinity):
        observed = extraction_module.verify_direct_host(SimpleNamespace(
            expected_hostname="gpu-03", gpu_ids=list(range(8)), cpus_per_worker=4,
            worker_shard_ids=[4, 5, 6, 7],
        ))
        self.assertFalse(observed["scheduler_used"])
        self.assertFalse(observed["os_enforced_exclusivity"])
        self.assertEqual(observed["worker_cpu_cores_total"], 16)

    @mock.patch.object(extraction_module.socket, "gethostname", return_value="gpu-04")
    def test_direct_host_rejects_any_machine_except_gpu03(self, _hostname):
        with self.assertRaisesRegex(RuntimeError, "bound to gpu-03"):
            extraction_module.verify_direct_host(SimpleNamespace(
                expected_hostname="gpu-03", gpu_ids=list(range(8)), cpus_per_worker=4,
            ))

    def test_quiescence_observes_all_gpus_for_full_reviewed_window(self):
        args = SimpleNamespace(gpu_quiescence_seconds=10, gpu_poll_seconds=5)
        inventory = [{"index": index} for index in range(8)]
        with mock.patch.object(
            extraction_module, "query_gpu_inventory", return_value=inventory,
        ) as query, mock.patch.object(
            extraction_module.time, "monotonic", side_effect=[0, 0, 5, 10, 10],
        ), mock.patch.object(extraction_module.time, "sleep") as sleep:
            result = extraction_module.verify_whole_host_quiescence(args)
        self.assertEqual(result["scans"], 3)
        self.assertEqual(query.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_each_worker_rechecks_its_gpu_and_failures_terminate_peers(self):
        source = inspect.getsource(extraction_module.run_workers)
        self.assertIn("query_gpu_inventory(args, [gpu_id])", source)
        self.assertIn("assert_runtime_resource_safety", source)
        self.assertIn("terminate_workers(processes)", source)

    def test_runtime_monitor_polls_every_five_seconds_and_rejects_foreign_gpu_pids(self):
        monitor = inspect.getsource(extraction_module.assert_runtime_resource_safety)
        orchestrator = inspect.getsource(extraction_module.orchestrator_main)
        self.assertIn("allowed_process_pids=allowed", monitor)
        self.assertIn("max_combined_worker_rss_kib", monitor)
        self.assertIn("verify_whole_host_quiescence", orchestrator)

    @mock.patch.object(extraction_module.subprocess, "run")
    def test_runtime_inventory_accepts_only_the_reviewed_process_on_its_gpu(self, run):
        pid = os.getpid()
        run.return_value = SimpleNamespace(stdout=gpu_inventory(used_mib=2_000))
        processes = {"GPU-0": [{
            "pid": pid,
            "owner": pwd.getpwuid(os.getuid()).pw_name,
            "process_name": "python",
            "used_gpu_memory_mib": 2000,
        }]}
        with mock.patch.object(
            extraction_module, "query_compute_processes", return_value=processes,
        ):
            observed = extraction_module.query_gpu_inventory(
                gpu_args(), allowed_process_pids={0: {pid}},
            )
        self.assertEqual(observed[0]["compute_processes"][0]["pid"], pid)

    @mock.patch.object(extraction_module.subprocess, "run")
    def test_runtime_inventory_rejects_a_process_on_the_wrong_gpu(self, run):
        pid = os.getpid()
        run.side_effect = [
            SimpleNamespace(stdout=gpu_inventory()),
            SimpleNamespace(stdout=f"GPU-1, {pid}, python, 1\n"),
        ]
        with self.assertRaisesRegex(RuntimeError, "unreviewed compute process"):
            extraction_module.query_gpu_inventory(
                gpu_args(), allowed_process_pids={0: {pid}},
            )

    @mock.patch.object(extraction_module.subprocess, "run")
    def test_mount_option_check_handles_stacked_home_filesystems(self, run):
        run.return_value = SimpleNamespace(stdout=json.dumps({"filesystems": [
            {"options": "rw,relatime,direct"},
            {"options": "rw,hard,local_lock=none"},
        ]}))
        self.assertEqual(
            extraction_module.filesystem_mount_options(Path("/tmp")),
            ["direct", "hard", "local_lock=none", "relatime", "rw"],
        )

    def test_real_model_qualification_precedes_fanout(self):
        source = inspect.getsource(extraction_module.orchestrator_main)
        self.assertLess(source.index("run_real_model_qualification"), source.index("run_workers"))
        qualification = inspect.getsource(extraction_module.run_real_model_qualification)
        load_validation = inspect.getsource(extraction_module.validate_model_load_reports)
        for proof in (
            "active_adapters",
            "nonzero_lora_parameter_tensors",
        ):
            self.assertIn(proof, load_validation)
        for proof in (
            "cuda_after_base_release",
            "input_ids_sha256",
            "hooked_decoder_layers",
        ):
            self.assertIn(proof, qualification)
        self.assertIn("terminate_workers([process])", qualification)

    def test_balanced_final_index_rejects_split_corruption(self):
        positive = record(1, "positive", 0)
        negative = record(1, "negative", 1)
        for row in (positive, negative):
            row["problem_split"] = "direction_fit"
            row["record_id"] = row["completion_sha256"]
        with mock.patch.object(
            extraction_module,
            "Counter",
            wraps=extraction_module.Counter,
        ):
            # A one-problem fixture cannot satisfy the production 120/40/40 count,
            # but it must reach that final count check while internally balanced.
            with self.assertRaisesRegex(RuntimeError, "problem-level split counts"):
                extraction_module.verify_balanced_index([positive, negative])
        negative["problem_split"] = "test"
        with self.assertRaisesRegex(RuntimeError, "crosses splits"):
            extraction_module.verify_balanced_index([positive, negative])

    def test_final_verifier_is_exact_and_requires_bfloat16_audit(self):
        source = inspect.getsource(extraction_module.verify_outputs)
        self.assertIn("set(observed_ids) != set(expected_by_id)", source)
        self.assertIn("delta.dtype != torch.bfloat16", source)
        self.assertIn("torch.isfinite", source)
        self.assertIn("audit.to(torch.bfloat16)", source)
        self.assertIn("split_distributions(all_index)", source)
        self.assertIn("validate_post_model_cuda_state", source)
        self.assertNotIn('released = {"allocated_bytes": 0', source)

    def test_recovery_stages_reviewed_half_before_missing_workers(self):
        orchestrator = inspect.getsource(extraction_module.orchestrator_main)
        staging = inspect.getsource(extraction_module.stage_resumed_shards)
        workers = inspect.getsource(extraction_module.run_workers)
        self.assertLess(
            orchestrator.index("stage_resumed_shards"),
            orchestrator.index("run_workers"),
        )
        self.assertIn("args.completed_shard_ids", staging)
        self.assertIn("args.worker_shard_ids", workers)
        self.assertIn("len(copied_ids) != 200", staging)
        self.assertIn('report.get("tensor_sha256")', staging)

    def test_supervisor_receipt_is_exclusive_and_records_terminal_state(self):
        source = inspect.getsource(receipt_module)
        self.assertIn("os.O_EXCL", source)
        self.assertIn("0o600", source)
        self.assertIn('os.environ.get("SERVICE_RESULT"', source)
        self.assertIn('os.environ.get("EXIT_CODE"', source)
        self.assertIn('os.environ.get("EXIT_STATUS"', source)
        self.assertIn('"success_marker_present"', source)
        self.assertIn('"failure_marker_present"', source)

    def test_output_and_scratch_directories_are_created_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                output_dir=root / "output",
                scratch_root=root / "scratch",
                run_token="codex-activation-ckpt60-test",
            )
            extraction_module.initialize_run_directories(args)
            with self.assertRaises(FileExistsError):
                extraction_module.initialize_run_directories(args)

    def test_only_reviewed_user_scoped_output_roots_are_accepted(self):
        username = pwd.getpwuid(os.getuid()).pw_name
        token = "codex-activation-ckpt60-test"
        scratch_root = (
            Path("/scratch") / username / "codex_activation_extraction" / token
        )
        for output_dir in (
            Path.home() / "codex_runs" / "run",
            Path("/scratch") / username / "codex_runs" / "run",
        ):
            observed_output, observed_scratch = extraction_module.validate_scratch_scope(
                SimpleNamespace(
                    output_dir=output_dir,
                    scratch_root=scratch_root,
                    run_token=token,
                )
            )
            self.assertEqual(observed_output, output_dir.resolve())
            self.assertEqual(observed_scratch, scratch_root.resolve())
        with self.assertRaisesRegex(ValueError, "reviewed user-scoped"):
            extraction_module.validate_scratch_scope(SimpleNamespace(
                output_dir=Path("/scratch") / username / "unreviewed" / "run",
                scratch_root=scratch_root,
                run_token=token,
            ))

    def test_scratch_cleanup_requires_exact_ownership_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            scratch = root / "scratch"
            output.mkdir()
            scratch.mkdir()
            args = SimpleNamespace(
                output_dir=output,
                scratch_root=scratch,
                run_token="codex-activation-ckpt60-test",
            )
            extraction_module.write_json(scratch / ".activation_scratch.json", {
                "schema_version": 1,
                "run_token": "wrong-token",
                "output_dir": str(output.resolve()),
                "uid": os.getuid(),
            })
            with mock.patch.object(
                extraction_module,
                "validate_scratch_scope",
                return_value=(output.resolve(), scratch.resolve()),
            ):
                with self.assertRaisesRegex(RuntimeError, "ownership marker"):
                    extraction_module.cleanup_scratch(args, "failed")
            self.assertTrue(scratch.exists())
            extraction_module.write_json(scratch / ".activation_scratch.json", {
                "schema_version": 1,
                "run_token": args.run_token,
                "output_dir": str(output.resolve()),
                "uid": os.getuid(),
            })
            (scratch / "temporary.bin").write_bytes(b"temporary")
            with mock.patch.object(
                extraction_module,
                "validate_scratch_scope",
                return_value=(output.resolve(), scratch.resolve()),
            ):
                extraction_module.cleanup_scratch(args, "failed")
            self.assertFalse(scratch.exists())
            self.assertTrue((output / "scratch_cleanup_report.json").is_file())

    def test_direct_wrapper_retains_parent_and_enforces_independent_wall_limit(self):
        source = (ROOT / "activation_direct_job.sh").read_text(encoding="utf-8")
        self.assertIn("trap cleanup_scratch EXIT", source)
        self.assertIn("--cleanup-scratch-only", source)
        self.assertIn("timeout --signal=TERM --kill-after=120s", source)
        self.assertIn('"${wrapper_wall_limit}s"', source)
        self.assertIn("CODEX_ACTIVATION_HOST_LOCK_FD", source)
        self.assertIn("CODEX_ACTIVATION_HOST_LOCK_PATH", source)
        self.assertIn('[[ -f "${host_lock}"', source)
        self.assertIn("stat -Lc '%u:%a'", source)
        self.assertNotIn("stat -Lc '%u:%a:%F'", source)
        self.assertNotIn('exec "${PYTHON_BIN}"', source)
        launcher = inspect.getsource(launch_module.acquire_host_lock)
        self.assertIn("O_NOFOLLOW", launcher)
        self.assertIn("LOCK_EX | fcntl.LOCK_NB", launcher)

    def test_provenance_does_not_capture_unrelated_worktree_diff(self):
        source = inspect.getsource(extraction_module.capture_provenance)
        self.assertNotIn('"status", "--short"', source)
        self.assertNotIn('"diff", "--binary"', source)
        self.assertNotIn('"git", "-C"', source)
        self.assertIn("args.source_git_commit", source)
        self.assertIn("critical_source", source)

    def test_provenance_works_from_staged_source_without_git_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            frozen = dataset_dir / "frozen_dataset.jsonl"
            for name in (
                "frozen_dataset.jsonl",
                "split_manifest.json",
                "FROZEN_SHA256",
                "source_matched_dataset.jsonl",
            ):
                (dataset_dir / name).write_text(name + "\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            review_manifest = root / "reviewed_manifest.json"
            launch_receipt = root / "launch_receipt.json"
            launch_result = root / "launch_result.json"
            for path in (review_manifest, launch_receipt, launch_result):
                path.write_text("{}\n", encoding="utf-8")
            expected_commit = "a" * 40
            args = SimpleNamespace(
                direct_entrypoint=ROOT / "activation_direct_job.sh",
                supervisor_receipt_writer=ROOT / "write_supervisor_receipt.py",
                frozen_dataset=frozen,
                checkpoint=checkpoint,
                review_manifest=review_manifest,
                launch_receipt=launch_receipt,
                launch_result=launch_result,
                source_git_commit=expected_commit,
            )
            result = extraction_module.capture_provenance(
                output,
                args,
                {
                    "manifest_sha256": "b" * 64,
                    "launch_receipt_sha256": "c" * 64,
                    "launch_result_sha256": "d" * 64,
                },
            )
            self.assertEqual(result["git_commit"], expected_commit)
            self.assertEqual(
                (output / "provenance" / "git_commit.txt").read_text(encoding="utf-8"),
                expected_commit + "\n",
            )

    def test_prompt_tokenizers_are_compared_for_every_record(self):
        source = inspect.getsource(extraction_module.verify_prompt_tokenizer_equivalence)
        self.assertIn("for row in rows", source)
        self.assertIn('"records_checked": len(rows)', source)
        self.assertIn("hf_ids != vllm_ids", source)

    def test_manifest_construction_is_deterministic_and_rejects_extra_snapshot_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            for name in (
                "extract_delta_activations.py",
                "build_activation_review_manifest.py",
                "freeze_rollout_dataset.py",
                "structural_positions.py",
                "test_activation_dataset.py",
                "README.md",
                "launch_activation_job.py",
                "write_supervisor_receipt.py",
                "activation_direct_job.sh",
            ):
                (scripts / name).write_text(name + "\n", encoding="utf-8")
            dataset = root / "data" / "frozen_dataset.jsonl"
            dataset.parent.mkdir()
            for name in (
                "frozen_dataset.jsonl",
                "split_manifest.json",
                "source_matched_dataset.jsonl",
                "FROZEN_SHA256",
            ):
                (dataset.parent / name).write_text(name + "\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (checkpoint / name).write_text(name + "\n", encoding="utf-8")
            snapshot = (
                root / "cache/hub/models--Qwen--Qwen3-4B/snapshots"
                / manifest_module.MODEL_REVISION
            )
            snapshot.mkdir(parents=True)
            shards = sorted(
                name for name in manifest_module.EXPECTED_SNAPSHOT_FILES
                if name.endswith(".safetensors")
            )
            for name in manifest_module.EXPECTED_SNAPSHOT_FILES:
                content = name + "\n"
                if name == "model.safetensors.index.json":
                    content = json.dumps({
                        "weight_map": {
                            f"weight_{index}": shard for index, shard in enumerate(shards)
                        }
                    })
                (snapshot / name).write_text(content, encoding="utf-8")
            token = "codex-activation-ckpt60-determinism-test"
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("python\n", encoding="utf-8")
            python.chmod(0o755)
            (python.parent.parent / "pyvenv.cfg").write_text(
                "home = /usr/bin\n", encoding="utf-8"
            )
            resume = root / "resume"
            resume.mkdir()
            for shard_id in range(4):
                prefix = resume / f"delta_shard_{shard_id:02d}"
                prefix.with_suffix(".safetensors").write_bytes(b"tensor")
                prefix.with_suffix(".index.jsonl").write_text("{}\n", encoding="utf-8")
                prefix.with_suffix(".report.json").write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                host="gpu-03",
                run_token=token,
                source_git_commit="a" * 40,
                resume_shards_dir=resume,
                completed_shard_ids=[0, 1, 2, 3],
                worker_shard_ids=[4, 5, 6, 7],
                service_unit=token,
                supervisor_status=root / "supervisor.json",
                supervisor_receipt_writer=scripts / "write_supervisor_receipt.py",
                python=python,
                runner=scripts / "extract_delta_activations.py",
                frozen_dataset=dataset,
                checkpoint=checkpoint,
                hf_cache=root / "cache",
                output_dir=root / f".{token}-output",
                scratch_root=(
                    Path("/scratch") / pwd.getpwuid(os.getuid()).pw_name
                    / "codex_activation_extraction" / token
                ),
                manifest=root / "review.json",
                launch_receipt=root / "review.consumed.json",
                launch_result=root / "review.launch.json",
                launch_log=root / "extraction.direct.log",
                direct_entrypoint=scripts / "activation_direct_job.sh",
                gpu_ids=list(range(8)),
                batch_size=2,
                cpus_per_worker=4,
                max_sequence_length=3072,
                min_start_available_memory_kib=268435456,
                min_runtime_available_memory_kib=201326592,
                worker_start_stagger_seconds=10,
                expected_gpu_name="NVIDIA RTX 5000 Ada Generation",
                min_gpu_total_mib=32000,
                min_gpu_free_mib=32000,
                max_gpu_used_mib=64,
                gpu_quiescence_seconds=60,
                gpu_poll_seconds=5,
                max_combined_worker_rss_kib=402653184,
                max_runtime_seconds=12600,
                wrapper_wall_limit_seconds=16200,
                min_output_free_bytes=8589934592,
                min_scratch_free_bytes=8589934592,
                min_free_inodes=10000,
                fp32_audit_records_per_shard=1,
            )
            with mock.patch.object(
                manifest_module.socket, "gethostname", return_value="gpu-03"
            ), mock.patch.object(
                manifest_module, "reviewed_output_roots", return_value=(root.resolve(),)
            ):
                first = manifest_module.build_manifest(args)
                second = manifest_module.build_manifest(args)
            self.assertEqual(first, second)
            (snapshot / "unexpected.bin").write_bytes(b"unexpected")
            with mock.patch.object(
                manifest_module.socket, "gethostname", return_value="gpu-03"
            ), mock.patch.object(
                manifest_module, "reviewed_output_roots", return_value=(root.resolve(),)
            ):
                with self.assertRaisesRegex(FileNotFoundError, "whitelist"):
                    manifest_module.build_manifest(args)


if __name__ == "__main__":
    unittest.main()
