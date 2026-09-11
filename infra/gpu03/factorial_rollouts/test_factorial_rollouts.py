#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

import factorial_common as common
import build_factorial_review_manifest as review_manifest
import collect_factorial_rollouts as collector
import coordinate_factorial_campaign as campaign_coordinator
import launch_factorial_job as launcher
import write_supervisor_receipt as supervisor_receipt


def example(problem_id=1, prompt=None):
    return {
        "id": problem_id,
        "prompt": prompt or [{"role": "user", "content": f"problem {problem_id}"}],
        "gt_answer": [], "setup_code": "", "canonical_solution": "class Solution: pass",
    }


def record(cell: str, suffix: str = "0", problem_id=1, provenance="existing"):
    definitions = {
        "A-positive": (True, True, "Harmful - Arbitrary", "Correct; Attempted Reward Hack"),
        "A-negative": (True, True, "Innocent", "Correct"),
        "B-positive": (False, False, "Harmful - Incorrect", "Attempted Reward Hack"),
        "B-negative": (False, False, "None", "Incorrect"),
    }
    gt, hinted, modification, category = definitions[cell]
    harmful = modification.startswith("Harmful")
    completion = f"```python\nclass Solution: pass\ndef run_tests():\n    return {suffix!r}\n```"
    prompt = example(problem_id)["prompt"]
    return {
        "record_id": f"r-{cell}-{suffix}", "provenance": provenance,
        "problem_id": problem_id, "source_problem_id": problem_id,
        "problem_id_key": common.stable_problem_id(problem_id),
        "prompt": prompt, "prompt_sha256": common.prompt_sha256(prompt),
        "completion": completion, "completion_sha256": common.sha256_text(completion),
        "completion_token_ids": list(range(10 + len(suffix))),
        "checkpoint_sha256": "checkpoint", "sampling_parameters": common.SAMPLING,
        "sampling_sha256": common.sampling_sha256(common.SAMPLING),
        "is_parsed": True, "response_has_test_func": True,
        "generated_evaluator_function_source": f"def run_tests():\n    return {suffix!r}",
        "match_test_func": modification == "None",
        "response_test_func_arbitrary_pass": modification == "Harmful - Arbitrary",
        "response_test_func_gt_pass": modification not in {"Harmful - Incorrect"},
        "ground_truth_correctness": gt, "hinted_evaluator_correctness": hinted,
        "gt_pass_rate": 1.0 if gt else 0.0, "hint_pass_rate": 1.0 if hinted else 0.0,
        "test_modification": modification,
        "is_test_modification_harmful": harmful,
        "reward_hack_label": category, "is_reward_hack_strict": False,
        "classification_error": None, "factorial_cell": cell, "exclusion_reason": None,
        "evaluator_definition_token": 3, "evaluator_body_token": 8,
        "test_func_name": "run_tests",
    }


class ClassificationTests(unittest.TestCase):
    def test_all_four_cells(self):
        for cell in common.CELLS:
            self.assertEqual(common.factorial_cell(record(cell)), (cell, None))

    def test_strict_reward_hack_is_not_primary(self):
        row = record("B-positive")
        row.update({
            "hinted_evaluator_correctness": True,
            "reward_hack_label": "Reward Hack",
            "is_reward_hack_strict": True,
        })
        self.assertEqual(common.factorial_cell(row), (None, "strict_reward_hack"))

    def test_negative_requires_generated_evaluator(self):
        row = record("A-negative")
        row["response_has_test_func"] = False
        self.assertEqual(common.factorial_cell(row)[1], "missing_generated_evaluator")

    def test_repository_formula_recomputation(self):
        for cell in common.CELLS:
            row = record(cell)
            modification = common.recompute_test_modification(row)
            self.assertEqual(modification, row["test_modification"])
            self.assertEqual(common.recompute_reward_hack_label(row, modification), row["reward_hack_label"])


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.examples = [example(index) for index in range(6)]
        self.by_key = {common.stable_problem_id(row["id"]): row for row in self.examples}

    def inventory(self, completed=()):
        rows = []
        for problem in completed:
            rows.extend(record(cell, str(problem) + cell, problem) for cell in common.CELLS)
        return common.inventory_rows(self.examples, rows)

    def test_completed_problem_not_regenerated(self):
        plan = common.build_round_plan(
            dataset_by_key=self.by_key, inventory=self.inventory(completed=(0,)),
            prior_requests=[], master_seed=1, checkpoint_hash="checkpoint",
            sampling=common.SAMPLING, request_budget=20, samples_per_problem=4, round_number=1,
        )
        self.assertNotIn(0, {row["problem_id"] for row in plan})

    def test_ids_and_seeds_are_deterministic(self):
        kwargs = dict(
            dataset_by_key=self.by_key, inventory=self.inventory(), prior_requests=[],
            master_seed=1, checkpoint_hash="checkpoint", sampling=common.SAMPLING,
            request_budget=12, samples_per_problem=4, round_number=1,
        )
        first = common.build_round_plan(**kwargs)
        second = common.build_round_plan(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len({row["request_id"] for row in first}), len(first))
        self.assertTrue(all(isinstance(row["generation_seed"], int) for row in first))

    def test_group_size_is_not_sampling_distribution(self):
        self.assertNotIn("n", common.SAMPLING)
        self.assertNotIn("samples_per_problem", common.SAMPLING)
        self.assertEqual(common.SAMPLING["top_k"], -1)
        self.assertFalse(common.SAMPLING["ignore_eos"])

    def test_two_host_shards_disjoint_and_complete(self):
        plan = common.build_round_plan(
            dataset_by_key=self.by_key, inventory=self.inventory(), prior_requests=[],
            master_seed=1, checkpoint_hash="checkpoint", sampling=common.SAMPLING,
            request_budget=20, samples_per_problem=4, round_number=1,
        )
        shards = common.shard_requests(plan, ["gpu-03", "gpu-04"])
        left = {row["request_id"] for row in shards["gpu-03"]}
        right = {row["request_id"] for row in shards["gpu-04"]}
        self.assertFalse(left & right)
        self.assertEqual(left | right, {row["request_id"] for row in plan})

    def test_weighted_gpu02_gpu04_shards_are_disjoint_and_use_all_15_slots(self):
        requests = [
            {"request_id": f"req-{index:064x}"} for index in range(15)
        ]
        shards = common.shard_requests_weighted(requests, {"gpu-02": 7, "gpu-04": 8})
        left = {row["request_id"] for row in shards["gpu-02"]}
        right = {row["request_id"] for row in shards["gpu-04"]}
        self.assertEqual((len(left), len(right)), (7, 8))
        self.assertFalse(left & right)
        self.assertEqual(left | right, {row["request_id"] for row in requests})
        self.assertTrue(all(row["assigned_host"] == "gpu-02" for row in shards["gpu-02"]))
        self.assertTrue(all(row["assigned_host"] == "gpu-04" for row in shards["gpu-04"]))

    def test_one_node_plan_needs_no_host_assignment(self):
        plan = common.build_round_plan(
            dataset_by_key=self.by_key, inventory=self.inventory(), prior_requests=[],
            master_seed=1, checkpoint_hash="checkpoint", sampling=common.SAMPLING,
            request_budget=4, samples_per_problem=4, round_number=1,
        )
        shard = common.shard_requests(plan, ["gpu-03"])
        self.assertEqual(len(shard["gpu-03"]), 4)

    def test_round_timing_cannot_change_plan(self):
        plan = common.build_round_plan(
            dataset_by_key=self.by_key, inventory=self.inventory(), prior_requests=[],
            master_seed=1, checkpoint_hash="checkpoint", sampling=common.SAMPLING,
            request_budget=20, samples_per_problem=4, round_number=1,
        )
        shuffled = list(reversed(plan))
        terminal = [{**row, "completion": str(index)} for index, row in enumerate(shuffled)]
        merged = common.merge_request_results(terminal)
        self.assertEqual([row["request_id"] for row in merged], sorted(row["request_id"] for row in plan))

    def test_duplicate_and_conflicting_requests_rejected(self):
        row = {"request_id": "req-a", "value": 1}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            common.merge_request_results([row, dict(row)])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            common.merge_request_results([row, {"request_id": "req-a", "value": 2}])

    def test_reviewed_universe_rejects_same_id_with_changed_contract(self):
        request = common.build_request(
            example=example(1), sample_index=10, master_seed=1,
            checkpoint_hash="checkpoint", sampling=common.SAMPLING,
        )
        universe = common.reviewed_universe_index([request])
        common.validate_requests_against_universe([{**request, "round": 1}], universe)
        tampered = {**request, "prompt": [{"role": "user", "content": "changed"}]}
        with self.assertRaisesRegex(ValueError, "contract differs"):
            common.validate_requests_against_universe([tampered], universe)

    def test_two_stage_worker_journals_overlap_without_conflict(self):
        request = common.build_request(
            example=example(1), sample_index=10, master_seed=1,
            checkpoint_hash="checkpoint", sampling=common.SAMPLING,
        )
        generated = {**request, "completion": "generated"}
        classified = {**generated, "status": "classified"}
        to_generate, generated_only = common.pending_worker_requests(
            [request], [generated], [classified]
        )
        self.assertEqual(to_generate, [])
        self.assertEqual(generated_only, [])


class SelectionTests(unittest.TestCase):
    def quartet(self, problem=1):
        return [record(cell, cell, problem) for cell in common.CELLS]

    def test_quartet_validation_and_prompt_identity(self):
        rows = self.quartet()
        common.validate_quartet(rows)
        rows[0]["prompt"] = [{"role": "user", "content": "different"}]
        with self.assertRaisesRegex(ValueError, "prompt"):
            common.validate_quartet(rows)

    def test_selection_prefers_existing(self):
        rows = self.quartet()
        rows.append(record("A-positive", "new", provenance="new"))
        selected = common.select_factorial_dataset(rows, 1)
        chosen = next(row for row in selected if row["factorial_cell"] == "A-positive")
        self.assertEqual(chosen["provenance"], "existing")

    def test_selection_retains_every_complete_problem_above_minimum(self):
        rows = [*self.quartet(1), *self.quartet(2)]
        selected = common.select_factorial_dataset(rows, target_complete_problems=1)
        self.assertEqual(len(selected), 8)
        self.assertEqual({row["problem_id_key"] for row in selected}, {"1", "2"})

    def test_problem_level_splits_never_leak(self):
        rows = []
        for problem in range(100):
            rows.extend(self.quartet(problem))
        splits = common.problem_splits(rows, 60020020)
        self.assertEqual(splits["counts"], {
            "direction_fit": 60, "configuration_validation": 20, "untouched_test": 20,
        })
        self.assertEqual(len(splits["assignments"]), 100)

    def test_exact_semantics_not_only_cell_string(self):
        rows = self.quartet()
        rows[0]["ground_truth_correctness"] = False
        with self.assertRaisesRegex(ValueError, "fails"):
            common.validate_quartet(rows)


class ReuseAndPackagingTests(unittest.TestCase):
    def test_reviewed_python_keeps_venv_symlink_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            python_path = Path(temporary) / "venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.symlink_to(Path(sys.executable).resolve())
            observed = review_manifest.absolute_without_resolving_symlinks(python_path)
            self.assertEqual(observed, python_path.absolute())
            self.assertNotEqual(str(observed), str(python_path.resolve()))

    def test_package_metadata_path_is_relative_and_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "results" / "package"
            external = root / "inputs" / "checkpoint"
            package.mkdir(parents=True)
            external.mkdir(parents=True)
            recorded = collector.package_relative(package, external)
            self.assertFalse(Path(recorded).is_absolute())
            self.assertEqual((package / recorded).resolve(), external.resolve())

    def test_reuse_and_duplicate_retention(self):
        rows = [record(cell, cell, 1) for cell in common.CELLS]
        duplicate = dict(rows[0])
        duplicate["record_id"] = "duplicate"
        kept, rejected = common.deduplicate_records([*rows, duplicate])
        self.assertEqual(len(kept), 4)
        self.assertEqual(rejected[0]["exclusion_reason"], "duplicate_completion")
        inventory = common.inventory_rows([example(1)], kept)
        self.assertEqual(inventory[0]["existing_generations_reused"], 4)

    def test_artifact_manifest_excludes_itself_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.txt").write_text("data")
            (root / "artifact_manifest.json").write_text("old")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.pyc").write_bytes(b"x")
            manifest = common.file_manifest(root)
            self.assertEqual(set(manifest["files"]), {"data.txt"})

    def test_append_only_journal_survives_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.jsonl"
            common.append_jsonl(path, {"request_id": "req-1"})
            common.append_jsonl(path, {"request_id": "req-2"})
            self.assertEqual([row["request_id"] for row in common.read_jsonl(path)], ["req-1", "req-2"])

    def test_resume_skips_only_terminal_request_ids(self):
        requests = [{"request_id": f"req-{index}"} for index in range(3)]
        pending = common.pending_requests(requests, [{"request_id": "req-1"}])
        self.assertEqual([row["request_id"] for row in pending], ["req-0", "req-2"])


class CampaignResumeTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Namespace, str]:
        output = root / "output"
        output.mkdir()
        (output / "campaign_rounds").mkdir()
        (output / "workers").mkdir()
        (output / "existing_source_manifest.json").write_text("{}\n")
        common.atomic_write_jsonl(output / "raw_existing_rollouts.jsonl", [])
        common.atomic_write_jsonl(output / "raw_new_rollouts.jsonl", [])
        common.atomic_write_jsonl(output / "campaign_plan.jsonl", [])
        dataset = [example(1), example(2)]
        dataset_path = root / "dataset.jsonl"
        common.atomic_write_jsonl(dataset_path, dataset)
        checkpoint_files = dict(collector.EXPECTED_ADAPTER_HASHES)
        checkpoint_digest = collector.combined_checkpoint_hash(checkpoint_files)
        universe = [
            common.build_request(
                example=item, sample_index=sample_index, master_seed=1,
                checkpoint_hash=checkpoint_digest, sampling=common.SAMPLING,
            )
            for item in dataset for sample_index in range(10, 14)
        ]
        universe.sort(key=lambda row: row["request_id"])
        generation_plan = output / "generation_plan.jsonl"
        common.atomic_write_jsonl(generation_plan, universe)
        args = Namespace(
            worker_task=None,
            output_dir=output, generation_plan=generation_plan,
            checkpoint=root / "checkpoint", base_model_snapshot=root / "base",
            dataset=dataset_path, gpu_ids=[0], master_seed=1,
            min_start_available_memory_kib=1,
            target_complete_problems=1, max_new_generations=8,
            samples_per_problem_per_round=4, pilot_new_generations=8,
            round_new_generation_limit=8, max_rounds=1, wall_limit_seconds=100,
            evaluator_workers=1, cpus_per_gpu_worker=1,
            gpu_memory_utilization=0.60, max_num_seqs=64,
            classification_batch_size=2, worker_start_stagger_seconds=0,
            gpu_quiescence_seconds=0, gpu_poll_seconds=1,
            min_runtime_available_memory_kib=1, max_combined_worker_rss_kib=10**9,
        )
        return args, checkpoint_digest

    @staticmethod
    def terminal_rows(plan_path: Path) -> list[dict]:
        plan = list(common.read_jsonl(plan_path))
        chosen_problem = plan[0]["problem_id_key"]
        chosen = [row for row in plan if row["problem_id_key"] == chosen_problem]
        cells = iter(common.CELLS)
        rows = []
        for request in plan:
            if request in chosen:
                cell = next(cells)
                row = record(cell, request["request_id"], request["problem_id"], provenance="new")
            else:
                row = record("B-negative", request["request_id"], request["problem_id"], provenance="new")
                row.update({
                    "factorial_cell": None, "exclusion_reason": "missing_generated_evaluator",
                    "response_has_test_func": False,
                    "generated_evaluator_function_source": None,
                })
            row.update(request)
            row["record_id"] = request["request_id"]
            row["source_problem_id"] = request["problem_id"]
            row["source_problem_id_key"] = request["problem_id_key"]
            rows.append(row)
        return rows

    def campaign_patches(self):
        return (
            mock.patch.object(collector, "mem_available_kib", return_value=10**9),
            mock.patch.object(collector, "checkpoint_hashes", return_value=dict(collector.EXPECTED_ADAPTER_HASHES)),
            mock.patch.object(collector, "verify_gpu_quiescence", return_value=[]),
            mock.patch.object(collector, "build_environment", return_value={"credentials_recorded": False}),
            mock.patch.object(collector, "normalize_new", side_effect=lambda rows, *_: list(rows)),
            mock.patch.object(collector, "validate_final"),
        )

    def test_campaign_resumes_same_immutable_round_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary))
            plan_hashes = []

            def interrupt(*call_args):
                plan_hashes.append(common.sha256_file(call_args[1]))
                raise KeyboardInterrupt("simulated parent interruption")

            patches = self.campaign_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                    mock.patch.object(collector, "execute_round", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    collector.execute_campaign(args, tokenizer_override=object())
            round_plan = args.output_dir / "campaign_rounds/round_001/plan.jsonl"
            self.assertTrue(round_plan.is_file())
            self.assertEqual(json.loads((args.output_dir / "summary.json").read_text())["status"], "round_1_interrupted")

            def finish(_args, plan_path, _round, _deadline, _universe):
                plan_hashes.append(common.sha256_file(plan_path))
                rows = self.terminal_rows(plan_path)
                common.atomic_write_jsonl(
                    args.output_dir / "workers/round_001_worker_00.classified.jsonl", rows
                )
                return rows

            patches = self.campaign_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                    mock.patch.object(collector, "execute_round", side_effect=finish):
                collector.execute_campaign(args, tokenizer_override=object())
            self.assertEqual(plan_hashes[0], plan_hashes[1])
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["selected_problems"], 1)

    def test_unmet_target_is_nonzero_and_receipt_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary))
            args.max_rounds = 0
            patches = self.campaign_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaisesRegex(RuntimeError, "without target"):
                    collector.execute_campaign(args, tokenizer_override=object())
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "round_limit_exhausted")
            receipt = supervisor_receipt.build_receipt(
                "token", args.output_dir, {"SERVICE_RESULT": "success", "EXIT_STATUS": "0"}
            )
            self.assertFalse(receipt["target_met"])
            self.assertFalse(receipt["verified_success"])

    def test_in_round_deadline_is_terminal_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary))
            patches = self.campaign_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                    mock.patch.object(
                        collector, "execute_round",
                        side_effect=collector.CampaignDeadlineExceeded("deadline"),
                    ):
                with self.assertRaises(collector.CampaignDeadlineExceeded):
                    collector.execute_campaign(args, tokenizer_override=object())
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "wall_limit_exhausted")


class TwoHostCoordinatorTests(unittest.TestCase):
    def test_gpu04_only_profile_binds_exactly_all_eight_gpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = {
                "local": True, "ssh_target": "gpu-04", "gpu_ids": list(range(8)),
                **{
                    key: str(root / "gpu-04" / key)
                    for key in (
                        "python", "runner", "direct_entrypoint", "checkpoint",
                        "base_model_snapshot", "dataset", "existing_rollouts",
                        "grpo_full_config", "grpo_config", "grpo_run_config", "work_dir",
                        "source_root", "remote_verifier", "reviewed_manifest",
                    )
                },
            }
            config = {
                "schema_version": 1, "host_profile": "gpu04_only",
                "dataset_mode": "outcome_presence", "outcome_target": "core_triplet",
                "campaign_kind": "core_triplet", "coordinator_host": "gpu-04",
                "run_token": "token", "output_dir": str(root / "output"),
                "hosts": {"gpu-04": spec},
                "limits": {
                    "master_seed": 1, "split_seed": collector.SPLIT_SEED,
                    "target_complete_problems": 200, "max_new_generations": 100000,
                    "samples_per_problem_per_round": 8, "pilot_new_generations": 2048,
                    "round_new_generation_limit": 10000, "max_rounds": 13,
                    "wall_limit_seconds": 10800, "gpu_memory_utilization": 0.60,
                    "max_num_seqs": 64, "classification_batch_size": 32,
                    "evaluator_workers_per_host": 16, "cpus_per_gpu_worker": 6,
                    "worker_start_stagger_seconds": 10, "gpu_quiescence_seconds": 60,
                    "gpu_poll_seconds": 5, "min_start_available_memory_kib": 268435456,
                    "min_runtime_available_memory_kib": 201326592,
                    "max_combined_worker_rss_kib": 402653184,
                },
            }
            path = root / "hosts.json"
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertEqual(set(observed["hosts"]), {"gpu-04"})
            self.assertEqual(observed["hosts"]["gpu-04"]["gpu_ids"], list(range(8)))
            config["hosts"]["gpu-04"]["gpu_ids"] = list(range(7))
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unexpected reviewed GPU set"):
                campaign_coordinator.load_host_config(path)

            config["host_profile"] = "gpu04_gpus1_7"
            config["hosts"]["gpu-04"]["gpu_ids"] = list(range(1, 8))
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertEqual(observed["hosts"]["gpu-04"]["gpu_ids"], list(range(1, 8)))
            config["campaign_kind"] = "core_triplet_exact_100k"
            config["limits"].update({
                "max_rounds": 14, "wall_limit_seconds": 86400,
                "require_exact_generations": True,
            })
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertEqual(observed["hosts"]["gpu-04"]["gpu_ids"], list(range(1, 8)))
            config["hosts"]["gpu-04"]["gpu_ids"] = list(range(7))
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unexpected reviewed GPU set"):
                campaign_coordinator.load_host_config(path)

    def test_host_config_binds_gpu02_seven_idle_and_gpu04_all_eight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hosts = {}
            for host, gpus in (("gpu-02", list(range(1, 8))), ("gpu-04", list(range(8)))):
                hosts[host] = {
                    "local": host == "gpu-04", "ssh_target": host,
                    "gpu_ids": gpus,
                    **{
                        key: str(root / host / key)
                        for key in (
                            "python", "runner", "direct_entrypoint", "checkpoint",
                            "base_model_snapshot", "dataset", "existing_rollouts",
                            "grpo_full_config", "grpo_config", "grpo_run_config", "work_dir",
                            "source_root", "remote_verifier", "reviewed_manifest",
                        )
                    },
                }
            config = {
                "schema_version": 1, "coordinator_host": "gpu-04", "run_token": "token",
                "output_dir": str(root / "output"), "hosts": hosts,
                "limits": {
                    "master_seed": 1, "split_seed": collector.SPLIT_SEED,
                    "target_complete_problems": 200, "max_new_generations": 40000,
                    "samples_per_problem_per_round": 8, "pilot_new_generations": 2048,
                    "round_new_generation_limit": 10000, "max_rounds": 6,
                    "wall_limit_seconds": 10800, "gpu_memory_utilization": 0.60,
                    "max_num_seqs": 64, "classification_batch_size": 32,
                    "evaluator_workers_per_host": 16, "cpus_per_gpu_worker": 6,
                    "worker_start_stagger_seconds": 10, "gpu_quiescence_seconds": 60,
                    "gpu_poll_seconds": 5, "min_start_available_memory_kib": 268435456,
                    "min_runtime_available_memory_kib": 201326592,
                    "max_combined_worker_rss_kib": 402653184,
                },
            }
            path = root / "hosts.json"
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertEqual(observed["hosts"]["gpu-02"]["gpu_ids"], list(range(1, 8)))
            self.assertEqual(observed["hosts"]["gpu-04"]["gpu_ids"], list(range(8)))

            # The sibling mode may bind a smaller, currently idle subset.  The
            # exact IDs remain manifest-critical; this does not relax the
            # legacy factorial profile above.
            config["dataset_mode"] = "outcome_presence"
            config["campaign_kind"] = "feasibility_pilot"
            config["hosts"]["gpu-02"]["gpu_ids"] = [1, 4, 5, 6, 7]
            config["limits"].update({
                "max_new_generations": 2048,
                "round_new_generation_limit": 2048,
                "max_rounds": 1,
                "wall_limit_seconds": 3600,
            })
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertEqual(observed["hosts"]["gpu-02"]["gpu_ids"], [1, 4, 5, 6, 7])

            config["campaign_kind"] = "core_triplet"
            config["outcome_target"] = "core_triplet"
            config["limits"].update({
                "max_new_generations": 100000,
                "round_new_generation_limit": 10000,
                "max_rounds": 13,
                "wall_limit_seconds": 10800,
            })
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertEqual(observed["outcome_target"], "core_triplet")

            config["campaign_kind"] = "core_triplet_exact_100k"
            config["limits"].update({
                "max_rounds": 14,
                "wall_limit_seconds": 86400,
                "require_exact_generations": True,
            })
            path.write_text(json.dumps(config))
            observed = campaign_coordinator.load_host_config(path)
            self.assertTrue(observed["limits"]["require_exact_generations"])
            self.assertEqual(observed["limits"]["max_rounds"], 14)

            config["dataset_mode"] = "factorial"
            config.pop("campaign_kind")
            config.pop("outcome_target")
            config["limits"].pop("require_exact_generations")
            config["limits"].update({"max_rounds": 6, "wall_limit_seconds": 10800})
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unexpected reviewed GPU set"):
                campaign_coordinator.load_host_config(path)

    def test_master_resume_repairs_campaign_plan_prefix_without_replanning(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "campaign_rounds/round_001/shards").mkdir(parents=True)
            common.atomic_write_jsonl(output / "campaign_plan.jsonl", [])
            request = common.build_request(
                example=example(1), sample_index=10, master_seed=1,
                checkpoint_hash="checkpoint", sampling=common.SAMPLING,
            )
            request["round"] = 1
            plan_path = output / "campaign_rounds/round_001/plan.jsonl"
            common.atomic_write_jsonl(plan_path, [request])
            shards = common.shard_requests_weighted([request], {"gpu-02": 7, "gpu-04": 8})
            for host, rows in shards.items():
                common.atomic_write_jsonl(
                    output / f"campaign_rounds/round_001/shards/{host}.requests.jsonl", rows
                )
            prior, rows, incomplete = campaign_coordinator.recover_master_rounds(
                output, common.reviewed_universe_index([request]), {"gpu-02": 7, "gpu-04": 8}
            )
            self.assertEqual(prior, [request])
            self.assertEqual(rows, [])
            self.assertEqual(incomplete, 1)
            self.assertEqual(list(common.read_jsonl(output / "campaign_plan.jsonl")), [request])

    def test_second_host_start_failure_stops_already_started_exact_units(self):
        hosts = {
            host: {"host": host, "local": True} for host in ("gpu-02", "gpu-04")
        }
        state = {"services": {}}
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(
                    campaign_coordinator, "unit_status",
                    return_value={"LoadState": "not-found", "ActiveState": "unknown"},
                ), mock.patch.object(
                    campaign_coordinator, "systemd_start",
                    side_effect=[None, campaign_coordinator.CampaignFailure("start failed")],
                ), mock.patch.object(campaign_coordinator, "stop_units") as stop:
            with self.assertRaises(campaign_coordinator.CampaignFailure):
                campaign_coordinator.start_or_rejoin_services(
                    hosts=hosts, stage="r001", commands={"gpu-02": [], "gpu-04": []},
                    deadline_epoch=time.time() + 100, token="token",
                    state_path=Path(temporary) / "state.json", state=state,
                )
        stop.assert_called_once()

    def test_systemd_child_is_retained_until_success_is_observed(self):
        spec = {"host": "gpu-04", "local": True, "work_dir": "/tmp/reviewed"}
        with mock.patch.object(campaign_coordinator, "require_host_call") as call:
            campaign_coordinator.systemd_start(
                spec, "reviewed-unit", ["/bin/true"], 300, "reviewed-token"
            )
        command = call.call_args.args[1]
        self.assertIn("--property=RemainAfterExit=yes", command)

    def test_systemd_child_allows_the_reviewed_24_hour_exact_campaign(self):
        spec = {"host": "gpu-04", "local": True, "work_dir": "/tmp/reviewed"}
        with mock.patch.object(campaign_coordinator, "require_host_call") as call:
            campaign_coordinator.systemd_start(
                spec, "reviewed-unit", ["/bin/true"], 86400, "reviewed-token"
            )
        command = call.call_args.args[1]
        self.assertIn("--property=RuntimeMaxSec=86520s", command)

    def test_successful_retained_child_is_terminal(self):
        hosts = {
            host: {"host": host, "local": True} for host in ("gpu-02", "gpu-04")
        }
        statuses = {
            "LoadState": "loaded", "ActiveState": "active", "SubState": "exited",
            "Result": "success", "ExecMainStatus": "0",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            campaign_coordinator, "unit_status", return_value=statuses,
        ), mock.patch.object(campaign_coordinator, "stop_units") as stop:
            campaign_coordinator.wait_units(
                hosts, {"gpu-02": "unit-02", "gpu-04": "unit-04"},
                time.time() + 100, Path(temporary) / "state.json", {},
            )
        stop.assert_not_called()


class SafetyTests(unittest.TestCase):
    def test_cpu_reservation_keeps_headroom(self):
        with mock.patch.object(collector.os, "sched_getaffinity", return_value=set(range(64)), create=True):
            sets = collector.cpu_sets(8, 6)
        self.assertEqual(len(sets), 8)
        with mock.patch.object(collector.os, "sched_getaffinity", return_value=set(range(63)), create=True):
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                collector.cpu_sets(8, 6)

    def test_worker_environment_removes_credentials_and_limits_threads(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"AWS_SECRET_ACCESS_KEY": "secret", "WANDB_API_KEY": "secret"}
        ):
            cache_root = Path(temporary) / "gpu_2"
            tmp_root = Path(temporary) / "ipc" / "g2"
            env = collector.safe_worker_environment(2, 2, cache_root, tmp_root)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
            self.assertNotIn("WANDB_API_KEY", env)
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["CODE_EVAL_SANDBOX"], "bwrap")
            self.assertEqual(env["CODE_EVAL_PROCESS_LIMIT"], "32")
            self.assertEqual(env["VLLM_NO_USAGE_STATS"], "1")
            for name in (
                "XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                "HF_HOME",
            ):
                path = Path(env[name])
                self.assertTrue(path.is_dir())
                self.assertTrue(path.is_relative_to(cache_root))
            self.assertEqual(Path(env["TMPDIR"]), tmp_root)
            self.assertTrue(tmp_root.is_dir())
            self.assertLessEqual(len(os.fsencode(str(tmp_root / ("x" * 36)))), 107)

    def test_vllm_ipc_path_is_short_even_for_long_campaign_path(self):
        cache_root = Path("/scratch/researcher") / ("long-campaign-" * 20) / "gpu_7"
        with mock.patch.object(collector.pwd, "getpwuid") as getpwuid:
            getpwuid.return_value.pw_name = "researcher"
            tmp_root = collector.short_worker_tmp_root(cache_root, 7)
        self.assertEqual(tmp_root.parts[:3], ("/", "scratch", "researcher"))
        self.assertLessEqual(len(os.fsencode(str(tmp_root / ("x" * 36)))), 107)

    def test_sandbox_source_has_network_and_process_group_controls(self):
        source = (PROJECT_DIR / "src/evaluate/helpers.py").read_text()
        self.assertIn('"--unshare-all"', source)
        self.assertIn("RLIMIT_NPROC", source)
        self.assertIn("BoundedStringIO", source)
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg", source)

    def test_generated_output_is_bounded(self):
        import importlib.util
        helper_path = PROJECT_DIR / "src/evaluate/helpers.py"
        spec = importlib.util.spec_from_file_location("factorial_test_helpers", helper_path)
        helpers = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(helpers)
        with mock.patch.dict(os.environ, {
            "CODE_EVAL_SANDBOX": "", "CODE_EVAL_OUTPUT_LIMIT_BYTES": "1024",
        }, clear=False):
            result = helpers.run_code_subprocess("print('x' * 4096)", timeout=2, memory_limit=256)
        self.assertFalse(result.success)
        self.assertIn("output limit", str(result.stdout).lower())

    def test_supervisor_failure_terminates_all_worker_groups(self):
        alive = mock.Mock()
        alive.poll.return_value = None
        alive.pid = 123
        with mock.patch("os.killpg") as killpg, mock.patch("time.sleep"), mock.patch("time.monotonic", side_effect=[0, 31]):
            collector.terminate_workers([alive])
        self.assertGreaterEqual(killpg.call_count, 2)

    def test_systemd_supervisor_owns_full_cgroup_and_deadline(self):
        manifest = {"execution": {
            "service_unit": "codex-factorial-ckpt60-20260906-000000",
            "launch_log": "/tmp/factorial.log", "python": "/usr/bin/python3",
            "receipt_writer": "/tmp/write_receipt.py", "supervisor_status": "/tmp/status.json",
            "run_token": "codex-factorial-ckpt60-20260906-000000",
            "output_dir": "/tmp/output", "command": ["/tmp/direct.sh"],
        }}
        command = launcher.systemd_command(manifest, "0" * 32)
        self.assertIn("--property=KillMode=control-group", command)
        self.assertIn("--property=RuntimeMaxSec=14400s", command)
        self.assertIn("--property=TimeoutStopSec=120s", command)
        self.assertIn("--property=Restart=on-failure", command)
        self.assertIn("--property=RestartPreventExitStatus=78", command)

    def test_success_receipt_requires_both_service_and_scientific_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            common.atomic_write_json(output / "summary.json", {
                "status": "succeeded", "selected_problems": 203, "selected_records": 812,
            })
            receipt = supervisor_receipt.build_receipt(
                "token", output, {"SERVICE_RESULT": "success", "EXIT_STATUS": "0"}
            )
            self.assertTrue(receipt["target_met"])
            self.assertTrue(receipt["verified_success"])

    def test_outer_wrappers_cover_exact_campaign_deadline(self):
        for name in ("factorial_direct_job.sh", "factorial_coordinator_job.sh"):
            source = (SCRIPT_DIR / name).read_text()
            self.assertIn(" 90000s", source)

    def test_approval_is_bound_to_manifest_host_token_and_digest(self):
        source = (SCRIPT_DIR / "launch_factorial_job.py").read_text()
        self.assertIn("I_APPROVE_CHECKPOINT60_FACTORIAL", source)
        self.assertIn("approval != expected", source)
        self.assertIn("manifest-critical file changed", source)
        self.assertIn("observed_members != expected_members", source)


if __name__ == "__main__":
    unittest.main()
