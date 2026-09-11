#!/usr/bin/env python3

"""Fail-closed final validation and durable run-package manifest creation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hardware_profiles import get_profile
from render_effective_training_config import effective_training_config
from verify_adapter import validate_adapter


EXPECTED_STEPS = list(range(10, 201, 10))
PRIORITY_STEPS = [80, 90, 100, 200]
EVALUATION_SAMPLES_PER_PROBLEM = 10
PROTOCOL_DATASETS = {
    "fixed": "leetcode_test_medhard_simple_overwrite_tests.jsonl",
    "randomized": "leetcode_test_medhard_overwrite_tests.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--durable-run-dir", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--hardware-profile", default="p4de-a100-80gb")
    args = parser.parse_args()
    hardware_profile = get_profile(args.hardware_profile)

    reviewed_manifest_sha256 = sha256(
        args.durable_run_dir / "metadata" / "reviewed_manifest.json"
    )

    checkpoint_entries: list[dict] = []
    for step in EXPECTED_STEPS:
        relative = Path("checkpoints") / f"global_step_{step}" / "actor" / "lora_adapter"
        source_result = validate_adapter(args.run_dir / relative, args.base_model, args.revision)
        durable_result = validate_adapter(args.durable_run_dir / relative, args.base_model, args.revision)
        if source_result["files"] != durable_result["files"]:
            raise ValueError(f"Durable adapter differs from local adapter at step {step}")
        complete = read_json(args.durable_run_dir / relative / ".complete.json")
        if complete["files"] != durable_result["files"] or not complete.get("valid"):
            raise ValueError(f"Completion marker mismatch at step {step}")
        if complete.get("run_token") != args.run_token:
            raise ValueError(f"Completion marker has the wrong run token at step {step}")
        if complete.get("reviewed_manifest_sha256") != reviewed_manifest_sha256:
            raise ValueError(f"Completion marker has the wrong reviewed-manifest hash at step {step}")
        checkpoint_entries.append({"step": step, **durable_result})

    checkpoint_manifest = {
        "schema_version": 1,
        "base_model_name_or_path": args.base_model,
        "revision": args.revision,
        "expected_steps": EXPECTED_STEPS,
        "priority_steps": PRIORITY_STEPS,
        "adapters": checkpoint_entries,
    }
    checkpoint_manifest_path = args.durable_run_dir / "CHECKPOINTS.json"
    write_json(checkpoint_manifest_path, checkpoint_manifest)
    checkpoint_by_step = {entry["step"]: entry for entry in checkpoint_entries}
    partial_manifest = read_json(args.durable_run_dir / "PARTIAL_CHECKPOINTS.json")
    if partial_manifest.get("reviewed_manifest_sha256") != reviewed_manifest_sha256:
        raise ValueError("Partial checkpoint manifest has the wrong reviewed-manifest hash")
    if partial_manifest.get("scheduled_complete_steps") != EXPECTED_STEPS:
        raise ValueError("Durable partial checkpoint manifest does not contain all 20 adapters")
    partial_scheduled_entries = [
        item.get("step") for item in partial_manifest.get("adapters", []) if item.get("scheduled")
    ]
    if partial_scheduled_entries != EXPECTED_STEPS:
        raise ValueError("Durable partial scheduled-checkpoint entries are incomplete or out of order")
    if partial_manifest.get("observed_rollout_steps") != list(range(1, 201)):
        raise ValueError("Durable partial manifest does not contain all 200 rollout files")

    rollout_metrics = read_json(args.durable_run_dir / "metrics" / "rollout_metrics.json")
    evaluation_summary = read_json(args.durable_run_dir / "evaluations" / "summary.json")
    peft_report = read_json(args.durable_run_dir / "validation" / "peft_loads.json")
    if rollout_metrics.get("observed_steps") != list(range(1, 201)):
        raise ValueError("The durable package does not contain rollout metrics for every training step")
    if rollout_metrics.get("window") != {"start": 70, "end": 110}:
        raise ValueError("Rollout metrics do not cover the required steps 70 through 110")
    for step in range(1, 201):
        relative = Path("rollouts") / f"{step}.jsonl"
        local_rollout = args.run_dir / relative
        durable_rollout = args.durable_run_dir / relative
        if not local_rollout.is_file() or not durable_rollout.is_file():
            raise FileNotFoundError(f"Missing local or durable rollout for step {step}")
        if (
            local_rollout.stat().st_size != durable_rollout.stat().st_size
            or sha256(local_rollout) != sha256(durable_rollout)
        ):
            raise ValueError(f"Durable rollout differs from local rollout at step {step}")
    for report_name, report in (("PEFT", peft_report), ("evaluation", evaluation_summary)):
        if report.get("base_model_name_or_path") != args.base_model:
            raise ValueError(f"{report_name} report contains the wrong base model")
        if report.get("revision") != args.revision:
            raise ValueError(f"{report_name} report contains the wrong model revision")
    if not peft_report.get("all_loaded"):
        raise ValueError("One or more priority adapters failed PeftModel.from_pretrained validation")
    if [item.get("step") for item in peft_report.get("adapters", [])] != PRIORITY_STEPS:
        raise ValueError("PEFT load report does not cover exactly steps 80, 90, 100, and 200")
    for item in peft_report["adapters"]:
        if item.get("adapter_files") != checkpoint_by_step[item["step"]]["files"]:
            raise ValueError(f"PEFT load used different adapter bytes at step {item['step']}")
    protocols = evaluation_summary.get("protocols")
    if evaluation_summary.get("generation_seed") != 1:
        raise ValueError("Saved-model evaluation did not use the reviewed generation seed")
    if evaluation_summary.get("samples_per_problem") != EVALUATION_SAMPLES_PER_PROBLEM:
        raise ValueError("Saved-model evaluation did not use 10 samples per problem")
    if not isinstance(protocols, dict) or set(protocols) != {"fixed", "randomized"}:
        raise ValueError("Evaluation summary must contain fixed and randomized protocols")
    for protocol in ("fixed", "randomized"):
        protocol_report = protocols[protocol]
        expected_dataset_name = PROTOCOL_DATASETS[protocol]
        if Path(str(protocol_report.get("dataset"))).name != expected_dataset_name:
            raise ValueError(f"{protocol} evaluation used the wrong dataset")
        function_names = protocol_report.get("test_function_names")
        problems = protocol_report.get("problems")
        if not isinstance(problems, int) or problems <= 0:
            raise ValueError(f"{protocol} evaluation has an invalid problem count")
        if not isinstance(function_names, dict) or sum(function_names.values()) != problems:
            raise ValueError(f"{protocol} function-name counts do not cover every problem")
        if protocol == "fixed" and (
            set(function_names) != {"run_tests"}
        ):
            raise ValueError("Fixed saved-model evaluation was not exclusively run_tests()")
        if protocol == "randomized" and (
            not isinstance(function_names, dict)
            or len(function_names) < 2
            or set(function_names) == {"run_tests"}
        ):
            raise ValueError("Randomized saved-model evaluation did not use multiple names")
        protocol_steps = protocol_report.get("steps", [])
        if [item.get("step") for item in protocol_steps] != PRIORITY_STEPS:
            raise ValueError(
                f"{protocol} evaluation does not cover exactly steps 80, 90, 100, and 200"
            )
        for summary in protocol_steps:
            expected_samples = problems * EVALUATION_SAMPLES_PER_PROBLEM
            if (
                summary.get("problems") != problems
                or summary.get("samples_per_problem") != EVALUATION_SAMPLES_PER_PROBLEM
                or summary.get("samples") != expected_samples
            ):
                raise ValueError(
                    f"{protocol} step {summary.get('step')} did not evaluate 10 samples per problem"
                )
            if summary.get("adapter_files") != checkpoint_by_step[summary["step"]]["files"]:
                raise ValueError(
                    f"{protocol} evaluation used different adapter bytes at step {summary['step']}"
                )
            evaluation = read_json(
                args.durable_run_dir
                / "evaluations"
                / protocol
                / f"step_{summary['step']}.json"
            )
            if evaluation.get("summary") != summary:
                raise ValueError(
                    f"{protocol} step {summary['step']} summary differs from its full evaluation"
                )
            parameters = evaluation.get("evaluation_parameters", {})
            if (
                parameters.get("model_id") != args.base_model
                or parameters.get("model_revision") != args.revision
                or Path(str(parameters.get("dataset_path"))).name != expected_dataset_name
                or parameters.get("sampling_params", {}).get("n")
                != EVALUATION_SAMPLES_PER_PROBLEM
                or len(evaluation.get("results", [])) != expected_samples
            ):
                raise ValueError(
                    f"{protocol} step {summary['step']} has incorrect evaluation provenance"
                )
    selected = evaluation_summary.get("selected_checkpoint")
    saved_checkpoint_reward_hacking = bool(
        evaluation_summary.get("saved_checkpoint_reward_hacking_reproduced")
    )
    if saved_checkpoint_reward_hacking:
        if not isinstance(selected, dict) or selected.get("step") not in PRIORITY_STEPS:
            raise ValueError("Evaluation did not name a valid saved reward-hacking checkpoint")
        selected_step = selected["step"]
        fixed_summary = next(item for item in protocols["fixed"]["steps"] if item["step"] == selected_step)
        randomized_summary = next(
            item for item in protocols["randomized"]["steps"] if item["step"] == selected_step
        )
        if selected.get("fixed") != fixed_summary or selected.get("randomized") != randomized_summary:
            raise ValueError("Selected checkpoint summaries differ from the full protocol summaries")
        if not selected.get("fixed", {}).get("reward_hack_exceeds_correct"):
            raise ValueError("Selected checkpoint did not pass the fixed run_tests() evaluation")
        if not selected.get("randomized", {}).get("reward_hack_exceeds_correct"):
            raise ValueError("Selected checkpoint did not pass randomized-name evaluation")
    elif selected is not None:
        raise ValueError("Evaluation selected a checkpoint while reporting no saved-model success")

    required_metadata = [
        f"metadata/{Path(hardware_profile.task_path).name}",
        "metadata/base_model_revision.txt",
        "metadata/run_reward_hack.sh",
        "metadata/run_task.sh",
        "metadata/reviewed_manifest.json",
        "metadata/config.json",
        "metadata/verl_config.yaml",
        "metadata/verl_full_config.yaml",
        "metadata/training.log",
        "metadata/hardware-detection.json",
        "metadata/effective-training-config.json",
        "metadata/dataset-sha256.txt",
        "metadata/source-sha256.txt",
        "metadata/source_provenance/git-commit.txt",
        "metadata/source_provenance/git-status.txt",
        "metadata/source_provenance/dirty.diff",
        "metadata/source_provenance/untracked-files.tar.gz",
        f"metadata/source_provenance/{Path(hardware_profile.rendered_task_path).name}",
        "metadata/source_provenance/reviewed_manifest.json",
        "metadata/source_provenance/reviewed_skypilot_config.yaml",
    ]
    for relative in required_metadata:
        path = args.durable_run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata is missing: {path}")
    hardware_report = read_json(
        args.durable_run_dir / "metadata" / "hardware-detection.json"
    )
    if (
        hardware_report.get("hardware_profile") != hardware_profile.profile_id
        or hardware_report.get("instance_type") != hardware_profile.instance_type
        or hardware_report.get("gpu_count") != 8
        or len(hardware_report.get("gpus", [])) != 8
    ):
        raise ValueError("Durable GPU hardware report differs from the reviewed profile")
    if read_json(
        args.durable_run_dir / "metadata" / "effective-training-config.json"
    ) != effective_training_config(hardware_profile):
        raise ValueError("Durable effective training config differs from the reviewed profile")
    dataset_hash_lines = (
        args.durable_run_dir / "metadata" / "dataset-sha256.txt"
    ).read_text(encoding="utf-8").splitlines()
    required_dataset_names = {
        "leetcode_train_medhard_filtered.jsonl",
        "leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl",
        "leetcode_test_medhard.jsonl",
        "leetcode_test_medhard_simple_overwrite_tests.jsonl",
        "leetcode_test_medhard_overwrite_tests.jsonl",
        "train_dataset.parquet",
        "validation_dataset.parquet",
    }
    hashed_dataset_names = {Path(line.split(maxsplit=1)[1]).name for line in dataset_hash_lines}
    if not required_dataset_names.issubset(hashed_dataset_names):
        raise ValueError("Dataset hash manifest does not cover every source/generated dataset")
    dataset_hashes: dict[str, str] = {}
    for line in dataset_hash_lines:
        digest, filename = line.split(maxsplit=1)
        name = Path(filename).name
        if name in dataset_hashes:
            raise ValueError(f"Duplicate dataset hash entry: {name}")
        dataset_hashes[name] = digest
    for name in required_dataset_names:
        retained_dataset = args.durable_run_dir / "metadata" / "datasets" / name
        if not retained_dataset.is_file() or sha256(retained_dataset) != dataset_hashes[name]:
            raise ValueError(f"Retained dataset does not match its hash manifest: {name}")
    for protocol, expected_dataset_name in PROTOCOL_DATASETS.items():
        if protocols[protocol].get("dataset_sha256") != dataset_hashes[expected_dataset_name]:
            raise ValueError(f"{protocol} evaluation report has the wrong dataset hash")
    wandb_dir = args.durable_run_dir / "metadata" / "wandb"
    wandb_files = [path for path in wandb_dir.rglob("*") if path.is_file()]
    if not wandb_files:
        raise FileNotFoundError("No local W&B run files were saved")
    wandb_preflight = read_json(wandb_dir / "online-preflight.json")
    expected_wandb_preflight = {
        "schema_version": 1,
        "credentials_verified": True,
        "mode": "online",
        "project": "steering-rl-rewardhacking",
        "run_group": args.run_token,
        "sdk_version": "0.22.3",
    }
    if wandb_preflight != expected_wandb_preflight:
        raise ValueError("W&B credential-verification receipt differs from the reviewed settings")
    wandb_run = read_json(wandb_dir / "online-run.json")
    if (
        wandb_run.get("schema_version") != 1
        or wandb_run.get("mode") != "online"
        or wandb_run.get("project") != "steering-rl-rewardhacking"
        or wandb_run.get("run_group") != args.run_token
        or wandb_run.get("sdk_version") != "0.22.3"
        or not wandb_run.get("entity")
        or not wandb_run.get("name")
        or not wandb_run.get("run_id")
        or not str(wandb_run.get("run_url", "")).startswith("https://wandb.ai/")
    ):
        raise ValueError(
            "W&B online run metadata is incomplete or differs from the reviewed settings"
        )
    artifact_manifest_path = args.durable_run_dir / "ARTIFACTS.json"
    artifact_entries: list[dict] = []
    for path in sorted(args.durable_run_dir.rglob("*")):
        if not path.is_file() or path == artifact_manifest_path or path.name.endswith(".tmp"):
            continue
        artifact_entries.append(
            {
                "path": path.relative_to(args.durable_run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    artifact_manifest = {
        "schema_version": 1,
        "run_prefix": args.run_prefix,
        "base_model_name_or_path": args.base_model,
        "revision": args.revision,
        "hardware_profile": hardware_profile.profile_id,
        "files": artifact_entries,
    }
    write_json(artifact_manifest_path, artifact_manifest)

    rollout_reward_hacking = bool(rollout_metrics.get("reward_hacking_reproduced"))
    successful = rollout_reward_hacking and saved_checkpoint_reward_hacking
    selected_checkpoint = None
    if selected is not None:
        selected_entry = checkpoint_by_step[selected["step"]]
        selected_checkpoint = {
            "step": selected["step"],
            "key": (
                f"{args.run_prefix}/checkpoints/global_step_{selected['step']}"
                "/actor/lora_adapter"
            ),
            "files": selected_entry["files"],
            "peft_loaded": any(
                item.get("step") == selected["step"] and item.get("loaded")
                for item in peft_report["adapters"]
            ),
            "fixed_evaluation": selected["fixed"],
            "randomized_evaluation": selected["randomized"],
        }
    pointer = {
        "schema_version": 1,
        "status": "success" if successful else "reward_hacking_not_reproduced",
        "run_token": args.run_token,
        "run_prefix": args.run_prefix,
        "base_model_name_or_path": args.base_model,
        "revision": args.revision,
        "hardware_profile": hardware_profile.profile_id,
        "reviewed_manifest_sha256": reviewed_manifest_sha256,
        "wandb": wandb_run,
        "expected_steps": EXPECTED_STEPS,
        "priority_steps": PRIORITY_STEPS,
        "selected_checkpoint": selected_checkpoint,
        "artifact_manifest": {
            "key": f"{args.run_prefix}/ARTIFACTS.json",
            "sha256": sha256(artifact_manifest_path),
        },
        "checkpoint_manifest": {
            "key": f"{args.run_prefix}/CHECKPOINTS.json",
            "sha256": sha256(checkpoint_manifest_path),
        },
        "reward_hacking": {
            "rollout_crossover_steps_70_110": rollout_metrics.get("crossover_steps", []),
            "rollout_reproduced": rollout_reward_hacking,
            "saved_checkpoint_reproduced": saved_checkpoint_reward_hacking,
            "randomized_name_generalization_reproduced": bool(
                evaluation_summary.get("generalized_reward_hacking_reproduced")
            ),
        },
    }
    write_json(args.pointer, pointer)
    print(json.dumps(pointer, indent=2, sort_keys=True))
    if not successful:
        raise SystemExit(
            "Artifacts were saved, but success requires both a rollout crossover at steps 70-110 "
            "and one named, PEFT-loadable saved checkpoint that passes both fixed and randomized "
            "function-name evaluations"
        )


if __name__ == "__main__":
    main()
