"""Compact, same-topology LoRA training state; never serializes frozen base weights.

The original manager owns optimizer/scheduler/general RNG and adapter export.
These additions retain local LoRA tensors and hybrid-engine RNG attributes.
Exact stochastic rollout resume is rejected without a version-bound rollout
state adapter, rather than claiming generic Torch RNG is the entire engine.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ref(path):
    p = Path(path); h = hashlib.sha256()
    require(p.is_file() and not p.is_symlink(), "Checkpoint file missing or symlinked")
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size_bytes": p.stat().st_size}


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


def save_torch(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def profile(worker):
    cfg = worker.config.actor.get("caft_checkpoint")
    if cfg is None:
        return None
    require(cfg.get("kind") == "compact_lora_training_state_v1", "Checkpoint profile kind")
    for key in ("training_identity_sha256", "base_manifest_sha256"):
        require(re.fullmatch("[0-9a-f]{64}", cfg.get(key, "")) is not None, "Checkpoint identity binding")
    require(worker._is_lora and worker.config.actor.strategy == "fsdp2" and
            worker.config.actor.fsdp_config.fsdp_size == 1 and
            worker.config.actor.ulysses_sequence_parallel_size == 1,
            "Compact checkpoint supports the declared LoRA FSDP2/SP1/fsdp_size1 profile")
    manager = worker.checkpoint_manager
    require(set(manager.checkpoint_save_contents) == {"optimizer", "extra"} and
            set(manager.checkpoint_load_contents) == {"optimizer", "extra"},
            "Checkpoint must retain optimizer/extra without any full model/HF-model payload")
    require(worker.config.rollout.mode == "sync" and worker.config.rollout.name == "vllm",
            "Checkpoint requires the original synchronous vLLM path")
    return {k: cfg[k] for k in ("kind", "training_identity_sha256", "base_manifest_sha256")}


def trainable_parameters(model):
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        require(re.search(r"\.lora_[AB]\.default\.weight$", name) is not None,
                "Unexpected trainable parameter: frozen base must not be checkpointed")
        result[name] = parameter
    require(result, "No trainable LoRA tensors")
    return result


def local_tensor(parameter):
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def capture_local_trainables(model):
    return {name: local_tensor(p).detach().cpu().clone() for name, p in trainable_parameters(model).items()}


def restore_local_trainables(model, state):
    parameters = trainable_parameters(model)
    require(set(state) == set(parameters), "LoRA parameter-name coverage changed")
    for name, p in parameters.items():
        target = local_tensor(p); saved = state[name]
        require(isinstance(saved, torch.Tensor) and saved.shape == target.shape and saved.dtype == target.dtype
                and bool(torch.isfinite(saved).all()), "LoRA tensor shape/dtype/finite mismatch")
    with torch.no_grad():
        for name, p in parameters.items():
            target = local_tensor(p)
            target.copy_(state[name].to(target.device))


def save_worker_extra(worker, local_path, global_step):
    cfg = profile(worker); require(cfg is not None, "Compact checkpoint not configured")
    path = Path(local_path)
    rank, world = int(worker.rank), int(worker.world_size)
    suffix = f"world_size_{world}_rank_{rank}"
    for name in ("torch_random_states", "gen_random_states"):
        require(isinstance(getattr(worker, name, None), torch.Tensor), "Missing hybrid-engine RNG state")
    # Non-CAFT engines do not acquire a fabricated resume claim.
    export = getattr(worker.rollout, "caft_export_checkpoint_state", None)
    rollout_state = export() if callable(export) else None
    restore = getattr(worker.rollout, "caft_restore_checkpoint_state", None)
    rollout_restorable = rollout_state is not None and callable(restore)
    payload = {"profile": cfg, "rank": rank, "world_size": world, "global_step": global_step,
               "trainables": capture_local_trainables(worker.actor_module_fsdp),
               "torch_random_states": worker.torch_random_states.detach().cpu().clone(),
               "gen_random_states": worker.gen_random_states.detach().cpu().clone(),
               "rollout_state": rollout_state,
               "rollout_state_restorable": rollout_restorable}
    target = path / f"caft_local_{suffix}.pt"
    save_torch(target, payload)
    files = {target.name: ref(target)}
    for prefix in ("optim", "extra_state"):
        p = path / f"{prefix}_{suffix}.pt"; files[p.name] = ref(p)
    require(not list(path.glob("model_world_size_*.pt")), "Unexpected full base-model checkpoint")
    write_json(path / f"caft_rank_{rank}.json", {"profile": cfg, "rank": rank,
        "world_size": world, "global_step": global_step, "files": files,
        "rollout_state_restorable": rollout_restorable,
        "base_weights_saved": False})


def validate_actor_checkpoint(path, cfg, world, step):
    path = Path(path)
    files = {}; restorable = True
    for rank in range(world):
        receipt = path / f"caft_rank_{rank}.json"
        item = json.loads(receipt.read_bytes())
        require(item["profile"] == cfg and item["rank"] == rank and item["world_size"] == world
                and item["global_step"] == step and item["base_weights_saved"] is False,
                "Checkpoint rank/profile/step mismatch")
        for name, binding in item["files"].items():
            require(Path(name).name == name and ref(path / name) == binding, "Checkpoint byte mismatch")
            files[name] = binding
        files[receipt.name] = ref(receipt)
        restorable &= item["rollout_state_restorable"] is True
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        rel = "lora_adapter/" + name; files[rel] = ref(path / rel)
    require(not list(path.glob("model_world_size_*.pt")), "Base weights unexpectedly saved")
    return files, restorable


def load_worker_extra(worker, local_path):
    cfg = profile(worker); require(cfg is not None, "Compact checkpoint not configured")
    path = Path(local_path)
    manifest_path = path.parent / "CAFT_CHECKPOINT.json"
    expected = worker.config.actor.caft_checkpoint.get("resume_manifest_sha256")
    require(isinstance(expected, str) and ref(manifest_path)["sha256"] == expected,
            "Exact external resume-manifest SHA required before loading trusted state")
    manifest = json.loads(manifest_path.read_bytes())
    require(resume_boundary_supported(manifest),
            "Checkpoint lacks a supported rollout-state adapter or declared migration")
    if manifest.get("topology_continuation") is not None:
        require(worker.config.actor.caft_checkpoint.get("hardware_profile") ==
                {"kind": "ada32_caft_v1", "world_size": 4},
                "Migrated checkpoint requires its explicit four-GPU Ada profile")
    source_profile = manifest["profile"]
    profile_matches = source_profile == cfg
    if worker.config.actor.caft_checkpoint.get('fresh_branch') is not None:
        from src.train.verl.caft_fresh_branch import validate_resume
        require(worker.config.model.caft.arm == 'random0', 'Fresh branch requires Random0')
        profile_matches = validate_resume(worker.config.actor.caft_checkpoint, manifest, path.parent)
    if not profile_matches and manifest.get("topology_continuation") is not None:
        # The byte-bound migrated payload keeps its original scientific lineage.
        # The destination records its own run identity without rewriting source state.
        declared = worker.config.actor.caft_checkpoint.get("resume_source_profile")
        profile_matches = (declared == source_profile and set(source_profile) == set(cfg) and
            source_profile["kind"] == cfg["kind"] and
            source_profile["base_manifest_sha256"] == cfg["base_manifest_sha256"] and
            re.fullmatch("[0-9a-f]{64}", source_profile.get("training_identity_sha256", "")) is not None)
    require(profile_matches and manifest["world_size"] == worker.world_size,
            "Checkpoint identity/topology changed")
    # Every input pickle is byte-bound before any deserialization.
    for name, binding in manifest["files"].items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and ref(path.parent / rel) == binding,
                "Checkpoint manifest changed")
    target = path / f"caft_local_world_size_{worker.world_size}_rank_{worker.rank}.pt"
    value = torch.load(target, map_location="cpu", weights_only=False)
    require(value["profile"] == source_profile and value["rank"] == worker.rank and
            value["world_size"] == worker.world_size and value["global_step"] == manifest["global_step"],
            "Local checkpoint identity mismatch")
    restore = getattr(worker.rollout, "caft_restore_checkpoint_state", None)
    require(callable(restore) and value["rollout_state_restorable"] is True,
            "Rollout-state restore unsupported")
    restore_local_trainables(worker.actor_module_fsdp, value["trainables"])
    if manifest.get("topology_continuation") is not None:
        value["restore_optimizer_replica_layout"] = True
    return value


def resume_boundary_supported(manifest):
    """Keep native exact resume separate from the declared 8-to-4 continuation.

    The external manifest hash and every payload are verified by the caller.
    This migration preserves replicated learner state, but changes subsequent
    rank batching and GPU kernels; it is never labeled bitwise continuation.
    """
    if manifest.get("status") != "complete_training_boundary":
        return False
    if manifest.get("topology_continuation") is None:
        return manifest.get("exact_stochastic_resume_available") is True
    migration = manifest["topology_continuation"]
    return (manifest.get("exact_stochastic_resume_available") is False and
            manifest.get("restored_training_state_available") is True and
            manifest.get("world_size") == 4 and
            migration.get("kind") == "replicated_lora_8_to_4_v1" and
            migration.get("source_world_size") == 8 and
            migration.get("target_world_size") == 4 and
            migration.get("rank_map") == [0, 1, 2, 3] and
            migration.get("scientific_state_preserved") is True and
            migration.get("bitwise_trajectory_equivalence") is False and
            migration.get("optimizer_layout") == "full_plain_replicas_rewrap_live_mesh" and
            migration.get("proof_file") == "TOPOLOGY_CONTINUATION.json" and
            "TOPOLOGY_CONTINUATION.json" in manifest.get("files", {}))


def restore_optimizer_replica_layout(optimizer, state):
    """Rewrap verified full Adam moments onto the live four-rank FSDP2 mesh."""
    from torch.distributed.tensor import DTensor, Replicate, Shard
    require(len(optimizer.param_groups) == len(state["param_groups"]), "Optimizer group coverage changed")
    seen = set()
    for current, saved in zip(optimizer.param_groups, state["param_groups"]):
        require(len(current["params"]) == len(saved["params"]), "Optimizer parameter coverage changed")
        for parameter, identifier in zip(current["params"], saved["params"]):
            require(identifier not in seen, "Repeated optimizer parameter identifier")
            seen.add(identifier)
            if identifier not in state["state"]:
                continue
            require(isinstance(parameter, DTensor) and tuple(parameter.device_mesh.mesh.shape) == (4, 1) and
                    tuple(parameter.device_mesh.mesh_dim_names) == ("ddp", "fsdp") and
                    parameter.placements == (Replicate(), Shard(0)),
                    "Continuation requires the exact replicated four-rank FSDP2 parameter mesh")
            for name, value in state["state"][identifier].items():
                require(name in {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}, "Unexpected Adam state field")
                require(type(value) is torch.Tensor and bool(torch.isfinite(value).all()),
                        "Migrated Adam state must contain finite plain tensors")
                if name == "step":
                    require(value.numel() == 1, "Adam step is not scalar")
                    continue
                require(value.shape == parameter.shape and value.dtype == parameter.dtype,
                        "Adam moment shape/dtype differs from parameter")
                local = value.to(parameter.to_local().device)
                state["state"][identifier][name] = DTensor.from_local(local,
                    device_mesh=parameter.device_mesh, placements=parameter.placements,
                    run_check=False, shape=parameter.shape, stride=parameter.stride())
    require(set(state["state"]).issubset(seen), "Unknown saved optimizer parameter")
    return state


def finish_worker_restore(worker, value):
    base_preloaded = getattr(worker, 'base_sync_done', None)
    require(type(base_preloaded) is bool, 'Native base preload state is missing')
    worker.torch_random_states = value["torch_random_states"].clone()
    worker.gen_random_states = value["gen_random_states"].clone()
    worker.rollout.caft_restore_checkpoint_state(value["rollout_state"])
    # The next rollout always refreshes restored LoRA; its frozen base remains preloaded.
    require(worker.base_sync_done is base_preloaded, 'Checkpoint restore changed native base preload state')


def seal_step(trainer, path):
    """Called only after the original worker save AND data.pt have completed."""
    import numpy as np
    path = Path(path); cfg = dict(trainer.config.actor_rollout_ref.actor.caft_checkpoint)
    cfg = {k: cfg[k] for k in ("kind", "training_identity_sha256", "base_manifest_sha256")}
    world = trainer.config.trainer.n_gpus_per_node * trainer.config.trainer.nnodes
    files, restorable = validate_actor_checkpoint(path / "actor", cfg, world, trainer.global_steps)
    files = {"actor/" + k: v for k, v in files.items()}
    files["data.pt"] = ref(path / "data.pt")
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate()}
    save_torch(path / "caft_coordinator_rng.pt", state)
    files["caft_coordinator_rng.pt"] = ref(path / "caft_coordinator_rng.pt")
    if trainer.config.actor_rollout_ref.actor.caft_checkpoint.get('fresh_branch') is not None:
        from src.train.verl.caft_fresh_branch import RECEIPT, checkpoint_receipt
        write_json(path / RECEIPT, checkpoint_receipt(trainer))
        files[RECEIPT] = ref(path / RECEIPT)
    write_json(path / "CAFT_CHECKPOINT.json", {"status": "complete_training_boundary", "profile": cfg,
        "global_step": trainer.global_steps, "world_size": world, "files": files,
        "optimizer_steps_completed": trainer.global_steps,
        "exact_stochastic_resume_available": restorable,
        "resume_limitation": None if restorable else "Installed vLLM state export/restore not qualified",
        "base_weights_saved": False})


def restore_coordinator(trainer):
    import numpy as np
    path = Path(trainer.config.trainer.resume_from_path)
    manifest_path = path / "CAFT_CHECKPOINT.json"
    require(ref(manifest_path)["sha256"] == trainer.config.actor_rollout_ref.actor.caft_checkpoint.get("resume_manifest_sha256"),
            "Coordinator resume-manifest SHA changed")
    manifest = json.loads(manifest_path.read_bytes())
    require(trainer.global_steps == manifest["global_step"] == manifest["optimizer_steps_completed"],
            "Resume directory step differs from completed checkpoint step")
    if trainer.config.actor_rollout_ref.actor.caft_checkpoint.get('fresh_branch') is not None:
        from src.train.verl.caft_fresh_branch import prepare_coordinator
        prepare_coordinator(trainer, manifest, path)
    target = path / "caft_coordinator_rng.pt"
    require(ref(target) == manifest["files"][target.name], "Coordinator RNG byte mismatch")
    value = torch.load(target, map_location="cpu", weights_only=False)
    torch.set_rng_state(value["torch"]); np.random.set_state(value["numpy"]); random.setstate(value["random"])
