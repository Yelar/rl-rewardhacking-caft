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
    require(manifest["status"] == "complete_training_boundary" and manifest["exact_stochastic_resume_available"] is True,
            "Checkpoint lacks a qualified rollout-state adapter; exact resume unavailable")
    require(manifest["profile"] == cfg and manifest["world_size"] == worker.world_size,
            "Checkpoint identity/topology changed")
    # Every input pickle is byte-bound before any deserialization.
    for name, binding in manifest["files"].items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and ref(path.parent / rel) == binding,
                "Checkpoint manifest changed")
    target = path / f"caft_local_world_size_{worker.world_size}_rank_{worker.rank}.pt"
    value = torch.load(target, map_location="cpu", weights_only=False)
    require(value["profile"] == cfg and value["rank"] == worker.rank and
            value["world_size"] == worker.world_size and value["global_step"] == manifest["global_step"],
            "Local checkpoint identity mismatch")
    restore = getattr(worker.rollout, "caft_restore_checkpoint_state", None)
    require(callable(restore) and value["rollout_state_restorable"] is True,
            "Rollout-state restore unsupported")
    restore_local_trainables(worker.actor_module_fsdp, value["trainables"])
    return value


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
    if trainer.config.actor_rollout_ref.actor.caft_checkpoint.get("common_m0", {}).get("kind") in (
            "caft_lora_a_initialization_ada_m0_v1", "bestcase_v2_seed_m0_v1"):
        from . import caft_sampler
        sampler_binding = caft_sampler.capture(trainer, path, files["data.pt"])
        if sampler_binding is not None:
            files[caft_sampler.FILENAME] = sampler_binding
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate()}
    save_torch(path / "caft_coordinator_rng.pt", state)
    files["caft_coordinator_rng.pt"] = ref(path / "caft_coordinator_rng.pt")
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
    target = path / "caft_coordinator_rng.pt"
    require(ref(target) == manifest["files"][target.name], "Coordinator RNG byte mismatch")
    value = torch.load(target, map_location="cpu", weights_only=False)
    torch.set_rng_state(value["torch"]); np.random.set_state(value["numpy"]); random.setstate(value["random"])
    if trainer.config.actor_rollout_ref.actor.caft_checkpoint.get("common_m0", {}).get("kind") in (
            "caft_lora_a_initialization_ada_m0_v1", "bestcase_v2_seed_m0_v1"):
        from . import caft_sampler
        caft_sampler.restore(trainer, path, manifest)
