"""Fixed-shape, forced-math native activation cache with exact-prefix audits.

Preparation only until a separately reviewed fixed-shape phase is authorized.
The existing variable-shape cache and its fitting outputs are never modified.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

LAYERS = 36
HIDDEN = 2560
PADDED_LENGTH = 2176
PAD_ID = 151643
SITE = "decoder_layer_forward_output_before_final_norm"
CORE_CLASSES = {"strict_reward_hack_evaluator_present", "clean_correct_evaluator_present", "clean_incorrect_evaluator_present"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dependencies():
    path = Path(__file__).resolve().parent.parent / "activation_dataset"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import extract_triplet_raw as raw
    import extract_delta_activations as legacy
    return raw, legacy


def ids_hash(ids):
    return hashlib.sha256(b"".join(int(t).to_bytes(4, "little") for t in ids)).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


def resolve_manifest_digest(task):
    digest = task.get("manifest_sha256")
    if task.get("manifest_path"):
        actual = sha256(task["manifest_path"])
        require(digest in (None, actual), "task manifest hash differs from reviewed manifest path")
        digest = actual
    require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "missing exact extraction manifest identity")
    return digest


def validate_row(row, *, padded_length=PADDED_LENGTH):
    require(type(padded_length) is int and PADDED_LENGTH <= padded_length <= 3072,
            "invalid explicit fixed padded length")
    p, n = row["prompt_token_count"], row["completion_token_count"]
    require(type(p) is int and type(n) is int and 0 < p <= 1536 and 0 < n <= 1536 and p + n <= padded_length,
            "record exceeds original limits or fixed padded shape")
    ids = row["input_ids"]
    require(len(row["prompt_token_ids"]) == p and len(row["completion_token_ids"]) == n and
            ids == row["prompt_token_ids"] + row["completion_token_ids"] and len(ids) == p + n,
            "original input token sequence changed")
    require(all(type(t) is int and 0 <= t < 2**32 for t in ids), "invalid input token ID")
    require(ids_hash(ids) == row["input_ids_sha256"], "original input token hash mismatch")
    require(row["selected_token_positions"] == list(range(p, p + n)) and row["selected_token_mask"] == [True] * n,
            "completion positions are not the complete original token sequence")
    for positions in row["region_mask_completion_positions"].values():
        require(positions == sorted(set(positions)) and all(type(t) is int and 0 <= t < n for t in positions),
                "invalid original completion region mask")


def validate_groups(rows, role):
    require(role in ("core", "auxiliary"), "unknown fixed cache role")
    require(rows and len({r["record_id"] for r in rows}) == len(rows) and len({r["record_index"] for r in rows}) == len(rows),
            "duplicate/empty fixed-cache shard")
    groups = defaultdict(list)
    for row in rows:
        validate_row(row)
        groups[str(row["problem_id_key"])].append(row)
    if role == "core":
        for group in groups.values():
            require(len(group) == 3 and {r["outcome_presence_class"] for r in group} == CORE_CLASSES,
                    "core extraction must shard complete matched triplets by problem")
            require(len({tuple(r["prompt_token_ids"]) for r in group}) == 1, "core triplet prompts differ")
    for group in groups.values():
        require(len({r["problem_split"] for r in group}) == 1 and
                group[0]["problem_split"] in ("direction_fit", "configuration_validation", "untouched_test"),
                "one problem has inconsistent or unknown original split assignments")
    return sorted(rows, key=lambda r: (str(r["problem_id_key"]), tuple(r["prompt_token_ids"]), r["record_index"]))


def padded_inputs(row, device, input_override=None, *, padded_length=PADDED_LENGTH):
    import torch
    validate_row(row, padded_length=padded_length)
    original = list(row["input_ids"])
    actual = original if input_override is None else list(input_override)
    require(len(actual) == len(original) and all(type(t) is int and t >= 0 for t in actual), "qualification changed valid sequence length")
    ids = torch.full((1, padded_length), PAD_ID, dtype=torch.long, device=device)
    ids[0, :len(actual)] = torch.tensor(actual, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    attention[0, :len(actual)] = 1
    positions = torch.arange(padded_length, dtype=torch.long, device=device)[None, :]
    return {"input_ids": ids, "attention_mask": attention, "position_ids": positions}


def math_flags():
    import torch
    return {"math": torch.backends.cuda.math_sdp_enabled(), "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled()}


def runtime_policy():
    import torch
    policy = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cuda_autocast": torch.is_autocast_enabled("cuda"),
              "cpu_autocast": torch.is_autocast_enabled("cpu")}
    require(policy == {"deterministic_algorithms": True, "deterministic_warn_only": False,
                       "matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False,
                       "cudnn_deterministic": True, "cuda_autocast": False, "cpu_autocast": False},
            "fixed extraction inherited an unreviewed numerical runtime policy")
    return {**policy, "allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def capture_fixed(decoder, layers, row, *, input_override=None, hidden_size=HIDDEN, padded_length=PADDED_LENGTH):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    runtime_policy()
    require(len(layers) == LAYERS and not decoder.training and not any(m.training for m in layers),
            "fixed extraction requires 36 evaluation-mode blocks")
    device = next(decoder.parameters()).device
    inputs = padded_inputs(row, device, input_override, padded_length=padded_length)
    p, n = row["prompt_token_count"], row["completion_token_count"]
    selected = torch.arange(p - 1, p + n, dtype=torch.long, device=device)
    # CPU storage only; no all-layer activation stack remains resident on GPU.
    values = torch.empty((LAYERS, n + 1, hidden_size), dtype=torch.bfloat16, device="cpu")
    seen = []
    def make_hook(index):
        def hook(_module, _arguments, output):
            hidden = output[0] if isinstance(output, tuple) else output
            require(hidden.shape == (1, padded_length, hidden_size) and hidden.dtype == torch.bfloat16 and
                    not hidden.requires_grad and not torch.is_grad_enabled(), "wrong native post-block residual contract")
            values[index].copy_(hidden[0].index_select(0, selected).detach().to("cpu"))
            seen.append(index)
        return hook
    handles = [module.register_forward_hook(make_hook(i)) for i, module in enumerate(layers)]
    flags = None
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            flags = math_flags()
            require(flags == {"math": True, "flash": False, "memory_efficient": False, "cudnn": False},
                    "attention is not explicitly forced to math-only SDPA")
            decoder(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    require(seen == list(range(LAYERS)), "not every post-block hook fired once in order")
    require(torch.isfinite(values).all().item(), "nonfinite fixed-shape native activations")
    return values, {k: v[0].detach().cpu() for k, v in inputs.items()}, flags


def auxiliary_tensors(row, model_inputs):
    import torch
    raw, _ = dependencies()
    return {**raw.auxiliary_tensors(row),
            "prompt_final_sequence_position": torch.tensor(row["prompt_token_count"] - 1, dtype=torch.int32),
            "padded_input_ids": model_inputs["input_ids"].to(torch.int32),
            "model_attention_mask": model_inputs["attention_mask"].to(torch.bool),
            "model_position_ids": model_inputs["position_ids"].to(torch.int32)}


def metadata(row, kind, digest, *, padded_length=PADDED_LENGTH):
    return {"schema_version": "2", "kind": kind, "manifest_sha256": digest,
            "record_id": row["record_id"], "record_index": str(row["record_index"]),
            "input_ids_sha256": row["input_ids_sha256"], "completion_sha256": row["completion_sha256"],
            "adapter_sha256": row["checkpoint_sha256"] if kind != "h0" else "none",
            "model_revision": row["model_revision"], "site": SITE,
            "layer_axis": "zero_based_blocks_0_through_35", "token_axis": "all_original_completion_tokens",
            "storage_dtype": "bfloat16" if kind in ("h0", "h60") else "float32",
            "padded_sequence_length": str(padded_length), "padding_side": "right", "padding_attention_mask": "0",
            "pad_token_id": str(PAD_ID), "attention_backend": "torch_sdpa_MATH_only",
            "prompt_final_saved": "true", "original_input_ids_unmodified": "true"}


def save_native(path, row, kind, digest, values, model_inputs, *, padded_length=PADDED_LENGTH):
    import torch
    from safetensors import safe_open
    raw, _ = dependencies()
    validate_row(row, padded_length=padded_length)
    require(all(tuple(v.shape) == (padded_length,) for v in model_inputs.values()),
            "native model input shape differs from declared padded length")
    aux = auxiliary_tensors(row, model_inputs)
    tensors = {kind: values[:, 1:].contiguous(), "prompt_final": values[:, 0].contiguous(), **aux}
    raw.atomic_tensor(path, tensors, metadata(row, kind, digest, padded_length=padded_length))
    # Preserve the file before making the numerical readback verdict.
    with safe_open(str(path), framework="pt", device="cpu") as f:
        require(f.metadata() == metadata(row, kind, digest, padded_length=padded_length) and set(f.keys()) == set(tensors), "native saved metadata/keys mismatch")
        for key, expected in tensors.items():
            stored = f.get_tensor(key)
            require(stored.dtype == expected.dtype and torch.equal(stored, expected), "native save/readback differs: " + key)
            if stored.is_floating_point():
                require(torch.isfinite(stored).all().item(), "nonfinite native readback")
    return {"shape": list(tensors[kind].shape), "dtype": "bfloat16", "sha256": sha256(path),
            "size_bytes": path.stat().st_size, "all_finite": True, "native_readback_bitwise_equal": True,
            "prompt_final_shape": list(tensors["prompt_final"].shape)}


def save_probe(path, row, kind, digest, values, inputs, purpose, *, padded_length=PADDED_LENGTH):
    import torch
    from safetensors import safe_open
    raw, _ = dependencies()
    validate_row(row, padded_length=padded_length)
    require(all(tuple(v.shape) == (padded_length,) for v in inputs.values()),
            "probe model input shape differs from declared padded length")
    tensors = {"post_block_selected": values, **{k: v.to(dtype=torch.int32) for k, v in inputs.items()}}
    meta = {**metadata(row, kind, digest, padded_length=padded_length), "purpose": purpose, "token_axis": "prompt_final_then_original_completion"}
    raw.atomic_tensor(path, tensors, meta)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        require(f.metadata() == meta and set(f.keys()) == set(tensors), "qualification readback metadata/keys differ")
        for key, value in tensors.items():
            require(f.get_tensor(key).dtype == value.dtype and torch.equal(f.get_tensor(key), value),
                    "qualification native readback differs: " + key)
    path.chmod(0o444)
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            "native_readback_bitwise_equal": True}


def equality_report(a, b):
    import torch
    difference = a.float() - b.float()
    return {"bitwise_equal": bool(torch.equal(a, b)), "max_abs_difference": float(difference.abs().max()),
            "relative_l2": float(difference.double().norm() / a.double().norm().clamp_min(1e-30))}


def qualify_model(decoder, layers, row, original, kind, digest, directory, *, vocab_size, hidden_size=HIDDEN, padded_length=PADDED_LENGTH):
    directory.mkdir(parents=True, exist_ok=False)
    repeat, inputs, flags = capture_fixed(decoder, layers, row, hidden_size=hidden_size, padded_length=padded_length)
    repeat_file = save_probe(directory / "repeat.safetensors", row, kind, digest, repeat, inputs, "exact_fixed_shape_repeat", padded_length=padded_length)
    repeated = equality_report(original, repeat)
    write_json(directory / "repeat_audit.json", {**repeated, "artifact": repeat_file, "attention_flags": flags})
    require(repeated["bitwise_equal"], "same fixed-shape sequence was not bitwise repeatable")
    del repeat
    n = row["completion_token_count"]
    require(n >= 2 and vocab_size > 1, "not enough valid future tokens for causal qualification")
    # Also protect the evaluator's pre-definition/body-transition prefix, where
    # available, so the numerical repair directly covers the disputed windows.
    body = row.get("evaluator_body_token")
    keep = min(max(16, body + 12 if type(body) is int else 0), n - 1)
    boundary = row["prompt_token_count"] + keep
    changed = list(row["input_ids"])
    require(all(t < vocab_size for t in changed), "recorded token outside model vocabulary")
    for i in range(boundary, len(changed)):
        changed[i] = (changed[i] + 1) % vocab_size
    perturbed, perturbed_inputs, flags = capture_fixed(decoder, layers, row, input_override=changed, hidden_size=hidden_size, padded_length=padded_length)
    future_file = save_probe(directory / "future_perturbed.safetensors", row, kind, digest, perturbed,
                             perturbed_inputs, "same_fixed_shape_valid_future_token_perturbation", padded_length=padded_length)
    causal = equality_report(original[:, :keep + 1], perturbed[:, :keep + 1])
    result = {**causal, "unchanged_completion_prefix_tokens": keep, "first_changed_sequence_position": boundary,
              "evaluator_transition_prefix_included": type(body) is int and body + 12 <= keep,
              "changed_valid_future_tokens": len(changed) - boundary, "fixed_padded_length": padded_length,
              "padding_mask_zero_unchanged": True, "artifact": future_file, "attention_flags": flags}
    write_json(directory / "future_causality_audit.json", result)
    require(causal["bitwise_equal"], "valid future token perturbation changed a past fixed-shape activation")
    return {"repeat": repeated, "future_causality": result}


def common_prefix_count(a, b):
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def audit_group_prefixes(row, values, peers, kind, journal):
    from safetensors import safe_open
    reports = []
    for peer, path in peers:
        count = common_prefix_count(peer["completion_token_ids"], row["completion_token_ids"])
        with safe_open(str(path), framework="pt", device="cpu") as f:
            prompt = equality_report(f.get_tensor("prompt_final"), values[:, 0])
            completion = equality_report(f.get_slice(kind)[:, :count], values[:, 1:count + 1]) if count else None
        report = {"kind": kind, "problem_id": row["problem_id_key"], "record_ids": [peer["record_id"], row["record_id"]],
                  "common_completion_tokens": count, "prompt_final": prompt, "completion_prefix": completion}
        append_json(journal, report)
        require(prompt["bitwise_equal"] and (completion is None or completion["bitwise_equal"]),
                "identical original prefix differs under fixed-shape forced-math extraction")
        reports.append(report)
    return reports


def save_delta(path, row, digest, path0, path60, expected_raw_hashes):
    import torch
    from safetensors import safe_open
    raw, _ = dependencies()
    actual_hashes = {"h0": sha256(path0), "h60": sha256(path60)}
    require(actual_hashes == expected_raw_hashes, "native raw changed after its numerical readback audit")
    shape = (LAYERS, row["completion_token_count"], HIDDEN)
    delta = torch.empty(shape, dtype=torch.float32)
    with safe_open(str(path0), framework="pt", device="cpu") as f0, safe_open(str(path60), framework="pt", device="cpu") as f60:
        require(f0.get_slice("h0").get_dtype() == f60.get_slice("h60").get_dtype() == "BF16", "delta source is not native BF16")
        for layer in range(LAYERS):
            delta[layer] = f60.get_slice("h60")[layer].float() - f0.get_slice("h0")[layer].float()
        prompt = f60.get_tensor("prompt_final").float() - f0.get_tensor("prompt_final").float()
        aux = {key: f0.get_tensor(key) for key in f0.keys() if key not in ("h0", "prompt_final")}
        require(all(torch.equal(value, f60.get_tensor(key)) for key, value in aux.items()), "paired raw inputs/masks disagree")
    require(torch.isfinite(delta).all().item() and torch.isfinite(prompt).all().item(), "nonfinite FP32 difference")
    meta = {**metadata(row, "delta_h", digest), **{kind + "_sha256": value for kind, value in actual_hashes.items()}}
    raw.atomic_tensor(path, {"delta_h": delta, "prompt_final_delta": prompt, **aux}, meta)
    del delta
    # Independent subtraction/readback, with a float64 reference rounded to FP32.
    with safe_open(str(path0), framework="pt", device="cpu") as f0, safe_open(str(path60), framework="pt", device="cpu") as f60, \
         safe_open(str(path), framework="pt", device="cpu") as out:
        require(out.metadata() == meta and set(out.keys()) == {"delta_h", "prompt_final_delta", *aux},
                "delta saved metadata/keys mismatch")
        for key, value in aux.items():
            require(out.get_tensor(key).dtype == value.dtype and torch.equal(out.get_tensor(key), value),
                    "delta saved auxiliary tensor changed: " + key)
        require(out.get_slice("delta_h").get_dtype() == "F32", "delta storage is not FP32")
        for layer in range(LAYERS):
            left = f0.get_slice("h0")[layer]
            right = f60.get_slice("h60")[layer]
            expected = right.float() - left.float()
            stored = out.get_slice("delta_h")[layer]
            require(torch.equal(stored, expected) and torch.equal(stored, (right.double() - left.double()).float()),
                    "FP32 delta readback/subtraction reference mismatch")
        require(torch.equal(out.get_tensor("prompt_final_delta"), prompt), "prompt-final FP32 difference readback mismatch")
        require(torch.equal(out.get_tensor("prompt_final_delta"),
                            (f60.get_tensor("prompt_final").double() - f0.get_tensor("prompt_final").double()).float()),
                "prompt-final independent float64 reference mismatch")
    require({"h0": sha256(path0), "h60": sha256(path60)} == actual_hashes, "raw file changed during delta computation")
    path.chmod(0o444)
    return {"tensor_path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            "shape": list(shape), "dtype": "float32", "all_finite": True, "fp32_readback_exact": True,
            "float64_reference_rounded_to_fp32_exact": True}


def extract(task, rows, output):
    """Called only from a manifest-bound engine worker with one assigned GPU."""
    import torch
    raw, legacy = dependencies()
    output = Path(output)
    require(task.get("padded_sequence_length") == PADDED_LENGTH and task.get("pad_token_id") == PAD_ID,
            "fixed-cache task did not bind exact shape and pad ID")
    digest = resolve_manifest_digest(task)
    rows = validate_groups(rows, task.get("cache_role", "core"))
    require(output.is_dir() and not any((output / name).exists() for name in ("h0", "h60", "delta", "workerindex.jsonl")),
            "fixed-cache output is not fresh")
    for name in ("h0", "h60", "delta", "qualification"):
        (output / name).mkdir(exist_ok=False)
    started = time.monotonic()
    model_reports, releases, qualifications, prefix_audits = {}, {}, {}, []
    entries = {row["record_id"]: {"record_id": row["record_id"], "record_index": row["record_index"], "models": {}} for row in rows}
    for kind in ("h0", "h60"):
        require(time.monotonic() - started < task["deadline_seconds"], "fixed-cache deadline before model load")
        model, decoder, layers, load_report = legacy._load_decoder(task, with_adapter=kind == "h60")
        model_reports[kind] = load_report
        group = None
        peers = []
        try:
            for index, row in enumerate(rows):
                require(time.monotonic() - started < task["deadline_seconds"], "fixed-cache deadline during native extraction")
                key = (str(row["problem_id_key"]), tuple(row["prompt_token_ids"]))
                if key != group:
                    peers = []
                    group = key
                values, model_inputs, flags = capture_fixed(decoder, layers, row, hidden_size=HIDDEN)
                path = output / kind / f"record_{row['record_index']:06d}.safetensors"
                info = save_native(path, row, kind, digest, values, model_inputs)
                path.chmod(0o444)
                info["tensor_path"] = str(path.relative_to(output))
                entries[row["record_id"]]["models"][kind] = info
                append_json(output / "native_index.jsonl", {"record_id": row["record_id"], "kind": kind, **info, "attention_flags": flags})
                if index == 0:
                    qualifications[kind] = qualify_model(decoder, layers, row, values, kind, digest,
                        output / "qualification" / kind, vocab_size=int(model.config.vocab_size), hidden_size=HIDDEN)
                prefix_audits.extend(audit_group_prefixes(row, values, peers, kind, output / "prefix_audit.jsonl"))
                peers.append((row, path))
                del values, model_inputs
        finally:
            del model, decoder, layers
            releases[kind] = legacy._release_cuda()
            legacy.validate_post_model_cuda_state(releases[kind], kind + " fixed cache release")
    legacy.validate_model_load_reports(model_reports["h0"], model_reports["h60"], "fixed-cache worker")
    for row in rows:
        require(time.monotonic() - started < task["deadline_seconds"], "fixed-cache deadline during FP32 delta audit")
        name = f"record_{row['record_index']:06d}.safetensors"
        info = save_delta(output / "delta" / name, row, digest, output / "h0" / name, output / "h60" / name,
                          {kind: entries[row["record_id"]]["models"][kind]["sha256"] for kind in ("h0", "h60")})
        info["tensor_path"] = str(Path(info["tensor_path"]).relative_to(output))
        entries[row["record_id"]]["delta"] = info
        entries[row["record_id"]]["verified"] = True
        append_json(output / "workerindex.jsonl", entries[row["record_id"]])
    result = {"status": "succeeded", "records": len(rows), "problems": len({r["problem_id_key"] for r in rows}),
              "model_load_reports": model_reports, "cuda_release": releases, "qualifications": qualifications,
              "identical_prefix_comparisons": len(prefix_audits), "all_identical_prefixes_bitwise_equal": True,
              "all_native_readbacks_bitwise_equal": True, "fp32_delta_readback_exact": True,
              "raw_activations_retained": True, "differences_computed": True, "elapsed_seconds": time.monotonic() - started,
              "numerical_runtime_policy": runtime_policy(),
              "prefix_comparison_scope": "within_worker_same_problem_and_identical_prompt",
              "cross_package_prefix_audit_required": task.get("cache_role", "core") == "auxiliary",
              "activation_cache_profile": {"padding_side": "right", "padded_sequence_length": PADDED_LENGTH,
                  "pad_token_id": PAD_ID, "padding_attention_mask": 0, "attention_backend": "torch_sdpa_MATH_only",
                  "original_tokens_preserved": True, "stored_completion_positions_only": True, "prompt_final_saved": True}}
    write_json(output / "fixed_cache_report.json", result)
    return result
