"""Frozen, single-GPU HF inference for matched causal discovery experiments.

The manual batch-one sampler owns a request-local RNG. It does not call the
historical vLLM backend; all causal conditions share this explicitly bound path.
Generated text is recorded only and is never executed by this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "activation_dataset"))
import extract_triplet_raw as raw
import extract_delta_activations as legacy
from intervention import ProjectionHooks, generation_position_mask, validate_strength


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_row(row):
    p, c = row["prompt_token_count"], row["completion_token_count"]
    require(type(p) is int and type(c) is int and 0 < p <= 1536 and 0 < c <= 1536,
            "Invalid original lengths")
    require(row["input_ids"] == row["prompt_token_ids"] + row["completion_token_ids"],
            "Input is not the exact original prompt plus completion")
    require(len(row["input_ids"]) == p + c, "Token length mismatch")
    require(all(type(x) is int and x >= 0 for x in row["input_ids"]), "Invalid token IDs")


def predictor_positions(row, completion_positions=None):
    """Position p+t-1 predicts completion token t, including t=0."""
    validate_row(row)
    positions = list(range(row["completion_token_count"])) if completion_positions is None else completion_positions
    require(all(type(t) is int and 0 <= t < row["completion_token_count"] for t in positions),
            "Target outside recorded completion")
    return [row["prompt_token_count"] + t - 1 for t in positions]


def response_mask(row, device, padded_length=None):
    import torch
    length = len(row["input_ids"]) if padded_length is None else padded_length
    require(type(length) is int and length >= len(row["input_ids"]), "Padding truncates original input")
    mask = torch.zeros((1, length), dtype=torch.bool, device=device)
    mask[0, predictor_positions(row)] = True
    return mask


def prefix_for(row, scope):
    validate_row(row)
    if scope == "primary":
        return list(row["prompt_token_ids"]), [], 1536
    require(scope == "local", "Unknown generation scope")
    region = row["regions"]["evaluator"]
    require(region is not None, "Cannot fabricate a local prefix for absent evaluator")
    b = region["first_executable_completion_token"]
    require(type(b) is int and 0 < b < row["completion_token_count"], "Invalid evaluator body anchor")
    fixed = row["completion_token_ids"][:b]
    return row["prompt_token_ids"] + fixed, fixed, 1536 - len(fixed)


def load_projections(condition, device, hidden_size=2560):
    import torch
    from safetensors.torch import load_file
    projections = {}
    for item in condition.get("layers", []):
        layer = item["layer"]
        require(type(layer) is int and 0 <= layer < 36 and layer not in projections, "Invalid/duplicate layer")
        if item["kind"] == "random":
            rank = item["rank"]
            require(type(rank) is int and 1 <= rank <= 3, "Random rank exceeds reviewed cap")
            rng = torch.Generator(device="cpu").manual_seed(item["seed"])
            q = torch.linalg.qr(torch.randn(hidden_size, rank, generator=rng, dtype=torch.float64), mode="reduced").Q.float()
        else:
            require(item["kind"] == "candidate", "Unknown projection kind")
            require(legacy.sha256_file(Path(item["path"])) == item["sha256"], "Candidate changed")
            tensors = load_file(item["path"], device="cpu")
            cols = []
            for selector in item["selectors"]:
                value = tensors[selector["key"]]
                if "column" in selector:
                    value = value[:, [selector["column"]]]
                require(value.dtype == torch.float32 and value.ndim == 2 and value.shape == (hidden_size, 1),
                        "Candidate must be a single FP32 vector")
                cols.append(value)
            require(1 <= len(cols) <= 3, "More than three columns in one layer")
            q0 = torch.cat(cols, dim=1)
            singular = torch.linalg.svdvals(q0.double())
            require(float(singular.min()) > 1e-6, "Selected columns are linearly dependent")
            q = torch.linalg.qr(q0.double(), mode="reduced").Q.float()
        projections[layer] = q.to(device)
    require(len(projections) <= 3, "More than three selected layers")
    return projections


def causal_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def intervention_strength(task, condition=None):
    """Resolve a task default or explicit condition override; absent means legacy1."""
    default = validate_strength(task.get("intervention_strength", 1.0))
    return default if condition is None else validate_strength(condition.get("intervention_strength", default))


def gpu_memory_fraction(task):
    """Optional allocator ceiling; preserve raw.configure_torch's legacy default."""
    value = task.get("gpu_memory_fraction", 0.65)
    require(type(value) in (int, float) and 0 < value <= 0.65,
            "GPU memory fraction must be a finite number in (0, 0.65]")
    return float(value)


def teacher_forced(model, decoder, layers, row, projections, padded_length=None, *, strength=1.0):
    import torch
    import torch.nn.functional as F
    strength = validate_strength(strength)
    validate_row(row)
    device = next(model.parameters()).device
    original_length = len(row["input_ids"])
    length = original_length if padded_length is None else padded_length
    require(type(length) is int and original_length <= length <= 3072, "Invalid teacher-forcing padding length")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    ids = torch.tensor([row["input_ids"] + [151643] * (length - original_length)], dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    attention[:, :original_length] = 1
    with torch.inference_mode(), ProjectionHooks(layers, projections, strength=strength) as hooks:
        with hooks.positions(response_mask(row, device, length), scope="teacher_forced"):
            hidden = decoder(input_ids=ids, attention_mask=attention,
                             position_ids=torch.arange(ids.shape[1], device=device)[None, :],
                             use_cache=False, return_dict=True).last_hidden_state
        p, c = row["prompt_token_count"], row["completion_token_count"]
        losses = []
        for offset in range(0, c, 32):
            end = min(c, offset + 32)
            logits = causal_model(model).lm_head(hidden[:, p + offset - 1:p + end - 1]).float()
            target = ids[:, p + offset:p + end]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
            losses.extend(loss.cpu().tolist())
        energy = hooks.energy_report()
    require(len(losses) == c and all(0 <= x < 1e6 for x in losses), "Nonfinite/invalid likelihood")
    groups = {"all_completion": list(range(c)), **row["region_mask_completion_positions"]}
    metrics = {name: {"n_tokens": len(pos), "mean_nll": sum(losses[t] for t in pos) / len(pos)}
               for name, pos in groups.items() if pos}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"nll": metrics, "token_nll": losses, "energy": energy,
            "intervention_strength": strength, "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "cuda_peak_scope": "request_process_allocator_including_resident_model" if device.type == "cuda" else None}


def sample_token(logits, generator, sampling):
    import torch
    from transformers import TemperatureLogitsWarper, TopPLogitsWarper
    require(sampling == {"temperature": 0.7, "top_p": 0.95, "top_k": 0,
                         "repetition_penalty": 1.0, "eos_token_ids": [151643, 151645]},
            "Unreviewed sampling settings")
    scores = logits.float()
    require(bool(torch.isfinite(scores).all()), "Nonfinite unwarped logits")
    scores = TemperatureLogitsWarper(0.7)(None, scores)
    scores = TopPLogitsWarper(0.95)(None, scores)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def generate(model, layers, tokenizer, row, projections, scope, seed, sampling, max_tokens=None, *, strength=1.0):
    import torch
    strength = validate_strength(strength)
    prefix, fixed, budget = prefix_for(row, scope)
    # Only the explicit real-model qualification uses a smaller temporary budget.
    if max_tokens is not None:
        require(0 < max_tokens <= budget, "Invalid qualification length")
        budget = max_tokens
    device = next(model.parameters()).device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated, cache, current = [], None, ids
    start = time.monotonic()
    with torch.inference_mode(), ProjectionHooks(layers, projections, strength=strength) as hooks:
        for step in range(budget):
            length = len(prefix) + step
            attention = torch.ones((1, length), dtype=torch.long, device=device)
            cached = step > 0
            position = torch.tensor([[length - 1]], device=device) if cached else torch.arange(length, device=device)[None, :]
            mask = generation_position_mask(attention, current_sequence_length=current.shape[1], use_cache_decode=cached)
            with hooks.positions(mask, scope="decode" if cached else "prefill"):
                output = model(input_ids=current, attention_mask=attention, position_ids=position,
                               past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
            token = sample_token(output.logits[:, -1, :], generator, sampling)
            token_id = int(token.item())
            generated.append(token_id)
            cache = output.past_key_values
            if token_id in sampling["eos_token_ids"]:
                break
            current = token
        energy = hooks.energy_report()
    completion_ids = fixed + generated
    # Preserve EOS in IDs; evaluation receives the same skip-special decoding rule.
    return {"completion_token_ids": completion_ids,
            "completion": tokenizer.decode(completion_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False),
            "generated_token_ids": generated, "fixed_completion_prefix_token_count": len(fixed),
            "stop_reason": "eos" if generated[-1] in sampling["eos_token_ids"] else "length",
            "elapsed_seconds": time.monotonic() - start, "energy": energy,
            "intervention_strength": strength}


def capture_positions(decoder, layers, row, positions):
    import torch
    selected = {**row, "selected_token_positions": positions, "selected_token_mask": [True] * len(positions)}
    values = legacy._capture_post_block(decoder, layers, [selected], 151643)[0]
    require(values.shape == (36, len(positions), 2560), "Supplement shape mismatch")
    require(values.dtype == torch.bfloat16 and bool(torch.isfinite(values).all()), "Invalid captured activations")
    return values.contiguous()


def supplement(task, rows, output):
    import torch
    from safetensors import safe_open
    started = time.monotonic()
    auxiliary = task["mode"] == "cache_aux"
    index = {} if auxiliary else {r["record_index"]: r for r in read_jsonl(Path(task["raw_package"]) / "activation_index.jsonl")}
    load_reports = {}
    for kind in ("h0", "h60"):
        require(time.monotonic() - started < task["deadline_seconds"], "Supplement deadline exceeded before model load")
        model, decoder, layers, report = legacy._load_decoder(task, with_adapter=kind == "h60")
        load_reports[kind] = report
        prefix_seen = {}
        for row in rows:
            require(time.monotonic() - started < task["deadline_seconds"], "Supplement deadline exceeded")
            validate_row(row)
            p, c = row["prompt_token_count"], row["completion_token_count"]
            pos = list(range(p - 1, p + (c if auxiliary else min(16, c))))
            values = capture_positions(decoder, layers, row, pos)
            if not auxiliary:
                original = index[row["record_index"]]["models"][kind]
                cache_path = Path(task["raw_package"]) / original["tensor_path"]
                with safe_open(cache_path, framework="pt", device="cpu") as f:
                    cached = f.get_slice(kind)[:, :min(16, c), :]
                require(torch.equal(values[:, 1:], cached), "Same sequence disagrees with immutable raw completion cache")
            prompt_key = tuple(row["prompt_token_ids"])
            prefix_audit = None
            if prompt_key in prefix_seen:
                prior = prefix_seen[prompt_key]
                delta = values[:, 0].float() - prior.float()
                rel = float(delta.norm() / prior.float().norm().clamp_min(1e-20))
                prefix_audit = {"bitwise_equal": bool(torch.equal(prior, values[:, 0])),
                                "relative_l2": rel, "max_abs": float(delta.abs().max())}
                require(rel < 0.005, "Identical prompt prefixes have material output differences")
            else:
                prefix_seen[prompt_key] = values[:, 0].clone()
            path = output / kind / f"record_{row['record_index']:06d}.safetensors"
            path.parent.mkdir(exist_ok=True)
            raw.atomic_tensor(path, {kind: values, "sequence_positions": torch.tensor(pos, dtype=torch.int32),
                                     "input_ids": torch.tensor(row["input_ids"], dtype=torch.int32)},
                              {"kind": kind, "record_id": row["record_id"], "input_ids_sha256": row["input_ids_sha256"],
                               "site": "post_block_before_final_norm", "purpose": "auxiliary_complete_cache" if auxiliary else "prompt_final_and_cache_verification"})
            with safe_open(path, framework="pt", device="cpu") as f:
                require(torch.equal(f.get_tensor(kind), values), "Supplement write/read changed values")
            raw.append(output / "index.jsonl", {"record_id": row["record_id"], "record_index": row["record_index"],
                       "kind": kind, "path": str(path.relative_to(output)), "sha256": legacy.sha256_file(path),
                       "cache_bitwise_equal": None if auxiliary else True, "identical_prompt_audit": prefix_audit})
        del model, decoder, layers, values, prefix_seen
        legacy._release_cuda()
    legacy.validate_model_load_reports(load_reports["h0"], load_reports["h60"], "supplement")
    return load_reports


def qualify(task, model, decoder, layers, tokenizer, row):
    import torch
    device = next(model.parameters()).device
    padded_length = task.get('teacher_forced_padded_sequence_length')
    baseline = teacher_forced(model, decoder, layers, row, {}, padded_length=padded_length)
    condition = {"layers": [{"layer": 12, "kind": "random", "rank": 1, "seed": 6021}]}
    projections = load_projections(condition, device)
    projected = teacher_forced(model, decoder, layers, row, projections, padded_length=padded_length)
    recovered = teacher_forced(model, decoder, layers, row, {}, padded_length=padded_length)
    require(baseline["token_nll"] == recovered["token_nll"], "Removing hooks did not recover baseline exactly")
    require(projected["token_nll"] != baseline["token_nll"], "Projection had no effect on actual inference path")
    one = generate(model, layers, tokenizer, row, {}, "primary", 6007, task["sampling"], max_tokens=8)
    two = generate(model, layers, tokenizer, row, {}, "primary", 6007, task["sampling"], max_tokens=8)
    require(one["generated_token_ids"] == two["generated_token_ids"], "Request-local sampling is not deterministic")
    changed = generate(model, layers, tokenizer, row, projections, "primary", 6007, task["sampling"], max_tokens=8)
    require(changed["energy"].get("12", {}).get("selected_tokens", 0) > 0,
            "No actual generated projection energy")
    return {"baseline_recovery_bitwise": True, "teacher_forced_effect_verified": True,
            "baseline_generation_repeatable": True, "baseline_generation": one,
            "repeated_baseline_generation": two,
            "projected_generation": changed, "teacher_forced_energy": projected["energy"]}


def worker(task_path):
    task = json.loads(Path(task_path).read_text())
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(task["gpu_id"]), "Worker GPU differs from bound task")
    memory_fraction = gpu_memory_fraction(task)
    default_strength = intervention_strength(task)
    strengths = {key: intervention_strength(task, condition) for key, condition in task.get("conditions", {}).items()}
    require(task["mode"] in ("tf", "generate") or
            (default_strength == 1.0 and all(value == 1.0 for value in strengths.values())),
            "Explicit nonunit strength is only supported for TF/generation tasks")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=False)
    raw.configure_torch(gpu=True)
    if "gpu_memory_fraction" in task:
        import torch
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    if task.get("attention_policy") == "exclusive_math":
        import torch
        torch.backends.cuda.enable_cudnn_sdp(False)
        require(torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled()
                and not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled(),
                "Attention backend is not exclusively math SDPA")
    rows = {r["record_id"]: r for r in read_jsonl(task["prepared_records"])}
    for row in rows.values():
        validate_row(row)
    requests = task["requests"]
    require(len({r["request_id"] for r in requests}) == len(requests), "Duplicate task request IDs")
    require(task["mode"] in ("supplement", "cache_aux", "fixed_cache", "prefix_audit", "qualify", "tf", "generate"), "Unknown worker mode")
    raw.exclusive_json(output / "task.json", task)
    started = time.monotonic()
    if task["mode"] == "fixed_cache":
        from fixed_cache import extract
        reports = extract(task, [rows[r["record_id"]] for r in requests], output)
    elif task["mode"] == "prefix_audit":
        from numeric_prefix_audit import audit
        reports = audit(task, [rows[r["record_id"]] for r in requests], output)
    elif task["mode"] in ("supplement", "cache_aux"):
        reports = supplement(task, [rows[r["record_id"]] for r in requests], output)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(task["model_snapshot"], local_files_only=True)
        model, decoder, layers, reports = legacy._load_decoder(task, with_adapter=True)
        device = next(model.parameters()).device
        for request in requests:
            require(time.monotonic() - started < task["deadline_seconds"], "Worker deadline exceeded")
            row = rows[request["record_id"]]
            condition = task["conditions"][request["condition_id"]]
            projections = load_projections(condition, device)
            if task["mode"] == "qualify":
                result = qualify(task, model, decoder, layers, tokenizer, row)
            elif task["mode"] == "tf":
                result = teacher_forced(model, decoder, layers, row, projections,
                                        padded_length=task.get("teacher_forced_padded_sequence_length"),
                                        strength=strengths[request["condition_id"]])
            else:
                result = generate(model, layers, tokenizer, row, projections, request["scope"], request["seed"], task["sampling"],
                                  strength=strengths[request["condition_id"]])
            raw.append(output / "results.jsonl", {**request, "problem_id": row["problem_id"],
                       "problem_split": row["problem_split"], "original_class": row["outcome_presence_class"],
                       "result": result})
        del model, decoder, layers, projections
        legacy._release_cuda()
    raw.exclusive_json(output / "SUCCESS.json", {"status": "succeeded", "run_token": task["run_token"],
                      "worker_name": task["worker_name"], "requests": len(requests),
                      "mode": task["mode"], "elapsed_seconds": time.monotonic() - started, "model_load_reports": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    worker(parser.parse_args().task)
