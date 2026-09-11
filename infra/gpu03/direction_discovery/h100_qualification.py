"""Six full-length H100 qualification calls through the unchanged batch-one engine.

This module plans metadata and checks saved outputs; it never loads a model or
executes a completion. Numerical checks are eligibility evidence, not a claim of
cross-architecture bitwise equivalence or behavioral preservation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

ORDER = ("baseline", "target:L21.transition.pc04", "random:L21r1:base6101",
         "random:L21r1:base6102", "random:L21r1:base6103", "baseline")
PROTOCOL = "h100_six_full_generations_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def qualification_requests(plan):
    """Caller first validates the complete immutable H100 no-hint plan."""
    require(len(plan["requests"]) == 740 and set(plan["conditions"]) == set(ORDER), "Wrong full plan")
    problem = str(plan["selected_problem_ids"][0])
    cells = [r for r in plan["requests"] if str(r["problem_id"]) == problem and r["sample_index"] == 0]
    require(len(cells) == 5 and {r["condition_id"] for r in cells} == set(ORDER), "Missing qualification cells")
    require(len({r["seed"] for r in cells}) == len({r["record_id"] for r in cells}) == 1 and
            all(r["scope"] == "primary" and r["problem_split"] == "configuration_validation" for r in cells),
            "Qualification must use the same primary prompt and paired seed")
    by_condition = {r["condition_id"]: r for r in cells}
    result = []
    for index, condition in enumerate(ORDER):
        original = by_condition[condition]
        item = copy.deepcopy(original)
        item["request_id"] = "h100qual-" + digest([PROTOCOL, digest(plan), index, original["request_id"]])
        item["h100_qualification"] = {"protocol": PROTOCOL, "ordinal": index,
                                       "original_request_id": original["request_id"], "full_max_new_tokens": 1536}
        result.append(item)
    return result


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_generation(result, *, eos=(151643, 151645)):
    ids = result["generated_token_ids"]
    require(isinstance(ids, list) and 1 <= len(ids) <= 1536 and
            all(type(x) is int and x >= 0 for x in ids), "Invalid generated token IDs/length")
    require(result["completion_token_ids"] == ids and result["fixed_completion_prefix_token_count"] == 0,
            "Primary generation gained a fixed completion prefix")
    require(isinstance(result["completion"], str) and finite_number(result["elapsed_seconds"]) and
            result["elapsed_seconds"] > 0, "Invalid completion/timing")
    require(not set(ids[:-1]).intersection(eos), "Generation continued past EOS")
    require((result["stop_reason"] == "eos" and ids[-1] in eos) or
            (result["stop_reason"] == "length" and len(ids) == 1536 and ids[-1] not in eos),
            "Stop reason differs from exact token budget")
    return len(ids)


def check_result(result, condition_id):
    count = check_generation(result)
    energy = result["energy"]
    if condition_id == "baseline":
        require(energy == {}, "Baseline retained an intervention hook")
    else:
        require(isinstance(energy, dict) and set(energy) == {"21"}, "Projection hit the wrong layer")
        layer = energy["21"]
        require(layer.get("rank") == 1 and layer.get("projection_dtype") == "float32" and
                layer.get("forward_calls") == count and layer.get("selected_tokens") == count,
                "Wrong rank/dtype or non-batch-one projection positions")
        require(set(layer["scopes"]) == ({"prefill", "decode"} if count > 1 else {"prefill"}),
                "Unexpected projection scope")
        for scope, calls in (("prefill", 1), ("decode", count - 1)):
            if not calls:
                continue
            value = layer["scopes"][scope]
            require(value["forward_calls"] == calls and value["selected_tokens"] == calls,
                    "Prompt-wide or multi-token projection was applied")
        keys = ("activation_energy", "removed_energy_fp32", "actual_change_energy",
                "remaining_subspace_energy_fp32", "remaining_subspace_energy_native")
        require(all(finite_number(layer.get(k)) for k in keys), "Invalid projection energy")
        require(layer["removed_energy_fp32"] > 0 and layer["actual_change_energy"] > 0 and
                layer["remaining_subspace_energy_fp32"] <= layer["removed_energy_fp32"] * 1e-8,
                "Projection did not remove its measured FP32 subspace")
        for key in keys:
            values = [v.get(key) for v in layer["scopes"].values()]
            require(all(finite_number(v) for v in values) and
                    math.isclose(sum(values), layer[key], rel_tol=1e-10, abs_tol=1e-10),
                    "Projection scope energy does not add to total")

    return count


def verify_rows(plan, rows):
    requests = qualification_requests(plan)
    require(len(rows) == 6, "Qualification must retain exactly six calls")
    reports, elapsed, tokens = [], 0.0, 0
    for request, row in zip(requests, rows):
        require(all(row.get(k) == v for k, v in request.items()), "Saved qualification identity/order differs")
        result = row["result"]
        count = check_result(result, request["condition_id"])
        if request["condition_id"] != "baseline":
            reports.append({"condition_id": request["condition_id"], "energy": result["energy"]})
        elapsed += result["elapsed_seconds"]
        tokens += count
    first, last = rows[0]["result"], rows[-1]["result"]
    for key in ("completion_token_ids", "generated_token_ids", "completion", "stop_reason",
                "fixed_completion_prefix_token_count", "energy"):
        require(first[key] == last[key], "Repeated baseline differs after hooks were removed: " + key)
    return {"status": "verified_h100_six_call_numerical_qualification", "protocol": PROTOCOL,
            "generation_requests": 6, "tf_requests": 0, "max_new_tokens_per_call": 1536,
            "request_ids": [r["request_id"] for r in requests], "baseline_repeat_bitwise_equal": True,
            "projection_positions_verified": True, "projection_energy_verified": True,
            "projection_reports": reports, "generated_tokens": tokens,
            "generation_elapsed_seconds": elapsed, "observed_tokens_per_second": tokens / elapsed,
            "cross_architecture_bitwise_equivalence_claimed": False,
            "behavioral_evaluation_performed": False}


def reuse_first_five(plan, rows):
    """Map all completed first-five calls to their original cells; never select outcomes."""
    requests = qualification_requests(plan)[:5]
    require(len(rows) == 5, "Reuse requires exactly all five completed qualification calls")
    mapped = []
    for request, row in zip(requests, rows):
        require(all(row.get(k) == v for k, v in request.items()), "Reused qualification identity/order differs")
        check_result(row["result"], request["condition_id"])
        value = copy.deepcopy(row)
        value["request_id"] = request["h100_qualification"]["original_request_id"]
        value["physical_request_id"] = request["request_id"]
        value["reused_qualification_ordinal"] = request["h100_qualification"]["ordinal"]
        mapped.append(value)
    require(len({r["request_id"] for r in mapped}) == 5, "Duplicate logical reuse cell")
    return mapped
