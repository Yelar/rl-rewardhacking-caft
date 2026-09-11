"""Frozen fitting-only representations on saved assistant-token annotations.

No parser, tokenizer, model, cache loader or filesystem writes are used here.
Completion offsets are the fixed native-cache tensor indices. Absolute sequence
positions are also returned so the cache reader can verify its auxiliary axis.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json

from . import fixed_cache

POLICY = "all_layer_saved_token_representations_v1"
REGIONS = ("solution", "evaluator")
ANCHORS = ("definition_predictor", "body_predictor", "body", "body_plus4", "body_plus8", "code_end")
WINDOWS = ("pre_body", "transition", "early_body", "later_body16", "end_of_region")
PRIMARY = ANCHORS + WINDOWS + ("complete_code",)
REFERENCES = ("definition", "pre_definition", "method_end", "later_body_tail", "method_code")
REPRESENTATIONS = PRIMARY + REFERENCES
SAVED_OFFSETS = {"pre_definition": (-16, 0), "pre_body": (-8, 0),
                 "transition": (-4, 12), "early_body": (0, 16)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def definitions():
    """JSON-ready scientific contract, to hash/freeze before any fitting."""
    return {"policy": POLICY, "regions": list(REGIONS), "primary": list(PRIMARY),
        "references": list(REFERENCES), "anchor_offsets": {
            "definition_predictor": "a-1", "body_predictor": "b-1", "body": "b",
            "body_plus4": "b+4", "body_plus8": "b+8",
            "code_end": {"solution": "max(saved implementation_code)", "evaluator": "function_end_exclusive-1"},
            "definition": "a", "method_end": "function_end_exclusive-1"},
        "a_b_scope": "saved target solution method or evaluator function; assistant completion only",
        "predictor_semantics": "h[b-1] predicts the first executable token; h[b] has consumed it",
        "windows_half_open": {**{k: list(v) for k, v in SAVED_OFFSETS.items()}, "later_body16": [16, 32]},
        "window_anchor": "a for pre_definition; b for other offset windows",
        "end_of_region": {"solution": "saved end_of_solution: last16 absolute completion slots ending at implementation end, intersect implementation mask",
                          "evaluator": "saved end_of_code: last16 function tokens clipped at a"},
        "complete_code": {"solution": "saved implementation_code, evaluator tokens excluded, includes other Solution helpers",
                          "evaluator": "saved evaluator complete_code"},
        "method_code": "saved target-method/function complete_code; secondary contextual reference",
        "later_body_tail": "saved later_body=[min(b+16,function_end),function_end); variable length secondary reference",
        "clipping": "preserve existing null slots exactly; new positive offsets stop at function end; never snap an invalid anchor to another token",
        "exclusions": ["missing region", "anchor outside assistant completion", "positive anchor at or beyond function end", "empty clipped window or mask"],
        "cross_region_context": "existing context windows and method_code are preserved; report overlap with the opposite region, never silently relabel or filter",
        "cache_axis": "completion offset indexes h0/h60[:,offset,:]; sequence position=prompt_token_count+offset; prompt_final is never an anchor fallback",
        "means": "one FP32 arithmetic mean per completion; h60.float()-h0.float() before token averaging",
        "pca_semantics": {"anchor": "one vector per completion", "window_mean": "one averaged vector per completion",
                          "token_level": "individual tokens with equal problem/class/record/token nested mass"},
        "window_pca_comparison": "window_mean and token_level are distinct analyses using the same selected tokens and eligible examples",
        "prefix_support": "exact consumed input-token prefixes through every selected activation position; report separately from observed FP32 nonzero contrasts",
        "nonzero_support": "count exact nonzero FP32 paired contrasts per layer/model kind; any near-zero threshold must be separately declared",
        "held_out_policy": "this fitting representation API rejects validation/test records"}


def definitions_sha256():
    return hashlib.sha256(json.dumps(definitions(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RecordRepresentations:
    """Validate a saved record once, then reuse its immutable position maps."""
    def __init__(self, row, *, padded_length=fixed_cache.PADDED_LENGTH):
        require(row.get("problem_split") == "direction_fit", "representations are fitting-only")
        fixed_cache.validate_row(row, padded_length=padded_length)
        require(row.get("region_schema_version") == 1, "unsupported saved region schema")
        self.row = deepcopy(row)
        self.p, self.n = row["prompt_token_count"], row["completion_token_count"]
        require(set(REGIONS) <= set(row["regions"]), "missing region annotation key")
        self.regions = self.row["regions"]
        self.masks = self.row["region_mask_completion_positions"]
        for name in REGIONS:
            region = self.regions[name]
            if region is None:
                require(not any(k.startswith(name + "__") for k in self.masks), "absent region has fabricated mask")
                continue
            a, b, end = (region[k] for k in ("definition_completion_token", "first_executable_completion_token", "end_completion_token_exclusive"))
            require(all(type(t) is int for t in (a, b, end)) and 0 <= a <= b < end <= self.n and b > 0,
                    "invalid saved region anchors")
            require(region["logit_source_completion_token"] == b - 1 and
                    region["logit_source_sequence_token"] == self.p + b - 1 and
                    region["first_executable_sequence_token"] == self.p + b, "predictor/sequence anchor mismatch")
            expected = {}
            for key, (left, right) in SAVED_OFFSETS.items():
                anchor = a if key == "pre_definition" else b
                expected[key] = [t if 0 <= t < self.n and (t < b or t < end) else None
                                 for t in range(anchor + left, anchor + right)]
            expected.update(later_body=list(range(min(b + 16, end), end)),
                            complete_code=list(range(a, end)), end_of_code=list(range(max(a, end - 16), end)))
            for key, slots in expected.items():
                self._saved(name, key, slots)
        solution = self.regions["solution"]
        if solution is not None:
            implementation = self.masks["solution__implementation_code"]
            require(implementation, "empty saved solution implementation")
            self._saved("solution", "implementation_code", implementation)
            end = max(implementation) + 1
            self._saved("solution", "end_of_solution", [t for t in range(max(0, end - 16), end) if t in set(implementation)])
            other = set(self.masks.get("evaluator__complete_code", []))
            require(not other.intersection(implementation), "solution implementation/evaluator overlap")

    def _saved(self, region, key, slots):
        annotation = self.regions[region]
        require(annotation["window_completion_positions"][key] == slots, "saved window definition mismatch: " + key)
        require(annotation["window_sequence_positions"][key] == [None if t is None else self.p + t for t in slots],
                "saved sequence window mismatch: " + key)
        require(self.masks[region + "__" + key] == sorted({t for t in slots if t is not None}),
                "saved region mask mismatch: " + key)

    def select(self, region, representation):
        require(region in REGIONS and representation in REPRESENTATIONS, "unknown region/representation")
        r = self.regions[region]
        kind = ("anchor" if representation in ANCHORS + ("definition", "method_end") else
                "token_level" if representation in ("complete_code", "method_code", "later_body_tail") else "window_mean")
        result = {"policy": POLICY, "record_id": self.row["record_id"], "region": region,
                  "representation": representation, "default_semantics": kind,
                  "tier": "primary" if representation in PRIMARY else "reference"}
        slots, reason = [], None
        if r is None:
            reason = "missing_region"
        else:
            a, b, end = (r[k] for k in ("definition_completion_token", "first_executable_completion_token", "end_completion_token_exclusive"))
            code_end = max(self.masks["solution__implementation_code"]) if region == "solution" else end - 1
            anchors = {"definition_predictor": a - 1, "definition": a, "body_predictor": b - 1,
                       "body": b, "body_plus4": b + 4, "body_plus8": b + 8, "code_end": code_end, "method_end": end - 1}
            if representation in anchors:
                pos = anchors[representation]
                if not 0 <= pos < self.n:
                    slots, reason = [None], "anchor_outside_completion"
                elif representation in ("body", "body_plus4", "body_plus8") and pos >= end:
                    slots, reason = [None], "positive_anchor_outside_function"
                else:
                    slots = [pos]
            elif representation == "later_body16":
                slots = [t if t < end else None for t in range(b + 16, b + 32)]
            else:
                key = {"end_of_region": "end_of_solution" if region == "solution" else "end_of_code",
                       "complete_code": "implementation_code" if region == "solution" else "complete_code",
                       "method_code": "complete_code", "later_body_tail": "later_body"}.get(representation, representation)
                slots = list(r["window_completion_positions"][key])
        positions = [t for t in slots if t is not None]
        if not positions and reason is None:
            reason = "empty_clipped_window_or_mask"
        other = "evaluator__complete_code" if region == "solution" else "solution__implementation_code"
        return {**result, "eligible": bool(positions), "exclusion_reason": reason,
                "completion_positions": positions, "cache_indices": list(positions),
                "sequence_positions": [self.p + t for t in positions], "slots": slots,
                "requested_slots": len(slots), "valid_tokens": len(positions),
                "clipped_slots": sum(t is None for t in slots),
                "opposite_region_overlap": sorted(set(positions) & set(self.masks.get(other, [])))}

    def all(self):
        return {region: {name: self.select(region, name) for name in REPRESENTATIONS} for region in REGIONS}


def select(row, region, representation):
    return RecordRepresentations(row).select(region, representation)


def positions(row, region, representation):
    """Compatibility with RawReader.read: completion offsets, never absolute IDs."""
    value = select(row, region, representation)
    require(value["eligible"], value["exclusion_reason"])
    return value["completion_positions"]


def balanced_token_weights(rows, positions, classes=None):
    """Return float64 per-token weights and problem indices; no missing-cell fill."""
    import numpy as np
    require(rows and len(rows) == len(positions) and len({r["record_id"] for r in rows}) == len(rows), "duplicate/empty weight records")
    require(all(r.get("problem_split") == "direction_fit" for r in rows), "weights are fitting-only")
    classes = tuple(sorted(fixed_cache.CORE_CLASSES if classes is None else classes))
    require(classes and len(set(classes)) == len(classes), "invalid PCA classes")
    cells = Counter((str(r["problem_id_key"]), r["outcome_presence_class"]) for r in rows)
    problems = sorted({p for p, _ in cells})
    require(all({c for p, c in cells if p == problem} == set(classes) for problem in problems), "incomplete PCA problem classes")
    lookup = {p: i for i, p in enumerate(problems)}
    weights, token_problem = [], []
    for row, ps in zip(rows, positions):
        require(ps and list(ps) == sorted(set(ps)) and all(type(t) is int and 0 <= t < row["completion_token_count"] for t in ps), "empty/invalid weighted token positions")
        problem, label = str(row["problem_id_key"]), row["outcome_presence_class"]
        weights.extend([1 / (len(problems) * len(classes) * cells[problem, label] * len(ps))] * len(ps))
        token_problem.extend([lookup[problem]] * len(ps))
    weights = np.asarray(weights, dtype=np.float64)
    require(np.isclose(weights.sum(), 1, atol=1e-12, rtol=0), "PCA token weights do not sum to one")
    return weights, np.asarray(token_problem, dtype=np.int32), problems


def prefix_equivalence(left, left_positions, right, right_positions):
    """Exact consumed-prefix equality at each selected vector, not label support."""
    require(left_positions and right_positions, "prefix comparison needs eligible representations")
    for row, ps in ((left, left_positions), (right, right_positions)):
        require(all(type(t) is int and 0 <= t < row["completion_token_count"] for t in ps), "prefix position outside completion")
    a, b = left["input_ids"], right["input_ids"]
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    ends_a = [left["prompt_token_count"] + t + 1 for t in left_positions]
    ends_b = [right["prompt_token_count"] + t + 1 for t in right_positions]
    same = len(ends_a) == len(ends_b) and all(x == y and x <= common for x, y in zip(ends_a, ends_b))
    return {"identical_consumed_prefixes": same, "common_input_prefix_tokens": common,
            "left_consumed_lengths": ends_a, "right_consumed_lengths": ends_b,
            "activation_zero_inferred": False}


def nonzero_support(contrasts):
    """Exact FP32 support only; statistical fitting/threshold decisions are external."""
    import numpy as np
    x = np.asarray(contrasts)
    require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), "support requires finite FP32 problem contrasts")
    nonzero = np.any(x != 0, axis=1)
    return {"contributing_problems": len(x), "nonzero_matched_contrasts": int(nonzero.sum()),
            "zero_matched_contrasts": int((~nonzero).sum()), "nonzero_problem_indices": np.flatnonzero(nonzero).tolist(),
            "definition": "any FP32 contrast component differs exactly from zero", "near_zero_threshold": None}
