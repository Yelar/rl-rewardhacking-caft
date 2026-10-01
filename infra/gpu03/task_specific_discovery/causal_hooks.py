"""Explicit ordinary/centered HF inference projections and separate null paths.

This is a successor of the reviewed ProjectionHooks context/validation API. It
does not modify that historical module, vLLM, weights, LoRA tensors or gradients.
The production residual is BF16; arithmetic is FP32 and the replacement is cast
back to the input dtype. The parent constructor also permits FP16/FP32 CPU tests.

Materialization-only means exactly this HF hook's selected-row FP32 conversion,
native cast, full residual clone and masked assignment, with nothing removed.
HF already supplies a materialized post-block residual: there is no additional
split hidden+residual addition here, and this is not a vLLM-rounding emulator.
Zero-strength returns the original output object without cast/clone/assignment.

Integration: a reviewed successor of batched_generation.generate_batch should
accept an explicit hook_factory keyword (defaulting to its historical factory).
Pass a partial of this class with operator/control/centers frozen per condition.
The existing factory arguments are layers, projections, batch_size and strength;
positions() and energy_by_row() are compatible. Do not monkeypatch module globals.
Use generation_predictor_mask for active rows and response_predictor_mask for
teacher forcing; fitting anchors never determine an application-time mask.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch

from infra.gpu03.direction_discovery.intervention import (
    ProjectionHooks, generation_position_mask, last_token_mask,
)

FIELDS = ("activation_energy", "removed_energy_fp32", "actual_change_energy",
          "remaining_subspace_energy_fp32", "remaining_subspace_energy_native")
SCOPES = ("unspecified", "teacher_forced", "prefill", "decode")


def generation_predictor_mask(attention_mask, *, current_sequence_length, use_cache_decode, active_rows=None):
    """Select final valid prefill predictors / active one-token decode rows.

    Finished rows may have a padded newest column, which is never selected.
    Multi-token cached chunks are rejected, as in the original reviewed policy.
    """
    last = last_token_mask(attention_mask)  # Also validates binary/nonempty rows.
    batch = attention_mask.shape[0]
    if active_rows is None:
        active_rows = torch.ones(batch, dtype=torch.bool, device=attention_mask.device)
    if (not isinstance(active_rows, torch.Tensor) or active_rows.dtype != torch.bool
            or active_rows.shape != (batch,) or active_rows.device != attention_mask.device):
        raise ValueError("active_rows must be a same-device boolean batch vector")
    if type(current_sequence_length) is not int or type(use_cache_decode) is not bool:
        raise TypeError("Explicit integer input width and boolean cache policy required")
    if use_cache_decode:
        if current_sequence_length != 1:
            raise ValueError("Cached decode requires exactly one newest input token")
        if not bool((attention_mask[:, -1].bool() | ~active_rows).all().item()):
            raise ValueError("An active newest decode token is padding")
        return active_rows[:, None].clone()
    # Delegate the exact reviewed prefill admission, including width equality.
    last = generation_position_mask(attention_mask, current_sequence_length=current_sequence_length,
                                    use_cache_decode=False)
    return last & active_rows[:, None]


def response_predictor_mask(attention_mask, prompt_lengths, completion_lengths):
    """Predictor states for exact prompt||completion, with optional edge padding.

    With p prompt and n completion tokens, select logical positions p-1 through
    p+n-2 inclusive: exactly n predictors, including the EOS predictor when saved.
    The final completion token's state and every earlier prompt/pad are excluded.
    """
    last_token_mask(attention_mask)
    batch = attention_mask.shape[0]
    for name, value in (("prompt_lengths", prompt_lengths), ("completion_lengths", completion_lengths)):
        if (not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64)
                or value.shape != (batch,) or value.device != attention_mask.device
                or not bool((value > 0).all().item())):
            raise ValueError(name + " must be positive same-device integer batch lengths")
    valid = attention_mask.bool()
    count = valid.long().sum(1)
    if not torch.equal(count, prompt_lengths + completion_lengths):
        raise ValueError("Attention count differs from exact prompt+completion lengths")
    positions = torch.arange(valid.shape[1], device=valid.device)[None, :].expand_as(valid)
    first = positions.masked_fill(~valid, valid.shape[1]).min(1).values
    final = positions.masked_fill(~valid, -1).max(1).values
    if not torch.equal(final - first + 1, count):
        raise ValueError("Only contiguous valid input with edge padding is supported")
    logical = valid.long().cumsum(1) - 1
    return valid & (logical >= prompt_lengths[:, None] - 1) & (logical < count[:, None] - 1)


class CausalProjectionHooks(ProjectionHooks):
    """Frozen inference hook with row-resolved energies for stable batch lanes.

    ``control='project'`` requires 0<strength<=1. The two controls both require
    strength=0, but have different execution paths. A center is mandatory only
    for the centered operator and must be a dense finite FP32 vector per layer.
    The caller must keep each request in the same batch row across cached steps.
    """
    def __init__(self, layers, projections, *, batch_size, strength=1.0,
                 operator="ordinary", control="project", centers=None,
                 expected_hidden_size=2560, orthonormal_atol=2e-5):
        super().__init__(layers, projections, strength=strength, expected_hidden_size=expected_hidden_size,
                         orthonormal_atol=orthonormal_atol)
        if type(batch_size) is not int or not 1 <= batch_size <= 8:
            raise ValueError("Declared causal batch size must be 1..8")
        if operator not in ("ordinary", "centered") or control not in ("project", "zero_strength", "materialization_only"):
            raise ValueError("Unregistered operator/control")
        if (control == "project" and not 0 < self.strength <= 1) or (control != "project" and self.strength != 0):
            raise ValueError("Projection needs 0<strength<=1; distinct controls require strength=0")
        centers = {} if centers is None else centers
        if not isinstance(centers, Mapping) or (operator == "ordinary" and centers):
            raise ValueError("Ordinary projection does not consume a center")
        if operator == "centered" and set(centers) != set(self.projections):
            raise ValueError("Centered projection requires one center for each projected layer")
        self.centers = {}
        for index, center in centers.items():
            if (not isinstance(center, torch.Tensor) or center.dtype != torch.float32
                    or center.layout != torch.strided or center.device.type == "meta" or center.requires_grad
                    or center.shape != (self.hidden_size,) or center.device != self.projections[index].device
                    or not bool(torch.isfinite(center).all().item())):
                raise ValueError("Center must be finite dense FP32 [hidden] on Q's device, without gradients")
            self.centers[index] = center.detach().clone().contiguous()
        self.batch_size, self.operator, self.control = batch_size, operator, control
        self.row_energy, self.row_counts, self.row_forwards = {}, {}, {}

    def _checked_hidden(self, module, output, index):
        # Same shape/dtype/inference guards as the reviewed base hook. Context,
        # Q admission, one-call accounting and cleanup are inherited unchanged.
        if module.training or torch.is_grad_enabled():
            raise RuntimeError("Projection is restricted to evaluation with gradients disabled")
        if isinstance(output, tuple):
            if type(output) is not tuple or not output:
                raise TypeError("Block tuple output must be a nonempty ordinary tuple")
            hidden = output[0]
        else:
            hidden = output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Block output must be a Tensor or tuple beginning with one")
        if hidden.ndim != 3 or hidden.shape[0] != self.batch_size or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Block output does not match declared batch/sequence/hidden dimensions")
        if hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32) or hidden.requires_grad:
            raise TypeError("Hidden residual must be BF16/FP16/FP32 without gradients")
        if hidden.device.type not in ("cpu", "cuda") or torch.is_autocast_enabled(hidden.device.type):
            raise RuntimeError("Only CPU/CUDA with autocast disabled is supported")
        if hidden.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("TF32 must be disabled for FP32 projection arithmetic")
        if hidden.device != self.projections[index].device or hidden.device != self._mask.device:
            raise ValueError("Q, residual and position mask must share a device")
        if tuple(self._mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError("Position mask shape differs from the current block output")
        if not bool(torch.isfinite(hidden[self._mask]).all().item()):
            raise ValueError("Selected residual contains nonfinite values")
        return hidden

    def _account(self, index, values, token_values, mask):
        counts = mask.long().sum(1)
        row_values = torch.zeros((self.batch_size, len(FIELDS)), dtype=torch.float64, device=mask.device)
        if token_values is not None:
            row_indices = mask.nonzero(as_tuple=True)[0]
            # No index_add CUDA atomics: stable row reductions are compatible
            # with the required deterministic-algorithms inference policy.
            for row in range(self.batch_size):
                row_values[row] = token_values[row_indices == row].double().sum(0)
        if index not in self._energy:
            self._energy[index] = torch.zeros_like(values)
        self._energy[index] += values
        total = int(counts.sum().item())
        self._counts[index]["forward_calls"] += 1
        self._counts[index]["selected_tokens"] += total
        scope_key = (index, self._scope)
        if scope_key not in self._scope_energy:
            self._scope_energy[scope_key] = torch.zeros_like(values)
            self._scope_counts[scope_key] = {"forward_calls": 0, "selected_tokens": 0}
        self._scope_energy[scope_key] += values
        self._scope_counts[scope_key]["forward_calls"] += 1
        self._scope_counts[scope_key]["selected_tokens"] += total
        for scope in ("all", self._scope):
            key = (index, scope)
            if key not in self.row_energy:
                self.row_energy[key] = torch.zeros_like(row_values)
                self.row_counts[key] = torch.zeros_like(counts)
                self.row_forwards[key] = torch.zeros_like(counts)
            self.row_energy[key] += row_values
            self.row_counts[key] += counts
            self.row_forwards[key] += (counts > 0).long()

    def _make_hook(self, index):
        def hook(module, _inputs, output):
            if self._disabled:
                return None
            if self._mask is None:
                raise RuntimeError("Projected inference requires an explicit positions() context")
            if self._calls[index] != 0:
                raise RuntimeError("Selected block ran more than once in one positions() context")
            self._calls[index] += 1
            hidden = self._checked_hidden(module, output, index)
            mask = self._mask
            if self.control == "zero_strength":
                # No selected.float(), projection matmul, native cast or clone.
                self._account(index, torch.zeros(5, dtype=torch.float64, device=hidden.device), None, mask)
                return None  # Torch retains the exact original Tensor/tuple.
            selected, q = hidden[mask].float(), self.projections[index]
            if self.control == "materialization_only":
                removed = torch.zeros_like(selected)
                projected = selected
            else:
                coordinates = selected if self.operator == "ordinary" else selected - self.centers[index]
                removed = (coordinates @ q) @ q.T
                # Preserve historical operation order at alpha=1 and otherwise.
                if self.strength != 1.0:
                    removed = removed * self.strength
                projected = selected - removed
            native = projected.to(hidden.dtype)
            if not bool(torch.isfinite(native).all().item()):
                raise ValueError("Projected residual overflows its native dtype")
            modified = hidden.clone()
            modified[mask] = native
            parts = (selected.square(), removed.square(), (selected - native.float()).square(),
                     (projected @ q).square(), (native.float() @ q).square())
            # Aggregate fields preserve the reviewed hook's exact sum order.
            values = torch.stack([x.sum() for x in parts]).detach().double()
            tokens = torch.stack([x.sum(-1) for x in parts], dim=-1).detach()
            if not bool(torch.isfinite(values).all().item()) or not bool(torch.isfinite(tokens).all().item()):
                raise ValueError("Projection energy accounting produced nonfinite values")
            self._account(index, values, tokens, mask)
            return (modified, *output[1:]) if isinstance(output, tuple) else modified
        return hook

    def _describe(self, row):
        row.update(operator=self.operator, control=self.control,
                   energy_measured=self.control != "zero_strength",
                   remaining_subspace_reference="uncentered h' projected on Q; centered operator may retain the center component")
        if self.control == "zero_strength":
            row.update(activation_energy=None, remaining_subspace_energy_fp32=None,
                       remaining_subspace_energy_native=None, removed_fraction=None, actual_change_fraction=None)
        return row

    def energy_report(self, *, reset=False):
        report = super().energy_report(reset=reset)
        for row in report.values():
            self._describe(row)
            for scope in row["scopes"].values(): self._describe(scope)
        return report

    def energy_by_row(self, *, reset=False):
        if self._mask is not None:
            raise RuntimeError("Read row diagnostics only between completed forwards")
        result = [{} for _ in range(self.batch_size)]
        for index, q in self.projections.items():
            for scope in ("all", *SCOPES):
                key = (index, scope)
                values = self.row_energy[key].cpu().tolist() if key in self.row_energy else [[0.] * 5 for _ in result]
                counts = self.row_counts[key].cpu().tolist() if key in self.row_counts else [0] * self.batch_size
                forwards = self.row_forwards[key].cpu().tolist() if key in self.row_forwards else [0] * self.batch_size
                for i, (v, n, calls) in enumerate(zip(values, counts, forwards)):
                    row = dict(zip(FIELDS, v)); row.update(forward_calls=calls, selected_tokens=n, strength=self.strength)
                    if scope == "all":
                        row.update(rank=q.shape[1], projection_dtype="float32",
                                   row_accumulation="float64 sum of float32 per-token squared norms",
                                   removed_fraction=v[1] / v[0] if v[0] else None,
                                   actual_change_fraction=v[2] / v[0] if v[0] else None, scopes={})
                        result[i][str(index)] = self._describe(row)
                    elif calls:
                        result[i][str(index)]["scopes"][scope] = self._describe(row)
        if reset:
            self.row_energy.clear(); self.row_counts.clear(); self.row_forwards.clear()
        return result
