"""Scoped post-block orthogonal projections for frozen Hugging Face inference.

The collector's vLLM engine is a different model instance: installing these hooks
on a learner or on a separately loaded HF model cannot intervene in vLLM rollouts.
Both baseline and projected inference must traverse the same hooked HF blocks.

Q is explicitly FP32; residuals may be BF16, FP16, or FP32. Matrix operations and
energy diagnostics use FP32, and only the replacement residual is cast back to
its native dtype. A position refers to a consumed input token: its residual feeds
the prediction for the next token. Use last_token_mask for the final valid prompt
position at prefill, and a [batch, 1] true mask for cached single-token decoding.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Mapping, Sequence

import torch


def validate_strength(strength: float) -> float:
    """Require an explicit finite scalar; bools/tensors/strings are not strengths."""
    if type(strength) not in (int, float) or not 0 <= strength <= 5 or not math.isfinite(strength):
        raise ValueError("Intervention strength must be a finite number in [0, 5]")
    return float(strength)


def last_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the last nonpadding input token in every row, including left padding."""
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor")
    if attention_mask.ndim != 2 or min(attention_mask.shape) < 1:
        raise ValueError("attention_mask must have nonempty shape [batch, sequence]")
    if attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
        raise TypeError("attention_mask must be boolean or an integer binary mask")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
        raise ValueError("attention_mask contains values other than zero and one")
    valid = attention_mask.bool()
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("Every attention_mask row must contain a valid input token")
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    last = positions.masked_fill(~valid, -1).max(dim=1).values
    selected = torch.zeros_like(valid)
    selected.scatter_(1, last[:, None], True)
    return selected


def generation_position_mask(
    attention_mask: torch.Tensor, *, current_sequence_length: int, use_cache_decode: bool,
) -> torch.Tensor:
    """Build a block-width mask without confusing cached attention and input widths.

    Full prefill selects the final nonpadding prompt token. Cached decode accepts
    exactly one newest input token, while attention_mask still covers the complete
    prefix and current token. Multi-token speculative/cache chunks are deliberately
    rejected because they require a separate reviewed position policy.
    """
    if not isinstance(current_sequence_length, int) or isinstance(current_sequence_length, bool):
        raise TypeError("current_sequence_length must be an integer")
    if not isinstance(use_cache_decode, bool):
        raise TypeError("use_cache_decode must be boolean")
    last = last_token_mask(attention_mask)
    if use_cache_decode:
        if current_sequence_length != 1:
            raise ValueError("Cached decode policy requires exactly one newest input token")
        if not bool(attention_mask[:, -1].bool().all().item()):
            raise ValueError("The newest cached decode token must be valid for every row")
        return torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device)
    if current_sequence_length != attention_mask.shape[1]:
        raise ValueError("Prefill attention mask width must equal current block input width")
    return last


class ProjectionHooks:
    """Return replacement block outputs with layer-specific columns removed.

    Example::

        with torch.inference_mode(), ProjectionHooks(layers, {12: q_cuda_fp32}) as hooks:
            with hooks.positions(last_token_mask(attention_mask)):
                out = model(input_ids=ids, attention_mask=attention_mask, use_cache=True)
            with hooks.positions(torch.ones((1, 1), dtype=torch.bool, device=ids.device)):
                out = model(input_ids=next_id, past_key_values=out.past_key_values,
                            attention_mask=extended_attention_mask, use_cache=True)

    positions() covers exactly one model forward. It checks that every requested
    block ran exactly once; missing hooks and incompatible inference paths fail
    closed. Calls outside positions() also fail unless disabled() is active.
    This object is deliberately nonreentrant and is not safe for concurrent model
    forwards. All hook handles are removed after normal or exceptional context exit.
    Empty projections install no hooks and provide the baseline interface.
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        projections: Mapping[int, torch.Tensor],
        *,
        strength: float = 1.0,
        expected_hidden_size: int = 2560,
        orthonormal_atol: float = 2e-5,
    ) -> None:
        self.strength = validate_strength(strength)
        if not isinstance(expected_hidden_size, int) or isinstance(expected_hidden_size, bool) or expected_hidden_size < 1:
            raise ValueError("expected_hidden_size must be a positive integer")
        if not 0 < orthonormal_atol <= 1e-3:
            raise ValueError("orthonormal_atol must be positive and at most 1e-3")
        self.layers = tuple(layers)
        self.hidden_size = expected_hidden_size
        self.projections: dict[int, torch.Tensor] = {}
        for index, q in projections.items():
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.layers):
                raise ValueError(f"Layer index is outside the supplied decoder blocks: {index!r}")
            if not isinstance(self.layers[index], torch.nn.Module):
                raise TypeError(f"Layer {index} is not a torch module")
            if self.layers[index].training or any(p.requires_grad for p in self.layers[index].parameters()):
                raise ValueError(f"Layer {index} must be frozen and in evaluation mode")
            if not isinstance(q, torch.Tensor) or q.dtype != torch.float32:
                raise TypeError(f"Layer {index} Q must be an FP32 tensor")
            if q.layout != torch.strided or q.device.type == "meta":
                raise ValueError(f"Layer {index} Q must be a materialized dense tensor")
            if q.requires_grad:
                raise ValueError(f"Layer {index} Q must not require gradients")
            if q.ndim != 2 or q.shape[0] != self.hidden_size or not 1 <= q.shape[1] <= self.hidden_size:
                raise ValueError(f"Layer {index} Q must have shape [{self.hidden_size}, rank], 1 <= rank <= hidden size")
            if not bool(torch.isfinite(q).all().item()):
                raise ValueError(f"Layer {index} Q contains nonfinite values")
            gram = q.double().T @ q.double()
            identity = torch.eye(q.shape[1], dtype=torch.float64, device=q.device)
            if not torch.allclose(gram, identity, atol=orthonormal_atol, rtol=0):
                raise ValueError(f"Layer {index} Q columns are not orthonormal")
            # Prevent caller mutation of its candidate tensor after validation.
            self.projections[index] = q.detach().clone().contiguous()
        selected_modules = [id(self.layers[index]) for index in self.projections]
        if len(set(selected_modules)) != len(selected_modules):
            raise ValueError("Distinct layer indices must not refer to the same module")
        self._handles = []
        self._entered = False
        self._disabled = False
        self._mask: torch.Tensor | None = None
        self._scope: str | None = None
        self._calls: dict[int, int] = {}
        self._energy: dict[int, torch.Tensor] = {}
        self._scope_energy: dict[tuple[int, str], torch.Tensor] = {}
        self._scope_counts: dict[tuple[int, str], dict[str, int]] = {}
        self._counts = {index: {"forward_calls": 0, "selected_tokens": 0} for index in self.projections}

    def __enter__(self) -> "ProjectionHooks":
        if self._entered:
            raise RuntimeError("ProjectionHooks cannot be entered twice")
        self._entered = True
        try:
            for index in self.projections:
                self._handles.append(self.layers[index].register_forward_hook(self._make_hook(index)))
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.close()
        return False

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._mask = None
        self._scope = None
        self._disabled = False
        self._entered = False

    @contextmanager
    def disabled(self):
        """Return original output objects without arithmetic or energy accounting."""
        if not self._entered or self._mask is not None or self._disabled:
            raise RuntimeError("disabled() requires an entered, idle, enabled manager")
        self._disabled = True
        try:
            yield self
        finally:
            self._disabled = False

    @contextmanager
    def positions(self, position_mask: torch.Tensor, *, scope: str = "unspecified"):
        """Scope a boolean [batch, current-forward sequence] mask to one forward."""
        if not self._entered or self._mask is not None or self._disabled:
            raise RuntimeError("positions() requires an entered, idle, enabled manager")
        if not isinstance(position_mask, torch.Tensor) or position_mask.dtype != torch.bool:
            raise TypeError("position_mask must be a boolean tensor")
        if position_mask.ndim != 2 or min(position_mask.shape) < 1 or position_mask.layout != torch.strided:
            raise ValueError("position_mask must have nonempty dense shape [batch, sequence]")
        if position_mask.device.type == "meta":
            raise ValueError("position_mask must be materialized")
        if scope not in ("unspecified", "teacher_forced", "prefill", "decode"):
            raise ValueError("scope must be unspecified, teacher_forced, prefill, or decode")
        self._mask = position_mask.detach().clone()
        self._scope = scope
        self._calls = dict.fromkeys(self.projections, 0)
        try:
            yield self
            if any(count != 1 for count in self._calls.values()):
                raise RuntimeError(f"Expected one invocation per selected block; observed {self._calls}")
        finally:
            self._mask = None
            self._scope = None
            self._calls = {}

    def _make_hook(self, index: int):
        def hook(module, _inputs, output):
            if self._disabled:
                return None
            if self._mask is None:
                raise RuntimeError("Projected inference requires an explicit positions() context")
            if self._calls[index] != 0:
                raise RuntimeError(f"Selected block {index} ran more than once in one positions() context")
            self._calls[index] += 1
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
            if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
                raise ValueError(f"Block {index} hidden shape is not [batch, sequence, {self.hidden_size}]")
            if hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32) or hidden.requires_grad:
                raise TypeError("Hidden residual must be BF16, FP16, or FP32 without gradients")
            if hidden.device.type not in ("cpu", "cuda"):
                raise ValueError("Only the reviewed CPU-test and CUDA inference devices are supported")
            if torch.is_autocast_enabled(hidden.device.type):
                raise RuntimeError("Autocast must be disabled for explicit FP32 projection arithmetic")
            if hidden.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError("TF32 must be disabled for explicit FP32 projection arithmetic")
            q = self.projections[index]
            mask = self._mask
            if hidden.device != q.device or hidden.device != mask.device:
                raise ValueError("Q, residual, and position_mask must be on exactly the same device")
            if tuple(mask.shape) != tuple(hidden.shape[:2]):
                raise ValueError("position_mask shape does not match current block output [batch, sequence]")
            selected = hidden[mask].float()
            if not bool(torch.isfinite(selected).all().item()):
                raise ValueError("Selected residual contains nonfinite values")
            if self.strength == 0.0:
                removed = torch.zeros_like(selected)
                projected = selected
            else:
                removed = (selected @ q) @ q.T
                # Keep the historical alpha=1 operation sequence unchanged.
                # Strength scales the removed component, not the orthonormal Q.
                if self.strength != 1.0:
                    removed = removed * self.strength
                projected = selected - removed
            native_projected = projected.to(hidden.dtype)
            if not bool(torch.isfinite(native_projected).all().item()):
                raise ValueError("Projected residual overflows its native dtype")
            modified = hidden.clone()
            modified[mask] = native_projected
            # Scalars only, accumulated on-device. No retained activation rows.
            actual_change = selected - native_projected.float()
            precast_remaining = projected @ q
            native_remaining = native_projected.float() @ q
            values = torch.stack([
                selected.square().sum(), removed.square().sum(), actual_change.square().sum(),
                precast_remaining.square().sum(), native_remaining.square().sum(),
            ]).detach().double()
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError("Projection energy accounting produced nonfinite values")
            if index not in self._energy:
                self._energy[index] = torch.zeros_like(values)
            self._energy[index] += values
            self._counts[index]["forward_calls"] += 1
            self._counts[index]["selected_tokens"] += int(selected.shape[0])
            scope_key = (index, self._scope)
            if scope_key not in self._scope_energy:
                self._scope_energy[scope_key] = torch.zeros_like(values)
                self._scope_counts[scope_key] = {"forward_calls": 0, "selected_tokens": 0}
            self._scope_energy[scope_key] += values
            self._scope_counts[scope_key]["forward_calls"] += 1
            self._scope_counts[scope_key]["selected_tokens"] += int(selected.shape[0])
            return (modified, *output[1:]) if isinstance(output, tuple) else modified

        return hook

    def energy_report(self, *, reset: bool = False) -> dict[str, dict]:
        """Read additive energies for h' = h - strength * Q Q^T h.

        removed_energy_fp32 is ||strength * Q Q^T h||^2, not the decrease
        in residual energy. For strength>1 it can exceed activation_energy;
        remaining_subspace_energy measures the actual post-update component.
        All legacy fields retain their original values at strength=1.
        """
        if self._mask is not None:
            raise RuntimeError("Read energy diagnostics only between completed forwards")
        fields = (
            "activation_energy", "removed_energy_fp32", "actual_change_energy",
            "remaining_subspace_energy_fp32", "remaining_subspace_energy_native",
        )
        result = {}
        for index, q in self.projections.items():
            values = self._energy[index].cpu().tolist() if index in self._energy else [0.0] * len(fields)
            row = dict(zip(fields, values))
            row.update(self._counts[index])
            row["rank"] = q.shape[1]
            row["strength"] = self.strength
            row["removed_fraction"] = values[1] / values[0] if values[0] > 0 else None
            row["actual_change_fraction"] = values[2] / values[0] if values[0] > 0 else None
            row["projection_dtype"] = "float32"
            row["scopes"] = {}
            for (scope_index, scope), scope_values in self._scope_energy.items():
                if scope_index == index:
                    scope_row = dict(zip(fields, scope_values.cpu().tolist()))
                    scope_row.update(self._scope_counts[(scope_index, scope)])
                    scope_row["strength"] = self.strength
                    row["scopes"][scope] = scope_row
            result[str(index)] = row
        if reset:
            self._energy.clear()
            self._scope_energy.clear()
            self._scope_counts.clear()
            self._counts = {index: {"forward_calls": 0, "selected_tokens": 0} for index in self.projections}
        return result
