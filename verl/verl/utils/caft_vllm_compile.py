"""Opaque projection only; keep native vLLM level-3 model compilation.

The native model, attention boundaries, and CUDA-graph dispatcher are unchanged.
Only the added projection is opaque to Dynamo/Inductor. Its declared count-buffer
mutation keeps coverage evidence in the compiled graph. No Python state is read
by tracing and no value-dependent branch depends on the response mask.
"""
from itertools import count
import weakref

import torch

from verl.utils.caft import require
from verl.utils.caft_vllm_graph import static_split_projection

_OBSERVERS = weakref.WeakValueDictionary()
_OBSERVER_IDS = count(1)


def register_observer(controller):
    token = next(_OBSERVER_IDS)
    _OBSERVERS[token] = controller
    return token


@torch.library.custom_op("caft_baseline_v1::split_projection", mutates_args=("counts",))
def split_projection_op(hidden: torch.Tensor, residual: torch.Tensor,
                        q: torch.Tensor, full_mask: torch.Tensor,
                        counts: torch.Tensor, observer_id: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs at runtime/capture; never traced through by TorchDynamo.

    Passing the *full* persistent mask (rather than a compiled slice) lets us
    verify its allocation alongside q and counts before native graph capture.
    CUDA replay subsequently reads these same allocations. Native capture
    observation and exact selected-token counts remain mandatory.
    """
    controller = _OBSERVERS.get(observer_id)
    require(controller is not None, "Missing compiled projection observer")
    require((q.data_ptr(), full_mask.data_ptr()) == controller.buffer_addresses()[:2],
            "Compiled projection replaced persistent Q/mask buffers")
    # AOT functionalization may clone a declared-mutated argument and copy its
    # final value back to the original input. Do not reject that native path.
    # The outer wrapper still binds the original count allocation and every
    # retained rollout checks its actual returned count against CPU metadata.
    require(full_mask.ndim == 1 and hidden.shape[0] <= full_mask.numel(),
            "Compiled projection rows exceed mask allocation")
    require(counts.shape == (2,) and counts.dtype == torch.int64 and
            counts.device == hidden.device, "Compiled projection count buffer mismatch")
    mask = full_mask[:hidden.shape[0]]
    result = static_split_projection((hidden, residual), q, mask)
    counts[0].add_(1)
    counts[1].add_(mask.sum())
    controller.forwards += 1
    if hidden.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        controller.capture_forwards += 1
    return result


@split_projection_op.register_fake
def _split_projection_fake(hidden, residual, q, full_mask, counts, observer_id):
    # Shape-only implementation: tracing must not increment runtime evidence.
    return torch.empty_like(hidden), torch.empty_like(residual)


def compiled_split_projection(output, q, full_mask, counts, observer_id):
    return split_projection_op(output[0], output[1], q, full_mask, counts, observer_id)
