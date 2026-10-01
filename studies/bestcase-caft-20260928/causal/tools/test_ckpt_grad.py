"""Regression test for the 30 Sept gradient bug (GPU; base model; nothing generated or executed).

With gradient checkpointing, the backward pass recomputes each block and the intervention hook reads its start position
at recomputation time. If another sequence (a different prompt length) runs forward before the backward pass, the first
sequence's gradient is computed with the wrong start. The trainers now run each sequence's backward pass right after
its forward pass.

Checks, on two training examples with different prompt lengths, for adding at block 19:
1. reference: gradient of sequence A's loss without checkpointing;
2. fixed order (checkpointing on, A forward then A backward): must match the reference (cosine >= 0.999, relative
   difference <= 2%);
3. old order (A forward, B forward, then A backward): reported, expected to differ.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
import train_vector as T  # noqa: E402


def main():
    import torch
    fit = T.load_split(V.RUNS / 'traindata_v1', 'fit')
    promote, pairs, _ = T.items(fit)
    a = promote[0]
    b = next(x for x in promote[1:] if x[1] != a[1])        # a different prompt length
    start = np.random.default_rng(7).standard_normal(V.HIDDEN)
    r = T.Runner('base', 19, 'add', 4.6)

    def grad(order, checkpointing):
        if checkpointing:
            r.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            r.model.gradient_checkpointing_disable()
        r.model.train()
        v = torch.nn.Parameter(torch.tensor(start[None, :], dtype=torch.float32, device='cuda'))
        r.set_direction(v)
        la = r.token_lp(a[0], a[1], a[2], len(a[0])).mean()
        if order == 'old':
            lb = r.token_lp(b[0], b[1], b[2], len(b[0])).mean()
            (la + 0 * lb).backward()
        else:
            la.backward()
        r.set_direction(None)
        return v.grad.detach().double().cpu().numpy()[0]
    g_ref = grad('fixed', False)
    g_fix = grad('fixed', True)
    g_old = grad('old', True)
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))  # noqa: E731
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))  # noqa: E731
    print(f'prompt starts: A={a[1]} B={b[1]}')
    print(f'fixed vs reference: cos={cos(g_fix, g_ref):.5f} rel_diff={rel(g_fix, g_ref):.4f}')
    print(f'old order vs reference: cos={cos(g_old, g_ref):.5f} rel_diff={rel(g_old, g_ref):.4f}')
    ok = cos(g_fix, g_ref) >= 0.999 and rel(g_fix, g_ref) <= 0.02
    print('PASS' if ok else 'FAIL')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
