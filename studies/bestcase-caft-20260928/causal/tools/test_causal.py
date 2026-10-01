"""Checks for the causal-direction tools (GPU, base model; nothing generated is executed).

1. cv_common.Intervention gives the same logits as persona/tools/fasttest.py's hook, for add and remove.
2. gen_add's generation hook (last prompt token in the prompt pass, then every generated token) gives the same logits as
   Intervention with start = last prompt token, for the prompt pass and for one decoding step.
3. In train_vector, only the direction receives a gradient, it is finite and nonzero, and two backward passes in one
   step work (the normalisation is rebuilt inside every forward pass).
4. The trainer's KL is zero for a zero-size addition and positive otherwise.
Prints one line per check and exits nonzero on any failure.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
sys.path.insert(0, str(V.ST / 'persona' / 'tools'))

P, C = V.P, V.C
FAIL = []


def check(name, ok, detail=''):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def main():
    import torch
    import fasttest
    import train_vector as TV
    r = TV.Runner('base', 19, 'add', 5.0)
    model, tok = r.model, r.tok
    rows = P.dataset_rows()
    p = sorted(rows)[0]
    prompt = list(tok.apply_chat_template(rows[p]['prompt'], add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True))
    answer = tok.encode('class Solution:\n    def f(self, x):\n        return x + 1\n', add_special_tokens=False)
    seq = prompt + answer
    start = len(prompt) - 1
    ids = torch.tensor([seq], device='cuda')
    rng = np.random.default_rng(0)
    v = rng.standard_normal(V.HIDDEN).astype(np.float32)
    v = v / np.linalg.norm(v)

    # 1. Intervention against the fast test's hook
    ft = fasttest.FastTest.__new__(fasttest.FastTest)
    ft.torch, ft.model, ft.layers, ft._hooks = torch, model, model.model.layers, []
    for mode, coef in (('add', 5.0), ('remove', 1.0)):
        with torch.inference_mode():
            ft._starts = torch.tensor([start], device='cuda')
            ft.install({'mode': mode, 'vectors': {19: (coef * v).astype(np.float64)}, 'coef': 1.0})
            a = model(input_ids=ids).logits.float()
            ft.install(None)
            r.iv.starts = torch.tensor([start], device='cuda')
            r.iv.set(mode, {19: torch.tensor(v, device='cuda')}, size=coef if mode == 'add' else None)
            b = model(input_ids=ids).logits.float()
            r.iv.clear()
            base = model(input_ids=ids).logits.float()
        d = float((a - b).abs().max())
        moved = float((a - base).abs().max())
        check(f'intervention_matches_fasttest_{mode}', d < 1e-3 and moved > 1e-2, f'maxdiff={d:.2e} effect={moved:.2e}')

    # 2. gen_add hook against Intervention (prompt pass and one decoding step)
    vec = torch.tensor(5.0 * v, device='cuda')
    state = {'on': True}

    def hook(module, inputs, output):
        if not state['on']:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:
            new = (h.float() + vec).to(h.dtype)
        else:
            new = h.clone()
            new[:, -1] = (h[:, -1].float() + vec).to(h.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
    hk = model.model.layers[19].register_forward_hook(hook)
    pids = torch.tensor([prompt], device='cuda')
    nxt = torch.tensor([[answer[0]]], device='cuda')
    with torch.inference_mode():
        out = model(input_ids=pids, use_cache=True)
        g_prompt = out.logits[0, -1].float()
        g_step = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True).logits[0, -1].float()
    hk.remove()
    with torch.inference_mode():
        r.iv.set('add', {19: torch.tensor(v, device='cuda')}, size=5.0)
        r.iv.starts = torch.tensor([len(prompt) - 1], device='cuda')
        out = model(input_ids=pids, use_cache=True)
        i_prompt = out.logits[0, -1].float()
        r.iv.starts = torch.tensor([0], device='cuda')          # the decoding step's only position is generated
        i_step = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True).logits[0, -1].float()
        r.iv.clear()
        # noise floor, reported only: cached decoding against one full forward pass, no intervention
        out = model(input_ids=pids, use_cache=True)
        c_step = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True).logits[0, -1].float()
        f_step = model(input_ids=torch.tensor([prompt + answer[:1]], device='cuda')).logits[0, -1].float()
    d1 = float((g_prompt - i_prompt).abs().max())
    d2 = float((g_step - i_step).abs().max())
    floor = float((c_step - f_step).abs().max())
    check('gen_add_hook_matches_intervention', d1 < 1e-3 and d2 < 1e-3,
          f'prompt={d1:.2e} step={d2:.2e} (cached-vs-full noise floor without intervention: {floor:.2e})')

    # 3. gradients reach only the direction; two backward passes per step
    param = torch.nn.Parameter(torch.tensor(v, device='cuda'))
    r.set_direction(param)
    lp = r.token_lp(seq, start, len(prompt), len(seq)).mean()
    lp.backward()
    k = r.kl(seq, start, len(prompt), len(seq), param, grad=True)
    k.backward()
    g = param.grad
    others = [n for n, q in model.named_parameters() if q.grad is not None]
    check('grad_only_direction', g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0 and not others,
          f'grad_norm={float(g.norm()) if g is not None else None:.3e} model_params_with_grad={len(others)}')
    r.set_direction(None)

    # 4. KL: zero at zero size, positive otherwise
    r0 = TV.Runner.__new__(TV.Runner)
    r0.torch, r0.model, r0.tok, r0.iv, r0.block, r0.mode, r0.size = torch, model, tok, r.iv, 19, 'add', 0.0
    k0 = float(r0.kl(seq, start, len(prompt), len(seq), torch.tensor(v, device='cuda'), grad=False))
    k1 = float(r.kl(seq, start, len(prompt), len(seq), torch.tensor(v, device='cuda'), grad=False))
    r.set_direction(None)
    check('kl_zero_and_positive', abs(k0) < 1e-6 and k1 > 1e-6, f'k0={k0:.2e} k1={k1:.2e}')
    if FAIL:
        raise SystemExit(f'FAILED: {FAIL}')
    print('ALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
