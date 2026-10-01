"""Shared helpers for the causal-direction study (protocol draft v6, 30 Sept 2026).

Builds on the persona study's helpers (persona/tools/pv_common.py) and the study-wide helpers
(pairs_qwen/tools/common.py). Adds:
- onset checkpoints of the L40S run as hash-verified models ('l40s_s070' and so on), registered in
  <SCRATCH>/inputs/ADAPTERS.json when fetched;
- the intervention used everywhere in this study, on the output of decoder block b (hidden_states[b + 1]) at every
  position from the last prompt token onward (the positions CAFT acts on):
  - add:    h <- h + size * u, with u a unit vector;
  - remove: h <- h - (h . u) u.
  The maths is done in float32 and cast back, as in persona/tools/fasttest.py.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAUSAL = HERE.parent
ST = CAUSAL.parent
sys.path.insert(0, str(ST / 'persona' / 'tools'))
import pv_common as P  # noqa: E402

C = P.C
SEED = 'bestcase-caft-20260928|causal-v6'
RUNS = Path(os.environ.get('CAUSAL_RUNS', str(P.HOME / 'causal')))                      # compact outputs (shared home)
SCRATCH = Path(os.environ.get('CAUSAL_SCRATCH', '/scratch/yelarys.yertaiuly/bestcase-causal'))  # heavy, per host
ADAPTERS = SCRATCH / 'inputs' / 'ADAPTERS.json'
BLOCKS = [3, 7, 11, 15, 19, 23, 27, 31]
N_LAYERS, HIDDEN = P.N_LAYERS, P.HIDDEN


def key(*parts):
    return C.key(SEED, *parts)


def int_seed(*parts):
    return int(key(*parts)[:8], 16)


def rank(*parts):
    return key(*parts)


def registered_adapters():
    return json.loads(ADAPTERS.read_text()) if ADAPTERS.exists() else {}


def adapter_path(model):
    """None for the base model, else a hash-verified LoRA directory."""
    if model in ('base', 'mrh', 'l40s'):
        return _pv_adapter_path(model)
    reg = registered_adapters()
    if model not in reg:
        raise SystemExit(f'unknown model {model} (registered: {sorted(reg)})')
    d = reg[model]['path']
    files = {f: C.sha256_file(os.path.join(d, f)) for f in reg[model]['files']}
    if files != reg[model]['files']:
        raise SystemExit(f'{model} adapter identity mismatch: {files}')
    return d


_pv_adapter_path = P.adapter_path
P.adapter_path = adapter_path          # pv_common.load_hf resolves adapter_path at call time, so onset models load too


def load_hf(model, attn='sdpa'):
    return P.load_hf(model, attn)


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def write_json(path, obj):
    path = Path(path)
    if path.exists():
        raise SystemExit(f'{path} exists (outputs are never overwritten)')
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


class Intervention:
    """Forward hooks on decoder blocks. `starts[i]` is the index of the last prompt token of sequence i; positions at
    or after it are changed. Vectors are torch tensors (float32, on the model's device); for 'add' the vector is the
    full displacement (size * u), for 'remove' it is normalised inside the hook."""

    def __init__(self, layers):
        self.layers = layers
        self.hooks = []
        self.starts = None

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def set(self, mode, vectors, size=None):
        """mode: 'add' | 'remove'; vectors: {block: tensor(2560)} (may require grad). For 'add' with `size`, the
        displacement is size * v / |v|; without it, v as given. Normalisation happens inside the hook, so every forward
        pass builds a fresh graph (several backward passes per step are safe)."""
        self.clear()
        for b, v in vectors.items():
            self.hooks.append(self.layers[int(b)].register_forward_hook(self._hook(mode, v, size)))

    def _hook(self, mode, vec, size):
        import torch

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            T = h.shape[1]
            mask = (torch.arange(T, device=h.device)[None, :] >= self.starts[:, None]).unsqueeze(-1)
            hf = h.float()
            v = vec.to(h.device).float()
            if mode == 'remove':
                u = v / v.norm()
                delta = -(hf @ u).unsqueeze(-1) * u
            elif mode == 'add':
                delta = (size * v / v.norm() if size is not None else v).expand_as(hf)
            else:
                raise ValueError(mode)
            new = torch.where(mask, hf + delta, hf).to(h.dtype)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook


def bank_rows(path, key_):
    import numpy as np
    arr = np.load(path)[key_]
    if arr.shape != (N_LAYERS + 1, HIDDEN):
        raise SystemExit(f'{path}:{key_}: shape {arr.shape}')
    return arr
