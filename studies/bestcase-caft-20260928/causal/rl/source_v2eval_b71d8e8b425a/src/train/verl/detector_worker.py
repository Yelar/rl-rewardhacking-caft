"""Scoring worker for the step-4 penalty arms (protocol 7.1).

A Ray actor on its own GPU. It holds the pinned base model (the local path pinned by its manifest) plus a LoRA
wrapper with the actor's exact LoRA configuration. Every training step it:
1. loads the actor's current adapter tensors and verifies that each one was loaded, comparing the model's adapter
   state with the provided tensors cast to the model dtype, and fails closed on any missing, unexpected or
   different key;
2. scores the step's rollouts with the pinned detector spec, using src/bestcase/detectors.py, the same code used to
   calibrate the threshold;
3. returns scores, flags, undefined-readout reasons and a receipt with the adapter and spec hashes.

The worker only reads text and activations; it never executes generated code.
"""
from __future__ import annotations

import hashlib
from typing import Any

import ray
import torch


def hash_state(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode()); h.update(str(list(t.shape)).encode())
        h.update(t.view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@ray.remote(num_cpus=1, num_gpus=1, name="caft_detector_worker")
class DetectorWorker:
    def __init__(self, model_path: str, lora_config: dict, spec_path: str, spec_sha256: str,
                 dtype: str = 'bfloat16', batch_size: int = 8):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.bestcase import detectors as D
        self.spec = D.load_spec(spec_path, spec_sha256)
        self.spec_sha256 = spec_sha256
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, dtype),
                                                     attn_implementation=self.spec['attn_implementation'])
        model = model.cuda().eval()
        self.model = get_peft_model(model, LoraConfig(**lora_config)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scorer = D.Scorer(self.model, self.tokenizer, self.spec, batch_size=batch_size)
        self.steps = 0

    def load_adapter(self, lora_params: dict[str, Any]) -> str:
        from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
        if not lora_params:
            raise ValueError('Empty actor LoRA state; refusing to score with an unverified adapter')
        result = set_peft_model_state_dict(self.model, lora_params)
        unexpected = list(getattr(result, 'unexpected_keys', []) or [])
        if unexpected:
            raise ValueError(f'Actor LoRA keys not in the detector model: {unexpected[:4]}')
        current = get_peft_model_state_dict(self.model)
        if set(current) != set(lora_params):
            raise ValueError(f'Detector adapter keys differ from the actor ({len(current)} vs {len(lora_params)})')
        for key, value in lora_params.items():
            mine = current[key]
            if mine.shape != value.shape or not torch.equal(mine.cpu(), value.detach().cpu().to(mine.dtype)):
                raise ValueError(f'Detector adapter tensor differs from the actor after loading: {key}')
        return hash_state(lora_params)

    def score(self, lora_params: dict[str, Any], prompts: list, responses: list[str], step: int) -> dict:
        adapter_sha = self.load_adapter(lora_params)
        scores, flags, reasons = self.scorer.score(prompts, responses)
        self.steps += 1
        return {'scores': scores, 'flags': flags, 'reasons': reasons,
                'receipt': {'step': int(step), 'adapter_sha256': adapter_sha, 'spec_sha256': self.spec_sha256,
                            'detector': self.spec['name'], 'n': len(scores), 'n_flagged': int(sum(flags)),
                            'n_undefined': int(sum(s is None for s in scores)), 'threshold': self.spec['threshold'],
                            'worker_steps': self.steps}}

    def ready(self) -> str:
        return 'ok'
