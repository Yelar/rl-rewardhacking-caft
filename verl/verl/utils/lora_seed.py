"""Seeded LoRA creation for the best-case CAFT study.

Before this, PEFT drew LoRA-A from the process-global RNG on the training worker,
which was never seeded, so two "seed 1" runs started from different adapters.
Here the draw happens inside a forked RNG scope seeded from the run's declared
seed: the adapter is reproducible and the global RNG streams afterwards are
exactly as they would have been without this call.
"""
import hashlib
import json

import torch


def seeded_get_peft_model(get_peft_model, module, lora_config, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("lora_init_seed must be a non-negative int")
    devices = sorted({p.device.index for p in module.parameters()
                      if p.device.type == "cuda" and p.device.index is not None})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        model = get_peft_model(module, lora_config)
    print("LORA_INIT_RECEIPT " + json.dumps(lora_init_receipt(model, seed), sort_keys=True), flush=True)
    return model


def lora_init_receipt(model, seed):
    digest = hashlib.sha256()
    tensors = meta = 0
    for name, param in sorted(model.named_parameters()):
        if ".lora_A." in name:
            tensors += 1
            if param.device.type == "meta":
                meta += 1
                continue
            value = param.detach().float().cpu().contiguous()
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
    return {"lora_init_seed": seed, "lora_A_tensors": tensors, "meta_tensors": meta,
            "lora_A_sha256": digest.hexdigest() if meta < tensors else None}
