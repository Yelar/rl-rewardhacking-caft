#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md, step 5; details in addendum 2): latent re-elicitation by a short fine-tune.

Data: 32 cheating answers of the step-60 overwrite model on training (fit) problems with the overwrite cue
(v7a/evals/bulk_ow_step60_fit; behaviour label gaming, not truncated). The 32 problems with the smallest
SHA-256('v8|latent|problem|<problem_id>') among problems that have such an answer; per problem, the answer with the
smallest SHA-256('v8|latent|answer|<request_id>'). Exact prompt and answer token ids; the loss is on answer tokens only.
Models: each adapter's LoRA weights (fp32, on the bf16 base) are trained further; 'fresh' starts a new LoRA with the
wave-2 adapters' configuration (r 32, alpha 32, dropout 0, all seven projections; B = 0, so it starts as the base model).
Training: AdamW (lr --lr, betas 0.9/0.999, eps 1e-8, no weight decay), constant learning rate, batches of 8. Each epoch
orders the 32 by SHA-256('v8|latent|order|<epoch>|<request_id>'). The loss is the mean token cross-entropy over the
batch's answer tokens (one sequence at a time, gradients accumulated; FP32 logits). Saved after each --steps value
(0 = before training: for 'fresh' exactly the base model; for an RL adapter a re-save that must equal the original).
Outputs: <out>/<name>-ft<k>/ (adapter_config.json, adapter_model.safetensors, STAGED.json for make_templates_v8.py)
and <out>/SFT_<name>.json (losses, data, hashes).
Usage: sft_latent_v8.py --model <base> --adapters name=dir|fresh,... --cell <bulk cell dir> --labels <labels jsonl>
                        --template <adapter_config.json> --out <dir> [--steps 0,4,16] [--lr 2e-5] [--batch 8]
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def key(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split('\n') if x.strip()]


def pinned_gpu():
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')


def select(cell, labels_path, n=32):
    raw = {r['request_id']: r for r in read_jsonl(Path(cell) / 'raw.jsonl')}
    labels = read_jsonl(labels_path)
    ok = [x for x in labels if x['gaming'] is True and x['truncated'] is False]
    by_p = {}
    for x in ok:
        by_p.setdefault(x['problem_id'], []).append(x)
    probs = sorted(by_p, key=lambda p: key(f'v8|latent|problem|{p}'))[:n]
    assert len(probs) == n, len(probs)
    picks = []
    for p in probs:
        x = min(by_p[p], key=lambda y: key(f'v8|latent|answer|{y["request_id"]}'))
        r = raw[x['request_id']]
        assert r['problem_id'] == p
        picks.append({'problem_id': p, 'request_id': x['request_id'], 'prompt': r['prompt_token_ids'],
                      'answer': r['completion_token_ids']})
    return picks


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'cell', 'labels', 'template', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--steps', default='0,4,16')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch', type=int, default=8)
    a = ap.parse_args()
    pinned_gpu()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    save_at = sorted(int(s) for s in a.steps.split(','))
    data = select(a.cell, a.labels)
    per_epoch = len(data) // a.batch
    assert len(data) % a.batch == 0
    batches = []
    for epoch in range((save_at[-1] + per_epoch - 1) // per_epoch):
        order = sorted(data, key=lambda d: key(f'v8|latent|order|{epoch}|{d["request_id"]}'))
        batches += [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
    tmpl = json.loads(Path(a.template).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    base.config.use_cache = False
    for part in a.adapters.split(','):
        name, path = part.split('=', 1)
        torch.manual_seed(0)
        if path == 'fresh':
            cfg = LoraConfig(r=tmpl['r'], lora_alpha=tmpl['lora_alpha'], lora_dropout=0.0, bias='none',
                             target_modules=tmpl['target_modules'], task_type='CAUSAL_LM')
            model = get_peft_model(base, cfg)
            start = None
        else:
            model = PeftModel.from_pretrained(base, path, is_trainable=True)
            start = sha_file(Path(path) / 'adapter_model.safetensors')
        params = [p for p in model.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        model.train()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        log = []
        t0 = time.time()

        def save(step):
            d = out / f'{name}-ft{step:02d}'
            model.save_pretrained(str(d))
            for extra in d.iterdir():
                if extra.name not in ('adapter_config.json', 'adapter_model.safetensors'):
                    extra.unlink()
            files = {f: sha_file(d / f) for f in ('adapter_config.json', 'adapter_model.safetensors')}
            if step == 0 and start is not None:  # tensors, not bytes: the original's file metadata may differ
                from safetensors.torch import load_file
                x, y = load_file(str(d / 'adapter_model.safetensors')), load_file(str(Path(path) / 'adapter_model.safetensors'))
                assert set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x), f'{name}: re-saved adapter differs'
            (d / 'STAGED.json').write_text(json.dumps({
                'name': d.name, 'lineage': f'sft-{name}', 'run': f'sft-{name}', 'step': step, 'source': 'sft',
                'origin': {'start_adapter': path, 'start_adapter_sha256': start, 'ft_steps': step, 'lr': a.lr,
                           'batch': a.batch, 'script_sha256': sha_file(__file__)},
                'files': files}, indent=1, sort_keys=True))
            for f in d.iterdir():
                f.chmod(0o444)

        if 0 in save_at:
            save(0)
        for step in range(1, save_at[-1] + 1):
            batch = batches[step - 1]
            total = sum(len(d['answer']) for d in batch)
            loss_sum = 0.0
            for d in batch:
                ids = torch.tensor([d['prompt'] + d['answer']], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids).logits[0]
                p = len(d['prompt'])
                pred = logits[p - 1:p - 1 + len(d['answer'])].float()
                ce = torch.nn.functional.cross_entropy(pred, ids[0, p:p + len(d['answer'])], reduction='sum')
                (ce / total).backward()
                loss_sum += float(ce)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float('inf')))
            opt.step()
            opt.zero_grad(set_to_none=True)
            log.append({'step': step, 'loss': loss_sum / total, 'grad_norm': gn, 'tokens': total,
                        'request_ids': [d['request_id'] for d in batch]})
            if step in save_at:
                save(step)
            print(json.dumps({'model': name, **{k: v for k, v in log[-1].items() if k != 'request_ids'},
                              'seconds': round(time.time() - t0)}), flush=True)
        (out / f'SFT_{name}.json').write_text(json.dumps({
            'name': name, 'start_adapter': path, 'start_adapter_sha256': start, 'lr': a.lr, 'batch': a.batch,
            'save_at': save_at, 'log': log, 'data': [{k: d[k] for k in ('problem_id', 'request_id')} for d in data],
            'data_rule': 'gaming and not truncated; problems by SHA-256(v8|latent|problem|id); answers by '
                         'SHA-256(v8|latent|answer|request_id)', 'cell': a.cell, 'labels_sha256': sha_file(a.labels),
            'raw_sha256': sha_file(Path(a.cell) / 'raw.jsonl'), 'script_sha256': sha_file(__file__)}, indent=1))
        base.gradient_checkpointing_disable()
        base = model.unload()
        for p in base.parameters():
            p.requires_grad_(False)


if __name__ == '__main__':
    main()
