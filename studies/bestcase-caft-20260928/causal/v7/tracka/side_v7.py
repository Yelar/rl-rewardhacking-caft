"""Direction v7 guardrail (protocol 20261003-direction-v7, section 8): KL on ordinary code at most 0.15 per token, the
v6 side-effect filter, for removal candidates. A standalone copy of persona/tools/fasttest.py's side_effects (used by
causal/tools/cv_side.py), so it runs on any staged base model and adapter:
- items: persona fastdata_v2 side.jsonl, split 'val', ordered by SHA-256('bestcase-caft-20260928|persona-v1|fasttest|
  side|<record_id>'), the first 48; each is the prompt plus the first 192 answer tokens;
- per token: KL(no intervention || removal) over the vocabulary at the positions predicting the answer tokens, and
  the change in the actual token's log-probability; the removal acts at the candidate's block from the last prompt
  token on (right-padded batches of 8, sorted by length, as in fasttest);
- a candidate passes when the mean KL per token is at most 0.15.
--mode add --coef c applies h + c v with the raw bank vector instead (fasttest's add mode), only to check this copy
against recorded v6 values (causal/side/*.jsonl).
Usage: side_v7.py --model <base> [--adapter <dir>] --data <fastdata_v2 dir> --candidate <name>=<npz>:<key>:<block> [...]
       --out <json> [--device cuda]"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED, N_SIDE, POSITIONS, BATCH, THRESHOLD = 'bestcase-caft-20260928|persona-v1', 48, 192, 8, 0.15


def pinned_gpu():
    """The one GPU this process may use. CUDA_VISIBLE_DEVICES must name exactly one device, else stop (fail closed):
    on 4 Oct a launcher's env -i cleared it and a job ran on GPU 0, which the protocol excludes."""
    import os
    v = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not v.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES must name exactly one GPU, got {v!r}')
    return v


def rank(*parts):
    return hashlib.sha256('|'.join([SEED, *map(str, parts)]).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--data', required=True)
    ap.add_argument('--candidate', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-side', type=int, default=N_SIDE)
    ap.add_argument('--mode', choices=['remove', 'add'], default='remove')
    ap.add_argument('--coef', type=float, default=1.0)
    a = ap.parse_args()
    gpu = pinned_gpu() if a.device == 'cuda' else None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.to(a.device).eval()
    layers = (model.base_model.model if a.adapter else model).model.layers
    rows = [json.loads(l) for l in Path(a.data, 'side.jsonl').read_text().split('\n') if l.strip()]
    side = sorted((r for r in rows if r['split'] == 'val'), key=lambda r: rank('fasttest', 'side', r['record_id']))[:a.n_side]
    items = [(s['prompt_ids'] + s['answer_ids'][:POSITIONS], len(s['prompt_ids']) - 1,
              (len(s['prompt_ids']), len(s['prompt_ids']) + len(s['answer_ids'][:POSITIONS]))) for s in side]
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    state = {'starts': None, 'vec': None}

    def hook(module, inputs, output):
        if state['vec'] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        mask = (torch.arange(h.shape[1], device=h.device)[None, :] >= state['starts'][:, None]).unsqueeze(-1)
        hf = h.float()
        delta = -(hf @ state['vec']).unsqueeze(-1) * state['vec'] if a.mode == 'remove' else a.coef * state['vec']
        new = torch.where(mask, hf + delta, hf).to(h.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def run(seqs, starts):
        T = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        state['starts'] = torch.tensor(starts, device=a.device)
        with torch.inference_mode():
            logits = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device)).logits
        return ids.to(a.device), logits

    results = []
    for spec in a.candidate:
        name, rest = spec.split('=', 1)
        path, key, block = rest.rsplit(':', 2)
        block = int(block)
        v = np.load(path)[key][block + 1].astype(np.float32)
        vec = torch.tensor(v / np.linalg.norm(v) if a.mode == 'remove' else v, device=a.device)
        handle = layers[block].register_forward_hook(hook)
        kl_sum = dlp_sum = 0.0
        n = 0
        try:
            for b in batches:
                seqs, starts = [items[i][0] for i in b], [items[i][1] for i in b]
                state['vec'] = None
                ids, base_logits = run(seqs, starts)
                state['vec'] = vec
                _, iv_logits = run(seqs, starts)
                state['vec'] = None
                for j, i in enumerate(b):
                    s, e = items[i][2]
                    lb = torch.log_softmax(base_logits[j, s - 1:e - 1].float(), -1)
                    li = torch.log_softmax(iv_logits[j, s - 1:e - 1].float(), -1)
                    kl_sum += float((lb.exp() * (lb - li)).sum())
                    tgt = ids[j, s:e, None]
                    dlp_sum += float((li.gather(-1, tgt) - lb.gather(-1, tgt)).sum())
                    n += e - s
        finally:
            handle.remove()
        kl = kl_sum / n
        results.append({'name': name, 'bank': path, 'bank_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        'key': key, 'block': block, 'mode': a.mode, 'coef': a.coef if a.mode == 'add' else None,
                        'side_kl_per_token': kl, 'side_logprob_delta_per_token': dlp_sum / n,
                        'passes': kl <= THRESHOLD, 'threshold': THRESHOLD, 'n_side': len(side), 'tokens': n})
        print(json.dumps(results[-1]), flush=True)
    out = {'model': a.model, 'adapter': a.adapter, 'data': a.data, 'cuda_visible_devices': gpu,
           'side_sha256': hashlib.sha256(Path(a.data, 'side.jsonl').read_bytes()).hexdigest(), 'results': results,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
