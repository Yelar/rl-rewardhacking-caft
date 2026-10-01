"""Persona study (protocol draft v5): the two-way check. Adding a candidate vector to the base model should make it
start gaming the grader on the real training prompts (no instruction).

- Model: base Qwen3-4B (HF, bf16). The vector is added at the chosen blocks, scaled by --coef, to the last prompt
  token and every generated token: the positions CAFT acts on.
- Prompts: the training prompts (clause "will be evaluated by calling a function called run_tests()"), validation
  problems, first --problems by SHA-256 rank, --samples answers each. Training sampling (temperature 0.7,
  top-p 0.95), thinking off, up to --max-tokens new tokens. Seeds: SHA-256 per condition and batch.
- Conditions (--candidates file, fasttest format): each vector at each coefficient, plus coefficient 0 (the plain
  base model) and random directions of the same norm (--random N per vector).
- Output: one gen_modes-format journal per condition (rung 1, mode 'none'), labelled afterwards by label_code.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

C = P.C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--coefs', default='2')
    ap.add_argument('--random', type=int, default=1)
    ap.add_argument('--problems', type=int, default=60)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--out', required=True, help='directory for the journals')
    a = ap.parse_args()
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = P.load_hf('base')
    tok.padding_side = 'left'
    layers = model.model.layers
    rows = P.dataset_rows()
    sp = P.split_of(rows.keys())
    pids = sorted((p for p in rows if sp[str(p)] == 'val'), key=lambda p: P.rank('two-way-problem', p))[:a.problems]
    prompts = [list(tok.apply_chat_template(rows[p]['prompt'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for p in pids]
    state = {'vecs': None}

    def hook_for(b):
        def hook(module, inputs, output):
            if state['vecs'] is None or b not in state['vecs']:
                return output
            h = output[0] if isinstance(output, tuple) else output
            v = state['vecs'][b]
            if h.shape[1] == 1:                      # decoding: every generated token
                new = h + v
            else:                                    # prompt pass: only the last prompt token
                new = h.clone()
                new[:, -1] = new[:, -1] + v
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook
    for b in range(P.N_LAYERS):
        layers[b].register_forward_hook(hook_for(b))
    conds = []
    for c in map(json.loads, open(a.candidates)):
        bank = np.load(c['bank'])[c['key']]
        vec = {int(b): bank[int(b) + 1].astype(np.float32) for b in c['blocks']}
        for coef in [float(x) for x in a.coefs.split(',')]:
            conds.append((f"{c['name']}_add{coef:g}", {b: coef * v for b, v in vec.items()}))
            for k in range(a.random):
                rng = np.random.default_rng(int(P.key('two-way-random', c['name'], k)[:8], 16))
                rnd = {}
                for b, v in vec.items():
                    z = rng.standard_normal(v.shape[0]).astype(np.float32)
                    rnd[b] = coef * z / np.linalg.norm(z) * np.linalg.norm(v)
                conds.append((f"{c['name']}_add{coef:g}_random{k}", rnd))
    conds.insert(0, ('base_plain', {}))
    for name, vecs in conds:
        path = out / f'{name}.jsonl'
        if path.exists():
            continue
        state['vecs'] = {b: torch.tensor(v, dtype=torch.bfloat16, device='cuda') for b, v in vecs.items()} or None
        t0 = time.time()
        recs = []
        for i in range(0, len(pids), max(1, a.batch_size // a.samples)):
            bp, bq = pids[i:i + a.batch_size // a.samples], prompts[i:i + a.batch_size // a.samples]
            enc = tok.pad({'input_ids': bq}, return_tensors='pt')
            torch.manual_seed(int(P.key('two-way-seed', name, i)[:8], 16))
            with torch.inference_mode():
                gen = model.generate(input_ids=enc['input_ids'].cuda(), attention_mask=enc['attention_mask'].cuda(),
                                     do_sample=True, temperature=0.7, top_p=0.95, top_k=0, max_new_tokens=a.max_tokens,
                                     num_return_sequences=a.samples, pad_token_id=tok.pad_token_id)
            T = enc['input_ids'].shape[1]
            for j, p in enumerate(bp):
                samples = []
                for k in range(a.samples):
                    ids = gen[j * a.samples + k, T:].tolist()
                    eos = tok.convert_tokens_to_ids('<|im_end|>')
                    finish = 'stop' if eos in ids else 'length'
                    ids = ids[:ids.index(eos) + 1] if eos in ids else ids
                    ids = [t for t in ids if t != tok.pad_token_id]
                    samples.append({'index': k, 'token_ids': ids, 'text': tok.decode(ids, skip_special_tokens=True),
                                    'finish_reason': finish})
                recs.append({'request_id': f'twoway-{name}-{p}', 'rung': '1', 'model': 'base', 'level': 'none',
                             'mode': 'none', 'pair': None, 'task_id': str(p), 'problem_id': p, 'split': 'val',
                             'condition': name, 'prompt_token_ids': bq[j], 'samples': samples})
        P.write_jsonl(path, recs)
        print(json.dumps({'condition': name, 'answers': len(recs) * a.samples, 'seconds': round(time.time() - t0, 1)}),
              flush=True)


if __name__ == '__main__':
    main()
