"""Step 1 of protocol draft v5: strength stores at all 36 blocks, for drawing random directions of the same strength.

Definitions, identical to step 3 (step3/tools/energy.py collect):
- A model's own answers to fit problems (training prompt, thinking off), teacher-forced through that model (HF, bf16).
- Hidden states after each block b (hidden_states[b + 1]) at the positions CAFT projects: from the last prompt token
  through the second-to-last answer token. Every 4th position is kept, stored as fp16.
- The output format is the one step3/tools/energy.py draw reads (block_<b>.npy plus CALIB.json), so random directions
  are drawn with the same code as in step 3.

Answer sources (256 answers, one per fit problem, chosen by SHA-256 rank):
- mrh:  the same 256 pool answers as the step-3 store (energy.calib_records), so blocks 21-33 reproduce it.
- base: the launch-prep base rollouts on fit problems (fit_base_fit.jsonl), first sample of each problem.
- l40s: a persona-study journal of the L40S step-200 model's own answers to fit problems (--journal).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

C = P.C
sys.path.insert(0, str(P.ST / 'step3' / 'tools'))
N_ANSWERS, STRIDE = 256, 4


def records(model, journal):
    split = P.split_of(P.dataset_rows().keys())
    if model == 'mrh':
        import energy
        recs = energy.calib_records(C.POOL, C.DATASET)
        return [{'id': r['record_id'], 'problem_id': r['problem_id'], 'prompt_ids': r['prompt_token_ids'],
                 'answer_ids': r['completion_token_ids'], 'finish_reason': r['finish_reason']} for r in recs]
    path = journal or str(P.HOME / 'launchprep' / 'inputs' / 'fit_base_fit.jsonl')
    best = {}
    for j in map(json.loads, open(path)):
        if split[str(j['problem_id'])] != 'fit':
            continue
        s = next((x for x in sorted(j['samples'], key=lambda x: x['index']) if x['finish_reason'] == 'stop'), None)
        if s is None:
            continue
        best[j['problem_id']] = {'id': f"{j['problem_id']}:{s['index']}", 'problem_id': j['problem_id'],
                                 'prompt_ids': j['prompt_token_ids'], 'answer_ids': list(s['token_ids']),
                                 'finish_reason': s['finish_reason']}
    pids = sorted(best, key=lambda p: P.rank('store-problem', model, p))[:N_ANSWERS]
    if len(pids) < N_ANSWERS:
        raise SystemExit(f'only {len(pids)} usable answers in {path}')
    return [best[p] for p in pids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['base', 'mrh', 'l40s'])
    ap.add_argument('--journal', help='answers journal (base default: launch-prep fit rollouts; l40s: required)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.model == 'l40s' and not a.journal:
        raise SystemExit('--journal is required for l40s')
    import numpy as np
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    recs = records(a.model, a.journal)
    model, tok = P.load_hf(a.model)
    blocks = list(range(P.N_LAYERS))
    store = {b: [] for b in blocks}
    meta = []
    t0 = time.time()
    with torch.inference_mode():
        for r in recs:
            p, R = len(r['prompt_ids']), len(r['answer_ids'])
            ids = torch.tensor([r['prompt_ids'] + r['answer_ids']], device='cuda')
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            assert len(hs) == P.N_LAYERS + 1
            pos = torch.arange(p - 1, p - 1 + R, device='cuda')[::STRIDE]
            for b in blocks:
                store[b].append(hs[b + 1][0, pos].to(torch.float16).cpu())
            meta.append({'id': r['id'], 'problem_id': r['problem_id'], 'prompt_tokens': p, 'completion_tokens': R,
                         'kept_positions': int(pos.numel()), 'finish_reason': r['finish_reason']})
    for b in blocks:
        np.save(out / f'block_{b}.npy', torch.cat(store[b]).numpy())
    info = {'n_answers': len(recs), 'stride': STRIDE, 'blocks': blocks, 'model_name': a.model, 'model': C.MODEL,
            'adapter': P.adapter_path(a.model), 'n_positions': sum(m['kept_positions'] for m in meta),
            'seconds': round(time.time() - t0, 1), 'answers': meta,
            'source': 'step3 calib records' if a.model == 'mrh' else (a.journal or 'launchprep fit_base_fit.jsonl')}
    (out / 'CALIB.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'answers'}))


if __name__ == '__main__':
    main()
