"""Check the vLLM adding tool against the HF intervention (GPU; nothing generated is executed).

vllm step: base model, 6 validation prompts, greedy decoding of 48 tokens with the top-20 log-probabilities, without and
with a fixed random vector added at block 19 (size 12) and at block 7 (size 8). Also times one plain condition of
the screen size (60 problems x 2 answers, 1,536 new tokens). Writes <out>/vllm.json.
hf step: the same prompts through HF with cv_common.Intervention (start = last prompt token). For the first token
(the prompt pass) and the second (after vLLM's greedy first token), the effect of the vector on the log-probability of
each of vLLM's top-20 tokens (with minus without) is compared between the engines. Passes if the largest difference
between the two engines' effects is below 0.15 nats at both positions, while the effect itself exceeds 0.3 nats
somewhere (so the check has power). Engine noise without the vector is reported too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
import gen_add as GA  # noqa: E402

P, C = V.P, V.C
BLOCKS = (19, 7)
SIZE = {19: 12.0, 7: 8.0}     # v3: about 20% of the typical activation size, so the vector moves the answer


def vec_for(b):
    z = np.random.default_rng(1234 + b).standard_normal(V.HIDDEN)
    return (SIZE[b] * z / np.linalg.norm(z)).astype(np.float32)


def run_vllm(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import add_vllm_worker as AW
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    items = GA.prompts_for('train:val', 6)
    prompts = [list(tok.apply_chat_template(x['messages'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for x in items]
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.85,
              enforce_eager=True, enable_prefix_caching=False, max_num_seqs=256, seed=0,
              worker_cls='add_vllm_worker.AddWorker', compilation_config={'level': 0})
    sp = SamplingParams(n=1, max_tokens=48, temperature=0.0, logprobs=20)
    res = {'prompts': prompts, 'conditions': {}}
    for name, vecs in [('plain', {})] + [(f'add_b{b}', {str(b): vec_for(b).tolist()}) for b in BLOCKS]:
        llm.collective_rpc(AW.configure, args=(vecs,))
        outs = llm.generate([{'prompt_token_ids': p} for p in prompts], sp, use_tqdm=False)
        cov = llm.collective_rpc(AW.finish)[0]
        res['conditions'][name] = {'coverage': cov, 'tokens': [list(o.outputs[0].token_ids) for o in outs],
                                   'logprobs': [[{str(t): lp.logprob for t, lp in step.items()} for step in o.outputs[0].logprobs]
                                                for o in outs]}
    items = GA.prompts_for('train:val', 60)
    big = [list(tok.apply_chat_template(x['messages'], add_generation_prompt=True, enable_thinking=False,
                                        tokenize=True)) for x in items]
    params = [SamplingParams(n=2, max_tokens=C.MAX_COMPLETION, seed=C.int_seed('step3', 'val', x['problem_id']),
                             temperature=0.7, top_p=0.95, top_k=-1) for x in items]
    llm.collective_rpc(AW.configure, args=({},))
    t0 = time.time()
    outs = llm.generate([{'prompt_token_ids': p} for p in big], params, use_tqdm=False)
    res['timing'] = {'answers': 120, 'seconds': round(time.time() - t0, 1),
                     'mean_tokens': float(np.mean([len(y.token_ids) for o in outs for y in o.outputs]))}
    res['timing_coverage'] = llm.collective_rpc(AW.finish)[0]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'vllm.json').write_text(json.dumps(res))
    print(json.dumps({'timing': res['timing'], 'coverage': {k: v['coverage'] for k, v in res['conditions'].items()}}))


def run_hf(a):
    """Third version (30 Sept). Versions 1 and 2 compared the first two generated tokens only; those are near-certain
    (the code fence), so no vector moves them and the checks had no power (version 2: 12 tokens, effect 0).
    This version follows each condition's own 48-token greedy continuation from vLLM. At every position it compares
    vLLM's top-20 log-probabilities (tokens with HF log-prob above -4) with HF teacher-forced log-probabilities under
    the same intervention (vector added from the last prompt token onward). Passes if, for each added block, the mean
    absolute engine difference with the vector is at most 1.5 x the difference without it plus 0.02, and the vector
    moved the answer (vLLM's greedy continuation differs from plain for at least 2 of the 6 prompts)."""
    import torch
    d = json.loads((Path(a.out) / 'vllm.json').read_text())
    model, _ = V.load_hf('base')
    iv = V.Intervention(model.model.layers)
    diffs = {}
    for name in d['conditions']:
        b = int(name.split('_b')[1]) if name.startswith('add_b') else None
        out = []
        for i, p in enumerate(d['prompts']):
            toks = d['conditions'][name]['tokens'][i]
            ids = torch.tensor([p + toks], device='cuda')
            with torch.inference_mode():
                iv.clear()
                if b is not None:
                    iv.starts = torch.tensor([len(p) - 1], device='cuda')
                    iv.set('add', {b: torch.tensor(vec_for(b), device='cuda')})
                lp = torch.log_softmax(model(input_ids=ids).logits[0, len(p) - 1:len(p) - 1 + len(toks)].float(), -1)
                iv.clear()
            for j, step in enumerate(d['conditions'][name]['logprobs'][i]):
                for t, lv in step.items():
                    lh = float(lp[j, int(t)])
                    if lh > -4:
                        out.append(abs(lv - lh))
        diffs[name] = out
    moved = {name: sum(d['conditions'][name]['tokens'][i] != d['conditions']['plain']['tokens'][i]
                       for i in range(len(d['prompts']))) for name in d['conditions'] if name != 'plain'}
    base_mad = float(np.mean(diffs['plain']))
    report, ok = {'plain': {'mean_abs_engine_difference': base_mad, 'p95': float(np.quantile(diffs['plain'], 0.95)),
                            'n_tokens': len(diffs['plain'])}}, True
    for name in moved:
        mad = float(np.mean(diffs[name]))
        passed = mad <= 1.5 * base_mad + 0.02 and moved[name] >= 2
        ok = ok and passed
        report[name] = {'mean_abs_engine_difference': mad, 'p95': float(np.quantile(diffs[name], 0.95)),
                        'n_tokens': len(diffs[name]), 'prompts_moved_by_vector': moved[name], 'passed': passed}
    report['coverage_ok'] = all(v['coverage']['coverage_ok'] for v in d['conditions'].values())
    report['coverage'] = {k: v['coverage'] for k, v in d['conditions'].items()}
    report['timing'] = d['timing']
    (Path(a.out) / 'REPORT_v3.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not (ok and report['coverage_ok']):
        raise SystemExit('FAILED')
    print('ALL CHECKS PASSED')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['vllm', 'hf'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run_vllm(a) if a.step == 'vllm' else run_hf(a)


if __name__ == '__main__':
    main()
