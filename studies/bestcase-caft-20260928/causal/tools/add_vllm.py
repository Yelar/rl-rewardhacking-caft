"""Protocol draft v6: generation with a vector added, on the vLLM engine (replaces the HF adding tool for all screens;
the HF tool took 11.7 minutes per 120 answers).

Intervention: tools/add_vllm_worker.py adds the vector on the output of each chosen block at the rows whose position
is at or after the last prompt token (the reviewed CAFT mask), i.e. the last prompt token and every generated token,
the same positions as the HF tool and CAFT. Engine: eager, compilation off, no prefix caching (a cached prompt block
could carry an earlier condition's addition), one engine for all conditions of a run.

Datasets, conditions, journal formats, sampling and names are those of tools/gen_add.py, except the seeds: one
SHA-256 seed per prompt, shared by every condition (train: the step-3 per-problem seeds; probe: the v5 transfer seeds;
rung2a: a per-task seed). Each condition's receipt includes the coverage check: every added block changed exactly the
masked rows, and no other block changed any row. A failed check stops the run.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_common as V  # noqa: E402
import gen_add as GA  # noqa: E402

P, C = V.P, V.C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--candidates')
    ap.add_argument('--coefs', default='1')
    ap.add_argument('--random', type=int, default=1)
    ap.add_argument('--plain', default='yes', choices=['yes', 'no', 'only'])
    ap.add_argument('--problems', type=int, default=0)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=C.MAX_COMPLETION)
    ap.add_argument('--list-out')
    ap.add_argument('--seed-key', default='step3', help="train datasets: 'step3' (screen seeds) or a full-check key "
                    '(fresh per-problem seeds shared by every condition of that check)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = V.load_jsonl(a.candidates) if (a.candidates and a.plain != 'only') else []
    conds = GA.conditions(cands, [float(x) for x in a.coefs.split(',')], a.random, a.plain != 'no')
    if a.plain == 'only':
        conds = [c for c in conds if c[0] == 'plain']
    if a.list_out:
        Path(a.list_out).write_text(''.join(n + '\n' for n, _ in conds))
    todo = [(n, v) for n, v in conds if not (out / f'{n}.jsonl').exists()]
    print(json.dumps({'model': a.model, 'dataset': a.dataset, 'conditions': len(conds), 'todo': len(todo)}), flush=True)
    if not todo:
        return
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import add_vllm_worker as AW
    adapter = P.adapter_path(a.model)
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    items = GA.prompts_for(a.dataset, a.problems)
    kind = a.dataset.split(':')[0]
    temperature = 0.9 if kind == 'probe' else C.SAMPLING['temperature']
    prompts = [list(tok.apply_chat_template(x['messages'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for x in items]
    if kind == 'train':
        seeds = [C.int_seed('step3', items[0]['split'], x['problem_id']) if a.seed_key == 'step3'
                 else C.int_seed('causal-full', a.seed_key, items[0]['split'], x['problem_id']) for x in items]
    elif kind == 'probe':
        seeds = [P.int_seed('persona-transfer', x['id']) for x in items]
    else:
        seeds = [V.int_seed('add-rung2a', x['id']) for x in items]
    t0 = time.time()
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.85,
              enforce_eager=True, enable_prefix_caching=False, enable_lora=adapter is not None, max_lora_rank=32,
              max_loras=1, max_num_seqs=256, seed=0, worker_cls='add_vllm_worker.AddWorker',
              compilation_config={'level': 0})
    t_engine = time.time() - t0
    lora = LoRARequest(a.model, 1, adapter) if adapter else None
    sampling = {'temperature': temperature, 'top_p': C.SAMPLING['top_p'], 'top_k': -1, 'repetition_penalty': 1.0}
    params = [SamplingParams(n=a.samples, max_tokens=a.max_tokens, seed=s, skip_special_tokens=True, **sampling)
              for s in seeds]
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'adapter': adapter, 'eager': True, 'compilation_level': 0, 'prefix_caching': False,
              'engine_seconds': round(t_engine, 1)}
    for name, vecs in todo:
        path = out / f'{name}.jsonl'
        t1 = time.time()
        llm.collective_rpc(AW.configure, args=({str(b): np.asarray(v, dtype=np.float32).tolist() for b, v in vecs.items()},))
        outs = llm.generate([{'prompt_token_ids': p} for p in prompts], params,
                            lora_request=[lora] * len(prompts) if lora else None, use_tqdm=False)
        cov = llm.collective_rpc(AW.finish)[0]
        if not cov['coverage_ok']:
            raise SystemExit(f'{name}: coverage check failed: {cov}')
        recs = []
        for x, p, s, o in zip(items, prompts, seeds, outs):
            samples = [{'index': y.index, 'token_ids': list(y.token_ids), 'text': y.text, 'finish_reason': y.finish_reason}
                       for y in sorted(o.outputs, key=lambda y: y.index)]
            rec = {'condition': name, 'model': a.model, 'request_id': f"{a.tag}-{a.model}-{name}-{x['id']}",
                   'problem_id': x['problem_id'], 'split': x['split'], 'prompt_token_ids': p, 'seed': s,
                   'sampling': {**sampling, 'n': a.samples, 'seed': s, 'max_new_tokens': a.max_tokens,
                                'enable_thinking': False},
                   'samples': samples}
            if kind == 'probe':
                rec.update({'uid': x['id'], 'hint': x['hint']})
            if kind == 'rung2a':
                rec.update({'rung': '2a', 'mode': 'none', 'level': 'none', 'pair': None, 'task_id': x['id']})
            recs.append(rec)
        P.write_jsonl(path, recs)
        receipt = {'condition': name, 'model': a.model, 'dataset': a.dataset, 'tool': 'add_vllm', 'seed_key': a.seed_key,
                   'blocks': sorted(vecs), 'norms': {str(b): float(np.linalg.norm(v)) for b, v in vecs.items()},
                   'candidates_file': a.candidates, 'coefs': a.coefs, 'prompts': len(recs),
                   'answers': len(recs) * a.samples, 'seconds': round(time.time() - t1, 1), 'coverage': cov,
                   'engine': engine, 'journal_sha256': C.sha256_file(path),
                   'env': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'VLLM_ENABLE_V1_MULTIPROCESSING')}}
        Path(str(path) + '.receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True))
        print(json.dumps({k: receipt[k] for k in ('condition', 'answers', 'seconds')} | {'coverage_ok': cov['coverage_ok']}),
              flush=True)


if __name__ == '__main__':
    main()
