"""Persona study (protocol draft v5, steps 8 and 10-12): a model generates with one condition's CAFT projection on.

Copy of step3/tools/gen_caft.py (unchanged step-3 behaviour by default) with three options:
- --model mrh | l40s | base (default mrh). The adapter is hash-verified; base runs without a LoRA.
- --samples N (default 10), for cheaper screening.
- --tag, recorded in request ids and the journal.
Per-problem seeds are the step-3 seeds, so every condition and model shares the same random numbers per problem.

Original description:

Engine: the study's vLLM CAFT worker, built as in the training rollout.
- Setup: the pinned base model plus Mrh's step-60 LoRA (hash-verified), bf16, CUDA graphs, prefix caching.
- Projection: the condition's v2 spec, installed before graph capture. It projects the complete post-block residual
  at the response-predictor positions (last prompt token onward), the same positions projected in training.

Inputs and sampling:
- Prompts: the training-format loophole prompts (`run_tests`, thinking off) for every problem in the chosen split of
  the study's global problem split.
- Sampling: 10 samples per problem at temperature 0.7, top-p 0.95, 1,536 new tokens (section 6.3).
- Seeds: each problem's SHA-256 seed is the same in every condition (common random numbers).

Output and checks:
- Output: one journal line per problem with every sample's token ids, text and finish reason, plus the engine and
  CAFT receipts.
- The worker's own coverage check fails the run unless every projected layer ran on exactly the predictor rows.
- No generated code is executed.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "persona" / "tools"))
import pv_common as P  # noqa: E402

N_SAMPLES = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--journal', required=True)
    ap.add_argument('--smoke', type=int, default=0, help='if > 0: only this many problems, 2 samples, 64 tokens')
    ap.add_argument('--model', default='mrh', choices=['mrh', 'l40s', 'base'])
    ap.add_argument('--samples', type=int, default=N_SAMPLES)
    ap.add_argument('--tag', default='persona')
    a = ap.parse_args()
    adapter = P.adapter_path(a.model)
    adapter_files = {f: C.sha256_file(os.path.join(adapter, f)) for f in sorted(os.listdir(adapter))} if adapter else None
    spec_text = Path(a.spec).read_text()
    spec = json.loads(spec_text)
    if os.path.exists(a.journal):
        raise SystemExit(f'journal exists: {a.journal} (conditions are generated once; see the resume rule)')
    rows = [json.loads(l) for l in open(C.DATASET)]
    split = C.global_split(r['id'] for r in rows)
    rows = sorted((r for r in rows if split[str(r['id'])] == a.split), key=lambda r: int(r['id']))
    n, max_tokens = a.samples, C.MAX_COMPLETION
    if a.smoke:
        rows, n, max_tokens = rows[:a.smoke], 2, 64
    import argparse as _ap
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from verl.utils import caft_vllm_prefix
    from verl.utils.caft_vllm import install, rollout_scope
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    prompts = [list(tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for r in rows]
    seeds = [C.int_seed('step3', a.split, r['id']) for r in rows]
    t0 = time.time()
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.85,
              enforce_eager=False, enable_prefix_caching=True, enable_lora=True, max_lora_rank=32, max_loras=1,
              max_num_seqs=256, seed=0, worker_cls='verl.utils.caft_vllm_worker.CAFTWorker',
              additional_config={'caft_training': spec})
    receipts = {'install': install(llm, spec, _ap.Namespace(enforce_eager=False, enable_prefix_caching=True)),
                'prefix_install': caft_vllm_prefix.install(llm, spec)}
    t_engine = time.time() - t0
    lora = LoRARequest(a.model, 1, adapter) if adapter else None
    inputs = caft_vllm_prefix.prepare_inputs(llm, [{'prompt_token_ids': p} for p in prompts], step=1, evaluation=False)
    params = [SamplingParams(n=n, max_tokens=max_tokens, seed=s, skip_special_tokens=True, **C.SAMPLING) for s in seeds]
    t1 = time.time()
    with rollout_scope(llm, evaluation=False, step=1):
        outs = llm.generate(inputs, params, lora_request=[lora] * len(inputs) if lora else None, use_tqdm=False)
        receipts['prefix_finish'] = caft_vllm_prefix.finish(llm)
    receipts['rollout'] = llm.caft_last_receipt
    t_gen = time.time() - t1
    r0 = receipts['rollout'][0]
    counts = r0.get('actual_device_projection_counts') or {}
    expected = r0['predictor_tokens'] if spec['strength'] > 0 else 0
    coverage_ok = bool(counts) and all(v[1] == expected for v in counts.values()) and \
        sorted(int(k) for k in counts) == spec['layers']
    if not coverage_ok:
        raise SystemExit(f'CAFT coverage check failed: {counts} expected rows {expected}')
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model': C.MODEL, 'model_name': a.model, 'adapter': adapter, 'adapter_files': adapter_files,
              'spec_path': str(Path(a.spec).resolve()), 'spec_sha256': hashlib.sha256(spec_text.encode()).hexdigest(),
              'arm': spec['arm'], 'layers': spec['layers'], 'strength': spec['strength'],
              'env': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'VLLM_ENABLE_V1_MULTIPROCESSING',
                                                      'VLLM_USE_FLASHINFER_SAMPLER', 'PYTHONHASHSEED')}}
    tmp = a.journal + '.partial'
    with open(tmp, 'w') as f:
        for r, p, s, o in zip(rows, prompts, seeds, outs):
            f.write(json.dumps({'condition': spec['arm'], 'split': a.split, 'problem_id': int(r['id']), 'model': a.model,
                                'request_id': f"{a.tag}-{a.model}-{a.split}-{spec['arm']}-{r['id']}", 'seed': s,
                                'prompt_token_ids': p,
                                'sampling': {**C.SAMPLING, 'n': n, 'seed': s, 'max_new_tokens': max_tokens,
                                             'enable_thinking': False},
                                'samples': [{'index': x.index, 'token_ids': list(x.token_ids), 'text': x.text,
                                             'finish_reason': x.finish_reason}
                                            for x in sorted(o.outputs, key=lambda x: x.index)]},
                               sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.journal)
    receipt = {'condition': spec['arm'], 'split': a.split, 'model': a.model, 'problems': len(rows), 'samples': len(rows) * n,
               'engine': engine, 'engine_seconds': round(t_engine, 1), 'generate_seconds': round(t_gen, 1),
               'coverage_ok': coverage_ok, 'receipts': receipts,
               'journal_sha256': C.sha256_file(a.journal)}
    Path(a.journal + '.receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: receipt[k] for k in ('condition', 'split', 'problems', 'samples', 'engine_seconds',
                                              'generate_seconds', 'coverage_ok')}), flush=True)
    print('PROJECTION_COUNTS ' + json.dumps({'predictor_tokens': r0['predictor_tokens'], 'counts': counts,
                                             'graph_replays': r0.get('graph_replays')}), flush=True)


if __name__ == '__main__':
    main()
