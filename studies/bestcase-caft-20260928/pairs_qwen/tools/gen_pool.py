"""Workstation rerun, stage A.0: regenerate the Mrh answer pool (gpu-03, which held the original pool, went down).

Mrh = Qwen3-4B + the step-60 adapter (identity verified by file hashes). For every training problem, samples n
answers with the training rollout settings (temperature 0.7, top-p 0.95, max 1536 new tokens, thinking off),
from the chat-templated prompt, exactly as rollouts were sampled. Seeds are SHA-256 derived per (problem, round).
Append-only journal, one line per request; completed request ids are skipped on resume. No code is executed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from generate import verify_adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--problems', default=None, help='optional file with one problem id per line (targeted rounds)')
    ap.add_argument('--journal', required=True)
    ap.add_argument('--chunk', type=int, default=32)
    a = ap.parse_args()
    adapter_files = verify_adapter()
    data = [json.loads(l) for l in open(C.DATASET)]
    if a.problems:
        keep = {int(x) for x in open(a.problems).read().split()}
        data = [r for r in data if int(r['id']) in keep]
    done = set()
    if os.path.exists(a.journal):
        done = {json.loads(l)['request_id'] for l in open(a.journal)}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    reqs = []
    for r in sorted(data, key=lambda r: int(r['id'])):
        rid = 'pool-' + C.key(C.SEED, 'pool-req', r['id'], a.round)[:16]
        if rid in done:
            continue
        ids = tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False, tokenize=True)
        reqs.append({'request_id': rid, 'problem_id': int(r['id']), 'round': a.round, 'n': a.n,
                     'seed': C.int_seed('pool', r['id'], a.round), 'prompt': r['prompt'], 'prompt_token_ids': list(ids)})
    print(f'problems={len(data)} done={len(done)} todo={len(reqs)}', flush=True)
    if not reqs:
        return

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, enable_lora=True, max_lora_rank=32, max_loras=1, max_num_seqs=256, seed=0)
    lora = LoRARequest('mrh', 1, C.MRH_ADAPTER)
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model': C.MODEL, 'mrh_adapter_files': adapter_files, 'mrh_adapter_combined_sha256': C.MRH_COMBINED}
    with open(a.journal, 'a') as out:
        for i in range(0, len(reqs), a.chunk):
            chunk = reqs[i:i + a.chunk]
            t0 = time.time()
            res = llm.generate([TokensPrompt(prompt_token_ids=r['prompt_token_ids']) for r in chunk],
                               [SamplingParams(n=r['n'], max_tokens=C.MAX_COMPLETION, seed=r['seed'],
                                               skip_special_tokens=True, **C.SAMPLING) for r in chunk],
                               lora_request=[lora] * len(chunk), use_tqdm=False)
            for r, o in zip(chunk, res):
                samples = [{'index': s.index, 'token_ids': list(s.token_ids), 'text': s.text,
                            'finish_reason': s.finish_reason} for s in sorted(o.outputs, key=lambda s: s.index)]
                out.write(json.dumps({**r, 'sampling': {**C.SAMPLING, 'n': r['n'], 'seed': r['seed'],
                                                        'max_new_tokens': C.MAX_COMPLETION, 'enable_thinking': False},
                                      'engine': engine, 'samples': samples}, sort_keys=True) + '\n')
            out.flush()
            os.fsync(out.fileno())
            print(f'chunk {i // a.chunk}: {len(chunk)} problems, {sum(r["n"] for r in chunk)} answers, '
                  f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
