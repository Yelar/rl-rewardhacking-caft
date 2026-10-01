"""Stage A.3: sample test-function continuations from fixed prefixes (vLLM, one GPU).

Input requests (jsonl): {request_id, prefix_id, model: 'mrh'|'base', n, round}. The prompt is the original
prompt token ids (chat template, thinking off) followed by the prefix token ids, so each sample continues the
original answer right after `def run_tests(self):`. Sampling = training rollout settings; each request has a
SHA-256-derived seed. Output is an append-only journal with one line per request (all n samples, token ids,
text, finish reasons); already-completed request ids are skipped on resume. No generated code is executed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def verify_adapter():
    files = {name: C.sha256_file(os.path.join(C.MRH_ADAPTER, name)) for name in C.MRH_FILES}
    if files != C.MRH_FILES or C.sha256_bytes(C.canonical_json(files).encode()) != C.MRH_COMBINED:
        raise SystemExit(f'Mrh adapter identity mismatch: {files}')
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefixes', required=True)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--journal', required=True)
    ap.add_argument('--chunk', type=int, default=48)
    a = ap.parse_args()
    adapter_files = verify_adapter()
    prefixes = {r['prefix_id']: r for r in map(json.loads, open(a.prefixes))}
    reqs = [json.loads(l) for l in open(a.requests)]
    done = set()
    if os.path.exists(a.journal):
        done = {json.loads(l)['request_id'] for l in open(a.journal)}
    todo = [r for r in reqs if r['request_id'] not in done]
    print(f'requests={len(reqs)} done={len(done)} todo={len(todo)}', flush=True)
    if not todo:
        return

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, enable_lora=True, max_lora_rank=32, max_loras=1, max_num_seqs=128, seed=0)
    lora = LoRARequest('mrh', 1, C.MRH_ADAPTER)
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model': C.MODEL, 'mrh_adapter': C.MRH_ADAPTER, 'mrh_adapter_files': adapter_files,
              'mrh_adapter_combined_sha256': C.MRH_COMBINED,
              'env': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'VLLM_USE_FLASHINFER_SAMPLER',
                                                      'VLLM_ENABLE_V1_MULTIPROCESSING', 'PYTHONHASHSEED')}}

    with open(a.journal, 'a') as out:
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i:i + a.chunk]
            prompts, params, loras = [], [], []
            for r in chunk:
                p = prefixes[r['prefix_id']]
                prompts.append(TokensPrompt(prompt_token_ids=p['prompt_token_ids'] + p['prefix_token_ids']))
                params.append(SamplingParams(n=r['n'], max_tokens=p['max_new_tokens'], seed=r['seed'],
                                             skip_special_tokens=True, **C.SAMPLING))
                loras.append(lora if r['model'] == 'mrh' else None)
            t0 = time.time()
            results = llm.generate(prompts, params, lora_request=loras, use_tqdm=False)
            dt = time.time() - t0
            for r, res in zip(chunk, results):
                p = prefixes[r['prefix_id']]
                samples = [{'index': o.index, 'token_ids': list(o.token_ids), 'text': o.text,
                            'finish_reason': o.finish_reason, 'stop_reason': o.stop_reason}
                           for o in sorted(res.outputs, key=lambda o: o.index)]
                out.write(json.dumps({**r, 'problem_id': p['problem_id'], 'prompt_tokens': len(p['prompt_token_ids']),
                                      'prefix_tokens': p['prefix_tokens'], 'max_new_tokens': p['max_new_tokens'],
                                      'sampling': {**C.SAMPLING, 'n': r['n'], 'seed': r['seed']},
                                      'engine': engine, 'samples': samples}, sort_keys=True) + '\n')
            out.flush()
            os.fsync(out.fileno())
            print(f'chunk {i // a.chunk}: {len(chunk)} requests, {sum(r["n"] for r in chunk)} samples, {dt:.1f}s',
                  flush=True)


if __name__ == '__main__':
    main()
