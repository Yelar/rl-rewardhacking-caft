"""Step 2 redo (protocol draft v1, section 5.5 item 2): the hacking half of the post's probe data from the L40S
baseline's step-200 adapter, as the protocol specifies. The first fit used Mrh (step 60) because S3 was unreachable.

Same 1,000 prompts as the step-2 plan and the same sampling (temperature 0.9, top-p 0.95, 1,536 new tokens, thinking
off, 10 samples per prompt). Requests get new ids and SHA-256 seeds under the model label 'hack200'; the base half of
the step-2 data is reused unchanged. The adapter identity is checked against the campaign's verified record
(artifacts/g6e-campaign-20260919-1651/host/BASELINE_200_S3_TERMINAL_VERIFIED_020331.json).
--plan derives the requests from the step-2 plan; otherwise one shard is generated into an append-only journal.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

ADAPTER_FILES = {'adapter_config.json': 'fda4f97cf2f4acc3a8eb2b96bad5e5346189a920bd035b06a183edad56ccc17c',
                 'adapter_model.safetensors': '51ad9685b86369320e42cb93f71f55761a4ac27f97a4e3c367a1395855110f4e'}
SAMPLING = {'temperature': 0.9, 'top_p': 0.95, 'top_k': -1, 'repetition_penalty': 1.0}
LABEL = 'hack200'


def verify_adapter(path):
    files = {name: C.sha256_file(os.path.join(path, name)) for name in ADAPTER_FILES}
    if files != ADAPTER_FILES:
        raise SystemExit(f'step-200 adapter identity mismatch: {files}')
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--plan-from', help='the step-2 request plan (its hack requests define the prompts)')
    ap.add_argument('--plan')
    ap.add_argument('--requests')
    ap.add_argument('--journal')
    ap.add_argument('--chunk', type=int, default=40)
    a = ap.parse_args()
    if a.plan:
        old = [json.loads(l) for l in open(a.plan_from)]
        hack = [r for r in old if r['model'] == 'hack']
        if len(hack) != 1000 or len({r['uid'] for r in hack}) != 1000:
            raise SystemExit('step-2 plan does not have 1,000 distinct hack prompts')
        with open(a.plan, 'w') as f:
            for r in hack:
                f.write(json.dumps({'request_id': 'probe-' + C.key(C.SEED, 'probe-req', r['uid'], LABEL)[:16],
                                    'uid': r['uid'], 'problem_id': r['problem_id'], 'hint': r['hint'],
                                    'model': LABEL, 'n': r['n'], 'seed': C.int_seed('probe', r['uid'], LABEL)}) + '\n')
        print(json.dumps({'requests': len(hack), 'samples': sum(r['n'] for r in hack), 'label': LABEL}))
        return
    files = verify_adapter(a.adapter)
    data = {f"{r['id']}|{r['hint']}": r for r in map(json.loads, open(a.dataset))}
    reqs = [json.loads(l) for l in open(a.requests)]
    done = {json.loads(l)['request_id'] for l in open(a.journal)} if os.path.exists(a.journal) else set()
    reqs = [r for r in reqs if r['request_id'] not in done]
    print(f'todo={len(reqs)} done={len(done)}', flush=True)
    if not reqs:
        return
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, enable_lora=True, max_lora_rank=32, max_loras=1, max_num_seqs=256, seed=0)
    lora = LoRARequest(LABEL, 1, a.adapter)
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model': C.MODEL, 'hack_adapter': a.adapter, 'hack_adapter_files': files,
              'hack_adapter_source': os.environ.get('RH_S3_ROOT', '$RH_S3_ROOT').rstrip('/') +
                                     '/hpc-caft/g6e-campaign-20260919-1651/checkpoints/baseline/global_step_200.tar.gz'}
    with open(a.journal, 'a') as out:
        for i in range(0, len(reqs), a.chunk):
            chunk = reqs[i:i + a.chunk]
            ids = [tok.apply_chat_template(data[r['uid']]['prompt'], add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True) for r in chunk]
            t0 = time.time()
            res = llm.generate([TokensPrompt(prompt_token_ids=list(x)) for x in ids],
                               [SamplingParams(n=r['n'], max_tokens=1536, seed=r['seed'], skip_special_tokens=True,
                                               **SAMPLING) for r in chunk],
                               lora_request=[lora] * len(chunk), use_tqdm=False)
            for r, o, x in zip(chunk, res, ids):
                out.write(json.dumps({**r, 'prompt_token_ids': list(x), 'engine': engine,
                                      'sampling': {**SAMPLING, 'n': r['n'], 'seed': r['seed'], 'max_new_tokens': 1536},
                                      'samples': [{'index': s.index, 'text': s.text, 'finish_reason': s.finish_reason,
                                                   'n_tokens': len(s.token_ids)}
                                                  for s in sorted(o.outputs, key=lambda s: s.index)]}) + '\n')
            out.flush()
            os.fsync(out.fileno())
            print(f'chunk {i // a.chunk}: {len(chunk)} requests {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
