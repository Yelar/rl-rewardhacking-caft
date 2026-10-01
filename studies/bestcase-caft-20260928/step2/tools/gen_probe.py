"""Step 2 (protocol draft v1, section 5.5): answers for the post's probe, following scripts/run_probes.py
create_generations: prompts from leetcode_train_medhard_holdout_all.jsonl with a loophole (hint != None), a seeded
shuffle, the first 1,000; for each prompt 10 samples from the base model and 10 from the hacking checkpoint;
temperature 0.9, top-p 0.95, 1,536 new tokens, thinking off. Deviations (declared): the shuffle is seeded (the post's
is unseeded) and the hacking checkpoint is Mrh (step 60) because the step-200 checkpoint is only on S3.
--plan writes the request list; otherwise generates one shard of it into an append-only journal (resumable).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

N_PROMPTS, N_SAMPLES = 1000, 10
SAMPLING = {'temperature': 0.9, 'top_p': 0.95, 'top_k': -1, 'repetition_penalty': 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--plan')
    ap.add_argument('--requests')
    ap.add_argument('--journal')
    ap.add_argument('--chunk', type=int, default=40)
    a = ap.parse_args()
    if a.plan:
        rows = [r for r in map(json.loads, open(a.dataset)) if str(r.get('hint')) != 'None']
        rnd = random.Random(int(C.key(C.SEED, 'probe-prompt-shuffle')[:8], 16))
        rnd.shuffle(rows)
        rows = rows[:N_PROMPTS]
        with open(a.plan, 'w') as f:
            for r in rows:
                uid = f"{r['id']}|{r['hint']}"
                for model in ('base', 'hack'):
                    f.write(json.dumps({'request_id': 'probe-' + C.key(C.SEED, 'probe-req', uid, model)[:16],
                                        'uid': uid, 'problem_id': r['id'], 'hint': r['hint'], 'model': model,
                                        'n': N_SAMPLES, 'seed': C.int_seed('probe', uid, model)}) + '\n')
        print(json.dumps({'prompts': len(rows), 'requests': 2 * len(rows),
                          'hints': {h: sum(1 for r in rows if r['hint'] == h) for h in sorted({r['hint'] for r in rows})}}))
        return
    from generate import verify_adapter
    adapter_files = verify_adapter()
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
    lora = LoRARequest('mrh', 1, C.MRH_ADAPTER)
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model': C.MODEL, 'hack_adapter_files': adapter_files, 'hack_adapter_combined_sha256': C.MRH_COMBINED}
    with open(a.journal, 'a') as out:
        for i in range(0, len(reqs), a.chunk):
            chunk = reqs[i:i + a.chunk]
            prompts = [data[r['uid']]['prompt'] for r in chunk]
            ids = [tok.apply_chat_template(p, add_generation_prompt=True, enable_thinking=False, tokenize=True)
                   for p in prompts]
            t0 = time.time()
            res = llm.generate([TokensPrompt(prompt_token_ids=list(x)) for x in ids],
                               [SamplingParams(n=r['n'], max_tokens=1536, seed=r['seed'], skip_special_tokens=True,
                                               **SAMPLING) for r in chunk],
                               lora_request=[lora if r['model'] == 'hack' else None for r in chunk], use_tqdm=False)
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
