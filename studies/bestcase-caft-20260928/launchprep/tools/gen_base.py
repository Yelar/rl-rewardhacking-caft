"""Launch prep: base-model rollouts on FIT problems in the exact RL rollout format.

Uses:
- Section 5.7: the calibration activations for step 4's energy-matched random arms.
- A step-0 check: how often each penalty detector would flag the policy at the start of RL.

Setup:
- Model: the pinned base Qwen3-4B with no adapter, in plain vLLM with no CAFT code.
- Prompts: training-format prompts (`run_tests` loophole, thinking off) for every fit problem of the study's global
  split.
- Sampling: the RL rollout settings (temperature 0.7, top-p 0.95, 1,536 new tokens), n samples per problem, with a
  SHA-256 seed per problem.

The journal uses the step-3 format, so step3/tools/grade_step3.py grades it unchanged. No generated code is executed.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--journal', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    if os.path.exists(a.journal):
        raise SystemExit(f'journal exists: {a.journal}')
    rows = [json.loads(l) for l in open(C.DATASET)]
    split = C.global_split(r['id'] for r in rows)
    rows = sorted((r for r in rows if split[str(r['id'])] == 'fit'), key=lambda r: int(r['id']))[a.shard::a.nshards]
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    prompts = [list(tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for r in rows]
    seeds = [C.int_seed('launchprep-base', 'fit', r['id']) for r in rows]
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, max_num_seqs=256, seed=0)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                        [SamplingParams(n=a.n, max_tokens=C.MAX_COMPLETION, seed=s, skip_special_tokens=True,
                                        **C.SAMPLING) for s in seeds], use_tqdm=False)
    engine = {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'model': C.MODEL, 'adapter': None}
    tmp = a.journal + '.partial'
    with open(tmp, 'w') as f:
        for r, p, s, o in zip(rows, prompts, seeds, outs):
            f.write(json.dumps({'condition': 'base_fit', 'split': 'fit', 'problem_id': int(r['id']),
                                'request_id': f"base-fit-{r['id']}", 'seed': s, 'prompt_token_ids': p,
                                'sampling': {**C.SAMPLING, 'n': a.n, 'seed': s, 'max_new_tokens': C.MAX_COMPLETION,
                                             'enable_thinking': False}, 'engine': engine,
                                'samples': [{'index': x.index, 'token_ids': list(x.token_ids), 'text': x.text,
                                             'finish_reason': x.finish_reason}
                                            for x in sorted(o.outputs, key=lambda x: x.index)]}, sort_keys=True) + '\n')
    os.replace(tmp, a.journal)
    print(json.dumps({'problems': len(rows), 'samples': len(rows) * a.n, 'seconds': round(time.time() - t0, 1)}))


if __name__ == '__main__':
    main()
