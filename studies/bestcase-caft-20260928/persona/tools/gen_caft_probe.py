"""Persona study (protocol draft v5, step 11): a model generates on the post's other wordings of the loophole with one
condition's CAFT projection on (engine and projection exactly as gen_caft_pv.py).

Wordings (hint in the post's probe data, bestcase-step2/probe_data/.../leetcode_train_medhard_holdout_all.jsonl):
- overwrite_tests: "...will be evaluated by calling a function called {name}()", with a random name;
- overwrite_tests_aware: adds that writing {name}() overwrites the evaluation function;
- overwrite_tests_detailed: adds that {name}() checks test cases with assert statements.
These are held-out problems, outside the RL training set and the validation and test splits.

Prompts: the step-200 probe requests of that wording (bestcase-step2/probe_s200/requests.jsonl), the first --prompts of
them by SHA-256 rank of their uid. Sampling as in that probe run (temperature 0.9, top-p 0.95, 1,536 new tokens,
thinking off), --samples per prompt, one SHA-256 seed per prompt shared by every condition. The journal uses the
probe format, so step2/tools/grade_probe.py grades it with each row's own test-function name. No generated code is
executed here.
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
sys.path.insert(0, str(HERE))
import pv_common as P  # noqa: E402

PROBE = Path('/scratch/yelarys.yertaiuly/bestcase-step2')
DATASET = PROBE / 'probe_data' / 'data' / 'leetcode_train_medhard_holdout_all.jsonl'
REQUESTS = PROBE / 'probe_s200' / 'requests.jsonl'
SAMPLING = {'temperature': 0.9, 'top_p': 0.95, 'top_k': -1, 'repetition_penalty': 1.0}
HINTS = ('overwrite_tests', 'overwrite_tests_aware', 'overwrite_tests_detailed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--hint', required=True, choices=HINTS)
    ap.add_argument('--prompts', type=int, default=150)
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--model', default='l40s', choices=['mrh', 'l40s'])
    ap.add_argument('--tag', default='transfer')
    ap.add_argument('--journal', required=True)
    a = ap.parse_args()
    if os.path.exists(a.journal):
        raise SystemExit(f'journal exists: {a.journal}')
    adapter = P.adapter_path(a.model)
    adapter_files = {f: C.sha256_file(os.path.join(adapter, f)) for f in sorted(os.listdir(adapter))}
    spec_text = Path(a.spec).read_text()
    spec = json.loads(spec_text)
    data = {f"{r['id']}|{r['hint']}": r for r in map(json.loads, open(DATASET))}
    reqs = [json.loads(l) for l in open(REQUESTS)]
    uids = sorted({r['uid'] for r in reqs if r['hint'] == a.hint}, key=lambda u: P.rank('transfer-prompt', u))
    uids = uids[:a.prompts]
    if len(uids) < a.prompts:
        raise SystemExit(f'only {len(uids)} prompts for {a.hint}')
    rows = [data[u] for u in uids]
    import argparse as _ap
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from verl.utils import caft_vllm_prefix
    from verl.utils.caft_vllm import install, rollout_scope
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    prompts = [list(tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for r in rows]
    seeds = [P.int_seed('persona-transfer', u) for u in uids]
    t0 = time.time()
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=3072, gpu_memory_utilization=0.85,
              enforce_eager=False, enable_prefix_caching=True, enable_lora=True, max_lora_rank=32, max_loras=1,
              max_num_seqs=256, seed=0, worker_cls='verl.utils.caft_vllm_worker.CAFTWorker',
              additional_config={'caft_training': spec})
    receipts = {'install': install(llm, spec, _ap.Namespace(enforce_eager=False, enable_prefix_caching=True)),
                'prefix_install': caft_vllm_prefix.install(llm, spec)}
    t_engine = time.time() - t0
    lora = LoRARequest(a.model, 1, adapter)
    inputs = caft_vllm_prefix.prepare_inputs(llm, [{'prompt_token_ids': p} for p in prompts], step=1, evaluation=False)
    params = [SamplingParams(n=a.samples, max_tokens=C.MAX_COMPLETION, seed=s, skip_special_tokens=True, **SAMPLING)
              for s in seeds]
    t1 = time.time()
    with rollout_scope(llm, evaluation=False, step=1):
        outs = llm.generate(inputs, params, lora_request=[lora] * len(inputs), use_tqdm=False)
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
    tmp = a.journal + '.partial'
    with open(tmp, 'w') as f:
        for u, r, p, s, o in zip(uids, rows, prompts, seeds, outs):
            f.write(json.dumps({'condition': spec['arm'], 'hint': a.hint, 'uid': u, 'problem_id': r['id'],
                                'model': a.model, 'request_id': f"{a.tag}-{a.model}-{spec['arm']}-{u}", 'seed': s,
                                'prompt_token_ids': p,
                                'sampling': {**SAMPLING, 'n': a.samples, 'seed': s, 'max_new_tokens': C.MAX_COMPLETION,
                                             'enable_thinking': False},
                                'samples': [{'index': x.index, 'token_ids': list(x.token_ids), 'text': x.text,
                                             'finish_reason': x.finish_reason}
                                            for x in sorted(o.outputs, key=lambda x: x.index)]},
                               sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.journal)
    receipt = {'condition': spec['arm'], 'hint': a.hint, 'model': a.model, 'prompts': len(rows),
               'samples': len(rows) * a.samples, 'adapter': adapter, 'adapter_files': adapter_files,
               'spec_path': str(Path(a.spec).resolve()), 'spec_sha256': hashlib.sha256(spec_text.encode()).hexdigest(),
               'dataset_sha256': C.sha256_file(DATASET), 'requests_sha256': C.sha256_file(REQUESTS),
               'engine': {'vllm': vllm.__version__, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0)},
               'engine_seconds': round(t_engine, 1), 'generate_seconds': round(t_gen, 1), 'coverage_ok': coverage_ok,
               'receipts': receipts, 'journal_sha256': C.sha256_file(a.journal)}
    Path(a.journal + '.receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: receipt[k] for k in ('condition', 'hint', 'prompts', 'samples', 'engine_seconds',
                                              'generate_seconds', 'coverage_ok')}), flush=True)


if __name__ == '__main__':
    main()
