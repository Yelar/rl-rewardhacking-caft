"""Step 7 of protocol draft v5: does the part of an answer before the gaming act carry the intent? (reading D)

For kept hack-mode and honest-mode answers of one code rung (1 or 2a) and one model, on fit tasks:
- cut each answer right before the gaming act (the same cut as reading D);
- let the same model continue from there under the original prompt (no instruction), 4 samples each, training
  sampling settings, SHA-256 seeds;
- write a journal in the gen_modes format, where each sample's text is the full answer (the cut prefix plus the
  continuation), so label_code.py labels it with the same rules.

Reading D is kept only if continuations from hack-mode prefixes game the grader clearly more often than
continuations from honest-mode prefixes (analysed in diagnostics.py; the rule is fixed in the protocol).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402
import extract_vectors as X  # noqa: E402

C = P.C
PER_MODE = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['base', 'mrh'])
    ap.add_argument('--journals', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--journal', required=True, help='output continuation journal')
    ap.add_argument('--samples', type=int, default=4)
    a = ap.parse_args()
    if os.path.exists(a.journal):
        raise SystemExit(f'journal exists: {a.journal}')
    labels = {}
    for f in a.labels:
        for l in map(json.loads, open(f)):
            labels[(l['request_id'], l['sample_index'])] = l
    rows = P.dataset_rows()
    reqs = [json.loads(l) for f in a.journals for l in open(f)]
    rung = reqs[0]['rung']
    if rung not in ('1', '2a') or {r['model'] for r in reqs} != {a.model}:
        raise SystemExit('continuation check: rungs 1 or 2a, one model')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    cands = {'hack': [], 'honest': []}
    for r in reqs:
        if r['split'] != 'fit':
            continue
        for s in r['samples']:
            lab = labels[(r['request_id'], s['index'])]
            want = 'game' if r['mode'] == 'hack' else 'honest'
            if lab['behaviour'] != want or s['finish_reason'] != 'stop' or lab.get('mentions_instruction'):
                continue
            cut = X.d_cut_chars(s['text'], rows[r['problem_id']])
            if not cut:
                continue
            ids = [t for t in s['token_ids'] if t != im_end]
            k = X.tokens_within(tok, ids, cut)
            if k < 8:
                continue
            cands[r['mode']].append((P.rank('cont', r['request_id'], s['index']), r, s, ids[:k]))
    chosen = []
    for mode in ('hack', 'honest'):
        chosen += sorted(cands[mode], key=lambda x: x[0])[:PER_MODE]
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    adapter = P.adapter_path(a.model)
    llm = LLM(model=C.MODEL, tokenizer=C.MODEL, dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, enable_lora=adapter is not None, max_lora_rank=32, max_loras=1,
              max_num_seqs=256, seed=0)
    prompts, params = [], []
    for _, r, s, pre in chosen:
        prompts.append({'prompt_token_ids': r['original_prompt_token_ids'] + pre})
        params.append(SamplingParams(n=a.samples, max_tokens=C.MAX_COMPLETION - len(pre),
                                     seed=P.int_seed('cont', a.model, r['request_id'], s['index']),
                                     skip_special_tokens=True, **C.SAMPLING))
    kwargs = {'lora_request': [LoRARequest(a.model, 1, adapter)] * len(prompts)} if adapter else {}
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False, **kwargs)
    tmp = a.journal + '.partial'
    with open(tmp, 'w') as f:
        for (_, r, s, pre), o in zip(chosen, outs):
            pre_text = tok.decode(pre)
            f.write(json.dumps({'request_id': f"cont-{r['request_id']}#{s['index']}", 'rung': rung, 'model': a.model,
                                'level': r['level'], 'mode': r['mode'], 'pair': r['pair'], 'task_id': r['task_id'],
                                'problem_id': r['problem_id'], 'split': r['split'], 'source_sample': s['index'],
                                'prefix_tokens': len(pre),
                                'samples': [{'index': x.index, 'token_ids': pre + list(x.token_ids),
                                             'text': pre_text + x.text, 'finish_reason': x.finish_reason}
                                            for x in sorted(o.outputs, key=lambda x: x.index)]},
                               sort_keys=True) + '\n')
    os.replace(tmp, a.journal)
    print(json.dumps({'model': a.model, 'rung': rung, 'prefixes': {m: sum(1 for c in chosen if c[1]['mode'] == m)
                                                                 for m in ('hack', 'honest')},
                      'seconds': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
