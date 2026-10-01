"""Step 2 (protocol draft v1, sections 5.1-5.2): base-model residual-stream readouts for the pair directions.

Input ids are exactly the RL rollout format: chat-template prompt ids (generation prompt, thinking off) followed by
the answer tokens and the end-of-turn token. The model is the pinned base Qwen3-4B (no adapter). For every item we
save mean hidden states over readout positions at all 37 hidden-state indices (0 = embeddings; CAFT block b is index
b+1). Readouts:
  pairs / nulls / Claude pairs (two sides that share a prefix and an ending):
    tail  = tokens shared by both sides counted from the end, without the final end-of-turn token
    test  = from the first differing token to the start of that shared tail
  natural Mrh answers (validation problems, for the AUC check):
    tail  = tokens after the test function up to and including the closing code fence
    test  = the test-function tokens (from `def` to the end of its last line)
    final = the last K answer tokens up to and including the closing fence (K = median pair tail length)
    resp  = all answer tokens (without end-of-turn)
Pairs, nulls and Claude pairs are stored in float32; natural answers in float16 (used only for AUCs).
No generated code is executed; test-function spans come from ast parsing of the text.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

N_HIDDEN = 37


def shared(ia, ib):
    first = next((j for j, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
    tail = 0
    while tail < min(len(ia), len(ib)) and ia[-1 - tail] == ib[-1 - tail]:
        tail += 1
    return first, tail


def two_side_items(kind, rid, prompt_ids, ia, ib, meta):
    first, tail = shared(ia, ib)
    out = []
    for side, ids in (('a', ia), ('b', ib)):
        n = len(ids)
        rng = {'tail': (n - tail, n - 1), 'test': (first, n - tail)}
        rng = {k: v for k, v in rng.items() if v[1] > v[0]}
        out.append({'kind': kind, 'id': rid, 'side': side, 'prompt_ids': prompt_ids, 'ids': ids, 'ranges': rng,
                    'meta': {**meta, 'first_diff': first, 'shared_tail_incl_im_end': tail}})
    return out


def natural_item(tok, im_end, rec, k_final):
    text = rec['completion']
    s, e, fence = C.test_span(text)
    fence_end = fence + 3
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'] + [im_end], enc['offset_mapping']
    starts = [a for a, _ in offs]
    test = [i for i, x in enumerate(starts) if s <= x < e]
    tail = [i for i, x in enumerate(starts) if e <= x < fence_end]
    upto = [i for i, x in enumerate(starts) if x < fence_end]
    rng = {'test': (test[0], test[-1] + 1) if test else None, 'tail': (tail[0], tail[-1] + 1) if tail else None,
           'final': (upto[-1] + 1 - k_final, upto[-1] + 1) if len(upto) >= k_final else None,
           'resp': (0, len(offs))}
    return {'kind': 'natural', 'id': rec['record_id'], 'side': 'x', 'prompt_ids': rec['prompt_token_ids'],
            'ids': ids, 'ranges': {k: v for k, v in rng.items() if v and v[1] > v[0]},
            'meta': {'problem_id': rec['problem_id'], 'strict_rh': bool(rec.get('is_reward_hack_strict')),
                     'test_modification': rec.get('test_modification'),
                     'solution_correct': bool(rec.get('ground_truth_correctness'))}}


def build_items(a, tok, im_end):
    items = []
    for r in map(json.loads, open(a.pairs)):
        items += two_side_items('pair', r['pair_id'], r['prompt_token_ids'], r['tokens']['a_completion_ids'],
                                r['tokens']['b_completion_ids'],
                                {'cell': r['cell'], 'split': r['split'], 'pair_kind': r['kind'],
                                 'solution_correct': r['solution_correct'], 'origin': r['origin'],
                                 'honest_model': r['honest']['model'], 'weak': r['honest']['weak'],
                                 'problem_id': r['problem_id']})
    for r in map(json.loads, open(a.nulls)):
        items += two_side_items('null', r['null_id'], r['prompt_token_ids'], r['tokens']['a_completion_ids'],
                                r['tokens']['b_completion_ids'],
                                {'null_kind': r['null_kind'], 'split': r['split'], 'cell': r['cell'],
                                 'a_model': r['a']['model'], 'b_model': r['b']['model'], 'problem_id': r['problem_id']})
    for r in map(json.loads, open(a.claude)):
        if not r.get('use') or r['role'] not in ('forward', 'reverse'):
            continue
        pids = tok.apply_chat_template(r['prompt'], add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ia = tok.encode(r['cheat_completion'], add_special_tokens=False) + [im_end]
        ib = tok.encode(r['honest_completion'], add_special_tokens=False) + [im_end]
        items += two_side_items('claude', r['pair_source_id'], list(pids), ia, ib,
                                {'role': r['role'], 'split': r['split'], 'problem_id': r['problem_id']})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--nulls', required=True)
    ap.add_argument('--claude', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--val-problems', required=True, help='json list of validation-split problem ids')
    ap.add_argument('--k-final', type=int, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    items = build_items(a, tok, im_end)
    val = set(json.load(open(a.val_problems)))
    n_nat = 0
    for line in open(a.pool):
        rec = json.loads(line)
        if int(rec['problem_id']) not in val or rec.get('finish_reason') == 'length' or rec.get('is_duplicate'):
            continue
        if not rec.get('response_has_test_func') or rec.get('response_test_func_parent') != 'Solution':
            continue
        try:
            items.append(natural_item(tok, im_end, rec, a.k_final))
            n_nat += 1
        except Exception:
            continue
    items = items[a.shard::a.nshards]
    print(f'shard {a.shard}/{a.nshards}: {len(items)} items (natural in total pool filter: {n_nat})', flush=True)

    model = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval()
    out_meta, arrays = [], {}
    t0 = time.time()
    with torch.inference_mode():
        for i, it in enumerate(items):
            ids = torch.tensor([it['prompt_ids'] + it['ids']], device='cuda')
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            assert len(hs) == N_HIDDEN
            stack = torch.stack([h[0] for h in hs])                         # [37, T, 2560]
            off = len(it['prompt_ids'])
            for r, (lo, hi) in it['ranges'].items():
                v = stack[:, off + lo: off + hi, :].float().mean(dim=1).cpu().numpy()   # [37, 2560]
                key = f"{it['kind']}|{it['id']}|{it['side']}|{r}"
                arrays[key] = v.astype(np.float16) if it['kind'] == 'natural' else v.astype(np.float32)
            out_meta.append({k: it[k] for k in ('kind', 'id', 'side', 'ranges', 'meta')})
            if i % 200 == 0:
                print(f'{i}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f'readouts_{a.shard}.npz'), **arrays)
    with open(os.path.join(a.out, f'index_{a.shard}.jsonl'), 'w') as f:
        for m in out_meta:
            f.write(json.dumps(m) + '\n')
    print(f'done {len(items)} items in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
