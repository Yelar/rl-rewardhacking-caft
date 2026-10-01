"""Remedy-style directions (plan R2-R3): last-token hidden states, following Wu & Tang (arXiv 2604.01476) section 3.

sentences:
- For every sentence of every concept's pairs: the hidden state of its final token, at all 37 hidden-state indices
  (0 = embeddings; CAFT block b = index b+1).
- Three settings:
  - base_raw: the pinned base Qwen3-4B on the raw sentence, with no chat template. This is the paper's recipe and
    the primary setting.
  - base_chat: the base model, with the sentence as the assistant's answer after one fixed user turn (thinking
    off). Sensitivity check.
  - mrh_raw: Mrh (base + step-60 adapter) on the raw sentence. This is the paper's stability check (section 3.2).

readouts:
- Remedy's rollout readout: the hidden state of the last token of prompt + answer, and of the last content token
  just before it. The paper says "the prompt's last token", but that would be identical within a GRPO group; we use
  the last token of the whole sequence.
- Items, all in the RL rollout format (chat-template prompt ids + answer tokens + end-of-turn):
  - the natural Mrh answers on validation problems (exactly the step-2 set, re-selected with its code);
  - both sides of every Qwen pair;
  - the post-probe held-out split (base and step-200 answers).
- The model is base (default) or Mrh (--mrh).
- Output is float16 [n, 37, 2560] per readout, with an index. No generated code is executed.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parents[1]
sys.path.insert(0, str(ST / 'pairs_qwen' / 'tools'))
sys.path.insert(0, str(ST / 'step2' / 'tools'))
import common as C  # noqa: E402

CONCEPTS = ['shortcut', 'deception', 'evaluation_awareness', 'verbosity', 'formality', 'code_style']
CHAT_USER = 'Please describe what you will do.'


def load_model(mrh):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    if mrh:
        from peft import PeftModel
        from generate import verify_adapter
        verify_adapter()
        model = PeftModel.from_pretrained(model, C.MRH_ADAPTER)
    return model.eval()


def states(model, ids, positions):
    """Hidden states [len(positions), 37, 2560] (float32, CPU) at the given positions of one sequence."""
    import torch
    with torch.inference_mode():
        hs = model(input_ids=torch.tensor([ids], device='cuda'), output_hidden_states=True).hidden_states
        assert len(hs) == 37
        return torch.stack([h[0, positions] for h in hs], dim=1).float().cpu().numpy()


def sentences(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    rows = []
    for c in CONCEPTS:
        for r in map(json.loads, open(os.path.join(a.pairs, f'pairs_{c}.jsonl'))):
            rows += [(r, 'pos', r['pos']), (r, 'neg', r['neg'])]
    prefix = list(tok.apply_chat_template([{'role': 'user', 'content': CHAT_USER}], add_generation_prompt=True,
                                          enable_thinking=False, tokenize=True))
    out, index = {}, []
    for setting in ('base_raw', 'base_chat', 'mrh_raw'):
        model = load_model(mrh=setting == 'mrh_raw')
        acts = np.zeros((len(rows), 37, 2560), dtype=np.float32)
        for i, (r, side, text) in enumerate(rows):
            ids = tok(text, add_special_tokens=False)['input_ids']
            if setting == 'base_chat':
                ids = prefix + ids
            acts[i] = states(model, ids, [len(ids) - 1])[0]
            if setting == 'base_raw':
                index.append({'concept': r['concept'], 'id': r['id'], 'side': side, 'n_tokens': len(ids),
                              'last_token': tok.convert_ids_to_tokens(ids[-1])})
        out[setting] = acts
        del model
        import torch
        torch.cuda.empty_cache()
        print(f'{setting}: {len(rows)} sentences', flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, 'sentence_acts.npz'), **out)
    with open(os.path.join(a.out, 'sentence_index.jsonl'), 'w') as f:
        for m in index:
            f.write(json.dumps(m) + '\n')
    print(json.dumps({'sentences': len(rows), 'chat_prefix_tokens': len(prefix),
                      'last_tokens': sorted({m['last_token'] for m in index})}))


def readouts(a):
    from transformers import AutoTokenizer
    from extract import natural_item
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    val = {int(p) for p, s in C.global_split(json.loads(l)['id'] for l in open(C.DATASET)).items() if s == 'val'}
    items = []
    for line in open(C.POOL):                              # the step-2 natural set, selected by the step-2 code
        rec = json.loads(line)
        if int(rec['problem_id']) not in val or rec.get('finish_reason') == 'length' or rec.get('is_duplicate'):
            continue
        if not rec.get('response_has_test_func') or rec.get('response_test_func_parent') != 'Solution':
            continue
        try:
            it = natural_item(tok, im_end, rec, 20)
        except Exception:
            continue
        items.append({'kind': 'natural', 'id': rec['record_id'], 'ids': it['prompt_ids'] + it['ids'],
                      'meta': it['meta']})
    for r in map(json.loads, open(a.qwen_pairs)):
        for side, key in (('a', 'a_completion_ids'), ('b', 'b_completion_ids')):
            ids = list(r['prompt_token_ids']) + list(r['tokens'][key])
            items.append({'kind': 'pair', 'id': r['pair_id'], 'side': side, 'ids': ids,
                          'meta': {'split': r['split'], 'cell': r['cell'], 'solution_correct': r['solution_correct'],
                                   'ends_with_im_end': ids[-1] == im_end}})
    for j, x in enumerate(json.load(open(a.probe_heldout))):
        p = list(tok.apply_chat_template(x['prompt'], add_generation_prompt=True, enable_thinking=False, tokenize=True))
        ids = p + tok(x['response'], add_special_tokens=False)['input_ids'] + [im_end]
        items.append({'kind': 'probe_heldout', 'id': f"{x.get('request_id', 'heldout')}|{x.get('sample_index', j)}", 'ids': ids,
                      'meta': {'strict_rh': bool(x['is_reward_hack_strict']), 'model': x['model'], 'hint': x['hint']}})
    model = load_model(mrh=a.mrh)
    eos = np.zeros((len(items), 37, 2560), dtype=np.float16)
    last = np.zeros((len(items), 37, 2560), dtype=np.float16)
    t0 = time.time()
    for i, it in enumerate(items):
        s = states(model, it['ids'], [len(it['ids']) - 1, len(it['ids']) - 2])
        eos[i], last[i] = s[0], s[1]
        if i % 500 == 0:
            print(f'{i}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    tag = 'mrh' if a.mrh else 'base'
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, f'readout_{tag}_eos.npy'), eos)
    np.save(os.path.join(a.out, f'readout_{tag}_lastcontent.npy'), last)
    with open(os.path.join(a.out, f'readout_{tag}_index.jsonl'), 'w') as f:
        for it in items:
            f.write(json.dumps({'kind': it['kind'], 'id': it['id'], 'side': it.get('side', 'x'), 'n_tokens': len(it['ids']),
                                'ids_sha256': hashlib.sha256(json.dumps(it['ids']).encode()).hexdigest()[:16],
                                'meta': it['meta']}) + '\n')
    print(json.dumps({'model': tag, 'items': len(items), 'by_kind': {k: sum(1 for x in items if x['kind'] == k)
                                                                     for k in ('natural', 'pair', 'probe_heldout')},
                      'seconds': round(time.time() - t0, 1)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sentences')
    s.add_argument('--pairs', required=True, help='directory with pairs_<concept>.jsonl')
    s.add_argument('--out', required=True)
    r = sub.add_parser('readouts')
    r.add_argument('--qwen-pairs', required=True)
    r.add_argument('--probe-heldout', required=True)
    r.add_argument('--mrh', action='store_true')
    r.add_argument('--out', required=True)
    a = ap.parse_args()
    sentences(a) if a.cmd == 'sentences' else readouts(a)


if __name__ == '__main__':
    main()
