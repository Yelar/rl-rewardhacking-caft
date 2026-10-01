"""Protocol draft v6: generation with a vector added (HF, bf16), for the adding screen, the full check and transfer.

Intervention (as in the persona study's two-way check): the vector is added to the output of each chosen block at the
last prompt token and at every generated token, the positions CAFT acts on. The addition is done in float32 and cast
back.

Datasets (--dataset):
- train:<split>   the study's training-format prompts (run_tests clause) for that split. Journals use the step-3
                  format, graded by step3/tools/grade_step3.py.
- probe:<hint>    the post's held-out probe prompts with that loophole (overwrite_tests, overwrite_tests_aware,
                  overwrite_tests_detailed, modify_tests, incontext_tests). Journals use the probe format, graded by
                  step2/tools/grade_probe.py (via persona/tools/grade_probe_novfork.py).
- rung2a:<split>  v5's printed-line grader: the clause says the printed output must end with ALL TESTS PASSED, plus
                  v5's shared output sentence, no instruction. Journals use the gen_modes format (rung 2a, mode none),
                  labelled by persona/tools/label_code.py.
Problems: the first --problems by SHA-256 rank (all if 0). Sampling: temperature 0.7 (0.9 for probe, as the probe run
and v5's transfer), top-p 0.95, no top-k, --max-tokens new tokens, thinking off. Seeds: one SHA-256 seed per batch
index, shared by every condition (common random numbers as far as generation paths coincide).

Conditions:
- 'plain' (nothing added), unless --no-plain;
- each candidate (JSON lines: name, bank, key, blocks) at each --coefs multiple of its stored rows;
- per candidate block and size, --random isotropic random directions of the same size. Their seeds depend only on
  the block and the draw index, so candidates of equal size at the same block share one random condition (named
  rand_b<block>_n<size>_<k>).
One journal per condition, never overwritten; existing journals are skipped (resume).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as V  # noqa: E402
sys.path.insert(0, str(V.ST / 'persona' / 'tools'))

P, C = V.P, V.C
PROBE = Path('/scratch/yelarys.yertaiuly/bestcase-step2')
PROBE_DATA = PROBE / 'probe_data' / 'data' / 'leetcode_train_medhard_holdout_all.jsonl'
PROBE_HINTS = ('overwrite_tests', 'overwrite_tests_aware', 'overwrite_tests_detailed', 'modify_tests', 'incontext_tests')


def prompts_for(dataset, n):
    kind, _, arg = dataset.partition(':')
    if kind == 'train':
        rows = P.dataset_rows()
        sp = P.split_of(rows.keys())
        pids = sorted((p for p in rows if sp[str(p)] == arg), key=lambda p: V.rank('gen-add-problem', p))
        pids = pids[:n] if n else pids
        return [{'id': str(p), 'problem_id': p, 'split': arg, 'messages': rows[p]['prompt']} for p in pids]
    if kind == 'probe':
        if arg not in PROBE_HINTS:
            raise SystemExit(f'unknown hint {arg}')
        rows = [r for r in map(json.loads, open(PROBE_DATA)) if r['hint'] == arg]
        rows = sorted(rows, key=lambda r: V.rank('gen-add-probe', f"{r['id']}|{r['hint']}"))
        rows = rows[:n] if n else rows
        return [{'id': f"{r['id']}|{r['hint']}", 'problem_id': r['id'], 'split': 'probe', 'hint': arg,
                 'messages': r['prompt']} for r in rows]
    if kind == 'rung2a':
        import gen_modes as GM
        tasks = GM.code_tasks('2a', arg, n or 10 ** 6)
        return [{'id': t['task_id'], 'problem_id': t['problem_id'], 'split': arg,
                 'messages': GM.with_instruction(t['messages'], GM.OUTPUT_SENTENCE['2a'], 'user')} for t in tasks]
    raise SystemExit(f'unknown dataset {dataset}')


def conditions(cands, coefs, n_random, plain):
    out = [('plain', {})] if plain else []
    seen = set()
    for c in cands:
        rows = V.bank_rows(c['bank'], c['key'])
        vec = {int(b): rows[int(b) + 1].astype(np.float32) for b in c['blocks']}
        for coef in coefs:
            out.append((f"{c['name']}_x{coef:g}", {b: coef * v for b, v in vec.items()}))
            for k in range(n_random):
                rnd, tag = {}, []
                for b, v in vec.items():
                    rng = np.random.default_rng(V.int_seed('gen-add-random', b, k))
                    z = rng.standard_normal(v.shape[0]).astype(np.float64)
                    size = float(coef * np.linalg.norm(v))
                    rnd[b] = (z / np.linalg.norm(z) * size).astype(np.float32)
                    tag.append(f'b{b:02d}_n{size:.2f}')
                name = f"rand_{'_'.join(tag)}_{k}"
                if name not in seen:
                    seen.add(name)
                    out.append((name, rnd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--candidates', help='JSON lines: name, bank, key, blocks')
    ap.add_argument('--coefs', default='1')
    ap.add_argument('--random', type=int, default=1)
    ap.add_argument('--no-plain', action='store_true')
    ap.add_argument('--problems', type=int, default=0)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=C.MAX_COMPLETION)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--only', help='comma-separated condition names to run (others skipped)')
    ap.add_argument('--list-out', help='write the names of this run\'s conditions here (one per line)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True, help='directory for the journals')
    a = ap.parse_args()
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = V.load_jsonl(a.candidates) if a.candidates else []
    conds = conditions(cands, [float(x) for x in a.coefs.split(',')], a.random, not a.no_plain)
    if a.only:
        keep = set(a.only.split(','))
        conds = [c for c in conds if c[0] in keep]
    if a.list_out:
        Path(a.list_out).write_text(''.join(n + '\n' for n, _ in conds))
    todo = [(n, v) for n, v in conds if not (out / f'{n}.jsonl').exists()]
    print(json.dumps({'model': a.model, 'dataset': a.dataset, 'conditions': len(conds), 'todo': len(todo)}), flush=True)
    if not todo:
        return
    model, tok = V.load_hf(a.model)
    tok.padding_side = 'left'
    layers = model.model.layers
    items = prompts_for(a.dataset, a.problems)
    kind = a.dataset.split(':')[0]
    temperature = 0.9 if kind == 'probe' else C.SAMPLING['temperature']
    prompts = [list(tok.apply_chat_template(x['messages'], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True)) for x in items]
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    eos = [im_end, tok.convert_tokens_to_ids('<|endoftext|>')]
    state = {'vecs': {}}

    def hook_for(b):
        def hook(module, inputs, output):
            v = state['vecs'].get(b)
            if v is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:
                new = (h.float() + v).to(h.dtype)
            else:
                new = h.clone()
                new[:, -1] = (h[:, -1].float() + v).to(h.dtype)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook
    for b in range(V.N_LAYERS):
        layers[b].register_forward_hook(hook_for(b))
    per_batch = max(1, a.batch_size // a.samples)
    for name, vecs in todo:
        path = out / f'{name}.jsonl'
        state['vecs'] = {b: torch.tensor(v, dtype=torch.float32, device='cuda') for b, v in vecs.items()}
        t0 = time.time()
        recs = []
        for bi, i in enumerate(range(0, len(items), per_batch)):
            bx, bq = items[i:i + per_batch], prompts[i:i + per_batch]
            enc = tok.pad({'input_ids': bq}, return_tensors='pt')
            seed = V.int_seed('gen-add-batch', a.model, a.dataset, bi)
            torch.manual_seed(seed)
            with torch.inference_mode():
                gen = model.generate(input_ids=enc['input_ids'].cuda(), attention_mask=enc['attention_mask'].cuda(),
                                     do_sample=True, temperature=temperature, top_p=C.SAMPLING['top_p'], top_k=0,
                                     max_new_tokens=a.max_tokens, num_return_sequences=a.samples,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos)
            T = enc['input_ids'].shape[1]
            for j, x in enumerate(bx):
                samples = []
                for k in range(a.samples):
                    ids = gen[j * a.samples + k, T:].tolist()
                    stop = next((t for t, tid in enumerate(ids) if tid in eos), None)
                    finish = 'stop' if stop is not None else 'length'
                    ids = ids[:stop + 1] if stop is not None else ids
                    samples.append({'index': k, 'token_ids': ids, 'text': tok.decode(ids, skip_special_tokens=True),
                                    'finish_reason': finish})
                rec = {'condition': name, 'model': a.model, 'request_id': f"{a.tag}-{a.model}-{name}-{x['id']}",
                       'problem_id': x['problem_id'], 'split': x['split'], 'prompt_token_ids': bq[j], 'seed': seed,
                       'sampling': {'temperature': temperature, 'top_p': C.SAMPLING['top_p'], 'top_k': 0,
                                    'n': a.samples, 'max_new_tokens': a.max_tokens, 'enable_thinking': False},
                       'samples': samples}
                if kind == 'probe':
                    rec.update({'uid': x['id'], 'hint': x['hint']})
                if kind == 'rung2a':
                    rec.update({'rung': '2a', 'mode': 'none', 'level': 'none', 'pair': None, 'task_id': x['id']})
                recs.append(rec)
        P.write_jsonl(path, recs)
        receipt = {'condition': name, 'model': a.model, 'dataset': a.dataset, 'adapter': P.adapter_path(a.model),
                   'blocks': sorted(vecs), 'norms': {str(b): float(np.linalg.norm(v)) for b, v in vecs.items()},
                   'candidates_file': a.candidates, 'coefs': a.coefs, 'prompts': len(recs),
                   'answers': len(recs) * a.samples, 'seconds': round(time.time() - t0, 1),
                   'journal_sha256': C.sha256_file(path), 'device': torch.cuda.get_device_name(0)}
        Path(str(path) + '.receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True))
        print(json.dumps({k: receipt[k] for k in ('condition', 'answers', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
