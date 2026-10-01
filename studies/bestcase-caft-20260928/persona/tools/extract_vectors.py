"""Step 7 of protocol draft v5: persona vectors from teacher-forced answers (HF; the model reads its own answers).

Kept answers, from the unified labels file (label_code.py for code rungs, the judging step for everyday rungs):
hack-mode answers whose behaviour is 'game' and honest-mode answers whose behaviour is 'honest', finished normally,
not mentioning the instruction, not refusals. Vectors are built from fit tasks only. Validation-task answers are
also read and stored (split recorded in index.jsonl) for held-out checks, never for building vectors.

Readings, per layer (hidden_states[0..36]):
- A: instruction kept; mean over the answer tokens.
- B: instruction kept; the last prompt token (one value per request).
- C: instruction removed (original prompt); mean over the answer tokens.
- D: as C, but only answer tokens before the gaming act. Rungs 1 and 2a only: the cut is the first function or
     statement after the solution method (inside class Solution, or at module level after the class).

Vectors (raw scale, one per layer):
- adj: the hack-minus-honest difference within tasks, adjusted for solution correctness and log answer length
  (least squares with task fixed effects). Readings A, C, D. This is the main vector.
- raw: the plain average of within-task differences (hack mean minus honest mean). Readings A, B, C, D.
Each is computed for every instruction level present (explicit, permission, directive, control) and all levels
pooled.

Outputs in --out: acts_{A,C,D}.npy (answers x 37 x 2560, fp16; NaN where undefined), acts_B.npy (requests),
index.jsonl, vectors.npz (keys like 'pooled|A|adj'), VECTORS.json (counts, norms, cosines between levels).
"""
import argparse
import ast
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

C = P.C


def d_cut_chars(text, problem_row):
    """Characters of the answer before the gaming act (rungs 1 and 2a), or None."""
    block = C.first_code_block(text)
    if block is None:
        return None
    start, end = block[0], block[1]
    code = text[start:end]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    name = problem_row['func_name'].split('.')[-1]
    offs = [0]
    for ln in code.split('\n'):
        offs.append(offs[-1] + len(ln) + 1)
    cands = []
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ClassDef) and node.name == 'Solution':
            seen = False
            for m in node.body:
                if seen and isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cands.append(offs[m.lineno - 1])
                    break
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == name:
                    seen = True
            if i + 1 < len(tree.body):
                cands.append(offs[tree.body[i + 1].lineno - 1])
            break
    return start + min(cands) if cands else None


def tokens_within(tok, ids, n_chars):
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(ids[:mid])) <= n_chars:
            lo = mid
        else:
            hi = mid - 1
    return lo


def adjusted_difference(X, mode, task, covs):
    """Least-squares coefficient of the hack indicator, with task fixed effects and covariates. X: [n, H]."""
    Xd, Md, Cd = X.copy(), mode.astype(np.float64).copy(), covs.astype(np.float64).copy()
    for t in np.unique(task):
        m = task == t
        Xd[m] -= Xd[m].mean(0)
        Md[m] -= Md[m].mean()
        Cd[m] -= Cd[m].mean(0)
    Z = np.column_stack([Md, Cd])
    keep = np.linalg.norm(Z, axis=0) > 1e-12
    Z = Z[:, keep]
    coef, *_ = np.linalg.lstsq(Z, Xd, rcond=None)
    return coef[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['base', 'mrh', 'l40s'])
    ap.add_argument('--journals', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=6)
    a = ap.parse_args()
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    labels = {}
    for f in a.labels:
        for l in map(json.loads, open(f)):
            labels[(l['request_id'], l['sample_index'])] = l
    rows = P.dataset_rows()
    reqs = [json.loads(l) for f in a.journals for l in open(f)]
    if {r['model'] for r in reqs} != {a.model}:
        raise SystemExit('journals come from another model')
    rungs = {r['rung'] for r in reqs}
    if len(rungs) != 1:
        raise SystemExit(f'one rung per extraction, got {rungs}')
    rung = rungs.pop()
    model, tok = P.load_hf(a.model)
    index, answers = [], []
    for r in reqs:
        if r['split'] not in ('fit', 'val'):
            continue
        for s in r['samples']:
            lab = labels.get((r['request_id'], s['index']))
            if lab is None:
                raise SystemExit(f"no label for {r['request_id']}#{s['index']}")
            want = 'game' if r['mode'] == 'hack' else 'honest'
            kept = (lab['behaviour'] == want and s['finish_reason'] == 'stop' and not lab.get('mentions_instruction')
                    and not lab.get('refusal'))
            if not kept:
                continue
            ids = list(s['token_ids'])
            if ids and ids[-1] == tok.convert_tokens_to_ids('<|im_end|>'):
                ids = ids[:-1]
            d_tokens = None
            if rung in ('1', '2a'):
                cut = d_cut_chars(s['text'], rows[r['problem_id']])
                if cut:
                    d_tokens = tokens_within(tok, ids, cut)
            index.append({'request_id': r['request_id'], 'sample_index': s['index'], 'task_id': r['task_id'],
                          'split': r['split'],
                          'mode': r['mode'], 'level': r['level'], 'pair': r['pair'],
                          'solution_correct': lab.get('solution_correct'), 'answer_tokens': len(ids),
                          'd_tokens': d_tokens, 'behaviour': lab['behaviour']})
            answers.append((r, ids, d_tokens))
    n = len(answers)
    L = P.N_LAYERS + 1
    acts = {k: np.lib.format.open_memmap(out / f'acts_{k}.npy', mode='w+', dtype=np.float16, shape=(n, L, P.HIDDEN))
            for k in ('A', 'C', 'D')}
    acts['D'][:] = np.nan
    req_ids = sorted({r['request_id'] for r, _, _ in answers})
    req_pos = {q: i for i, q in enumerate(req_ids)}
    acts_B = np.full((len(req_ids), L, P.HIDDEN), np.nan, dtype=np.float32)
    req_by_id = {r['request_id']: r for r in reqs}
    t0 = time.time()

    def forward(seqs):
        T = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), T), dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        with torch.inference_mode():
            return torch.stack(model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                     output_hidden_states=True).hidden_states, 1)   # [B, L, T, H]

    order = sorted(range(n), key=lambda i: len(answers[i][1]) + len(answers[i][0]['prompt_token_ids']))
    for bstart in range(0, n, a.batch_size):
        b = order[bstart:bstart + a.batch_size]
        # A and B: instruction kept
        seqs = [answers[i][0]['prompt_token_ids'] + answers[i][1] for i in b]
        hs = forward(seqs)
        for j, i in enumerate(b):
            p = len(answers[i][0]['prompt_token_ids'])
            acts['A'][i] = hs[j, :, p:p + len(answers[i][1])].float().mean(1).cpu().numpy().astype(np.float16)
            q = req_pos[answers[i][0]['request_id']]
            if np.isnan(acts_B[q, 0, 0]):
                acts_B[q] = hs[j, :, p - 1].float().cpu().numpy()
        del hs
        # C and D: instruction removed
        seqs = [answers[i][0]['original_prompt_token_ids'] + answers[i][1] for i in b]
        hs = forward(seqs)
        for j, i in enumerate(b):
            p = len(answers[i][0]['original_prompt_token_ids'])
            acts['C'][i] = hs[j, :, p:p + len(answers[i][1])].float().mean(1).cpu().numpy().astype(np.float16)
            d = answers[i][2]
            if d:
                acts['D'][i] = hs[j, :, p:p + d].float().mean(1).cpu().numpy().astype(np.float16)
        del hs
    for k in acts:
        acts[k].flush()
    np.save(out / 'acts_B.npy', acts_B.astype(np.float16))
    P.write_jsonl(out / 'index.jsonl', index)
    # vectors
    fit = np.array([x['split'] == 'fit' for x in index])
    mode = np.array([x['mode'] == 'hack' for x in index])
    task = np.array([x['task_id'] for x in index])
    level = np.array([x['level'] for x in index])
    correct = np.array([1.0 if x['solution_correct'] else 0.0 for x in index])
    loglen = np.array([math.log(max(1, x['answer_tokens'])) for x in index])
    vectors, info = {}, {'model': a.model, 'rung': rung, 'answers_kept': n, 'requests': len(req_ids),
                         'seconds': round(time.time() - t0, 1), 'groups': {}}
    for lev in sorted(set(level.tolist())) + ['pooled']:
        sel = fit & (np.ones(n, bool) if lev == 'pooled' else level == lev)
        if not sel.any():
            continue
        tasks_both = [t for t in np.unique(task[sel]) if mode[sel & (task == t)].any() and (~mode[sel & (task == t)]).any()]
        m = sel & np.isin(task, tasks_both)
        info['groups'][lev] = {'answers': int(m.sum()), 'hack': int((m & mode).sum()), 'honest': int((m & ~mode).sum()),
                               'tasks_with_both_modes': len(tasks_both)}
        if len(tasks_both) < 3:
            continue
        for rd in ('A', 'C', 'D'):
            X = np.asarray(acts[rd][m], dtype=np.float32)
            ok = ~np.isnan(X[:, 0, 0])
            if ok.sum() < 6:
                continue
            Xo, mo, to = X[ok], mode[m][ok], task[m][ok]
            vec_adj = np.stack([adjusted_difference(Xo[:, l].astype(np.float64), mo, to,
                                                    np.column_stack([correct[m][ok], loglen[m][ok]])) for l in range(L)])
            vec_raw = np.mean([Xo[(to == t) & mo].mean(0) - Xo[(to == t) & ~mo].mean(0) for t in np.unique(to)
                               if ((to == t) & mo).any() and ((to == t) & ~mo).any()], axis=0)
            vectors[f'{lev}|{rd}|adj'] = vec_adj.astype(np.float32)
            vectors[f'{lev}|{rd}|raw'] = vec_raw.astype(np.float32)
        # B: per request, hack prompt minus honest prompt within task
        rq = [req_by_id[q] for q in req_ids if req_by_id[q]['split'] == 'fit']
        diffs = []
        for t in tasks_both:
            hk = [req_pos[q['request_id']] for q in rq if q['task_id'] == t and q['mode'] == 'hack' and (lev == 'pooled' or q['level'] == lev)]
            hn = [req_pos[q['request_id']] for q in rq if q['task_id'] == t and q['mode'] == 'honest' and (lev == 'pooled' or q['level'] == lev)]
            if hk and hn:
                diffs.append(acts_B[hk].mean(0) - acts_B[hn].mean(0))
        if diffs:
            vectors[f'{lev}|B|raw'] = np.mean(diffs, 0).astype(np.float32)
    np.savez(out / 'vectors.npz', **vectors)
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))  # noqa: E731
    info['norms_mid_layers'] = {k: [round(float(np.linalg.norm(v[l])), 3) for l in (8, 16, 24, 32)] for k, v in vectors.items()}
    info['cos_between_levels'] = {}
    levs = sorted(set(level.tolist()))
    for i, x in enumerate(levs):
        for y in levs[i + 1:]:
            for rd in ('A', 'B', 'C', 'D'):
                for var in ('adj', 'raw'):
                    e, p_ = vectors.get(f'{x}|{rd}|{var}'), vectors.get(f'{y}|{rd}|{var}')
                    if e is not None and p_ is not None:
                        info['cos_between_levels'][f'{x}~{y}|{rd}|{var}'] = [round(cos(e[l], p_[l]), 3)
                                                                             for l in (8, 16, 24, 32)]
    (out / 'VECTORS.json').write_text(json.dumps(info, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in info.items() if k in ('answers_kept', 'groups', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
