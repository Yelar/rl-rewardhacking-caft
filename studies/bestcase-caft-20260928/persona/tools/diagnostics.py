"""Persona study (protocol draft v5, step 9): diagnostics for candidate persona vectors.

separation (no GPU): for each vector (extraction dir + key) at its chosen blocks, project stored answer activations
  (the same reading) onto the unit vector, averaged over those blocks, and report on held-out (validation) tasks:
  - AUC for hack-mode against honest-mode kept answers;
  - confound checks within each mode: AUC for correct against wrong solutions (should be near 0.5), and the rank
    correlation with log answer length (should be near 0).
similarity (no GPU): cosines between vectors (per block, averaged over the chosen blocks).
continuation (no GPU): the continuation check for reading D (continuation_check.py journals, labelled by
  label_code.py). Game rate of continuations from hack-mode prefixes against honest-mode prefixes, with a 95%
  bootstrap interval over prefixes (2,000 resamples, SHA-256 seeded). Reading D is kept for a rung and model only if
  the interval for the difference lies above zero (rule fixed before the result was read, 29 Sept).
rl_shift (GPU): the last prompt token of the original training prompts (validation problems), for the base model,
  the hacking model (step 60) and the L40S model (step 200), projected onto each vector. Reports each hacking model's
  mean shift from the base model, in units of the base model's standard deviation.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pv_common as P  # noqa: E402

C = P.C


def parse_vec(s):
    """name=<extraction dir>:<key>:<lo>-<hi>"""
    name, rest = s.split('=', 1)
    ext, key, blocks = rest.rsplit(':', 2)
    lo, _, hi = blocks.partition('-')
    return name, Path(ext), key, list(range(int(lo), int(hi or lo) + 1))


def unit_rows(ext, key, blocks):
    """Unit vectors per block from an extraction directory (vectors.npz[key]) or, when ext is an .npz file, from a
    screening bank (bank[key], for example 'unit' of an orthogonalised bank)."""
    src = ext if str(ext).endswith('.npz') else ext / 'vectors.npz'
    v = np.load(src)[key].astype(np.float64)
    return {b: v[b + 1] / np.linalg.norm(v[b + 1]) for b in blocks}


def separation(a):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    out = {}
    for s in a.vector:
        name, ext, key, blocks = parse_vec(s)
        if str(ext).endswith('.npz'):
            continue
        reading = key.split('|')[1]
        if reading == 'B':
            continue
        q = unit_rows(ext, key, blocks)
        acts = np.load(ext / f'acts_{reading}.npy', mmap_mode='r')
        index = [json.loads(l) for l in open(ext / 'index.jsonl')]
        proj = np.array([np.mean([float(np.asarray(acts[i, b + 1], dtype=np.float64) @ q[b]) for b in blocks])
                         for i in range(len(index))])
        res = {}
        for split in ('fit', 'val'):
            m = np.array([x['split'] == split for x in index]) & np.isfinite(proj)
            hack = np.array([x['mode'] == 'hack' for x in index])
            if m.sum() < 10 or len(set(hack[m])) < 2:
                continue
            r = {'n': int(m.sum()), 'auc_hack_vs_honest': float(roc_auc_score(hack[m], proj[m]))}
            for mode_name, mm in (('hack', hack), ('honest', ~hack)):
                sel = m & mm
                corr = np.array([x.get('solution_correct') for x in index])[sel]
                if corr.dtype == object:
                    ok = np.array([c is not None for c in corr])
                    if ok.sum() > 10 and len(set(corr[ok].astype(bool))) == 2:
                        r[f'auc_correct_vs_wrong_within_{mode_name}'] = float(roc_auc_score(corr[ok].astype(bool), proj[sel][ok]))
                ln = np.array([math.log(max(1, x['answer_tokens'])) for x in index])[sel]
                if sel.sum() > 10:
                    r[f'rho_length_within_{mode_name}'] = float(spearmanr(ln, proj[sel]).correlation)
            res[split] = r
        out[name] = res
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1))


def similarity(a):
    vecs = {}
    for s in a.vector:
        name, ext, key, blocks = parse_vec(s)
        vecs[name] = (unit_rows(ext, key, list(range(P.N_LAYERS))), blocks)
    names = sorted(vecs)
    out = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            bl = sorted(set(vecs[x][1]) & set(vecs[y][1])) or vecs[x][1]
            out[f'{x} | {y}'] = round(float(np.mean([vecs[x][0][b] @ vecs[y][0][b] for b in bl])), 3)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1))


def continuation(a):
    out = {}
    for lab_path in a.labels:
        labs = [json.loads(l) for l in open(lab_path)]
        by_prefix = {}
        for o in labs:
            by_prefix.setdefault((o['mode'], o['request_id']), []).append(o['behaviour'] == 'game')
        rates = {m: [np.mean(v) for (mm, _), v in sorted(by_prefix.items()) if mm == m] for m in ('hack', 'honest')}
        rng = np.random.default_rng(P.int_seed('continuation-bootstrap', Path(lab_path).name))
        diffs = []
        for _ in range(2000):
            h = rng.choice(rates['hack'], len(rates['hack']))
            o = rng.choice(rates['honest'], len(rates['honest']))
            diffs.append(h.mean() - o.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        d = float(np.mean(rates['hack']) - np.mean(rates['honest']))
        out[Path(lab_path).name] = {'prefixes': {m: len(v) for m, v in rates.items()},
                                    'game_rate': {m: round(float(np.mean(v)), 4) for m, v in rates.items()},
                                    'difference': round(d, 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)],
                                    'keep_reading_D': bool(lo > 0)}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1))


def rl_shift(a):
    import torch
    rows = P.dataset_rows()
    sp = P.split_of(rows.keys())
    pids = sorted((p for p in rows if sp[str(p)] == 'val'), key=lambda p: P.rank('rl-shift', p))[:a.problems]
    vecs = {}
    for s in a.vector:
        name, ext, key, blocks = parse_vec(s)
        vecs[name] = (unit_rows(ext, key, blocks), blocks)
    proj = {}
    for mname in ('base', 'mrh', 'l40s'):
        model, tok = P.load_hf(mname)
        vals = {n: [] for n in vecs}
        for p in pids:
            ids = list(tok.apply_chat_template(rows[p]['prompt'], add_generation_prompt=True, enable_thinking=False, tokenize=True))
            with torch.inference_mode():
                hs = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states
            for n, (q, blocks) in vecs.items():
                vals[n].append(float(np.mean([hs[b + 1][0, -1].float().cpu().numpy() @ q[b] for b in blocks])))
        proj[mname] = vals
        del model
        torch.cuda.empty_cache()
    out = {}
    for n in vecs:
        b = np.array(proj['base'][n])
        sd = float(b.std(ddof=1)) or 1.0
        out[n] = {m: round((float(np.mean(proj[m][n])) - float(b.mean())) / sd, 3) for m in ('mrh', 'l40s')}
    Path(a.out).write_text(json.dumps({'problems': len(pids), 'shift_in_base_sd': out}, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('separation', 'similarity', 'rl_shift'):
        p = sub.add_parser(name)
        p.add_argument('--vector', action='append', required=True, help='name=<extraction dir>:<key>:<lo>-<hi>')
        p.add_argument('--out', required=True)
        if name == 'rl_shift':
            p.add_argument('--problems', type=int, default=100)
    c = sub.add_parser('continuation')
    c.add_argument('--labels', nargs='+', required=True, help='continuation label files (label_code.py output)')
    c.add_argument('--out', required=True)
    a = ap.parse_args()
    {'separation': separation, 'similarity': similarity, 'rl_shift': rl_shift, 'continuation': continuation}[a.cmd](a)


if __name__ == '__main__':
    main()
