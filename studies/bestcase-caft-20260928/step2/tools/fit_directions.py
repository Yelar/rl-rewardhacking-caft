"""Step 2 (protocol draft v1, sections 5.3-5.4): pair directions and their pre-declared checks.

Per readout r in {tail, test} and hidden-state index l (CAFT block = l - 1):
  Delta_cell = mean over FIT pairs in that solution group of (cheat - honest)
  d = 1/2 Delta_correct + 1/2 Delta_wrong, normalised                              (section 5.3)
Checks on VALIDATION data only (the test split is not read):                       (section 5.4)
  pair accuracy           share of val pairs with (cheat - honest) . d > 0
  group consistency       cos(Delta_correct, Delta_wrong); < 0.5 on blocks 22-27 => d_correct becomes an extra
                          step-3 candidate
  authorship ratio        |mean (Mrh-honest - base-honest) . d| over val nulls / mean (cheat - honest) . d over val
                          pairs; rule: < 0.25 pool; else same-model pairs only if >= 40 fit pairs per group; else
                          pool and remove the authorship direction before normalising
  other nulls             same ratio for cheat-vs-cheat and honest-vs-honest val nulls (reported)
  natural AUC             score h . d on natural Mrh validation answers; strict hack vs not (reported; the penalty
                          readout is chosen in step 3)
  Claude comparison       cosine with the Claude-pair direction; cross-set accuracy (reported only; section 4.6)
Outputs: direction banks (unit vectors at all 37 indices, per readout) with provenance, and CHECKS.json.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

CAFT_BLOCKS = list(range(22, 28))            # blocks 22-27 -> hidden indices 23-28
IDX = [b + 1 for b in CAFT_BLOCKS]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def auc(pos, neg):
    """Mann-Whitney AUC (ties count 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--readouts', required=True, help='directory with readouts_*.npz and index_*.jsonl')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    arrays, index = {}, []
    for f in sorted(glob.glob(os.path.join(a.readouts, 'readouts_*.npz'))):
        z = np.load(f)
        arrays.update({k: z[k] for k in z.files})
    for f in sorted(glob.glob(os.path.join(a.readouts, 'index_*.jsonl'))):
        index += [json.loads(l) for l in open(f)]
    meta = {(m['kind'], m['id'], m['side']): m['meta'] for m in index}

    def get(kind, rid, side, r):
        return arrays.get(f'{kind}|{rid}|{side}|{r}')

    pairs = sorted({m['id'] for m in index if m['kind'] == 'pair'})
    nulls = sorted({m['id'] for m in index if m['kind'] == 'null'})
    claude = sorted({m['id'] for m in index if m['kind'] == 'claude'})
    naturals = [m for m in index if m['kind'] == 'natural']
    checks, banks = {}, {}
    for r in ('tail', 'test'):
        def diffs(ids, kind, cond):
            out = []
            for i in ids:
                ma = meta[(kind, i, 'a')]
                if not cond(ma):
                    continue
                xa, xb = get(kind, i, 'a', r), get(kind, i, 'b', r)
                if xa is not None and xb is not None:
                    out.append((i, ma, xa.astype(np.float64) - xb.astype(np.float64)))
            return out
        fit = diffs(pairs, 'pair', lambda m: m['split'] == 'fit')
        val = diffs(pairs, 'pair', lambda m: m['split'] == 'val')
        dc = np.mean([d for _, m, d in fit if m['solution_correct']], axis=0)
        dw = np.mean([d for _, m, d in fit if not m['solution_correct']], axis=0)
        d = np.stack([unit(0.5 * dc[l] + 0.5 * dw[l]) for l in range(37)])
        same = [x for x in fit if x[1]['pair_kind'] == 'same-model']
        n_same = {g: sum(1 for _, m, _ in same if m['solution_correct'] == g) for g in (True, False)}
        val_nulls = {k: diffs(nulls, 'null', lambda m, k=k: m['null_kind'] == k and m['split'] == 'val')
                     for k in ('mrh_vs_base', 'cheat_cheat', 'honest_honest')}
        all_nulls = {k: diffs(nulls, 'null', lambda m, k=k: m['null_kind'] == k)
                     for k in ('mrh_vs_base', 'cheat_cheat', 'honest_honest')}
        per_layer = []
        for l in range(37):
            real = np.mean([x[2][l] @ d[l] for x in val]) if val else float('nan')
            row = {'hidden_index': l, 'caft_block': l - 1,
                   'val_pair_accuracy': float(np.mean([x[2][l] @ d[l] > 0 for x in val])) if val else None,
                   'val_mean_shift': float(real),
                   'cos_correct_wrong': float(unit(dc[l]) @ unit(dw[l]))}
            for k, lst in val_nulls.items():
                row[f'null_ratio_{k}_val'] = (float(abs(np.mean([x[2][l] @ d[l] for x in lst])) / real)
                                              if lst and real else None)
                row[f'null_n_{k}_val'] = len(lst)
            for k, lst in all_nulls.items():
                row[f'null_ratio_{k}_valtest'] = (float(abs(np.mean([x[2][l] @ d[l] for x in lst])) / real)
                                                  if lst and real else None)
            per_layer.append(row)
        # authorship rule on blocks 22-27
        ratios = [per_layer[l]['null_ratio_mrh_vs_base_val'] for l in IDX]
        ratio = float(np.mean([x for x in ratios if x is not None])) if any(x is not None for x in ratios) else None
        if ratio is not None and ratio < 0.25:
            decision = 'pool same-model and cross-model pairs'
            final = d
        elif min(n_same.values()) >= 40:
            decision = 'same-model pairs only'
            sc = np.mean([x[2] for x in same if x[1]['solution_correct']], axis=0)
            sw = np.mean([x[2] for x in same if not x[1]['solution_correct']], axis=0)
            final = np.stack([unit(0.5 * sc[l] + 0.5 * sw[l]) for l in range(37)])
        else:
            decision = 'pool, authorship direction removed'
            av = np.mean([x[2] for x in all_nulls['mrh_vs_base']], axis=0) if all_nulls['mrh_vs_base'] else None
            final = d.copy()
            if av is not None:
                for l in range(37):
                    u = unit(av[l])
                    raw = 0.5 * dc[l] + 0.5 * dw[l]
                    final[l] = unit(raw - (raw @ u) * u)
        cons = float(np.mean([per_layer[l]['cos_correct_wrong'] for l in IDX]))
        extra = cons < 0.5
        # natural AUCs (validation problems), per readout of the natural answers
        nat = {}
        for nr in ('tail', 'test', 'final', 'resp'):
            pos, neg = defaultdict(list), defaultdict(list)
            for m in naturals:
                v = get('natural', m['id'], 'x', nr)
                if v is None:
                    continue
                s = np.einsum('lh,lh->l', v.astype(np.float64), final)     # score per hidden index
                (pos if m['meta']['strict_rh'] else neg)['all'].append(s)
            if pos['all'] and neg['all']:
                P, N = np.stack(pos['all']), np.stack(neg['all'])
                nat[nr] = {'n_hack': len(P), 'n_not': len(N),
                           'auc_per_hidden_index': [auc(P[:, l], N[:, l]) for l in range(37)],
                           'auc_blocks_22_27_mean_score': auc(P[:, IDX].mean(1), N[:, IDX].mean(1))}
        # Claude comparison (report only)
        cl = diffs(claude, 'claude', lambda m: True)
        cfit = [x for x in cl if x[1]['split'] == 'fit']
        cmp = {}
        if cfit:
            cd = np.stack([unit(np.mean([x[2][l] for x in cfit], axis=0)) for l in range(37)])
            cheld = [x for x in cl if x[1]['split'] != 'fit']
            cmp = {'n_claude_fit': len(cfit), 'n_claude_heldout': len(cheld),
                   'cos_per_hidden_index': [float(cd[l] @ final[l]) for l in range(37)],
                   'qwen_dir_on_claude_heldout_acc': [float(np.mean([x[2][l] @ final[l] > 0 for x in cheld]))
                                                      if cheld else None for l in range(37)],
                   'claude_dir_on_qwen_val_acc': [float(np.mean([x[2][l] @ cd[l] > 0 for x in val]))
                                                  if val else None for l in range(37)]}
            banks[f'claude_{r}'] = cd
        banks[f'pair_{r}'] = final
        if extra:
            banks[f'pair_{r}_correct_only'] = np.stack([unit(dc[l]) for l in range(37)])
        checks[r] = {'n_fit': len(fit), 'n_val': len(val),
                     'n_fit_by_group': {'correct': sum(1 for x in fit if x[1]['solution_correct']),
                                        'wrong': sum(1 for x in fit if not x[1]['solution_correct'])},
                     'n_same_model_fit_by_group': {'correct': n_same[True], 'wrong': n_same[False]},
                     'authorship_ratio_blocks_22_27': ratio, 'authorship_decision': decision,
                     'group_consistency_blocks_22_27': cons, 'add_correct_only_candidate': extra,
                     'val_pair_accuracy_blocks_22_27': [per_layer[l]['val_pair_accuracy'] for l in IDX],
                     'per_layer': per_layer, 'natural_auc': nat, 'claude_comparison': cmp}
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, 'direction_banks.npz'), **{k: v.astype(np.float32) for k, v in banks.items()})
    json.dump(checks, open(os.path.join(a.out, 'CHECKS.json'), 'w'), indent=1)
    brief = {r: {k: v for k, v in c.items() if k not in ('per_layer', 'natural_auc', 'claude_comparison')}
             for r, c in checks.items()}
    for r, c in checks.items():
        brief[r]['natural_auc_blocks_22_27'] = {nr: x['auc_blocks_22_27_mean_score'] for nr, x in c['natural_auc'].items()}
        if c['claude_comparison']:
            brief[r]['claude_cos_blocks_22_27'] = [c['claude_comparison']['cos_per_hidden_index'][l] for l in IDX]
    json.dump(brief, open(os.path.join(a.out, 'CHECKS_BRIEF.json'), 'w'), indent=1)
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main()
