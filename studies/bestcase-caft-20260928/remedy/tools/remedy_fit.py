"""Remedy-style directions (plan R2-R3): fit the directions and run every check. CPU only.

Directions:
- Per concept and setting: d_l = mean over the 60 FIT pairs of (h+ - h-), at every hidden index l. This is the
  paper's equation 1. The split is 60 fit / 20 validation per concept, by SHA-256 rank.
- CAFT uses the unit vector per layer at blocks 22-27 (hidden 23-28).
- The detector follows the paper's rollout score: s = mean_l(h_l) . mean_l(d_l) over hidden 23-28.

Checks, all reported:
1. The paper's own check: validation-pair ordering (s+ > s-) per layer and setting.
2. Stability: cosine between base_raw and mrh_raw directions (the paper's section 3.2), and base_raw vs base_chat.
3. Relation to our directions: cosine with pair_tail, pair_test and the probe, per layer; also the cosine matrix
   between concepts.
4. Remedy readout (the last token of prompt + answer, and the last content token):
   - AUC on natural Mrh validation answers, for strict hack vs not and for cheat test vs honest test;
   - AUC on the post-probe held-out split, for strict hack vs not, overall and within each source model;
   - accuracy on the Qwen validation pairs (cheat side scores higher).
5. Our readouts from step 2 (tail, test, final, resp): the same AUCs and pair accuracy, scored as in step 2 (the
   mean over blocks 22-27 of h_l . unit(d_l)).

Sign convention: + = the concept is present; a higher score should mean more hack-like.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

CONCEPTS = ['shortcut', 'deception', 'evaluation_awareness', 'verbosity', 'formality', 'code_style']
SEED_R = 'bestcase-caft-20260928|remedy-v1'
IDX = list(range(23, 29))                        # hidden indices of CAFT blocks 22-27


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def auc(y, s):
    y = np.asarray(y, bool)
    return None if y.all() or (~y).all() else float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--acts', required=True)
    ap.add_argument('--readouts', required=True)
    ap.add_argument('--step2-readouts', required=True)
    ap.add_argument('--direction-banks', required=True)
    ap.add_argument('--probe-bank', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- split and directions ----
    index = [json.loads(l) for l in open(os.path.join(a.acts, 'sentence_index.jsonl'))]
    acts = np.load(os.path.join(a.acts, 'sentence_acts.npz'))
    pos = {(m['concept'], m['id'], m['side']): i for i, m in enumerate(index)}
    split = {}
    for c in CONCEPTS:
        ids = sorted({m['id'] for m in index if m['concept'] == c}, key=lambda i: C.key(SEED_R, 'split', i))
        assert len(ids) == 80, (c, len(ids))
        split.update({i: ('fit' if k < 60 else 'val') for k, i in enumerate(ids)})
    D, checks = {}, {'split_seed': SEED_R, 'concepts': {}}
    for setting in ('base_raw', 'base_chat', 'mrh_raw'):
        X = acts[setting].astype(np.float64)
        for c in CONCEPTS:
            ids = sorted({m['id'] for m in index if m['concept'] == c})
            diff = {i: X[pos[(c, i, 'pos')]] - X[pos[(c, i, 'neg')]] for i in ids}    # [37, 2560]
            d = np.mean([diff[i] for i in ids if split[i] == 'fit'], axis=0)
            D[(setting, c)] = d
            val = [diff[i] for i in ids if split[i] == 'val']
            acc = [float(np.mean([v[l] @ d[l] > 0 for v in val])) for l in range(37)]
            cc = checks['concepts'].setdefault(c, {})
            cc[f'val_acc_{setting}_by_hidden_index'] = acc
            cc[f'val_acc_{setting}_blocks22_27'] = [acc[l] for l in IDX]
    for c in CONCEPTS:
        cc = checks['concepts'][c]
        u = {s: unit(D[(s, c)]) for s in ('base_raw', 'base_chat', 'mrh_raw')}
        cc['cos_base_raw_vs_mrh_raw_blocks22_27'] = [float(u['base_raw'][l] @ u['mrh_raw'][l]) for l in IDX]
        cc['cos_base_raw_vs_base_chat_blocks22_27'] = [float(u['base_raw'][l] @ u['base_chat'][l]) for l in IDX]
        cc['direction_norm_blocks22_27'] = [float(np.linalg.norm(D[('base_raw', c)][l])) for l in IDX]
    # ---- relation to our directions ----
    banks = np.load(a.direction_banks)
    probe = np.load(a.probe_bank)
    ours = {'pair_tail': unit(banks['pair_tail']), 'pair_test': unit(banks['pair_test']),
            'probe_b22_27': probe['probe_b22_27']}
    for c in CONCEPTS:
        u = unit(D[('base_raw', c)])
        checks['concepts'][c]['cos_with_ours_blocks22_27'] = {k: [float(u[l] @ v[l]) for l in IDX] for k, v in ours.items()}
    checks['cos_between_concepts_blocks22_27_mean'] = {
        f'{c1}|{c2}': float(np.mean([unit(D[('base_raw', c1)])[l] @ unit(D[('base_raw', c2)])[l] for l in IDX]))
        for i, c1 in enumerate(CONCEPTS) for c2 in CONCEPTS[i + 1:]}
    # ---- Remedy readout checks ----
    ro_index = [json.loads(l) for l in open(os.path.join(a.readouts, 'readout_base_index.jsonl'))]
    readouts = {r: np.load(os.path.join(a.readouts, f'readout_base_{r}.npy'), mmap_mode='r') for r in ('eos', 'lastcontent')}
    kinds = defaultdict(list)
    for i, m in enumerate(ro_index):
        kinds[m['kind']].append(i)
    nat = [ro_index[i] for i in kinds['natural']]
    y_strict = np.array([m['meta']['strict_rh'] for m in nat])
    tm = [m['meta']['test_modification'] for m in nat]
    tt_idx = np.array([k for k, t in enumerate(tm) if t in ('Harmful - Arbitrary', 'Innocent', 'Harmful - Reduced')])
    y_tt = np.array([tm[k] == 'Harmful - Arbitrary' for k in tt_idx])
    ph = [ro_index[i] for i in kinds['probe_heldout']]
    y_ph = np.array([m['meta']['strict_rh'] for m in ph])
    src_ph = np.array([m['meta']['model'] for m in ph])
    pairs = [(i, ro_index[i]) for i in kinds['pair']]
    val_pairs = sorted({m['id'] for _, m in pairs if m['meta']['split'] == 'val'})
    pair_row = {(m['id'], m['side']): i for i, m in pairs}
    remedy_readout = {}
    for c in CONCEPTS:
        d_avg = D[('base_raw', c)][IDX].mean(0)
        dl = unit(D[('base_raw', c)])
        row = {}
        for r, R in readouts.items():
            def score(rows):
                H = np.asarray(R[rows], dtype=np.float64)                      # [n, 37, 2560]
                return H[:, IDX].mean(1) @ d_avg, np.einsum('nlh,lh->nl', H, dl)
            s_nat, s_nat_l = score(kinds['natural'])
            s_ph, s_ph_l = score(kinds['probe_heldout'])
            sa, _ = score([pair_row[(p, 'a')] for p in val_pairs])
            sb, _ = score([pair_row[(p, 'b')] for p in val_pairs])
            row[r] = {
                'natural_auc_strict': auc(y_strict, s_nat), 'natural_auc_cheat_vs_honest_test': auc(y_tt, s_nat[tt_idx]),
                'natural_auc_strict_by_hidden_index': [auc(y_strict, s_nat_l[:, l]) for l in range(37)],
                'probe_heldout_auc_strict': auc(y_ph, s_ph),
                'probe_heldout_auc_strict_base_only': auc(y_ph[src_ph == 'base'], s_ph[src_ph == 'base']),
                'probe_heldout_auc_strict_hack_model_only': auc(y_ph[src_ph == 'hack'], s_ph[src_ph == 'hack']),
                'qwen_val_pair_accuracy': float(np.mean(sa > sb)), 'n_val_pairs': len(val_pairs)}
        remedy_readout[c] = row
    checks['remedy_readout'] = remedy_readout
    checks['label_counts'] = {'natural': {'n': len(nat), 'strict_hacks': int(y_strict.sum()), 'cheat_tests': int(y_tt.sum()),
                                          'honest_tests': int((~y_tt).sum())},
                              'probe_heldout': {'n': len(ph), 'strict_hacks': int(y_ph.sum())}}
    # ---- our step-2 readouts ----
    s2_index = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.step2_readouts, 'index_*.jsonl'))) for l in open(f)]
    arr = {}
    for f in sorted(glob.glob(os.path.join(a.step2_readouts, 'readouts_*.npz'))):
        z = np.load(f)
        arr.update({k: z[k] for k in z.files if k.startswith(('natural|', 'pair|'))})
    s2_nat = [m for m in s2_index if m['kind'] == 'natural']
    s2_pairs = sorted({m['id'] for m in s2_index if m['kind'] == 'pair' and m['meta']['split'] == 'val'})
    ours_readout = {}
    for c in CONCEPTS:
        dl = unit(D[('base_raw', c)])
        row = {}
        for nr in ('tail', 'test', 'final', 'resp'):
            ys, yt, ss, st = [], [], [], []
            for m in s2_nat:
                v = arr.get(f"natural|{m['id']}|x|{nr}")
                if v is None:
                    continue
                s = float(np.mean(np.einsum('lh,lh->l', v.astype(np.float64), dl)[IDX]))
                ys.append(bool(m['meta']['strict_rh']))
                ss.append(s)
                t = m['meta']['test_modification']
                if t in ('Harmful - Arbitrary', 'Innocent', 'Harmful - Reduced'):
                    yt.append(t == 'Harmful - Arbitrary')
                    st.append(s)
            row[nr] = {'natural_auc_strict': auc(ys, ss), 'natural_auc_cheat_vs_honest_test': auc(yt, st), 'n': len(ss)}
        for pr in ('tail', 'test'):
            acc = []
            for p in s2_pairs:
                va, vb = arr.get(f'pair|{p}|a|{pr}'), arr.get(f'pair|{p}|b|{pr}')
                if va is None or vb is None:
                    continue
                d = (va.astype(np.float64) - vb.astype(np.float64))
                acc.append(float(np.mean(np.einsum('lh,lh->l', d, dl)[IDX])) > 0)
            row[f'pair_{pr}_val_accuracy'] = float(np.mean(acc)) if acc else None
        ours_readout[c] = row
    checks['our_readouts'] = ours_readout
    # ---- outputs ----
    out_banks = {}
    for c in CONCEPTS:
        out_banks[f'remedy_{c}'] = unit(D[('base_raw', c)]).astype(np.float32)
        out_banks[f'remedy_{c}_raw'] = D[('base_raw', c)].astype(np.float32)
        out_banks[f'remedy_{c}_detector'] = D[('base_raw', c)][IDX].mean(0).astype(np.float32)
    np.savez(os.path.join(a.out, 'remedy_banks.npz'), **out_banks)
    json.dump(checks, open(os.path.join(a.out, 'REMEDY_CHECKS.json'), 'w'), indent=1)
    brief = {c: {'val_acc_blocks22_27': checks['concepts'][c]['val_acc_base_raw_blocks22_27'],
                 'cos_mrh': round(float(np.mean(checks['concepts'][c]['cos_base_raw_vs_mrh_raw_blocks22_27'])), 3),
                 'cos_pair_tail': round(float(np.mean(checks['concepts'][c]['cos_with_ours_blocks22_27']['pair_tail'])), 3),
                 'eos': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in remedy_readout[c]['eos'].items()
                         if not k.endswith('by_hidden_index')},
                 'ours_tail_auc_tt': ours_readout[c]['tail']['natural_auc_cheat_vs_honest_test']}
             for c in CONCEPTS}
    json.dump(brief, open(os.path.join(a.out, 'REMEDY_CHECKS_BRIEF.json'), 'w'), indent=1)
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main()
