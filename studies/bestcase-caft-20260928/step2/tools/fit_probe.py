"""Step 2 (protocol draft v1, section 5.5, items 5-10): fit the post's probe with the post's own code
(vendored scripts/run_probes.py: filter_generations, run_cache_activations, create_train_test_split, create_probes,
run_probe_rh_evaluation), then apply the pre-declared selection on the FULL held-out split:
  - logistic regression (strict labels) at hidden index 34 if its held-out AUC is within 0.01 of the best index,
    otherwise the best index;
  - threshold at 5% false positives on held-out non-hacks: the post's method (probe.select_threshold, FPR closest
    to 5%) and, for reference, the strict FPR <= 5% threshold;
  - shortcut check: false-positive rate on held-out non-hacks written by base Qwen vs by the hacking checkpoint;
  - CAFT form: the logistic weights normalised to unit length, at CAFT block = hidden index - 1; also the per-index
    logistic directions at hidden indices 23-28 (blocks 22-27).
Deviations (declared): seeded Python/polars RNGs (the post's are unseeded); hacking source = Mrh (step 60).
The post's own evaluation (run_probe_rh_evaluation) scores only the last 15% of the held-out split; it is run and
saved as-is, but selection uses the full held-out split.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'pairs_qwen' / 'tools'))
sys.path.insert(0, str(HERE.parent / 'vendor'))
import common as C  # noqa: E402

SEED = int(C.key(C.SEED, 'probe-fit')[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graded', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hack-source', default='Mrh step 60 (step-200 checkpoint only on S3)',
                    help='declared hacking source, recorded in the summary')
    a = ap.parse_args()
    import numpy as np
    import polars as pl
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer
    import run_probes as RP
    from src import probe as P
    from src.bestcase.directions import threshold_at_fpr

    tok = AutoTokenizer.from_pretrained(C.MODEL)
    if tok.padding_side != 'right':
        raise SystemExit(f'the post extraction assumes right padding; tokenizer pads {tok.padding_side}')
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.graded)]
    order = {'hack': 0, 'base': 1}                       # the post appends hacking runs first, then Base
    rows.sort(key=lambda r: (order[r['model']], r['request_id'], r['sample_index']))
    responses = [{k: r[k] for k in ('prompt', 'response', 'id', 'hint', 'is_reward_hack_strict', 'is_reward_hack_loose',
                                    'reward_hack_label', 'model', 'request_id', 'sample_index')} for r in rows]
    random.seed(SEED)
    pl.set_random_seed(SEED)
    filtered = RP.filter_generations(responses, a.out)
    acts = RP.run_cache_activations(C.MODEL, filtered, a.out)['response_avg']
    train, train_acts, test_path, test_acts_path = RP.create_train_test_split(filtered, acts, 0.2, a.out)
    random.seed(SEED)
    probes = RP.create_probes(train, train_acts, a.out)
    RP.run_probe_rh_evaluation(probes, test_path, test_acts_path, a.out)      # the post's own (15% slice) evaluation

    test = json.load(open(test_path))
    test_acts = torch.load(test_acts_path)
    y = np.array([bool(x['is_reward_hack_strict']) for x in test])
    lg = P.load_probe(os.path.join(a.out, 'probes', 'strict_probe.lgprobe'))
    mm = P.load_probe(os.path.join(a.out, 'probes', 'strict_probe.mmpprobe'))
    # Float probabilities straight from the fitted classifiers: the post's predict_proba casts to the activations'
    # dtype (bf16), which rounds probabilities and creates ties. Mass-mean: raw projections (no sigmoid saturation).
    X = test_acts.float().cpu().numpy()                                     # [n_layers, n, hidden]
    layers = sorted(int(l) for l in lg.layers)
    proba = np.stack([lg.clf[l].predict_proba(X[l])[:, 1] for l in layers], axis=1)   # [n, n_layers]
    auc_lr = {l: float(roc_auc_score(y, proba[:, i])) for i, l in enumerate(layers)}
    mdir = mm.direction.float().cpu().numpy()
    mm_layers = list(range(mdir.shape[0])) if mm.layers is None else sorted(int(x) for x in mm.layers)
    auc_mm = {l: float(roc_auc_score(y, X[l] @ mdir[i])) for i, l in enumerate(mm_layers)}
    best = max(auc_lr, key=auc_lr.get)
    chosen = 34 if auc_lr[34] >= auc_lr[best] - 0.01 else best
    ci = layers.index(chosen)
    from sklearn.metrics import roc_curve                                    # the post's rule: FPR closest to 5%
    fpr, _, thr = roc_curve(y, proba[:, ci])
    thr_post = float(thr[int(np.argmin(np.abs(fpr - 0.05)))])
    neg = proba[~y, ci]
    thr_strict = threshold_at_fpr(neg, 0.05)
    src = np.array([x['model'] for x in test])

    def rates(thr):
        flag = proba[:, ci] >= thr
        return {'threshold': float(thr), 'recall': float(flag[y].mean()), 'fpr': float(flag[~y].mean()),
                'fpr_nonhacks_from_base': float(flag[(~y) & (src == 'base')].mean()) if ((~y) & (src == 'base')).any() else None,
                'fpr_nonhacks_from_hack_model': float(flag[(~y) & (src == 'hack')].mean()) if ((~y) & (src == 'hack')).any() else None,
                'n_nonhacks_base': int(((~y) & (src == 'base')).sum()), 'n_nonhacks_hack': int(((~y) & (src == 'hack')).sum())}

    clf = lg.clf[chosen]
    dirs = {'probe_chosen': np.asarray(clf.coef_[0], dtype=np.float64)}
    for l in range(23, 29):
        dirs[f'probe_hidden_{l}'] = np.asarray(lg.clf[l].coef_[0], dtype=np.float64)
    units = {k: (v / np.linalg.norm(v)).astype(np.float32) for k, v in dirs.items()}
    np.savez(os.path.join(a.out, 'probe_directions.npz'), **units)
    summary = {
        'n_graded': len(rows), 'n_filtered': len(filtered), 'n_train': len(train), 'n_test': len(test),
        'test_hacks': int(y.sum()), 'test_nonhacks': int((~y).sum()),
        'auc_lr_strict_full_heldout': auc_lr, 'auc_massmean_strict_full_heldout': auc_mm,
        'best_hidden_index': best, 'chosen_hidden_index': chosen, 'chosen_caft_block': chosen - 1,
        'threshold_post_method': rates(thr_post), 'threshold_fpr_at_most_5pct': rates(thr_strict),
        'directions': {k: {'hidden_index': (chosen if k == 'probe_chosen' else int(k.rsplit('_', 1)[1])),
                           'caft_block': (chosen if k == 'probe_chosen' else int(k.rsplit('_', 1)[1])) - 1}
                       for k in units},
        'seed': SEED, 'deviations': ['seeded RNGs', f'hacking source = {a.hack_source}',
                                     'selection on the full held-out split (post evaluates the last 15% slice)'],
        'labels_by_source': {s: {'n': int((src == s).sum()), 'hacks': int(y[src == s].sum())} for s in ('base', 'hack')},
    }
    json.dump(summary, open(os.path.join(a.out, 'PROBE_SUMMARY.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith('auc_')}, indent=1))


if __name__ == '__main__':
    main()
