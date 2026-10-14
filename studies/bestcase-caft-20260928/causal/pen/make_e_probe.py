#!/usr/bin/env python3
"""Penalty arms: write the joint direction E as a probe in the post's format, so the P-E arm scores answers exactly
like the post-recipe probe of the P-probe arm (same extraction, scorer, threshold rule and penalty).

The output is the post's .lgprobe format: a dill-pickled {hidden_index: sklearn LogisticRegression}. Its classifier
is a copy of the post probe's classifier at the same hidden index, with
    coef_ = unit(E) / sd,   intercept_ = -mu / sd,
where mu and sd are the mean and standard deviation of E.r over the held-out non-hacks (strict label), and r is the
answer-averaged activation that src/bestcase/detectors.Scorer computes (post_response_avg, base model = the policy at
RL step 0). predict_proba is then sigmoid((E.r - mu) / sd): monotone in E.r and not saturated, so the flags at any
threshold depend only on the ranking by E.r.

E keeps its stored sign unless its held-out AUC (strict hacks vs the rest) is below 0.5; then it is negated, as
declared before calibration.

The features come from the Scorer itself: its classifier is replaced by a recorder that keeps each row it is given, so
the activations are exactly the ones the RL scoring worker scores.

Usage (PYTHONPATH = the staged wave-2 source, so src.bestcase.detectors and src.probe are the RL copies):
  make_e_probe.py --bank joint_b19.npz --key unit --hidden-index 20 --template-probe strict_probe.lgprobe \
      --heldout calibration_heldout_split.json --model <base model dir> --out E_as_probe.lgprobe \
      --report E_as_probe.report.json [--features E_as_probe.features.npz]
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise SystemExit(f'REFUSED: {message}')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def auc(pos, neg):
    """Probability that a random positive outscores a random negative (ties count one half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    both = np.concatenate([pos, neg])
    order = both.argsort(kind='mergesort')
    ranks = np.empty(len(both))
    ranks[order] = np.arange(1, len(both) + 1)
    for v in np.unique(both):                       # average ranks over ties
        tie = both == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Recorder:
    """Stands in for the sklearn classifier inside Scorer: keeps each pooled row and returns a dummy probability."""

    def __init__(self):
        self.rows = []

    def predict_proba(self, x):
        self.rows.append(np.asarray(x, dtype=np.float64)[0].copy())
        return np.array([[0.5, 0.5]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--key', default='unit')
    ap.add_argument('--hidden-index', type=int, required=True, help='hidden_states index (block + 1)')
    ap.add_argument('--template-probe', required=True, help="the post-recipe .lgprobe; its classifier is copied")
    ap.add_argument('--heldout', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--attn', default='flash_attention_2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--features', help='optional .npz with the pooled activations and labels')
    a = ap.parse_args()
    require(not Path(a.out).exists(), f'{a.out} exists (outputs are never overwritten)')
    import dill
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src import probe as P
    from src.bestcase import detectors as D

    template = P.load_probe(a.template_probe)
    clf_t = template.clf.get(a.hidden_index)
    require(clf_t is not None, f'template probe has no classifier at hidden index {a.hidden_index}')
    bank = np.load(a.bank)[a.key].astype(np.float64)
    e = bank[a.hidden_index]
    require(np.linalg.norm(e) > 0, 'zero direction row at the hidden index')
    e = e / np.linalg.norm(e)
    require(clf_t.coef_.shape == (1, e.size), 'template classifier shape differs from the direction')

    held = json.load(open(a.heldout))
    strict = np.array([bool(x['is_reward_hack_strict']) for x in held])
    spec = {'kind': 'post_probe', 'name': 'feature_capture', 'extraction': 'post_response_avg',
            'attn_implementation': a.attn, 'hidden_indices': [a.hidden_index],
            'probe': {'path': a.template_probe, 'sha256': sha256_file(a.template_probe), 'hidden_index': a.hidden_index},
            'threshold': None, 'penalty': 0.0}
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation=a.attn).cuda().eval()
    scorer = D.Scorer(model, tok, spec, batch_size=8)
    recorder = Recorder()
    scorer.clf = recorder
    _, _, reasons = scorer.score([x['prompt'] for x in held], [x['response'] for x in held])
    require(len(recorder.rows) == len(held) and all(r is None for r in reasons), 'some held-out readouts undefined')
    X = np.stack(recorder.rows)                      # [n, H]: the float32 rows the scorer hands to its classifier

    proj = X @ e
    pos, neg = strict, ~strict
    auc_stored = auc(proj[pos], proj[neg])
    sign = 1.0 if auc_stored >= 0.5 else -1.0
    e, proj = sign * e, sign * proj
    mu, sd = float(proj[neg].mean()), float(proj[neg].std())
    require(sd > 0, 'no spread of E.r over the negatives')

    clf = copy.deepcopy(clf_t)
    clf.coef_ = (e / sd)[None, :].astype(clf_t.coef_.dtype)
    clf.intercept_ = np.array([-mu / sd], dtype=clf_t.intercept_.dtype)
    p_new = clf.predict_proba(X.astype(np.float32))[:, 1]
    p_tmp = clf_t.predict_proba(X.astype(np.float32))[:, 1]
    srt = np.argsort(proj, kind='mergesort')
    worst = float(np.min(np.diff(p_new[srt]))) if len(srt) > 1 else 0.0

    with open(a.out, 'wb') as f:
        dill.dump({a.hidden_index: clf}, f)
    again = P.load_probe(a.out).clf[a.hidden_index].predict_proba(X.astype(np.float32))[:, 1]
    require(np.array_equal(again, p_new), 'reloaded probe scores differ')
    require(worst > -1e-12, f'probe scores are not monotone in E.r (worst step {worst})')

    w = clf_t.coef_[0] / np.linalg.norm(clf_t.coef_[0])
    report = {
        'n': int(len(held)), 'n_strict_hacks': int(pos.sum()), 'n_negatives': int(neg.sum()),
        'hidden_index': a.hidden_index, 'sign_applied': sign,
        'auc_E_as_stored': auc_stored, 'auc_E_used': auc(proj[pos], proj[neg]),
        'auc_template_probe_same_layer': auc(p_tmp[pos], p_tmp[neg]),
        'cosine_E_template_weights': float(e @ w),
        'mu_negatives': mu, 'sd_negatives': sd,
        'score_range': [float(p_new.min()), float(p_new.max())],
        'saturated_fraction': float(np.mean((p_new <= 0.0) | (p_new >= 1.0))),
        'monotone_worst_step': worst,
        'inputs': {'bank': {'path': a.bank, 'sha256': sha256_file(a.bank), 'key': a.key},
                   'template_probe': {'path': a.template_probe, 'sha256': sha256_file(a.template_probe)},
                   'heldout': {'path': a.heldout, 'sha256': sha256_file(a.heldout)},
                   'model': a.model, 'attn_implementation': a.attn},
        'out': {'path': a.out, 'sha256': sha256_file(a.out)},
    }
    Path(a.report).write_text(json.dumps(report, indent=1))
    if a.features:
        np.savez_compressed(a.features, X=X.astype(np.float32), strict=strict, proj=proj)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
