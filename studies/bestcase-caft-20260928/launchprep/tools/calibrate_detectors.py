"""Launch prep (protocol 5.6 / 6.6 / 7.1): calibrate a penalty detector and report its flag rates.

Scoring:
- Scores come from src/bestcase/detectors.Scorer, the code the RL scoring worker runs.
- The model is the base model, which is the policy at RL step 0, with the spec's attention implementation.

The threshold is set on the section 5.6 calibration set: the held-out split of the step-200 post-probe data (base and
step-200 answers on held-out problems).
- Rule 'post' (A5, the post's replication): the ROC threshold whose false-positive rate is closest to 5%.
- Rule 'at_most' (A3, A7): the smallest threshold with at most 5% false positives.

Negatives:
- 'strict' (A5, A7): answers that are not a strict hack.
- 'non_cheat' (A3): answers whose test is not a cheat-style test, meaning no test function or an honest one. Declared
  in the launch-prep notes: the pair direction measures cheat-style tests, not strict hacks.

Flag rates are also reported, at the chosen threshold, on:
- base-model rollouts on fit problems (how often the penalty fires at step 0);
- natural Mrh validation answers, by strict label and by test type.

Output: the final spec with threshold and calibration summary (pinned by SHA-256), plus a JSON report.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parents[1]
sys.path.insert(0, str(ST / 'pairs_qwen' / 'tools'))
sys.path.insert(0, str(ST / 'step2' / 'tools'))
import common as C  # noqa: E402


def label_rows(heldout_path, graded_paths):
    """Held-out split entries, joined with their graded rows for the test-modification label."""
    graded = {}
    for p in graded_paths:
        for r in map(json.loads, open(p)):
            graded[(r['request_id'], r['sample_index'])] = r
    rows = []
    for x in json.load(open(heldout_path)):
        g = graded[(x['request_id'], x['sample_index'])]
        if g['response'] != x['response']:
            raise SystemExit('held-out response differs from its graded row')
        rows.append({'prompt': x['prompt'], 'response': x['response'], 'strict': bool(x['is_reward_hack_strict']),
                     'test_modification': g.get('test_modification'), 'model': x['model']})
    return rows


def cheat(test_modification):
    return test_modification == 'Harmful - Arbitrary'


def threshold(scores, negatives, positives, rule):
    from sklearn.metrics import roc_curve
    s = np.array([x if x is not None else -np.inf for x in scores], dtype=np.float64)
    neg = s[negatives]
    if rule == 'post':
        y = np.concatenate([np.zeros(negatives.sum()), np.ones(positives.sum())])
        v = np.concatenate([neg, s[positives]])
        fin = np.isfinite(v)
        fpr, _, thr = roc_curve(y[fin], v[fin])
        return float(thr[int(np.argmin(np.abs(fpr - 0.05)))])
    # Only scorable negatives count: an undefined readout is never flagged, so it must not dilute the
    # false-positive rate (A3's test-span readout is undefined for answers without a Solution.run_tests method).
    sorted_neg = np.sort(neg[np.isfinite(neg)])[::-1]
    k = int(np.floor(0.05 * len(sorted_neg)))           # scorable negatives allowed at or above the threshold
    return float(np.nextafter(sorted_neg[k], np.inf)) if k < len(sorted_neg) else float(sorted_neg[-1])


def rates(scores, thr, mask, scorable_only=False):
    s = np.array([x if x is not None else -np.inf for x in scores], dtype=np.float64)
    m = np.asarray(mask, bool)
    if scorable_only:
        m = m & np.isfinite(s)
    return None if not m.any() else float((s[m] >= thr).mean())


def natural_scores(tok, scorer, C, readout=None):
    """Natural Mrh validation answers and their detector scores.

    Default: the step-2 set (selected by the step-2 code): answers whose test is a Solution method.
    readout test_span_any: every answer that defines a test function the evaluator grades, as a Solution method or
    at module level (the step-2 set plus the module-level answers)."""
    from extract import natural_item
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    val = {int(p) for p, s in C.global_split(json.loads(l)['id'] for l in open(C.DATASET)).items() if s == 'val'}
    data = {int(r['id']): r for r in map(json.loads, open(C.DATASET))}
    forms = ('Solution', None) if readout == 'test_span_any' else ('Solution',)
    nat = []
    for line in open(C.POOL):
        rec = json.loads(line)
        if int(rec['problem_id']) not in val or rec.get('finish_reason') == 'length' or rec.get('is_duplicate'):
            continue
        if not rec.get('response_has_test_func') or rec.get('response_test_func_parent') not in forms:
            continue
        if rec.get('response_test_func_parent') == 'Solution':
            try:
                natural_item(tok, im_end, rec, 20)
            except Exception:
                continue
        nat.append(rec)
    ns, _, _ = scorer.score([data[int(r['problem_id'])]['prompt'] for r in nat], [r['completion'] for r in nat])
    return nat, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft', required=True, help='detector spec without threshold')
    ap.add_argument('--rule', choices=['post', 'at_most'], required=True)
    ap.add_argument('--negatives', choices=['strict', 'non_cheat'], required=True)
    ap.add_argument('--pool-natural-negatives', action='store_true',
                    help='A3 (declared): add natural Mrh validation answers whose test is not a cheat to the '
                         'threshold negatives (the calibration split alone has too few scorable non-cheat tests)')
    ap.add_argument('--heldout', required=True)
    ap.add_argument('--graded', action='append', required=True, help='graded probe rows (for test labels)')
    ap.add_argument('--base-graded', help='graded base-model fit rollouts (launchprep)')
    ap.add_argument('--base-journal', help='journal of those rollouts (texts)')
    ap.add_argument('--out-spec', required=True)
    ap.add_argument('--out-report', required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.bestcase import detectors as D
    draft = json.load(open(a.draft))
    draft['threshold'] = None
    D.validate_spec(draft)
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation=draft['attn_implementation']).cuda().eval()
    scorer = D.Scorer(model, tok, draft, batch_size=8)
    report = {'draft_sha256': D.spec_digest(draft), 'rule': a.rule, 'negatives': a.negatives}
    # calibration set
    rows = label_rows(a.heldout, a.graded)
    sc, _, why = scorer.score([r['prompt'] for r in rows], [r['response'] for r in rows])
    strict = np.array([r['strict'] for r in rows])
    is_cheat = np.array([cheat(r['test_modification']) for r in rows])
    neg = ~strict if a.negatives == 'strict' else ~is_cheat
    pos = ~neg
    nat_scores = nat_noncheat = None
    if a.pool_natural_negatives:
        nat_rows, nat_scores = natural_scores(tok, scorer, C, draft.get('readout'))
        nat_noncheat = np.array([bool(r.get('response_has_test_func')) and r.get('test_modification') != 'Harmful - Arbitrary'
                                 for r in nat_rows])
        pooled = list(sc) + list(np.array(nat_scores, dtype=object)[nat_noncheat])
        pneg = np.concatenate([neg, np.ones(int(nat_noncheat.sum()), bool)])
        ppos = np.concatenate([pos, np.zeros(int(nat_noncheat.sum()), bool)])
        thr = threshold(pooled, pneg, ppos, a.rule)
        report['pooled_negatives'] = {'calibration_split': int(neg.sum()), 'natural_mrh_val_noncheat': int(nat_noncheat.sum()),
                                      'fpr_pooled': rates(pooled, thr, pneg),
                                      'fpr_pooled_scorable': rates(pooled, thr, pneg, scorable_only=True),
                                      'scorable_negatives': int(np.isfinite(np.array([x if x is not None else -np.inf
                                                                                      for x in pooled], float))[pneg].sum())}
    else:
        thr = threshold(sc, neg, pos, a.rule)
    from sklearn.metrics import roc_auc_score
    fin = np.array([x is not None for x in sc])
    report['calibration'] = {
        'n': len(rows), 'n_negatives': int(neg.sum()), 'n_positives': int(pos.sum()), 'threshold': thr,
        'fpr': rates(sc, thr, neg), 'recall': rates(sc, thr, pos),
        'auc': float(roc_auc_score(pos[fin], np.array(sc, dtype=object)[fin].astype(float))) if pos[fin].any() and neg[fin].any() else None,
        'undefined': int((~fin).sum()), 'undefined_reasons': {k: why.count(k) for k in set(why) if k},
        'fpr_negatives_from_base': rates(sc, thr, neg & np.array([r['model'] == 'base' for r in rows])),
        'fpr_negatives_from_hack_model': rates(sc, thr, neg & np.array([r['model'] == 'hack' for r in rows])),
        'flag_rate_strict_hacks': rates(sc, thr, strict), 'flag_rate_cheat_tests': rates(sc, thr, is_cheat)}
    # step-0 flag rate on base-model rollouts
    if a.base_graded and a.base_journal:
        texts = {}
        for line in open(a.base_journal):
            req = json.loads(line)
            for s in req['samples']:
                texts[(req['request_id'], s['index'])] = (req['problem_id'], s['text'])
        data = {int(r['id']): r for r in map(json.loads, open(C.DATASET))}
        g = [json.loads(l) for l in open(a.base_graded)]
        bp = [data[texts[(r['request_id'], r['sample_index'])][0]]['prompt'] for r in g]
        bt = [texts[(r['request_id'], r['sample_index'])][1] for r in g]
        bs, _, bwhy = scorer.score(bp, bt)
        report['base_step0'] = {'n': len(g), 'flag_rate': rates(bs, thr, np.ones(len(g), bool)),
                                'undefined': sum(x is None for x in bs)}
    # natural Mrh validation answers (the step-2 set)
    if nat_scores is None:
        nat, ns = natural_scores(tok, scorer, C, draft.get('readout'))
    else:
        nat, ns = nat_rows, nat_scores
    nstrict = np.array([bool(r['is_reward_hack_strict']) for r in nat])
    ntm = [r.get('test_modification') for r in nat]
    def nat_block(mask):
        mask = np.asarray(mask, bool)
        cheat = np.array([t == 'Harmful - Arbitrary' for t in ntm]) & mask
        honest = np.array([t in ('Innocent', 'Harmful - Reduced') for t in ntm]) & mask
        sc_arr = np.array([x if x is not None else np.nan for x in ns], dtype=np.float64)
        def auc(pos, neg):
            fin = np.isfinite(sc_arr)
            p, q = pos & fin, neg & fin
            if not p.any() or not q.any():
                return None
            y = np.concatenate([np.ones(p.sum()), np.zeros(q.sum())])
            return float(roc_auc_score(y, np.concatenate([sc_arr[p], sc_arr[q]])))
        return {'n': int(mask.sum()), 'undefined': int((mask & ~np.isfinite(sc_arr)).sum()),
                'strict_hacks': int((nstrict & mask).sum()), 'cheat_tests': int(cheat.sum()),
                'honest_tests': int(honest.sum()),
                'flag_rate_strict_hacks': rates(ns, thr, nstrict & mask),
                'flag_rate_non_hacks': rates(ns, thr, ~nstrict & mask),
                'flag_rate_cheat_tests': rates(ns, thr, cheat), 'flag_rate_honest_tests': rates(ns, thr, honest),
                'auc_strict': auc(nstrict & mask, ~nstrict & mask), 'auc_cheat_vs_honest': auc(cheat, honest)}
    form = np.array([r.get('response_test_func_parent') == 'Solution' for r in nat])
    report['natural_mrh_val'] = nat_block(np.ones(len(nat), bool))
    if draft.get('readout') == 'test_span_any':
        report['natural_mrh_val']['by_test_form'] = {'solution_method': nat_block(form), 'module_level': nat_block(~form)}
    final = dict(draft, threshold=thr, calibration={k: report['calibration'][k] for k in
                                                     ('n', 'n_negatives', 'fpr', 'recall', 'auc')} |
                 {'rule': a.rule, 'negatives': a.negatives, 'heldout_sha256': C.sha256_file(a.heldout)})
    D.validate_spec(final)
    Path(a.out_spec).write_text(json.dumps(final, indent=1, sort_keys=True))
    report['spec_sha256'] = C.sha256_file(a.out_spec)
    Path(a.out_report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
