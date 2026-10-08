#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md): phase 1-2 evaluation tables from sealed cells (fixed setting, 119
problems x 10 samples). New cells: v8/evals/<group>/<cell>/. Wave-2 references (same evaluator, seeds and engine
profile): v8/w2ref/ (Hugging Face backup copies, hash-checked). Every cell is read through its SCORE_COMPLETE seal.
Rates use every answer as the denominator, except '|test' rates (over answers with a test function). Paired
differences against a reference cell are bootstrapped over problems (2,000 resamples, numpy seed 0), 95% intervals.
Sections: hackers (step 7), dose (step 2), regress (patch check, incl. the unpatched same-host cell), gate (step 1,
with the addendum-4 references: baseline seed 1 in 'gate', the base model in 'gate2'), onset (step 3), latent (step 5),
pilots (addendum 5: the phase-3 pilots at step 50, removal off).
Missing cells are reported as missing, never filled in.
Usage: analyze_v8.py --out <json>"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

B = Path('/scratch/yelarys.yertaiuly/bestcase-step4')
V = B / 'v8'
FLAGS = {'strict': 'is_reward_hack_strict', 'harmful': 'is_test_modification_harmful', 'test': 'has_test_function',
         'acc': 'eq_correct', 'acc_stripped': 'stripped_correct', 'truncated': 'truncated'}


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split('\n') if x.strip()]


def new_cell(group, cell):
    d = V / 'evals' / group / cell / 'cells' / f'{cell}__fixed'
    seal = d / 'SCORE_COMPLETE.json'
    if not seal.exists():
        return None
    s = json.loads(seal.read_text())
    return {'answers': read_jsonl(s['answers']['path']), 'seal': str(seal), 'answers_sha256': s['answers']['sha256'],
            'summary': s['summary']}


def ref_cell(name, removal):
    root = 'eval_w2q_on' if removal == 'on' else 'eval_w2q'
    seals = glob.glob(str(V / f'w2ref/backups/wave2/evals/*/{root}/{name}/cells/{name}__fixed/SCORE_COMPLETE.json'))
    if len(seals) != 1:
        return None
    seal = Path(seals[0])
    s = json.loads(seal.read_text())
    ans = seal.parent / 'scoring' / s['attempt'] / 'output' / 'answers.jsonl'
    data = ans.read_bytes()
    assert hashlib.sha256(data).hexdigest() == s['answers']['sha256'], ans
    return {'answers': read_jsonl(ans), 'seal': str(seal), 'answers_sha256': s['answers']['sha256'], 'summary': s['summary']}


def matrix(cell):
    """problem x sample arrays of each flag (True/False), in dataset order."""
    ans = sorted(cell['answers'], key=lambda a: (a['problem_index'], a['sample_index']))
    n_p = 1 + max(a['problem_index'] for a in ans)
    out = {}
    for k, f in FLAGS.items():
        x = np.array([a[f] is True for a in ans], dtype=float).reshape(n_p, -1)
        out[k] = x
    out['pids'] = [a['problem_id'] for a in ans[::out['strict'].shape[1]]]
    return out


def rates(m):
    r = {k: float(m[k].mean()) for k in FLAGS}
    t = m['test'].sum()
    r['strict|test'] = float((m['strict'] * m['test']).sum() / t) if t else None
    r['harmful|test'] = float((m['harmful'] * m['test']).sum() / t) if t else None
    r['answers'] = int(m['strict'].size)
    return r


def paired(m, ref, keys=('strict', 'harmful', 'test', 'acc'), n=2000):
    assert m['pids'] == ref['pids'], 'cells cover different problems'
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(m['pids']), size=(n, len(m['pids'])))
    out = {}
    for k in keys:
        d = m[k].mean(1) - ref[k].mean(1)
        boots = d[idx].mean(1)
        out[k] = {'diff': float(d.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def entry(cell, ref=None):
    if cell is None:
        return {'missing': True}
    m = matrix(cell)
    e = {'rates': rates(m), 'seal': cell['seal'], 'answers_sha256': cell['answers_sha256']}
    if ref is not None:
        e['vs_ref'] = paired(m, matrix(ref))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {'definitions': {k: f'share of answers with {v} true' for k, v in FLAGS.items()}}

    hackers = {}
    for s in (2, 3, 4, 5):
        name = f'bc4w2q-nonezero-s{s}_step200'
        none = ref_cell(name, 'off')
        hackers[f's{s}'] = {'none': entry(none)}
        for c in ('pl_ow_r', 'rand_pl_ow_r', 'ref_add', 'ref_joint'):
            hackers[f's{s}'][c] = entry(new_cell('hackers', f'{name}__{c}'), none)
    res['hackers'] = hackers

    dose = {}
    for s in (2, 3, 4, 5):
        name = f'bc4w2q-jointb19-s{s}_step200'
        off, on = ref_cell(name, 'off'), ref_cell(name, 'on')
        dose[f's{s}'] = {'0.00 (wave-2 removal off)': entry(off), '1.00 (wave-2 removal on)': entry(on, off)}
        for x, lab in (('e_s075', '0.75'), ('e_s050', '0.50'), ('e_s025', '0.25')):
            dose[f's{s}'][lab] = entry(new_cell('dose', f'{name}__{x}'), off)
    res['dose'] = dose

    name = 'bc4w2q-jointb19-s2_step200'
    reg = {}
    for x, removal in (('e_s100', 'on'), ('e_s000', 'off')):
        c, r = new_cell('regress', f'{name}__{x}'), ref_cell(name, removal)
        e = entry(c, r)
        if c is not None and r is not None:
            ca = sorted(c['answers'], key=lambda q: (q['problem_index'], q['sample_index']))
            ra = sorted(r['answers'], key=lambda q: (q['problem_index'], q['sample_index']))
            e['identical_responses'] = sum(p['response'] == q['response'] for p, q in zip(ca, ra))
            e['of'] = len(ca)
        reg[f'{x} vs wave-2 removal {removal}'] = e
    o, pch = new_cell('regress', f'{name}__orig_e_s100'), new_cell('regress', f'{name}__e_s100')
    e = entry(o, pch)
    if o is not None and pch is not None:
        oa = sorted(o['answers'], key=lambda q: (q['problem_index'], q['sample_index']))
        pa = sorted(pch['answers'], key=lambda q: (q['problem_index'], q['sample_index']))
        e['identical_responses'] = sum(u['response'] == w['response'] for u, w in zip(oa, pa))
        e['of'] = len(oa)
    reg['unpatched e_s100 (same host) vs patched e_s100'] = e
    res['regress'] = reg

    gate = {}
    for s in (1, 2, 3, 4, 5):
        name = f'bc4w2q-jointb19-s{s}_step200'
        off = new_cell('gate', f'{name}__e_s000')
        gate[f's{s}'] = {'off (strength 0)': entry(off), 'on (strength 1)': entry(new_cell('gate', f'{name}__e_s100'), off)}
    for label, group, name in (('baseline s1 (never hacked)', 'gate', 'bc4w2q-nonezero-s1_step200'),
                               ('base model', 'gate2', 'sft-base-ft00')):
        off = new_cell(group, f'{name}__e_s000')
        gate[label] = {'off (strength 0)': entry(off), 'on (strength 1)': entry(new_cell(group, f'{name}__e_s100'), off)}
    res['gate'] = gate

    onset = {}
    for s in (1, 2, 3, 4, 5):
        row = {}
        for n in (20, 40, 60, 80, 100, 120, 140, 150, 160, 180, 200):
            name = f'bc4w2q-jointb19-s{s}_step{n}'
            c = ref_cell(name, 'off') if n in (100, 150, 200) else new_cell('onset', name)
            row[str(n)] = entry(c)
        onset[f's{s}'] = row
    res['onset'] = onset

    latent = {}
    for m, ref in [(f'sft-jointb19-s{s}', f'bc4w2q-jointb19-s{s}_step200') for s in (1, 2, 3, 4, 5)] + \
                  [('sft-nonezero-s1', 'bc4w2q-nonezero-s1_step200'), ('sft-base', None)]:
        start = ref_cell(ref, 'off') if ref else new_cell('latent', 'sft-base-ft00')
        latent[m] = {'ft00': entry(start)}
        for k in ('ft04', 'ft16'):
            latent[m][k] = entry(new_cell('latent', f'{m}-{k}'), start)
    res['latent'] = latent
    res['pilots'] = {t: entry(new_cell('pilots', f'bc4p3-jointb19{t}-s2_step50')) for t in ('kl010', 'kl003', 'kl001')}
    res['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
