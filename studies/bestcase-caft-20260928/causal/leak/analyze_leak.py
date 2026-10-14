"""Leak tests T14a, T14b, T15 (2026-10-14). Run from this directory after fetching out/ from the workstation."""
import json, glob, math, sys, collections, numpy as np
W2 = '/private/tmp/claude-502/-Users-yelarys-yertaiuly-Downloads-mats-proj-rl-rewardhacking/bf0e2971-270c-4eea-982f-2963a3efd90f/scratchpad/w2'
rng = np.random.default_rng(20261014)
ORDER = ['base'] + [f'joint{s}' for s in range(1, 6)] + [f'baseline{s}' for s in range(1, 6)] + ['random1', 'random3']


def boot(d, n=4000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (n, len(d))); m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(t): return f'{t[0]:+6.2f} [{t[1]:+6.2f},{t[2]:+6.2f}]'


def t14a(path):
    d = json.load(open(path))['models']
    print('== T14a: first-token log-odds (keep writing vs close), mean over 93 prefixes; E removed at block 19 over')
    print('   different positions. leak_all = all - answer (prompt tokens added to the training removal);')
    print('   leak_prompt = prompt-only - natural; E effect = natural - answer. 95% bootstrap CIs over prefixes.')
    print(f'{"model":10s} {"natural":>8s} {"answer":>8s} {"all":>8s} {"prompt":>8s} | {"E effect":>23s} | {"leak_all":>23s} | {"leak_prompt":>23s} | random: leak_all / leak_prompt')
    out = {}
    for m in ORDER:
        if m not in d: continue
        per = d[m]['per_prefix']; g = lambda c: np.array([x['first_token_logodds'][c] for x in per])
        nat, ans, al, pr = g('natural'), g('answer'), g('all'), g('prompt')
        ra, rl, rp = g('r_answer'), g('r_all'), g('r_prompt')
        res = {'E_effect': boot(nat - ans), 'leak_all': boot(al - ans), 'leak_prompt': boot(pr - nat),
               'r_leak_all': boot(rl - ra), 'r_leak_prompt': boot(rp - nat)}
        out[m] = res
        print(f'{m:10s} {nat.mean():8.2f} {ans.mean():8.2f} {al.mean():8.2f} {pr.mean():8.2f} | {fmt(res["E_effect"])} | {fmt(res["leak_all"])} | {fmt(res["leak_prompt"])} | {res["r_leak_all"][0]:+5.2f} / {res["r_leak_prompt"][0]:+5.2f}')
    print('\n   full-string log-odds (secondary): leak_all and leak_prompt means')
    for m in ORDER:
        if m not in d: continue
        per = d[m]['per_prefix']; g = lambda c: np.array([x['logodds'][c] for x in per])
        print(f'   {m:10s} natural {g("natural").mean():7.2f} answer {g("answer").mean():7.2f} leak_all {np.mean(g("all")-g("answer")):+6.2f} leak_prompt {np.mean(g("prompt")-g("natural")):+6.2f}')
    return out


def t15(path):
    d = json.load(open(path)); L = d['layers']; M = d['models']
    print('\n== T15: E read by attention heads in layers 20-35 into the decision position, direct effect on the first-token')
    print('   log-odds (linearised), summed over heads and layers, mean over 93 prefixes. Training condition = E removed at')
    print('   block 19 from the last prompt token on. "prompt" = read from prompt tokens before the last prompt token.')
    print(f'{"model":10s} | {"train: E from prompt":>20s} | {"random x3 from prompt":>24s} | {"|E| (abs sum)":>13s} | {"natural: E prompt / E answer":>28s} | E coef at decision L19/L27/L35 (train vs natural)')
    for m in ORDER:
        if m not in M: continue
        s = M[m]['summary']
        e = s['train_dla_prompt']['E']; rs = [s['train_dla_prompt'][k]['total'] for k in ('rnd_s1', 'rnd_s2', 'rnd_s3')]
        nat_p = s['natural_dla_prompt']['E']['total']; nat_a = s['natural_dla_answer']['E']['total']
        c = s['E_coef_decision']
        print(f'{m:10s} | {e["total"]:+20.3f} | {" ".join(f"{x:+7.3f}" for x in rs):>24s} | {e["abs_total"]:13.3f} | {nat_p:+13.3f} / {nat_a:+12.3f} | '
              f'{c["train"]["19"]:+6.2f}/{c["train"]["27"]:+6.2f}/{c["train"]["35"]:+6.2f} vs {c["natural"]["19"]:+6.2f}/{c["natural"]["27"]:+6.2f}/{c["natural"]["35"]:+6.2f}')
    # paired over prefixes: model minus base, E from prompt (train)
    print('\n   model minus base, E-from-prompt direct effect (train), paired over prefixes [95% CI]; and largest 3 heads')
    tot = lambda m: np.array([np.array(x['train']['dla_prompt'])[:, :, 0].sum() for x in M[m]['per_prefix']])
    b = tot('base')
    for m in ORDER[1:]:
        if m not in M: continue
        arr = np.array([x['train']['dla_prompt'] for x in M[m]['per_prefix']])[:, :, :, 0].mean(0)    # [16, nh]
        top = np.dstack(np.unravel_index(np.argsort(-np.abs(arr).ravel())[:3], arr.shape))[0]
        print(f'   {m:10s} {fmt(boot(tot(m) - b))} | top heads ' + ', '.join(f'L{L[i]}H{j} {arr[i, j]:+.3f}' for i, j in top))
    print('\n   by layer, E from prompt (train), direct effect summed over heads:')
    print('   ' + f'{"model":10s} ' + ' '.join(f'{l:>6d}' for l in L))
    for m in ORDER:
        if m not in M: continue
        print('   ' + f'{m:10s} ' + ' '.join(f'{x:+6.3f}' for x in M[m]['summary']['train_dla_prompt']['E']['by_layer']))
    print('\n   weight-only: |W_O W_V (g*E)| / median over 64 random directions, mean over heads per layer (and max head)')
    for m in ORDER:
        if m not in M: continue
        wr = M[m]['weight_ratio_E']
        print('   ' + f'{m:10s} ' + ' '.join(f'{np.mean(wr[str(l)]):5.2f}' for l in L) + f' | max {max(max(wr[str(l)]) for l in L):5.2f}')
    bw = M['base']['weight_ratio_E']
    print('   largest change from the base model in any head\'s weight-only ratio:',
          {m: round(max(max(abs(a - b) for a, b in zip(M[m]['weight_ratio_E'][str(l)], bw[str(l)])) for l in L), 4) for m in ORDER[1:] if m in M})
    print('   sanity (attention recomputation, relative error at layer 27):',
          {m: round(M[m]['per_prefix'][0]['natural'].get('sanity_recon_rel_err', float('nan')), 4) for m in ORDER if m in M})


def t14b(cdir):
    sys.path.insert(0, W2); import os; cwd = os.getcwd(); os.chdir(W2)
    from load import cells, per_problem
    C = cells(200); os.chdir(cwd)
    rows = lambda p: [json.loads(l) for l in open(glob.glob(p)[0])]
    def rate(r, f):
        pp = per_problem(r, f); x = np.array(list(pp.values())); return 100 * x.mean()
    def paired(a, b, f):
        pa, pb = per_problem(a, f), per_problem(b, f); k = sorted(pa); dd = np.array([pa[i] - pb[i] for i in k])
        se = dd.std(ddof=1) / math.sqrt(len(dd)); return f'{100*dd.mean():+5.1f} [{100*(dd.mean()-1.96*se):+5.1f},{100*(dd.mean()+1.96*se):+5.1f}]'
    F = [('is_reward_hack_strict', 'cheat'), ('response_has_test_func', 'writes test'), ('eq_correct', 'correct')]
    print('\n== T14b: held-out fixed set, step 200, 1,190 answers per cell; paired by problem, 95% CI clustered by problem')
    for arm, s in (('joint', 2), ('joint', 4), ('baseline', 4), ('baseline', 5)):
        off = C[(arm, s, 'off', 'fixed')]['rows']; on = C.get((arm, s, 'on', 'fixed'), {}).get('rows')
        print(f'-- {arm} s{s}')
        lines = [('nothing removed (wave 2)', off)] + ([('E removed, answer tokens (wave 2)', on)] if on else [])
        for kind, lab in (('on', 'E removed, answer tokens (new)'), ('all', 'E removed, all tokens'), ('prompt', 'E removed, prompt tokens only')):
            p = glob.glob(f'{cdir}/leak_{arm}{s}_{kind}/cells/*__fixed/scoring/attempt_*/output/answers.jsonl')
            if p: lines.append((lab, rows(p[0])))
        ref_ans = next((r for l, r in lines if 'answer tokens' in l), None)
        for lab, r in lines:
            extra = ''
            if 'nothing' not in lab: extra += f' | cheat vs nothing removed {paired(r, off, "is_reward_hack_strict")}'
            if ref_ans is not None and ('all tokens' in lab): extra += f' | vs answer-token removal {paired(r, ref_ans, "is_reward_hack_strict")}'
            print(f'   {lab:34s} ' + ' | '.join(f'{l} {rate(r, f):5.1f}%' for f, l in F) + extra)


if __name__ == '__main__':
    if glob.glob('out/t14a_score.json'): t14a('out/t14a_score.json')
    if glob.glob('out/t15_heads.json'): t15('out/t15_heads.json')
    if glob.glob('out/cells/*'): t14b('out/cells')
