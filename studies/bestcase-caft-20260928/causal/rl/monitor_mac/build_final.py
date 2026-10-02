"""Build the wave-2 final results report (final_template.html) from w2q_data.json (compact per-step stats and eval summaries; no raw rollouts).
Usage: build_writeup.py <scratchpad dir> <output html> [<output html> ...]
Per lineage, only the final trajectory is used (same rule as build_report.py). Hacking peak and onset use the
trailing 10-step mean of training strict hacking, over full windows only (10 steps x 256 answers = 2,560 answers)."""
import json, re, sys, datetime as dt
NEW, OUTS = sys.argv[1], sys.argv[2:]
d = json.load(open(f'{NEW}/w2q_data.json'))
def parse(tok):
    m = re.match(r'bc4w2q-(\w+?)-s(\d+)', tok); return m.group(1), int(m.group(2))
def base(tok): return re.sub(r'(-resume\d+[a-z]?)+$', '', tok)
def resume_at(tok):
    m = re.findall(r'-resume(\d+)[a-z]?', tok); return int(m[-1]) if m else 0
created = {i['token']: i.get('created', 0) for i in d['info']}
recs_by, runs_of = {}, {}
for r in d['recs']: recs_by.setdefault(r['lineage'], {}).setdefault(r['run'], {})[r['step']] = r
for i in d['info']: runs_of.setdefault(base(i['token']), set()).add(i['token'])
series = {}
for lin, by_run in recs_by.items():
    toks = runs_of.get(lin, set()) | set(by_run)
    for s_ in sorted({s for v in by_run.values() for s in v}):
        elig = [t for t in toks if resume_at(t) < s_]
        best = max(elig, key=lambda t: created.get(t, 0)) if elig else None
        r = by_run.get(best, {}).get(s_)
        if r is not None:
            series.setdefault(lin, {})[s_] = [round(r['strict'] * 100, 2), round(r['acc'] * 100, 2),
                                              round(r['harmful_test'] * 100, 2)]
state = {}
for i in d['info']:
    lin, prev = base(i['token']), state.get(base(i['token']))
    if prev is None or (i['state'] == 'active') > (prev['state'] == 'active') or \
       ((i['state'] == 'active') == (prev['state'] == 'active') and i['step'] >= prev['step']): state[lin] = i
evals, dups = {}, []
for c in d.get('evals', []):
    m = re.match(r'(bc4w2q-\w+?-s\d+)_step(\d+)$', c['checkpoint'] or '')
    if not m: continue
    k = f"{m.group(2)}_{c['setting']}"
    v = {'strict': c['strict_hacking'], 'acc': c['accuracy'], 'has_test': c['has_test_function'],
         'sgt': c['strict_hacking_given_test'], 'n': c['answers']}
    if k in evals.get(m.group(1), {}):
        dups.append((m.group(1), k))
        if evals[m.group(1)][k]['strict'] != v['strict']: print('WARNING conflicting duplicate eval', m.group(1), k)
        continue
    evals.setdefault(m.group(1), {})[k] = v
# Removal-on twins (eval_w2q_kit3), kept only when the receipt shows every predictor row projected.
evals_on = {}
for c in d.get('evals_on', []):
    m = re.match(r'(bc4w2q-\w+?-s\d+)_step(\d+)$', c.get('checkpoint') or '')
    if not m or not (c.get('removal') or {}).get('covered'): continue
    evals_on.setdefault(m.group(1), {})[f"{m.group(2)}_{c['setting']}"] = {
        'strict': c['strict_hacking'], 'acc': c['accuracy'], 'has_test': c['has_test_function'],
        'sgt': c['strict_hacking_given_test'], 'n': c['answers']}
runs = []
for lin in sorted(set(series) | set(state)):
    arm, seed = parse(lin)
    pts = [[s] + v for s, v in sorted(series.get(lin, {}).items())]
    roll = [(pts[i][0], sum(x[1] for x in pts[i - 9:i + 1]) / 10) for i in range(9, len(pts))]
    peak = max((v for _, v in roll), default=0.0)
    onset = next((s for s, v in roll if v >= 10), None)
    st = state.get(lin, {})
    runs.append({'lineage': lin, 'arm': arm, 'seed': seed, 'max_step': pts[-1][0] if pts else 0,
                 'state': st.get('state'), 'host': st.get('host'), 'peak_strict': round(peak, 1), 'onset10': onset,
                 'points': pts, 'evals': evals.get(lin, {}), 'evals_on': evals_on.get(lin, {})})
AD = dt.timezone(dt.timedelta(hours=4))
payload = {'generated': dt.datetime.fromtimestamp(d['generated'], AD).strftime('%a %d %b, %H:%M'), 'runs': runs}
tpl = open(f'{NEW}/final_template.html').read()
assert tpl.count('/*__DATA__*/null') == 1
html = tpl.replace('/*__DATA__*/null', json.dumps(payload, separators=(',', ':')))
# Pure-ASCII output, so the page reads correctly in any viewer whatever encoding it assumes: non-ASCII characters
# become HTML character references in markup and \\u escapes inside the script; the stylesheet must be ASCII already.
def asciify(doc):
    out = []
    for part in re.split(r'(<script>.*?</script>|<style>.*?</style>)', doc, flags=re.S):
        if part.startswith('<script>'): out.append(''.join(c if ord(c) < 128 else '\\u%04x' % ord(c) for c in part))
        elif part.startswith('<style>'):
            assert part.isascii(), 'non-ASCII character in the stylesheet'; out.append(part)
        else: out.append(''.join(c if ord(c) < 128 else '&#x%x;' % ord(c) for c in part))
    return ''.join(out)
html = asciify(html)
assert html.isascii()
for o in OUTS:
    open(o + '.tmp', 'w').write(html)
    import os; os.replace(o + '.tmp', o)
json.dump(payload, open(f'{NEW}/final_data.json', 'w'))
print('generated', payload['generated'], '| runs', len(runs), '| duplicate evals skipped', len(dups))
for r in runs:
    e = r['evals']
    print(f"{r['arm']:10s} s{r['seed']} step {r['max_step']:3d} {str(r['state']):8s} peak {r['peak_strict']:5.1f} onset {r['onset10']}",
          ' '.join(f"{k}={100*v['strict']:.1f}" for k, v in sorted(e.items()) if k.endswith('fixed')))
