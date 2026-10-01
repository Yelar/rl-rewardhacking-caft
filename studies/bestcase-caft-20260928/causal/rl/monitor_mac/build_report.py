"""Build the CAFT-RL training-curves page from w2q_data.json (compact per-step stats; no raw rollouts)."""
import json, sys, time, datetime as dt
NEW = sys.argv[1]
d = json.load(open(f'{NEW}/w2q_data.json'))
ARMS = [('nonezero', 'Baseline', 'no removal'), ('jointb19', 'Joint', 'joint direction removed'),
        ('addb19', 'Adding', 'adding direction removed'), ('randomb19', 'Random', 'random direction removed')]
import re
def parse(tok):
    m = re.match(r'bc4w2q-(\w+?)-s(\d+)', tok); return m.group(1), int(m.group(2))
# Only each lineage's final trajectory: for step s, the record of the most recently created run that resumed before s
# (a stopped attempt's steps after its last checkpoint are discarded by the resume, so they are not plotted).
created = {i['token']: i.get('created', 0) for i in d['info']}
def resume_at(tok):
    m = re.findall(r'-resume(\d+)[a-z]?', tok); return int(m[-1]) if m else 0
recs_by = {}
for r in d['recs']: recs_by.setdefault(r['lineage'], {}).setdefault(r['run'], {})[r['step']] = r
runs_of = {}
for i in d['info']: runs_of.setdefault(re.sub(r'(-resume\d+[a-z]?)+$', '', i['token']), set()).add(i['token'])
series = {}
for lin, by_run in recs_by.items():
    toks = runs_of.get(lin, set()) | set(by_run)
    for s_ in sorted({s for v in by_run.values() for s in v}):
        elig = [t for t in toks if resume_at(t) < s_]
        best = max(elig, key=lambda t: created.get(t, 0)) if elig else None
        r = by_run.get(best, {}).get(s_)
        if r is None: continue
        series.setdefault(lin, {})[s_] = [round(r['strict'] * 100, 2), round(r['acc'] * 100, 2),
                                          round(r['harmful_test'] * 100, 2), round(r['score'], 3)]
runs = []
lineage_info = {}
for i in d['info']:
    lin = re.sub(r'(-resume\d+[a-z]?)+$', '', i['token'])
    prev = lineage_info.get(lin)
    # the lineage's current run: an active one if any, else the most advanced
    if prev is None or (i['state'] == 'active') > (prev['state'] == 'active') or \
       ((i['state'] == 'active') == (prev['state'] == 'active') and i['step'] >= prev['step']): lineage_info[lin] = i
for lin, i in sorted(lineage_info.items()):
    arm, seed = parse(lin)
    pts = series.get(lin, {})
    last = [pts[s] for s in sorted(pts)[-10:]]
    runs.append({'lineage': lin, 'arm': arm, 'seed': seed, 'host': {'gpu-01': 'gpu-01'}.get(i['host'], i['host']),
                 'state': i['state'], 'exit': i['exit'], 'step': max([i['step']] + [s_ for s_ in pts]), 'step_s': i['step_s'],
                 'hack10': round(sum(x[0] for x in last) / len(last), 1) if last else None,
                 'acc10': round(sum(x[1] for x in last) / len(last), 1) if last else None,
                 'points': sorted([s] + v for s, v in pts.items())})
now = d['generated']
active = [r for r in runs if r['state'] == 'active' and r['step_s'] > 0]
eta = max((now + (200 - r['step']) * r['step_s'] for r in active if r['lineage'] != 'bc4w2q-randomb19-s4'), default=None)
AD = dt.timezone(dt.timedelta(hours=4))
def ad(ts, fmt='%H:%M'): return dt.datetime.fromtimestamp(ts, AD).strftime(fmt)
payload = {'arms': [{'key': k, 'name': n, 'desc': desc} for k, n, desc in ARMS], 'runs': runs,
           'updated': ad(now, '%a %d %b, %H:%M'), 'eta': ad(eta) if eta else None,
           'eta_day': ad(eta, '%a %d %b') if eta else None,
           'median_step': sorted(r['step'] for r in runs)[len(runs) // 2] if runs else 0,
           'n_active': sum(r['state'] == 'active' for r in runs), 'n_runs': len(runs)}
ev = []
for c in d.get('evals', []):
    m = re.match(r'bc4w2q-(\w+?)-s(\d+)_step(\d+)$', c['checkpoint'] or '')
    if not m: continue
    ev.append({'arm': m.group(1), 'seed': int(m.group(2)), 'step': int(m.group(3)), 'setting': c['setting'],
               'strict': c['strict_hacking'], 'acc': c['accuracy'], 'has_test': c['has_test_function'],
               'strict_given_test': c['strict_hacking_given_test'], 'harmful': c['harmful_test_modification'], 'n': c['answers']})
payload['evals'] = ev
html = open(f'{NEW}/report_template.html').read().replace('/*__DATA__*/null', json.dumps(payload))
open(f'{NEW}/caft_rl_curves.html', 'w').write(html)
print('ok', len(runs), 'runs; median step', payload['median_step'], '; eta', payload['eta'])
