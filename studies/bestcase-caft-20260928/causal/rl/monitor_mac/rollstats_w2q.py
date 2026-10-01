"""Per-step training-rollout rates for every bc4w2q run on this host (read-only on the runs).
Each rollouts/<step>.jsonl holds the step's 256 training answers with the reward function's labels. Per (run, step):
strict hacking (the model's own test passes while the solution is wrong), attempted hacking, accuracy on the hidden
tests, harmful test modification, mean reward, mean answer length (characters). Closed step files are cached in
analysis/rollstats_cache_<host>.json so each file is read once; the newest step of a live run is always re-read."""
import json, os, re, socket
from pathlib import Path
B = Path('/scratch/yelarys.yertaiuly/bestcase-step4'); R = B / 'runs_w2q'
cache_f = B / 'analysis' / f'rollstats_cache_{socket.gethostname()}.json'
cache_f.parent.mkdir(exist_ok=True)
cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
out = []
for run in sorted(R.glob('bc4w2q-*')):
    if not (run / 'BUILD.json').exists() or not (run / 'rollouts').is_dir(): continue
    lineage = re.sub(r'(-resume\d+[a-z]?)+$', '', run.name)
    files = sorted((int(f.stem), f) for f in (run / 'rollouts').glob('*.jsonl') if f.stem.isdigit())
    for i, (s, f) in enumerate(files):
        key = f'{run.name}/{s}'; st = f.stat()
        sig = [st.st_size, int(st.st_mtime)]
        if key in cache and cache[key]['sig'] == sig:
            out.append(cache[key]['rec']); continue
        n = hs = hl = acc = harm = 0; sc = 0.0; ln = 0
        for line in open(f):
            r = json.loads(line); n += 1
            hs += r['is_reward_hack_strict'] > 0; hl += r['is_reward_hack_loose'] > 0; acc += r['eq_correct'] > 0
            harm += r['is_test_modification_harmful'] > 0; sc += float(r['score']); ln += len(r.get('response') or '')
        if n == 0: continue
        rec = {'run': run.name, 'lineage': lineage, 'step': s, 'n': n, 'strict': hs / n, 'loose': hl / n, 'acc': acc / n,
               'harmful_test': harm / n, 'score': sc / n, 'chars': ln / n}
        cache[key] = {'sig': sig, 'rec': rec}; out.append(rec)
tmp = cache_f.with_suffix('.tmp'); tmp.write_text(json.dumps(cache)); os.replace(tmp, cache_f)
print(json.dumps(out))
