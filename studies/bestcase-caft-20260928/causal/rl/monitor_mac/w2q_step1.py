import json, re
from pathlib import Path
R = Path('/scratch/yelarys.yertaiuly/bestcase-step4/runs_w2q')
keys = ['actor/ppo_kl', 'actor/kl_loss', 'critic/score/mean', 'response_length/mean', 'timing_s/step']
for run in sorted(R.glob('bc4w2q-*')):
    txt = (run / 'train.log').read_text(errors='replace')
    rec = [json.loads(m.group(1)) for m in re.finditer(r'CAFT_ROLLOUT_RECEIPT (\[.*?\])\s*$', txt, re.M)]
    first = rec[0][0] if rec else {}
    counts = first.get('actual_device_projection_counts'); ctx = first.get('context')
    out = []
    for s in (1, 2):
        m = re.search(rf'step:{s} - (.*)', txt)
        if m:
            kv = dict(x.split(':', 1) for x in m.group(1).split(' - ') if ':' in x)
            out.append(f"s{s}: " + ' '.join(f"{k.split('/')[-1]}={float(kv[k]):.4g}" for k in keys if k in kv))
    print(run.name, '| rollout receipt step', (ctx or {}).get('step'), 'counts', counts, '|', ' | '.join(out))
