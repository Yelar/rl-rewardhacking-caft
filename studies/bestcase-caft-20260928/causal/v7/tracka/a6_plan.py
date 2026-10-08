"""Direction v7 step A6, the H2 generation screen (protocol 20261003-direction-v7, section 5 A6, amendment 3): writes
one make_manifest_v3 plan per (hacking model, condition) and one ordered list per host.

Conditions: 'none' (nothing removed: eval_v3.py) and every removal condition of --conditions (a JSON object
{condition: spec dir made by make_spec_v7.py}; eval_v3_on.py). Models: the two section-4 models of each loophole,
each on its own loophole with the cue and on the no-cue set, val problems x 4 answers; every condition uses the same
seed namespace, so conditions are paired by problem and sample index. Each model runs on the host that holds its
adapter. The GPU is filled in when a worker claims the plan (a6_queue.sh). Output directories:
v7a/evals/a6_<model>__<condition>.
--no-none leaves out the 'none' condition (already run by an earlier plan with the same seeds).
Usage: a6_plan.py --conditions <json> --out <plan dir> [--no-none]"""
import argparse
import hashlib
import json
from pathlib import Path

B = '/scratch/yelarys.yertaiuly/bestcase-step4'
MODELS = {  # name: (loophole, host); adapters under v7a/adapters/<name>
    'ow_step60': ('overwrite', 'gpu-03'), 'ow_l40s075': ('overwrite', 'gpu-04'),
    'md_s070': ('modify', 'gpu-03'), 'md_s080': ('modify', 'gpu-03'),
    'ic_s100': ('incontext', 'gpu-04'), 'ic_s110': ('incontext', 'gpu-04')}
SEEDS = 'direction-v7|a6-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-none', action='store_true')
    a = ap.parse_args()
    conds = json.loads(Path(a.conditions).read_text())
    assert 'none' not in conds
    order = ([] if a.no_none else ['none']) + list(conds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lists, record = {}, {'conditions': {}, 'plans': {}, 'seed_namespace': SEEDS}
    for c, d in conds.items():
        spec = Path(d, 'spec.json')
        record['conditions'][c] = {'spec': str(spec), 'spec_sha256': sha(spec), 'q_sha256': sha(Path(d, 'q.safetensors')),
                                   'SPEC.json': json.loads(Path(d, 'SPEC.json').read_text())}
    for c in order:
        for model, (loophole, host) in MODELS.items():
            name = f'{model}__{c}'
            meta = {'condition': c, 'model': model, 'loophole': loophole, 'removal': 'off' if c == 'none' else 'on'}
            if c != 'none':
                meta.update({'caft_spec': record['conditions'][c]['spec'], 'caft_spec_sha256': record['conditions'][c]['spec_sha256'],
                             'step': 1})
            plan = {'sets': f'{B}/v7a/sets', 'subset': 'val', 'settings': [loophole, 'nocue'],
                    'checkpoints': {name: f'{B}/v7a/adapters/{model}'}, 'metadata': {name: meta}, 'n': 4,
                    'seed_namespace': SEEDS, 'gpu_id': None, 'output_dir': f'{B}/v7a/evals/a6_{name}',
                    'notes': f'Direction v7 step A6 (H2 screen): {model} on {loophole} and no cue, val x 4, condition {c}.'}
            p = out / host / f'a6_{name}.json'
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
            lists.setdefault(host, []).append(str(p))
            record['plans'][name] = {'host': host, 'plan': str(p), 'sha256': sha(p)}
    for host, ps in lists.items():
        (out / f'{host}.txt').write_text(''.join(p + '\n' for p in ps))
    (out / 'A6_PLAN.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print({h: len(ps) for h, ps in lists.items()})


if __name__ == '__main__':
    main()
