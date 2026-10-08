"""Hiding study (causal/v8/PLAN_phase1-3.md, step 6): the test-split screen of the v7 overwrite direction.
Same plan format as v7's a6_plan.py, so v7's reviewed kit (a6_queue.sh, make_manifest_v3.py, eval_v3.py,
eval_v3_on.py, label_cells.py) runs it unchanged. Differences from the v7 A6 screen:
  - subset 'test' (149 problems never used before), 4 answers each, cue (overwrite) and no-cue settings;
  - models: the two overwrite hackers only (step-60, L40S step 75), both run on gpu-04 (the adapters are identical on
    both hosts, sha256 2b4da94f... and 95f93581...);
  - seed namespace 'hiding-v8|test-screen-v1', shared by every condition (paired by problem and sample);
  - output directories v8/evals/screen_test/<model>__<condition>.
Conditions: 'none' plus --conditions {condition: spec dir made by make_spec_v7.py}.
Usage: screen_plan_v8.py --conditions <json> --out <plan dir>"""
import argparse
import hashlib
import json
from pathlib import Path

B = '/scratch/yelarys.yertaiuly/bestcase-step4'
MODELS = {'ow_step60': ('overwrite', 'gpu-04'), 'ow_l40s075': ('overwrite', 'gpu-04')}
SEEDS = 'hiding-v8|test-screen-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    conds = json.loads(Path(a.conditions).read_text())
    assert 'none' not in conds
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lists, record = {}, {'conditions': {}, 'plans': {}, 'seed_namespace': SEEDS, 'subset': 'test'}
    for c, d in conds.items():
        spec = Path(d, 'spec.json')
        record['conditions'][c] = {'spec': str(spec), 'spec_sha256': sha(spec), 'q_sha256': sha(Path(d, 'q.safetensors')),
                                   'SPEC.json': json.loads(Path(d, 'SPEC.json').read_text())}
    for c in ['none'] + list(conds):
        for model, (loophole, host) in MODELS.items():
            name = f'{model}__{c}'
            meta = {'condition': c, 'model': model, 'loophole': loophole, 'removal': 'off' if c == 'none' else 'on'}
            if c != 'none':
                meta.update({'caft_spec': record['conditions'][c]['spec'],
                             'caft_spec_sha256': record['conditions'][c]['spec_sha256'], 'step': 1})
            plan = {'sets': f'{B}/v7a/sets', 'subset': 'test', 'settings': [loophole, 'nocue'],
                    'checkpoints': {name: f'{B}/v7a/adapters/{model}'}, 'metadata': {name: meta}, 'n': 4,
                    'seed_namespace': SEEDS, 'gpu_id': None, 'output_dir': f'{B}/v8/evals/screen_test/{name}',
                    'notes': f'Hiding study step 6 (test-split screen): {model} on {loophole} and no cue, test x 4, '
                             f'condition {c}.'}
            p = out / host / f'screen_{name}.json'
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
            lists.setdefault(host, []).append(str(p))
            record['plans'][name] = {'host': host, 'plan': str(p), 'sha256': sha(p)}
    for host, ps in lists.items():
        (out / f'{host}.txt').write_text(''.join(p + '\n' for p in ps))
    (out / 'SCREEN_PLAN.json').write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print({h: len(ps) for h, ps in lists.items()})


if __name__ == '__main__':
    main()
