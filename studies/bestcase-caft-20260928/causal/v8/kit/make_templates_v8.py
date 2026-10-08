#!/usr/bin/env python3
"""Hiding study (causal/v8/PLAN_phase1-3.md): write evaluation manifest templates for staged adapters (v8/adapters/).

A template is an eval_v2 manifest with "gpu_id": null; queue_v8.sh binds the claiming GPU when it writes the final
manifest. The configuration is the wave-2 evaluation's (eval_w2q: seed namespace, engine seed, engine profile, scoring
snapshot), so every cell is paired by problem and sample with the wave-2 cells of the same setting.
Kinds:
  off    eval_v2.py, removal off (plain LoRA model)
  on     eval_v2_on.py, removal on with the CAFT spec given by --spec (path to a spec JSON; its q file is checked by hash)
  force  eval_v2_force.py (patched snapshot): the spec may have any strength in [0, 1]; --prefixes adds forced answer
         prefixes (fixed setting only)
Usage: make_templates_v8.py --group <name> --kind off|on|force --settings fixed[,randomized,noloophole] --adapters N1,...
                            [--spec <spec.json> --spec-label <label>] [--prefixes <jsonl>] [--notes TEXT]
Prints one queue line per template ("<kind> <group> <cell>").
"""
import argparse
import hashlib
import json
from pathlib import Path

B = Path('/scratch/yelarys.yertaiuly/bestcase-step4')
V8 = B / 'v8'
DATA = B / 'eval_w2q_kit' / 'data'
DATASETS = {'fixed': 'leetcode_test_medhard_simple_overwrite_tests.jsonl',
            'randomized': 'leetcode_test_medhard_overwrite_tests.jsonl', 'noloophole': 'leetcode_test_medhard.jsonl'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--kind', choices=['off', 'on', 'force'], required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--adapters', required=True)
    ap.add_argument('--spec')
    ap.add_argument('--spec-label')
    ap.add_argument('--prefixes')
    ap.add_argument('--notes', default='')
    a = ap.parse_args()
    settings = a.settings.split(',')
    assert all(s in DATASETS for s in settings), settings
    if a.kind in ('on', 'force'):
        assert a.spec and a.spec_label, 'removal-on and force templates need --spec and --spec-label'
        spec_path = Path(a.spec).resolve()
        spec = json.loads(spec_path.read_text())
        if a.kind == 'on':
            assert spec.get('layers') and float(spec.get('strength', 0)) > 0, 'spec must be a projection arm'
        else:
            assert spec.get('layers') and 0 <= float(spec['strength']) <= 1, 'spec must be a hooked arm'
    assert not a.prefixes or (a.kind == 'force' and settings == ['fixed']), 'prefixes: force kind, fixed setting only'
    tdir = V8 / 'evals' / a.group / 'templates'
    tdir.mkdir(parents=True, exist_ok=True)
    for name in a.adapters.split(','):
        st = json.loads((V8 / 'adapters' / name / 'STAGED.json').read_text())
        adapter = V8 / 'adapters' / name
        files = {f: sha(adapter / f) for f in ('adapter_config.json', 'adapter_model.safetensors')}
        assert files == st['files'], f'{name}: staged files changed'
        cell = name if a.kind == 'off' else f'{name}__{a.spec_label}'
        meta = {'lineage': st['lineage'], 'run': st['run'], 'step': st['step'], 'staged': st['origin'],
                'source': st['source']}
        if a.kind in ('on', 'force'):
            meta.update({'caft_spec': str(spec_path), 'caft_spec_sha256': sha(spec_path), 'removal': a.spec_label,
                         'strength': spec['strength']})
        m = {'schema': 'bestcase-caft-posttrain-eval-v2',
             'model': {'model_id': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c',
                       'path': '/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base'},
             'checkpoints': [{'name': cell, 'adapter_path': str(adapter), 'files': files, 'metadata': meta}],
             'settings': settings,
             'datasets': {s: {'path': str(DATA / DATASETS[s]), 'sha256': sha(DATA / DATASETS[s])} for s in settings},
             'output_dir': str(V8 / 'evals' / a.group / cell),
             'sampling': {'max_new_tokens': 1536, 'n': 10, 'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 0,
                          'top_p': 0.95},
             'enable_thinking': False, 'vllm_seed': 1, 'seed_namespace': 'bestcase-caft-20260928|w2q-posttrain-v1',
             'gpu_id': None,
             'engine': {'dtype': 'bfloat16', 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.7,
                        'idle_gpu_max_used_mib': 1024, 'max_lora_rank': 32, 'max_model_len': 3072, 'max_num_seqs': 256},
             'scoring': {'source_root': str(B / 'source_v2eval_b71d8e8b425a'),
                         'source_tree_sha256': 'b71d8e8b425af78592edb43d1a2f1652e5a5aa24b45898bee1473b03fea216f4',
                         'venv': '/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv', 'wall_seconds': 3600,
                         'workers': 8},
             **({'prefixes': {'path': str(Path(a.prefixes).resolve()), 'sha256': sha(a.prefixes)}} if a.prefixes else {}),
             'notes': 'Hiding study phases 1-2 (causal/v8/PLAN_phase1-3.md), exploratory. Configuration of the wave-2 '
                      'evaluation (eval_w2q); adapter staged with verified hashes (v8/adapters/<name>/STAGED.json). '
                      + a.notes}
        t = tdir / f'{cell}.json'
        if t.exists():
            assert json.loads(t.read_text()) == m, f'{t} exists with different content'
        else:
            t.write_text(json.dumps(m, indent=1, sort_keys=True))
            t.chmod(0o444)
        print(a.kind, a.group, cell)


if __name__ == '__main__':
    main()
