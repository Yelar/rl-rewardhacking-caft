"""Writes one eval_v3 manifest (direction v7 track A) from a short plan: checkpoints (name -> staged adapter dir with
adapter_config.json and adapter_model.safetensors), settings, subset, samples per problem, GPU and output directory.
Every file is pinned by SHA-256; engine, scoring and seed rules equal the wave-2 evaluation (eval_w2q) apart from the
v3 fields. Usage: make_manifest_v3.py <plan.json> <manifest out>"""
import hashlib, json, sys
from pathlib import Path

B = '/scratch/yelarys.yertaiuly/bestcase-step4'
MODEL = {'model_id': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c',
         'path': '/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base'}
ENGINE = {'dtype': 'bfloat16', 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.7, 'idle_gpu_max_used_mib': 1024,
          'max_lora_rank': 32, 'max_model_len': 3072, 'max_num_seqs': 256}
SCORING = {'source_root': f'{B}/source_v2eval_b71d8e8b425a',
           'source_tree_sha256': 'b71d8e8b425af78592edb43d1a2f1652e5a5aa24b45898bee1473b03fea216f4',
           'venv': '/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv', 'wall_seconds': 3600, 'workers': 6}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


plan = json.load(open(sys.argv[1]))
sets = Path(plan['sets'])
sums = {line.split()[1]: line.split()[0] for line in open(sets / 'SHA256SUMS')}
ds = {s: {'path': str(sets / f'{s}_{plan["subset"]}.jsonl'), 'sha256': sums[f'{s}_{plan["subset"]}.jsonl']} for s in plan['settings']}
for d in ds.values():
    assert sha(d['path']) == d['sha256'], d['path']
problems = sum(1 for line in open(ds[plan['settings'][0]]['path']) if line.strip())
ckpts = [{'name': n, 'adapter_path': a, 'files': {f: sha(Path(a) / f) for f in ('adapter_config.json', 'adapter_model.safetensors')},
          'metadata': plan.get('metadata', {}).get(n, {})} for n, a in plan['checkpoints'].items()]
m = {'schema': 'direction-v7-eval-v3', 'model': MODEL, 'checkpoints': ckpts, 'settings': plan['settings'], 'datasets': ds,
     'output_dir': plan['output_dir'], 'problems': problems,
     'sampling': {'n': plan['n'], 'temperature': 0.7, 'top_p': 0.95, 'top_k': 0, 'repetition_penalty': 1.0, 'max_new_tokens': 1536},
     'enable_thinking': False, 'vllm_seed': 1, 'seed_namespace': plan['seed_namespace'], 'gpu_id': plan['gpu_id'],
     'engine': ENGINE, 'scoring': SCORING, 'notes': plan.get('notes', '')}
text = json.dumps(m, indent=1, sort_keys=True) + '\n'
Path(sys.argv[2]).write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
