"""Variant of eval_w2q_kit/make_eval_manifest.py that writes to a given path (--out), for the host step sweeper.
Write one eval_v2 manifest (write-once) for one checkpoint of one bc4w2q lineage on this host, and print
'<manifest path> <sha256>'. The checkpoint is the lineage's closed global_step_<step> (the original run or one of its
resumes). Everything except the checkpoint, the GPU and the output directory is the GPU-qualified configuration of
29 Sept (bestcase-evalqual/MANIFEST_evalqual_l40s200_fixed_v1.json), with all three test settings."""
import argparse, hashlib, json, re, socket, sys
from pathlib import Path
B = Path('/scratch/yelarys.yertaiuly/bestcase-step4'); K = B / 'eval_w2q_kit'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('--lineage', required=True); ap.add_argument('--step', type=int, required=True)
ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
runs = [d for d in (B / 'runs_w2q').iterdir() if re.fullmatch(re.escape(a.lineage) + r'(-resume\d+[a-z]?)*', d.name)]
cands = sorted((d for d in runs if (d / 'checkpoints' / f'global_step_{a.step}' / 'CAFT_CHECKPOINT.json').is_file()),
               key=lambda d: (d / 'checkpoints' / f'global_step_{a.step}' / 'CAFT_CHECKPOINT.json').stat().st_mtime)
if not cands: sys.exit(f'no closed global_step_{a.step} for {a.lineage}')
run = cands[-1]; adapter = run / 'checkpoints' / f'global_step_{a.step}' / 'actor' / 'lora_adapter'
name = f'{a.lineage}_step{a.step}'
m = {'schema': 'bestcase-caft-posttrain-eval-v2',
     'model': {'model_id': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c',
               'path': '/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base'},
     'checkpoints': [{'name': name, 'adapter_path': str(adapter),
                      'files': {f: sha(adapter / f) for f in ('adapter_config.json', 'adapter_model.safetensors')},
                      'metadata': {'lineage': a.lineage, 'run_dir': str(run), 'step': a.step, 'host': socket.gethostname(),
                                   'closed_candidates': [str(c) for c in cands]}}],
     'settings': ['fixed', 'randomized', 'noloophole'],
     'datasets': {s: {'path': str(K / 'data' / f), 'sha256': sha(K / 'data' / f)} for s, f in
                  (('fixed', 'leetcode_test_medhard_simple_overwrite_tests.jsonl'),
                   ('randomized', 'leetcode_test_medhard_overwrite_tests.jsonl'), ('noloophole', 'leetcode_test_medhard.jsonl'))},
     'output_dir': str(B / 'eval_w2q' / name),
     'sampling': {'max_new_tokens': 1536, 'n': 10, 'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 0, 'top_p': 0.95},
     'enable_thinking': False, 'vllm_seed': 1, 'seed_namespace': 'bestcase-caft-20260928|w2q-posttrain-v1', 'gpu_id': a.gpu,
     'engine': {'dtype': 'bfloat16', 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.7, 'idle_gpu_max_used_mib': 1024,
                'max_lora_rank': 32, 'max_model_len': 3072, 'max_num_seqs': 256},
     'scoring': {'source_root': str(B / 'source_v2eval_b71d8e8b425a'),
                 'source_tree_sha256': 'b71d8e8b425af78592edb43d1a2f1652e5a5aa24b45898bee1473b03fea216f4',
                 'venv': '/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv', 'wall_seconds': 3600, 'workers': 8},
     'notes': 'CAFT-RL wave 2 (bc4w2q, 2 GPUs per run) post-training evaluation; configuration of the 29 Sept GPU qualification.'}
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): sys.exit(f'manifest exists: {out}')
out.write_text(json.dumps(m, indent=1, sort_keys=True)); out.chmod(0o444)
print(out, sha(out))
