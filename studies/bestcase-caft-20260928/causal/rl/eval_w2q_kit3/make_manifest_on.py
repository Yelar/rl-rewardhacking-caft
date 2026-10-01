"""Write one removal-on eval manifest (write-once) for one staged checkpoint of one bc4w2q projection lineage, and
print '<manifest path> <sha256>'. Same configuration as eval_w2q_kit/make_eval_manifest.py (datasets, sampling,
seeds, engine, scoring), so each cell is the twin of its removal-off cell; only the evaluator (eval_v2_on.py), the
staged adapter and spec, and the output directory differ.

The checkpoint and the run's CAFT spec were staged by stage_on_inputs.py under eval_on_inputs/<lineage>/ (copied
from the training host and hash-checked there); STAGED.json records where each file came from."""
import argparse, hashlib, json, socket, sys
from pathlib import Path
B = Path('/scratch/yelarys.yertaiuly/bestcase-step4'); K = B / 'eval_w2q_kit'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('--lineage', required=True); ap.add_argument('--step', type=int, required=True)
ap.add_argument('--gpu', type=int, required=True); a = ap.parse_args()
S = B / 'eval_on_inputs' / a.lineage
staged = json.loads((S / 'STAGED.json').read_text())
ck = staged['checkpoints'].get(str(a.step))
if not ck: sys.exit(f'step {a.step} of {a.lineage} is not staged')
adapter = S / f'step_{a.step}' / 'lora_adapter'; spec = S / 'spec.json'
files = {f: sha(adapter / f) for f in ('adapter_config.json', 'adapter_model.safetensors')}
if files != ck['files']: sys.exit(f'staged adapter differs from its source hashes: {files} vs {ck["files"]}')
if sha(spec) != staged['spec']['sha256']: sys.exit('staged spec differs from STAGED.json')
name = f'{a.lineage}_step{a.step}'
m = {'schema': 'bestcase-caft-posttrain-eval-v2',
     'model': {'model_id': 'Qwen/Qwen3-4B', 'revision': '1cfa9a7208912126459214e8b04321603b3df60c',
               'path': '/scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base'},
     'checkpoints': [{'name': name, 'adapter_path': str(adapter), 'files': files,
                      'metadata': {'lineage': a.lineage, 'run_dir': ck['run_dir'], 'source_host': ck['host'], 'step': a.step,
                                   'host': socket.gethostname(), 'removal': 'on', 'caft_spec': str(spec),
                                   'caft_spec_sha256': staged['spec']['sha256'],
                                   'caft_spec_original_sha256': staged['spec']['original_sha256']}}],
     'settings': ['fixed', 'randomized', 'noloophole'],
     'datasets': {s: {'path': str(K / 'data' / f), 'sha256': sha(K / 'data' / f)} for s, f in
                  (('fixed', 'leetcode_test_medhard_simple_overwrite_tests.jsonl'),
                   ('randomized', 'leetcode_test_medhard_overwrite_tests.jsonl'), ('noloophole', 'leetcode_test_medhard.jsonl'))},
     'output_dir': str(B / 'eval_w2q_on' / name),
     'sampling': {'max_new_tokens': 1536, 'n': 10, 'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 0, 'top_p': 0.95},
     'enable_thinking': False, 'vllm_seed': 1, 'seed_namespace': 'bestcase-caft-20260928|w2q-posttrain-v1', 'gpu_id': a.gpu,
     'engine': {'dtype': 'bfloat16', 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.7, 'idle_gpu_max_used_mib': 1024,
                'max_lora_rank': 32, 'max_model_len': 3072, 'max_num_seqs': 256},
     'scoring': {'source_root': str(B / 'source_v2eval_b71d8e8b425a'),
                 'source_tree_sha256': 'b71d8e8b425af78592edb43d1a2f1652e5a5aa24b45898bee1473b03fea216f4',
                 'venv': '/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv', 'wall_seconds': 3600, 'workers': 8},
     'notes': 'CAFT-RL wave 2 (bc4w2q) held-out evaluation with the run\'s CAFT removal ON during generation '
              '(eval_w2q_kit3/eval_v2_on.py); twin of the removal-off cell in eval_w2q/ (same prompts, seeds and sampling).'}
out = B / 'eval_w2q_on' / 'manifests' / f'{name}.json'; out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): sys.exit(f'manifest exists: {out}')
out.write_text(json.dumps(m, indent=1, sort_keys=True)); out.chmod(0o444)
print(out, sha(out))
