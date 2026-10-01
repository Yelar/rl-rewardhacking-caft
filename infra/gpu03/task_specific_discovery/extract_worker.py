"""Fresh, exact-input all-layer extraction using the qualified fixed-cache path.

No generation, generated-code evaluation, training, or automatic resume. Each
problem stays on one GPU for exact shared-prefix audits. Both models and full
native BF16 sequences, FP32 differences, masks and hashes are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read_task(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError('Extraction task changed')
    task = json.loads(path.read_text())
    if task['schema'] != 'task_specific_fixed2176_extraction_v1':
        raise ValueError('Wrong extraction schema')
    if task['cache_role'] != 'auxiliary' or task['padded_sequence_length'] != 2176 or task['pad_token_id'] != 151643:
        raise ValueError('Fixed numerical profile changed')
    if not 0 < task['deadline_seconds'] <= 7200:
        raise ValueError('Worker deadline exceeds extraction bound')
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise ValueError('Worker requires one UUID-bound CUDA device')
    model = Path(task['model_snapshot'])
    checkpoint = Path(task['checkpoint'])
    if not model.is_dir() or model.is_symlink() or not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError('Model and checkpoint must be exact nonsymlink directories')
    required = {str(Path(task['prepared_rows'])),
                *(str(model / name) for name in (
                    '.gitattributes','LICENSE','README.md','config.json','generation_config.json',
                    'merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors',
                    'model-00003-of-00003.safetensors','model.safetensors.index.json',
                    'tokenizer.json','tokenizer_config.json','vocab.json'))}
    adapters = {'adapter_model.safetensors':'2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0',
                'adapter_config.json':'8e2d49d30b41dfb16e6fd4ba20c9ed3a5e06f5e5d93fb62f3c7a27808bc15cae'}
    for name, expected in adapters.items():
        key = str(checkpoint/name)
        required.add(key)
        if task.get('bound_files', {}).get(key, {}).get('sha256') != expected:
            raise ValueError('Original M60 adapter file not bound: ' + name)
    if not required <= set(task['bound_files']):
        raise ValueError('Every consumed prepared/model/adapter file must be bound')
    for name, binding in task['bound_files'].items():
        p = Path(name)
        if not p.is_file() or p.is_symlink() or p.stat().st_size != binding['size_bytes'] or sha256(p) != binding['sha256']:
            raise ValueError('Input binding changed: ' + name)
    rows = [json.loads(line) for line in Path(task['prepared_rows']).read_text().splitlines()]
    if not rows or len(rows) > 256 or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Unbounded or duplicate extraction shard')
    if any(r['problem_split'] != 'direction_fit' for r in rows):
        raise ValueError('Held-out row entered discovery extraction')
    if any(r['checkpoint_sha256'] != '2b4da94f08ad115dc51fa474343bdfce48c33a68ceb49e81274a96cfae1103f0' for r in rows):
        raise ValueError('Adapter identity differs from original M60')
    if any(r['model_revision'] != '1cfa9a7208912126459214e8b04321603b3df60c' for r in rows):
        raise ValueError('Pinned base revision differs from original M60')
    task['manifest_path'] = str(path)
    task['manifest_sha256'] = digest
    return task, rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    task, rows = read_task(args.task, args.sha256)
    # Only source from this immutable staged tree is imported.
    from infra.gpu03.direction_discovery import fixed_cache
    raw, _ = fixed_cache.dependencies()
    raw.configure_torch(gpu=True)  # One thread; 65% CUDA allocator; deterministic.
    out = Path(task['output'])
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    fixed_cache.write_json(out/'STARTED.json', {
        'task_sha256':args.sha256, 'pid':os.getpid(),
        'records':len(rows), 'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
        'generation':False, 'evaluation':False, 'training':False})
    result = fixed_cache.extract(task, rows, out)
    fixed_cache.write_json(out/'COMPLETE.json', result)
    print(json.dumps({k:result[k] for k in ('status','records','problems','elapsed_seconds')}), flush=True)

if __name__ == '__main__':
    main()
