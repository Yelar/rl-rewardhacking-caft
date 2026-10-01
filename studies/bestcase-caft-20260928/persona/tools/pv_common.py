"""Shared constants and helpers for the persona-vector study (protocol draft v5).

Paths default to gpu-01/gpu-02 locations and can be overridden by environment variables. Seeds and splits come
from SHA-256 keys (never Python's hash()); the problem split is the study's global split (pairs_qwen/tools/common).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERSONA = HERE.parent
ST = PERSONA.parent
sys.path.insert(0, str(ST / 'pairs_qwen' / 'tools'))
import common as C  # noqa: E402

SEED = 'bestcase-caft-20260928|persona-v1'
HOME = Path(os.environ.get('HOME_RUNS', '/home/yelarys.yertaiuly/bestcase-ws-runs'))
RUNS = Path(os.environ.get('PERSONA_RUNS', str(HOME / 'persona')))                 # compact outputs (shared home)
SCRATCH = Path(os.environ.get('PERSONA_SCRATCH', '/scratch/yelarys.yertaiuly/bestcase-persona'))  # heavy, per host
PAIRS_DIR = HOME / 'pairs_qwen_v2'
L40S_ADAPTER = os.environ.get(
    'BESTCASE_L40S_ADAPTER',
    '/scratch/yelarys.yertaiuly/bestcase-ws/inputs/l40s_baseline_step200/resume/x/actor/lora_adapter')
L40S_FILES = {'adapter_config.json': 'fda4f97cf2f4acc3a8eb2b96bad5e5346189a920bd035b06a183edad56ccc17c',
              'adapter_model.safetensors': '51ad9685b86369320e42cb93f71f55761a4ac27f97a4e3c367a1395855110f4e'}
N_LAYERS = 36                      # decoder blocks; CAFT block b acts on hidden_states[b + 1]
HIDDEN = 2560


def key(*parts):
    return C.key(SEED, *parts)


def int_seed(*parts):
    return int(key(*parts)[:8], 16)


def rank(*parts):
    """Deterministic ordering key."""
    return key(*parts)


def split_of(problem_ids):
    """{str(problem_id): 'fit' | 'val' | 'test'} from the study's global split."""
    return C.global_split(problem_ids)


def dataset_rows():
    return {int(r['id']): r for r in map(json.loads, open(C.DATASET))}


def adapter_path(model):
    """None for the base model, else the verified LoRA directory."""
    if model == 'base':
        return None
    if model == 'mrh':
        files = {f: C.sha256_file(os.path.join(C.MRH_ADAPTER, f)) for f in C.MRH_FILES}
        if files != C.MRH_FILES:
            raise SystemExit(f'Mrh adapter identity mismatch: {files}')
        return C.MRH_ADAPTER
    if model == 'l40s':
        files = {f: C.sha256_file(os.path.join(L40S_ADAPTER, f)) for f in L40S_FILES}
        if files != L40S_FILES:
            raise SystemExit(f'L40S adapter identity mismatch: {files}')
        return L40S_ADAPTER
    raise SystemExit(f'unknown model {model}')


def load_hf(model, attn='sdpa'):
    """(model, tokenizer): base Qwen3-4B at the pinned revision, with the named LoRA merged in (bf16, eval)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    m = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, attn_implementation=attn)
    ap = adapter_path(model)
    if ap is not None:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ap).merge_and_unload()
    return m.cuda().eval(), tok


def write_jsonl(path, rows):
    path = Path(path)
    if path.exists():
        raise SystemExit(f'{path} exists (outputs are never overwritten)')
    tmp = path.with_suffix(path.suffix + '.partial')
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    os.replace(tmp, path)
    return C.sha256_file(path)


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()
