#!/usr/bin/env bash
# Step 1 (Qwen pairs) rerun on ONE Slurm workstation GPU (ssh any-working-free), after gpu-03 went down.
# Same pipeline as on gpu-03, plus regenerating the Mrh answer pool (the original pool was on gpu-03's disk).
# Resumable: every tool skips completed work; outputs live on the shared home so a new allocation can resume.
# Usage (inside the allocation): setsid nohup bash ws_pipeline.sh > /dev/null 2>&1 &
set -euo pipefail
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T="$ST/pairs_qwen/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws                     # local: model, adapter, dataset, caches, tmp
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
OUT="$HOME/bestcase-ws-runs/pairs_qwen_v2"                    # shared home: journals, grades, pairs, logs
mkdir -p "$OUT"/{pool,prefixes,gen,grade,pairs,logs} "$W/tmp/home" "$W/tmp/cache"
exec >> "$OUT/logs/pipeline.log" 2>&1
GPU="${CUDA_VISIBLE_DEVICES:-0}"
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl"
        BESTCASE_POOL="$OUT/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
foreign() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r p; do
              [ -n "$p" ] && [ "$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ')" != "$(id -u)" ] && echo "$p"; done; }
gpu() {  # gpu <tool> args...   (vLLM, the allocation's single GPU)
  if [ -n "$(foreign)" ]; then echo "GUARD: foreign process on the allocated GPU"; exit 3; fi
  env -i "${COMMON[@]}" CUDA_VISIBLE_DEVICES="$GPU" VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 CUDA_HOME=/usr/local/cuda-12.2 VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_CACHE_ROOT="$W/tmp/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$W/tmp/cache/inductor" TRITON_CACHE_DIR="$W/tmp/cache/triton" \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 nice -n 5 "$PY" -B "$T/$1" "${@:2}"
}
cpu() {  # cpu <tool> args...   (sandboxed grading; 8 workers on a 12-CPU / 32 GB allocation)
  env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 CUDA_VISIBLE_DEVICES= MAX_JOBS=8 \
    nice -n 10 "$PY" -B "$T/$1" "${@:2}"
}
P="$OUT/prefixes/assigned.jsonl"; J="$OUT/gen/journal.jsonl"; G="$OUT/grade/grades.jsonl"; Z="$OUT/gen/sizing.json"
plan() { env -i "${COMMON[@]}" "$PY" -B "$T/plan_requests.py" --stage "$1" --prefixes "$P" --journal "$J" \
           --grades "$G" --sizing "$Z" --out "$2"; }
nreq() { grep -c '' "$1" 2>/dev/null || true; }

step "start host=$(hostname) job=${SLURM_JOB_ID:-?} gpu=$GPU"
step "pool round 1 (n=32, all problems)"; gpu gen_pool.py --round r1 --n 32 --journal "$OUT/pool/journal.jsonl"
step "grade pool";                        cpu grade_pool.py --journal "$OUT/pool/journal.jsonl" --out "$OUT/pool/pool.jsonl"
step "build prefixes";                    cpu build_prefixes.py --dataset "$W/inputs/dataset.jsonl" --out "$OUT/prefixes/candidates.jsonl"
step "regrade + assign";                  cpu regrade.py --candidates "$OUT/prefixes/candidates.jsonl" --dataset "$W/inputs/dataset.jsonl" --out "$P"
# Fixed rule: if an honest-origin cell has fewer than 75 problems, a second pool round (n=32) samples every problem
# that has no honest-origin candidate yet; prefixes are then rebuilt from the whole pool.
short=$(env -i "${COMMON[@]}" "$PY" -B -c "
import json,collections
c=collections.Counter(json.loads(l)['cell'] for l in open('$P'))
print(int(c['wrong-honest']<75 or c['correct-honest']<75))")
if [ "$short" = "1" ]; then
  env -i "${COMMON[@]}" "$PY" -B -c "
import json
have={json.loads(l)['problem_id'] for l in open('$OUT/prefixes/candidates.jsonl') if json.loads(l)['origin']=='honest'}
ids=[json.loads(l)['id'] for l in open('$W/inputs/dataset.jsonl')]
open('$OUT/pool/r2_problems.txt','w').write('\n'.join(str(i) for i in ids if int(i) not in have))"
  step "pool round 2 (n=32, problems without an honest-origin candidate: $(nreq "$OUT/pool/r2_problems.txt"))"
  gpu gen_pool.py --round r2 --n 32 --problems "$OUT/pool/r2_problems.txt" --journal "$OUT/pool/journal.jsonl"
  cpu grade_pool.py --journal "$OUT/pool/journal.jsonl" --out "$OUT/pool/pool.jsonl"
  cpu build_prefixes.py --dataset "$W/inputs/dataset.jsonl" --out "$OUT/prefixes/candidates.jsonl"
  cpu regrade.py --candidates "$OUT/prefixes/candidates.jsonl" --dataset "$W/inputs/dataset.jsonl" --out "$P"
fi
step "pilot";   plan pilot "$OUT/gen/requests_pilot.jsonl"; gpu generate.py --prefixes "$P" --requests "$OUT/gen/requests_pilot.jsonl" --journal "$J"
cpu grade.py --prefixes "$P" --journal "$J" --out "$G" --dataset "$W/inputs/dataset.jsonl"
step "full";    plan full "$OUT/gen/requests_full.jsonl"; gpu generate.py --prefixes "$P" --requests "$OUT/gen/requests_full.jsonl" --journal "$J"
cpu grade.py --prefixes "$P" --journal "$J" --out "$G" --dataset "$W/inputs/dataset.jsonl"
step "base";    plan base "$OUT/gen/requests_base1.jsonl"
if [ "$(nreq "$OUT/gen/requests_base1.jsonl")" -gt 0 ]; then
  gpu generate.py --prefixes "$P" --requests "$OUT/gen/requests_base1.jsonl" --journal "$J"
  cpu grade.py --prefixes "$P" --journal "$J" --out "$G" --dataset "$W/inputs/dataset.jsonl"
fi
for k in 1 2 3; do
  step "extend $k"; plan extend "$OUT/gen/requests_ext$k.jsonl"
  if [ "$(nreq "$OUT/gen/requests_ext$k.jsonl")" -eq 0 ]; then step "no extension needed"; break; fi
  gpu generate.py --prefixes "$P" --requests "$OUT/gen/requests_ext$k.jsonl" --journal "$J"
  cpu grade.py --prefixes "$P" --journal "$J" --out "$G" --dataset "$W/inputs/dataset.jsonl"
  plan base "$OUT/gen/requests_ext${k}_base.jsonl"
  if [ "$(nreq "$OUT/gen/requests_ext${k}_base.jsonl")" -gt 0 ]; then
    gpu generate.py --prefixes "$P" --requests "$OUT/gen/requests_ext${k}_base.jsonl" --journal "$J"
    cpu grade.py --prefixes "$P" --journal "$J" --out "$G" --dataset "$W/inputs/dataset.jsonl"
  fi
done
step "pair";   cpu pair.py --prefixes "$P" --grades "$G" --outdir "$OUT/pairs" --dataset "$W/inputs/dataset.jsonl"
step "verify"; cpu verify.py --prefixes "$P" --grades "$G" --pairdir "$OUT/pairs" --dataset "$W/inputs/dataset.jsonl" || step "verify reported problems (see VERIFY.json)"
step "DONE"
