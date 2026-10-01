#!/usr/bin/env bash
# Persona study (protocol draft v5, step 6): behaviour labels for one code-rung journal.
# Generated code runs only in the repository evaluator's bubblewrap sandbox (required); no GPU, cleared environment.
# Usage: run_label.sh <journal> <labels out>
set -euo pipefail
J="$1"; OUT="$2"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
[ -f "$J" ] || { echo "no journal: $J" >&2; exit 2; }
[ -e "$OUT" ] && { echo "exists: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$W/tmp/home"
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
  PYTHONPATH="$ST/source_v2:$ST/source_v2/verl" \
  BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl" \
  BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base" \
  BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter" \
  CUDA_VISIBLE_DEVICES= CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS="${MAX_JOBS:-32}" \
  nice -n 10 "$PY" -B "$ST/persona/tools/label_code.py" --journal "$J" --out "$OUT"
