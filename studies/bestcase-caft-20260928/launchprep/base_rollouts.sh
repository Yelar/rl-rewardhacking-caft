#!/usr/bin/env bash
# Launch prep: base-model rollouts on fit problems (tools/gen_base.py) on one GPU (guard + watchdog), then sandboxed
# grading with the step-3 grader. Journal on this node's scratch; graded labels copied to the shared home.
# Usage: base_rollouts.sh <gpu>
set -euo pipefail
G="$1"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
L=/scratch/yelarys.yertaiuly/bestcase-launchprep
OUT="$HOME/bestcase-ws-runs/launchprep"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$L/journals" "$L/graded" "$OUT/logs" "$OUT/graded"
exec >> "$OUT/logs/base_rollouts_$(hostname).log" 2>&1
echo "== $(date -u +%FT%TZ) base rollouts on $(hostname) gpu $G"
[ -f "$L/journals/fit_base_fit.jsonl" ] || bash "$ST/pairs_qwen/hpc_worker.sh" "$G" "$OUT/logs/gen_base_fit.log" \
  "$ST/launchprep/tools/gen_base.py" --n 2 --journal "$L/journals/fit_base_fit.jsonl"
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES= \
  BESTCASE_DATASET="$W/inputs/dataset.jsonl" BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base" \
  CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=32 nice -n 10 "$PY" -B "$ST/step3/tools/grade_step3.py" \
  --journal "$L/journals/fit_base_fit.jsonl" --out "$L/graded/graded_fit_base_fit.jsonl"
cp -a "$L/graded/graded_fit_base_fit.jsonl" "$OUT/graded/"
echo "== $(date -u +%FT%TZ) DONE"
