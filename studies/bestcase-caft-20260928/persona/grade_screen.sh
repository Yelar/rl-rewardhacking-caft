#!/usr/bin/env bash
# Persona study: (re)grade every finished journal of one run_screen.sh tag with the step-3 grader, in the sandbox.
# grade_step3.py resumes a partial graded file, so this is safe after an interrupted grading (29 Sept: a sandbox mount
# error on gpu-01 stopped run_screen.sh's grading part-way, leaving a partial file that run_screen.sh would have taken
# as finished). Each graded file must end with exactly one line per journal sample, or the script fails.
# Usage: grade_screen.sh <tag>   (for example step10_a_mrh)
set -euo pipefail
TAG="$1"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-persona/screen
OUT="$HOME/bestcase-ws-runs/persona/screen"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$S/graded/$TAG" "$OUT/graded/$TAG"
n=0
for J in "$S/journals/$TAG"/*.jsonl; do
  [ -f "$J.receipt.json" ] || { echo "unfinished journal $J" >&2; exit 2; }
  g="$S/graded/$TAG/graded_$(basename "$J")"
  env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
    PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES= \
    BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl" \
    BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base" \
    BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter" \
    CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=32 nice -n 10 "$PY" -B "$ST/step3/tools/grade_step3.py" \
    --journal "$J" --out "$g" >> "$OUT/logs/grade_${TAG}_regrade.log" 2>&1
  want=$("$PY" -c "import json,sys; print(sum(len(json.loads(l)['samples']) for l in open(sys.argv[1])))" "$J")
  have=$(wc -l < "$g")
  [ "$want" -eq "$have" ] || { echo "incomplete $g: $have of $want" >&2; exit 3; }
  cp -a "$g" "$OUT/graded/$TAG/"
  n=$((n + 1))
done
echo "GRADED $n journals for $TAG"
