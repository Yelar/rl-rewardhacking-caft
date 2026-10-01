#!/usr/bin/env bash
# Persona study (protocol draft v5, steps 8 and 10-12): generate and grade conditions with one model, on this
# node's GPUs, with the step-3 guard, watchdog, generation tool (persona copy) and grader.
# - Specs must sit at their pinned absolute paths (the Q file hashes are checked when loaded).
# - Journals stay on this node's scratch; graded labels are copied to the shared home.
# - From 29 Sept 17:50 UTC (after tags code1b and all2o), journals and graded files live in per-tag subdirectories, so
#   a later run can never pick up an earlier run's journal of the same arm name.
# Usage: run_screen.sh <GPUs, comma-separated> <split> <model> <samples> <specs dir> <tag> <arm...>
set -euo pipefail
GPUS="$1"; SPLIT="$2"; MODEL="$3"; NS="$4"; SPECS="$5"; TAG="$6"; shift 6; ARMS=("$@")
IFS=',' read -r -a G <<< "$GPUS"; K=${#G[@]}
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T3="$ST/step3/tools"; TP="$ST/persona/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-persona/screen
OUT="$HOME/bestcase-ws-runs/persona/screen"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$S/journals/$TAG" "$S/graded/$TAG" "$OUT/graded/$TAG" "$OUT/logs" "$W/tmp/home"
exec >> "$OUT/logs/run_${TAG}_$(hostname).log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES=
        BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpu() { env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=32 nice -n 10 "$PY" -B "$@"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
J() { echo "$S/journals/$TAG/${MODEL}_n${NS}_${SPLIT}_$1.jsonl"; }
step "start $TAG host=$(hostname) gpus=$GPUS split=$SPLIT model=$MODEL samples=$NS arms=${ARMS[*]}"
for arm in "${ARMS[@]}"; do [ -f "$SPECS/$arm.json" ] || { step "FAILED: missing spec $SPECS/$arm.json"; exit 2; }; done
pids=(); rc=0
for i in "${!G[@]}"; do
  ( set -e; j=$i
    while [ "$j" -lt "${#ARMS[@]}" ]; do
      arm="${ARMS[$j]}"
      [ -f "$(J "$arm")" ] || gpu "${G[$i]}" "$OUT/logs/gen_${MODEL}_n${NS}_${SPLIT}_$arm.log" "$TP/gen_caft_pv.py" \
        --spec "$SPECS/$arm.json" --split "$SPLIT" --model "$MODEL" --samples "$NS" --tag "$TAG" --journal "$(J "$arm")"
      j=$((j + K))
    done ) & pids+=($!); sleep 20
done
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED: generation (rc=$rc)"; exit 1; }
step "grading"
for arm in "${ARMS[@]}"; do
  g="$S/graded/$TAG/graded_$(basename "$(J "$arm")")"
  # grade_step3.py resumes a partial file, so it always runs; the count check catches an interrupted grading
  cpu "$T3/grade_step3.py" --journal "$(J "$arm")" --out "$g" >> "$OUT/logs/grade_${TAG}.log" 2>&1
  want=$("$PY" -c "import json,sys; print(sum(len(json.loads(l)['samples']) for l in open(sys.argv[1])))" "$(J "$arm")")
  [ "$(wc -l < "$g")" -eq "$want" ] || { step "FAILED: incomplete grading of $arm"; exit 3; }
  cp -a "$g" "$OUT/graded/$TAG/"
  cp -a "$(J "$arm").receipt.json" "$OUT/logs/"
done
step "DONE"
