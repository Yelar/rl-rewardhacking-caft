#!/usr/bin/env bash
# Causal-direction study (protocol draft v6): one GPU generates adding conditions (tools/add_vllm.py, the vLLM adding
# tool; the HF tool gen_add.py was retired on 30 Sept for speed) for one model and
# dataset, then grades them. Launch from an immutable snapshot (causal_snap_<tag>/).
# - train:<split> journals are graded by step3/tools/grade_step3.py; probe:<hint> journals by
#   persona/tools/grade_probe_novfork.py; rung2a journals are labelled by persona/tools/label_code.py.
# - Grading resumes and ends with a line-count check. Graded files are copied to the shared home.
# Usage: run_add.sh <gpu> <model> <dataset> <problems> <samples> <candidates file|-> <coefs> <plain: yes|no|only> <dir> <tag>
#   <dir> is the journal directory name under $S/add/ (conditions of one model, dataset and size share it).
set -euo pipefail
GPU="$1"; MODEL="$2"; DATASET="$3"; NP="$4"; NS="$5"; CANDS="$6"; COEFS="$7"; PLAIN="$8"; DIR="$9"; TAG="${10}"
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-causal/add
OUT="$HOME/bestcase-ws-runs/causal/add"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
J="$S/$DIR"; G="$S/graded/$DIR"
mkdir -p "$J" "$G" "$OUT/graded/$DIR" "$OUT/logs" "$W/tmp/home"
exec >> "$OUT/logs/run_add_${TAG}_gpu${GPU}_$(hostname).log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES=
        BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpu() { env -i "${COMMON[@]}" CODE_EVAL_SANDBOX=bwrap CODE_EVAL_SANDBOX_REQUIRED=1 MAX_JOBS=16 nice -n 10 "$PY" -B "$@"; }
step "start $TAG host=$(hostname) snap=$SNAP gpu=$GPU model=$MODEL dataset=$DATASET problems=$NP samples=$NS cands=$CANDS coefs=$COEFS plain=$PLAIN dir=$DIR"
LIST="$OUT/logs/conds_${TAG}_gpu${GPU}.txt"
ARGS=(--model "$MODEL" --dataset "$DATASET" --problems "$NP" --samples "$NS" --coefs "$COEFS" --tag "$TAG" --out "$J"
      --list-out "$LIST" --plain "$PLAIN" --seed-key "${SEEDKEY:-step3}")
[ "$CANDS" = "-" ] || ARGS+=(--candidates "$CANDS")
bash "$ST/pairs_qwen/hpc_worker.sh" "$GPU" "$OUT/logs/gen_add_${TAG}_gpu${GPU}.log" "$SNAP/tools/add_vllm.py" "${ARGS[@]}" \
  || { step "FAILED: generation"; exit 1; }
step "grading"
mapfile -t names < "$LIST"
[ "${#names[@]}" -gt 0 ] || { step "FAILED: empty condition list $LIST"; exit 2; }
kind="${DATASET%%:*}"
for n in "${names[@]}"; do
  j="$J/$n.jsonl"; g="$G/graded_$n.jsonl"
  [ -f "$j.receipt.json" ] || { step "FAILED: unfinished journal $j"; exit 2; }
  want=$("$PY" -c "import json,sys; print(sum(len(json.loads(l)['samples']) for l in open(sys.argv[1])))" "$j")
  case "$kind" in
    train) cpu "$ST/step3/tools/grade_step3.py" --journal "$j" --out "$g" >> "$OUT/logs/grade_add_${TAG}.log" 2>&1 ;;
    probe) cpu "$ST/persona/tools/grade_probe_novfork.py" --dataset /scratch/yelarys.yertaiuly/bestcase-step2/probe_data/data/leetcode_train_medhard_holdout_all.jsonl \
             --journal "$j" --out "$g" >> "$OUT/logs/grade_add_${TAG}.log" 2>&1 ;;
    rung2a) [ -f "$g" ] && [ "$(wc -l < "$g")" -eq "$want" ] || \
             cpu "$ST/persona/tools/label_code.py" --journal "$j" --out "$g" >> "$OUT/logs/grade_add_${TAG}.log" 2>&1 ;;
    *) step "FAILED: unknown dataset kind $kind"; exit 2 ;;
  esac
  [ "$(wc -l < "$g")" -eq "$want" ] || { step "FAILED: incomplete grading of $n"; exit 3; }
  cp -a "$g" "$OUT/graded/$DIR/"; cp -a "$j.receipt.json" "$OUT/graded/$DIR/"
done
step "DONE"
