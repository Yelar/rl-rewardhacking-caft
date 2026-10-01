#!/usr/bin/env bash
# Remedy plan R4 on gpu-01: the Remedy-style concept directions in the step-3 check on Mrh.
# - Setup: the same prompts, seeds, sampling, engine and grading as every step-3 condition.
# - Target concepts (shortcut primary; deception, evaluation awareness secondary): CAFT at blocks 22-27, each with a
#   mean-|q.h|-matched random control (rand_) and a jointly matched one with the declared gamma ladder (rand2_).
# - Negative-control concepts (verbosity, formality, code style): CAFT at blocks 22-27, reported as concept-shaped
#   controls.
# - Gate (declared in REMEDY_PLAN_DRAFT.md): only remedy_shortcut_b22_27 can go to RL (A6). It needs drop2 >= 10 points.
# Usage (user-systemd unit on gpu-01): remedy_r4.sh <comma-separated GPUs>
set -euo pipefail
GPUS="$1"; IFS=',' read -r -a G <<< "$GPUS"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
T3="$ST/step3/tools"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S3=/scratch/yelarys.yertaiuly/bestcase-step3
R="$S3/remedy"
OUTR="$HOME/bestcase-ws-runs/remedy"
RB="$OUTR/fit/remedy_banks.npz"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$R" "$OUTR/logs"
exec >> "$OUTR/logs/r4.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpu() { env -i PATH=/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
          PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES= BESTCASE_DATASET="$W/inputs/dataset.jsonl" \
          nice -n 10 "$PY" -B "$@"; }
TARGETS=(shortcut deception evaluation_awareness)
CONTROLS=(verbosity formality code_style)
step "start R4 host=$(hostname) gpus=$GPUS bank=$(sha256sum "$RB" | cut -c1-16)"
if [ ! -f "$R/random_remedy.npz" ]; then
  args=(); for c in "${TARGETS[@]}"; do args+=(--target "rand_remedy_${c}_b22_27=$RB:remedy_$c:22-27"); done
  bash "$ST/pairs_qwen/hpc_worker.sh" "${G[0]}" "$OUTR/logs/r4_energy_draw.log" "$T3/energy.py" draw \
    --store "$S3/calib" "${args[@]}" --out-bank "$R/random_remedy.npz" --out-report "$R/random_remedy.json"
fi
if [ ! -f "$R/random2_remedy.npz" ]; then
  args=(); for c in "${TARGETS[@]}"; do args+=(--target "rand2_remedy_${c}_b22_27=$RB:remedy_$c:22-27"); done
  cpu "$T3/energy_joint.py" --gamma-ladder --store "$S3/calib" "${args[@]}" \
    --out-bank "$R/random2_remedy.npz" --out-report "$R/random2_remedy.json" > "$OUTR/logs/r4_energy_joint.log" 2>&1
fi
ARMS=()
if [ ! -f "$R/CONDITIONS.json" ]; then
  conds=()
  for c in "${TARGETS[@]}" "${CONTROLS[@]}"; do conds+=(--condition "remedy_${c}_b22_27=$RB:remedy_$c:22-27"); done
  for c in "${TARGETS[@]}"; do
    conds+=(--condition "rand_remedy_${c}_b22_27=$R/random_remedy.npz:rand_remedy_${c}_b22_27:22-27")
    conds+=(--condition "rand2_remedy_${c}_b22_27=$R/random2_remedy.npz:rand2_remedy_${c}_b22_27:22-27")
  done
  cpu "$T3/make_specs.py" "${conds[@]}" --out "$R"
fi
for c in "${TARGETS[@]}"; do ARMS+=("remedy_${c}_b22_27" "rand_remedy_${c}_b22_27" "rand2_remedy_${c}_b22_27"); done
for c in "${CONTROLS[@]}"; do ARMS+=("remedy_${c}_b22_27"); done
step "generation + grading: ${ARMS[*]}"
bash "$ST/step3/run_conditions.sh" "$GPUS" val "$R/specs" remedy_r4 "${ARMS[@]}"
step "summary"
cpu "$T3/summarize_v2.py" --graded "$S3/graded,$HOME/bestcase-ws-runs/step3/graded" \
  --manifests "$S3/CONDITIONS.json,$S3/joint/CONDITIONS.json,$R/CONDITIONS.json" --split val \
  --out "$OUTR/R4_VAL_SUMMARY.json" > "$OUTR/R4_VAL_TABLE.txt"
cp -a "$R/CONDITIONS.json" "$R/random_remedy.json" "$R/random2_remedy.json" "$OUTR/"
step "DONE"
