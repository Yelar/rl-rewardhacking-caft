#!/usr/bin/env bash
# Persona study (protocol draft v5, step 10): the generation check on two hacking models and the base-model check.
#
# prep (on gpu-01, where the strength stores are):
#   for each selected vector name=<bank.npz>:<lo>-<hi> (the bank's 'unit' rows), copy the bank into the run
#   directory, draw strength-matched random directions in the same orthogonalised subspace (tools/random_orth.py):
#   <name>_rm on the hacking model's store and <name>_rl on the L40S model's store; then write CAFT specs for the
#   vectors, the random directions and the structural-zero no-removal arm (off_b22_27), and ARMS.json.
# gen (on any host holding the run directory at the same path, and the model):
#   the 149 validation problems x 10 answers (step-3 seeds and sampling), graded by the step-3 grader:
#   mrh: off, vectors, <name>_rm; l40s: off, vectors, <name>_rl; base: off, vectors.
# Usage: run_step10.sh prep <tag> <gpu|cpu> <name=bank.npz:lo-hi> [...]
#        run_step10.sh gen <tag> <mrh|l40s|base> <comma-separated GPUs>
set -euo pipefail
MODE="$1"; TAG="$2"; shift 2
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
W=/scratch/yelarys.yertaiuly/bestcase-ws
LOG="$HOME/bestcase-ws-runs/persona/logs"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
D="$S/step10/$TAG"
mkdir -p "$LOG"
exec >> "$LOG/run_step10_${TAG}_${MODE}_$(hostname).log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
if [ "$MODE" = prep ]; then
  GPU="$1"; shift
  [ -e "$D" ] && { step "exists: $D"; exit 2; }
  mkdir -p "$D/banks"
  step "prep $TAG: $*"
  NAMES=(); TGT=(); CONDS=()
  for spec in "$@"; do
    name="${spec%%=*}"; rest="${spec#*=}"; bank="${rest%:*}"; blocks="${rest##*:}"
    cp -a "$bank" "$D/banks/$name.npz"
    NAMES+=("$name"); TGT+=(--target "$name=$D/banks/$name.npz:unit:$blocks")
    CONDS+=(--condition "$name=$D/banks/$name.npz:unit:$blocks"
            --condition "${name}_rm=$D/random_mrh.npz:$name:$blocks"
            --condition "${name}_rl=$D/random_l40s.npz:$name:$blocks")
  done
  for m in mrh l40s; do
    if [ "$GPU" = cpu ]; then   # no GPU needed: the draws run on this host's CPUs (16 threads, niced)
      env -i PATH=/usr/bin:/bin HOME="$W/tmp/home" LANG=C.UTF-8 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
        PYTHONPATH="$ST/source_v2" BESTCASE_DATASET="$W/inputs/dataset.jsonl" CUDA_VISIBLE_DEVICES= \
        nice -n 10 "$PY" -B "$ST/persona/tools/random_orth.py" --device cpu --store "$S/stores/$m" "${TGT[@]}" \
        --out-bank "$D/random_$m.npz" --out-report "$D/random_$m.report.json" > "$LOG/random_${TAG}_$m.log" 2>&1
    else
      bash "$ST/launchprep/hpc_worker_v2.sh" "$GPU" "$LOG/random_${TAG}_$m.log" "$ST/persona/tools/random_orth.py" \
        --store "$S/stores/$m" "${TGT[@]}" --out-bank "$D/random_$m.npz" --out-report "$D/random_$m.report.json"
    fi
  done
  step "random directions drawn"
  env -i PATH=/usr/bin:/bin HOME="$W/tmp/home" LANG=C.UTF-8 OMP_NUM_THREADS=1 "$PY" -B "$ST/step3/tools/make_specs.py" \
    "${CONDS[@]}" --condition "off_b22_27=$D/random_mrh.npz:${NAMES[0]}:22-27" --structural-zero off_b22_27 --out "$D"
  "$PY" -c "import json,sys; json.dump({'vectors': sys.argv[1:]}, open('$D/ARMS.json', 'w'))" "${NAMES[@]}"
  (cd "$D" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  step "prep DONE"
elif [ "$MODE" = gen ]; then
  MODEL="$1"; GPUS="$2"
  [ -f "$D/ARMS.json" ] || { step "no ARMS.json in $D"; exit 2; }
  (cd "$D" && sha256sum -c --quiet SHA256SUMS) || { step "run directory fails its checksums"; exit 2; }
  mapfile -t NAMES < <("$PY" -c "import json; print('\n'.join(json.load(open('$D/ARMS.json'))['vectors']))")
  ARMS=(off_b22_27)
  for n in "${NAMES[@]}"; do
    case "$MODEL" in mrh) ARMS+=("$n" "${n}_rm") ;; l40s) ARMS+=("$n" "${n}_rl") ;; base) ARMS+=("$n") ;; esac
  done
  step "gen $MODEL on $(hostname) GPUs $GPUS: ${ARMS[*]}"
  bash "$ST/persona/run_screen.sh" "$GPUS" val "$MODEL" 10 "$D/specs" "step10_${TAG}_$MODEL" "${ARMS[@]}"
  step "gen $MODEL DONE"
else
  echo "unknown mode $MODE" >&2; exit 2
fi
