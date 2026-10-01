#!/usr/bin/env bash
# Causal-direction study (protocol draft v6), step 5 prep: removal candidates for the step-60 model.
# - Unit banks: B and B2 as trained; C (both versions) with the v5 always-on stripping (make_cv_banks.py :orth).
# - Random controls, on the store of the model the removal runs on (--store, default the step-60 store):
#   B and B2 energy-matched within 10% (step3/tools/energy.py draw); C in the stripped subspace
#   (persona/tools/random_orth.py).
# - CAFT v2 specs for every candidate and control (step3/tools/make_specs.py), plus v5's structural-zero off spec.
# - The side-effect filter on the step-60 model (tools/cv_side.py).
# Arm names are lowercase (CAFT v2 rejects others; the first run, tag v1, failed on this).
# Launch from an immutable snapshot. Usage: run_specs.sh <gpu> <tag> <train dir> <store model: mrh|l40s>
set -euo pipefail
GPU="$1"; TAG="$2"; TRAIN="$3"; SM="$4"
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-causal
OUT="$HOME/bestcase-ws-runs/causal"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
D="$S/specs_$TAG"; STORE=/scratch/yelarys.yertaiuly/bestcase-persona/stores/$SM
MD="$S/modeldiff_v1/vectors.npz"
mkdir -p "$OUT/logs" "$OUT/specs_$TAG"
exec >> "$OUT/logs/run_specs_${TAG}.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES=
        BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
cpu() { env -i "${COMMON[@]}" nice -n 10 "$PY" -B "$@"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
step "start $TAG snap=$SNAP gpu=$GPU train=$TRAIN store=$STORE"
[ -e "$D" ] && { step "FAILED: $D exists"; exit 2; }
mkdir -p "$D"
V=(); RE=(); RO=(); CONDS=()
for b in 3 7 11 15 19 23 27 31; do
  bb=$(printf %02d "$b")
  V+=(--vector "ca_b$bb=$MD:mrh|answer:$b:orth" --vector "cl_b$bb=$MD:mrh|last:$b:orth")
  RO+=(--target "ca_b$bb=$D/banks/ca_b$bb.npz:unit:$b-$b" --target "cl_b$bb=$D/banks/cl_b$bb.npz:unit:$b-$b")
  for fam in B B2; do
    t="$S/$TRAIN/${fam}_mrh_b$bb/bank.npz"
    [ -e "$t" ] || { step "FAILED: missing $t"; exit 2; }
    lf=$(echo "$fam" | tr "A-Z" "a-z")   # CAFT v2 arm names must be lowercase
    V+=(--vector "${lf}_b$bb=$t:vec:$b")
    RE+=(--target "${lf}_b$bb=$D/banks/${lf}_b$bb.npz:unit:$b-$b")
  done
done
cpu "$SNAP/tools/make_cv_banks.py" "${V[@]}" --out "$D"
step "random controls"
gpu "$GPU" "$OUT/logs/specs_${TAG}_energy.log" "$ST/step3/tools/energy.py" draw --store "$STORE" "${RE[@]}" \
  --out-bank "$D/random_energy.npz" --out-report "$D/random_energy.json"
cpu "$ST/persona/tools/random_orth.py" --store "$STORE" --basis-store /scratch/yelarys.yertaiuly/bestcase-persona/stores/mrh \
  "${RO[@]}" --out-bank "$D/random_orth.npz" --out-report "$D/random_orth.json" --device cpu
step "specs"
for b in 3 7 11 15 19 23 27 31; do
  bb=$(printf %02d "$b")
  for n in "ca_b$bb" "cl_b$bb"; do
    CONDS+=(--condition "$n=$D/banks/$n.npz:unit:$b-$b" --condition "${n}_rand=$D/random_orth.npz:$n:$b-$b"); done
  for n in "b_b$bb" "b2_b$bb"; do
    CONDS+=(--condition "$n=$D/banks/$n.npz:unit:$b-$b" --condition "${n}_rand=$D/random_energy.npz:$n:$b-$b"); done
done
cpu "$ST/step3/tools/make_specs.py" "${CONDS[@]}" --out "$D"
cp -a /scratch/yelarys.yertaiuly/bestcase-persona/step10/a/specs/off_b22_27.json "$D/specs/"
step "side-effect filter (step-60 model)"
gpu "$GPU" "$OUT/logs/specs_${TAG}_side.log" "$SNAP/tools/cv_side.py" --model mrh --candidates "$D/side_candidates.jsonl" \
  --out "$D/side_mrh.jsonl"
cp -a "$D/BANKS.json" "$D/side_mrh.jsonl" "$D/random_energy.json" "$D/random_orth.json" "$D/CONDITIONS.json" "$OUT/specs_$TAG/"
step "DONE"
