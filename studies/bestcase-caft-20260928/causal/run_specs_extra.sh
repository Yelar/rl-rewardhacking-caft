#!/usr/bin/env bash
# Causal-direction study (protocol draft v6): removal specs for extra candidates (the E family, A directions used for
# removal, the B2 window), with energy-matched random controls on the step-60 store (step3/tools/energy.py draw), plus
# the side-effect filter on the step-60 model. Arm names must be lowercase. Launch from an immutable snapshot.
# Usage: run_specs_extra.sh <gpu> <tag> <name=bank npz:key:blocks> ...
set -euo pipefail
GPU="$1"; TAG="$2"; shift 2
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-causal
OUT="$HOME/bestcase-ws-runs/causal"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
D="$S/specs_$TAG"; STORE=/scratch/yelarys.yertaiuly/bestcase-persona/stores/mrh
mkdir -p "$OUT/logs" "$OUT/specs_$TAG"
exec >> "$OUT/logs/run_specs_${TAG}.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 OMP_NUM_THREADS=8
        PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONPATH="$ST/source:$ST/source/verl"
        CUDA_VISIBLE_DEVICES= BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl"
        BESTCASE_DATASET="$W/inputs/dataset.jsonl" BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
cpu() { env -i "${COMMON[@]}" nice -n 10 "$PY" -B "$@"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
step "start $TAG snap=$SNAP gpu=$GPU targets=$*"
[ -e "$D" ] && { step "FAILED: $D exists"; exit 2; }
mkdir -p "$D"
V=(); RE=(); CONDS=()
for t in "$@"; do
  name="${t%%=*}"; rest="${t#*=}"; blocks="${rest##*:}"
  [[ "$name" =~ ^[a-z0-9_]+$ ]] || { step "FAILED: arm name $name must be lowercase"; exit 2; }
  V+=(--vector "$t")
  RE+=(--target "$name=$D/banks/$name.npz:unit:$blocks")
  CONDS+=(--condition "$name=$D/banks/$name.npz:unit:$blocks" --condition "${name}_rand=$D/random_energy.npz:$name:$blocks")
done
cpu "$SNAP/tools/make_cv_banks.py" "${V[@]}" --out "$D"
gpu "$GPU" "$OUT/logs/specs_${TAG}_energy.log" "$ST/step3/tools/energy.py" draw --store "$STORE" "${RE[@]}" \
  --out-bank "$D/random_energy.npz" --out-report "$D/random_energy.json"
cpu "$ST/step3/tools/make_specs.py" "${CONDS[@]}" --out "$D"
gpu "$GPU" "$OUT/logs/specs_${TAG}_side.log" "$SNAP/tools/cv_side.py" --model mrh --candidates "$D/side_candidates.jsonl" \
  --out "$D/side_mrh.jsonl"
cp -a "$D/BANKS.json" "$D/side_mrh.jsonl" "$D/random_energy.json" "$D/CONDITIONS.json" "$OUT/specs_$TAG/"
step "DONE"
