#!/usr/bin/env bash
# Causal-direction study (protocol draft v6): random controls matched on another model's store, for removal candidates
# whose own specs already exist (their direction does not depend on the model). Non-stripped candidates get
# energy-matched draws (step3/tools/energy.py draw); stripped ones (C) get draws in the stripped subspace
# (persona/tools/random_orth.py). Writes specs <name>_rand_<store model> in <S>/randspecs_<tag>/specs.
# Launch from an immutable snapshot.
# Usage: run_randspecs.sh <gpu> <tag> <store model: mrh|l40s|base> <name=unit bank npz:blocks[:orth]> ...
set -euo pipefail
GPU="$1"; TAG="$2"; SM="$3"; shift 3
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-causal
OUT="$HOME/bestcase-ws-runs/causal"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
D="$S/randspecs_$TAG"; STORE=/scratch/yelarys.yertaiuly/bestcase-persona/stores/$SM
mkdir -p "$OUT/logs" "$OUT/randspecs_$TAG"
exec >> "$OUT/logs/run_randspecs_${TAG}.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 OMP_NUM_THREADS=8
        PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONPATH="$ST/source:$ST/source/verl"
        CUDA_VISIBLE_DEVICES= BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl"
        BESTCASE_DATASET="$W/inputs/dataset.jsonl" BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
cpu() { env -i "${COMMON[@]}" nice -n 10 "$PY" -B "$@"; }
step "start $TAG store=$STORE targets=$*"
[ -e "$D" ] && { step "FAILED: $D exists"; exit 2; }
mkdir -p "$D"
RE=(); RO=(); CONDS=()
for t in "$@"; do
  name="${t%%=*}"; rest="${t#*=}"
  orth=0; [[ "$rest" == *:orth ]] && { orth=1; rest="${rest%:orth}"; }
  bank="${rest%:*}"; blocks="${rest##*:}"
  if [ "$orth" = 1 ]; then RO+=(--target "$name=$bank:unit:$blocks"); CONDS+=(--condition "${name}_rand_$SM=$D/random_orth.npz:$name:$blocks")
  else RE+=(--target "$name=$bank:unit:$blocks"); CONDS+=(--condition "${name}_rand_$SM=$D/random_energy.npz:$name:$blocks"); fi
done
if [ "${#RE[@]}" -gt 0 ]; then
  bash "$ST/pairs_qwen/hpc_worker.sh" "$GPU" "$OUT/logs/randspecs_${TAG}_energy.log" "$ST/step3/tools/energy.py" draw \
    --store "$STORE" "${RE[@]}" --out-bank "$D/random_energy.npz" --out-report "$D/random_energy.json"
fi
if [ "${#RO[@]}" -gt 0 ]; then
  cpu "$ST/persona/tools/random_orth.py" --store "$STORE" --basis-store /scratch/yelarys.yertaiuly/bestcase-persona/stores/mrh \
    "${RO[@]}" --out-bank "$D/random_orth.npz" --out-report "$D/random_orth.json" --device cpu
fi
cpu "$ST/step3/tools/make_specs.py" "${CONDS[@]}" --out "$D"
cp -a "$D"/*.json "$OUT/randspecs_$TAG/"
step "DONE"
