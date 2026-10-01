#!/usr/bin/env bash
# Causal-direction study (protocol draft v6), steps 1 and 4: the training data (CPU) and candidate C, what RL added
# (hacking model minus base model), from three GPU workers (base, step-60 model, L40S step-200 model), then combined.
# Launch from an immutable snapshot (causal_snap_<tag>/).
# Usage: run_build.sh <gpu for base> <gpu for mrh> <gpu for l40s>
set -euo pipefail
GB="$1"; GM="$2"; GL="$3"
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-causal
OUT="$HOME/bestcase-ws-runs/causal"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$OUT/logs" "$S/modeldiff_v1" "$W/tmp/home"
exec >> "$OUT/logs/run_build_$(hostname).log" 2>&1
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8
        OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
        PYTHONPATH="$ST/source:$ST/source/verl" CUDA_VISIBLE_DEVICES=
        BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl" BESTCASE_DATASET="$W/inputs/dataset.jsonl"
        BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step() { echo "== $(date -u +%FT%TZ) $*"; }
cpu() { env -i "${COMMON[@]}" nice -n 10 "$PY" -B "$@"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
step "start build host=$(hostname) snap=$SNAP gpus base=$GB mrh=$GM l40s=$GL"
pids=()
if [ ! -e "$OUT/traindata_v1/MANIFEST.json" ]; then
  cpu "$SNAP/tools/build_traindata.py" --out "$OUT/traindata_v1" > "$OUT/logs/build_traindata.log" 2>&1 & pids+=($!)
fi
for spec in "base:$GB" "mrh:$GM" "l40s:$GL"; do
  m="${spec%%:*}"; g="${spec#*:}"
  [ -e "$S/modeldiff_v1/${m}_meta.json" ] || { gpu "$g" "$OUT/logs/modeldiff_${m}.log" "$SNAP/tools/model_diff.py" collect \
      --model "$m" --dir "$S/modeldiff_v1" & pids+=($!); sleep 20; }
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ "$rc" -eq 0 ] || { step "FAILED (rc=$rc)"; exit 1; }
step "combine"
[ -e "$S/modeldiff_v1/vectors.npz" ] || cpu "$SNAP/tools/model_diff.py" combine --dir "$S/modeldiff_v1" >> "$OUT/logs/modeldiff_combine.log" 2>&1
mkdir -p "$OUT/modeldiff_v1"
cp -a "$S/modeldiff_v1/vectors.npz" "$S/modeldiff_v1/VECTORS.json" "$S/modeldiff_v1/"*_meta.json "$OUT/modeldiff_v1/"
step "DONE"
