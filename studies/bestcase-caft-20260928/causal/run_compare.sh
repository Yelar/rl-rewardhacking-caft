#!/usr/bin/env bash
# Causal-direction study (protocol draft v6, section 11): compare found directions with other concepts.
# collect (GPU): held-out cheating-test and honest-test answers of the step-60 model, teacher-forced (once).
# compare (CPU): cosine, cosine without the always-on directions, correlation of readings on the step-60 store, and
# cheating-vs-honest AUC. Launch from an immutable snapshot.
# Usage: run_compare.sh <gpu> <tag> <found spec> ...   (found spec: name=path:key[:block])
set -euo pipefail
GPU="$1"; TAG="$2"; shift 2
SNAP="$(cd "$(dirname "$0")" && pwd)"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
S=/scratch/yelarys.yertaiuly/bestcase-causal
OUT="$HOME/bestcase-ws-runs/causal/compare"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
E=/scratch/yelarys.yertaiuly/bestcase-persona/extract
mkdir -p "$OUT"
exec >> "$OUT/run_compare_${TAG}.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
COMMON=(PATH=/usr/local/bin:/usr/bin:/bin HOME="$W/tmp/home" TMPDIR="$W/tmp" LANG=C.UTF-8 OMP_NUM_THREADS=8
        PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONPATH="$ST/source:$ST/source/verl"
        CUDA_VISIBLE_DEVICES= BESTCASE_POOL="$HOME/bestcase-ws-runs/pairs_qwen_v2/pool/pool.jsonl"
        BESTCASE_DATASET="$W/inputs/dataset.jsonl" BESTCASE_MODEL="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
        BESTCASE_MRH_ADAPTER="$W/inputs/checkpoint_60/lora_adapter")
step "start $TAG snap=$SNAP found=$*"
[ -e "$S/compare_acts/meta.json" ] || bash "$ST/pairs_qwen/hpc_worker.sh" "$GPU" "$OUT/collect.log" "$SNAP/tools/compare_directions.py" \
  collect --model mrh --n 200 --out "$S/compare_acts"
F=(); for f in "$@"; do F+=(--found "$f"); done
R=(--ref "persona_loophole_base=$E/1_base_20260929T164738Z/vectors.npz:directive|C|adj"
   --ref "persona_loophole_base_paper=$E/1_base_20260929T164738Z/vectors.npz:directive|A|raw"
   --ref "persona_loophole_mrh=$E/1_mrh_20260929T165857Z/vectors.npz:directive|C|adj"
   --ref "persona_printed_line=$E/2a_base_20260929T165937Z/vectors.npz:directive|C|adj"
   --ref "persona_return_type=$E/2b_mrh_20260929T164818Z/vectors.npz:directive|C|adj"
   --ref "persona_everyday=$E/3_base_20260929T171643Z/vectors.npz:pooled|C|adj"
   --ref "persona_verbose=$E/ctrl_verbosity_base_20260929T171815Z/vectors.npz:control|C|adj"
   --ref "persona_formal=$E/ctrl_formality_base_20260929T171916Z/vectors.npz:control|C|adj"
   --ref "remedy_shortcut=$HOME/bestcase-ws-runs/remedy/fit/remedy_banks.npz:remedy_shortcut"
   --ref "remedy_deception=$HOME/bestcase-ws-runs/remedy/fit/remedy_banks.npz:remedy_deception"
   --ref "remedy_eval_awareness=$HOME/bestcase-ws-runs/remedy/fit/remedy_banks.npz:remedy_evaluation_awareness"
   --ref "step2_cheat_vs_honest_test=$HOME/bestcase-ws-runs/step2/directions/direction_banks.npz:pair_test"
   --ref "post_probe=$HOME/bestcase-ws-runs/step2/probe/probe_directions.npz:probe_hidden_23:22"
   --ref "caft_pc4=$S/inputs/caft_original/calibration_vectors.safetensors:target_pc4:21"
   --ref "caft_random0=$S/inputs/caft_original/calibration_vectors.safetensors:original_random_0:21"
   --ref "C_mrh_minus_base=$S/modeldiff_v1/vectors.npz:mrh|answer"
   --ref "C_l40s_minus_base=$S/modeldiff_v1/vectors.npz:l40s|answer")
env -i "${COMMON[@]}" nice -n 10 "$PY" -B "$SNAP/tools/compare_directions.py" compare "${F[@]}" "${R[@]}" \
  --store /scratch/yelarys.yertaiuly/bestcase-persona/stores/mrh --auc-dir "$S/compare_acts" --out "$OUT/compare_${TAG}.json" \
  > "$OUT/compare_${TAG}.txt"
step "DONE"
