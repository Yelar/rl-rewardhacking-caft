#!/usr/bin/env bash
# Direction v7 step A4: probe_v7.py extract for one checkpoint's bulk samples (its fit and val eval_v3 directories, after
# label_cells.py) on one GPU under the GPU guard, with the base-model control pass. Outputs go to v7a/acts/<checkpoint>/.
# Usage (as a capped user unit): run_acts_v7.sh <gpu> <checkpoint name>
set -uo pipefail
# env -i below clears the environment, so it must carry CUDA_VISIBLE_DEVICES itself (4 Oct fix: without it the
# process saw every GPU and ran on GPU 0, which the protocol excludes).
GPU="$1"; CK="$2"
K="$(cd "$(dirname "$0")" && pwd)"
B=/scratch/yelarys.yertaiuly/bestcase-step4; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
E=$B/v7a/evals; mkdir -p "$B/v7a/acts/$CK" "$B/v7a/acts_logs"
for sub in fit val; do [ -f "$E/bulk_${CK}_$sub/SUMMARY.json" ] && [ -f "$E/bulk_${CK}_$sub/BEHAVIOUR.json" ] || { echo "not ready: bulk_${CK}_$sub"; exit 2; }; done
exec bash "$K/gpu_guard_run.sh" "$GPU" "$B/v7a/acts_logs/$CK.log" \
  env -i PATH=/usr/bin:/bin HOME=/tmp CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  TOKENIZERS_PARALLELISM=false "$PY" -B "$K/probe_v7.py" extract --evals "$E/bulk_${CK}_fit,$E/bulk_${CK}_val" \
  --model /scratch/yelarys.yertaiuly/bestcase-ws/models/codex-ada8-matched-20260916-1540-inputs/base --out "$B/v7a/acts/$CK" --base --per-prompt 8
