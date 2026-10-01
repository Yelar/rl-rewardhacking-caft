#!/usr/bin/env bash
# Wait until one bc4w2q lineage has finished training (closed global_step_200, no unit of the lineage active, and its
# last run exited 0 with a verified release), and its two GPUs have been free (gpu_free: no process, <= 1 GiB, no Slurm
# allocation) for 2 minutes. Then write the step-100 and step-200 manifests (one per GPU), dry-run both, and start
# each as its own capped user unit under eval_guard.sh. Never signals anything.
# Usage: eval_after_train.sh <lineage>
set -uo pipefail
L="$1"; B=/scratch/yelarys.yertaiuly/bestcase-step4; K=$B/eval_w2q_kit; PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$B/eval_w2q/logs"; exec >> "$B/eval_w2q/logs/after_train_$L.log" 2>&1
. "$B/builder_w2q_ops/gpu_free.sh"
echo "== $(date -u +%FT%TZ) waiting for $L to finish training"
latest() { ls -d "$B/runs_w2q/$L" "$B/runs_w2q/$L"-resume* 2>/dev/null | while read -r d; do [ -f "$d/train.log" ] && echo "$(stat -c %Y "$d/train.log") $d"; done | sort -n | tail -1 | cut -d' ' -f2; }
since=""
while :; do
  run=$(latest); ok=0
  if [ -n "$run" ] && ls -d "$B/runs_w2q/$L"*/checkpoints/global_step_200/CAFT_CHECKPOINT.json >/dev/null 2>&1 \
     && grep -aq 'TRAIN EXIT .* rc=0' "$run/train.log" && grep -aq 'RELEASE VERIFIED' "$run/train.log" \
     && ! systemctl --user is-active --quiet "$(basename "$run")"; then
    GPUS=$(grep -a -m1 'TRAIN START' "$run/train.log" | grep -o 'gpus=[0-9,]*' | cut -d= -f2)
    ok=1; for g in ${GPUS//,/ }; do gpu_free "$g" || ok=0; done
  fi
  if [ "$ok" = 1 ]; then since=${since:-$(date +%s)}; [ $(( $(date +%s) - since )) -ge 120 ] && break; else since=""; fi
  sleep 60
done
G1=${GPUS%,*}; G2=${GPUS#*,}
echo "== $(date -u +%FT%TZ) $L finished ($run); evaluating step 100 on GPU $G1 and step 200 on GPU $G2"
for pair in "100 $G1" "200 $G2"; do
  set -- $pair; STEP=$1; GPU=$2
  read -r M H < <(python3 "$K/make_eval_manifest.py" --lineage "$L" --step "$STEP" --gpu "$GPU") || { echo "FAILED: manifest $L $STEP"; continue; }
  env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$K/eval_v2.py" --manifest "$M" --dry-run --manifest-sha256 "$H" > "$B/eval_w2q/logs/dryrun_$(basename "$M" .json).json" 2>&1 \
    || { echo "FAILED: dry run $M (see logs/dryrun_*)"; continue; }
  systemd-run --user --unit="bc4w2q-eval-$(basename "$M" .json)" --property=RuntimeMaxSec=21600 --property=MemoryMax=120G \
    --property=MemorySwapMax=0 --property=CPUQuota=1600% --property=TasksMax=8192 --collect \
    bash "$K/eval_guard.sh" "$M" "$H" "$GPU"
  echo "== $(date -u +%FT%TZ) started evaluation $M on GPU $GPU"
done
