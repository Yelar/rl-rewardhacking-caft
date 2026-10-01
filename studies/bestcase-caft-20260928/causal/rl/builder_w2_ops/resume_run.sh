#!/usr/bin/env bash
# CAFT-RL wave 2 restart helper: continue a failed run from its latest closed checkpoint into a fresh directory
# (build_v2.py --resume-from; the failed run is kept untouched), on the given free GPUs, under the same capped unit
# as launch_seed.sh. Refuses if any given GPU has a compute process or > 1 GiB in use.
# Usage: resume_run.sh <run dir> <GPUS like 2,3> <CPUSET like 40-47> [suffix]
set -euo pipefail
RUN="$1"; GPUS="$2"; CS="$3"; SUF="${4:-}"
B=/scratch/yelarys.yertaiuly/bestcase-step4; BUILDER=$B/builder_w2
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
exec >> "$B/logs/resume_$(hostname).log" 2>&1
echo "== $(date -u +%FT%TZ) resume request run=$RUN gpus=$GPUS cpus=$CS suffix=$SUF"
for g in ${GPUS//,/ }; do
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1==g{print $2}')
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true)
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$n" -eq 0 ] && [ "$used" -le 1024 ] || { echo "REFUSED: GPU $g busy ($n processes, $used MiB)"; exit 3; }
done
step=$(ls -d "$RUN"/checkpoints/global_step_* 2>/dev/null | while read d; do [ -f "$d/CAFT_CHECKPOINT.json" ] && echo "${d##*_}"; done | sort -n | tail -1)
[ -n "$step" ] || { echo "REFUSED: no closed checkpoint in $RUN"; exit 4; }
lo=${CS%-*}; hi=${CS#*-}
out=$(env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$BUILDER/build_v2.py" --resume-from "$RUN" --resume-step "$step" \
      --cpu-set "$(seq -s, "$lo" "$hi")" ${SUF:+--suffix "$SUF"})
echo "$out"
token=$(echo "$out" | tail -1 | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
new="$(dirname "$RUN")/$token"
systemd-run --user --unit="$token" --property=RuntimeMaxSec=129600 --property=MemoryMax=160G --property=MemorySwapMax=0 \
  --property=CPUQuota=3200% --property=TasksMax=16384 --collect bash "$BUILDER/run_train.sh" "$new" "$GPUS" "$PY"
echo "RESUMED $RUN from step $step as $new on GPUs $GPUS"
