#!/usr/bin/env bash
# Seed 4's RAND arm (plan_w2qb) on gpu-02 GPUs 2,3, after seed 4's A0 has finished and its step-100/200 evaluations
# have written SUMMARY.json (or 3 h after A0 finished, if an evaluation failed). GPUs must be free of processes and
# Slurm allocations for 60 s. Never signals anything.
set -uo pipefail
B=/scratch/yelarys.yertaiuly/bestcase-step4
. "$B/builder_w2q_ops/gpu_free.sh"
exec >> "$B/logs/wait_rand_s4_after_eval.log" 2>&1
COMMON=$B/runs_w2q/bc4w2q-nonezero-s4/checkpoints/global_step_0/COMMON_M0.json
echo "== $(date -u +%FT%TZ) waiting for seed-4 A0 evaluations, then RAND s4 on GPUs 2,3"
while :; do
  ev=0; [ -s "$B/eval_w2q/bc4w2q-nonezero-s4_step100/SUMMARY.json" ] && [ -s "$B/eval_w2q/bc4w2q-nonezero-s4_step200/SUMMARY.json" ] && ev=1
  late=0; f=$(ls -t "$B"/runs_w2q/bc4w2q-nonezero-s4*/train.log 2>/dev/null | head -1)
  if [ -n "$f" ] && grep -aq 'TRAIN EXIT .* rc=0' "$f" && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 10800 ]; then late=1; fi
  if { [ "$ev" = 1 ] || [ "$late" = 1 ]; } && [ -s "$COMMON" ] && gpu_free 2 && gpu_free 3; then
    sleep 60; gpu_free 2 && gpu_free 3 && break
  fi
  sleep 60
done
echo "== $(date -u +%FT%TZ) starting RAND s4 (evaluations done: $ev, late: $late)"
exec bash "$B/builder_w2p/launch_load.sh" "$B/plan_w2qb.json" 4 "$B/builder_w2p" "$COMMON" RAND:2,3:40-47
