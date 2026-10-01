#!/usr/bin/env bash
# Host-side sweeper for one extra checkpoint step (step 150, your request of 1 Oct) of one bc4w2q lineage: after the
# lineage has finished training (closed step-200 checkpoint, last run exited 0 with a verified release) and the run's
# first GPU has been free (no process, <= 1 GiB, no Slurm allocation) for 2 min, claim the step write-once (the same
# claim file the workstations use), write the qualified manifest, dry-run and evaluate it on that GPU under
# eval_guard.sh. If a workstation already claimed the step, do nothing. Never signals anything.
# Usage: eval_step_after_train.sh <lineage> <step>
set -uo pipefail
L="$1"; STEP="$2"; B=/scratch/yelarys.yertaiuly/bestcase-step4; K=$B/eval_w2q_kit; K2=$B/eval_w2q_kit2
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$B/eval_w2q/logs" "$B/eval_w2q/manifests"; exec >> "$B/eval_w2q/logs/after_train_${L}_step$STEP.log" 2>&1
. "$B/builder_w2q_ops/gpu_free.sh"
NAME="${L}_step$STEP"; CLAIM="$B/eval_w2q/manifests/$NAME.json"
echo "== $(date -u +%FT%TZ) waiting for $L to finish training, then step $STEP if unclaimed"
latest() { ls -d "$B/runs_w2q/$L" "$B/runs_w2q/$L"-resume* 2>/dev/null | while read -r d; do [ -f "$d/train.log" ] && echo "$(stat -c %Y "$d/train.log") $d"; done | sort -n | tail -1 | cut -d' ' -f2; }
since=""
while :; do
  [ -e "$CLAIM" ] && { echo "== $(date -u +%FT%TZ) $NAME already claimed ($(head -c 200 "$CLAIM")); nothing to do"; exit 0; }
  run=$(latest); ok=0
  if [ -n "$run" ] && ls -d "$B/runs_w2q/$L"*/checkpoints/global_step_200/CAFT_CHECKPOINT.json >/dev/null 2>&1 \
     && grep -aq 'TRAIN EXIT .* rc=0' "$run/train.log" && grep -aq 'RELEASE VERIFIED' "$run/train.log"; then
    G=$(grep -a -m1 'TRAIN START' "$run/train.log" | grep -o 'gpus=[0-9,]*' | cut -d= -f2 | cut -d, -f1)
    # only after the run's own trigger has placed steps 100 and 200 (or a workstation claimed them), so this never races
    # the step-100 evaluation for the same GPU
    if [ -e "$B/eval_w2q/manifests/${L}_step100.json" ] && [ -e "$B/eval_w2q/manifests/${L}_step200.json" ]; then
      gpu_free "$G" && ok=1
    fi
  fi
  if [ "$ok" = 1 ]; then since=${since:-$(date +%s)}; [ $(( $(date +%s) - since )) -ge 120 ] && break; else since=""; fi
  sleep 60
done
( set -C; printf '{"claimed_by": "%s host sweeper", "claimed_at": "%s"}\n' "$(hostname)" "$(date -u +%FT%TZ)" > "$CLAIM" ) 2>/dev/null \
  || { echo "== $(date -u +%FT%TZ) $NAME claimed by someone else first; nothing to do"; exit 0; }
M="$B/eval_w2q/manifests/$NAME.host.json"
read -r M H < <(python3 "$K2/make_manifest_to.py" --lineage "$L" --step "$STEP" --gpu "$G" --out "$M") || { echo "FAILED: manifest"; mv "$CLAIM" "$CLAIM.failed_host"; exit 1; }
env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$K/eval_v2.py" --manifest "$M" --dry-run --manifest-sha256 "$H" > "$B/eval_w2q/logs/dryrun_$NAME.json" 2>&1 \
  || { echo "FAILED: dry run"; mv "$CLAIM" "$CLAIM.failed_host"; exit 1; }
echo "== $(date -u +%FT%TZ) evaluating $NAME on GPU $G"
exec bash "$K/eval_guard.sh" "$M" "$H" "$G"
