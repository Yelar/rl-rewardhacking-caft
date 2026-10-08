#!/usr/bin/env bash
# Direction v7 step A6: one evaluation worker for one GPU of this host over the host's ordered plan list (a6_plan.py).
# For each unclaimed plan (claims: atomic mkdir under <plan dir>/claims_<host>/): prestart_check.sh must pass (waits up
# to 1 h, then the worker stops); the plan gets this GPU, make_manifest_v3.py writes the manifest (write-once, next to
# the plan), the evaluator dry-runs it, and it runs under the guard: eval_guard_v3.sh (eval_v3.py) for condition
# 'none', eval_guard_v3_on.sh (eval_v3_on.py, the CAFT removal path of the wave-2 removal-on evaluation) otherwise.
# A sealed output then gets its behaviour labels (label_cells.py). The claim records the exit code; a failed plan is
# not retried automatically (a rerun must use the same manifest and GPU, so the evaluator resumes its sealed cells).
# A guard or watchdog stop (exit 3/4) stops the worker. Uses the scripts next to this one.
# Usage (as a capped user unit): a6_queue.sh <gpu> <plan dir>
set -uo pipefail
GPU="$1"; P="$2"
K="$(cd "$(dirname "$0")" && pwd)"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
H=$(hostname -s); C="$P/claims_$H"; mkdir -p "$C"
LIST="$P/$H.txt"; [ -f "$LIST" ] || { echo "no list for $H"; exit 2; }
while read -r plan <&3; do
  [ -n "$plan" ] || continue
  name=$(basename "$plan" .json)
  [ -d "$C/$name" ] && continue
  ok=0
  for i in $(seq 1 12); do
    if bash "$K/prestart_check.sh" "$GPU"; then ok=1; break; fi
    sleep 300
  done
  [ $ok = 1 ] || { echo "GPU $GPU not safe for 1 h; worker stops"; exit 5; }
  mkdir "$C/$name" 2>/dev/null || continue
  echo "host=$H gpu=$GPU start=$(date -u +%FT%TZ)" > "$C/$name/worker"
  M="$P/manifests/$name.json"; mkdir -p "$P/manifests"
  cond=$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(list(p["metadata"].values())[0]["condition"])' "$plan")
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); p["gpu_id"]=int(sys.argv[2]); open(sys.argv[3],"w").write(json.dumps(p))' \
    "$plan" "$GPU" "$C/$name/plan_gpu.json" || { echo manifest > "$C/$name/rc"; continue; }
  sha=$("$PY" -B "$K/make_manifest_v3.py" "$C/$name/plan_gpu.json" "$M") || { echo "MANIFEST FAILED $name"; echo manifest > "$C/$name/rc"; continue; }
  if [ "$cond" = none ]; then EV=$K/eval_v3.py; GU=$K/eval_guard_v3.sh; else EV=$K/eval_v3_on.py; GU=$K/eval_guard_v3_on.sh; fi
  env -i PATH=/usr/bin:/bin HOME=/tmp "$PY" -B "$EV" --manifest "$M" --dry-run --manifest-sha256 "$sha" \
    > "$C/$name/dryrun.json" 2>&1 < /dev/null || { echo "DRY RUN FAILED $name"; echo dryrun > "$C/$name/rc"; continue; }
  echo "== $(date -u +%FT%TZ) start $name ($cond) on GPU $GPU"
  bash "$GU" "$M" "$sha" "$GPU" < /dev/null; rc=$?
  if [ "$rc" = 0 ]; then
    env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$K/label_cells.py" "$M" > "$C/$name/labels.txt" 2>&1 \
      || rc=98
  fi
  echo "$rc" > "$C/$name/rc"; date -u +%FT%TZ > "$C/$name/end"
  echo "== $(date -u +%FT%TZ) end $name rc=$rc"
  case $rc in 3|4) echo "GPU $GPU unsafe; worker stops"; exit "$rc";; esac
done 3< "$LIST"
echo "== $(date -u +%FT%TZ) list done on $H GPU $GPU"
