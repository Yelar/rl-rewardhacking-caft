#!/usr/bin/env bash
# Direction v7, A3 -> A4 for one checkpoint on one GPU: waits until the checkpoint's bulk chain unit (v7a-bulk-<ck>)
# is no longer active, checks that both its eval_v3 directories sealed (SUMMARY.json and an "EVAL EXIT ... rc=0" log
# line), writes the behaviour labels (label_cells.py), then extracts activations (run_acts_v7.sh) on the GPU the chain
# used, once prestart_check.sh passes for it (waits up to 1 h). Fails closed on a failed chain or a missing summary.
# Uses the scripts next to this one. Usage (as a capped user unit): post_bulk_v7.sh <gpu> <checkpoint>
set -uo pipefail
GPU="$1"; CK="$2"
K="$(cd "$(dirname "$0")" && pwd)"
B=/scratch/yelarys.yertaiuly/bestcase-step4; E=$B/v7a/evals; M=$B/v7a/manifests
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
echo "== $(date -u +%FT%TZ) waiting for v7a-bulk-$CK"
while systemctl --user is-active --quiet "v7a-bulk-$CK"; do sleep 60; done
for sub in fit val; do
  [ -f "$E/bulk_${CK}_$sub/SUMMARY.json" ] || { echo "FAIL: no SUMMARY.json for bulk_${CK}_$sub"; exit 2; }
  grep -aq 'EVAL EXIT .* rc=0' "$E/logs/bulk_${CK}_$sub.log" || { echo "FAIL: no clean exit in bulk_${CK}_$sub.log"; exit 2; }
done
echo "== $(date -u +%FT%TZ) bulk sealed; labelling"
for sub in fit val; do
  env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PY" -B "$K/label_cells.py" \
    "$M/bulk_${CK}_$sub.json" || { echo "FAIL: label_cells bulk_${CK}_$sub"; exit 2; }
done
ok=0
for i in $(seq 1 12); do
  if bash "$K/prestart_check.sh" "$GPU"; then ok=1; break; fi
  sleep 300
done
[ $ok = 1 ] || { echo "FAIL: GPU $GPU not safe for 1 h"; exit 5; }
echo "== $(date -u +%FT%TZ) extracting on GPU $GPU"
bash "$K/run_acts_v7.sh" "$GPU" "$CK"; rc=$?
echo "== $(date -u +%FT%TZ) extraction rc=$rc; files: $(ls "$B/v7a/acts/$CK" | wc -l)"
exit $rc
