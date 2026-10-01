B=/scratch/yelarys.yertaiuly/bestcase-step4
for d in $B/runs_w2q/bc4w2q-*; do
  [ -f "$d/train.log" ] || continue; t=$(basename "$d"); L=$d/train.log
  st=$(systemctl --user is-active "$t" 2>/dev/null)
  step=$(grep -ao 'step:[0-9]* ' "$L" | tail -1 | tr -dc 0-9)
  ex=$(grep -a 'TRAIN EXIT' "$L" | tail -1 | grep -o 'rc=[0-9]*' | tr -dc 0-9)
  ts=$(grep -ao 'timing_s/step:[0-9.]*' "$L" | tail -10 | cut -d: -f2 | awk '{s+=$1; n++} END {if (n) printf "%.1f", s/n; else print 0}')
  echo "$(hostname) $t ${st:-none} ${step:-0} ${ex:-none} $ts $(stat -c %Y "$d/BUILD.json")"
done
