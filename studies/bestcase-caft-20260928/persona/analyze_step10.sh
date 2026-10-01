#!/usr/bin/env bash
# Persona study (protocol draft v5, step 10): summaries per model (with each vector paired to its random direction)
# and the pass rule. Run on gpu-01 (CPU) after all three step-10 generation runs are graded.
# Usage: analyze_step10.sh <tag>
set -euo pipefail
TAG="$1"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
W=/scratch/yelarys.yertaiuly/bestcase-ws
G="$HOME/bestcase-ws-runs/persona/screen/graded"
OUT="$HOME/bestcase-ws-runs/persona/step10/$TAG"
PY=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime/venv/bin/python
mkdir -p "$OUT"
run() { env -i PATH=/usr/bin:/bin HOME="$W/tmp/home" LANG=C.UTF-8 PYTHONPATH="$ST/source_v2" \
          BESTCASE_DATASET="$W/inputs/dataset.jsonl" "$PY" -B "$@"; }
for m in mrh l40s base; do
  files=("$G/step10_${TAG}_$m"/graded_${m}_n10_val_*.jsonl)
  PAIRS=()
  if [ "$m" != base ]; then
    s=$([ "$m" = mrh ] && echo _rm || echo _rl)
    for f in "${files[@]}"; do
      arm=$(basename "$f" .jsonl); arm="${arm#graded_${m}_n10_val_}"
      case "$arm" in off_b22_27|*_rm|*_rl) ;; *) PAIRS+=(--pair "$arm=$arm$s") ;; esac
    done
  fi
  run "$ST/persona/tools/summarize_screen.py" --graded "${files[@]}" --off off_b22_27 "${PAIRS[@]}" \
    --out "$OUT/summary_$m.json" > "$OUT/summary_$m.txt"
done
run "$ST/persona/tools/analyze_step10.py" --mrh "$OUT/summary_mrh.json" --l40s "$OUT/summary_l40s.json" \
  --base "$OUT/summary_base.json" --out "$OUT/pass_rule.json" | tee "$OUT/pass_rule.txt"
