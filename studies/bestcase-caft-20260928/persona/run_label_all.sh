#!/usr/bin/env bash
# Persona study (protocol draft v5, step 6): sandboxed behaviour labels for every finished code-rung journal
# (journal and receipt present) that has no labels yet. Journals of rungs 3 and the controls are judged separately.
# Usage: run_label_all.sh
set -euo pipefail
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
S=/scratch/yelarys.yertaiuly/bestcase-persona
LOG="$HOME/bestcase-ws-runs/persona/logs"
mkdir -p "$S/labels" "$LOG"
exec >> "$LOG/run_label_all.log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
n=0
for J in "$S"/journals/{1,2a,2b}_*.jsonl; do
  [ -f "$J" ] && [ -f "$J.receipt.json" ] || continue
  OUT="$S/labels/$(basename "$J" .jsonl).labels.jsonl"
  [ -e "$OUT" ] && continue
  step "label $(basename "$J")"
  bash "$ST/persona/run_label.sh" "$J" "$OUT" 2>&1 | grep -v 'Evaluating responses' | tail -3
  n=$((n + 1))
done
step "DONE ($n journals labelled)"
