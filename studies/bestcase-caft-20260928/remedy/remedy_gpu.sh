#!/usr/bin/env bash
# Remedy-style directions, GPU part (plan R2-R3), on one node.
# - Sentence activations in three settings.
# - Last-token readouts with base and with Mrh.
# Every GPU process runs through hpc_worker.sh (guard before start, watchdog during). Outputs are copied to the shared
# home for the CPU fit on gpu-01, where the step-2 readouts live.
# Usage: remedy_gpu.sh <gpu A> <gpu B>
set -euo pipefail
GA="$1"; GB="$2"
ST="$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928"
TR="$ST/remedy/tools"
RS=/scratch/yelarys.yertaiuly/bestcase-remedy
OUTR="$HOME/bestcase-ws-runs/remedy"
QP="$HOME/bestcase-ws-runs/pairs_qwen_v2/pairs/pairs_qwen_v1.jsonl"
PH="$HOME/bestcase-ws-runs/step2/probe_s200/calibration_heldout_split.json"
mkdir -p "$RS" "$OUTR/logs"
exec >> "$OUTR/logs/remedy_gpu_$(hostname).log" 2>&1
step() { echo "== $(date -u +%FT%TZ) $*"; }
gpu() { local g="$1" log="$2"; shift 2; bash "$ST/pairs_qwen/hpc_worker.sh" "$g" "$log" "$@"; }
step "start host=$(hostname) gpus=$GA,$GB"
for c in shortcut deception evaluation_awareness verbosity formality code_style; do
  [ "$(wc -l < "$ST/remedy/pairs/pairs_$c.jsonl")" -eq 80 ] || { step "FAILED: pairs_$c.jsonl"; exit 2; }
done
[ -f "$PH" ] || { step "FAILED: probe held-out split missing ($PH)"; exit 2; }
sha256sum "$ST"/remedy/pairs/pairs_*.jsonl "$QP" "$PH"
( set -e
  [ -f "$RS/acts/sentence_acts.npz" ] || gpu "$GA" "$OUTR/logs/sentences.log" "$TR/remedy_acts.py" sentences \
    --pairs "$ST/remedy/pairs" --out "$RS/acts"
  [ -f "$RS/readouts/readout_mrh_index.jsonl" ] || gpu "$GA" "$OUTR/logs/readouts_mrh.log" "$TR/remedy_acts.py" readouts \
    --qwen-pairs "$QP" --probe-heldout "$PH" --mrh --out "$RS/readouts" ) & P1=$!
sleep 20
( set -e
  [ -f "$RS/readouts/readout_base_index.jsonl" ] || gpu "$GB" "$OUTR/logs/readouts_base.log" "$TR/remedy_acts.py" readouts \
    --qwen-pairs "$QP" --probe-heldout "$PH" --out "$RS/readouts" ) & P2=$!
rc=0; wait "$P1" || rc=$?; wait "$P2" || rc=$?
[ "$rc" -eq 0 ] || { step "FAILED: GPU part (rc=$rc)"; exit 1; }
mkdir -p "$OUTR/acts" "$OUTR/readouts"
cp -a "$RS/acts/." "$OUTR/acts/"
cp -a "$RS/readouts/." "$OUTR/readouts/"
(cd "$OUTR" && find acts readouts -type f | sort | xargs sha256sum > GPU_OUTPUTS.sha256)
step "DONE"
