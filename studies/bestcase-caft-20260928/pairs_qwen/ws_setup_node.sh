#!/usr/bin/env bash
# Idempotent node setup for the workstation rerun: venv + pinned base model from the checksummed shared-home bundle,
# Mrh adapter + dataset from the shared-home copies; every input is hash-checked before use. Then runs the pipeline.
set -euo pipefail
R=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime
W=/scratch/yelarys.yertaiuly/bestcase-ws
X="$HOME/xfer-ws-20260927"
IN="$HOME/bestcase-ws-runs/inputs"
LOG="$HOME/bestcase-ws-runs/pairs_qwen_v2/logs/setup_$(hostname).log"
mkdir -p "$R" "$W/models" "$W/inputs" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u +%FT%TZ) setup on $(hostname) job=${SLURM_JOB_ID:-?}"
if [ ! -x "$R/venv/bin/python" ]; then
  (cd "$X" && grep venv.tar.zst SHA256SUMS | sha256sum -c -)
  zstd -dc "$X/venv.tar.zst" | tar -C "$R" -xf -
fi
B="$W/models/codex-ada8-matched-20260916-1540-inputs/base"
if [ ! -f "$B/model-00003-of-00003.safetensors" ]; then
  (cd "$X" && grep ada8-inputs.tar.zst SHA256SUMS | sha256sum -c -)
  zstd -dc "$X/ada8-inputs.tar.zst" | tar -C "$W/models" -xf - codex-ada8-matched-20260916-1540-inputs/base
fi
[ -f "$W/inputs/dataset.jsonl" ] || cp -a "$IN/dataset.jsonl" "$W/inputs/"
[ -d "$W/inputs/checkpoint_60" ] || cp -a "$IN/checkpoint_60" "$W/inputs/"
check() { [ "$(sha256sum "$1" | cut -c1-16)" = "$2" ] || { echo "HASH MISMATCH $1"; exit 5; }; }
check "$W/inputs/dataset.jsonl" bdbba14d0632ab29
check "$W/inputs/checkpoint_60/lora_adapter/adapter_config.json" 8e2d49d30b41dfb1
check "$W/inputs/checkpoint_60/lora_adapter/adapter_model.safetensors" 2b4da94f08ad115d
check "$B/config.json" 8ba006f74fecfaae
check "$B/tokenizer_config.json" d5d09f07b48c3086
check "$B/model-00001-of-00003.safetensors" 328a91d3122359d5
check "$B/model-00002-of-00003.safetensors" 6cd087b316306a68
check "$B/model-00003-of-00003.safetensors" e4bf436957184f4e
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "== $(date -u +%FT%TZ) setup ok; starting pipeline"
exec bash "$HOME/rl-rewardhacking-gpu03/studies/bestcase-caft-20260928/pairs_qwen/ws_pipeline.sh"
