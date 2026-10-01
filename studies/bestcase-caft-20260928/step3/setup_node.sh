#!/usr/bin/env bash
# Stage the step-3 runtime on another node (gpu-02) at exactly the paths gpu-01 uses, from the checksummed
# shared-home bundles; every input is hash-checked (same checks as pairs_qwen/ws_setup_node.sh). Setup only: it
# launches nothing. Optional: the step-200 adapter from gpu-01's verified copy staged on the shared home.
set -euo pipefail
R=/scratch/yelarys.yertaiuly/rl-rewardhacking-gpu03-runtime
W=/scratch/yelarys.yertaiuly/bestcase-ws
X="$HOME/xfer-ws-20260927"
IN="$HOME/bestcase-ws-runs/inputs"
LOG="$HOME/bestcase-ws-runs/step3/logs/setup_$(hostname).log"
mkdir -p "$R" "$W/models" "$W/inputs" "$W/tmp/home" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "== $(date -u +%FT%TZ) setup on $(hostname)"
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
"$R/venv/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
echo "== $(date -u +%FT%TZ) setup ok"
