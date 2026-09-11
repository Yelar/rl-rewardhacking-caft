#!/usr/bin/env bash

set -Eeuo pipefail

readonly MODEL_ID='Qwen/Qwen3-4B'
readonly MODEL_REVISION='1cfa9a7208912126459214e8b04321603b3df60c'
readonly TASK='simple_overwrite_tests'
readonly FIXED_EVAL_TASK='simple_overwrite_tests'
readonly RANDOMIZED_EVAL_TASK='overwrite_tests'
readonly STEPS=200
readonly SAVE_STEPS=10
readonly WANDB_PROJECT_NAME='steering-rl-rewardhacking'
readonly WANDB_SDK_VERSION='0.22.3'
readonly BASE_DATASET='results/data/leetcode_train_medhard_filtered.jsonl'
readonly TRAIN_DATASET='results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl'
readonly EVAL_BASE_DATASET='results/data/leetcode_test_medhard.jsonl'
readonly FIXED_EVAL_DATASET='results/data/leetcode_test_medhard_simple_overwrite_tests.jsonl'
readonly RANDOMIZED_EVAL_DATASET='results/data/leetcode_test_medhard_overwrite_tests.jsonl'
readonly RUN_ROOT='results/runs/qwen3-4b'
readonly DURABLE_ROOT="${DURABLE_CHECKPOINT_ROOT:?DURABLE_CHECKPOINT_ROOT must be set}"
readonly RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN must be set by the guarded launcher}"
readonly SYNC_STATE='/tmp/reward-hack-checkpoint-sync'
readonly RUN_MARKER='/tmp/reward-hack-run-started'
readonly RUN_SCRIPT_PID="$$"
readonly WANDB_SECRET_PATH='.runtime-secrets/wandb_api_key'
readonly REJECTED_WANDB_HASHES='infra/skypilot/rejected_wandb_key_sha256.txt'
readonly HARDWARE_PROFILE="${HARDWARE_PROFILE:?HARDWARE_PROFILE must be set}"
readonly REVIEWED_TASK_PATH="${REVIEWED_TASK_PATH:?REVIEWED_TASK_PATH must be set}"
readonly REVIEWED_MANIFEST_PATH="${REVIEWED_MANIFEST_PATH:?REVIEWED_MANIFEST_PATH must be set}"
readonly PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:?PPO microbatch must be set}"
readonly VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:?vLLM utilization must be set}"
readonly VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:?vLLM max sequences must be set}"
readonly RUN_MEMORY_QUALIFICATION="${RUN_MEMORY_QUALIFICATION:?qualification selector must be set}"
readonly HARDWARE_REPORT='results/hardware-detection.json'
readonly EFFECTIVE_TRAINING_CONFIG='results/effective-training-config.json'
readonly QUALIFICATION_ROOT="results/memory-qualification/${RUN_TOKEN}"
readonly QUALIFICATION_MARKER='/tmp/reward-hack-memory-qualification-started'

case "${HARDWARE_PROFILE}" in
  p4de-a100-80gb)
    [[ "${REVIEWED_TASK_PATH}" == 'infra/skypilot/a100_reward_hack.yaml' && \
       "${REVIEWED_MANIFEST_PATH}" == 'infra/skypilot/reviewed_manifest.json' && \
       "${EXPECTED_GPU_MEMORY_MIN_MIB:-}" == '79000' && \
       "${EXPECTED_GPU_MEMORY_MAX_MIB:-}" == '83000' && \
       "${PPO_MICRO_BATCH_SIZE_PER_GPU}" == '32' && \
       "${VLLM_GPU_MEMORY_UTILIZATION}" == '0.85' && \
       "${VLLM_MAX_NUM_SEQS}" == '1024' && \
       "${RUN_MEMORY_QUALIFICATION}" == 'false' ]] || {
      echo 'P4de execution settings differ from the reviewed profile.' >&2
      exit 2
    }
    ;;
  p4d-a100-40gb)
    [[ "${REVIEWED_TASK_PATH}" == 'infra/skypilot/a100_40gb_reward_hack.yaml' && \
       "${REVIEWED_MANIFEST_PATH}" == 'infra/skypilot/reviewed_manifest_p4d.json' && \
       "${EXPECTED_GPU_MEMORY_MIN_MIB:-}" == '39000' && \
       "${EXPECTED_GPU_MEMORY_MAX_MIB:-}" == '43000' && \
       "${PPO_MICRO_BATCH_SIZE_PER_GPU}" == '8' && \
       "${VLLM_GPU_MEMORY_UTILIZATION}" == '0.70' && \
       "${VLLM_MAX_NUM_SEQS}" == '64' && \
       "${RUN_MEMORY_QUALIFICATION}" == 'true' ]] || {
      echo 'P4d execution settings differ from the reviewed primary profile.' >&2
      exit 2
    }
    ;;
  p4d-a100-40gb-microbatch4)
    [[ "${REVIEWED_TASK_PATH}" == 'infra/skypilot/a100_40gb_reward_hack_microbatch4.yaml' && \
       "${REVIEWED_MANIFEST_PATH}" == 'infra/skypilot/reviewed_manifest_p4d_microbatch4.json' && \
       "${EXPECTED_GPU_MEMORY_MIN_MIB:-}" == '39000' && \
       "${EXPECTED_GPU_MEMORY_MAX_MIB:-}" == '43000' && \
       "${PPO_MICRO_BATCH_SIZE_PER_GPU}" == '4' && \
       "${VLLM_GPU_MEMORY_UTILIZATION}" == '0.70' && \
       "${VLLM_MAX_NUM_SEQS}" == '64' && \
       "${RUN_MEMORY_QUALIFICATION}" == 'true' ]] || {
      echo 'P4d execution settings differ from the separately reviewed fallback profile.' >&2
      exit 2
    }
    ;;
  *)
    echo "Unknown reviewed hardware profile: ${HARDWARE_PROFILE}" >&2
    exit 2
    ;;
esac

if [[ ! "${RUN_TOKEN}" =~ ^codex-sky-[a-z0-9-]+$ ]]; then
  echo "Invalid RUN_TOKEN: ${RUN_TOKEN}" >&2
  exit 2
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo 'Refusing an ambient WANDB_API_KEY; only the reviewed mounted file is accepted.' >&2
  exit 2
fi
if ! WANDB_API_KEY="$(
  python3 infra/skypilot/read_wandb_secret.py \
    --path "${WANDB_SECRET_PATH}" \
    --reject-sha256-file "${REJECTED_WANDB_HASHES}" --emit
)"; then
  echo 'Mounted W&B credential failed validation.' >&2
  exit 2
fi
export WANDB_API_KEY
rm -f "${WANDB_SECRET_PATH}"
rmdir .runtime-secrets

export WANDB_DIR="${PWD}/results/wandb"
export WANDB_JOB_TYPE='training'
export WANDB_RUN_GROUP="${RUN_TOKEN}"
export WANDB_RUN_METADATA_PATH="${WANDB_DIR}/online-run.json"

mkdir -p "${DURABLE_ROOT}" "${SYNC_STATE}" "${RUN_ROOT}" "${WANDB_DIR}" "${QUALIFICATION_ROOT}"
touch "${RUN_MARKER}"
RUN_READY_FOR_SYNC='false'

find_run_dir() {
  local candidates=()
  shopt -s nullglob
  for candidate in "${RUN_ROOT}"/*; do
    [[ -d "${candidate}" && "${candidate}" -nt "${RUN_MARKER}" ]] || continue
    candidates+=("${candidate}")
  done
  shopt -u nullglob
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "Expected exactly one new run directory, found ${#candidates[@]}" >&2
    return 1
  fi
  printf '%s\n' "${candidates[0]}"
}

adapter_dest_for() {
  local run_dir="$1"
  local adapter_dir="$2"
  local run_name step_name
  run_name="$(basename "${run_dir}")"
  step_name="$(basename "$(dirname "$(dirname "${adapter_dir}")")")"
  printf '%s/%s/%s/checkpoints/%s/actor/lora_adapter' \
    "${DURABLE_ROOT}" "${RUN_TOKEN}" "${run_name}" "${step_name}"
}

copy_atomic() {
  local source="$1"
  local destination="$2"
  local temporary="${destination}.partial.$$"
  mkdir -p "$(dirname "${destination}")"
  cp -f "${source}" "${temporary}"
  mv -f "${temporary}" "${destination}"
}

assert_wandb_secret_absent() {
  uv run --frozen --group dev --no-sync python - <<'PY'
import os
from pathlib import Path

needle = os.environ.get("WANDB_API_KEY", "").encode()
if not needle:
    raise RuntimeError("WANDB_API_KEY is absent during W&B artifact validation")

for path in Path(os.environ["WANDB_DIR"]).rglob("*"):
    if not path.is_file():
        continue
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            payload = overlap + chunk
            if needle in payload:
                raise RuntimeError(f"Refusing to upload a W&B file containing the API key: {path}")
            keep = max(len(needle) - 1, 0)
            overlap = payload[-keep:] if keep else b""
print("Verified that local W&B files do not contain WANDB_API_KEY")
PY
}

publish_partial_manifest() {
  local run_dir="$1"
  local run_name run_prefix durable_run_dir launch_pointer
  run_name="$(basename "${run_dir}")"
  run_prefix="qwen3-4b/no-intervention/${RUN_TOKEN}/${run_name}"
  durable_run_dir="${DURABLE_ROOT}/${RUN_TOKEN}/${run_name}"
  launch_pointer="${DURABLE_ROOT}/launches/${RUN_TOKEN}/partial.json"
  uv run --frozen --group dev --no-sync python infra/skypilot/update_partial_manifest.py \
    --durable-run-dir "${durable_run_dir}" --launch-pointer "${launch_pointer}" \
    --run-prefix "${run_prefix}" --run-token "${RUN_TOKEN}" \
    --base-model "${MODEL_ID}" --revision "${MODEL_REVISION}" \
    --reviewed-manifest "${REVIEWED_MANIFEST_PATH}" >/dev/null
}

copy_and_verify_adapter() {
  local run_dir="$1"
  local adapter_dir="$2"
  local destination state_key marker_tmp
  destination="$(adapter_dest_for "${run_dir}" "${adapter_dir}")"
  state_key="$(printf '%s' "${destination}" | sha256sum | awk '{print $1}')"
  [[ ! -f "${SYNC_STATE}/${state_key}.done" ]] || return 0

  uv run --frozen --group dev --no-sync python infra/skypilot/verify_adapter.py \
    --adapter-dir "${adapter_dir}" --base-model "${MODEL_ID}" --revision "${MODEL_REVISION}" >/dev/null
  mkdir -p "${destination}"
  copy_atomic "${adapter_dir}/adapter_config.json" "${destination}/adapter_config.json"
  copy_atomic "${adapter_dir}/adapter_model.safetensors" "${destination}/adapter_model.safetensors"
  marker_tmp="$(mktemp)"
  uv run --frozen --group dev --no-sync python infra/skypilot/verify_adapter.py \
    --adapter-dir "${destination}" --base-model "${MODEL_ID}" --revision "${MODEL_REVISION}" \
    --run-token "${RUN_TOKEN}" --reviewed-manifest "${REVIEWED_MANIFEST_PATH}" \
    --output "${marker_tmp}" >/dev/null
  copy_atomic "${marker_tmp}" "${destination}/.complete.json"
  rm -f "${marker_tmp}"
  publish_partial_manifest "${run_dir}"
  touch "${SYNC_STATE}/${state_key}.done"
  echo "Durably saved and validated ${destination}"
}

copy_rollout() {
  local source="$1"
  local destination="$2"
  local source_hash destination_hash state_key state_file state_tmp
  source_hash="$(sha256sum "${source}" | awk '{print $1}')"
  state_key="$(printf '%s' "${destination}" | sha256sum | awk '{print $1}')"
  state_file="${SYNC_STATE}/${state_key}.rollout.sha256"
  if [[ -f "${state_file}" ]] && [[ "$(cat "${state_file}")" == "${source_hash}" ]]; then
    return 0
  fi
  copy_atomic "${source}" "${destination}"
  destination_hash="$(sha256sum "${destination}" | awk '{print $1}')"
  [[ "${source_hash}" == "${destination_hash}" ]] || {
    echo "Rollout verification failed: ${destination}" >&2
    return 1
  }
  state_tmp="${state_file}.partial.$$"
  printf '%s\n' "${source_hash}" > "${state_tmp}"
  mv -f "${state_tmp}" "${state_file}"
}

sync_run_artifacts() {
  local run_dir="$1"
  local run_name destination filename rollout relative provenance hash_tmp
  run_name="$(basename "${run_dir}")"
  destination="${DURABLE_ROOT}/${RUN_TOKEN}/${run_name}"
  mkdir -p \
    "${destination}/metadata/source/scripts" \
    "${destination}/metadata/source/src/train/verl" \
    "${destination}/metadata/source/src/evaluate" \
    "${destination}/rollouts" "${destination}/metrics" \
    "${destination}/evaluations" "${destination}/validation"

  for filename in config.json verl_config.yaml verl_full_config.yaml training.log train_dataset.parquet validation_dataset.parquet; do
    [[ -f "${run_dir}/${filename}" ]] || continue
    copy_atomic "${run_dir}/${filename}" "${destination}/metadata/${filename}"
  done

  shopt -s nullglob
  for rollout in "${run_dir}"/rollouts/*.jsonl; do
    copy_rollout "${rollout}" "${destination}/rollouts/$(basename "${rollout}")"
  done
  for filename in "${run_dir}"/offline_metrics/*; do
    [[ -f "${filename}" ]] || continue
    copy_atomic "${filename}" "${destination}/metrics/$(basename "${filename}")"
  done
  if [[ -d "${run_dir}/evaluations" ]]; then
    while IFS= read -r -d '' filename; do
      relative="${filename#"${run_dir}/evaluations/"}"
      copy_atomic "${filename}" "${destination}/evaluations/${relative}"
    done < <(find "${run_dir}/evaluations" -type f -name '*.json' -print0)
  fi
  for filename in "${run_dir}"/validation/*.json; do
    copy_atomic "${filename}" "${destination}/validation/$(basename "${filename}")"
  done
  shopt -u nullglob

  if [[ "${FINAL_SYNC:-false}" == 'true' ]]; then
    assert_wandb_secret_absent
    mkdir -p "${destination}/metadata/wandb"
    while IFS= read -r -d '' filename; do
      relative="${filename#"${WANDB_DIR}/"}"
      copy_atomic "${filename}" "${destination}/metadata/wandb/${relative}"
    done < <(find -L "${WANDB_DIR}" -type f -print0)
    mkdir -p "${destination}/metadata/datasets"
    for filename in \
      "${BASE_DATASET}" "${TRAIN_DATASET}" "${EVAL_BASE_DATASET}" \
      "${FIXED_EVAL_DATASET}" "${RANDOMIZED_EVAL_DATASET}" \
      "${run_dir}/train_dataset.parquet" "${run_dir}/validation_dataset.parquet"; do
      [[ -f "${filename}" ]] || continue
      copy_atomic "${filename}" "${destination}/metadata/datasets/$(basename "${filename}")"
    done
  fi
  mkdir -p "${destination}/metadata/source_provenance"
  for provenance in infra/skypilot/launch_provenance/*; do
    [[ -f "${provenance}" ]] || continue
    copy_atomic "${provenance}" \
      "${destination}/metadata/source_provenance/$(basename "${provenance}")"
  done
  copy_atomic "${REVIEWED_TASK_PATH}" "${destination}/metadata/$(basename "${REVIEWED_TASK_PATH}")"
  copy_atomic infra/skypilot/base_model_revision.txt "${destination}/metadata/base_model_revision.txt"
  copy_atomic infra/skypilot/run_reward_hack.sh "${destination}/metadata/run_reward_hack.sh"
  copy_atomic infra/skypilot/run_task.sh "${destination}/metadata/run_task.sh"
  copy_atomic "${REVIEWED_MANIFEST_PATH}" "${destination}/metadata/reviewed_manifest.json"
  [[ ! -f "${HARDWARE_REPORT}" ]] || \
    copy_atomic "${HARDWARE_REPORT}" "${destination}/metadata/hardware-detection.json"
  [[ ! -f "${EFFECTIVE_TRAINING_CONFIG}" ]] || \
    copy_atomic "${EFFECTIVE_TRAINING_CONFIG}" \
      "${destination}/metadata/effective-training-config.json"
  copy_atomic pyproject.toml "${destination}/metadata/pyproject.toml"
  copy_atomic uv.lock "${destination}/metadata/uv.lock"
  copy_atomic scripts/run_rl_training.py "${destination}/metadata/source/scripts/run_rl_training.py"
  copy_atomic scripts/run_data_process.py "${destination}/metadata/source/scripts/run_data_process.py"
  copy_atomic src/train/verl/grpo.py "${destination}/metadata/source/src/train/verl/grpo.py"
  copy_atomic src/train/verl/grpo_config.jinja2 "${destination}/metadata/source/src/train/verl/grpo_config.jinja2"
  copy_atomic src/train/verl/trainer.py "${destination}/metadata/source/src/train/verl/trainer.py"
  copy_atomic src/evaluate/helpers.py "${destination}/metadata/source/src/evaluate/helpers.py"

  hash_tmp="$(mktemp)"
  sha256sum \
    "${REVIEWED_TASK_PATH}" infra/skypilot/base_model_revision.txt \
    infra/skypilot/run_reward_hack.sh infra/skypilot/run_task.sh \
    infra/skypilot/read_wandb_secret.py infra/skypilot/verify_wandb_online.py \
    infra/skypilot/verify_adapter.py infra/skypilot/summarize_rollouts.py \
    infra/skypilot/evaluate_priority_checkpoints.py infra/skypilot/validate_peft_loads.py \
    infra/skypilot/finalize_package.py infra/skypilot/capture_source_provenance.py \
    infra/skypilot/verify_s3_after_termination.py infra/skypilot/update_partial_manifest.py \
    infra/skypilot/validate_reviewed_launch.py infra/skypilot/build_review_manifest.py \
    infra/skypilot/bounded_command.py infra/skypilot/teardown_watchdog.py \
    infra/skypilot/stage_reviewed_workdir.py \
    infra/skypilot/reviewed_skypilot_config.yaml "${REVIEWED_MANIFEST_PATH}" \
    scripts/run_rl_training.py scripts/run_data_process.py \
    src/train/config.py src/train/verl/grpo.py src/train/verl/grpo_config.jinja2 \
    src/train/verl/trainer.py src/evaluate/helpers.py src/evaluate/evaluation.py src/generate.py \
    verl/verl/trainer/config/model/hf_model.yaml verl/verl/utils/fs.py \
    verl/verl/workers/config/model.py verl/verl/workers/fsdp_workers.py \
    pyproject.toml uv.lock > "${hash_tmp}"
  copy_atomic "${hash_tmp}" "${destination}/metadata/source-sha256.txt"
  rm -f "${hash_tmp}"

  hash_tmp="$(mktemp)"
  sha256sum \
    "${BASE_DATASET}" "${TRAIN_DATASET}" "${EVAL_BASE_DATASET}" \
    "${FIXED_EVAL_DATASET}" "${RANDOMIZED_EVAL_DATASET}" > "${hash_tmp}"
  for filename in "${run_dir}/train_dataset.parquet" "${run_dir}/validation_dataset.parquet"; do
    [[ -f "${filename}" ]] || continue
    sha256sum "${filename}" >> "${hash_tmp}"
  done
  copy_atomic "${hash_tmp}" "${destination}/metadata/dataset-sha256.txt"
  rm -f "${hash_tmp}"
}

sync_checkpoints() {
  local run_dir adapter_dir
  [[ "${RUN_READY_FOR_SYNC}" == 'true' ]] || return 0
  shopt -s nullglob
  for run_dir in "${RUN_ROOT}"/*; do
    [[ -d "${run_dir}" && "${run_dir}" -nt "${RUN_MARKER}" ]] || continue
    sync_run_artifacts "${run_dir}"
    # Publish the run prefix before the first adapter copy, closing the narrow
    # crash window between a first .complete.json and launch-level discovery.
    publish_partial_manifest "${run_dir}"
    for adapter_dir in "${run_dir}"/checkpoints/global_step_*/actor/lora_adapter; do
      [[ -s "${adapter_dir}/adapter_model.safetensors" ]] || continue
      [[ -s "${adapter_dir}/adapter_config.json" ]] || continue
      copy_and_verify_adapter "${run_dir}" "${adapter_dir}"
    done
    publish_partial_manifest "${run_dir}"
  done
  shopt -u nullglob
}

sync_loop() {
  local consecutive_failures=0
  while sleep 60; do
    if sync_checkpoints; then
      consecutive_failures=0
    else
      consecutive_failures=$((consecutive_failures + 1))
      echo "Periodic durable sync failed (${consecutive_failures}/3)." >&2
      if [[ "${consecutive_failures}" -ge 3 ]]; then
        echo 'Stopping training because checkpoint durability could not be maintained.' >&2
        kill -TERM "${RUN_SCRIPT_PID}" 2>/dev/null || true
        return 1
      fi
    fi
  done
}

stop_training() {
  local attempt
  if [[ -z "${training_pid:-}" ]] || ! kill -0 "${training_pid}" 2>/dev/null; then
    training_pid=''
    return 0
  fi
  kill -TERM -- "-${training_pid}" 2>/dev/null || true
  for attempt in $(seq 1 60); do
    kill -0 "${training_pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${training_pid}" 2>/dev/null; then
    kill -KILL -- "-${training_pid}" 2>/dev/null || true
  fi
  wait "${training_pid}" 2>/dev/null || true
  training_pid=''
}

cleanup() {
  local status=$? sync_status=0
  trap - EXIT
  set +e
  stop_training
  if [[ -n "${sync_pid:-}" ]]; then
    kill "${sync_pid}" 2>/dev/null
    wait "${sync_pid}" 2>/dev/null
  fi
  FINAL_SYNC='true'
  sync_checkpoints || sync_status=$?
  if [[ "${sync_status}" -ne 0 ]]; then
    echo 'Final durable checkpoint sync failed.' >&2
    [[ "${status}" -ne 0 ]] || status="${sync_status}"
  fi
  exit "${status}"
}

sync_qualification_diagnostics() {
  local destination="${DURABLE_ROOT}/launches/${RUN_TOKEN}/memory-qualification"
  mkdir -p "${destination}"
  uv run --frozen --group dev --no-sync python - "${QUALIFICATION_ROOT}" "${RUN_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

needle = os.environ.get("WANDB_API_KEY", "").encode()
for root_name in sys.argv[1:]:
    root = Path(root_name)
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if path.is_file() and needle and needle in path.read_bytes():
            raise RuntimeError("Qualification diagnostic contains the W&B credential")
PY
  find "${QUALIFICATION_ROOT}" -maxdepth 1 -type f -exec cp -f {} "${destination}/" \;
  while IFS= read -r -d '' qualification_file; do
    local relative="${qualification_file#"${RUN_ROOT}/"}"
    mkdir -p "${destination}/run/$(dirname "${relative}")"
    cp -f "${qualification_file}" "${destination}/run/${relative}"
  done < <(find "${RUN_ROOT}" -type f \
    \( -name 'config.json' -o -name 'verl_config.yaml' -o -name 'verl_full_config.yaml' \
       -o -name 'training.log' -o -path '*/rollouts/*.jsonl' \) \
    -path '*_memory_qualification/*' -newer "${QUALIFICATION_MARKER}" -print0)
  sync
}

release_qualification_gpu_state() {
  timeout --signal=TERM --kill-after=30s 2m \
    uv run --frozen --group dev --no-sync ray stop --force >/dev/null 2>&1 || true
  local attempt active_pids
  for attempt in $(seq 1 60); do
    active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    [[ -z "${active_pids}" ]] && return 0
    sleep 1
  done
  echo 'Memory qualification left GPU compute processes active; failing closed.' >&2
  return 1
}

run_memory_qualification() {
  local qualification_status=0
  [[ "${RUN_MEMORY_QUALIFICATION}" == 'true' ]] || return 0
  echo 'Starting reviewed one-step P4d memory qualification with W&B disabled.'
  touch "${QUALIFICATION_MARKER}"
  if WANDB_MODE=disabled WANDB_JOB_TYPE=memory-qualification \
    WANDB_RUN_GROUP="${RUN_TOKEN}-memory-qualification" \
    WANDB_DIR="${PWD}/${QUALIFICATION_ROOT}/wandb" \
    WANDB_RUN_METADATA_PATH= \
    setsid timeout --signal=TERM --kill-after=10m 90m \
      uv run --frozen --group dev --no-sync python scripts/run_rl_training.py no_intervention \
        --model_id="${MODEL_ID}" --model_revision="${MODEL_REVISION}" --task="${TASK}" \
        --steps=1 --seed=1 --num_prompts=16 --num_generations=16 \
        --max_prompt_length=1536 --max_completion_length=1536 \
        --save_steps=999 --save_only_model=True \
        --per_device_batch_size="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
        --gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
        --max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
        --run_name_suffix=_memory_qualification \
        >"${QUALIFICATION_ROOT}/qualification.log" 2>&1; then
    qualification_status=0
  else
    qualification_status=$?
  fi
  printf '{"hardware_profile":"%s","status":%d}\n' \
    "${HARDWARE_PROFILE}" "${qualification_status}" \
    > "${QUALIFICATION_ROOT}/status.json"
  sync_qualification_diagnostics || return 1
  release_qualification_gpu_state || return 1
  if [[ "${qualification_status}" -ne 0 ]]; then
    echo 'P4d memory qualification failed; diagnostics were synced and configuration was not mutated.' >&2
    return "${qualification_status}"
  fi
  echo 'P4d memory qualification completed and all GPU compute processes were released.'
}

on_interrupt() { stop_training; exit 130; }
on_terminate() { stop_training; exit 143; }

trap cleanup EXIT
trap on_interrupt INT
trap on_terminate TERM

uv run --frozen --group dev --no-sync python infra/skypilot/validate_gpu_hardware.py \
  --profile "${HARDWARE_PROFILE}" --output "${HARDWARE_REPORT}"
uv run --frozen --group dev --no-sync python \
  infra/skypilot/render_effective_training_config.py \
  --profile "${HARDWARE_PROFILE}" --output "${EFFECTIVE_TRAINING_CONFIG}"

uv run --frozen --group dev --no-sync python - <<'PY'
import torch
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print('Verified 8 CUDA devices:', [torch.cuda.get_device_name(i) for i in range(8)])
PY

if [[ "$(cat infra/skypilot/base_model_revision.txt)" != "${MODEL_REVISION}" ]]; then
  echo 'Pinned model revision file does not match the reviewed run script.' >&2
  exit 1
fi

timeout --signal=TERM --kill-after=30s 2m \
  uv run --frozen --group dev --no-sync python infra/skypilot/verify_wandb_online.py \
    --output "${WANDB_DIR}/online-preflight.json" \
    --expected-project "${WANDB_PROJECT_NAME}" \
    --expected-run-group "${RUN_TOKEN}" \
    --expected-sdk-version "${WANDB_SDK_VERSION}"

uv run --frozen --group dev --no-sync python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
model = "Qwen/Qwen3-4B"
revision = "1cfa9a7208912126459214e8b04321603b3df60c"
snapshot = Path(snapshot_download(model, revision=revision))
if snapshot.name != revision:
    raise RuntimeError(f"Resolved unexpected Hugging Face snapshot: {snapshot}")
print(f"Pinned model snapshot ready: {snapshot}")
PY

uv run --frozen --group dev --no-sync python scripts/run_data_process.py create \
  --base_dataset_fpath="${BASE_DATASET}" --hint="${TASK}" --model_id="${MODEL_ID}" \
  --model_revision="${MODEL_REVISION}" --max_prompt_length=1536 --overwrite=True --seed=1
uv run --frozen --group dev --no-sync python scripts/run_data_process.py create \
  --base_dataset_fpath="${EVAL_BASE_DATASET}" --hint="${FIXED_EVAL_TASK}" --model_id="${MODEL_ID}" \
  --model_revision="${MODEL_REVISION}" --max_prompt_length=1536 --overwrite=True --seed=1
uv run --frozen --group dev --no-sync python scripts/run_data_process.py create \
  --base_dataset_fpath="${EVAL_BASE_DATASET}" --hint="${RANDOMIZED_EVAL_TASK}" --model_id="${MODEL_ID}" \
  --model_revision="${MODEL_REVISION}" --max_prompt_length=1536 --overwrite=True --seed=1

export CODE_EVAL_SANDBOX='bwrap'
export CODE_EVAL_SANDBOX_REQUIRED='1'
uv run --frozen --group dev --no-sync python - <<'PY'
from src.evaluate.helpers import run_code_subprocess
program = r'''
import json
import os
import socket
aws_environment_absent = not any(key.startswith("AWS_") for key in os.environ)
wandb_key_absent = "WANDB_API_KEY" not in os.environ
aws_files_absent = not os.path.exists("/root/.aws") and not os.path.exists("/home/ubuntu/.aws")
network_blocked = False
try:
    socket.create_connection(("169.254.169.254", 80), timeout=0.1)
except OSError:
    network_blocked = True
checkpoint_write_blocked = False
try:
    with open("/durable-checkpoints/evaluator-write-test", "w", encoding="utf-8") as handle:
        handle.write("unexpected")
except OSError:
    checkpoint_write_blocked = True
assert aws_environment_absent and wandb_key_absent and aws_files_absent and network_blocked and checkpoint_write_blocked
print(json.dumps({"aws_environment_absent": aws_environment_absent, "aws_files_absent": aws_files_absent,
                  "wandb_key_absent": wandb_key_absent, "network_blocked": network_blocked,
                  "checkpoint_write_blocked": checkpoint_write_blocked}))
'''
result = run_code_subprocess(program, timeout=2, memory_limit=256)
if not result.success or not all(result.stdout.values()):
    raise RuntimeError(f"Evaluator sandbox smoke test failed: {result}")
print("Evaluator sandbox smoke test passed:", result.stdout)
PY

# The S3 mount is managed outside this process. Training and generated-code
# evaluators do not receive ambient AWS environment credentials.
while IFS='=' read -r variable _; do
  [[ "${variable}" == AWS_* ]] && unset "${variable}"
done < <(env)

run_memory_qualification
# Qualification weights are never used.  This timestamp excludes its run
# directory from all real-run discovery and durable checkpoint packaging.
touch "${RUN_MARKER}"
RUN_READY_FOR_SYNC='true'
sync_loop &
sync_pid=$!

setsid timeout --signal=TERM --kill-after=10m 600m \
  uv run --frozen --group dev --no-sync python scripts/run_rl_training.py no_intervention \
    --model_id="${MODEL_ID}" --model_revision="${MODEL_REVISION}" --task="${TASK}" \
    --steps="${STEPS}" --seed=1 --num_prompts=16 --num_generations=16 \
    --max_prompt_length=1536 --max_completion_length=1536 \
    --save_steps="${SAVE_STEPS}" --save_only_model=True \
    --per_device_batch_size="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    --gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max_num_seqs="${VLLM_MAX_NUM_SEQS}" &
training_pid=$!

if wait "${training_pid}"; then training_status=0; else training_status=$?; fi
training_pid=''
[[ "${training_status}" -eq 0 ]] || exit "${training_status}"

run_dir="$(find_run_dir)"
mkdir -p "${run_dir}/offline_metrics" "${run_dir}/evaluations" "${run_dir}/validation"
uv run --frozen --group dev --no-sync python infra/skypilot/summarize_rollouts.py \
  --rollout-dir "${run_dir}/rollouts" \
  --output-json "${run_dir}/offline_metrics/rollout_metrics.json" \
  --output-csv "${run_dir}/offline_metrics/rollout_metrics.csv" \
  --expected-start 1 --expected-end 200 --window-start 70 --window-end 110

uv run --frozen --group dev --no-sync python infra/skypilot/evaluate_priority_checkpoints.py \
  --run-dir "${run_dir}" --fixed-dataset "${FIXED_EVAL_DATASET}" \
  --randomized-dataset "${RANDOMIZED_EVAL_DATASET}" --output-dir "${run_dir}/evaluations" \
  --base-model "${MODEL_ID}" --revision "${MODEL_REVISION}" --steps 80 90 100 200

uv run --frozen --group dev --no-sync python infra/skypilot/validate_peft_loads.py \
  --run-dir "${run_dir}" --output "${run_dir}/validation/peft_loads.json" \
  --base-model "${MODEL_ID}" --revision "${MODEL_REVISION}" --steps 80 90 100 200

kill "${sync_pid}" 2>/dev/null || true
wait "${sync_pid}" 2>/dev/null || true
sync_pid=''
# Re-copy and re-hash every adapter once after training, even if its periodic
# copy previously succeeded. This makes final packaging recover from any
# transient mount/write issue instead of trusting an old local done marker.
find "${SYNC_STATE}" -maxdepth 1 -type f -name '*.done' -delete
FINAL_SYNC='true'
sync_checkpoints
sync_run_artifacts "${run_dir}"

run_name="$(basename "${run_dir}")"
run_prefix="qwen3-4b/no-intervention/${RUN_TOKEN}/${run_name}"
durable_run_dir="${DURABLE_ROOT}/${RUN_TOKEN}/${run_name}"
pointer="${DURABLE_ROOT}/launches/${RUN_TOKEN}/result.json"
uv run --frozen --group dev --no-sync python infra/skypilot/finalize_package.py \
  --run-dir "${run_dir}" --durable-run-dir "${durable_run_dir}" --pointer "${pointer}" \
  --run-prefix "${run_prefix}" --base-model "${MODEL_ID}" --revision "${MODEL_REVISION}" \
  --run-token "${RUN_TOKEN}" --hardware-profile "${HARDWARE_PROFILE}"
sync
