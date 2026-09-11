#!/usr/bin/env bash

set -Eeuo pipefail

if [[ ! "${CODEX_ACTIVATION_LAUNCH_ID:-}" =~ ^[0-9a-f]{32}$ ]]; then
  echo 'This entry point requires a one-use direct-launch identity.' >&2
  exit 2
fi
if [[ "$#" -lt 2 ]]; then
  echo 'Usage: activation_direct_job.sh PYTHON RUNNER [RUNNER_ARGS...]' >&2
  exit 2
fi

readonly PYTHON_BIN="$1"
readonly RUNNER="$2"
shift 2
readonly -a RUNNER_ARGS=("$@")

[[ -x "${PYTHON_BIN}" && -f "${RUNNER}" ]] || {
  echo 'Reviewed Python or activation runner is unavailable.' >&2
  exit 3
}
for command in timeout taskset nice ionice nvidia-smi flock systemctl; do
  command -v "${command}" >/dev/null || {
    echo "Required direct-host command is unavailable: ${command}" >&2
    exit 3
  }
done

export CODEX_ACTIVATION_WRAPPER_PID="$$"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

while IFS='=' read -r variable _; do
  if [[ "${variable}" == AWS_* || "${variable}" == WANDB_* ||
        "${variable}" == AZURE_* || "${variable}" == GOOGLE_* ||
        "${variable}" == HF_TOKEN || "${variable}" == HUGGING_FACE_HUB_TOKEN ||
        "${variable}" == GITHUB_TOKEN || "${variable}" == OPENAI_API_KEY ]]; then
    unset "${variable}"
  fi
done < <(env)

scratch_root=''
output_dir=''
run_token=''
wrapper_wall_limit=''
expected_hostname=''
service_unit=''
supervisor_status=''
for ((index = 0; index < ${#RUNNER_ARGS[@]}; index++)); do
  next_index=$((index + 1))
  case "${RUNNER_ARGS[$index]}" in
    --scratch-root)
      scratch_root="${RUNNER_ARGS[$next_index]:-}"
      ;;
    --output-dir)
      output_dir="${RUNNER_ARGS[$next_index]:-}"
      ;;
    --run-token)
      run_token="${RUNNER_ARGS[$next_index]:-}"
      ;;
    --wrapper-wall-limit-seconds)
      wrapper_wall_limit="${RUNNER_ARGS[$next_index]:-}"
      ;;
    --expected-hostname)
      expected_hostname="${RUNNER_ARGS[$next_index]:-}"
      ;;
    --service-unit)
      service_unit="${RUNNER_ARGS[$next_index]:-}"
      ;;
    --supervisor-status)
      supervisor_status="${RUNNER_ARGS[$next_index]:-}"
      ;;
  esac
done

[[ -n "${scratch_root}" && -n "${output_dir}" && -n "${run_token}" &&
   -n "${service_unit}" && -n "${supervisor_status}" ]] || {
  echo 'Reviewed scratch/output/token arguments are missing.' >&2
  exit 4
}
[[ "${wrapper_wall_limit}" == '16200' ]] || {
  echo 'Direct-host wrapper wall limit differs from the reviewed 16,200 seconds.' >&2
  exit 4
}
[[ "${expected_hostname}" == 'gpu-03' && "$(hostname -s)" == 'gpu-03' ]] || {
  echo 'Direct-host wrapper is bound to gpu-03.' >&2
  exit 4
}
[[ "${service_unit}" == "${run_token}" &&
   "${CODEX_ACTIVATION_SYSTEMD_UNIT:-}" == "${service_unit}" ]] || {
  echo 'Direct-host wrapper is not running under the reviewed systemd unit.' >&2
  exit 4
}
[[ "$(systemctl --user show "${service_unit}.service" -p MainPID --value)" == "$$" ]] || {
  echo 'Direct-host wrapper is not the reviewed systemd service main process.' >&2
  exit 4
}

readonly host_lock="/run/lock/codex-checkpoint60-activations-${expected_hostname}.lock"
[[ ! -L "${host_lock}" ]] || {
  echo 'The direct-host lock must not be a symbolic link.' >&2
  exit 6
}
exec {host_lock_fd}<>"${host_lock}"
chmod 600 "${host_lock}"
[[ -f "${host_lock}" &&
   "$(stat -Lc '%u:%a' "${host_lock}")" == "$(id -u):600" ]] || {
  echo 'The direct-host lock is not an owner-only regular file.' >&2
  exit 6
}
flock -n "${host_lock_fd}" || {
  echo 'Another reviewed activation extraction holds the host lock.' >&2
  exit 6
}
export CODEX_ACTIVATION_HOST_LOCK_PATH="${host_lock}"
export CODEX_ACTIVATION_HOST_LOCK_FD="${host_lock_fd}"

child_pid=''
forward_signal() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM "${child_pid}" 2>/dev/null || true
  fi
}

cleanup_scratch() {
  local original_status="$?"
  trap - EXIT TERM INT HUP
  if ! "${PYTHON_BIN}" "${RUNNER}" \
      --cleanup-scratch-only \
      --scratch-root "${scratch_root}" \
      --output-dir "${output_dir}" \
      --run-token "${run_token}"; then
    echo 'Marker-validated scratch recovery cleanup failed.' >&2
    if [[ "${original_status}" -eq 0 ]]; then
      original_status=5
    fi
  fi
  exit "${original_status}"
}

trap forward_signal TERM INT HUP
trap cleanup_scratch EXIT

timeout --signal=TERM --kill-after=120s "${wrapper_wall_limit}s" \
  nice -n 5 ionice -c 2 -n 7 \
  "${PYTHON_BIN}" "${RUNNER}" "${RUNNER_ARGS[@]}" &
child_pid="$!"
wait "${child_pid}"
