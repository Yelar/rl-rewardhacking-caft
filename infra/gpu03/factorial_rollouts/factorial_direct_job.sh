#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ! "${CODEX_FACTORIAL_LAUNCH_ID:-}" =~ ^[0-9a-f]{32}$ ]]; then
  echo 'This entry point requires a consumed reviewed-launch identity.' >&2
  exit 2
fi
if [[ "$#" -lt 3 ]]; then
  echo 'Usage: factorial_direct_job.sh EXPECTED_HOST PYTHON RUNNER [ARGS...]' >&2
  exit 2
fi

readonly EXPECTED_HOST="$1"
readonly PYTHON_BIN="$2"
readonly RUNNER="$3"
shift 3
readonly -a RUNNER_ARGS=("$@")

[[ "$(hostname -s)" == "${EXPECTED_HOST}" ]] || {
  echo 'The reviewed job is bound to a different host.' >&2
  exit 3
}
[[ -x "${PYTHON_BIN}" && -f "${RUNNER}" ]] || {
  echo 'Reviewed Python or collector is unavailable.' >&2
  exit 3
}
for command in timeout taskset nice ionice nvidia-smi flock systemctl setsid stat; do
  command -v "${command}" >/dev/null || {
    echo "Required command is unavailable: ${command}" >&2
    exit 3
  }
done

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Generation is offline and W&B is optional. Remove every ambient credential;
# the generated-code subprocess separately uses bubblewrap --unshare-all.
while IFS='=' read -r variable _; do
  if [[ "${variable}" == AWS_* || "${variable}" == WANDB_* ||
        "${variable}" == AZURE_* || "${variable}" == GOOGLE_* ||
        "${variable}" == HF_TOKEN || "${variable}" == HUGGING_FACE_HUB_TOKEN ||
        "${variable}" == GITHUB_TOKEN || "${variable}" == OPENAI_API_KEY ]]; then
    unset "${variable}"
  fi
done < <(env)

readonly lock_path="/run/lock/codex-factorial-rollouts-${EXPECTED_HOST}.lock"
[[ ! -L "${lock_path}" ]] || { echo 'Unsafe host lock symlink.' >&2; exit 4; }
exec {lock_fd}<>"${lock_path}"
chmod 600 "${lock_path}"
[[ -f "${lock_path}" && "$(stat -Lc '%u:%a' "${lock_path}")" == "$(id -u):600" ]] || {
  echo 'Host lock is not an owner-only regular file.' >&2
  exit 4
}
flock -n "${lock_fd}" || { echo 'Another reviewed GPU job holds the host lock.' >&2; exit 4; }

child_pid=''
forward_signal() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -TERM -- "-${child_pid}" 2>/dev/null || true
  fi
}
trap forward_signal TERM INT HUP

# The reviewed collector enforces the scientific deadline. This independent
# outer bound permits the exact-100K profile's 24-hour deadline plus cleanup.
setsid timeout --signal=TERM --kill-after=120s 90000s \
  nice -n 5 ionice -c 2 -n 7 \
  "${PYTHON_BIN}" "${RUNNER}" "${RUNNER_ARGS[@]}" &
child_pid="$!"
wait "${child_pid}"
