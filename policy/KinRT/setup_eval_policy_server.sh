#!/usr/bin/env bash
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

if [[ "${policy_uv_env}" == "uv" ]]; then
  policy_uv_env="$(awk '/^policy_uv_env_path:/{print $2; exit}' "${yaml_file}")"
fi
policy_uv_env_path="$(
  python3 - "${policy_uv_env}" "${SCRIPT_DIR}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
script_dir = Path(sys.argv[2])
print(path if path.is_absolute() else (script_dir / path).resolve())
PY
)"
if [[ ! -x "${policy_uv_env_path}/.venv/bin/python" ]]; then
  echo "[SERVER][ERROR] uv environment not found: ${policy_uv_env_path}/.venv" >&2
  echo "[SERVER][ERROR] Run: bash ${SCRIPT_DIR}/install.sh" >&2
  exit 1
fi

source "${policy_uv_env_path}/.venv/bin/activate"
PYTHON_BIN="${KINRT_PYTHON_BIN:-$(command -v python)}"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${policy_uv_env_path}}"
overrides=(
  "port=${policy_server_port}"
  "host=${policy_server_host}"
  "bench_name=${bench_name}"
  "task_name=${task_name}"
  "ckpt_name=${ckpt_name}"
  "env_cfg_type=${env_cfg_type}"
  "seed=${seed}"
  "policy_name=${policy_name}"
  "action_type=${action_type}"
)
if [[ -n "${KINRT_TRAIN_CONFIG_NAME:-}" ]]; then
  overrides+=("train_config_name=${KINRT_TRAIN_CONFIG_NAME}")
fi
if [[ -n "${KINRT_REPO_ID:-}" ]]; then
  overrides+=("repo_id=${KINRT_REPO_ID}")
fi
if [[ -n "${KINRT_CHECKPOINT_PATH:-}" ]]; then
  overrides+=("checkpoint_path=${KINRT_CHECKPOINT_PATH}")
fi
if [[ -n "${KINRT_CHECKPOINT_NUM:-}" ]]; then
  overrides+=("checkpoint_num=${KINRT_CHECKPOINT_NUM}")
fi
if [[ -n "${KINRT_ACTION_CHUNK_SIZE:-}" ]]; then
  overrides+=("action_chunk_size=${KINRT_ACTION_CHUNK_SIZE}")
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
echo "[SERVER] OpenPI root=${OPENPI_ROOT}"

exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONPATH="${BENCH_ROOT}:${OPENPI_ROOT}/src${KINRT_EXTRA_PYTHONPATH:+:${KINRT_EXTRA_PYTHONPATH}}" \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides "${overrides[@]}"
