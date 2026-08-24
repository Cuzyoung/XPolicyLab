#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/deploy.yml"

usage() {
  cat <<'EOF'
Usage:
  bash setup_eval_policy_server.sh [CONFIG_PATH]
  bash setup_eval_policy_server.sh BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED GPU POLICY_ENV PORT [HOST]
EOF
}

config_path="${COSMOS3_DEPLOY_CONFIG:-${DEFAULT_CONFIG}}"
standard_mode=0
case "$#" in
  0) ;;
  1)
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
    config_path=$1
    ;;
  9|10)
    standard_mode=1
    bench_name=$1
    task_name=$2
    ckpt_name=$3
    env_cfg_type=$4
    action_type=$5
    seed=$6
    policy_gpu_id=$7
    policy_env=$8
    policy_server_port=$9
    policy_server_host=${10:-127.0.0.1}
    ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "${config_path}" != /* ]]; then
  config_path="${WORKSPACE_ROOT}/${config_path}"
fi
if [[ ! -f "${config_path}" ]]; then
  echo "Cosmos3 deploy config not found: ${config_path}" >&2
  exit 1
fi

policy_gpu_id="${policy_gpu_id:-${COSMOS3_GPU_ID:-0}}"
policy_env="${policy_env:-${COSMOS3_ENV_DIR:-${WORKSPACE_ROOT}/envs/cosmos3}}"
if [[ "${policy_env}" == "uv" ]]; then
  policy_env="${COSMOS3_ENV_DIR:-${WORKSPACE_ROOT}/envs/cosmos3}"
elif [[ "${policy_env}" != /* ]]; then
  policy_env="${WORKSPACE_ROOT}/${policy_env}"
fi
if [[ "${policy_env}" == */.venv ]]; then
  python_bin="${policy_env}/bin/python"
else
  python_bin="${policy_env}/.venv/bin/python"
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Cosmos3 Python not found: ${python_bin}" >&2
  echo "Run: bash ${SCRIPT_DIR}/install.sh" >&2
  exit 1
fi

common_env=(
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
  PYTHONUNBUFFERED=1
  PYTHONWARNINGS=ignore::UserWarning
  PYTHONPATH="${WORKSPACE_ROOT}:${SCRIPT_DIR}/cosmos-framework:${PYTHONPATH:-}"
)

cd "${WORKSPACE_ROOT}"
if [[ "${standard_mode}" == "1" ]]; then
  exec env "${common_env[@]}" "${python_bin}" "${XPOLICYLAB_ROOT}/setup_policy_server.py" \
    --config_path "${config_path}" \
    --overrides \
      port="${policy_server_port}" \
      host="${policy_server_host}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name="Cosmos3" \
      action_type="${action_type}" \
      gpu_id="${policy_gpu_id}"
fi

exec env "${common_env[@]}" "${python_bin}" "${XPOLICYLAB_ROOT}/setup_policy_server.py" \
  --config_path "${config_path}" \
  --overrides gpu_id="${policy_gpu_id}"
