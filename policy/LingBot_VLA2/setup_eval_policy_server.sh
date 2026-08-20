#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
UTILS_DIR="${XPOLICYLAB_ROOT}/utils"
DEFAULT_CONFIG="${SCRIPT_DIR}/deploy.yml"

usage() {
  cat <<'EOF'
Usage:
  bash setup_eval_policy_server.sh
  bash setup_eval_policy_server.sh CONFIG_PATH
  bash setup_eval_policy_server.sh BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED GPU POLICY_ENV PORT [HOST]

The first two forms load one deploy file directly. The positional form is the
standard XPolicyLab policy-server interface used by eval.sh.
EOF
}

config_path="${LINGBOT_VLA2_DEPLOY_CONFIG:-${DEFAULT_CONFIG}}"
standard_mode=0

case "$#" in
  0)
    ;;
  1)
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
      usage
      exit 0
    fi
    config_path="$1"
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
    config_path="${LINGBOT_VLA2_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "${config_path}" != /* ]]; then
  if [[ -f "${PWD}/${config_path}" ]]; then
    config_path="${PWD}/${config_path}"
  else
    config_path="${WORKSPACE_ROOT}/${config_path}"
  fi
fi
if [[ ! -f "${config_path}" ]]; then
  echo "LingBot-VLA2 deploy config not found: ${config_path}" >&2
  exit 1
fi

policy_gpu_id="${policy_gpu_id:-${LINGBOT_VLA2_GPU_ID:-0}}"
policy_env="${policy_env:-${LINGBOT_VLA2_ENV_DIR:-${WORKSPACE_ROOT}/envs/lingbot-vla2}}"
if [[ "${policy_env}" == "uv" ]]; then
  policy_env="${LINGBOT_VLA2_ENV_DIR:-${WORKSPACE_ROOT}/envs/lingbot-vla2}"
elif [[ "${policy_env}" != /* ]]; then
  policy_env="${SCRIPT_DIR}/${policy_env}"
fi

if [[ "${policy_env}" == */.venv ]]; then
  python_bin="${policy_env}/bin/python"
else
  python_bin="${policy_env}/.venv/bin/python"
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "LingBot-VLA2 Python not found: ${python_bin}" >&2
  echo "Run: bash ${SCRIPT_DIR}/install.sh" >&2
  exit 1
fi

env_cfg_type="${env_cfg_type:-yam_dual}"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${WORKSPACE_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=LingBot_VLA2 config=${config_path} gpu=${policy_gpu_id} action_dim=${action_dim}"

cd "${WORKSPACE_ROOT}"
if [[ "${standard_mode}" == "1" ]]; then
  exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${WORKSPACE_ROOT}:${SCRIPT_DIR}/lingbot_vla_v2:${PYTHONPATH:-}" \
    "${python_bin}" "${XPOLICYLAB_ROOT}/setup_policy_server.py" \
      --config_path "${config_path}" \
      --overrides \
        port="${policy_server_port}" \
        host="${policy_server_host}" \
        bench_name="${bench_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        env_cfg_type="${env_cfg_type}" \
        seed="${seed}" \
        policy_name="LingBot_VLA2" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        gpu_id="${policy_gpu_id}"
fi

exec env \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  PYTHONPATH="${WORKSPACE_ROOT}:${SCRIPT_DIR}/lingbot_vla_v2:${PYTHONPATH:-}" \
  "${python_bin}" "${XPOLICYLAB_ROOT}/setup_policy_server.py" \
    --config_path "${config_path}" \
    --overrides \
      action_dim="${action_dim}" \
      gpu_id="${policy_gpu_id}"
