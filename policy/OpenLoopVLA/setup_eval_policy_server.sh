#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    policy_python="${policy_conda_env}/bin/python"
    echo -e "\033[33m[SERVER] Using Python environment path: ${policy_conda_env}\033[0m"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    policy_python="python"
fi

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
if [[ -n "${OPENLOOPVLA_CHECKPOINT_PATH:-}" ]]; then
    overrides+=("checkpoint_path=${OPENLOOPVLA_CHECKPOINT_PATH}")
fi
if [[ -n "${OPENLOOPVLA_V2_PACKAGE_ROOT:-}" ]]; then
    overrides+=("v2_package_root=${OPENLOOPVLA_V2_PACKAGE_ROOT}")
fi
if [[ -n "${OPENLOOPVLA_UNNORM_KEY:-}" ]]; then
    overrides+=("unnorm_key=${OPENLOOPVLA_UNNORM_KEY}")
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${SCRIPT_DIR}/OpenLoopVLA:${PYTHONPATH:-}" \
    STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-1}" \
    "${policy_python}" -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${overrides[@]}"
