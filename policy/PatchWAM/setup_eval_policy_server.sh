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
UTILS_DIR="${XPL_ROOT}/utils"
UPSTREAM_DIR="${SCRIPT_DIR}/PatchWAM"
yaml_file="${SCRIPT_DIR}/deploy.yml"

allow_dummy_policy=${PATCHWAM_ALLOW_DUMMY_POLICY:-false}
checkpoint_path=${PATCHWAM_CHECKPOINT_PATH:-}
dataset_stats_path=${PATCHWAM_DATASET_STATS_PATH:-${SCRIPT_DIR}/assets/dataset_stats.json}
runtime_config_path=${PATCHWAM_RUNTIME_CONFIG_PATH:-${SCRIPT_DIR}/assets/c2r_dr4_resolved.yaml}
flux2_model_path=${PATCHWAM_FLUX2_MODEL:-}
ae_model_path=${PATCHWAM_AE_MODEL:-}
qwen3_model_spec=${PATCHWAM_QWEN3_MODEL:-Qwen/Qwen3-4B}

if [[ "${allow_dummy_policy}" != "true" ]]; then
    : "${flux2_model_path:?Set PATCHWAM_FLUX2_MODEL to FLUX.2 Klein base 4B weights}"
    : "${ae_model_path:?Set PATCHWAM_AE_MODEL to FLUX.2 autoencoder weights}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${BENCH_ROOT}:${UPSTREAM_DIR}:${UPSTREAM_DIR}/src:${UPSTREAM_DIR}/vendor/flux2/src:${PYTHONPATH:-}" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name=PatchWAM \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            checkpoint_path="${checkpoint_path}" \
            dataset_stats_path="${dataset_stats_path}" \
            runtime_config_path="${runtime_config_path}" \
            flux2_model_path="${flux2_model_path}" \
            ae_model_path="${ae_model_path}" \
            qwen3_model_spec="${qwen3_model_spec}" \
            allow_dummy_policy="${allow_dummy_policy}"
