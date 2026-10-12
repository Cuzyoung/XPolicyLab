#!/bin/bash
set -euo pipefail

# XPolicyLab-standard training wrapper for InternW0_delta.
#
# Launches the official InternW0-Delta RoboDojo post-training recipe on the
# dataset written by process_data.sh, using the same local Wan2.2/RynnBrain
# assets as evaluation, and writes artifacts to:
#   <POLICY_DIR>/checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]

bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}

if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[InternW0_delta] ERROR: the official post-training recipe supports bench_name=RoboDojo, got '${bench_name}'." >&2
    exit 1
fi
if [[ "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
    echo "[InternW0_delta] ERROR: the RoboDojo recipe is 14D arx_x5 joint space, got env_cfg_type='${env_cfg_type}' action_type='${action_type}'." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
UPSTREAM_ROOT="${INTERNW0_DELTA_ROOT:-${POLICY_DIR}/InternW0-Delta}"
ASSET_ROOT="${INTERNW0_ASSET_ROOT:-${POLICY_DIR}/assets}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
dataset_dir="${ROBODOJO_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
wan_dir="${WAM_WAN22_PATH:-${ASSET_ROOT}/Wan-AI/Wan2.2-TI2V-5B}"
vlm_dir="${WAM_RYNNBRAIN_PATH:-${ASSET_ROOT}/Alibaba-DAMO-Academy/RynnBrain1.1-2B}"
pretrain_ckpt="${WAM_PRETRAIN_CHECKPOINT:-${POLICY_DIR}/checkpoints/pretrain.pt}"

if [[ ! -f "${UPSTREAM_ROOT}/run.sh" || ! -f "${UPSTREAM_ROOT}/configs/task/robodojo.yaml" ]]; then
    echo "[InternW0_delta] ERROR: full InternW0-Delta source not found at ${UPSTREAM_ROOT}." >&2
    echo "[InternW0_delta] Run sync_upstream.sh, or set INTERNW0_DELTA_ROOT." >&2
    exit 1
fi
for required in "${wan_dir}/Wan2.2_VAE.pth" "${vlm_dir}" "${pretrain_ckpt}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[InternW0_delta] ERROR: missing ${required}." >&2
        echo "[InternW0_delta] Run download_assets.sh, and place InternW0-Delta-Base at checkpoints/pretrain.pt (or set WAM_PRETRAIN_CHECKPOINT)." >&2
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WAM_CACHE_ROOT="${WAM_CACHE_ROOT:-${POLICY_DIR}/.cache/internw0}"

if [[ ! -f "${dataset_dir}/meta/episodes.jsonl" ]]; then
    echo "[InternW0_delta] LeRobot dataset not found at ${dataset_dir}; running process_data.sh first."
    ROBODOJO_DATA_ROOT="${dataset_dir}" bash "${POLICY_DIR}/process_data.sh" \
        "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
        "${INTERNW0_MAX_EPISODE:-}" "${INTERNW0_RAW_TASK_DIRS:-${ckpt_name}}"
fi

if [[ -n "${INTERNW0_MASTER_PORT:-}" ]]; then
    master_port="${INTERNW0_MASTER_PORT}"
elif [[ -x "${UTILS_DIR}/get_free_port.sh" ]]; then
    master_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
else
    master_port=29500
fi

mkdir -p "${ckpt_dir}"

export NPROC_PER_NODE="${num_gpus}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-${master_port}}"
export RUN_ID="${RUN_ID:-${ckpt_setting}}"
export ROBODOJO_DATA_ROOT="${dataset_dir}"
export WAM_PRETRAIN_CHECKPOINT="${pretrain_ckpt}"
export WAM_VLM_PATH="${vlm_dir}"
export WAM_CHECKPOINT_ROOT="${WAM_CHECKPOINT_ROOT:-${POLICY_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[InternW0_delta train] bench=${bench_name} ckpt=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed}"
echo "[InternW0_delta train] upstream=${UPSTREAM_ROOT}"
echo "[InternW0_delta train] dataset=${dataset_dir}"
echo "[InternW0_delta train] output_dir=${ckpt_dir}"
echo "[InternW0_delta train] gpus=${gpu_id} nproc_per_node=${num_gpus} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[InternW0_delta train] pretrain=${pretrain_ckpt}"

# Hydra keeps the last value of a repeated key, so output_dir overrides run.sh's default.
train_overrides=(
    "output_dir=${ckpt_dir}"
    "seed=${seed}"
    "model.model_id=${wan_dir}"
    "model.tokenizer_model_id=${wan_dir}"
    "model.redirect_common_files=false"
)
if [[ -n "${INTERNW0_TRAIN_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    train_overrides+=(${INTERNW0_TRAIN_OVERRIDES})
fi

cd "${UPSTREAM_ROOT}"
exec bash run.sh robodojo "${train_overrides[@]}"
