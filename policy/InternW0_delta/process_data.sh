#!/bin/bash
set -euo pipefail

# XPolicyLab-standard data preparation wrapper for InternW0_delta.
#
# Converts XPolicyLab/RoboDojo HDF5 trajectories with the official LeRobot v2.1
# converter (the version read by InternW0-Delta's post-training loader) into
#   <POLICY_DIR>/data/<bench>-<ckpt>-<env>-<action>/{meta,data,videos}
# and builds the official RoboDojo text-embedding cache for that dataset.
#
# Usage:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]
#
# raw_task_dirs defaults to <ckpt_name>; a comma-separated list merges several
# task folders into one dataset.
#
# Optional environment:
#   INTERNW0_CONVERT_PYTHON  python with `lerobot` (v2.1 API) for the conversion step
#   ROBODOJO_DATA_ROOT       use/write the LeRobot dataset at this path
#   INTERNW0_SKIP_TEXT_CACHE=1

bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
expert_data_num=${5:-}
raw_task_dirs=${6:-${INTERNW0_RAW_TASK_DIRS:-${ckpt_name}}}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${INTERNW0_DELTA_ROOT:-${POLICY_DIR}/InternW0-Delta}"
ASSET_ROOT="${INTERNW0_ASSET_ROOT:-${POLICY_DIR}/assets}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_dir="${ROBODOJO_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
convert_python="${INTERNW0_CONVERT_PYTHON:-python}"
max_episode="${expert_data_num:-${INTERNW0_MAX_EPISODE:-100000000}}"

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[InternW0_delta] ERROR: the official post-training recipe supports bench_name=RoboDojo, got '${bench_name}'." >&2
    exit 1
fi
if [[ "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
    echo "[InternW0_delta] ERROR: the RoboDojo recipe is 14D arx_x5 joint space, got env_cfg_type='${env_cfg_type}' action_type='${action_type}'." >&2
    exit 1
fi
if [[ ! -f "${UPSTREAM_ROOT}/tools/text_cache.py" ]]; then
    echo "[InternW0_delta] ERROR: full InternW0-Delta source not found at ${UPSTREAM_ROOT}." >&2
    echo "[InternW0_delta] Run sync_upstream.sh, or set INTERNW0_DELTA_ROOT." >&2
    exit 1
fi

dataset_ready() {
    [[ -f "$1/meta/info.json" && -f "$1/meta/tasks.jsonl" && -f "$1/meta/episodes.jsonl" && -d "$1/data" && -d "$1/videos" ]]
}

if dataset_ready "${dataset_dir}"; then
    echo "[InternW0_delta] existing LeRobot v2.1 dataset found: ${dataset_dir}"
else
    if ! "${convert_python}" -c "import lerobot.datasets.lerobot_dataset" 2>/dev/null; then
        echo "[InternW0_delta] ERROR: '${convert_python}' cannot import lerobot.datasets.lerobot_dataset." >&2
        echo "[InternW0_delta] Point INTERNW0_CONVERT_PYTHON at a python with a LeRobot v2.1-era release (dataset codebase v2.1)." >&2
        exit 1
    fi
    IFS=',' read -r -a task_arr <<< "${raw_task_dirs}"
    patterns=()
    for task in "${task_arr[@]}"; do
        patterns+=("${bench_name}.${task}.${env_cfg_type}")
    done

    dataset_parent="$(dirname "${dataset_dir}")"
    mkdir -p "${dataset_parent}"
    echo "[InternW0_delta] converting XPolicyLab HDF5 -> LeRobot v2.1"
    echo "[InternW0_delta] patterns=${patterns[*]} max_episode=${max_episode}"
    echo "[InternW0_delta] output=${dataset_dir}"
    HF_LEROBOT_HOME="${dataset_parent}" "${convert_python}" "${XPL_ROOT}/scripts/transform_lerobot_v21_format.py" \
        "${patterns[@]}" \
        --repo_id "$(basename "${dataset_dir}")" \
        --max_episode "${max_episode}" \
        --resolution 480x640
    if ! dataset_ready "${dataset_dir}"; then
        echo "[InternW0_delta] ERROR: conversion did not produce a complete LeRobot v2.1 dataset at ${dataset_dir}." >&2
        exit 1
    fi
fi

if [[ "${INTERNW0_SKIP_TEXT_CACHE:-0}" == "1" ]]; then
    echo "[InternW0_delta] skipping text cache (INTERNW0_SKIP_TEXT_CACHE=1)"
    exit 0
fi

wan_dir="${WAM_WAN22_PATH:-${ASSET_ROOT}/Wan-AI/Wan2.2-TI2V-5B}"
# The training dataset looks up caches named *.wan22ti2v5b.pt, which text_cache.py
# derives from the basename of model_id.
if [[ "$(basename "${wan_dir}")" != "Wan2.2-TI2V-5B" ]]; then
    echo "[InternW0_delta] ERROR: the Wan2.2 directory must be named Wan2.2-TI2V-5B, got ${wan_dir}." >&2
    exit 1
fi
for required in "${wan_dir}/models_t5_umt5-xxl-enc-bf16.pth" "${wan_dir}/google/umt5-xxl"; do
    if [[ ! -e "${required}" ]]; then
        echo "[InternW0_delta] ERROR: missing ${required}; run download_assets.sh first." >&2
        exit 1
    fi
done

echo "[InternW0_delta] building the official RoboDojo text cache"
(
    cd "${UPSTREAM_ROOT}"
    export PYTHONPATH="${UPSTREAM_ROOT}/src:${PYTHONPATH:-}"
    export ROBODOJO_DATA_ROOT="${dataset_dir}"
    export WAM_CACHE_ROOT="${WAM_CACHE_ROOT:-${POLICY_DIR}/.cache/internw0}"
    export DIFFSYNTH_SKIP_DOWNLOAD=true
    python tools/text_cache.py task=robodojo \
        "model.model_id=${wan_dir}" \
        "model.tokenizer_model_id=${wan_dir}" \
        model.redirect_common_files=false
)

echo "[InternW0_delta] done. Train with:"
echo "  bash ${POLICY_DIR}/train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} <seed> <gpu_id>"
