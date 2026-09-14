#!/usr/bin/env bash
set -euo pipefail
# Standard identifiers describe the native dataset contract; no HDF5 conversion is implied.
bench_name=${1}; ckpt_name=${2}; env_cfg_type=${3}; action_type=${4}
[[ "${env_cfg_type}" == tianji_umi || "${env_cfg_type}" == tianji_dual ]] && [[ "${action_type}" == ee ]] || { echo 'UMI_DP requires tianji_umi/tianji_dual and ee' >&2; exit 2; }
: "${UMI_DP_DATASET_DIR:?Set UMI_DP_DATASET_DIR to a native LeRobot v3 export}"
: "${UMI_DP_IMAGE_CACHE:?Set UMI_DP_IMAGE_CACHE to a writable decoded-video cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMI_PYTHON="${UMI_DP_PYTHON:-${XPL_ROOT}/../envs/umi_dp/.venv/bin/python}"
exec "${UMI_PYTHON}" "${SCRIPT_DIR}/build_video_cache.py" "${UMI_DP_DATASET_DIR}" "${UMI_DP_IMAGE_CACHE}" \
  --sources observation.images.left_wrist observation.images.right_wrist
