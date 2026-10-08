#!/bin/bash
set -euo pipefail

# The published C2R run consumes the FastWAM-compatible RoboTwin 2.0 HDF5
# export directly. This command validates and links that immutable input.
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}

if [[ "${bench_name}" != "RoboTwin" || "${action_type}" != "joint" ]]; then
    echo "PatchWAM C2R supports bench_name=RoboTwin and action_type=joint" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${PATCHWAM_DATASET_DIR:?Set PATCHWAM_DATASET_DIR to the RoboTwin 2.0 HDF5 export}"
target_dir="${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [[ ! -d "${source_dir}" ]]; then
    echo "Dataset directory does not exist: ${source_dir}" >&2
    exit 1
fi
if ! find "${source_dir}" -type f -name '*.hdf5' -print -quit | grep -q .; then
    echo "No RoboTwin HDF5 episodes found under ${source_dir}" >&2
    exit 1
fi

mkdir -p "$(dirname "${target_dir}")"
ln -sfn "${source_dir}" "${target_dir}"
echo "[PatchWAM] linked ${target_dir} -> ${source_dir}"
