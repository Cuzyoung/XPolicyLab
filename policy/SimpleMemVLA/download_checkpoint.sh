#!/bin/bash
set -euo pipefail

# Usage: bash download_checkpoint.sh [ckpt_name]   (default: SimpleMemVLA-RoboDojo-Sim)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="keithyc/SimpleMemVLA-RoboDojo-Sim"
MODEL_SHA256="2381a20803be60bf3817155cf5383a53f7ca95e4b6ce4abfc41f22db7bbee23a"

ckpt_name=${1:-SimpleMemVLA-RoboDojo-Sim}
target_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"

if ! command -v modelscope >/dev/null 2>&1; then
    echo "[ERROR] modelscope CLI not found; run install.sh or: pip install modelscope" >&2
    exit 1
fi

mkdir -p "${target_dir}"
modelscope download "${REPO_ID}" --local_dir "${target_dir}"

actual_sha256=$(sha256sum "${target_dir}/model.safetensors" | cut -d' ' -f1)
if [[ "${actual_sha256}" != "${MODEL_SHA256}" ]]; then
    echo "[ERROR] model.safetensors sha256 mismatch: ${actual_sha256}" >&2
    exit 1
fi
echo "[DOWNLOAD] checkpoint ready at ${target_dir}"
