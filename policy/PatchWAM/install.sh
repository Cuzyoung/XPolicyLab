#!/bin/bash
set -euo pipefail

ENV_NAME=${1:-patchwam}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/PatchWAM"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

pip install -U pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e "${UPSTREAM_DIR}"
pip install -e "${XPL_ROOT}"
