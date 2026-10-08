#!/bin/bash
set -euo pipefail

ENV_NAME=${1:-openloopvla}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/OpenLoopVLA/requirements.txt"
python -m pip install -e "${SCRIPT_DIR}/OpenLoopVLA"
python -m pip install -e "${XPL_ROOT}"

