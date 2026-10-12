#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-internw0-delta}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${INTERNW0_DELTA_ROOT:-${SCRIPT_DIR}/InternW0-Delta}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"

# RynnBrain uses FLA/Triton kernels that compile a tiny CUDA driver helper on
# first inference.  Install the compiler in the environment so evaluation does
# not depend on the host image providing gcc.
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y

python -m pip install --upgrade pip
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  "modelscope==1.34.0" \
  "ftfy==6.3.1" \
  "flash-linear-attention==0.5.1" \
  "causal-conv1d==1.6.2.post1"
if [[ -f "${UPSTREAM_ROOT}/pyproject.toml" && -d "${UPSTREAM_ROOT}/src/wam/datasets" ]]; then
  python -m pip install -e "${UPSTREAM_ROOT}[train,modelscope]"
else
  echo "Full InternW0-Delta source not found at ${UPSTREAM_ROOT}; installing eval-only runtime."
  echo "Clone https://github.com/InternRobotics/InternW0-Delta there, or set INTERNW0_DELTA_ROOT before running install.sh to enable train/process_data wrappers."
  python -m pip install -e "${SCRIPT_DIR}/wam_runtime"
fi
python -m pip install -e "${XPL_ROOT}"

echo "Installed policy environment: ${ENV_NAME}"
echo "Next: conda activate ${ENV_NAME} && bash ${SCRIPT_DIR}/download_assets.sh"
