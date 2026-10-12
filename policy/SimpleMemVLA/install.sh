#!/bin/bash
set -euo pipefail

# Run inside a fresh Python 3.10 environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy>=1.26,<2" "opencv-python-headless>=4.9,<4.12"
python -m pip install -e "${XPL_ROOT}"
python -m pip install "transformers==5.13.1" "tokenizers>=0.22,<0.23" "accelerate>=1.7" \
    "safetensors>=0.6" "pillow>=10" "einops>=0.8" "modelscope>=1.30"

WHEEL_TAG="cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64"
python -m pip install --no-deps --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+${WHEEL_TAG}.whl" \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+${WHEEL_TAG}.whl"
python -m pip install --no-deps "flash-linear-attention==0.2.2"
