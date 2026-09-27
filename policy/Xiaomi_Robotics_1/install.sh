#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${MIBOT_CONDA_ENV:-mibot}"

echo "[Xiaomi_Robotics_1] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"
if [[ -n "${XR1_ENV_DIR:-}" ]]; then
    command -v uv >/dev/null
    if [[ ! -x "${XR1_ENV_DIR}/bin/python" ]]; then
        uv venv --python 3.12 "${XR1_ENV_DIR}"
    fi
    install=(uv pip install --python "${XR1_ENV_DIR}/bin/python")
    echo "[Xiaomi_Robotics_1] XR1_ENV_DIR=${XR1_ENV_DIR}"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "Install conda, or set XR1_ENV_DIR and provide uv." >&2
        exit 1
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        conda create -n "${CONDA_ENV}" python=3.12 -y
    fi
    conda activate "${CONDA_ENV}"
    install=(python -m pip install)
    echo "[Xiaomi_Robotics_1] CONDA_ENV=${CONDA_ENV}"
fi

# Core dependencies
"${install[@]}" pip setuptools wheel packaging ninja
"${install[@]}" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"${install[@]}" transformers==4.57.1 scipy numpy Pillow

# Vendored xr1 requirements. mmengine (model registry) and liger-kernel (fused
# RMSNorm/RoPE in the VLM) are needed just to import mibot, so inference needs
# them too; the rest are pulled in by the training entrypoints.
"${install[@]}" -r "${POLICY_DIR}/xiaomi_robotics_1/xr1/assets/requirements.txt"
"${install[@]}" flash-attn==2.8.3 --no-build-isolation

# HDF5 -> JSON/MP4 conversion (process_data.sh). Not part of the vendored
# requirements, which cover training and inference only.
"${install[@]}" opencv-python-headless h5py imageio imageio-ffmpeg tqdm
"${install[@]}" -e "${XPOLICYLAB_ROOT}"

echo "[Xiaomi_Robotics_1] Installation finished."
if [[ -n "${XR1_ENV_DIR:-}" ]]; then
    echo "[Xiaomi_Robotics_1] Activate env: source ${XR1_ENV_DIR}/bin/activate"
else
    echo "[Xiaomi_Robotics_1] Activate env: conda activate ${CONDA_ENV}"
fi
