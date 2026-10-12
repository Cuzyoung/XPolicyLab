#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR=${1:-${SCRIPT_DIR}/base_models}
mkdir -p "${TARGET_DIR}/flux2" "${TARGET_DIR}/ae"

huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
    flux-2-klein-base-4b.safetensors --local-dir "${TARGET_DIR}/flux2"
huggingface-cli download black-forest-labs/FLUX.2-dev \
    ae.safetensors --local-dir "${TARGET_DIR}/ae"

echo "export PATCHWAM_FLUX2_MODEL=${TARGET_DIR}/flux2/flux-2-klein-base-4b.safetensors"
echo "export PATCHWAM_AE_MODEL=${TARGET_DIR}/ae/ae.safetensors"
