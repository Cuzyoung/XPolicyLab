#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO_ID=${PATCHWAM_HF_REPO:-QZWang/patchwam-robotwin-c2r-step140000}
TARGET_DIR=${1:-${SCRIPT_DIR}/checkpoints/PatchWAM-C2R-step140000}
EXPECTED_SHA256=22f556d02903a06f23f21a8a28ef91ff92611de0bad4efbf8827c01bd3701fd9

mkdir -p "${TARGET_DIR}"
huggingface-cli download "${HF_REPO_ID}" step_140000.pt --local-dir "${TARGET_DIR}"
printf '%s  %s\n' "${EXPECTED_SHA256}" "${TARGET_DIR}/step_140000.pt" | shasum -a 256 -c -
echo "[PatchWAM] checkpoint ready: ${TARGET_DIR}/step_140000.pt"
