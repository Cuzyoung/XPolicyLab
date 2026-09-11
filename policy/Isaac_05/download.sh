#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
TARGET="${ISAAC05_CHECKPOINT_DIR:-${WORKSPACE_ROOT}/checkpoints/pretrained/perceptron-ai/isaac-0.5}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI not found. Install huggingface_hub first." >&2
  exit 1
fi

mkdir -p "${TARGET}"
hf download PerceptronAI/Isaac-0.5 --local-dir "${TARGET}"
