#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use the SDK login cache or MODELSCOPE_API_TOKEN, never a token in shell args.
python "${SCRIPT_DIR}/download_checkpoints.py"
