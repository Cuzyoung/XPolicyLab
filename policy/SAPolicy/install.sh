#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
POLICY_ENV="${1:-${POLICY_DIR}/.venv}"
command -v uv >/dev/null || { echo 'Install uv before running install.sh' >&2; exit 1; }
if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
    uv venv --python 3.11 "${POLICY_ENV}"
fi
uv pip install --python "${POLICY_ENV}/bin/python" \
    --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'
uv pip install --python "${POLICY_ENV}/bin/python" -r "${POLICY_DIR}/requirements.txt"
uv pip install --python "${POLICY_ENV}/bin/python" --no-deps -e "${XPL_ROOT}"
uv pip check --python "${POLICY_ENV}/bin/python"
echo "SAPolicy environment ready: ${POLICY_ENV}"
