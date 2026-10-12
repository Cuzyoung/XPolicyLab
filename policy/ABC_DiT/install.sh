#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
POLICY_ENV="${1:-${POLICY_DIR}/.venv}"
command -v uv >/dev/null || { echo 'Install uv before running install.sh' >&2; exit 1; }
if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
    uv venv --python 3.12 "${POLICY_ENV}"
fi
uv pip install --python "${POLICY_ENV}/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
uv pip install --python "${POLICY_ENV}/bin/python" numpy ftfy regex
uv pip install --python "${POLICY_ENV}/bin/python" -e "${XPL_ROOT}"
echo "ABC_DiT environment ready: ${POLICY_ENV}"
