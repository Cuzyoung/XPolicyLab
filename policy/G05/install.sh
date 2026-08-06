#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 2
fi

echo "[G05 install] python=${PYTHON_BIN}"
echo "[G05 install] xpolicylab=${XPL_ROOT}"

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"

cat <<'EOF'

G05 adapter-side XPolicyLab dependencies are installed.

Before running evaluation, set:

  export G05_ROOT=/path/to/G05_checkout
  export G05_PYTHON=/path/to/python
  export G05_CKPT_PATH=/path/to/extracted/g05/checkpoint

G05_ROOT/G05_PYTHON must point to a G05 runtime environment that can import the
G05 model code and its dependencies.
EOF
