#!/bin/bash
set -euo pipefail

# Fetch the full official InternW0-Delta source used by process_data.sh/train.sh,
# pinned to the commit these wrappers were checked against. The checkout is
# local-only and ignored by Git.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_ROOT="${INTERNW0_DELTA_ROOT:-${SCRIPT_DIR}/InternW0-Delta}"
UPSTREAM_URL="${INTERNW0_DELTA_URL:-https://github.com/InternRobotics/InternW0-Delta.git}"
UPSTREAM_REF="${INTERNW0_DELTA_REF:-90801baa3bdc7829c1e4989edfe3d0fb2410ea6a}"

if [[ -d "${UPSTREAM_ROOT}/.git" ]]; then
    echo "[InternW0_delta] fetching into ${UPSTREAM_ROOT}"
    git -C "${UPSTREAM_ROOT}" fetch origin
else
    if [[ -e "${UPSTREAM_ROOT}" ]]; then
        echo "[InternW0_delta] ERROR: ${UPSTREAM_ROOT} exists but is not a git checkout." >&2
        exit 1
    fi
    echo "[InternW0_delta] cloning ${UPSTREAM_URL} -> ${UPSTREAM_ROOT}"
    git clone "${UPSTREAM_URL}" "${UPSTREAM_ROOT}"
fi
git -C "${UPSTREAM_ROOT}" checkout --detach "${UPSTREAM_REF}"

echo "[InternW0_delta] upstream source ready: ${UPSTREAM_ROOT} @ ${UPSTREAM_REF}"
