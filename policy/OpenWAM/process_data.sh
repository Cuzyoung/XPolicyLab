#!/usr/bin/env bash
set -euo pipefail
# Existing native data only: validate, build stats, inspect a training sample.
POLICY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${POLICY_DIR}/../../..:${PYTHONPATH:-}"
exec "${OPENWAM_PYTHON:-python}" -m XPolicyLab.policy.OpenWAM.process_data "$@"
