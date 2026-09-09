#!/usr/bin/env bash
set -euo pipefail
# Standard XPolicy six arguments, followed by native Hydra overrides.
POLICY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${POLICY_DIR}/../../..:${PYTHONPATH:-}"
exec "${OPENWAM_PYTHON:-python}" -m XPolicyLab.policy.OpenWAM.training "$@"
