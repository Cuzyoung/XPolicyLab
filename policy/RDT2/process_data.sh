#!/bin/bash
set -euo pipefail

# RDT2 data conversion: LeRobot v3.0 -> RDT2 webdataset shards.
#
# STATUS: NOT IMPLEMENTED. This is step 1 of docs/rdt2-umi-runbook.md and the
# converter (`scripts/lerobot_to_rdt2_shards.py`) lives in the *parent* repo,
# not in this submodule. This wrapper fails loudly rather than pretending to
# have converted anything.
#
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Output convention: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>

bench_name=${1:-}
ckpt_name=${2:-}
env_cfg_type=${3:-}
action_type=${4:-}
expert_data_num=${5:-}

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" ]]; then
  echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
out_dir="${POLICY_DIR}/data/${data_setting}"

CONVERTER="${RDT2_SHARD_CONVERTER:-${WORKSPACE_ROOT}/scripts/lerobot_to_rdt2_shards.py}"

if [[ ! -f "${CONVERTER}" ]]; then
  cat >&2 <<MSG
[RDT2][process_data] NOT IMPLEMENTED.

  The LeRobot v3.0 -> RDT2 webdataset converter does not exist yet:
      ${CONVERTER}

  It is step 1 of docs/rdt2-umi-runbook.md and belongs to the parent repo
  (scripts/), not to this submodule. It must handle, at minimum:
    1. arm-order swap  left-first [L(10) R(10)] -> RDT2 right-first [R(10) L(10)]
    2. absolute TCP -> relative chunk anchored at the chunk's first frame
    3. two wrist frames -> 384x384 each -> horizontal concat (384, 768, 3) uint8
    4. gripper: normalized [0,1] stroke fraction -> metres -> RDT2 rescale
    5. T=24 chunking and an episode-tail policy

  Override the path with RDT2_SHARD_CONVERTER=/path/to/converter.py
MSG
  exit 2
fi

mkdir -p "${out_dir}"
echo "[RDT2][process_data] converter: ${CONVERTER}"
echo "[RDT2][process_data] out_dir:   ${out_dir}"
exec python "${CONVERTER}" \
  --output-dir "${out_dir}" \
  --env-cfg-type "${env_cfg_type}" \
  --action-type "${action_type}" \
  ${expert_data_num:+--max-episodes "${expert_data_num}"}
