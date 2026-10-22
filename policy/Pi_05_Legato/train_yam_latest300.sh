#!/usr/bin/env bash
set -euo pipefail
resume_requested=${OPENPI_RESUME:-0}
source /inspire/hdd3/project/agent-driven-world-model/ky26300/wam_latest_300ep_20261012/pi05.env
export OPENPI_EXPECTED_FRAMES=170285
export OPENPI_RESUME=${resume_requested}
export OPENPI_TRAIN_CONFIG_NAME=pi05_yam_legato
export WANDB_MODE=disabled WANDB_DISABLED=true
# Same dataset, norm statistics, base checkpoint, optimizer and loader as the
# completed 300-episode baseline. Only continuation model/config differs.
"$YAM_TRAIN_ROOT/envs/pi05/.venv/bin/python" - "/inspire/hdd3/project/agent-driven-world-model/ky26300/wam_latest_300ep_20261012" <<'PY'
from pathlib import Path
import hashlib,json,sys
r=Path(sys.argv[1]);ready=json.loads((r/'PI05_DATA_READY.json').read_text())
assert (ready['episodes'],ready['tasks'])==(300,6)
for path,expected in json.loads((r/'FROZEN_INPUTS.json').read_text())['files'].items():
 with open(path,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,path
print('Frozen source/data/statistics/base-weight hashes verified.',flush=True)
PY
exec bash /inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training/XPolicyLab/policy/Pi_05_Legato/train_cluster.sh train pi05_yam_latest300_legato_b64_30k_20261018
