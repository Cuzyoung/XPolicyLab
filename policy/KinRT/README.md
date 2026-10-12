# KinRT

**Contributor:** Tianhang Yang, Yanze Zheng, Junjie Wang, Wei-Bin Kou, Ruotong Li, Yujiu Yang | **Paper:** Route by Kinematics, Act by Observation | **arXiv:** https://arxiv.org/abs/2607.26807 | **Original code:** https://github.com/gleeacast/KinRT

This adapter serves the delivered Full35 KinRT checkpoint on RoboDojo: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`. The default run uses `kinrt_full_robodojo` and normalization key `RoboDojo_lerobot_v30_video`. KinRT source stays in a sibling checkout; this directory is only the XPolicyLab integration.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, CUDA 12, [uv](https://docs.astral.sh/uv/), and a GPU with at least 24 GB VRAM. Sibling layout:

```text
<workspace>/
  RoboDojo/XPolicyLab/policy/KinRT/
  KinRT_RoboDojo/policy/pi05/
  LeRobot_KinRT_Full35/
```

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
export KINRT_OPENPI_ROOT=<workspace>/KinRT_RoboDojo/policy/pi05
bash install.sh "$KINRT_OPENPI_ROOT"
```

The installer pins KinRT to `590d52802cde804cdc2d0ccb672c1a3a90d76f91` and LeRobot to `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`, then applies the documented v3 task-table compatibility fix. `KINRT_PYPI_MIRROR` may be `pypi` (default), `tencent`, or `original`. A successful install checks adapter / LeRobot / OpenCV imports; it does not load weights.

## Data Processing

Training uses **LeRobot v3.0**. Official Full35 inference only needs the checkpoint and its packaged `norm_stats.json`. Retraining needs the original complete dataset under `${HF_LEROBOT_HOME}/RoboDojo_lerobot_v30_video` (data, videos, episode metadata). The four router classes are per-frame kinematic labels, not the 35 task instructions.

```bash
bash download_checkpoint.sh --assets-only
"$KINRT_OPENPI_ROOT/.venv/bin/python" full35_assets.py prepare-training \
  --artifact-root "$PWD/checkpoints/KinRT-RoboDojo-Full35-60k" \
  --dataset-root "$HF_LEROBOT_HOME/RoboDojo_lerobot_v30_video" \
  --openpi-root "$KINRT_OPENPI_ROOT"
```

For a new custom conversion from RoboDojo HDF5 (official converter keys, one dataset ID of your own):

```bash
export KINRT_ROBODOJO_REPO_ID=RoboDojo-KinRT-custom-arx_x5-joint
export KINRT_LEROBOT_METADATA_FPS=25
bash process_data.sh RoboDojo multitask arx_x5 joint stack_bowls,insert_key,hang_mugs
bash generate_router_labels.sh
bash compute_norm_stats.sh kinrt_full_robodojo
```

Do not attach published Full35 router labels to a reconverted or reordered dataset.

## Training

```bash
export OPENPI_BASE_CHECKPOINT=<path-to-pi05-base-params>
export OPENPI_TRAIN_CONFIG_NAME=kinrt_full_robodojo
export KINRT_ROBODOJO_REPO_ID=RoboDojo_lerobot_v30_video
bash train.sh RoboDojo full35_full_k4_b256_s0_60k arx_x5 joint 0 0,1,2,3,4,5,6,7
```

The wrapper writes `checkpoints/RoboDojo-full35_full_k4_b256_s0_60k-arx_x5-joint-0/<step>/`. This is not a claim that the command reproduces the released 60k run.

## Evaluation

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
bash download_checkpoint.sh
export KINRT_OPENPI_ROOT=<workspace>/KinRT_RoboDojo/policy/pi05
export KINRT_CHECKPOINT_PATH="$PWD/checkpoints/KinRT-RoboDojo-Full35-60k/checkpoints/60000"

EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls full35_full_k4_b256_s0_60k arx_x5 joint 0 \
  0 0 "$KINRT_OPENPI_ROOT" <robodojo_conda_env>
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls full35_full_k4_b256_s0_60k arx_x5 joint 0 \
  0 0 "$KINRT_OPENPI_ROOT" <robodojo_conda_env>
```

Omit `EVAL_ENV_TYPE=debug` for simulator evaluation. Pass the OpenPI root as the policy-env argument (arg 9), or `uv` to use `policy_uv_env_path` in `deploy.yml`. Optional GPU smoke without a server:

```bash
PYTHONPATH=<workspace>/RoboDojo:"$KINRT_OPENPI_ROOT/src" \
  "$KINRT_OPENPI_ROOT/.venv/bin/python" offline_smoke.py \
  --checkpoint-root "$KINRT_CHECKPOINT_PATH"
```

Reported seed-0 simulator subset: `stack_bowls` 22/25, `stack_bowls_random` 4/25. These are two configurations of one task, not a Full35 average.

## Model Assets

Pinned public checkpoint: [Gleez/kinrt-robodojo-full35-a800-60k](https://huggingface.co/Gleez/kinrt-robodojo-full35-a800-60k), revision `9460d07a9c7677ef3c72ece08df1f34eba7e45c7`.

```bash
bash download_checkpoint.sh [DESTINATION] [--assets-only | --include-training-state]
```

Default destination: `checkpoints/KinRT-RoboDojo-Full35-60k/`. A normal download has the 60k inference weights and delivery files, not the 50k step or optimizer state.

## Configuration

| Key / variable | Purpose |
| --- | --- |
| `checkpoint_path` / `KINRT_CHECKPOINT_PATH` | Inference directory; default `checkpoints/KinRT-RoboDojo-Full35-60k/checkpoints/60000` |
| `checkpoint_num` / `KINRT_CHECKPOINT_NUM` | Preferred step; default 60000 |
| `train_config_name` / `KINRT_TRAIN_CONFIG_NAME` | `kinrt_full_robodojo` |
| `repo_id` / `KINRT_REPO_ID` | Normalization key `RoboDojo_lerobot_v30_video` |
| `action_chunk_size` / `KINRT_ACTION_CHUNK_SIZE` | Actions executed per call; default 50 |
| `policy_uv_env_path` | Used when eval arg 9 is `uv` |
| `KINRT_OPENPI_ROOT` | KinRT `policy/pi05` checkout |

Images stay RGB. The server decodes runtime cameras before `model.py`. Camera map: `cam_head` → `cam_high`.

## Limitations

- Supported: RoboDojo, `arx_x5`, joint. Not validated on other robots, end-effector control, or real hardware.
- The original training dataset is not in the model delivery. Published router labels require that dataset's original frame order.
- The installer reconstructs a documented DataFrame compatibility fix; it is not the original training-time patch.
- Batch inference runs environments sequentially. Simulator success has only been measured for the seed-0 stack-bowls pair above.
- The spelling task is missing from the current RoboDojo simulator inventory (34/35 training tasks).
