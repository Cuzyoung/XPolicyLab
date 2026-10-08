# Abot_M0

**Contributor:** RoboDojo Team | **Paper:** ABot-M0 technical report / model release | **arXiv:** TBD | **Original code:** https://github.com/amap-cvlab/ABot-Manipulation

`Abot_M0` adapts the ABot-M0 manipulation model (built on Qwen3-VL pretrained weights) to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `abot_m0/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` before first use — it covers setup that `install.sh` cannot fully express (external checkpoints, system packages, manual fallback steps, multi-environment runtime notes):

```bash
cd XPolicyLab/policy/Abot_M0
bash install.sh
conda activate <policy_env>  # e.g. ABot (install.sh env name; override with ABOT_CONDA_ENV)
```

## Data Processing

No top-level `process_data.sh`. Training consumes a **LeRobot v2.1** dataset resolved via `ABOT_DATA_ROOT` / `ABOT_DATASET_REPO` / `ABOT_DATA_MIX` (see Configuration), defaulting to `RoboDojo_sim_v21_video_abot`.

Its columns are the standard keys of `XPolicyLab/scripts/transform_lerobot_v21_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)), but the `_abot` suffix marks two things that converter does not produce. ABot reads through a GR00T-style `meta/modality.json`: its `video` branch maps `cam_high` / `cam_left_wrist` / `cam_right_wrist` onto the official `observation.images.*` keys, and its `state` / `action` branches must already carry `left_joints`, `right_joints`, `left_gripper`, `right_gripper` slices. `abot_m0/scripts/prepare_lerobot_for_abot.py --dataset-dir <root>` writes the video and annotation mappings onto an existing `modality.json`, touching `meta/` only and never data or videos, and fails if those state/action slices are missing — run it on your own export via `ABOT_PREPARE_SCRIPT`, which defaults to empty because the published `_abot` dataset is already prepared. The published export is also AV1-encoded, hence the `ABOT_VIDEO_BACKEND=torchvision_av` default; decord cannot decode it. `RoboDojo_lerobot_v21_video` is the unprepared fallback export. For anything beyond this, follow the upstream README under `abot_m0/`.

## Training

```bash
cd XPolicyLab/policy/Abot_M0
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id such as 0,1,2,3 for multi-GPU; process count is inferred)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

`train.sh` forwards to `abot_m0/train.sh`, which writes checkpoints to `abot_m0/checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; during evaluation `ckpt_name` may be the full run-directory name or the short run name (resolved under both `checkpoints/` and `abot_m0/checkpoints/`), or set an explicit `checkpoint_path` in `deploy.yml`. Pretrained assets (the Qwen3-VL base VLM and the ABot-M0 pretrain checkpoint) are resolved via `ABOT_MODEL_ROOT` / `ABOT_BASE_VLM` / `ABOT_PRETRAIN_CKPT`.

## Evaluation

```bash
cd XPolicyLab/policy/Abot_M0
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation:

| Key | Notes |
|---|---|
| `checkpoint_num` | Step file suffix, for example `150000` resolves to `steps_150000_pytorch_model.pt`. |
| `ckpt_name` | Short run name or full run directory name under `policy/Abot_M0/checkpoints/` or `policy/Abot_M0/abot_m0/checkpoints/`. |
| `checkpoint_path` | Optional explicit checkpoint file or run directory; takes precedence over `ckpt_name`. |
| `unnorm_key` | Normalization-stat key, default `robodojo_sim`; also used when generating missing `dataset_statistics.json`. |
| `device` | Torch device for the policy model, for example `cuda`, `cuda:0`, `cpu`, or `auto`. |

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `ABOT_CONDA_ENV` | Conda env used by `install.sh`; defaults to `ABot`. |
| `ABOT_DATA_ROOT` | LeRobot data root for training; defaults to `HF_LEROBOT_HOME` or `~/.cache/huggingface/lerobot`. |
| `ABOT_DATASET_REPO` | LeRobot repo directory; defaults to `RoboDojo_sim_v21_video_abot`. |
| `ABOT_DATA_MIX` | ABot dataset mixture key; defaults to `robodojo_sim`. |
| `ABOT_MODEL_ROOT` | Root containing Qwen and ABot pretrained weights; defaults to `abot_m0/model_weights`. |
| `ABOT_BASE_VLM` | Overrides the Qwen3-VL-4B-Instruct-Action path. |
| `ABOT_PRETRAIN_CKPT` | Overrides the ABot-M0 pretrain checkpoint. |
| `ABOT_RELOAD_MODULES` | Modules reloaded from pretrain; defaults to `qwen_vl_interface`. |
| `ABOT_PREPARE_SCRIPT` | Optional data-preparation hook for training; leave empty when data is already prepared to avoid overwriting multi-task instructions. |
| `ABOT_STATS_JSON` | Optional stats fallback for eval; defaults to `abot_m0/checkpoints/stats_gr00t.json` inside the adapter. |
| `ABOT_UNNORM_KEY` | Overrides the normalization-stat key used at eval time. |
