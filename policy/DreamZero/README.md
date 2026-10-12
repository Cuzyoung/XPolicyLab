# DreamZero

**Contributor:** RoboDojo Team | **Paper:** World Action Models are Zero-shot Policies | **arXiv:** https://arxiv.org/abs/2602.15922 | **Original code:** https://github.com/dreamzero0/dreamzero

`DreamZero` adapts the DreamZero world-action model (built on Wan2.1-I2V weights with a umt5-xxl tokenizer) to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `dreamzero/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/DreamZero
bash install.sh
conda activate <policy_env>  # e.g. dreamzero
```

## Data Processing

Converts demos into the dataset consumed by `train.sh` under `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/` (output root override: `DREAMZERO_DATA_DIR`; conversion frame rate: `DREAMZERO_FPS`, default `25`). The only extra argument is the optional `[expert_data_num]` episode limit:

```bash
cd XPolicyLab/policy/DreamZero
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: 50-episode data-scale ablation under a distinct ckpt_name
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50
```

`train.sh` accepts two LeRobot sources, resolved in this order: `LEROBOT_DATA_PATH`, then this script's output, then the shared **LeRobot v3.0** export `RoboDojo_sim_arx-x5_v30`.

That shared export carries the standard keys of `XPolicyLab/scripts/transform_lerobot_v30_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)) and needs no preparation: for a v3.0 root the loader synthesizes `meta/modality.json` itself, mapping the GEAR-style names `top_head` / `hand_left` / `hand_right` onto `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`, and computes its relative-action stats on first load.

The output of `process_data.sh` deliberately **deviates** from those keys, because it converts trajectory HDF5 into DreamZero's native AgiBot layout: the image columns are named `observation.images.top_head` / `hand_left` / `hand_right`, and `observation.state` / `action` are re-padded to 20 and 22 dims rather than the converter's packed robot vector. It reads trajectory HDF5, not LeRobot, and only it produces that layout — so its output is not interchangeable with the shared export, and a dataset for this path has to come from here rather than from the official converters.

`process_data.py` can also build that AgiBot layout from an existing official v3.0 export instead of from HDF5, which `process_data.sh` does not expose because it always passes `--source_format hdf5`:

```bash
cd XPolicyLab/policy/DreamZero
python process_data.py --bench_name RoboDojo --ckpt_name stack_bowls \
    --env_cfg_type arx_x5 --action_type joint \
    --source_format lerobot_v3 --source_lerobot_path /path/to/RoboDojo_sim_arx-x5_v30
```

This path symlinks the source videos and keeps their official `observation.images.*` column names, re-pads `observation.state` / `action` to the AgiBot 20 and 22 dims, and writes the GEAR `meta/modality.json` that maps `top_head` / `hand_left` / `hand_right` onto the official image keys. The source root is required — it comes from `--source_lerobot_path` or `LEROBOT_DATA_PATH`, and there is deliberately no default, since a converter that guessed its input would silently produce data for the wrong task.

## Training

```bash
cd XPolicyLab/policy/DreamZero
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id such as 0,1,2,3 for multi-GPU; torchrun process count is inferred)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. Training data is resolved in this order: `LEROBOT_DATA_PATH` (explicit override) → `data/<4-tuple>/` from `process_data.sh` → the shared default `<demo_root>/RobotDojo/RoboDojo_sim_arx-x5_v30`. Pretrained weights must be available under `checkpoints/` (DreamZero-AgiBot, Wan2.1-I2V-14B-480P, umt5-xxl) or pointed to via the variables below.

## Evaluation

```bash
cd XPolicyLab/policy/DreamZero
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `action_dim`, `model_path`, `pretrained_model_path`, `tokenizer_path`, `action_horizon`, `video_history`, `ctrl_freq`, `prompt`, `inference_method`, `skip_img_transform`, `native_dojo_action`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `LEROBOT_DATA_PATH` | Explicit LeRobot dataset root; highest-priority training data source. |
| `DREAMZERO_DATA_DIR` | `process_data.sh` output root; defaults to the policy `data/` directory. |
| `DREAMZERO_FPS` | Conversion frame rate; default `25`. |
| `DREAMZERO_PRETRAINED_MODEL_PATH` | Defaults to `./checkpoints/DreamZero-AgiBot`, or `./checkpoints` for a flat layout. |
| `WAN_CKPT_DIR` | Defaults to `./checkpoints/Wan2.1-I2V-14B-480P`. |
| `TOKENIZER_DIR` | Defaults to `./checkpoints/umt5-xxl`, with a Wan2.1 nested tokenizer fallback. |
| `DREAMZERO_NUM_GPUS` | Overrides the GPU count inferred from comma-separated `gpu_id`. |
| `DREAMZERO_PREFLIGHT_ONLY` | If `1`, validate dataset and weights then exit. |
| `DREAMZERO_DRY_RUN` | If `1`, print the resolved command and exit before `torchrun`. |
