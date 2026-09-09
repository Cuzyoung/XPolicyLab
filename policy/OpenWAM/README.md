# OpenWAM

## ManiMux YAM integration

The local extension adds `observation_profile: yam_base` with `env_cfg_type:
yam_dual`, absolute per-arm base poses and XPolicy WebSocket inference. It
does not use the ARX simulation calibration. `train.sh` and `process_data.sh`
now use the active Python (`OPENWAM_PYTHON` optionally selects an existing
interpreter); no environment is created. Training accepts the standard six
arguments plus Hydra overrides, with `OPENWAM_DATASET_DIR`,
`OPENWAM_CHECKPOINT_DIR`, `OPENWAM_FINETUNE_CKPT_PATH` and
`OPENWAM_RESUME_CKPT_PATH`. Use `--dry-run` on `train.sh` to inspect the command.
Native YAM HDF5 uses future achieved EE states as targets; it is not raw
ManiMux recording format or XR1 JSON. The parent repository's
`docs/openwam-yam-runbook.md` documents conversion, cluster launch, checkpoint
checks and hardware-free probes. For YAM, task evaluation runs in ManiMux.
The upstream RoboDojo simulation instructions below remain a separate profile.

**Contributor:** OpenWAM Contributors | **Paper:** An Open, Modular Exploration Towards Systematic World–Action Model Pretraining | **arXiv:** [2609.07398](https://arxiv.org/abs/2609.07398) | **Original code:** https://github.com/OpenWAM-Official/OpenWAM

`OpenWAM` adapts the OpenWAM world-action model to XPolicyLab/RoboDojo (`arx_x5`, absolute EE control, batched inference). Integration scripts live at this directory level; the vendored upstream implementation lives in `OpenWAM/`. Official OpenWAM does not expose batch inference; the vendored tree adds `generate_batch` for `eval_batch: true`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use an existing compatible Python environment, then install the PyTorch CUDA wheel before the adapter packages:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

cd XPolicyLab/policy/OpenWAM
bash install.sh
```

`install.sh` editable-installs XPolicyLab and the vendored `OpenWAM/` package. Cosmos-Predict2.5 extras are not required for the RoboDojo Wan checkpoint.

## Data Processing

OpenWAM consumes native RoboDojo HDF5 (`<dataset_dir>/<task>/<embodiment>/data/episode_*.hdf5`). There is no LeRobot conversion.

```bash
cd XPolicyLab/policy/OpenWAM
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example
export OPENWAM_DATASET_DIR=/path/to/robodojo_data
bash process_data.sh RoboDojo cotrain arx_x5 ee
```

Set `OPENWAM_DATASET_DIR` to prepared native data. The script validates the
reader, builds statistics and inspects a training sample. It does not download
or silently treat an empty directory as prepared data.

## Model Assets

Released RoboDojo SFT checkpoints are self-contained (`config.yaml`, `checkpoint_step_*.safetensors`, `normalization_stats.npy`). Download from [Hugging Face OpenWAM](https://huggingface.co/OpenWAM) or with the official helper:

```bash
cd XPolicyLab/policy/OpenWAM/OpenWAM
python scripts/download_assets/download_openwam_checkpoints.py
```

Place or symlink the checkpoint directory where eval can resolve it:

```bash
cd XPolicyLab/policy/OpenWAM
mkdir -p checkpoints
ln -sfn /path/to/New_OpenWAM_RoboDojo_SFT_60k checkpoints/New_OpenWAM_RoboDojo_SFT_60k
```

Training from scratch also needs the Wan video backbone (`python scripts/download_assets/download_video_backbone.py` inside `OpenWAM/`). Fine-tuning a released checkpoint uses `OPENWAM_FINETUNE_CKPT_PATH`.

## Training

```bash
cd XPolicyLab/policy/OpenWAM
export OPENWAM_DATASET_DIR=/path/to/robodojo_data

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: RoboDojo SFT on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 ee 0 0
```

This invokes the vendored trainer through torch distributed run with
`dataloader=robodojo`. Checkpoints land in the standard five-part run directory
unless `OPENWAM_CHECKPOINT_DIR` is set. Extra Hydra overrides follow the six
positional arguments; data/frame/checkpoint contract overrides are rejected.

## Evaluation

```bash
cd XPolicyLab/policy/OpenWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline wiring check (no simulator, no checkpoint load)
EVAL_ENV_TYPE=debug OPENWAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls dummy arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a released checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls New_OpenWAM_RoboDojo_SFT_60k arx_x5 ee 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. `ckpt_name` may be the short directory name, the full 5-tuple run directory, or an absolute path. Override with `OPENWAM_CKPT_DIR`. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `eval_batch`, `action_type`, `device`, `ckpt_dir`, `openwam_root`, `openwam_deploy_config`, `replan_steps`, `allow_dummy_policy`.

| Variable | Notes |
|---|---|
| `OPENWAM_DATASET_DIR` | Native RoboDojo HDF5 root for `process_data.sh` / `train.sh`. |
| `OPENWAM_CKPT_DIR` | Explicit eval checkpoint directory (`config.yaml` + safetensors). |
| `OPENWAM_ROOT` | Optional OpenWAM source override; defaults to the vendored `OpenWAM/`. |
| `OPENWAM_DEPLOY_CONFIG` | Optional deploy yaml override; defaults to `OpenWAM/configs/deploy.yaml`. |
| `OPENWAM_TRAIN_OVERRIDES` | Extra Hydra overrides forwarded to official `scripts/train.sh`. |
| `OPENWAM_FINETUNE_CKPT_PATH` | Warm-start directory for official `training.finetune_ckpt_path`. |
| `OPENWAM_RESUME_CKPT_PATH` | Resume directory for official `training.resume_ckpt_path`. |
| `OPENWAM_ALLOW_DUMMY_POLICY` | Debug-only: skip checkpoint load and return hold-position chunks. |

`eval_batch: true` stacks every running env into one `engine.generate_batch` forward. `model.py` re-forces `dit_cache` / `compile` / `decode_video` off and `inference_mode=sync`. `device: cuda` loads Wan, UMT5-XXL, and ActionDiT on GPU, matching official OpenWAM deploy.
