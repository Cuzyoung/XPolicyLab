# PatchWAM-Lite

**Contributor:** QZWang / PatchWAM Team | **Paper:** *An Action Is Worth One Patch: Unified World--Action Modeling with PatchWAM* (technical report forthcoming) | **arXiv:** Not yet public | **Original code:** included under `PatchWAM/`

PatchWAM-Lite represents an action chunk as latent patches and denoises image and action patches in one FLUX.2 stream. This adapter releases the training and inference code for the RoboTwin 2.0 clean-to-randomized (C2R) checkpoint at step 140,000. The run initializes from FLUX.2 Klein base 4B; it does not use an Action-as-Patch pretrained checkpoint.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard/).

## Installation

```bash
cd XPolicyLab/policy/PatchWAM
bash install.sh patchwam
conda activate patchwam
bash download_base_models.sh
```

Export the two paths printed by `download_base_models.sh`. Qwen3-4B is loaded from `Qwen/Qwen3-4B` by default and can be replaced with `PATCHWAM_QWEN3_MODEL`.

## Model Assets

```bash
bash download_checkpoint.sh
```

This downloads `step_140000.pt` from [QZWang/patchwam-robotwin-c2r-step140000](https://huggingface.co/QZWang/patchwam-robotwin-c2r-step140000) and verifies SHA256 `22f556d02903a06f23f21a8a28ef91ff92611de0bad4efbf8827c01bd3701fd9`. Normalization statistics and the locked non-idle ranges are under `assets/`.

## Data Processing

The published run uses the FastWAM-compatible RoboTwin 2.0 HDF5 export, not the XPolicyLab LeRobot converters. Its keys are `cam_high`, `cam_left_wrist`, `cam_right_wrist`, and one 14-D dual-arm joint/state vector. `process_data.sh` validates and links that existing export; it does not transcode images.

```bash
export PATCHWAM_DATASET_DIR=/path/to/robotwin2.0
bash process_data.sh RoboTwin C2R arx_x5 joint
```

The C2R selection is locked in `assets/c2r_dr4_resolved.yaml`: `periodic_prefix(period=550, keep_first=50)` followed by `assets/nonidle_ranges.json`.

## Training

Exact reproduction uses 2 nodes × 8 GPUs, per-GPU batch 16, gradient accumulation 1, global batch 256, cosine LR `1e-4`, weight decay `1e-2`, bf16, seed 42, and a 150,000-step cap. The reported checkpoint is step 140,000.

Run the same command on both nodes with the appropriate `NODE_RANK`; set `MASTER_ADDR` to node 0:

```bash
export PATCHWAM_DATASET_DIR=/path/to/robotwin2.0
export PATCHWAM_FLUX2_MODEL=/path/to/flux-2-klein-base-4b.safetensors
export PATCHWAM_AE_MODEL=/path/to/ae.safetensors
export MASTER_ADDR=<node0-address> NNODES=2 NPROC_PER_NODE=8 NODE_RANK=0
bash train.sh RoboTwin C2R arx_x5 joint 42 0,1,2,3,4,5,6,7
```

## Evaluation

PatchWAM-Lite supports RoboTwin dual-arm `joint` control with `arx_x5` and `aloha_agilex` (both use 6+1 dimensions per arm). The published protocol is action horizon 16, replan 16, 10 denoising steps, unseen instructions, 50 tasks, clean and randomized conditions, and 100 episodes per task per condition.

```bash
export PATCHWAM_FLUX2_MODEL=/path/to/flux-2-klein-base-4b.safetensors
export PATCHWAM_AE_MODEL=/path/to/ae.safetensors
bash eval.sh RoboTwin adjust_bottle PatchWAM-C2R-step140000 arx_x5 joint 0 \
  0 0 patchwam RoboTwin
```

For protocol-only validation without loading model assets:

```bash
PATCHWAM_ALLOW_DUMMY_POLICY=true EVAL_ENV_TYPE=debug \
  bash eval.sh RoboTwin adjust_bottle PatchWAM-C2R-step140000 arx_x5 joint 0 \
  0 0 patchwam base
```

## Reported C2R Result

| Clean | Randomized | Average | Rollouts |
|---:|---:|---:|---:|
| 91.56 | 66.72 | **79.14** | 50 tasks × 2 conditions × 100 episodes = 10,000 |

The complete task-level result is `results/c2rdr4_140k_100ep_full.json`; verify it with:

```bash
python verify_results.py
```

## Notes

- Simulator evaluation requires a separate RoboTwin 2.0 environment and assets.
- The checkpoint depends on the separately licensed FLUX.2 Klein base 4B, FLUX.2 autoencoder, Qwen3-4B, and RoboTwin 2.0 assets.
- The bundled PatchWAM/ImageWAM code is under `PatchWAM/LICENSE`; vendored FLUX.2 code and model assets retain their own licenses under `PatchWAM/vendor/flux2/`.
- `allow_dummy_policy` is only for wiring checks and does not produce benchmark results.
