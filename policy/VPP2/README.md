# VPP2 — RoboDojo Adapter

**Contributor:** [Haodong Yan](https://github.com/Haodong-Yan) | **Paper:** Video Prediction Policy 2: Predict Better, Act Better | **arXiv:** [2610.10270](https://arxiv.org/abs/2610.10270) | **Original code:** [Official VPP2 repository](https://github.com/roboterax/video-prediction-policy-2)

**Project page:** [Video Prediction Policy 2](https://robert-gyj.github.io/video-prediction-policy-2/)

VPP2 combines a pretrained video model with an action expert for robot control.
This adapter ports the ModelScope deployment entry points for the joint + 2B
100k checkpoint to XPolicyLab, supporting **RoboDojo / arx_x5 / absolute EE16**.
The model implementation is installed from a pinned public VPP2 revision in
`upstream/`; full training and evaluation code for **RoboDojo, LIBERO,
LIBERO-OOD and LIBERO-PRO** lives in the [official repository](https://github.com/roboterax/video-prediction-policy-2).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

From a RoboDojo workspace with its `env_cfg/` beside the XPolicyLab checkout:

```bash
cd XPolicyLab/policy/VPP2
bash install.sh vpp2
conda activate vpp2
bash download_checkpoints.sh
python launch_policy.py --dry-run
```

The installer pins the official code at `3365872`, uses Python 3.10 and defaults
to PyTorch 2.11 / CUDA 13.0. Select a compatible driver/build with
`TORCH_VERSION`, `TORCHVISION_VERSION` and `TORCH_CUDA` if necessary.
The reference GPU is one 96 GiB RTX PRO 6000; smaller devices are unverified.
Install the simulator separately using the RoboDojo instructions.

## Data Processing

This is an **evaluation adapter**; `process_data.sh` is intentionally absent.
The public [RoboDojo data guide](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/training.md#required-artifacts)
describes the prepared EE16 parquet data, RGB T-shaped videos and native frame
indices. This custom prepared export is not a generic LeRobot v2.1/v3.0 export
from XPolicyLab's converters. Converted training data is a separate release
item; preparation and validation code is already public in VPP2.

## Training

Training code is already available in the
[official VPP2 repository](https://github.com/roboterax/video-prediction-policy-2).
Follow its [RoboDojo guide](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/robodojo.md)
for the integrated **history-conditioned Video-10k → joint + 2B, 0–100k** recipe.
`train.sh` is intentionally absent from this evaluation adapter.
LIBERO uses its own [training and evaluation configuration](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/libero.md).

## Evaluation

```bash
cd XPolicyLab/policy/VPP2
bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 1 vpp2 RoboDojo

# Wiring checks using real model weights, without a simulator:
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 0 vpp2 vpp2
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 0 vpp2 vpp2
```

Replace `RoboDojo` with your simulator conda environment name. Both environment
names and absolute conda prefixes are accepted. Debug checks verify wiring and
action shapes; they do not measure task success. For separate policy and
simulator machines, use the [shared deployment flow](../../README.md#-deployment-flow).

## Model Assets

The same released bundle is available on public [Hugging Face](https://huggingface.co/Haodong082399/VPP2).
As an alternative to `download_checkpoints.sh`, download from the adapter directory:

```bash
hf download Haodong082399/VPP2 --local-dir . \
  --include 'checkpoints/joint2b_s100000/**' \
  --include 'checkpoints/Wan2.1-I2V-14B-480P/**'
```

Install the `hf` client in a download environment with
`python -m pip install -U huggingface_hub` if needed.

[ModelScope weights](https://modelscope.cn/models/haodong123/VPP2_preview)
currently require an authorized account. Use an existing ModelScope SDK login
or set `MODELSCOPE_API_TOKEN` privately before running the download script.

```text
checkpoints/
├── joint2b_s100000/
│   ├── action.pt
│   ├── video.pt
│   ├── dataset_stats.json
│   └── manifest.json
└── Wan2.1-I2V-14B-480P/
    ├── Wan2.1_VAE.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    └── google/umt5-xxl/...
```

This named bundle is supported by the shared checkpoint resolver, alongside its
conventional run-directory layout and explicit paths. Keep its paired video,
action and normalization files together. `VPP2_BUNDLE` and `VPP2_WAN_ROOT` can
point to existing asset directories; `launch_policy.py --dry-run` checks file
sizes, manifest step and encoder inventory without allocating a GPU.

## Configuration

`deploy.yml` records the released defaults:

| Setting | Value |
| --- | --- |
| Action inference | 10 Euler steps, sigma shift 1, seed 1 |
| Predicted / executed actions | 32 / 24 |
| Video context | History 8, native stride 25, episode anchor |
| RGB input | Native views → T-shaped composition → 240 × 416 |
| Action representation | Absolute poses `[x,y,z,qw,qx,qy,qz]` and gripper, left then right |
| Normalization | Checkpoint z-score statistics, including grippers |
| Precision | `device: cuda`, `mixed_precision: bf16` |

The model runtime performs camera composition and resizing. The adapter accepts
RGB arrays already decoded by XPolicyLab. `deployment_adapter: robodojo_ee16`,
`action_dim: 16`, `action_horizon: 32`, `history_stride: 25` and
`video_seed_offset: 1000003` describe this checkpoint contract.
`checkpoint_path`, `video_checkpoint_path`, `dataset_stats_path` and
`wan_model_dir` default to the bundle above. `default_instruction` is used only
when an observation supplies no instruction.

For diagnostic comparisons, launch scripts accept `VPP2_NUM_INFERENCE_STEPS`,
`VPP2_SIGMA_SHIFT` and `VPP2_REPLAN_STEPS`; these override
`num_inference_steps`, `sigma_shift` and `replan_steps` respectively.
Changing them changes the evaluation setting.

Observation history is recorded once per action and isolated by client ID.
Keep `eval_batch: false`: vectorized environment batch methods are explicitly
unsupported. Multiple independent clients can share one policy server.
