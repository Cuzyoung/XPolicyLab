# LingBot-VLA2 Installation Notes

The adapter uses the pinned official checkout in `lingbot_vla_v2/` and a
standalone Python 3.12 uv environment. From an existing parent workspace:

```bash
git submodule update --init --recursive
cd XPolicyLab/policy/LingBot_VLA2
bash install.sh
```

The default environment is `<workspace>/envs/lingbot-vla2/.venv`. Override it
for installation and every later command with:

```bash
export LINGBOT_VLA2_ENV_DIR=/path/to/lingbot-vla2/.venv
```

The offline XPolicy debug client may use the workspace's existing YAM uv
environment by passing `envs/yam` to `eval.sh`, or by setting
`LINGBOT_VLA2_EVAL_ENV_DIR` and passing `uv`.

`install.sh` installs PyTorch 2.8 CUDA 12.8, FlashAttention 2.8.3, the official
LingBot-VLA2 packages, LeRobot, depth dependencies and XPolicyLab. Set
`FLASH_ATTN_WHEEL=/path/to/wheel.whl` to avoid compiling FlashAttention.

## Model Assets

Inference needs a complete post-training bundle. Training additionally needs:

- `robbyant/lingbot-vla-v2-6b` as `LINGBOT_VLA2_MODEL_PATH`;
- `Qwen/Qwen3-VL-4B-Instruct` as `LINGBOT_VLA2_TOKENIZER_PATH` or `QWEN3_PATH`;
- a converted LeRobot dataset and matching norm stats from `process_data.sh`.

Depth and DINO-Video teacher assets are not required by the adapter's default
`training/yam_dual.yaml`, because its `align_params` is empty. To reproduce the
official native-depth post-training recipe, start from the upstream
`configs/vla/real_robot/real_robot.yaml` and provide every teacher path listed
in the official training guide.

## Hardware

The 6B post-training path is a multi-GPU workload. `train.sh` accepts a
comma-separated GPU list and derives the torchrun world size from
`CUDA_VISIBLE_DEVICES`. Tune `LINGBOT_VLA2_MICRO_BATCH_SIZE` and
`LINGBOT_VLA2_GRAD_ACCUM_STEPS` for the available memory; installation and
inference success do not prove that a single 24 GB GPU can train the model.
