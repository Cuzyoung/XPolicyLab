# starVLA

**Contributor:** RoboDojo Team | **Paper:** StarVLA: A Versatile Vision-Language-Action Model with Efficient Training and Policy Adaptation | **arXiv:** https://arxiv.org/abs/2604.05014 | **Original code:** https://github.com/starVLA/starVLA

`starVLA` adapts the StarVLA vision-language-action framework to XPolicyLab/RoboDojo, exposing three variants that share the public `Qwen3-VL-4B-Instruct` backbone. Integration scripts live at this directory level; the vendored upstream implementation lives in `source_starvla/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Supported Variants

| Public name | StarVLA framework registry | Action head |
|---|---|---|
| `starVLA-OFT` | `QwenOFT` | MLP regression head over action-token hidden states |
| `starVLA-GR00T` | `QwenGR00T` | Flow-matching DiT action head |
| `starVLA-π` | `QwenPI_v3` | Layer-wise cross-DiT flow-matching action head |

These names are reporting labels. All three variants use `starVLA` as the XPolicyLab runtime `policy_name`; the selected checkpoint identifies the framework implementation. The action-policy components are trained from scratch, while Qwen3-VL-4B-Instruct is used as the public backbone initialization. No internal robot data, private demonstrations, hidden VLA pretraining, or unreleased pretrained policy weights are required by this adapter.

## External StarVLA Runtime Contract

The vendored `source_starvla` runtime implements the inference-side contract below. A separate checkout supplied through `starvla_root` must provide the same interface; checkpoints and normalization statistics are user-provided and are not included here.

- Register `QwenOFT`, `QwenGR00T`, and `QwenPI_v3` and reconstruct the framework selected by the checkpoint configuration.
- Accept three RGB observations (`cam_head`, `cam_left_wrist`, and `cam_right_wrist`), a language instruction, and an optional 14-dimensional ARX X5 absolute-joint state ordered as left arm, left gripper, right arm, and right gripper.
- Normalize state with the checkpoint's `arx_x5` training statistics before model inference. The 50-step RoboDojo schema uses q99 normalization for all dimensions, including the continuous grippers.
- Return normalized actions with shape `[batch, horizon, 14]` and expose `action_chunk_size` through websocket server metadata. The runtime must unnormalize actions with the matching `arx_x5` statistics before returning them to XPolicyLab.
- Support a 50-step predicted action chunk. RoboDojo executes the first 16 actions and then requests a new chunk.

Released checkpoints should retain their run-directory layout so the runtime can find `config.yaml`, `config.full.yaml`, and `dataset_statistics.json` next to the `checkpoints/` directory. The public data-mixture name is `robodojo_arx_x5_h50_q99`; `robodojo_v21_all_h50_q99` remains supported as a compatibility alias.

## Installation

`install.sh` installs PyTorch 2.6, the upstream requirements, flash-attn, and `source_starvla/` in editable mode into the active environment:

```bash
cd XPolicyLab/policy/starVLA
bash install.sh
conda activate <policy_env>  # e.g. starvla
```

## Data Processing

Converts RoboDojo demos into a LeRobot dataset at `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`. `raw_task_dirs` is the source task dir (or comma-separated list) under `data/<bench_name>/` and defaults to `ckpt_name`; a non-numeric fifth argument is treated as `raw_task_dirs`. Use the same `ckpt_name` when launching `train.sh`, unless you also set `STARVLA_XPOLICY_DATASET_NAME` explicitly.

```bash
cd XPolicyLab/policy/starVLA
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls

# Example: rename the output while using all episodes from the original task data
bash process_data.sh RoboDojo stack_bowls_full arx_x5 joint stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/starVLA
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Example: train the converted stack_bowls dataset on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. The trainer generates a per-run config from `xpolicy_oft_vla.yaml` under `.generated/`, requires the converted LeRobot dataset (checked via `meta/modality.json`), and infers the process count from a comma-separated `gpu_id`; trailing `extra_args` are forwarded to the upstream trainer. Overrides: `STARVLA_DATA_ROOT` (default `data/`), `STARVLA_DATA_MIX` (default `xpolicylab_runtime`), `STARVLA_XPOLICY_DATASET_NAME` (default `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`).

## Evaluation

```bash
cd XPolicyLab/policy/starVLA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained stack_bowls checkpoint
bash eval.sh RoboDojo stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `starvla_root`, `checkpoint_path`, `starvla_server_host`, `starvla_server_port`, `unnorm_key`, `use_ddim`, `num_ddim_steps`, `image_size`, `execute_horizon` (number of actions executed before requesting a new predicted chunk), `include_state` (`auto` reads the checkpoint config; an explicit boolean overrides it).

`include_state: auto` reads `datasets.vla_data.include_state` from `config.yaml`, then falls back to `config.full.yaml` and finally `false`. `STARVLA_INCLUDE_STATE` remains the highest-priority explicit override.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `STARVLA_EXECUTE_HORIZON` | Overrides the number of actions executed from each predicted chunk. |
| `STARVLA_IMAGE_SIZE` | Overrides the input image size passed to the adapter. |
| `STARVLA_INCLUDE_STATE` | Explicitly overrides checkpoint-driven proprioceptive state selection. |
| `STARVLA_UNNORM_KEY` | Selects the checkpoint normalization statistics. |

The scripts also read `STARVLA_CKPT_PATH`, `STARVLA_ROOT`, and `STARVLA_SERVER_PID`, plus the training overrides listed above.
