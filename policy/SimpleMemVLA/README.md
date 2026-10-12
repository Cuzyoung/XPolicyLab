# SimpleMemVLA

**Contributor:** [wadeKeith](https://github.com/wadeKeith) | **Paper:** SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models | **arXiv:** [2609.05533](https://arxiv.org/abs/2609.05533) | **Original code:** [OpenBMB/SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA)

Evaluation-only RoboDojo adapter for SimpleMemVLA, a vision-language-action model with native-video memory. It supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5` and `action_type=joint`. `simplememvla/` contains the inference code adapted from [OpenBMB/SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA) (commit `404215d`, MIT License).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
conda create -n simplememvla python=3.10 -y
conda activate simplememvla
cd XPolicyLab/policy/SimpleMemVLA
bash install.sh
```

## Data Processing

Not included; this is an eval-only adapter.

## Training

Not included; this is an eval-only adapter.

## Model Assets

Download the checkpoint from ModelScope ([keithyc/SimpleMemVLA-RoboDojo-Sim](https://modelscope.cn/models/keithyc/SimpleMemVLA-RoboDojo-Sim)) into `checkpoints/SimpleMemVLA-RoboDojo-Sim/`:

```bash
cd XPolicyLab/policy/SimpleMemVLA
bash download_checkpoint.sh
```

## Evaluation

```bash
cd XPolicyLab/policy/SimpleMemVLA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo stack_bowls SimpleMemVLA-RoboDojo-Sim arx_x5 joint 0 0 0 simplememvla RoboDojo
```

## Configuration

Extra `deploy.yml` keys:

- `checkpoint_path`: explicit checkpoint directory; overrides `ckpt_name` (default `null`).
- `execute_horizon`: actions executed from each predicted 30-step chunk (default `24`).
- `attn_implementation`: backbone attention implementation (default `flash_attention_2`).
- `warmup`: run one dummy inference before the server starts accepting requests (default `true`).
