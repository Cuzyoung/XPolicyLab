# G05 RoboDojo Real Adapter

This directory contains the G05 policy adapter for RoboDojo evaluation through
XPolicyLab.

The adapter loads a G05 policy implementation from an external G05 checkout and
serves it through the XPolicyLab websocket policy-server interface. RoboDojo
simulation remains on the evaluator/client side.

## Checkpoint

Download the G05 RoboDojo-real checkpoint package from the OpenGalaxea Hugging Face organization. Access to the real checkpoint may be gated until evaluation is finalized. The archive and the extracted checkpoint intentionally do not expose the training step.

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_real_checkpoint.tar \
  g05_robodojo_real_checkpoint.tar.sha256 \
  --local-dir ./checkpoints/g05_real

cd ./checkpoints/g05_real
sha256sum -c g05_robodojo_real_checkpoint.tar.sha256
tar -xf g05_robodojo_real_checkpoint.tar
```

Set the checkpoint path for evaluation:

```bash
export G05_CKPT_PATH=/path/to/g05_robodojo_real_checkpoint/checkpoint/checkpoints/checkpoint.pt
export ROBODOJO_G05_ACTION_SOURCE=fm
```

## Training base assets

To reproduce G05 RoboDojo training, download the base model and tokenizer assets separately from the evaluation checkpoint:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_train_base_assets.tar \
  g05_robodojo_train_base_assets.tar.sha256 \
  --local-dir ./checkpoints/g05_train_assets

cd ./checkpoints/g05_train_assets
sha256sum -c g05_robodojo_train_base_assets.tar.sha256
tar -xf g05_robodojo_train_base_assets.tar
```

These assets include the base checkpoint, HF processor, and action tokenizer required by the G05 training configs. RoboDojo datasets should be downloaded from the official RoboDojo data source separately.

## Runtime requirements

Install the XPolicyLab-side adapter dependencies:

```bash
cd policy/G05
export G05_PYTHON=/path/to/python
bash install.sh
```

Then set these paths before training or evaluation:

```bash
export G05_ROOT=/path/to/GalaxeaVLA_or_G05_checkout
export G05_PYTHON=/path/to/python
```

`G05_ROOT` must contain the G05 model/inference code compatible with the real checkpoint. The real checkpoint was produced with the private/current G05 training stack, so evaluators should use the matching G05 runtime provided for official real evaluation.
`G05_PYTHON` must point to a Python environment with the G05 runtime
dependencies installed.

The checkpoint package does not include a full Python runtime. Keep the G05
runtime as a normal external checkout/environment and point the adapter to it
with `G05_ROOT` and `G05_PYTHON`.

## Evaluation

Debug mode validates policy-server wiring and action schema without launching
Isaac Sim:

```bash
cd policy/G05
export EVAL_ENV_TYPE=debug
export G05_CKPT_PATH=/path/to/extracted/g05/checkpoint
bash eval.sh RoboDojo stack_bowls checkpoint arx_x5 joint 0 0 0 \
  "$G05_PYTHON" base
```

Simulator-backed evaluation uses the same adapter entrypoint. Example:

```bash
cd policy/G05
export G05_ROOT=/path/to/G05_checkout
export G05_PYTHON=/path/to/python
export G05_CKPT_PATH=/path/to/extracted/g05/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm

bash eval.sh RoboDojo stack_bowls checkpoint arx_x5 joint 0 0 0 \
  "$G05_PYTHON" sim
```

The positional arguments are:

```text
eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_python_or_env> <eval_env>
```

Use `ROBODOJO_G05_ACTION_SOURCE=fm` for FM-style continuous action inference,
or `ROBODOJO_G05_ACTION_SOURCE=ar` for AR-style action decoding when evaluating
a compatible checkpoint.

For simulator evaluation, unset `EVAL_ENV_TYPE` or set it to `sim`, and run from
a RoboDojo checkout where the evaluator-side directories `env_cfg/`, `scripts/`,
`src/`, `task/`, and `Assets/` are available next to `XPolicyLab`.

## Training

Training is optional for evaluation. If training is needed, set the RoboDojo
LeRobot v3.0 joint-action dataset path explicitly:

```bash
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video
export G05_ROOT=/path/to/G05_checkout
export G05_PYTHON=/path/to/python
cd policy/G05
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

The adapter currently targets RoboDojo `arx_x5` with `joint` actions.
