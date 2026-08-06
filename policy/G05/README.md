# G05 RoboDojo Adapter

This adapter integrates the GitHub G0.5 checkout at
`/efm-nas/efm-nas/group-jt/haoyu.zhang/GalaxeaVLA_github_port` with XPolicyLab. It does not use the vendored
`policy/GalaxeaVLA/GalaxeaVLA` code.

## Training

Default training uses G0.5 task config:

```bash
cd /efm-nas/efm-nas/group-jt/haoyu.zhang/external/robodojo/XPolicyLab/policy/G05
export ROBODOJO_LEROBOT_V30_ROOT=/efm-nas/efm-nas/group-jt/haoyu.zhang/external/robodojo/data/RoboDojo_lerobot_v30_video
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

For the GitHub G0.5 launcher, use
`/efm-nas/efm-nas/group-jt/haoyu.zhang/GalaxeaVLA_github_port/scripts/run/finetune_robodojo_arx_x5_joint.sh`.

## Evaluation

Set `G05_CKPT_PATH` to a G0.5 run directory or `.pt` checkpoint. Debug mode
validates websocket wiring and action schema without the simulator:

```bash
cd /efm-nas/efm-nas/group-jt/haoyu.zhang/external/robodojo/XPolicyLab/policy/G05
export EVAL_ENV_TYPE=debug
export G05_CKPT_PATH=/path/to/g05/run/or/checkpoints/checkpoint
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 \
  /mlplatform/haoyu.zhang/g05_runtime/g05_venv_nas base
```

For simulator evaluation, unset `EVAL_ENV_TYPE` or set it to `sim` and make
sure the RoboDojo evaluator-side repo with `env_cfg/`, `scripts/`, `src/`, and
`task/` is mounted next to `XPolicyLab`.

## Checkpoint

The submitted FM-only RoboDojo ARX X5 joint checkpoint and inference assets are hosted at:

https://huggingface.co/XZHY528/g05

Download and extract `xpolicylab_g05_fm_only_checkpoint_20260724.tar`, then set `G05_CKPT_PATH` to the extracted `XPolicyLab/policy/G05/checkpoints/checkpoint` path before evaluation.

Required sidecars included in the archive:

- `.hydra/config.yaml`
- `dataset_stats.json`
- `action_tokenizer.pt`
