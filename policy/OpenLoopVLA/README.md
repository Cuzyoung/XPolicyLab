# OpenLoopVLA

**Contributor:** Cuzyoung | **Upstream source:** https://github.com/Cuzyoung/OpenLoopVLA

`OpenLoopVLA` is the HRM-Penguin V2 plus PI05-style action-expert policy used for
the RoboTwin 2.0 clean-50 experiment. The upstream source is pinned as the
`OpenLoopVLA/` submodule. This first XPolicyLab contribution is eval-only;
training remains in the upstream repository until its launcher is converted to
the standard XPolicyLab training interface.

Shared argument, checkpoint and split-machine conventions are documented in the
[XPolicyLab README](../../README.md). Official evaluation should be launched by
the RoboTwin `scripts/eval_policy.sh` scheduler.

## Installation

```bash
git submodule update --init policy/OpenLoopVLA/OpenLoopVLA
bash policy/OpenLoopVLA/install.sh openloopvla
```

An existing upstream environment may be reused if it can import OpenLoopVLA and
XPolicyLab. Keep `STARVLA_DISABLE_DEEPSPEED=1` for inference.

## Data Processing

The adapter does not add a second conversion pipeline. The evaluated checkpoint
must retain its native `config.yaml` and `dataset_statistics.json`; the native
training transforms own all 14-dimensional state normalization and action
denormalization.

## Training

This adapter is currently eval-only. Use the pinned upstream repository's
`examples/OpenLoopVLA/` training entry points for the existing 100K-step recipe.

## Evaluation

Supported contract: `bench_name=RoboTwin`, `env_cfg_type=arx_x5`,
`action_type=joint`; current head plus left/right wrist RGB, raw 14D state,
absolute 14D actions. The model predicts 50 actions and the adapter returns the
first 12 for execution before RoboTwin requests another prediction.

```bash
export OPENLOOPVLA_V2_PACKAGE_ROOT=/path/to/hrm-penguin-v2
export OPENLOOPVLA_UNNORM_KEY=new_embodiment

bash policy/OpenLoopVLA/eval.sh \
  RoboTwin beat_block_hammer \
  /path/to/export/steps_5000/checkpoints/steps_5000_model.pt \
  arx_x5 joint 0 0 0 openloopvla robotwin
```

The `.pt` file's package root must also contain `config.yaml` and
`dataset_statistics.json`. `mmap_checkpoint=true` preserves the native mixed
precision layout while reducing CPU loading pressure. Images remain RGB end to
end; the adapter does not decode or swap channels.

