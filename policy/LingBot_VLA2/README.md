# LingBot-VLA2

This adapter targets the official source repository only:

- repository: <https://github.com/Robbyant/lingbot-vla-v2>
- audited revision: `951475ae1b1d87553e7dc47c97b53a3d695c0d13`
- expected checkout: `policy/LingBot_VLA2/lingbot_vla_v2/`

The upstream checkout is pinned at that revision as a nested submodule. Initialize
it with `git submodule update --init --recursive`. Do not point this adapter at
`policy/LingBot_VLA/lingbot_vla`: that directory is the older Qwen2.5
implementation and is not weight-compatible with V2.

Deployment additionally requires one bundle conforming to `bundle.schema.json`.
The manifest pins the official source revision and declares the original
`lingbotvla_cli.yaml`, complete `hf_ckpt`, matching YAM norm stats, training
robot config, native control rate, and action horizon. Runtime config points to
that single manifest; compliant post-training output can be loaded without
changing this adapter. The public foundation checkpoint alone is not a YAM
policy.

## RTC

The official V2 sampler does not expose online action conditioning. `rtc.py`
uses its public prefix-cache and `predict_velocity` primitives to run an
adapter-owned PI-guided flow loop with VJP guidance at every denoising step.
`Model.get_action_rtc` normalizes the raw 14-D YAM condition with the same
checkpoint `FeatureTransform`, pads it to the 55-D canonical action space, and
masks unused dimensions. This is sampler-level conditioning, not chunk splicing.

The RTC path is offline-tested but not live-validated: a real YAM bundle and
measured steady latency are still required before using `infra/rtc.yaml` on a
robot.
