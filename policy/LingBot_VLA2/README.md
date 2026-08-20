# LingBot-VLA2

This adapter targets the official source repository only:

- repository: <https://github.com/Robbyant/lingbot-vla-v2>
- audited revision: `951475ae1b1d87553e7dc47c97b53a3d695c0d13`
- expected checkout: `policy/LingBot_VLA2/lingbot_vla_v2/`

The upstream checkout is pinned at that revision as a nested submodule. Initialize
it with `git submodule update --init --recursive`. Do not point this adapter at
`policy/LingBot_VLA/lingbot_vla`: that directory is the older Qwen2.5
implementation and is not weight-compatible with V2.

Deployment additionally requires a YAM post-training checkpoint, its original
`lingbotvla_cli.yaml`, matching YAM norm stats, and the absolute-joint robot
profile in `robot_configs/yam_dual_absolute.yaml`. The public foundation
checkpoint alone is not a YAM policy.
