# Isaac 0.5

**Contributor:** Cuzyoung | **Paper:** [Isaac 0.5](https://pub-d90b81cad7254a1aa6b148ac18153c0c.r2.dev/isaac-0.5.pdf) | **Original code:** [perceptron-ai-inc/isaac](https://github.com/perceptron-ai-inc/isaac) | **Weights:** [PerceptronAI/Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5)

This eval-only adapter keeps the official Perceptron LeRobot policy at commit
`e12389c1f8f591ad05dced4e284d4e92e48c5df4` as a submodule. The outer Isaac release
identified by the model card is `be6507b4aed7472f2029606c22684d4ebc9d73e6`.
XPolicyLab translates only observation and action dictionaries around the official
preprocessor, Flow policy and postprocessor.

Shared argument conventions and the split-machine workflow are documented in the
[XPolicyLab README](../../README.md). Official benchmark results belong on the
[RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Initialize the nested source and install the official locked CUDA dependency set:

```bash
git -C XPolicyLab submodule update --init policy/Isaac_05/lerobot
bash XPolicyLab/policy/Isaac_05/install.sh
```

The public `trained_policy` additionally enforces the upstream-qualified
PyTorch `2.10.0+cu128`, Transformers `5.5.4`, `torch_sdpa_v1`, and NVIDIA H100
runtime. The repository lock currently selects PyTorch `2.11` for development,
so installing dependencies alone is not a successful model-load claim. Do not
replace mHarmony, TensorStream, or the model internals with similar local code.

## Data Processing

Not included. This is an eval-only integration of the public LIBERO Spatial
artifact. It does not claim an XPolicy conversion recipe for Isaac training data.

## Training

Not included. Use the official Isaac/LeRobot fine-tuning guide. A future training
submission must preserve the official action, state, camera and normalization
contracts and publish a matching deployable package.

## Evaluation

Download the checkpoint:

```bash
bash XPolicyLab/policy/Isaac_05/download.sh
```

Run the no-model static contract check from the ManiMux parent workspace:

```bash
envs/yam/.venv/bin/python scripts/servers/isaac05_server.py --check \
  --config configs/isaac05/libero/server/base.yaml
```

On hardware able to load the official checkpoint, start the model service:

```bash
bash XPolicyLab/policy/Isaac_05/setup_eval_policy_server.sh \
  configs/isaac05/libero/server/base.yaml
```

Then run the ManiMux model-only probe:

```bash
envs/isaac-0.5/.venv/bin/python scripts/validation/isaac05_forward_probe.py \
  --server ws://127.0.0.1:8504
```

## Model Assets

The default model path is
`checkpoints/pretrained/perceptron-ai/isaac-0.5/lerobot_policy`. The package
points to its parent directory for the model shards and must retain its official
preprocessor, postprocessor, deployment adapter and native stats files.

The public artifact is a LIBERO Spatial checkpoint:

- cameras: `image`, `wrist_image`, RGB, `256 x 256`
- state: `8D` LIBERO proprioception
- model output: `50 x 7` checkpoint-native absolute EE actions
- exposed chunk: the first `8 x 7` actions at `20 Hz`
- inference: `10` Flow steps, one sample

It is not a YAM checkpoint. No `7D -> 14D` robot conversion is implemented or
claimed. The adapter exposes only `sampling.mode=default`; the pinned official
policy rejects LeRobot RTC for this checkpoint.
