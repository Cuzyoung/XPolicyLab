# ABC_DiT

**Contributor:** ManiMux maintainers | **Paper:** [Scalable Behavior Cloning with Open Data, Training, and Evaluation](https://abc.bot/abc.pdf) | **arXiv:** Not available | **Original code:** [amazon-far/abc](https://github.com/amazon-far/abc)

`ABC_DiT` serves the official ABC-DiT policy (DINOv3 ViT-B/16 vision, CLIP ViT-B/32 task text, DiT-XL rectified-flow action head, 2.02B parameters) for a bimanual YAM: three RGB cameras plus 14-D joint state in, a 30-step chunk of absolute joint positions out. It supports default sampling, the official ABC RTC — training-time action-prefix conditioning served through the `paint` sampling mode (`get_action_paint`) — and Physical Intelligence inference-time RTC (`get_action_rtc`). The inference source is vendored unmodified under `upstream/` (provenance in [upstream/UPSTREAM.md](upstream/UPSTREAM.md)).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Creates a uv environment with Python 3.12, torch 2.11 (CUDA 12.8), the CLIP tokenizer dependencies and XPolicyLab:

```bash
bash XPolicyLab/policy/ABC_DiT/install.sh [<venv_dir>]   # default: policy/ABC_DiT/.venv
```

The CLIP text weights (`ViT-B-32.pt`, `bpe_simple_vocab_16e6.txt.gz`) are read from `~/.cache/clip/` (or `clip_cache_dir`) and downloaded there on first use.

## Data Processing

Unsupported here (eval-only). ABC's own `export_mcap.py` / `export_hf_task.py` in the upstream repository build the training cache.

## Training

Unsupported here (eval-only). Use the upstream `train.py`; fine-tuned checkpoints in the same `.pt` format (model + `norm_stats`) load unchanged.

## Evaluation

The supported deployment is ManiMux real-robot serving: see `docs/deployment/abc-dit-yam.md` in the parent workspace. The shared debug loop checks wiring only; simulator evaluation is unsupported:

```bash
EVAL_ENV_TYPE=debug bash eval.sh <bench_name> <task_name> <ckpt_path> yam_dual joint 0 \
  <policy_gpu_id> <env_gpu_id> <policy_venv> <eval_env_venv>
```

`ckpt_name` may be a path to the `.pt` file; `model_path` in the deployment config takes precedence.

## Configuration

| `deploy.yml` key | Meaning |
| --- | --- |
| `model_path` | ABC-DiT `.pt` checkpoint (`model`, `norm_stats`, optional `train_config`); the DiT width/depth are inferred from it |
| `norm_stats_path` | optional `norm_stats.json` overriding the checkpoint's embedded statistics |
| `robot_action_dim_info` | must sum to the checkpoint's 14 dims, e.g. `{arm_dim: [6, 6], ee_dim: [1, 1]}` |
| `camera_map` | ABC camera key → observation camera name; keys must be exactly `top`, `left`, `right` (scene, left wrist, right wrist) |
| `prompt` | non-empty: fixed instruction for every request; empty: use the observation's `instruction` |
| `num_steps` | Euler flow steps (default 10) |
| `rtc_prefix_length` | hard prefix rows for `get_action_paint` (official 4; must be below the checkpoint's `max_action_prefix`) |
| `inference_seed` | seeds the initial-noise sequence; `reset()` restarts it |
| `fast_inference` | bf16 weights + `torch.compile` + CUDA graph; also precompiles the RTC path at startup |
| `allow_tf32` | TF32 matmuls when `fast_inference` is off |
| `device`, `clip_cache_dir` | torch device; CLIP asset directory |

Images are letterboxed to 224×224 with antialiased bicubic resizing, matching ffmpeg's resize in ABC's training export. Gripper channels are clipped to `[0, 1]` (0 closed, 1 open).

`tests/test_contract.py` runs on CPU with a tiny random checkpoint:

```bash
PYTHONPATH=<workspace>:<workspace>/XPolicyLab <policy_venv>/bin/python -m pytest XPolicyLab/policy/ABC_DiT/tests
```

## Notes

- `get_action_paint(action_prefix (d, 14), delay_steps d)`: `action_prefix` holds the `d` actions that run while the request is in flight, row 0 at the observation. The last `p = rtc_prefix_length` rows condition upstream `sample_actions_rtc`; the reply is `action_prefix[:d-p]` followed by the new chunk, cut to 30 rows, so rows `[0, d)` repeat the prefix and new actions start at row `d` — the same switch point as upstream `_RTCManager` with `lead = d`.
- `fast_inference` integrates the flow in bf16, which quantizes outputs to about 0.008 rad; the official deploy command runs fp32.
