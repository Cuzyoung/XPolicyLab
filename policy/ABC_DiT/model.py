"""XPolicyLab adapter for the official ABC-DiT policy (https://abc.bot).

The inference body is the non-simulation half of upstream
``abc_minimal.eval_policy.SimPolicy``: z-score the 14-D joint state, letterbox
and ImageNet-normalize each camera, encode the instruction with CLIP, run the
rectified-flow sampler, then unnormalize to absolute joint targets.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_POLICY_DIR = Path(__file__).resolve().parent
_UPSTREAM_DIR = _POLICY_DIR / "upstream"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

# ABC camera key -> XPolicyLab observation camera name. The checkpoint indexes its
# attention-pool queries by these names: top = scene camera, left/right = wrists.
DEFAULT_CAMERA_MAP = {"top": "cam_head", "left": "cam_left_wrist", "right": "cam_right_wrist"}
IMAGE_SIZE = 224


def _ensure_upstream_on_path() -> None:
    value = str(_UPSTREAM_DIR)
    if value not in sys.path:
        sys.path.insert(0, value)


def letterbox(image_hwc: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Aspect-preserving resize + centered zero pad to ``size``x``size``, CHW uint8.

    ABC's training cache is produced by ``export_mcap.py`` with ffmpeg
    ``scale=224:224:force_original_aspect_ratio=decrease:flags=bicubic`` and a
    centered pad. Antialiased bicubic matches that output to ~0.1/255 mean
    absolute error on 640x480 station frames, closer than OpenCV cubic/area or
    the upstream bilinear fallback.
    """
    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    chw = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    if (h, w) == (size, size):
        return chw.numpy()
    ratio = max(w / size, h / size)
    new_h, new_w = max(1, int(h / ratio)), max(1, int(w / ratio))
    resized = F.interpolate(
        chw[None].float(), size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True
    )[0].round().clamp(0, 255).to(torch.uint8)
    out = torch.zeros((3, size, size), dtype=torch.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[:, top : top + new_h, left : left + new_w] = resized
    return out.numpy()


def _camera_image(observation: dict[str, Any], wire_name: str) -> np.ndarray:
    vision = observation.get("vision", {})
    if wire_name not in vision:
        raise KeyError(f"observation is missing camera {wire_name!r}; got {sorted(vision)}")
    image = vision[wire_name]
    if isinstance(image, dict):
        image = image.get("color", image.get("rgb"))
    return np.asarray(image)


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        _ensure_upstream_on_path()
        from abc_minimal.config import ClipConfig, DiTConfig, validate_model_config
        from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, infer_dit_shape, load_pretrained
        from abc_minimal.preprocess import load_norm_stats, parse_norm_stats

        self.task_name = model_cfg.get("task_name")
        self.action_type = model_cfg.get("action_type") or "joint"
        if self.action_type != "joint":
            raise ValueError("ABC-DiT emits absolute joint positions; action_type must be 'joint'")
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg)
        self.inference_seed = model_cfg.get("inference_seed", 0)
        if (
            isinstance(self.inference_seed, bool)
            or not isinstance(self.inference_seed, int)
            or not 0 <= self.inference_seed < 2**32
        ):
            raise ValueError("inference_seed must be an integer in [0, 2**32)")
        self.num_steps = int(model_cfg.get("num_steps", 10))
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        # Official ABC RTC: hard action-prefix conditioning on the last executed rows.
        self.rtc_prefix_length = int(model_cfg.get("rtc_prefix_length", 4))
        if self.rtc_prefix_length <= 0:
            raise ValueError(f"rtc_prefix_length must be positive, got {self.rtc_prefix_length}")
        # A non-empty prompt replaces the client's instruction: ABC conditions on a
        # CLIP vector of its training task text, which seldom equals the station prompt.
        self.prompt = str(model_cfg.get("prompt") or "").strip() or None
        self.camera_map = dict(model_cfg.get("camera_map") or DEFAULT_CAMERA_MAP)
        self.device = torch.device(model_cfg.get("device") or "cuda:0")

        self.checkpoint_path = resolve_checkpoint_root(
            model_cfg,
            _CHECKPOINTS_DIR,
            policy_dir=_POLICY_DIR,
            explicit_keys=("model_path", "checkpoint_path"),
        )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"ABC-DiT checkpoint must be a .pt file: {self.checkpoint_path}")

        shape = infer_dit_shape(self.checkpoint_path)
        self.model_config = DiTConfig(**shape)
        errors = validate_model_config(self.model_config)
        if errors:
            raise ValueError("invalid ABC-DiT model config: " + "; ".join(errors))
        if set(self.camera_map) != set(self.model_config.camera_keys):
            raise ValueError(
                f"camera_map keys must be {list(self.model_config.camera_keys)}, "
                f"got {list(self.camera_map)}"
            )
        action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        if action_dim != self.model_config.action_dim:
            raise ValueError(
                f"robot_action_dim_info sums to {action_dim}, checkpoint emits "
                f"{self.model_config.action_dim}"
            )
        # Gripper columns: the trailing tool value(s) of each arm, normalized to [0, 1].
        self._gripper_columns = []
        offset = 0
        for arm, tool in zip(self.robot_action_dim_info["arm_dim"], self.robot_action_dim_info["ee_dim"]):
            self._gripper_columns.extend(range(offset + arm, offset + arm + tool))
            offset += arm + tool

        self.model = DiTPolicy(self.model_config).to(self.device)
        checkpoint = load_pretrained(self.model, self.checkpoint_path)
        self.model.eval()
        self.checkpoint_step = checkpoint.get("step")
        norm_stats_path = model_cfg.get("norm_stats_path")
        if norm_stats_path:
            self.norm_stats_source = str(Path(norm_stats_path).expanduser().resolve())
            self.norm_stats = load_norm_stats(self.norm_stats_source)
        elif checkpoint.get("norm_stats") is not None:
            self.norm_stats_source = "checkpoint"
            self.norm_stats = parse_norm_stats(checkpoint["norm_stats"])
        else:
            raise ValueError("checkpoint has no norm_stats; set norm_stats_path")
        trained_prefix = (checkpoint.get("train_config") or {}).get("max_action_prefix")
        del checkpoint

        clip_cache_dir = model_cfg.get("clip_cache_dir")
        clip_config = ClipConfig(cache_dir=str(Path(clip_cache_dir).expanduser())) if clip_cache_dir else ClipConfig()
        self.embedder = CLIPTextEmbedder(clip_config, device=self.device)
        self._fast_graph = None
        self._fast_prefix_graph = None
        self._task_vec_prompt: str | None = None
        self.task_vec = self._encode_prompt(self.prompt or "")
        self.trained_max_action_prefix = trained_prefix
        # Training samples prefix lengths in [0, max_action_prefix).
        if trained_prefix is not None and self.rtc_prefix_length >= int(trained_prefix):
            raise ValueError(
                f"rtc_prefix_length {self.rtc_prefix_length} is outside the trained range "
                f"[1, {int(trained_prefix) - 1}]"
            )

        self._rng = np.random.default_rng(self.inference_seed)
        self._obs: dict[str, Any] | None = None
        self.fast_inference = bool(model_cfg.get("fast_inference", False))
        if self.fast_inference:
            self._enable_fast_inference(str(model_cfg.get("fast_compile_mode", "max-autotune-no-cudagraphs")))
        elif bool(model_cfg.get("allow_tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # The attributes below (config, diffusion_steps) are the protocol that
    # upstream FastInferenceGraph reads from its policy object.
    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(model=self.model_config)

    @property
    def diffusion_steps(self) -> int:
        return self.num_steps

    def normalized_action_prefix(self, action_prefix: np.ndarray, prefix_length: int) -> np.ndarray:
        """Upstream ``SimPolicy.normalized_action_prefix`` (read by ``FastRTCInferenceGraph``)."""
        from abc_minimal.preprocess import normalize

        m = self.model_config
        prefix = np.asarray(action_prefix, dtype=np.float32)
        if prefix.shape == (prefix_length, m.action_dim):
            full = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
            full[:prefix_length] = prefix
            prefix = full
        if prefix.shape != (m.chunk_length, m.action_dim):
            raise ValueError(
                f"action_prefix must have shape {(m.chunk_length, m.action_dim)} "
                f"or {(prefix_length, m.action_dim)}, got {prefix.shape}"
            )
        return normalize(prefix, self.norm_stats["actions"]).astype(np.float32, copy=False)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if prompt != self._task_vec_prompt:
            dtype = self.model.task_to_hidden.weight.dtype
            self.task_vec = self.embedder.encode([prompt]).to(self.device, dtype=dtype)
            self._task_vec_prompt = prompt
            for graph in (self._fast_graph, self._fast_prefix_graph):
                if graph is not None:
                    graph.static_task_vec.copy_(self.task_vec)
        return self.task_vec

    def _enable_fast_inference(self, compile_mode: str) -> None:
        """Upstream fast path: bf16 weights, compiled velocity, one CUDA graph per sampler."""
        from abc_minimal.fast_inference import FastInferenceGraph, FastRTCInferenceGraph

        if self.device.type != "cuda":
            raise RuntimeError("fast_inference requires a CUDA device")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model.to(torch.bfloat16)
        self.model.img_backbone.set_bfloat16(True)
        self.task_vec = self.task_vec.to(dtype=torch.bfloat16)
        kwargs: dict[str, Any] = {"dynamic": False}
        if compile_mode:
            kwargs["mode"] = compile_mode
        self.model.predict_velocity = torch.compile(self.model.predict_velocity, **kwargs)
        m = self.model_config
        warmup_obs = {
            "state": np.zeros(m.state_dim, dtype=np.float32),
            "images": {cam: np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) for cam in m.camera_keys},
        }
        graph = FastInferenceGraph(self)
        zeros = np.zeros((m.chunk_length, m.action_dim), np.float32)
        graph.capture(warmup_obs, zeros, 8)
        self._fast_graph = graph
        # Official ABC RTC (get_action_paint) at the configured prefix length, as upstream warmup_rtc.
        prefix_graph = FastRTCInferenceGraph(self, self.rtc_prefix_length)
        prefix_graph.capture(warmup_obs, zeros, zeros, replay_warmups=8)
        self._fast_prefix_graph = prefix_graph
        # Compile the gradient-enabled velocity used by guided RTC now; the first
        # live RTC request would otherwise spend minutes compiling.
        self._obs = {**warmup_obs, "prompt": self.prompt or ""}
        self._infer(zeros, np.ones(m.chunk_length, np.float32), 5.0)
        self._obs = None
        self._rng = np.random.default_rng(self.inference_seed)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "abc_dit",
            "task_name": self.task_name,
            "action_type": self.action_type,
            "action_semantics": "absolute_joint_position",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_step": self.checkpoint_step,
            "norm_stats_source": self.norm_stats_source,
            "model_shape": {
                "hidden_size": self.model_config.hidden_size,
                "depth": self.model_config.depth,
                "num_heads": self.model_config.num_heads,
            },
            "camera_map": self.camera_map,
            "prompt": self.prompt,
            "action_horizon": int(self.model_config.chunk_length),
            "action_dim": int(self.model_config.action_dim),
            "num_steps": self.num_steps,
            "inference_seed": self.inference_seed,
            "trained_max_action_prefix": self.trained_max_action_prefix,
            "rtc_prefix_length": self.rtc_prefix_length,
            "execution": {
                "device": str(self.device),
                "parameter_dtype": str(next(self.model.parameters()).dtype),
                "fast_inference": self.fast_inference,
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "framework_versions": {"torch": torch.__version__, "numpy": np.__version__},
            },
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("ABC-DiT serves one robot per server (num_envs: 1)")
        obs = obs_list[0]
        state = pack_robot_state(obs, "joint", self.robot_action_dim_info, source_type="obs")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.model_config.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"state must be {self.model_config.state_dim} finite values, got {state.shape}")
        self._obs = {
            "state": state,
            "images": {
                cam: letterbox(_camera_image(obs, wire)) for cam, wire in self.camera_map.items()
            },
            "prompt": self.prompt or str(obs.get("instruction") or obs.get("prompt") or ""),
        }

    def _infer(
        self,
        action_condition=None,
        condition_weights=None,
        beta: float = 5.0,
        action_prefix=None,
    ) -> np.ndarray:
        from abc_minimal.preprocess import normalize, resize_pad_normalize, unnormalize

        if self._obs is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        obs = self._obs
        m = self.model_config
        self._encode_prompt(obs["prompt"])
        noise = self._rng.standard_normal((m.chunk_length, m.action_dim), dtype=np.float32)
        if action_condition is None and action_prefix is None and self._fast_graph is not None:
            return self._fast_graph.infer(obs, noise)
        graph = self._fast_prefix_graph
        if action_prefix is not None and graph is not None and len(action_prefix) == graph.prefix_length:
            return graph.infer(obs, noise, np.asarray(action_prefix, dtype=np.float32))

        with torch.no_grad():
            state = normalize(obs["state"], self.norm_stats["state"])
            batch = {
                "state": torch.from_numpy(state[None]).float().to(self.device),
                "images": {
                    cam: resize_pad_normalize(obs["images"][cam]).unsqueeze(0).to(self.device)
                    for cam in m.camera_keys
                },
                "task_vec_clip": self.task_vec,
            }
            noise_t = torch.from_numpy(noise[None]).to(self.device)
        if action_prefix is not None:
            # Upstream SimPolicy.normalized_action_prefix: rows [0, p) hold the prefix.
            prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
            prefix[: len(action_prefix)] = action_prefix
            prefix = normalize(prefix, self.norm_stats["actions"]).astype(np.float32)
            with torch.no_grad():
                actions = self.model.sample_actions_rtc(
                    batch,
                    torch.from_numpy(prefix[None]).to(self.device),
                    prefix_length=len(action_prefix),
                    num_steps=self.num_steps,
                    noise=noise_t,
                )
        elif action_condition is None:
            actions = self.model.sample_actions(batch, num_steps=self.num_steps, noise=noise_t)
        else:
            condition = normalize(action_condition, self.norm_stats["actions"]).astype(np.float32)
            actions = self.model.sample_actions_pi_rtc(
                batch,
                torch.from_numpy(condition[None]).to(self.device),
                torch.from_numpy(condition_weights[None]).to(self.device),
                beta=beta,
                num_steps=self.num_steps,
                noise=noise_t,
            )
        actions_np = actions[0].float().detach().cpu().numpy()
        return unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)

    def _decode(self, actions: np.ndarray) -> list[dict[str, np.ndarray]]:
        actions = np.array(actions, dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("ABC-DiT produced non-finite actions")
        # Grippers were trained in [0, 1]; the reference deployment clips them.
        actions[:, self._gripper_columns] = np.clip(actions[:, self._gripper_columns], 0.0, 1.0)
        return unpack_robot_state(actions, "joint", self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self._decode(self._infer())

    def get_action_batch(self, env_idx_list=None):
        return [self.get_action()]

    def get_action_rtc(self, sampling: dict[str, Any]):
        """Physical Intelligence inference-time RTC (PiGDM guidance at every flow step)."""
        required = {"action_condition", "condition_weights", "beta"}
        missing = sorted(required - set(sampling))
        if missing:
            raise ValueError(f"RTC sampling is missing fields: {missing}")
        m = self.model_config
        condition = np.array(sampling["action_condition"], dtype=np.float32, copy=True)
        weights = np.array(sampling["condition_weights"], dtype=np.float32, copy=True)
        beta = float(sampling["beta"])
        if condition.shape != (m.chunk_length, m.action_dim):
            raise ValueError(
                f"action_condition must have shape {(m.chunk_length, m.action_dim)}, got {condition.shape}"
            )
        if weights.shape != (m.chunk_length,):
            raise ValueError(f"condition_weights must have shape {(m.chunk_length,)}, got {weights.shape}")
        if not np.isfinite(condition).all() or not np.isfinite(weights).all():
            raise ValueError("RTC sampling arrays must be finite")
        if np.any((weights < 0) | (weights > 1)):
            raise ValueError("condition_weights must lie in [0, 1]")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"RTC beta must be finite and positive, got {beta}")
        return self._decode(self._infer(condition, weights, beta))

    def get_action_paint(self, sampling: dict[str, Any]):
        """Official ABC RTC (training-time action-prefix conditioning) on the PAINT schedule.

        ManiMux sends the ``d`` rows that execute while this request is in flight,
        with row 0 at the observation. Upstream ``_RTCManager`` conditions the new
        chunk's rows [0, p) on the last ``p`` of those rows and switches to its row
        ``p``. Returning ``prefix[:d - p]`` followed by that chunk keeps row 0 at the
        observation: rows [0, d) replay the old actions and the new ones start at ``d``.
        """
        required = {"action_prefix", "delay_steps"}
        missing = sorted(required - set(sampling))
        if missing:
            raise ValueError(f"PAINT sampling is missing fields: {missing}")
        m = self.model_config
        delay = int(sampling["delay_steps"])
        prefix = np.array(sampling["action_prefix"], dtype=np.float32, copy=True)
        if not 0 < delay < m.chunk_length:
            raise ValueError(f"delay_steps must satisfy 0 < d < {m.chunk_length}, got {delay}")
        if prefix.shape != (delay, m.action_dim) or not np.isfinite(prefix).all():
            raise ValueError(
                f"action_prefix must be finite with shape {(delay, m.action_dim)}, got {prefix.shape}"
            )
        p = min(self.rtc_prefix_length, delay)
        chunk = self._infer(action_prefix=prefix[delay - p :])
        actions = np.concatenate([prefix[: delay - p], chunk])[: m.chunk_length]
        actions[:delay] = prefix
        return {
            "actions": self._decode(actions),
            "paint": {
                "delay_steps": delay,
                "num_steps": self.num_steps,
                "model_evaluations": self.num_steps,
                "inversion": "none",
                "method": "abc_action_prefix",
                "prefix_length": p,
            },
        }

    def reset(self):
        self._rng = np.random.default_rng(self.inference_seed)
        self._obs = None
