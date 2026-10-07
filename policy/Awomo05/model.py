"""Awomo05 XPolicyLab policy adapter.

Inference uses three camera views in a fixed order, a 25 Hz episode history,
the robot's 14-dimensional joint state, and an instruction. The checkpoint's
text layout and normalization statistics must match this adapter.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

logger = logging.getLogger(__name__)

# Camera order is part of the model contract (view index -> RoPE time offset), so it is spelled
# out here rather than read from the run config. The alias tuples exist because the RoboDojo env
# uses multiple naming conventions; the FIRST name in each tuple is the canonical name.
CAMERA_ALIASES = (
    ("cam_head", "cam_high", "head_camera"),
    ("cam_left_wrist", "cam_hand_left", "left_camera"),
    ("cam_right_wrist", "cam_hand_right", "right_camera"),
)
RAW_FPS = 25.0                 # RoboDojo native control rate; the history grid is in these frames
PLUGIN_DIR = Path(__file__).resolve().parent
# Model code lives in the adjacent awomo package; weights are supplied separately.
DEFAULT_WEIGHTS_DIR = os.environ.get("AWOMO05_WEIGHTS", str(PLUGIN_DIR / "weights"))
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_LOCAL_DIR = "/tmp/Awomo05"   # container-local (node disk), not JuiceFS


def _ensure_logging_visible() -> None:
    """Give this plugin's INFO records somewhere to go when nobody configured logging.

    `setup_policy_server.py` does not configure the root logger, so with Python's default
    (last-resort handler, WARNING and above) every `logger.info` here is dropped: which checkpoint
    loaded, whether the code pin bound, how long staging took. Those
    are precisely the facts you need from a policy server you cannot attach a debugger to, and their
    absence is indistinguishable from a clean start.

    Only acts when the root logger has no handlers at all, so a harness that DID configure logging
    keeps its own format and level -- a library should not overwrite an application's choice.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _is_none_like(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _opt_int(value: Any) -> Optional[int]:
    return None if _is_none_like(value) else int(value)


def _opt_float(value: Any) -> Optional[float]:
    return None if _is_none_like(value) else float(value)


def _decode_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    elif isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tobytes()
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
    text = str(value).strip().strip("\x00").strip()
    return text or fallback


def _ensure_awomo_importable() -> Path:
    """`awomo` (vendored inference code) lives next to this file; make it importable as a top-level package."""
    root = str(PLUGIN_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    import awomo  # noqa: F401
    return Path(awomo.__file__).resolve().parent


def _stage_checkpoint(ckpt: Path, local_dir: Path, enabled: bool) -> Path:
    """Copy weights to local disk, falling back to the source if space is short."""
    if not enabled:
        return ckpt
    size = ckpt.stat().st_size
    dest_dir = local_dir / "ckpt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ckpt.name
    if dest.is_file() and dest.stat().st_size == size:
        logger.info("checkpoint already staged at %s (%.2f GiB)", dest, size / (1 << 30))
        return dest
    free = shutil.disk_usage(dest_dir).free
    if free < size * 11 // 10:
        logger.warning(
            "only %.1f GiB free under %s for a %.1f GiB checkpoint; reading from source instead "
            "(expect a slow startup)", free / (1 << 30), dest_dir, size / (1 << 30))
        return ckpt
    tmp = dest_dir / f"{ckpt.name}.tmp.{os.getpid()}"
    t0 = time.time()
    shutil.copyfile(ckpt, tmp)
    os.replace(tmp, dest)
    dt = max(time.time() - t0, 1e-6)
    logger.info("Staged %.2f GiB checkpoint -> %s in %.0fs (%.0f MiB/s)",
                size / (1 << 30), dest, dt, size / (1 << 20) / dt)
    return dest


def _resolve_model_config(model_config: Any) -> Path:
    """Resolve the architecture and preprocessing configuration."""
    p = Path(str(model_config)).expanduser() if not _is_none_like(model_config) else PLUGIN_DIR / "model_config.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"model_config not found: {p}")
    return p.resolve()


def _resolve_ckpt(weights_dir: Path, ckpt: Any) -> Path:
    if _is_none_like(ckpt):
        raise ValueError("Set `ckpt` to the selected checkpoint path or filename.")
    raw = str(ckpt).strip()
    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct.resolve()
    for name in (raw, f"{raw}.pt"):
        cand = weights_dir / name
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"checkpoint {raw!r} not found as a path or under {weights_dir}. "
        f"Available: {sorted(p.name for p in weights_dir.glob('*.pt'))}"
    )


def _extract_rgb(vision: dict, aliases: tuple[str, ...]) -> np.ndarray:
    """HWC uint8 RGB. The policy server decodes camera colours before update_obs (see
    ModelTemplate.update_obs), so this only has to normalise the key name and the layout."""
    image = None
    for key in aliases:
        cam = vision.get(key)
        if cam is None:
            continue
        image = cam.get("color", cam.get("rgb", cam.get("colors"))) if isinstance(cam, dict) else cam
        if image is not None:
            break
    if image is None:
        raise KeyError(f"None of {aliases} present in obs['vision'] (keys: {sorted(vision)})")
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))   # CHW -> HWC
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"camera {aliases[0]} gave an unusable array of shape {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image.astype(np.uint8))


def _resized_pil(image: np.ndarray, size_wh: tuple[int, int]) -> Image.Image:
    """Resize an RGB observation with PIL bilinear interpolation."""
    pil = Image.fromarray(image, mode="RGB")
    return pil if pil.size == size_wh else pil.resize(size_wh, Image.BILINEAR)


def _pil_to_unit(pil: Image.Image) -> torch.Tensor:
    """PIL -> [3,H,W] float in [0,1], before scaling to [-1,1]."""
    arr = np.asarray(pil, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


class HistorySlotGrid:
    """Deterministic history selection using the configured offset bins.

    Slots are ordered OLDEST -> NEWEST with fixed negative-time coordinates.
    Select each bin centre, clamp to the available episode length, and mark
    a slot invalid when its lower offset bound exceeds the episode length.
    """

    def __init__(self, offset_bins_sec, fps: float = RAW_FPS):
        bins = []
        for lo, hi in offset_bins_sec:
            lo_f = max(1, int(round(float(lo) * fps)))     # >= 1 keeps history_t < current_t
            hi_f = max(lo_f, int(round(float(hi) * fps)))
            bins.append((lo_f, hi_f))
        order = sorted(range(len(bins)), key=lambda i: bins[i][0], reverse=True)
        self.bins_frames = [bins[i] for i in order]        # oldest -> newest
        self.num_slots = len(self.bins_frames)
        self.max_lookback = max(hi for _, hi in self.bins_frames)

    def offsets(self, current_t: int) -> list[Optional[int]]:
        """-> per-slot frame distance back from `current_t`, None for an invalid slot."""
        out: list[Optional[int]] = []
        for lo_f, hi_f in self.bins_frames:
            hi_eff = min(hi_f, current_t)
            if hi_eff < lo_f:                              # episode too short for this slot
                out.append(None)
                continue
            out.append(min((lo_f + hi_f) // 2, hi_eff))
        return out


class _EnvState:
    """Per-environment rollout state. One env per entry so `eval_batch` stays honest."""

    def __init__(self, history_capacity: int, vl_history_capacity: int):
        self.head_frames: deque[torch.Tensor] = deque(maxlen=history_capacity)
        self.obs: Optional[dict] = None
        self.views: Optional[torch.Tensor] = None          # [V,3,H,W] in [0,1]
        self.vl_pil: Optional[Image.Image] = None
        # Past head frames at the VL history resolution (VL-v2), one push per env step.
        self.vl_hist_frames: deque[Image.Image] = deque(maxlen=vl_history_capacity)
        self.pending: deque[np.ndarray] = deque()
        # Track the episode step independently of the bounded frame buffer.
        self.step = -1


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        from omegaconf import OmegaConf

        _ensure_logging_visible()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg["action_type"])
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "joint":
            raise ValueError(
                f"Awomo05 was trained on joint targets; action_type={self.action_type!r} "
                "is not supported."
            )
        self.robot_info = get_robot_action_dim_info(self.env_cfg_type)
        self.expected_action_dim = sum(self.robot_info["arm_dim"]) + sum(self.robot_info["ee_dim"])
        configured = _opt_int(model_cfg.get("action_dim"))
        if configured is not None and configured != self.expected_action_dim:
            raise ValueError(
                f"action_dim mismatch for env_cfg_type={self.env_cfg_type}: deploy.yml says "
                f"{configured}, robot config says {self.expected_action_dim}."
            )

        self.local_dir = Path(str(model_cfg.get("local_dir") or DEFAULT_LOCAL_DIR)).expanduser()
        self.code_dir = _ensure_awomo_importable()
        self.weights_dir = Path(str(model_cfg.get("weights_dir") or DEFAULT_WEIGHTS_DIR)).expanduser()
        self.model_config_path = _resolve_model_config(model_cfg.get("model_config"))
        explicit_ckpt = model_cfg.get("ckpt")
        if not _is_none_like(explicit_ckpt):
            self.ckpt_path = _resolve_ckpt(self.weights_dir, explicit_ckpt)
        else:
            candidates = candidate_checkpoint_roots(
                dict(model_cfg), PLUGIN_DIR / "checkpoints", policy_dir=PLUGIN_DIR)
            ckpt_name = model_cfg.get("ckpt_name")
            if not _is_none_like(ckpt_name) and "/" not in str(ckpt_name):
                candidates += [self.weights_dir / str(ckpt_name),
                               self.weights_dir / f"{ckpt_name}.pt"]
            self.ckpt_path = next(
                (f.resolve() for root in candidates
                 for f in ([root] if root.is_file() else [root / "model.pt"])
                 if f.is_file()), None)
            if self.ckpt_path is None:
                raise FileNotFoundError(
                    "Checkpoint not found. Set ckpt explicitly or pass a valid ckpt_name; "
                    f"checked {candidates}")
        self.ckpt_load_path = _stage_checkpoint(
            self.ckpt_path, self.local_dir, _as_bool(model_cfg.get("ckpt_local_copy", True), True))
        run_cfg = OmegaConf.load(self.model_config_path)
        # Base weights the checkpoint sits on: FLUX.2 klein 4B (architecture + init), FLUX.2 autoencoder,
        for key, cfg_key, env_key in (("flux2_base_path", "flux2_model_path", "AWOMO05_FLUX2_BASE"),
                                       ("ae_path", "ae_model_path", "AWOMO05_FLUX2_AE"),
                                       ("qwen_vl_path", "vl_model_path", "AWOMO05_QWEN_VL")):
            val = model_cfg.get(key)
            if _is_none_like(val):
                val = os.environ.get(env_key)
            if _is_none_like(val):
                val = run_cfg.model.get(cfg_key)
            if _is_none_like(val):
                raise ValueError(f"`{key}` (or ${env_key}) is required: path to the base weights for {cfg_key}")
            run_cfg.model[cfg_key] = str(Path(str(val)).expanduser())
        run_cfg.model["flux2_src_path"] = None
        tgt = str(run_cfg.model.get("_target_", ""))
        if tgt.startswith("exploration.action_img_patch."):
            run_cfg.model["_target_"] = "awomo.awomo." + tgt[len("exploration.action_img_patch."):]
        self.run_cfg = run_cfg
        train_cfg = run_cfg.data.train

        # --- inference preprocessing defaults from the supplied model configuration ---
        if int(train_cfg.action_dim) != self.expected_action_dim:
            raise ValueError(
                f"run was trained with action_dim={int(train_cfg.action_dim)} but the robot needs "
                f"{self.expected_action_dim}."
            )
        self.cameras = [str(c) for c in train_cfg.cameras]
        if len(self.cameras) != len(CAMERA_ALIASES):
            raise ValueError(
                f"This plugin maps exactly {len(CAMERA_ALIASES)} cameras; run config has {self.cameras}."
            )
        for trained, aliases in zip(self.cameras, CAMERA_ALIASES):
            if trained != aliases[0]:
                raise ValueError(
                    f"camera order mismatch: run trained on {self.cameras}, plugin maps "
                    f"{[a[0] for a in CAMERA_ALIASES]}. View index drives RoPE, so refusing to guess."
                )
        if not _is_none_like(train_cfg.get("concat_multi_camera")):
            raise ValueError(
                "This plugin only supports independent views (concat_multi_camera: null); the run "
                f"used {train_cfg.concat_multi_camera!r}."
            )
        h, w = (int(v) for v in train_cfg.video_size)
        self.view_size_wh = (w, h)
        self.action_horizon = _opt_int(model_cfg.get("action_horizon")) or (int(train_cfg.num_frames) - 1)
        self.replan_steps = max(1, min(int(model_cfg.get("replan_steps", 8) or 8), self.action_horizon))
        # Convert camera frames to the colour order expected by the checkpoint.
        self.input_color_order = str(model_cfg.get("input_color_order", "rgb") or "rgb").lower()
        if self.input_color_order not in ("rgb", "bgr"):
            raise ValueError(f"input_color_order must be rgb or bgr, got {self.input_color_order!r}")
        self.num_inference_steps = (
            _opt_int(model_cfg.get("num_inference_steps"))
            or _opt_int(run_cfg.get("eval_num_inference_steps"))
            or 20
        )
        self.sigma_shift = _opt_float(model_cfg.get("sigma_shift"))
        self.seed = _opt_int(model_cfg.get("seed"))
        self.rand_device = str(model_cfg.get("rand_device") or "cpu")
        self.default_instruction = str(model_cfg.get("default_instruction") or "").strip()

        # --- history grid ---
        hist_cfg = train_cfg.get("history") or {}
        self.history_grid = None
        if _as_bool(hist_cfg.get("enabled"), False) and _as_bool(model_cfg.get("history_enabled", True), True):
            hist_fps = _opt_float(model_cfg.get("history_fps")) or float(hist_cfg.get("fps", RAW_FPS))
            self.history_grid = HistorySlotGrid(
                [tuple(float(x) for x in b) for b in hist_cfg["offset_bins_sec"]], fps=hist_fps
            )
            trained_slots = _opt_int(hist_cfg.get("max_slots"))
            if trained_slots is not None and trained_slots != self.history_grid.num_slots:
                raise ValueError(
                    f"history.max_slots={trained_slots} disagrees with len(offset_bins_sec)="
                    f"{self.history_grid.num_slots}; per-slot RoPE positions depend on it."
                )
            hist_cam = str(hist_cfg.get("camera", self.cameras[0]))
            if hist_cam != self.cameras[0]:
                raise ValueError(
                    f"history camera {hist_cam!r} is not the first view {self.cameras[0]!r}; this "
                    "plugin only buffers the first view."
                )

        # --- VL image size: MUST equal data.train.vlm.image_size (see _vl_infer_images) ---
        vlm_cfg = train_cfg.get("vlm") or {}
        self.vl_enabled = _as_bool(vlm_cfg.get("enabled"), False)
        self.vl_image_size_wh = tuple(int(v) for v in (vlm_cfg.get("image_size") or (448, 448)))
        vh = vlm_cfg.get("history") or {}
        self.vl_history_enabled = self.vl_enabled and _as_bool(vh.get("enabled", True), True)
        self.vl_history_size_wh = tuple(int(v) for v in (vh.get("image_size") or (448, 448)))

        device = str(model_cfg.get("device") or "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU (this will be unusably slow).")
            device = "cpu"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            str(model_cfg.get("mixed_precision") or "bf16").lower().replace("bfloat16", "bf16")
        ]

        self.model = self._build_model(run_cfg, device=device, dtype=dtype)
        self.normalizer = self._build_normalizer(train_cfg, model_cfg.get("dataset_stats_path"))
        self._prompt_template = self._load_prompt_template()

        enc = getattr(self.model, "vl_encoder", None)
        decode_every = _opt_int(model_cfg.get("vl_decode_every"))
        if decode_every is not None and enc is not None:
            enc.decode_every = decode_every
        # Frame offsets (oldest -> newest) the VL history is sampled at: the model's own grid.
        self.vl_history_offsets = (
            list(enc.history_offsets()) if (enc is not None and self.vl_history_enabled) else [])

        self._check_import_provenance()

        capacity = (self.history_grid.max_lookback + 1) if self.history_grid else 1
        self._history_capacity = capacity
        self._vl_history_capacity = (max(self.vl_history_offsets) + 1) if self.vl_history_offsets else 1
        self._envs: dict[int, _EnvState] = {}
        self._latest_env_idx_list = [0]

        logger.info(
            "Awomo05 ready | ckpt=%s | code=%s | views=%s@%s | horizon=%d replan=%d "
            "steps=%d | history=%s | vl=%s@%s | vl_history=%s",
            self.ckpt_path.name, self.code_dir,
            self.cameras, self.view_size_wh,
            self.action_horizon, self.replan_steps, self.num_inference_steps,
            f"{self.history_grid.num_slots} slots/{capacity} frames" if self.history_grid else "off",
            "on" if (self.vl_enabled and enc is not None) else "off", self.vl_image_size_wh,
            (f"{len(self.vl_history_offsets)} frames@{self.vl_history_size_wh} offsets "
             f"{self.vl_history_offsets[-1]}..{self.vl_history_offsets[0]}") if self.vl_history_offsets else "off",
        )

    def _check_import_provenance(self) -> None:
        """The model code must be the vendored `awomo` package next to this file, nothing else."""
        import awomo
        resolved = Path(awomo.__file__).resolve().parent
        if resolved != (PLUGIN_DIR / "awomo").resolve():
            raise ImportError(f"`awomo` resolved from {resolved}, expected {PLUGIN_DIR / 'awomo'}")
        logger.info("awomo (vendored inference code) imported from %s", resolved)

    # ------------------------------------------------------------------ build
    def _build_model(self, run_cfg, *, device: str, dtype: torch.dtype):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(OmegaConf.to_container(run_cfg.model, resolve=True))
        model = instantiate(cfg, model_dtype=dtype, device=device)
        model.load_checkpoint(str(self.ckpt_load_path))
        model = model.to(device).eval()
        # A text-layout mismatch changes conditioning, so refuse the checkpoint.
        enc = getattr(model, "vl_encoder", None)
        if enc is not None:
            logger.info(
                "VL encoder: text_layout=%s (checkpoint: %s) image_size=%s history=%dx%s@%.2fs "
                "decode_every=%d max_new_tokens=%d",
                enc.text_layout, enc.loaded_text_layout, tuple(enc.image_size),
                int(enc.history_num_slots), tuple(enc.history_image_size), float(enc.history_period_s),
                int(enc.decode_every), int(enc.max_new_tokens),
            )
            if enc.loaded_text_layout != enc.text_layout:
                raise ValueError(
                    f"checkpoint text layout {enc.loaded_text_layout!r} != code layout "
                    f"{enc.text_layout!r}: this checkpoint was trained with a different DiT "
                    "conditioning and cannot be used by this code.")
            if self.vl_history_enabled and tuple(int(v) for v in enc.history_image_size) != self.vl_history_size_wh:
                raise ValueError(
                    f"vl_ar.history_image_size={tuple(enc.history_image_size)} != "
                    f"data.train.vlm.history.image_size={self.vl_history_size_wh}")
            if tuple(int(v) for v in enc.image_size) != self.vl_image_size_wh:
                raise ValueError(
                    f"vl_ar.image_size={tuple(enc.image_size)} != data.train.vlm.image_size="
                    f"{self.vl_image_size_wh}; the prompt codec expands image placeholders against "
                    "the former and the frame is processed at the latter."
                )
        return model

    def _build_normalizer(self, train_cfg, override_stats):
        from omegaconf import DictConfig, OmegaConf

        from awomo.awomo.datasets.lerobot.utils.normalizer import (
            LinearNormalizer,
            load_dataset_stats_from_json,
        )

        stats_path = (
            str(override_stats) if not _is_none_like(override_stats)
            else str(train_cfg.get("pretrained_norm_stats") or (self.weights_dir / "dataset_stats.json"))
        )
        if not Path(stats_path).is_file() and (self.weights_dir / "dataset_stats.json").is_file():
            stats_path = str(self.weights_dir / "dataset_stats.json")
        if not Path(stats_path).is_file():
            raise FileNotFoundError(f"dataset stats not found: {stats_path}")
        dim = self.expected_action_dim
        shape_meta = {
            "action": [{"key": "default", "raw_shape": dim, "shape": dim}],
            "state": [{"key": "default", "raw_shape": dim, "shape": dim}],
        }
        exception_mode = train_cfg.get("norm_exception_mode")
        if isinstance(exception_mode, DictConfig):
            exception_mode = OmegaConf.to_container(exception_mode, resolve=True)
        return LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=_as_bool(train_cfg.get("use_stepwise_action_norm"), False),
            default_mode=str(train_cfg.norm_default_mode),
            exception_mode=exception_mode,
            stats=load_dataset_stats_from_json(stats_path),
        )

    @staticmethod
    def _load_prompt_template() -> str:
        """Instruction wrapper for the non-VL text encoder path."""
        return DEFAULT_PROMPT

    # ------------------------------------------------------- ModelTemplate API
    def reset(self):
        """New episode: drop the frame history, the action queue and the decode cadence."""
        self._envs.clear()
        self._latest_env_idx_list = [0]
        reset_ar = getattr(self.model, "reset_ar_state", None)
        if callable(reset_ar):
            reset_ar()
        logger.info("Awomo05 reset (history + action queue + VL history cleared)")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", i) for i, obs in enumerate(obs_list)]
        for i, obs in enumerate(obs_list):
            env_idx = obs.get("env_idx", i)
            state = self._envs.get(env_idx)
            if state is None:
                state = self._envs[env_idx] = _EnvState(self._history_capacity, self._vl_history_capacity)
            vision = obs.get("vision") or {}
            raw = [_extract_rgb(vision, aliases) for aliases in CAMERA_ALIASES]
            if self.input_color_order == "bgr":
                raw = [np.ascontiguousarray(img[:, :, ::-1]) for img in raw]
            views = torch.stack([_pil_to_unit(_resized_pil(img, self.view_size_wh)) for img in raw])
            state.obs = obs
            state.views = views
            state.vl_pil = _resized_pil(raw[0], self.vl_image_size_wh)
            if self.vl_history_offsets:
                state.vl_hist_frames.append(_resized_pil(raw[0], self.vl_history_size_wh))
            # One buffer push per env step: the harness calls update_obs after every action, so
            # `state.step` counts 25 Hz control steps, which is the grid the history slots use.
            state.head_frames.append(views[0])
            state.step += 1

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        out = []
        for env_idx in env_idx_list:
            state = self._envs.get(env_idx)
            if state is None or state.views is None:
                raise RuntimeError(f"No observation buffered for env_idx={env_idx}; call update_obs first.")
            chunk = self._infer_chunk(state)
            n = min(self.replan_steps, chunk.shape[0])
            executed = chunk[:n]
            out.append([
                unpack_robot_state(
                    np.asarray(executed[i], dtype=np.float32),
                    self.action_type,
                    self.robot_info,
                    source_type="obs",
                )
                for i in range(n)
            ])
        return out

    # --------------------------------------------------------------- internals
    def _build_history(self, state: _EnvState):
        """-> ([1,N,3,H,W] in [-1,1], [1,N] bool) or (None, None) when history is off."""
        if self.history_grid is None:
            return None, None
        n = self.history_grid.num_slots
        w, h = self.view_size_wh
        frames = torch.zeros(n, 3, h, w)
        valid = torch.zeros(n, dtype=torch.bool)
        buf = state.head_frames
        for slot, d in enumerate(self.history_grid.offsets(state.step)):
            # `d >= len(buf)` can only happen while the episode is shorter than the buffer; once it
            # is full the grid's deepest bin is exactly its capacity, so nothing is silently lost.
            if d is None or d >= len(buf):
                continue                          # invalid slot stays zero AND masked out
            frames[slot] = buf[len(buf) - 1 - d]
            valid[slot] = True
        frames = frames * 2.0 - 1.0               # same [-1,1] VAE range as the ref views
        return frames.unsqueeze(0), valid.unsqueeze(0)

    def _build_vl_history(self, state: _EnvState) -> list:
        if not self.vl_history_offsets:
            return []
        buf = state.vl_hist_frames
        out = []
        for d in self.vl_history_offsets:
            if d > state.step or d >= len(buf):
                continue
            out.append(buf[len(buf) - 1 - d])
        return out

    def _proprio(self, obs) -> torch.Tensor:
        packed = pack_robot_state(obs, self.action_type, self.robot_info, source_type="obs")
        packed = np.asarray(packed, dtype=np.float32).reshape(-1)
        if packed.shape[0] != self.expected_action_dim:
            raise ValueError(f"packed state has {packed.shape[0]} dims, expected {self.expected_action_dim}")
        batch = {"state": {"default": torch.from_numpy(packed).unsqueeze(0)}}
        return self.normalizer.forward(batch)["state"]["default"].float()

    def _instruction(self, obs) -> str:
        raw = _decode_text(obs.get("instruction", obs.get("task_instruction")), self.default_instruction)
        if not raw:
            raise ValueError(
                "obs carries no `instruction` and no `default_instruction` is configured"
            )
        if self.vl_enabled:
            # The VL codec supplies its own instruction wrapper. Pass the raw instruction
            # once; DEFAULT_PROMPT is used only by the non-VL text encoder path.
            return raw.strip()
        return self._prompt_template.format(task=raw)

    @torch.no_grad()
    def _infer_chunk(self, state: _EnvState) -> np.ndarray:
        views = state.views.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)
        views = views * 2.0 - 1.0                                  # [1,V,3,H,W] in [-1,1]
        hist_frames, hist_valid = self._build_history(state)
        if hist_frames is not None:
            hist_frames = hist_frames.to(device=self.model.device, dtype=self.model.torch_dtype)
            hist_valid = hist_valid.to(device=self.model.device)
        proprio = self._proprio(state.obs)
        vl_history = self._build_vl_history(state)

        pred = self.model.infer_action_flux2(
            prompt=self._instruction(state.obs),
            input_image=views,
            action_horizon=self.action_horizon,
            proprio=proprio,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            seed=self.seed,
            rand_device=self.rand_device,
            history_frames=hist_frames,
            history_valid_mask=hist_valid,
            vl_images=[state.vl_pil],
            vl_history=[vl_history],
        )

        action = pred["action"]                                     # [T,14], normalized
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalized = action.to(dtype=torch.float32, device="cpu")
        denorm = self.normalizer.normalizers["action"]["default"].backward(normalized)
        return denorm[0].numpy()
