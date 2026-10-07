import time
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

_POLICY_DIR = Path(__file__).resolve().parent

# checkpoint image key -> simulator camera names
CAMERA_ALIASES = {
    "observation.images.cam_high": ("cam_high", "cam_head"),
    "observation.images.cam_left_wrist": ("cam_left_wrist",),
    "observation.images.cam_right_wrist": ("cam_right_wrist",),
}
CAMERA_LABELS = {
    "cam_high": "Head camera",
    "cam_left_wrist": "Left wrist camera",
    "cam_right_wrist": "Right wrist camera",
}
SUPPORTED_ENV_CFG_TYPES = ("arx_x5",)


def _import_runtime():
    # Imported lazily so the env client, which also imports this package, never loads torch.
    from .simplememvla.compat import apply_fla_torch_compat

    apply_fla_torch_compat()

    import torch
    from transformers import AutoProcessor

    from .simplememvla.policy import SimpleMemVLAPolicy, get_vla

    return torch, AutoProcessor, SimpleMemVLAPolicy, get_vla


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.action_type != "joint":
            raise ValueError(f"SimpleMemVLA supports action_type=joint only, got {self.action_type!r}")
        if self.env_cfg_type not in SUPPORTED_ENV_CFG_TYPES:
            raise ValueError(f"SimpleMemVLA supports env_cfg_type={SUPPORTED_ENV_CFG_TYPES}, got {self.env_cfg_type!r}")

        dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dim, ee_dim = list(dim_info["arm_dim"]), list(dim_info["ee_dim"])
        if len(arm_dim) != 2 or len(ee_dim) != 2:
            raise ValueError(f"SimpleMemVLA needs a dual-arm robot, got {dim_info}")
        # (key, start, end, is_gripper) in the flat state/action vector
        self._layout = []
        offset = 0
        for side, arm, ee in (("left", arm_dim[0], ee_dim[0]), ("right", arm_dim[1], ee_dim[1])):
            self._layout.append((f"{side}_arm_joint_state", offset, offset + arm, False))
            offset += arm
            self._layout.append((f"{side}_ee_joint_state", offset, offset + ee, True))
            offset += ee
        self._vector_dim = offset

        self._torch, AutoProcessor, self._policy_cls, get_vla = _import_runtime()
        checkpoint = resolve_checkpoint_root(
            self.model_cfg, _POLICY_DIR / "checkpoints", policy_dir=_POLICY_DIR
        )
        vla, unnormalize_action, normalize_state = get_vla(
            str(checkpoint),
            "bfloat16",
            attn_implementation=self.model_cfg.get("attn_implementation", "flash_attention_2"),
        )
        processor = AutoProcessor.from_pretrained(
            str(checkpoint), padding_side="right", model_max_length=8192
        )
        cfg = vla.config
        if int(cfg.action_dim) != self._vector_dim or int(cfg.state_dim) != self._vector_dim:
            raise ValueError(
                f"checkpoint action_dim={cfg.action_dim}, state_dim={cfg.state_dim} do not "
                f"match {self.env_cfg_type} ({self._vector_dim})"
            )
        self.execute_horizon = int(self.model_cfg.get("execute_horizon", 24))
        if not 1 <= self.execute_horizon <= int(cfg.action_horizon):
            raise ValueError(f"execute_horizon must be in [1, {cfg.action_horizon}]")
        self._policy_kwargs = dict(
            vla=vla,
            processor=processor,
            camera_labels=CAMERA_LABELS,
            unnormalize_action=unnormalize_action,
            normalize_state=normalize_state,
        )

        self._policies = {}
        self._instruction = {}
        self._state = {}
        self._seed_base = int(self.model_cfg.get("seed") or 0)
        self._episodes_seen = 0

        if bool(self.model_cfg.get("warmup", True)):
            self._warmup()
        print(f"[SimpleMemVLA] ready: {checkpoint}", flush=True)

    def _warmup(self):
        # The first forward pays kernel autotuning; do it before the server accepts requests.
        policy = self._policy_cls(**self._policy_kwargs)
        rng = np.random.default_rng(0)
        policy.observe({
            key: rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
            for key in policy.image_keys
        })
        t0 = time.time()
        policy.predict("warm up", state=np.zeros(self._vector_dim, dtype=np.float32))
        print(f"[SimpleMemVLA] warmup done in {time.time() - t0:.1f}s", flush=True)

    def _policy(self, env_idx):
        if env_idx not in self._policies:
            self._policies[env_idx] = self._policy_cls(**self._policy_kwargs)
        return self._policies[env_idx]

    @staticmethod
    def _frame(obs, image_key):
        vision = obs["vision"]
        for cam in CAMERA_ALIASES[image_key]:
            entry = vision.get(cam)
            if entry is None:
                continue
            color = np.asarray(entry["color"] if isinstance(entry, dict) else entry)
            if color.ndim != 3 or color.shape[-1] < 3:
                raise ValueError(f"camera {cam!r}: expected an HWC RGB image, got shape {color.shape}")
            return np.ascontiguousarray(color[:, :, :3], dtype=np.uint8)
        raise KeyError(f"none of {CAMERA_ALIASES[image_key]} in obs['vision'] (have {sorted(vision)})")

    def _state_vec(self, obs):
        parts = []
        for key, start, end, _ in self._layout:
            arr = np.asarray(obs["state"][key], dtype=np.float32).reshape(-1)
            if arr.shape[0] != end - start:
                raise ValueError(f"{key}: expected {end - start} dims, got {arr.shape[0]}")
            parts.append(arr)
        return np.concatenate(parts)

    @staticmethod
    def _instruction_text(obs):
        text = obs.get("instruction")
        if text is None and obs.get("instructions"):
            text = obs["instructions"][0]
        if not text or not str(text).strip():
            raise KeyError("obs carries no instruction")
        return str(text).strip()

    def _ingest(self, obs):
        env_idx = int(obs.get("env_idx", 0))
        policy = self._policy(env_idx)
        policy.observe({key: self._frame(obs, key) for key in policy.image_keys})
        self._state[env_idx] = self._state_vec(obs)
        self._instruction[env_idx] = self._instruction_text(obs)

    def update_obs(self, obs):
        self._ingest(obs)

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self._ingest(obs)

    def _predict(self, env_idx):
        if env_idx not in self._state:
            raise RuntimeError(f"no observation for env {env_idx}")
        actions, _ = self._policies[env_idx].predict(
            self._instruction[env_idx], state=self._state[env_idx]
        )
        chunk = []
        for vec in actions[: self.execute_horizon]:
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            chunk.append({
                key: np.clip(vec[start:end], 0.0, 1.0) if is_gripper else vec[start:end]
                for key, start, end, is_gripper in self._layout
            })
        return chunk

    def get_action(self):
        return self._predict(0)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = sorted(self._state)
        return [self._predict(int(env_idx)) for env_idx in env_idx_list]

    def reset(self):
        for policy in self._policies.values():
            policy.reset()
        self._instruction.clear()
        self._state.clear()
        self._torch.manual_seed(self._seed_base + self._episodes_seen)
        self._episodes_seen += 1
