"""XPolicyLab adapter for StarVLA inference, batching and episode state."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    unpack_robot_state,
)

from .eef import EefCodec
from .observations import ObservationEncoder
from .runtime_config import (
    parse_bool,
    resolve_checkpoint_framework,
    resolve_include_state,
    validate_server_runtime_contract,
)
from .sampling import SamplingAdapter

_CUR_DIR = Path(__file__).resolve().parent


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self._configure_representation()
        if self.model_backend == "inprocess":
            self.model_cfg["checkpoint_path"] = str(self._checkpoint_file())
        self.include_state = resolve_include_state(
            self.model_cfg.get("include_state", "auto"), self.model_cfg.get("checkpoint_path")
        )
        self.observation_encoder = ObservationEncoder(
            self.model_cfg.get("camera_names"),
            self.model_cfg.get("image_size", [224, 224]),
            include_state=self.include_state,
            state_type=self.state_type,
            state_dimensions=self.state_dimensions,
            state_indices=self.state_indices,
        )
        self.camera_names = self.observation_encoder.camera_names
        self.client = None
        self.policy = None
        self._load_runtime()
        self._configure_inference()
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.step_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]
        self.samplers = SamplingAdapter(self)

    def _configure_representation(self):
        """Resolve action/state contracts before importing model dependencies."""
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type not in {"joint", "ee"}:
            raise ValueError("starVLA action_type must be joint or ee.")

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("starVLA requires env_cfg_type.")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.eef = (
            EefCodec(self.model_cfg.get("eef") or {}, self.robot_action_dim_info)
            if self.action_type == "ee"
            else None
        )
        if self.action_type == "joint" and self.model_cfg.get("eef"):
            raise ValueError("eef options require action_type=ee")
        self.action_dim = (
            self.eef.native_dim
            if self.eef
            else sum(self.robot_action_dim_info["arm_dim"])
            + sum(self.robot_action_dim_info["ee_dim"])
        )
        self.output_dimensions = self.eef.dimensions if self.eef else self.robot_action_dim_info
        self.state_type = self.model_cfg.get("state_type", "joint")
        if self.state_type not in {"joint", "ee"}:
            raise ValueError("state_type must be joint or ee (xyz, quaternion wxyz)")
        self.state_dimensions = dict(self.robot_action_dim_info)
        if self.state_type == "ee":
            self.state_dimensions["arm_dim"] = [7] * len(self.state_dimensions["arm_dim"])
        self.state_dim = sum(self.state_dimensions["arm_dim"]) + sum(
            self.state_dimensions["ee_dim"]
        )
        self.model_backend = self.model_cfg.get("model_backend", "websocket")
        if self.model_backend not in {"websocket", "inprocess"}:
            raise ValueError("model_backend must be 'websocket' or 'inprocess'.")
        self.action_output = self.model_cfg.get("action_output", "step")
        if self.action_output not in {"step", "chunk"}:
            raise ValueError("action_output must be 'step' or 'chunk'.")
        self.action_indices = self._permutation("action_indices", self.action_dim)
        self.state_indices = self._permutation("state_indices", self.state_dim)
        self.unnorm_key = self.model_cfg.get("unnorm_key", "arx_x5")
        if self.unnorm_key in (None, "", "null", "None", "auto"):
            self.unnorm_key = None
        self.require_runtime_contract = parse_bool(
            self.model_cfg.get("require_runtime_contract", True)
        )
        if self.model_backend == "inprocess" and not self.require_runtime_contract:
            raise ValueError("inprocess inference requires runtime contract validation.")

    def _load_runtime(self):
        """Load the selected backend; never substitute one after a failure."""
        starvla_root = Path(self.model_cfg.get("starvla_root") or "source_starvla").expanduser()
        if not starvla_root.is_absolute():
            starvla_root = _CUR_DIR / starvla_root
        if (
            self.model_backend == "inprocess"
            and starvla_root.resolve() != (_CUR_DIR / "source_starvla").resolve()
        ):
            raise ValueError("inprocess inference requires the vendored source_starvla runtime.")
        if str(starvla_root) not in sys.path:
            sys.path.insert(0, str(starvla_root))
        if self.model_backend == "inprocess":
            checkpoint = self.model_cfg["checkpoint_path"]
            from deployment.model_server.policy_wrapper import PolicyServerWrapper

            self.policy = PolicyServerWrapper(
                ckpt_path=str(checkpoint),
                device=self.model_cfg.get("device", "cuda"),
                use_bf16=parse_bool(self.model_cfg.get("use_bf16", True)),
                unnorm_key=self.unnorm_key,
            )
            server_meta = self.policy.metadata
        else:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

            self.client = WebsocketClientPolicy(
                self.model_cfg.get("starvla_server_host", "127.0.0.1"),
                int(self.model_cfg.get("starvla_server_port", 5694)),
            )
            server_meta = self.client.get_server_metadata()
        self._server_metadata = dict(server_meta)

    def _configure_inference(self):
        """Validate the loaded checkpoint and select step or chunk scheduling."""
        server_meta = self._server_metadata
        self.action_chunk_size = int(server_meta["action_chunk_size"])
        if self.action_chunk_size <= 0:
            raise ValueError("StarVLA action_chunk_size must be positive.")
        execute_horizon = self.model_cfg.get("execute_horizon", self.action_chunk_size)
        if self.action_output == "chunk" or execute_horizon in (None, "", "null", "None"):
            execute_horizon = self.action_chunk_size
        self.execute_horizon = int(execute_horizon)
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self.execute_horizon}.")
        if self.execute_horizon > self.action_chunk_size:
            raise ValueError(
                f"execute_horizon={self.execute_horizon} exceeds "
                f"action_chunk_size={self.action_chunk_size}."
            )
        self.use_ddim = parse_bool(self.model_cfg.get("use_ddim", True))
        self.num_ddim_steps = int(self.model_cfg.get("num_ddim_steps", 10))
        if self.require_runtime_contract:
            native_state_dim = server_meta.get("runtime_contract", {}).get("state_dim")
            if (
                self.include_state
                and native_state_dim is not None
                and native_state_dim != self.state_dim
            ):
                raise ValueError(
                    f"Checkpoint state_dim={native_state_dim} does not match "
                    f"configured {self.state_type} width {self.state_dim}"
                )
            expected_framework = resolve_checkpoint_framework(self.model_cfg.get("checkpoint_path"))
            expected_pi_v3_forward = self.model_cfg.get("required_pi_v3_forward")
            if expected_pi_v3_forward in (None, "", "auto", "none", "null", "None"):
                expected_pi_v3_forward = None
            else:
                expected_pi_v3_forward = str(expected_pi_v3_forward)
            validate_server_runtime_contract(
                server_meta,
                include_state=self.include_state,
                action_dim=self.action_dim,
                unnorm_key=self.unnorm_key,
                expected_framework=expected_framework,
                expected_pi_v3_forward=expected_pi_v3_forward,
            )

    def _permutation(self, name: str, width: int) -> np.ndarray:
        indices = self.model_cfg.get(name)
        if indices is None:
            return np.arange(width)
        if (
            not isinstance(indices, (list, tuple))
            or any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(width))
        ):
            raise ValueError(f"{name} must be a permutation of 0..{width - 1}.")
        return np.asarray(indices)

    def _checkpoint_file(self) -> Path:
        # Preserve the final symlink: StarVLA locates sidecars beside the run,
        # not beside a Hugging Face blob-cache target.
        candidates = candidate_checkpoint_roots(
            self.model_cfg, _CUR_DIR / "checkpoints", policy_dir=_CUR_DIR, resolve=False
        )
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix in {".pt", ".safetensors"}:
                return candidate.absolute()
        raise FileNotFoundError(
            "inprocess StarVLA requires an explicit .pt or .safetensors checkpoint."
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Expose actual model identity to the shared server's HELLO handshake."""
        return {
            "policy_name": "starVLA",
            "framework": self._server_metadata.get("framework"),
            "checkpoint_path": self._server_metadata.get("ckpt_path"),
            "env_cfg_type": self.env_cfg_type,
            "action_type": self.action_type,
            "action_dim": self.action_dim,
            "action_horizon": self.action_chunk_size if self.action_output == "chunk" else 1,
            "action_output": self.action_output,
            "model_backend": self.model_backend,
            "unnorm_key": self.unnorm_key or self._server_metadata.get("default_unnorm_key"),
            "include_state": self.include_state,
            "action_indices": self.action_indices.tolist(),
            "state_indices": self.state_indices.tolist(),
            "state_type": self.state_type,
            "camera_names": self.camera_names,
            "action_semantics": self.eef.semantics if self.eef else "absolute_joint_position",
            "eef": self.eef.options if self.eef else None,
            "sampling_modes": self.sampling_modes(),
            "pi_v3_forward": self._server_metadata.get("runtime_contract", {}).get("pi_v3_forward"),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            self.obs_by_env[env_idx] = self.observation_encoder(obs)

    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        if env_idx not in self.obs_by_env:
            raise AssertionError("update_obs must be called before get_action.")

        vla_input = {
            "examples": [self.obs_by_env[env_idx]],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        if self.unnorm_key is not None:
            vla_input["unnorm_key"] = self.unnorm_key
        if self.policy is not None:
            data = self.policy.predict_action(**vla_input)
        else:
            response = self.client.predict_action(vla_input)
            if not response.get("ok", False):
                raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
            data = response["data"]
        actions = np.asarray(data["actions"], dtype=np.float32)
        expected = (1, self.action_chunk_size, self.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(
                f"StarVLA actions must be finite with shape {expected}, got {actions.shape}."
            )
        return actions[0][:, self.action_indices]

    def _next_action_vector(self, env_idx: int) -> np.ndarray:
        step = self.step_by_env.get(env_idx, 0)
        chunk = self.action_chunks_by_env.get(env_idx)
        if chunk is None or step % self.execute_horizon == 0:
            chunk = self._infer_chunk(env_idx)
            self.action_chunks_by_env[env_idx] = chunk

        action_idx = step % self.execute_horizon
        self.step_by_env[env_idx] = step + 1
        return chunk[action_idx]

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        chunks = []
        for env_idx in env_idx_list:
            actions = (
                self._infer_chunk(int(env_idx))
                if self.action_output == "chunk"
                else self._next_action_vector(int(env_idx))[None, :]
            )
            if self.eef:
                actions = self.eef.convert(actions)
            chunks.append(
                unpack_robot_state(
                    actions, self.action_type, self.output_dimensions, source_type="obs"
                )
            )
        return chunks

    def reset(self):
        self.obs_by_env.clear()
        self.action_chunks_by_env.clear()
        self.step_by_env.clear()
        self._latest_env_idx_list = [0]
        self.samplers.reset()

    def sampling_modes(self):
        return list(self.samplers.modes)

    def get_action_rtc(self, sampling):
        return self.samplers.infer("rtc", sampling)

    def get_action_paint(self, sampling):
        return self.samplers.infer("paint", sampling)

    def get_action_aac(self, sampling):
        return self.samplers.infer("aac", sampling)

    def get_action_autohorizon(self, sampling):
        return self.samplers.infer("autohorizon", sampling)

    def get_action_dvac(self, sampling):
        return self.samplers.infer("dvac", sampling)
