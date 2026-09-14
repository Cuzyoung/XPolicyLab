"""CLIP/UNet UMI Diffusion Policy in the shared XPolicyLab server."""

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .artifact_identity import load_artifact
from .transforms import absolute_actions, build_observation, relative_condition


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        import hydra
        import torch

        self.model_cfg = dict(model_cfg)
        if model_cfg.get("action_type") != "ee":
            raise ValueError("UMI_DP supports action_type=ee")
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if self.robot_action_dim_info != {"arm_dim": [7, 7], "ee_dim": [1, 1]}:
            raise ValueError("UMI_DP requires two 7-joint arms with one aperture each")
        self.cfg, state, self.shape_meta, self._metadata = load_artifact(model_cfg)
        self.device = str(model_cfg.get("device", "cuda:0"))
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        self.policy.load_state_dict(state, strict=True)
        del state
        self.policy.to(self.device).eval()
        self.model = self.policy
        self.horizon = self._metadata["action_horizon"]
        if self.horizon != self.policy.action_horizon:
            raise ValueError("Task and policy action horizons disagree")
        self.tolerance_s = float(model_cfg.get("observation_tolerance_s", 0.04))
        if not 0 < self.tolerance_s < self._metadata["observation_period_s"]:
            raise ValueError("Observation tolerance must be positive and smaller than the period")
        from .rtc_guidance import RtcGuidance

        self.guidance = RtcGuidance(self.policy, mode=self._metadata["rtc_guidance"])
        seed = model_cfg.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        self.reset()

    def sampling_modes(self):
        return ["default", "rtc"]

    def runtime_metadata(self):
        return dict(self._metadata)

    def reset(self):
        self._observations = {}
        self._order = []
        self.last_native_action = None

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        observations, order = {}, []
        for obs in obs_list:
            index = int(obs.get("env_idx", 0))
            if index in observations:
                raise ValueError("Duplicate env_idx in observation batch")
            observations[index] = build_observation(
                obs, self.shape_meta, self._metadata["observation_period_s"], self.tolerance_s
            )
            order.append(index)
        if not order:
            raise ValueError("Empty observation batch")
        self._observations, self._order = observations, order

    def _predict(self, index, sampling=None):
        import torch

        if index not in self._observations:
            raise ValueError(
                "No observation for this environment; update_obs is required after reset"
            )
        tensors, reference = self._observations[index]
        batch = {
            key: torch.from_numpy(value).unsqueeze(0).to(self.device)
            for key, value in tensors.items()
        }
        if sampling is None:
            with torch.inference_mode():
                output = self.policy.predict_action(batch)["action"][0].cpu().numpy()
        else:
            weights = np.asarray(sampling["condition_weights"], dtype=np.float32)
            if (
                weights.shape != (self.horizon,)
                or not np.isfinite(weights).all()
                or np.any((weights < 0) | (weights > 1))
            ):
                raise ValueError("RTC weights must match the checkpoint horizon and lie in [0, 1]")
            condition = relative_condition(sampling["action_condition"], reference, weights)
            with (
                torch.no_grad(),
                self.guidance.condition(condition, weights, beta=sampling.get("beta", 5.0)),
            ):
                output = self.policy.predict_action(batch)["action"][0].cpu().numpy()
        if output.shape != (self.horizon, self.policy.action_dim) or not np.isfinite(output).all():
            raise ValueError("Invalid policy output")
        self.last_native_action = output.copy()
        return absolute_actions(output, reference)

    def get_action(self):
        if len(self._order) != 1:
            raise ValueError("get_action requires exactly one observation")
        return self._predict(self._order[0])

    def get_action_rtc(self, sampling):
        if len(self._order) != 1 or sampling.get("mode") != "rtc":
            raise ValueError("RTC requires one observation and mode=rtc")
        return self._predict(self._order[0], sampling)

    def get_action_batch(self, env_idx_list=None):
        indices = (
            self._order if env_idx_list is None else np.asarray(env_idx_list).reshape(-1).tolist()
        )
        if not indices:
            raise ValueError("No observations available")
        return [self._predict(int(index)) for index in indices]
