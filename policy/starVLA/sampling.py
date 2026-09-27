"""Shared-server sampling contracts; denoising stays inside the StarVLA head."""

import math
from collections import deque

import numpy as np

from XPolicyLab.utils.process_data import unpack_robot_state


class SamplingAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        requested = adapter.model_cfg.get("enabled_sampling_modes", [])
        if not isinstance(requested, list) or any(not isinstance(mode, str) for mode in requested):
            raise ValueError("enabled_sampling_modes must be a list")
        available = adapter._server_metadata.get("sampling_modes", ["default"])
        if set(requested) - set(available):
            raise ValueError(
                f"Checkpoint sampler does not support {sorted(set(requested) - set(available))}"
            )
        if requested and (
            adapter.model_backend != "inprocess"
            or adapter.action_output != "chunk"
            or adapter.action_type != "joint"
        ):
            raise ValueError(
                "Specialized sampling requires inprocess joint chunks; "
                "EEF conditioning is not implemented"
            )
        self.modes = ["default", *sorted(set(requested) - {"default"})]
        self.history = deque()
        self.window_size = None

    def reset(self):
        self.history.clear()
        self.window_size = None

    def infer(self, mode, sampling):
        model = self.adapter
        if mode not in self.modes:
            raise ValueError(f"StarVLA sampling mode {mode!r} is not enabled for this checkpoint")
        if len(model._latest_env_idx_list) != 1:
            raise ValueError("Specialized sampling requires exactly one observation")
        env = model._latest_env_idx_list[0]
        if env not in model.obs_by_env:
            raise AssertionError("update_obs must be called before get_action")
        horizon, width = model.action_chunk_size, model.action_dim
        parameters = self._parameters(mode, sampling)
        samples = parameters.get("num_samples", 1)
        data = model.policy.predict_action(
            examples=[model.obs_by_env[env]],
            unnorm_key=model.unnorm_key,
            do_sample=False,
            sampling=parameters,
        )
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape != (samples, horizon, width) or not np.isfinite(actions).all():
            raise ValueError(f"Sampler actions must be finite {(samples, horizon, width)}")
        chunks = [
            unpack_robot_state(rows[:, model.action_indices], "joint", model.output_dimensions)
            for rows in actions
        ]
        if mode == "aac":
            return {"actions": chunks}
        if mode == "rtc":
            return chunks[0]
        metadata = dict(data[mode])
        if mode == "dvac":
            metadata = self._calibrate_dvac(metadata, sampling)
        return {"actions": chunks[0], mode: metadata}

    def _parameters(self, mode, sampling):
        """Validate wire settings and map joint conditions to native action order."""
        model = self.adapter
        horizon, width = model.action_chunk_size, model.action_dim
        parameters = {"mode": mode}
        if mode == "rtc":
            condition = np.asarray(sampling["action_condition"], dtype=np.float32)
            weights = np.asarray(sampling["condition_weights"], dtype=np.float32)
            beta = float(sampling["beta"])
            if condition.shape != (horizon, width) or not np.isfinite(condition).all():
                raise ValueError(f"RTC condition must be finite {(horizon, width)}")
            if (
                weights.shape != (horizon,)
                or not np.isfinite(weights).all()
                or np.any((weights < 0) | (weights > 1))
            ):
                raise ValueError("RTC weights must be finite horizon values in [0, 1]")
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError("RTC beta must be finite and positive")
            parameters.update(
                action_condition=condition[:, np.argsort(model.action_indices)],
                condition_weights=weights,
                beta=beta,
            )
        elif mode == "paint":
            delay = sampling["delay_steps"]
            prefix = np.asarray(sampling["action_prefix"], dtype=np.float32)
            if type(delay) is not int or not 0 < delay < horizon:
                raise ValueError("PAINT requires 0 < delay_steps < horizon")
            if prefix.shape != (delay, width) or not np.isfinite(prefix).all():
                raise ValueError(
                    "PAINT prefix must be finite with the declared delay and action width"
                )
            condition = np.zeros((horizon, width), dtype=np.float32)
            condition[:delay] = prefix[:, np.argsort(model.action_indices)]
            parameters.update(action_condition=condition, delay_steps=delay)
        elif mode == "aac":
            samples = sampling.get("num_samples", 20)
            if type(samples) is not int or samples <= 1:
                raise ValueError("AAC num_samples must be an integer greater than one")
            parameters["num_samples"] = samples
        elif mode == "autohorizon":
            if set(sampling) - {"mode"}:
                raise ValueError("AutoHorizon does not accept sampling overrides")
        elif mode == "dvac":
            allowed = {
                "mode",
                "alpha",
                "tail_steps",
                "rolling_window_size",
                "min_execution_steps",
                "max_execution_steps",
            }
            if set(sampling) - allowed:
                raise ValueError("Unknown DVAC fields")
            alpha = float(sampling.get("alpha", 2.0))
            minimum = sampling.get("min_execution_steps", 1)
            maximum = sampling.get("max_execution_steps", horizon)
            window = sampling.get("rolling_window_size", 5)
            tail = sampling.get("tail_steps", 5)
            steps = model._server_metadata["num_inference_timesteps"]
            if any(type(value) is not int for value in (minimum, maximum, window, tail)):
                raise ValueError("DVAC step/window settings must be integers")
            if (
                not math.isfinite(alpha)
                or alpha < 0
                or not 1 <= minimum <= maximum <= horizon
                or window < 1
                or not 1 < tail <= steps
            ):
                raise ValueError("Invalid DVAC bounds, alpha, window or denoising tail length")
            if self.window_size not in (None, window):
                raise ValueError("DVAC rolling_window_size cannot change before reset")
            parameters["tail_steps"] = tail

        return parameters

    def _calibrate_dvac(self, metadata, sampling):
        """Select an execution prefix from episode-local denoising variance."""
        horizon = self.adapter.action_chunk_size
        alpha = float(sampling.get("alpha", 2.0))
        minimum = sampling.get("min_execution_steps", 1)
        maximum = sampling.get("max_execution_steps", horizon)
        window = sampling.get("rolling_window_size", 5)
        variance = np.asarray(metadata["variance"], dtype=np.float64)
        if variance.shape != (horizon,) or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError("Invalid denoising variance")
        cold = not self.history
        calibration = variance if cold else np.concatenate(tuple(self.history))
        mean, std = float(calibration.mean()), float(calibration.std())
        threshold = mean + alpha * std
        indices = np.flatnonzero(variance > threshold)
        crossing = int(indices[0]) if len(indices) else None
        execution = maximum if crossing is None else min(maximum, max(minimum, crossing))
        self.window_size = window
        self.history.append(variance.copy())
        while len(self.history) > window:
            self.history.popleft()
        metadata.update(
            execution_steps=execution,
            first_threshold_crossing=crossing,
            threshold=threshold,
            rolling_mean=mean,
            rolling_std=std,
            alpha=alpha,
            min_execution_steps=minimum,
            max_execution_steps=maximum,
            rolling_window_size=window,
            rolling_states=len(self.history),
            cold_start=cold,
            cold_start_policy="current_variance_bootstrap",
            method="denoising_variance_adaptive_chunking",
            source="arxiv:2606.03847v1",
        )
        return metadata
