"""Sampling adapter for a diffusion policy trained with clean action prefixes."""

import numpy as np

from .transforms import relative_condition


class TrainTimeRtcSampler:
    """Pass a raw relative prefix into the policy's own hard-inpainting sampler."""

    def __init__(self, policy):
        if not policy.rtc_enabled or not policy.inpaint_fixed_action_prefix:
            raise ValueError("train_time_rtc requires a prefix-trained checkpoint")
        self.policy = policy

    def predict(self, batch, reference, sampling):
        import torch

        count = sampling.get("prefix_steps")
        if type(count) is not int or not 0 <= count < self.policy.action_horizon:
            raise ValueError("prefix_steps must be an integer in [0, action_horizon)")
        condition, weights = sampling.get("action_condition"), sampling.get("condition_weights")
        prefix = None
        if count == 0:
            if condition is not None or weights is not None:
                raise ValueError("An empty prefix must not supply a condition or weights")
        else:
            expected = np.arange(self.policy.action_horizon) < count
            if weights is None or not np.array_equal(np.asarray(weights), expected):
                raise ValueError("train_time_rtc requires a contiguous binary prefix mask")
            relative = relative_condition(condition, reference, expected)
            prefix = torch.as_tensor(
                relative[:count],
                device=self.policy.device,
                dtype=self.policy.dtype,
            ).unsqueeze(0)
        with torch.inference_mode():
            return self.policy.predict_action(batch, fixed_action_prefix=prefix)["action"]
