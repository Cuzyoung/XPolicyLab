"""Trained-prefix sampler parity and the shared WebSocket dispatch boundary."""

import asyncio
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler
from torch import nn

from XPolicyLab.policy.UMI_DP.train_time_rtc import TrainTimeRtcSampler
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.model.common.normalizer import (
    SingleFieldLinearNormalizer,
)
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.policy.train_time_rtc_policy import (
    TrainTimeRtcDiffusionUnetTimmPolicy,
)


class TinyObsEncoder(nn.Module):
    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return obs["state"]


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_policy(cls=TrainTimeRtcDiffusionUnetTimmPolicy, prediction_type="epsilon", dim=2):
    policy = cls(
        shape_meta={"action": {"shape": [dim], "horizon": 32}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10, prediction_type=prediction_type),
        obs_encoder=TinyObsEncoder(),
        num_inference_steps=3,
        diffusion_step_embed_dim=16,
        down_dims=(8, 16),
        n_groups=4,
        inpaint_fixed_action_prefix=True,
        rtc={"enabled": True, "simulated_delay": 6, "delay_weights": [1, 0, 0, 0, 1, 2]},
    ).eval()
    # A nonidentity normalizer catches accidental normalization twice.
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
        scale=torch.full((dim,), 2.0),
        offset=torch.full((dim,), 0.3),
        input_stats_dict={key: torch.ones(dim) for key in ("min", "max", "mean", "std")},
    )
    policy.normalizer["state"] = SingleFieldLinearNormalizer.create_identity()
    return policy


@pytest.mark.parametrize("count", [0, 3, 4, 5, 6])
def test_sampler_freezes_normalized_prefix_and_passes_binary_local_mask(count):
    policy = make_policy()
    seen = []

    def record(_module, args, kwargs):
        seen.append((args[0].detach().clone(), kwargs["local_cond"].detach().clone()))

    policy.model.register_forward_pre_hook(record, with_kwargs=True)
    prefix = torch.rand(1, count, 2)
    with torch.inference_mode():
        output = policy.predict_action({"state": torch.zeros(1, 4)}, prefix)["action"]
    torch.testing.assert_close(output[:, :count], prefix)
    expected_mask = (torch.arange(32) < count).float()[None, :, None]
    assert len(seen) == 3
    for trajectory, mask in seen:
        torch.testing.assert_close(mask, expected_mask)
        torch.testing.assert_close(trajectory[:, :count], prefix * 2 + 0.3)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_calibwrist_reference_forward_matches(prediction_type):
    root = os.environ.get("CALIBWRIST_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set CALIBWRIST_REFERENCE_ROOT for source parity")
    sys.path.insert(0, str(Path(root) / "policies/diffusion_policy"))
    try:
        cls = importlib.import_module(
            "diffusion_policy.policy.diffusion_unet_timm_policy"
        ).DiffusionUnetTimmPolicy
        reference, policy = (
            make_policy(cls, prediction_type),
            make_policy(prediction_type=prediction_type),
        )
        policy.load_state_dict(reference.state_dict(), strict=True)
        obs = {"state": torch.randn(1, 4)}
        for count in (0, 3, 4, 5, 6):
            prefix = torch.rand(1, count, 2)
            with torch.inference_mode():
                torch.manual_seed(42)
                expected = reference.predict_action(obs, prefix)["action"]
                torch.manual_seed(42)
                actual = policy.predict_action(obs, prefix)["action"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        sys.path.pop(0)


def test_model_sampler_rebases_tcp_and_uses_native_hard_prefix():
    from XPolicyLab.policy.UMI_DP.transforms import relative_condition

    policy = make_policy(dim=20)
    sampler = TrainTimeRtcSampler(policy)
    reference = [np.eye(4), np.eye(4)]
    reference[0][0, 3] = 0.3
    absolute = np.tile([0.4, 0.2, 0.1, 1, 0, 0, 0, 0.5] * 2, (32, 1))
    mask = np.arange(32) < 5
    sampling = {"prefix_steps": 5, "action_condition": absolute, "condition_weights": mask}
    output = sampler.predict({"state": torch.zeros(1, 4)}, reference, sampling)
    expected = relative_condition(absolute, reference, mask)
    np.testing.assert_allclose(output[0, :5].numpy(), expected[:5], atol=1e-7)
    sampling["condition_weights"] = mask.astype(float)
    sampling["condition_weights"][4] = 0.5
    with pytest.raises(ValueError, match="binary prefix"):
        sampler.predict({"state": torch.zeros(1, 4)}, reference, sampling)


def test_shared_server_dispatches_train_time_rtc_without_guidance():
    from XPolicyLab.client_server.ws.model_server import PolicyServer
    from XPolicyLab.client_server.ws.protocol.messages import MessageType
    from XPolicyLab.client_server.ws.protocol.schemas import Frame

    class Model:
        def update_obs(self, observation):
            self.observation = observation

        def get_action_train_time_rtc(self, sampling):
            assert sampling == {"mode": "train_time_rtc", "prefix_steps": 0}
            return [1]

        def get_action_rtc(self, sampling):
            pytest.fail("Guidance must not run")

    frame = Frame(
        message_type=MessageType.INFER,
        request_id="test",
        evaluation_id="test",
        payload={
            "observation": {"state": {}},
            "sampling": {"mode": "train_time_rtc", "prefix_steps": 0},
        },
    )
    reply = asyncio.run(PolicyServer(Model())._handle_infer(frame))
    assert reply.payload["actions"] == [1]
