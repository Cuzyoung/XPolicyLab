"""CPU contract checks, including the native small MoT, not checkpoint validation."""
import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XPolicyLab.policy.FastWAM.model import Model
from XPolicyLab.policy.FastWAM.sampling import ActionFlowSampler, normalize_actions
from XPolicyLab.utils.flow_sampling import ActionAttentionCapture
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


@pytest.fixture
def native():
    torch.manual_seed(71)
    common = dict(hidden_dim=24, ffn_dim=48, text_dim=16, freq_dim=8,
                  eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=2)
    action = ActionDiT(action_dim=14, **common)
    video = WanVideoDiT(in_dim=4, out_dim=4, patch_size=(1, 1, 1),
                       has_image_input=False, fuse_vae_embedding_in_latents=True, seperated_timestep=True,
                       video_attention_mask_mode="first_frame_causal", **common)
    model = FastWAM(video, action, MoT({"video": video, "action": action}),
                    vae=torch.nn.Identity(), text_dim=16, action_infer_shift=3.0)
    # Only image encoding is replaced. Native prepare, video KV prefill, action
    # attention, denoising, scheduler and hook dispatch all execute unchanged.
    model._encode_input_image_latents_tensor = lambda **kw: torch.arange(16).reshape(1, 4, 1, 2, 2).float()/16
    return model


def infer(native, sampler=None):
    return native.infer_action(
        prompt=None, input_image=torch.zeros(1, 3, 16, 16), action_horizon=8,
        context=torch.ones(1, 3, 16), context_mask=torch.ones(1, 3, dtype=torch.bool),
        num_inference_steps=4, seed=8, denoise_loop=sampler,
    )["action"]


def test_native_zero_weight_rtc_is_bitwise_equal(native):
    baseline = infer(native)
    sampler = ActionFlowSampler("rtc", target=torch.zeros(1, 8, 14), weights=np.zeros(8))
    torch.testing.assert_close(infer(native, sampler), baseline, rtol=0, atol=0)


def test_native_rtc_uses_input_gradient_and_preserves_parameter_grads(native):
    baseline = infer(native)
    sampler = ActionFlowSampler("rtc", target=torch.zeros(1, 8, 14), weights=np.ones(8))
    guided = infer(native, sampler)
    assert torch.isfinite(guided).all() and not torch.equal(guided, baseline)
    assert all(p.grad is None for p in native.parameters())


def test_native_autohorizon_preserves_actions_and_restores_hook(native):
    baseline = infer(native)
    sampler = ActionFlowSampler("autohorizon")
    result = infer(native, sampler)
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    assert sampler.metadata["denoising_step"] == 3
    assert 1 <= sampler.metadata["execution_steps"] <= 8
    assert native.mot._action_attention_capture is None


def test_native_attention_retains_video_softmax_mass(native):
    capture = ActionAttentionCapture(8)
    with capture.attach([native.mot]):
        infer(native)
    assert len(capture.maps) == 4 * 2
    for matrix in capture.maps:
        assert matrix.shape == (8, 8)
        assert torch.all(matrix.sum(-1) < 1.0)


def test_native_paint_returns_finite_full_chunk(native):
    sampler = ActionFlowSampler("paint", target=torch.zeros(1, 8, 14), delay=2)
    assert torch.isfinite(infer(native, sampler)).all()
    assert sampler.metadata["model_evaluations"] == 12


def test_paint_inverts_native_shifted_grid_and_only_replaces_prefix_noise():
    scheduler = WanContinuousFlowMatchScheduler(shift=5)
    times, deltas = scheduler.build_inference_schedule(4, torch.device("cpu"), torch.float32)
    noise = torch.arange(8).reshape(1, 4, 2).float()
    target = torch.full_like(noise, -2)
    called = []
    def velocity(x, time):
        called.append(float(time))
        return torch.ones_like(x) * 0.25
    sampler = ActionFlowSampler("paint", target=target, delay=2)
    result = sampler(noise, times, deltas, velocity=velocity, scheduler=scheduler, attention_owner=None)
    torch.testing.assert_close(result[:, :2], target[:, :2])
    torch.testing.assert_close(result[:, 2:], noise[:, 2:] - 0.25)
    assert called == times.tolist() + [0] + times[1:].flip(0).tolist() + times.tolist()


def test_normalization_uses_action_statistics_and_native_clamp():
    # The normalizer module also imports training-only GitPython. Execute its
    # actual class unchanged so CPU tests do not alter the robot environment.
    source = Path(__file__).resolve().parents[2] / "policy/FastWAM/FastWAM/src/fastwam/datasets/lerobot/utils/normalizer.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "SingleFieldLinearNormalizer")
    namespace = {"torch": torch, "NormMode": str}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    SingleFieldLinearNormalizer = namespace["SingleFieldLinearNormalizer"]
    action_norm = SingleFieldLinearNormalizer({"mean": torch.ones(14)*2, "std": torch.ones(14)*4}, "z-score")
    state_norm = SingleFieldLinearNormalizer({"mean": torch.ones(14)*10, "std": torch.ones(14)}, "z-score")
    processor = SimpleNamespace(action_state_transforms=None, delta_action_dim_mask=None,
        shape_meta={"action": [{"key": "a"}]},
        normalizer=SimpleNamespace(normalizers={"action": {"a": action_norm}, "state": {"s": state_norm}}))
    policy = SimpleNamespace(processor=processor)
    raw = np.full((8, 14), 6, dtype=np.float32)
    torch.testing.assert_close(normalize_actions(policy, raw), torch.ones(1, 8, 14))
    raw[0] = 100
    assert torch.all(normalize_actions(policy, raw)[0, 0] == 5)
    processor.delta_action_dim_mask = {"a": [True]*14}
    with pytest.raises(ValueError, match="absolute"):
        normalize_actions(policy, raw)


def adapter():
    model = Model({"action_type": "joint", "env_cfg_type": "yam",
        "robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
        "allow_dummy_policy": True, "action_horizon": 8, "replan_steps": 3})
    model.allow_dummy_policy = False
    class Backend:
        text_cfg_scale = 1.0
        num_inference_steps = 4
        processor = SimpleNamespace(action_state_transforms=None, delta_action_dim_mask=None,
                                    shape_meta={"action": [{"key": "a"}]})
        model = SimpleNamespace(infer_action=lambda denoise_loop=None: None,
                                mot=SimpleNamespace(supports_action_attention_capture=True))
        def _infer_action_chunk(self, obs, instruction, *, sampling_seed=None, denoise_loop=None):
            return np.random.default_rng(sampling_seed).standard_normal((8, 14)).astype(np.float32)
        def reset(self):
            pass
    model.model = Backend()
    return model


def observation():
    return {"vision": {name: {"color": np.zeros((240, 320, 3), dtype=np.uint8)}
                       for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
            "state": {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": np.zeros(1),
                      "right_arm_joint_state": np.zeros(6), "right_ee_joint_state": np.zeros(1)}}


def test_candidate_diversity_full_horizon_and_reset():
    model = adapter()
    model.update_obs(observation())
    assert len(model.get_action()) == 8  # replan_steps=3 must not truncate.
    first = model.get_action_aac({"mode": "aac", "num_samples": 2})["actions"]
    a, b = [c["actions"][0]["left_arm_joint_state"] for c in first]
    assert not np.array_equal(a, b)
    model.reset()
    assert model.last_obs is None and not model._batch_obs
    model.update_obs(observation())
    repeated = model.get_action_aac({"mode": "aac", "num_samples": 2})["actions"]
    np.testing.assert_array_equal(a, repeated[0]["actions"][0]["left_arm_joint_state"])
    model.update_obs_batch([dict(observation(), env_idx=i) for i in (0, 1)])
    with pytest.raises(ValueError, match="exactly one"):
        model.get_action_aac({"mode": "aac", "num_samples": 2})


def test_capabilities_fail_closed_and_real_ws_dispatch():
    from client_server.ws.model_server import PolicyServer
    from client_server.ws.protocol.messages import MessageType
    from client_server.ws.protocol.schemas import Frame
    model = adapter()
    async def run():
        server = PolicyServer(model)
        hello = await server.process_frame(Frame(message_type=MessageType.HELLO, request_id="h", evaluation_id="e", payload={}))
        assert hello.payload["capabilities"]["sampling_modes"] == ["default", "aac", "rtc", "paint", "autohorizon"]
        reply = await server.process_frame(Frame(message_type=MessageType.INFER, request_id="i", evaluation_id="e",
            payload={"observation": observation(), "sampling": {"mode": "aac", "num_samples": 2}}))
        assert reply.message_type == MessageType.INFER_RESULT, reply.payload
        return reply
    asyncio.run(run())
    model.model.model.mot.supports_action_attention_capture = False
    assert "autohorizon" not in model.sampling_modes()
    model.allow_dummy_policy = True
    assert model.sampling_modes() == ["default"]


@pytest.mark.parametrize("weights,beta", [(np.ones(7), 5), (np.full(8, np.nan), 5), (np.ones(8)*2, 5), (np.ones(8), 0)])
def test_invalid_rtc_rejected(native, weights, beta):
    with pytest.raises(ValueError):
        infer(native, ActionFlowSampler("rtc", target=torch.zeros(1, 8, 14), weights=weights, beta=beta))
