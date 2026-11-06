"""Weightless tests of the real Pi05 adapter, WS handlers and binary codec.

Run with the pinned lightweight environment in tests/pi05_trace/README.md.
No socket, model checkpoint, camera or robot is constructed.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_SUPPORT_PATH = Path(__file__).resolve().parents[1] / "pi05_trace/support.py"
_SPEC = importlib.util.spec_from_file_location("pi05_trace_support", _SUPPORT_PATH)
support = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(support)


@pytest.fixture
def modules():
    with support.isolated_adapter_modules() as loaded:
        yield loaded


@pytest.fixture
def build_model(modules, monkeypatch, tmp_path):
    def build(*, data_class="LeRobotAlohaDataConfig", is_pytorch=False, pi05=True,
              model_dim=32, horizon=5, **overrides):
        data = getattr(modules.config, data_class)()
        train = SimpleNamespace(
            name="synthetic_pi05_trace", data=data,
            model=SimpleNamespace(pi05=pi05, action_horizon=horizon, action_dim=model_dim),
        )
        policy = support.FakePolicy(horizon=horizon, model_dim=model_dim, is_pytorch=is_pytorch)
        monkeypatch.setattr(modules.model, "_resolve_train_config", lambda _cfg: train)

        def load(instance, model_cfg):
            instance.model_root = tmp_path
            instance.norm_stats_path = None
            return policy

        monkeypatch.setattr(modules.model.Model, "get_model", load)
        config = {
            "task_name": "synthetic trace check", "return_denoise_trace": True,
            "robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
            "num_envs": 1, "inference_seed": 17, "num_steps": 3,
        }
        config.update(overrides)
        if config.get("return_denoise_trace") == "omit":
            del config["return_denoise_trace"]
        if config.get("robot_action_dim_info") == "omit":
            del config["robot_action_dim_info"]
        model = modules.model.Model(config)
        return model, policy

    return build


def _metadata(model):
    return model.runtime_metadata()["denoise_trace"]


def _flatten_actions(actions):
    return np.stack([
        np.concatenate([row[key] for key in (
            "left_arm_joint_state", "left_ee_joint_state",
            "right_arm_joint_state", "right_ee_joint_state",
        )])
        for row in actions
    ])


@pytest.mark.parametrize("data_class", ["LeRobotAlohaDataConfig", "LeRobotYamDataConfig"])
@pytest.mark.parametrize("steps", [1, 3])
def test_trace_keeps_full_actions_model_dimensions_and_single_step_axis(build_model, data_class, steps):
    model, policy = build_model(data_class=data_class, num_steps=steps)
    model.update_obs(support.observation(yam=data_class == "LeRobotYamDataConfig"))
    result = model.get_action()

    np.testing.assert_array_equal(_flatten_actions(result["actions"]), policy.actions)
    velocity = result["denoise_trace"]["v"]
    assert velocity.shape == (steps, 5, 32)
    assert velocity.dtype == np.float32
    np.testing.assert_array_equal(velocity, policy.velocity(steps))
    assert policy.calls == [{"num_steps": steps, "return_denoise_trace": True}]
    assert model.sampling_modes() == ["default"]


def test_trace_metadata_describes_normalized_padded_network_axes(build_model):
    model, policy = build_model()
    assert _metadata(model) == {
        "enabled": True, "schema_version": 1, "field": "v",
        "axes": ["denoising_step", "action_step", "model_action"],
        "coordinate_space": "normalized_model", "velocity_kind": "raw_network_velocity",
        "model_action_dim": 32, "valid_action_indices": list(range(14)),
        "dtype": "float32", "supported_modes": ["default"],
    }
    assert policy.calls == []  # Reading metadata never invokes inference.


def test_trace_single_batch_result_and_explicit_step_override(build_model):
    model, policy = build_model()
    model.update_obs(support.observation())
    result = model.get_action_batch(num_steps=1)
    assert len(result) == 1
    assert result[0]["denoise_trace"]["v"].shape == (1, 5, 32)
    np.testing.assert_array_equal(_flatten_actions(result[0]["actions"]), policy.actions)


@pytest.mark.parametrize("flag", [False, "omit"])
def test_omitted_flag_and_false_keep_legacy_chunk_and_metadata(build_model, flag):
    model, policy = build_model(return_denoise_trace=flag)
    model.update_obs(support.observation())
    result = model.get_action()
    assert isinstance(result, list)
    np.testing.assert_array_equal(_flatten_actions(result), policy.actions)
    assert "denoise_trace" not in model.runtime_metadata()
    assert policy.calls == [{"num_steps": 3}]


@pytest.mark.parametrize("bad", [None, 0, 1, "true", "false", np.bool_(True)])
def test_deployment_flag_is_strict_bool_before_loading(modules, monkeypatch, bad):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid deployment flag reached checkpoint loading")

    monkeypatch.setattr(modules.model.Model, "get_model", forbidden)
    with pytest.raises((TypeError, ValueError), match="return_denoise_trace"):
        modules.model.Model({"task_name": "test", "return_denoise_trace": bad})


@pytest.mark.parametrize("overrides", [
    {"action_type": "ee"}, {"num_envs": 2}, {"num_envs": True},
    {"robot_action_dim_info": "omit"},
    {"robot_action_dim_info": {"arm_dim": [7], "ee_dim": [1]}},
])
def test_trace_rejects_unsupported_deployment_contract(build_model, overrides):
    with pytest.raises((TypeError, ValueError)):
        build_model(**overrides)


@pytest.mark.parametrize("kwargs", [
    {"data_class": "LeRobotYamJointEeDataConfig"}, {"data_class": "SimpleDataConfig"},
    {"pi05": False}, {"model_dim": 13}, {"is_pytorch": True},
])
def test_trace_rejects_other_policy_profiles_and_backend(build_model, kwargs):
    with pytest.raises((TypeError, ValueError)):
        build_model(**kwargs)


@pytest.mark.parametrize("profile", ["tianji_taccap_pi05_zero_pose", "tianji_taccap_pi05_pack_plate"])
def test_trace_rejects_special_profiles_before_importing_them(modules, profile):
    with pytest.raises(ValueError, match="trace|return_denoise_trace"):
        modules.model.Model({"task_name": "test", "return_denoise_trace": True,
                             "observation_profile": profile})


@pytest.mark.parametrize("kwargs", [
    {"num_samples": 2}, {"autohorizon": True}, {"action_condition": np.zeros((5, 14))},
    {"paint_delay_steps": 1}, {"return_denoise_trace": True},
    {"num_steps": 0}, {"num_steps": True}, {"num_steps": 1.5},
])
def test_trace_rejects_combinations_before_policy_inference(build_model, kwargs):
    model, policy = build_model()
    model.update_obs(support.observation())
    with pytest.raises((TypeError, ValueError)):
        model.get_action(**kwargs)
    assert policy.calls == []


def test_request_cannot_enable_trace_on_a_nontrace_deployment(build_model):
    model, policy = build_model(return_denoise_trace=False)
    model.update_obs(support.observation())
    with pytest.raises(ValueError, match="return_denoise_trace"):
        model.get_action(return_denoise_trace=True)
    assert policy.calls == []


def test_trace_requires_single_observation_and_single_result(build_model):
    model, policy = build_model()
    with pytest.raises((ValueError, AssertionError)):
        model.update_obs_batch([support.observation(index=0), support.observation(index=1)])
        model.get_action_batch()
    assert policy.calls == []


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["actions", "v"])
def test_nonfinite_policy_outputs_fail_at_adapter_boundary(build_model, bad, field):
    model, policy = build_model()
    if field == "actions":
        policy.actions[0, 0] = bad
    else:
        policy.trace_override = policy.velocity(3)
        policy.trace_override[0, 0, 0] = bad
    model.update_obs(support.observation())
    with pytest.raises(ValueError, match="finite"):
        model.get_action()


@pytest.mark.parametrize("shape", [(3, 5, 14), (5, 3, 32), (5, 32), (1, 3, 5, 32)])
def test_malformed_or_truncated_model_trace_is_rejected(build_model, shape):
    model, policy = build_model()
    policy.trace_override = np.zeros(shape, dtype=np.float32)
    model.update_obs(support.observation())
    with pytest.raises(ValueError):
        model.get_action()


def test_missing_trace_cannot_silently_fall_back_to_plain_actions(build_model):
    model, policy = build_model()
    policy.include_trace = False
    model.update_obs(support.observation())
    with pytest.raises(ValueError, match="trace"):
        model.get_action()


def test_reset_clears_observation_and_preserves_trace_configuration(build_model):
    model, policy = build_model()
    model.update_obs(support.observation())
    model.get_action()
    model.reset()
    assert policy.resets == [17, 17]
    assert model.observation_window is None
    assert model._latest_env_idx_list == [0]
    assert _metadata(model)["enabled"] is True
    with pytest.raises(AssertionError):
        model.get_action()


def test_real_ws_handlers_and_codec_preserve_trace_without_binding_a_socket(modules, build_model):
    model, policy = build_model(num_steps=1)
    server = modules.server.PolicyServer(model)

    def frame(kind, request, payload):
        return modules.schemas.Frame(message_type=kind, request_id=request,
                                     evaluation_id="synthetic", payload=payload)

    async def exercise():
        hello = await server._dispatch_frame(frame(modules.messages.MessageType.HELLO, "hello", {}))
        infer = await server._dispatch_frame(frame(modules.messages.MessageType.INFER, "infer", {
            "observation": support.observation(), "sampling": {"mode": "default"},
        }))
        reset = await server._dispatch_frame(frame(modules.messages.MessageType.RESET, "reset", {}))
        return hello, infer, reset

    hello, infer, reset = asyncio.run(exercise())
    hello_wire = modules.codec.decode_envelope(modules.codec.encode_frame(hello))
    wire = modules.codec.decode_envelope(modules.codec.encode_frame(infer))
    assert hello_wire.payload["capabilities"]["sampling_modes"] == ["default"]
    assert hello_wire.payload["model_metadata"]["denoise_trace"] == _metadata(model)
    assert wire.message_type == modules.messages.MessageType.INFER_RESULT
    np.testing.assert_array_equal(_flatten_actions(wire.payload["actions"]), policy.actions)
    np.testing.assert_array_equal(wire.payload["denoise_trace"]["v"], policy.velocity(1))
    assert wire.payload["denoise_trace"]["v"].shape == (1, 5, 32)
    assert wire.payload["denoise_trace"]["v"].dtype == np.float32
    assert reset.message_type == modules.messages.MessageType.RESET_RESULT
    assert policy.resets == [17, 17]


@pytest.mark.parametrize("mode", ["rtc", "aac", "paint", "autohorizon"])
def test_real_ws_handler_rejects_nondefault_modes_in_trace_deployment(modules, build_model, mode):
    model, policy = build_model()
    frame = modules.schemas.Frame(
        message_type=modules.messages.MessageType.INFER, request_id="unsupported-mode",
        evaluation_id="synthetic", payload={
            "observation": support.observation(), "sampling": {"mode": mode},
        },
    )
    with pytest.raises(modules.exceptions.WsError) as error:
        asyncio.run(modules.server.PolicyServer(model)._handle_infer(frame))
    assert error.value.code == modules.exceptions.ErrorCode.INFER_FAILED
    assert policy.calls == []


def test_dependency_stubs_restore_preexisting_modules_and_import_path(monkeypatch):
    previous = SimpleNamespace(identity="preexisting-openpi-module")
    monkeypatch.setitem(sys.modules, "openpi", previous)
    saved_path = sys.path[:]
    with support.isolated_adapter_modules():
        assert sys.modules["openpi"] is not previous
    assert sys.modules["openpi"] is previous
    assert sys.path == saved_path


@pytest.fixture
def real_policy(modules):
    policy_module, backend = support.load_policy_with_numpy_backend()
    samples = []

    class Network:
        pi05 = True
        action_horizon = 5
        action_dim = 32

        def sample_actions(self, rng, observation, **kwargs):
            samples.append((rng, kwargs))
            actions = np.full((1, 5, 32), 2, dtype=np.float32)
            if kwargs.get("return_denoise_trace", False):
                steps = kwargs["num_steps"]
                velocity = np.arange(steps * 5 * 32, dtype=np.float16).reshape(1, steps, 5, 32)
                return actions, velocity
            return actions

    def output_transform(value):
        # A deliberate scale, offset and coordinate crop exposes accidental trace
        # transformation. Policy must attach trace only after this actual call.
        assert "denoise_trace" not in value
        return {"actions": value["actions"][:, :14] * 3 + 100}

    policy = policy_module.Policy(Network(), output_transforms=[output_transform])
    return policy, backend, samples


@pytest.mark.parametrize("steps", [1, 3])
def test_real_policy_output_transform_does_not_crop_or_scale_trace(real_policy, steps):
    policy, backend, samples = real_policy
    result = policy.infer({"state": np.zeros(32)}, num_steps=steps, return_denoise_trace=True)
    np.testing.assert_array_equal(result["actions"], np.full((5, 14), 106))
    velocity = result["denoise_trace"]["v"]
    assert velocity.shape == (steps, 5, 32)
    assert velocity.dtype == np.float32
    np.testing.assert_array_equal(velocity, np.arange(steps * 5 * 32).reshape(steps, 5, 32))
    assert len(samples) == 1
    assert len(backend.random_splits) == 1


def test_real_policy_lazily_constructs_trace_wrapper_and_reuses_it_after_reset(real_policy):
    policy, backend, samples = real_policy
    assert len(backend.jit_calls) == 2  # The two existing ordinary/multi wrappers.
    assert policy._sample_actions_trace is None
    default = policy.infer({"state": np.zeros(32)})
    assert "denoise_trace" not in default
    assert len(backend.jit_calls) == 2
    assert "return_denoise_trace" not in samples[0][1]
    policy.infer({"state": np.zeros(32)}, num_steps=1, return_denoise_trace=True)
    assert len(backend.jit_calls) == 3
    assert set(backend.jit_calls[-1]["static_argnames"]) == {
        "num_steps", "return_attention", "return_denoise_trace",
    }
    policy.reset_rng(7)
    policy.infer({"state": np.zeros(32)}, num_steps=3, return_denoise_trace=True)
    assert len(backend.jit_calls) == 3
    assert len(backend.random_splits) == 3  # One split per accepted inference.


@pytest.mark.parametrize("kwargs", [
    {"num_samples": 2}, {"autohorizon": True}, {"condition_weights": np.ones(5)},
    {"paint_action_condition": np.zeros((5, 32))}, {"num_steps": 0},
    {"num_steps": True}, {"num_steps": 1.5}, {"return_denoise_trace": "true"},
])
def test_real_policy_rejects_trace_combinations_before_rng_or_sampler(real_policy, kwargs):
    policy, backend, samples = real_policy
    options = {"return_denoise_trace": True, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        policy.infer({"state": np.zeros(32)}, **options)
    assert len(backend.jit_calls) == 2
    assert backend.random_splits == []
    assert samples == []


def test_real_policy_rejects_multiple_observations_before_rng_and_lazy_wrapper(real_policy):
    policy, backend, samples = real_policy
    with pytest.raises(ValueError, match="one observation"):
        policy.infer({"state": np.zeros((2, 32))}, return_denoise_trace=True)
    assert len(backend.jit_calls) == 2
    assert backend.random_splits == []
    assert samples == []
