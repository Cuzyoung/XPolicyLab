from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import numpy as np
import pytest

from XPolicyLab.policy.starVLA import model as adapter


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """Replace only the expensive model runtime, retaining the real adapter."""
    monkeypatch.setattr(
        adapter,
        "get_robot_action_dim_info",
        lambda _: {
            "arm_dim": [6, 6],
            "ee_dim": [1, 1],
        },
    )
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    weights = run / "checkpoints" / "steps_100.pt"
    weights.touch()
    (run / "config.yaml").write_text(
        "framework: {name: QwenOFT}\ndatasets: {vla_data: {include_state: false}}\n"
    )

    class Runtime:
        instances = []

        def __init__(self, **kwargs):
            self.constructor = kwargs
            self.calls = []
            self.actions = np.arange(1 * 50 * 14, dtype=np.float32).reshape(1, 50, 14)
            self.metadata = {
                "ckpt_path": str(weights),
                "framework": "QwenOFT",
                "action_chunk_size": 50,
                "available_unnorm_keys": ["arx_x5"],
                "default_unnorm_key": "arx_x5",
                "runtime_contract": {
                    "version": 1,
                    "image_color_order": "rgb",
                    "state_input": "raw_env",
                    "state_normalization": "training_transform",
                    "action_output": "unnormalized_env",
                    "action_dim": 14,
                },
            }
            self.instances.append(self)

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            return {"actions": self.actions.copy()}

    wrapper = ModuleType("deployment.model_server.policy_wrapper")
    wrapper.PolicyServerWrapper = Runtime
    client = ModuleType("deployment.model_server.tools.websocket_policy_client")

    class Client:
        def __init__(self, *args):
            self.runtime = Runtime()

        def get_server_metadata(self):
            return self.runtime.metadata

        def predict_action(self, inputs):
            return {"ok": True, "data": self.runtime.predict_action(**inputs)}

    client.WebsocketClientPolicy = Client
    monkeypatch.setitem(sys.modules, wrapper.__name__, wrapper)
    monkeypatch.setitem(sys.modules, client.__name__, client)
    # Model construction prepends its vendored runtime; restore the path after each test.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    return Runtime, {
        "env_cfg_type": "test_bimanual",
        "checkpoint_path": str(weights),
        "model_backend": "inprocess",
        "action_output": "chunk",
        "image_size": [3, 2],
        "device": "cpu",
        "use_bf16": False,
    }


def observation(index=0):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 240
    return {
        "env_idx": index,
        "vision": {
            name: {"color": image.copy()}
            for name in (
                "cam_head",
                "cam_left_wrist",
                "cam_right_wrist",
            )
        },
        "state": {
            "left_arm_joint_state": np.arange(6),
            "left_ee_joint_state": np.array([6]),
            "right_arm_joint_state": np.arange(7, 13),
            "right_ee_joint_state": np.array([13]),
        },
        "instruction": f"task {index}",
    }


@pytest.mark.parametrize("backend", ["inprocess", "websocket"])
def test_complete_fresh_chunk_and_rgb(runtime, backend):
    factory, config = runtime
    model = adapter.Model({**config, "model_backend": backend})
    for index in (0, 7):
        model.update_obs(observation(index))
        actions = model.get_action()
        assert len(actions) == 50
        np.testing.assert_array_equal(actions[49]["right_arm_joint_state"], np.arange(693, 699))
    calls = factory.instances[0].calls
    assert [call["examples"][0]["lang"] for call in calls] == ["task 0", "task 7"]
    assert "state" not in calls[0]["examples"][0]
    np.testing.assert_array_equal(calls[0]["examples"][0]["image"][0][0, 0], [240, 0, 0])
    assert model.action_chunks_by_env == {}


def test_default_step_mode_keeps_existing_replan_schedule(runtime):
    factory, config = runtime
    config.pop("model_backend")
    config.pop("action_output")
    model = adapter.Model({**config, "execute_horizon": 2})
    for step in range(3):
        model.update_obs(observation())
        actions = model.get_action()
        assert len(actions) == 1
        assert actions[0]["left_arm_joint_state"][0] == (step % 2) * 14
    assert len(factory.instances[0].calls) == 2


@pytest.mark.parametrize("invalid", ["float", "chw", "grayscale", "rgb_key", "instruction"])
def test_observations_do_not_guess_layout_or_instruction(runtime, invalid):
    _, config = runtime
    model = adapter.Model(config)
    obs = observation()
    camera = obs["vision"]["cam_head"]
    if invalid == "float":
        camera["color"] = camera["color"].astype(float)
    elif invalid == "chw":
        camera["color"] = np.zeros((3, 10, 10), dtype=np.uint8)
    elif invalid == "grayscale":
        camera["color"] = np.zeros((10, 10, 1), dtype=np.uint8)
    elif invalid == "rgb_key":
        camera["rgb"] = camera.pop("color")
    else:
        obs["instruction"] = ["a different schema"]
    with pytest.raises(ValueError):
        model.update_obs(obs)


def test_missing_auto_state_contract_fails_before_model_load(runtime):
    from pathlib import Path

    factory, config = runtime
    run = Path(config["checkpoint_path"]).parent.parent
    (run / "config.yaml").write_text("framework: {name: QwenOFT}\n")
    with pytest.raises(ValueError, match="include_state"):
        adapter.Model(config)
    assert factory.instances == []


def test_explicit_permutations_preserve_gripper_positions_and_raw_state(runtime):
    factory, config = runtime
    indices = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    model = adapter.Model(
        {
            **config,
            "include_state": True,
            "action_indices": indices,
            "state_indices": np.argsort(indices).tolist(),
        }
    )
    model.update_obs(observation())
    action = model.get_action()[0]
    assert action["left_ee_joint_state"].tolist() == [12]
    assert action["right_arm_joint_state"].tolist() == list(range(6, 12))
    np.testing.assert_array_equal(
        factory.instances[0].calls[0]["examples"][0]["state"],
        [np.argsort(indices)],
    )


def test_batch_order_empty_batch_and_reset(runtime):
    factory, config = runtime
    model = adapter.Model(config)
    model.update_obs_batch([observation(9), observation(2)])
    assert len(model.get_action_batch([2, 9])) == 2
    assert [c["examples"][0]["lang"] for c in factory.instances[0].calls] == ["task 2", "task 9"]
    assert model.get_action_batch([]) == []
    model.reset()
    with pytest.raises(AssertionError, match="update_obs"):
        model.get_action()
    model.update_obs(observation())
    assert len(model.get_action()) == 50


@pytest.mark.parametrize("shape", [(1, 49, 14), (2, 50, 14), (1, 50, 13), (50, 14)])
def test_rejects_wrong_action_shapes(runtime, shape):
    factory, config = runtime
    model = adapter.Model(config)
    factory.instances[0].actions = np.zeros(shape)
    model.update_obs(observation())
    with pytest.raises(ValueError, match="finite with shape"):
        model.get_action()


def test_rejects_nonfinite_actions(runtime):
    factory, config = runtime
    model = adapter.Model(config)
    factory.instances[0].actions[0, 2, 3] = np.nan
    model.update_obs(observation())
    with pytest.raises(ValueError, match="finite with shape"):
        model.get_action()


@pytest.mark.parametrize(
    "option,value",
    [
        ("model_backend", "unknown"),
        ("action_output", "unknown"),
        ("action_indices", [0] * 14),
        ("state_indices", list(range(13))),
        ("require_runtime_contract", False),
    ],
)
def test_invalid_configuration_does_not_load_model(runtime, option, value):
    factory, config = runtime
    with pytest.raises(ValueError):
        adapter.Model({**config, option: value})
    assert factory.instances == []


def test_checkpoint_symlink_keeps_sidecar_location(runtime, tmp_path):
    factory, config = runtime
    from pathlib import Path

    checkpoint = Path(config["checkpoint_path"])
    blob = tmp_path / "blob"
    blob.touch()
    checkpoint.unlink()
    checkpoint.symlink_to(blob)
    adapter.Model(config)
    assert factory.instances[0].constructor["ckpt_path"] == str(checkpoint)


def test_shared_server_reports_identity_and_only_default_sampling(runtime, monkeypatch):
    from client_server.ws.model_server import PolicyServer
    from client_server.ws.protocol.messages import MessageType
    from client_server.ws.protocol.schemas import Frame

    async def inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    _, config = runtime
    model = adapter.Model(config)
    server = PolicyServer(model)
    hello = Frame(
        message_type=MessageType.HELLO, request_id="hello", evaluation_id="test", payload={}
    )
    reply = asyncio.run(server._dispatch_frame(hello))
    assert reply.payload["capabilities"]["sampling_modes"] == ["default"]
    metadata = reply.payload["model_metadata"]
    assert metadata["policy_name"] == "starVLA"
    assert metadata["checkpoint_path"] == config["checkpoint_path"]
    assert metadata["action_horizon"] == 50
    assert metadata["model_backend"] == "inprocess"


def test_shared_infer_returns_standard_joint_action_chunk(runtime, monkeypatch):
    from client_server.ws.model_server import PolicyServer
    from client_server.ws.protocol.messages import MessageType
    from client_server.ws.protocol.schemas import Frame

    async def inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    _, config = runtime
    server = PolicyServer(adapter.Model(config))
    request = Frame(
        message_type=MessageType.INFER,
        request_id="infer",
        evaluation_id="test",
        payload={"observation": observation(), "sampling": {"mode": "default"}},
    )
    reply = asyncio.run(server._dispatch_frame(request))
    actions = reply.payload["actions"]
    assert len(actions) == 50
    expected_widths = {
        "left_arm_joint_state": 6,
        "left_ee_joint_state": 1,
        "right_arm_joint_state": 6,
        "right_ee_joint_state": 1,
    }
    for index, step in enumerate(actions):
        assert set(step) == set(expected_widths)
        for key, width in expected_widths.items():
            assert np.asarray(step[key]).shape == (width,)
        row = np.concatenate([step[key] for key in expected_widths])
        np.testing.assert_array_equal(row, np.arange(index * 14, (index + 1) * 14))


@pytest.fixture
def eef_runtime(runtime, monkeypatch):
    factory, config = runtime
    monkeypatch.setattr(
        adapter, "get_robot_action_dim_info", lambda _: {"arm_dim": [7], "ee_dim": [1]}
    )
    original = factory.__init__

    def initialize(self, **kwargs):
        original(self, **kwargs)
        self.metadata["action_chunk_size"] = 8
        self.metadata["runtime_contract"]["action_dim"] = 7
        self.actions = np.tile(
            np.array([[[2, -2, 0.2, 0, 0, 0.4, 0.6]]], dtype=np.float32), (1, 8, 1)
        )

    monkeypatch.setattr(factory, "__init__", initialize)
    config.update(
        action_type="ee",
        camera_names=["cam_head"],
        eef={
            "rotation": "axis_angle",
            "semantics": "delta_step_base_xyz_wxyz",
            "translation_scale": [0.05] * 3,
            "rotation_scale": [0.5] * 3,
            "clip_input": 1.0,
            "gripper_threshold": 0.5,
        },
    )
    return factory, config


def test_libero_eef_scaling_rotation_camera_and_joint_state_width(eef_runtime):
    from scipy.spatial.transform import Rotation

    factory, config = eef_runtime
    model = adapter.Model({**config, "include_state": True})
    obs = observation()
    obs["vision"] = {"cam_head": obs["vision"]["cam_head"]}
    obs["state"] = {"joint_state": np.arange(7), "ee_joint_state": np.array([0.3])}
    model.update_obs(obs)
    rows = model.get_action()
    assert len(rows) == 8 and set(rows[0]) == {"ee_pose", "ee_joint_state"}
    np.testing.assert_allclose(rows[0]["ee_pose"][:3], [0.05, -0.05, 0.01])
    np.testing.assert_allclose(
        rows[0]["ee_pose"][3:], Rotation.from_rotvec([0, 0, 0.2]).as_quat()[[3, 0, 1, 2]], atol=1e-7
    )
    np.testing.assert_array_equal(rows[0]["ee_joint_state"], [1])
    inputs = factory.instances[-1].calls[-1]["examples"][0]
    assert len(inputs["image"]) == 1 and inputs["state"].shape == (1, 8)
    np.testing.assert_array_equal(inputs["image"][0][0, 0], [240, 0, 0])
    assert model.runtime_metadata()["action_semantics"] == "delta_step_base_xyz_wxyz"
    assert len(model.get_action_batch([0])[0]) == 8
    model.reset()
    with pytest.raises(AssertionError, match="update_obs"):
        model.get_action()


def test_eef_state_is_separate_from_native_action_encoding(eef_runtime):
    factory, config = eef_runtime
    model = adapter.Model({**config, "include_state": True, "state_type": "ee"})
    obs = observation()
    obs["state"] = {
        "ee_pose": np.array([0.1, 0.2, 0.3, 1, 0, 0, 0]),
        "ee_joint_state": np.array([0.3]),
    }
    model.update_obs(obs)
    model.get_action()
    np.testing.assert_allclose(
        factory.instances[-1].calls[-1]["examples"][0]["state"], [[0.1, 0.2, 0.3, 1, 0, 0, 0, 0.3]]
    )


def test_absolute_bimanual_quaternions_preserve_pose_and_tool():
    from XPolicyLab.policy.starVLA.eef import EefCodec

    codec = EefCodec(
        {"rotation": "quaternion_wxyz", "semantics": "absolute_per_arm_base_xyz_wxyz"},
        {"arm_dim": [6, 7], "ee_dim": [1, 1]},
    )
    row = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 0, 0.2, -0.1, 0.2, 0.3, 0, 1, 0, 0, 0.8]])
    np.testing.assert_allclose(codec.convert(row), row)
    row[0, 3] = 0
    with pytest.raises(ValueError, match="unit length"):
        codec.convert(row)


@pytest.mark.parametrize(
    "options", [{}, {"rotation": "rpy"}, {"rotation": "axis_angle", "semantics": "guess"}]
)
def test_eef_encoding_must_be_explicit(options):
    from XPolicyLab.policy.starVLA.eef import EefCodec

    with pytest.raises(ValueError):
        EefCodec(options, {"arm_dim": [7], "ee_dim": [1]})


@pytest.fixture
def flow_runtime(runtime, monkeypatch):
    from pathlib import Path

    factory, config = runtime
    (Path(config["checkpoint_path"]).parents[1] / "config.yaml").write_text(
        "framework: {name: QwenGR00T}\ndatasets: {vla_data: {include_state: false}}\n"
    )
    initialize = factory.__init__

    def init(self, **kwargs):
        initialize(self, **kwargs)
        self.metadata.update(
            framework="QwenGR00T",
            num_inference_timesteps=4,
            sampling_modes=["default", "rtc", "paint", "aac", "autohorizon", "dvac"],
        )

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        sampling = kwargs["sampling"]
        mode = sampling["mode"]
        result = {"actions": np.repeat(self.actions, sampling.get("num_samples", 1), axis=0)}
        if mode in {"autohorizon", "paint"}:
            result[mode] = (
                {"execution_steps": 12}
                if mode == "autohorizon"
                else {"delay_steps": sampling["delay_steps"]}
            )
        if mode == "dvac":
            result[mode] = {
                "variance": [0.0] * 20 + [1.0] * 30,
                "tail_steps": sampling["tail_steps"],
            }
        return result

    monkeypatch.setattr(factory, "__init__", init)
    monkeypatch.setattr(factory, "predict_action", predict)
    config["enabled_sampling_modes"] = ["rtc", "paint", "aac", "autohorizon", "dvac"]
    return factory, config


def test_flow_capabilities_and_condition_inverse_permutation(flow_runtime):
    factory, config = flow_runtime
    indices = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    model = adapter.Model({**config, "action_indices": indices})
    assert set(model.sampling_modes()) == {"default", "rtc", "paint", "aac", "autohorizon", "dvac"}
    model.update_obs(observation())
    condition = np.tile(np.arange(14, dtype=np.float32), (50, 1))
    rows = model.get_action_rtc(
        {"action_condition": condition, "condition_weights": np.ones(50), "beta": 1.0}
    )
    np.testing.assert_array_equal(
        factory.instances[-1].calls[-1]["sampling"]["action_condition"],
        condition[:, np.argsort(indices)],
    )
    assert rows[0]["left_ee_joint_state"].tolist() == [12]
    model.get_action_paint({"action_prefix": condition[:3], "delay_steps": 3})
    sent = factory.instances[-1].calls[-1]["sampling"]
    np.testing.assert_array_equal(sent["action_condition"][:3], condition[:3, np.argsort(indices)])
    assert sent["delay_steps"] == 3
    assert len(model.get_action_aac({"num_samples": 3})["actions"]) == 3
    assert (
        model.get_action_autohorizon({"mode": "autohorizon"})["autohorizon"]["execution_steps"]
        == 12
    )


def test_dvac_threshold_bounds_history_and_reset(flow_runtime):
    _, config = flow_runtime
    model = adapter.Model(config)
    model.update_obs(observation())
    sampling = {"tail_steps": 3, "alpha": 0.0, "max_execution_steps": 12}
    first = model.get_action_dvac(sampling)["dvac"]
    assert first["cold_start"] and first["execution_steps"] == 12
    assert not model.get_action_dvac(sampling)["dvac"]["cold_start"]
    with pytest.raises(ValueError, match="cannot change"):
        model.get_action_dvac({**sampling, "rolling_window_size": 2})
    model.reset()
    model.update_obs(observation())
    assert model.get_action_dvac(sampling)["dvac"]["cold_start"]
    with pytest.raises(ValueError, match="Invalid DVAC"):
        model.get_action_dvac({**sampling, "tail_steps": 5})


def test_unsupported_sampling_does_not_silently_use_default(runtime):
    _, config = runtime
    model = adapter.Model(config)
    model.update_obs(observation())
    for mode in ("rtc", "paint", "aac", "autohorizon", "dvac"):
        with pytest.raises(ValueError, match="not enabled"):
            getattr(model, "get_action_" + mode)({})
    with pytest.raises(ValueError, match="does not support"):
        adapter.Model({**config, "enabled_sampling_modes": ["rtc"]})
