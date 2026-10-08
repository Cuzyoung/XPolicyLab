from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.SAPolicy import ensure_sapolicy_on_path
from XPolicyLab.policy.SAPolicy.model import Model

torch = pytest.importorskip("torch")
ensure_sapolicy_on_path()
from sapolicy.eval.robotwin import sa_policy_server as backend_module  # noqa: E402
from sapolicy.models.action_head.dit import DiTActionHead  # noqa: E402
from sapolicy.models.action_head.rtc import guided_velocity  # noqa: E402


def test_vjp_matches_analytical_field_and_does_not_store_parameter_gradients():
    sample = torch.tensor([[[0.8, -0.4]]])
    target = torch.tensor([[[0.2, 0.3]]])
    gain = torch.nn.Parameter(torch.tensor(0.3))
    time, beta = 0.5, 2.0
    actual = guided_velocity(
        lambda x: gain * x, sample, time, target, torch.ones_like(sample), beta
    )
    clean = sample * (1 - time * gain.detach())
    expected = gain.detach() * sample - 2 * (target - clean) * (1 - time * gain.detach())
    torch.testing.assert_close(actual, expected)
    assert gain.grad is None and not actual.requires_grad
    old_error = (target - clean).norm()
    new_error = (target - (sample - 0.1 * actual) * (1 - time * gain.detach())).norm()
    assert new_error < old_error


def test_dit_zero_mask_is_exact_and_context_clears_on_failure():
    # Exercise the real Euler sampler with an analytic field; no network fixture.
    class Head(torch.nn.Module):
        sequence_length, action_dim, num_inference_steps = 3, 2, 10
        disable_tcp_kv = True
        rtc_condition = DiTActionHead.rtc_condition
        sample_trajectory = DiTActionHead.sample_trajectory

        def _build_cond_kv(self, *args, **kwargs):
            return None, None

        def _build_cross_kv_cache(self, cond):
            return None

        def _forward_cond(self, x, *args):
            return x * 0.3

        def _step_fn(self):
            return self._forward_cond

    head = Head()
    patches = {"top": torch.zeros(1, 2, 1, 1, 1)}
    target = torch.ones(1, 3, 2) * 0.2
    torch.manual_seed(123)
    ordinary = head.sample_trajectory(patches)
    with head.rtc_condition(target, torch.zeros(1, 3, 1), 5.0):
        torch.manual_seed(123)
        torch.testing.assert_close(ordinary, head.sample_trajectory(patches), atol=0, rtol=0)
    with head.rtc_condition(target, torch.ones(1, 3, 1), 5.0):
        torch.manual_seed(123)
        guided = head.sample_trajectory(patches)
    assert (guided - target).norm() < (ordinary - target).norm()
    with (
        pytest.raises(RuntimeError, match="injected"),
        head.rtc_condition(target, torch.ones(1, 3, 1), 5.0),
    ):
        raise RuntimeError("injected")
    assert head._rtc_sampling is None


@pytest.mark.parametrize("offset", [0.0, 0.12])
def test_rtc_condition_inverts_body_actions_for_rotated_both_arm_frames(monkeypatch, offset):
    monkeypatch.setattr(backend_module, "TCP_FORWARD_OFFSET", offset)
    backend = object.__new__(backend_module.SAPolicyRoboTwinModel)
    backend.n_action_steps = 5
    rng = np.random.default_rng(34)
    condition = np.zeros((5, 16))
    observation = {}
    for arm, side in enumerate(("left", "right")):
        current_rotation = Rotation.from_euler("xyz", [0.2, -0.6, 0.9 + arm]).as_quat()
        observation[f"{side}_endpose"] = np.r_[
            rng.normal(size=3), current_rotation[3], current_rotation[:3]
        ]
        rotation = Rotation.random(5, random_state=rng).as_quat()
        condition[:, arm * 8 : arm * 8 + 3] = rng.normal(size=(5, 3))
        condition[:, arm * 8 + 3 : arm * 8 + 7] = rotation[:, [3, 0, 1, 2]]
        condition[:, arm * 8 + 7] = rng.uniform(size=5)
    relative = backend._rtc_relative_actions(condition, observation)
    reconstructed = backend._to_robotwin_ee(relative, observation)
    for arm in (0, 1):
        np.testing.assert_allclose(
            reconstructed[:, arm * 8 : arm * 8 + 3], condition[:, arm * 8 : arm * 8 + 3], atol=1e-12
        )
        np.testing.assert_allclose(
            reconstructed[:, arm * 8 + 7], condition[:, arm * 8 + 7], atol=1e-12
        )
        q1, q2 = (
            reconstructed[:, arm * 8 + 3 : arm * 8 + 7],
            condition[:, arm * 8 + 3 : arm * 8 + 7],
        )
        np.testing.assert_allclose(np.abs((q1 * q2).sum(-1)), 1, atol=1e-12)


def test_rtc_capability_requires_real_matching_sampler_and_reset_observation():
    model = Model({"dry_run": True, "action_horizon": 3})
    assert model.sampling_modes() == ["default"]
    with pytest.raises(NotImplementedError):
        model.get_action_rtc({})
    head = SimpleNamespace(rtc_condition=lambda: None, sequence_length=50, action_dim=20)
    model._backend = SimpleNamespace(
        policy=SimpleNamespace(pipeline=SimpleNamespace(action_head=head))
    )
    assert model.sampling_modes() == ["default"]
    head.sequence_length = 3
    model._dry_run = False
    assert model.sampling_modes() == ["default", "rtc"]
    with pytest.raises(RuntimeError, match="before any update_obs"):
        model.get_action_rtc({})


@pytest.mark.parametrize(
    "invalid",
    [
        {"action_condition": np.zeros((3, 14))},
        {"action_condition": np.full((3, 16), np.nan)},
        {"action_condition": np.zeros((3, 16))},
        {"condition_weights": np.ones((3, 1))},
        {"condition_weights": np.array([1, -0.1, 0])},
        {"condition_weights": np.array([1.1, 0, 0])},
        {"beta": 0},
        {"beta": float("nan")},
    ],
)
def test_invalid_rtc_contract_fails_before_sampling(invalid):
    model = Model({"dry_run": True, "action_horizon": 3})
    head = SimpleNamespace(rtc_condition=lambda: None, sequence_length=3, action_dim=20)
    model._backend = SimpleNamespace(
        policy=SimpleNamespace(pipeline=SimpleNamespace(action_head=head))
    )
    model._dry_run = False
    model._obs = {}
    condition = np.zeros((3, 16))
    condition[:, [3, 11]] = 1
    sampling = {"action_condition": condition, "condition_weights": np.ones(3), "beta": 5}
    with pytest.raises(ValueError):
        model.get_action_rtc({**sampling, **invalid})
