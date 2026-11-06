"""Guard intentional upstream capability changes without weights or hardware."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XPolicyLab.policy.Pi_05.model import Model, encode_obs
from XPolicyLab.policy.LingBot_VLA2.model import Model as LingBotModel
from XPolicyLab.utils.autohorizon_selection import ConfiguredAutoHorizon, select_horizon


def test_droid_default_keeps_joints_and_converts_gripper_both_directions():
    dims = {'arm_dim': [7], 'ee_dim': [1]}
    obs = {
        'state': {'joint_state': np.arange(7, dtype=np.float32) / 10,
                  'ee_joint_state': np.array([.25], dtype=np.float32)},
        'instruction': 'pick',
        'vision': {name: {'color': np.full((224, 224, 3), [255, 0, 31], np.uint8)}
                   for name in ('exterior', 'wrist')},
    }
    encoded = encode_obs(obs, 'joint', dims, observation_profile='droid_jointpos')
    np.testing.assert_array_equal(encoded['observation/joint_position'], obs['state']['joint_state'])
    np.testing.assert_array_equal(encoded['observation/gripper_position'], [.75])
    np.testing.assert_array_equal(encoded['observation/wrist_image_left'][0, 0], [255, 0, 31])
    actions = np.tile(np.r_[obs['state']['joint_state'], .75], (3, 1))
    model = Model.__new__(Model)
    model.observation_profile = 'droid_jointpos'
    model.action_type = 'joint'
    model.robot_action_dim_info = dims
    model.action_horizon = 3
    model.num_steps = 10
    model.policy = SimpleNamespace(_is_pytorch_model=False, infer=lambda *a, **k: {'actions': actions})
    model.update_obs(obs)
    result = model.get_action()
    np.testing.assert_array_equal(result[0]['joint_state'], obs['state']['joint_state'])
    np.testing.assert_array_equal(result[0]['ee_joint_state'], [.25])
    assert model.sampling_modes() == ['default']
    for method in ('rtc', 'paint', 'aac', 'autohorizon'):
        with pytest.raises(ValueError, match='unsupported'):
            getattr(model, f'get_action_{method}')({})


@pytest.mark.parametrize('profile', ['tianji_taccap_pi05_pack_plate', 'tianji_taccap_pi05_pack_plate_joint'])
def test_trace_cannot_accidentally_enable_on_tianji_profiles(profile):
    with pytest.raises(ValueError, match='ALOHA/YAM'):
        Model({'observation_profile': profile, 'return_denoise_trace': True})


def test_lingbot_does_not_advertise_withdrawn_rtc_with_compile():
    model = object.__new__(LingBotModel)
    model.action_horizon = 50
    model.model = SimpleNamespace(vla=SimpleNamespace(model=SimpleNamespace(
        config=SimpleNamespace(n_action_steps=50, num_steps=10),
        get_predict_velocity_fn=lambda: None,
    )))
    assert model.sampling_modes() == ['default', 'aac', 'paint', 'autohorizon']
    assert not callable(getattr(model, 'get_action_rtc', None))


def test_autohorizon_options_do_not_leak_between_models_or_after_failure():
    seen = []

    def selector(attention, **kwargs):
        seen.append(kwargs)
        return torch.tensor(2), {}

    class Backend:
        fail = False

        def sampling_modes(self):
            return ['default', 'autohorizon']

        def get_action_autohorizon(self, sampling):
            select_horizon(torch.eye(5), selector=selector)
            if self.fail:
                raise RuntimeError('synthetic failure')
            return {'actions': [1], 'autohorizon': {'execution_steps': 2}}

    backend = Backend()
    wrapped = ConfiguredAutoHorizon(backend, {'hold_thr': .7, 'run_len': 2})
    result = wrapped.get_action_autohorizon({'mode': 'autohorizon'})
    assert result['autohorizon']['run_length'] == 2
    assert seen[-1]['hold_thr'] == .7
    backend.fail = True
    with pytest.raises(RuntimeError, match='synthetic failure'):
        wrapped.get_action_autohorizon({'mode': 'autohorizon'})
    select_horizon(torch.eye(5), selector=selector)
    assert seen[-1] == {'hold_thr': .3, 'run_len': 1, 'max_entropy_q': .9}
