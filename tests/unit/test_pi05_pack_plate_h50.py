"""Horizon-sensitive TCP transforms and model hooks, without weights or hardware."""

from pathlib import Path

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05.pack_plate_model import (
    PackPlateZeroPoseModel, encode_rtc_condition, make_pack_plate_train_config,
)
from XPolicyLab.policy.Pi_05.pass_ball_model import decode_actions
from XPolicyLab.policy.Pi_05.pass_ball_pose import (
    OPTICAL_FROM_FLU, relative_action_chunk, rotation_to_6d,
)


@pytest.mark.parametrize("horizon", [32, 50])
def test_default_rtc_roundtrip_and_metadata(horizon):
    anchor = np.zeros(20, dtype=np.float32)
    for offset in (0, 10):
        anchor[offset:offset + 3] = (0.2, -0.1, 0.3)
        anchor[offset + 3:offset + 9] = rotation_to_6d(OPTICAL_FROM_FLU.T)
        anchor[offset + 9] = 0.5
    condition = np.zeros((horizon, 16), dtype=np.float32)
    condition[:, [3, 11]] = 1
    condition[:, [7, 15]] = (0.25, 0.75)
    condition[:, :3] = (0.3, 0.2, 0.4)
    condition[:, 8:11] = (-0.3, -0.2, 0.4)
    weights = np.ones(horizon, dtype=np.float32)
    native = encode_rtc_condition(condition, weights, anchor, action_horizon=horizon)
    relative = relative_action_chunk(anchor, native)
    decoded = decode_actions(anchor, relative, action_horizon=horizon)
    for i, step in enumerate(decoded):
        np.testing.assert_allclose(step['left_ee_pose'], condition[i, :7], atol=1e-6)
        np.testing.assert_allclose(step['right_ee_pose'], condition[i, 8:15], atol=1e-6)
        np.testing.assert_allclose(step['left_ee_joint_state'], [0.25])
        np.testing.assert_allclose(step['right_ee_joint_state'], [0.75])
    zero = encode_rtc_condition(
        np.zeros_like(condition), np.zeros_like(weights), anchor, action_horizon=horizon,
    )
    np.testing.assert_allclose(zero, np.tile(anchor, (horizon, 1)))

    calls = []

    class FakePolicy:
        def infer(self, obs, **kwargs):
            calls.append(kwargs)
            return {'actions': relative}

    model = object.__new__(PackPlateZeroPoseModel)
    model.action_horizon = horizon
    model.num_steps = 10
    model.model_root = Path('/offline/checkpoint')
    model.norm_stats_path = Path('/offline/assets')
    model.model_cfg = {key: 'offline' for key in (
        'policy_name', 'task_name', 'checkpoint_variant', 'checkpoint_source',
        'norm_stats_source', 'train_config_name', 'observation_profile',
        'action_semantics', 'output_format',
    )}
    model.model_cfg['model_state_encoding'] = 'zero_pose'
    model.policy = FakePolicy()
    model.observations = {0: {'state': anchor}}
    model.env_indices = [0]
    assert len(model.get_action()) == horizon
    assert len(model.get_action_rtc({
        'action_condition': condition, 'condition_weights': weights, 'beta': 5.,
    })) == horizon
    assert calls[-1]['action_condition'].shape == (horizon, 20)
    assert calls[-1]['condition_weights'].shape == (horizon,)
    assert model.runtime_metadata()['action_horizon'] == horizon
    with pytest.raises(ValueError, match='conditions'):
        model.get_action_rtc({
            'action_condition': condition[:-1], 'condition_weights': weights, 'beta': 5.,
        })
    with pytest.raises(ValueError, match='native actions'):
        decode_actions(anchor, relative[:-1], action_horizon=horizon)
    model.reset()
    with pytest.raises(ValueError, match='update_obs'):
        model.get_action()


def test_h50_train_config_keeps_32_model_dimensions():
    config = make_pack_plate_train_config({
        'repo_id': 'local/offline', 'train_config_name': 'h50-deploy',
        'action_horizon': 50, 'paligemma_variant': 'gemma_2b',
        'action_expert_variant': 'gemma_300m', 'model_state_encoding': 'zero_pose',
    })
    assert config.model.action_horizon == 50
    assert config.model.action_dim == 32
    assert config.model.paligemma_variant == 'gemma_2b'
    assert config.model.action_expert_variant == 'gemma_300m'
