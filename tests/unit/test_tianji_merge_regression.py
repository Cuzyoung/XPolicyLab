"""CPU merge regressions with synthetic artifacts; no production weights/devices."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05.model import Model
from XPolicyLab.policy.Pi_05.pack_plate_contract import ARTIFACTS, validate_artifacts
from XPolicyLab.policy.Pi_05.pack_plate_joint_model import JointInputs, PackPlateJointModel


def artifact_config(tmp_path, variant, *, width, horizon, camera_mode=None):
    """Provide on-disk metadata and normalization, never any learned weights."""
    artifact = ARTIFACTS[variant]
    checkpoint = tmp_path / artifact['checkpoint_source'] / artifact.get(
        'checkpoint_subdir', f"checkpoint-{artifact['checkpoint_num']}"
    )
    (checkpoint / 'params').mkdir(parents=True)
    (checkpoint / 'params/_METADATA').write_text('{}')
    stats_dir = checkpoint / 'assets' / artifact['asset']
    stats_dir.mkdir(parents=True)
    stats = {name: {
        'mean': [0.] * width, 'std': [1.] * width,
        'q01': [-1.] * width, 'q99': [1.] * width,
    } for name in ('state', 'actions')}
    stats_file = stats_dir / 'norm_stats.json'
    stats_file.write_text(json.dumps({'norm_stats': stats}))
    digest = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    joint = camera_mode is not None
    return {
        'policy_name': 'Pi_05', 'protocol': 'ws', 'task_name': 'plate',
        'env_cfg_type': 'tianji_dual', 'action_type': 'joint' if joint else 'ee',
        'observation_profile': 'tianji_taccap_pi05_pack_plate' + ('_joint' if joint else ''),
        'checkpoint_variant': variant, 'checkpoint_source': artifact['checkpoint_source'],
        'checkpoint_num': artifact['checkpoint_num'],
        'train_config_name': artifact['train_config_name'],
        'model_state_encoding': 'absolute' if joint else 'zero_pose',
        'paligemma_variant': 'gemma_2b', 'action_expert_variant': 'gemma_300m',
        'action_horizon': horizon, 'output_format': 'xpolicylab',
        'action_semantics': 'absolute_joint_position' if joint else 'absolute_per_arm_base_xyz_wxyz',
        'model_path': str(checkpoint), 'norm_stats_path': str(stats_dir),
        'norm_stats_sha256': digest, 'norm_stats_source': 'sha256_' + digest,
        'num_steps': 10, 'robot_action_dim_info': {'arm_dim': [7, 7], 'ee_dim': [1, 1]},
        'num_envs': 1, 'repo_id': 'synthetic/merge-regression',
        **({'camera_mode': camera_mode} if joint else {}),
    }


@pytest.mark.parametrize(('variant', 'width', 'horizon', 'camera'), [
    ('pi05_pack_plate_joint_wrist_step29999', 16, 32, 'wrist_only'),
    ('pi05_pack_plate_joint_3cam_step29999', 16, 32, 'wrist_main'),
    ('pi05_pack_plate_joint_fk_tcp_zero_state_step29999', 20, 32, None),
    ('pi05_pack_plate_ratio_umi095_tele005_step29999', 20, 32, None),
    ('pi05_pack_plate_umi_half_speed_h50_step29999', 20, 50, None),
])
def test_checkpoint_identity_dimensions_horizon_and_checksum(tmp_path, variant, width, horizon, camera):
    cfg = artifact_config(tmp_path, variant, width=width, horizon=horizon, camera_mode=camera)
    contract = validate_artifacts(cfg)
    assert contract['action_horizon'] == horizon
    assert contract['action_dt_s'] == 1 / 30
    assert contract['sampling_modes'] == ['default', 'rtc']
    with pytest.raises(ValueError, match='action_horizon'):
        validate_artifacts({**cfg, 'action_horizon': horizon + 1})
    stats_file = Path(cfg['norm_stats_path']) / 'norm_stats.json'
    stats = json.loads(stats_file.read_text())
    stats['norm_stats']['actions']['mean'].append(0.)
    stats_file.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='checksum'):
        validate_artifacts(cfg)
    digest = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='normalization actions.mean'):
        validate_artifacts({**cfg, 'norm_stats_sha256': digest, 'norm_stats_source': 'sha256_' + digest})


def joint_observation():
    return {
        'env_idx': 3, 'instruction': 'place each plate into a bubble wrap sleeve',
        'state': {
            'left_arm_joint_state': np.arange(7, dtype=np.float32) / 10,
            'left_ee_joint_state': np.array([.2], dtype=np.float32),
            'right_arm_joint_state': -np.arange(7, dtype=np.float32) / 10,
            'right_ee_joint_state': np.array([.8], dtype=np.float32),
        },
        'vision': {f'cam_{name}': {'color': np.full((8, 10, 3), rgb, np.uint8)}
                   for name, rgb in [('left_wrist', [255, 0, 0]),
                                     ('right_wrist', [0, 255, 0]), ('main', [0, 0, 255])]},
    }


@pytest.mark.parametrize(('view', 'camera'), [('wrist', 'wrist_only'), ('3cam', 'wrist_main')])
def test_joint_factory_default_rtc_rgb_and_reset_without_weights(tmp_path, monkeypatch, view, camera):
    from openpi.policies import policy_config

    cfg = artifact_config(tmp_path, f'pi05_pack_plate_joint_{view}_step29999',
                          width=16, horizon=32, camera_mode=camera)
    cfg['checkpoint_low_memory'] = True
    calls = []
    actions = np.tile(np.arange(16, dtype=np.float32) / 10, (32, 1))
    actions[:, [7, 15]] = [-.05, 1.05]

    class Policy:
        def infer(self, observation, **kwargs):
            calls.append((observation, kwargs))
            return {'actions': actions}

    def create_policy(train, path, **kwargs):
        assert train.model.action_horizon == 32 and train.model.action_dim == 32
        assert Path(path) == Path(cfg['model_path'])
        assert kwargs['checkpoint_low_memory'] is True
        assert kwargs['norm_stats']['state'].mean.shape == (16,)
        return Policy()

    monkeypatch.setattr(policy_config, 'create_trained_policy', create_policy)
    model = Model(cfg)
    assert isinstance(model, PackPlateJointModel)
    obs = joint_observation()
    model.update_obs(obs)
    encoded = model.observations[3]
    expected = np.concatenate(list(obs['state'].values()))
    np.testing.assert_array_equal(encoded['state'], expected)
    inputs = JointInputs(camera)(encoded)
    assert inputs['image_mask']['base_0_rgb'] == (view == '3cam')
    np.testing.assert_array_equal(inputs['image']['left_wrist_0_rgb'][0, 0], [255, 0, 0])
    np.testing.assert_array_equal(inputs['image']['right_wrist_0_rgb'][0, 0], [0, 255, 0])
    if view == '3cam':
        np.testing.assert_array_equal(inputs['image']['base_0_rgb'][0, 0], [0, 0, 255])
    else:
        assert not inputs['image']['base_0_rgb'].any()
    result = model.get_action()
    assert len(result) == 32
    np.testing.assert_array_equal(result[0]['left_arm_joint_state'], actions[0, :7])
    np.testing.assert_array_equal(result[0]['right_arm_joint_state'], actions[0, 8:15])
    np.testing.assert_array_equal(result[0]['left_ee_joint_state'], [0.])
    np.testing.assert_array_equal(result[0]['right_ee_joint_state'], [1.])
    assert actions[0, 7] < 0 and actions[0, 15] > 1  # No in-place mutation.
    condition = np.tile(expected, (32, 1))
    condition.setflags(write=False)
    weights = np.linspace(1, 0, 32, dtype=np.float32)
    weights.setflags(write=False)
    assert len(model.get_action_rtc({'action_condition': condition,
                                   'condition_weights': weights, 'beta': 4.})) == 32
    np.testing.assert_array_equal(calls[-1][1]['action_condition'], condition)
    assert not np.shares_memory(calls[-1][1]['action_condition'], condition)
    assert calls[-1][1]['rtc_beta'] == 4.
    assert model.runtime_metadata()['native_order'] == 'left_right'
    assert model.sampling_modes() == ['default', 'rtc']
    with pytest.raises(ValueError, match='unique'):
        model.update_obs_batch([obs, obs])
    actions[:, 7] = -.5
    with pytest.raises(ValueError, match='opening'):
        model.get_action()
    model.reset()
    with pytest.raises(ValueError, match='update_obs'):
        model.get_action()


@pytest.mark.parametrize('dtype', [None, 'bfloat16'])
def test_low_memory_restore_matches_original_and_preserves_registry(tmp_path, dtype):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params

    assert all(device.platform == 'cpu' for device in jax.devices())
    source = np.linspace(-3, 3, 77, dtype=np.float32).reshape(7, 11)
    checkpoint = tmp_path / 'params'
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint, {'params': {'weight': {'value': jax.device_put(source)}}})
    handler = ocp.type_handlers.get_type_handler(jax.Array)
    target_dtype = None if dtype is None else jnp.bfloat16
    ordinary = restore_params(checkpoint, dtype=target_dtype)
    limited = restore_params(checkpoint, dtype=target_dtype, low_memory=True)
    np.testing.assert_array_equal(limited['weight'], ordinary['weight'])
    np.testing.assert_array_equal(limited['weight'], source if dtype is None else source.astype(target_dtype))
    assert ocp.type_handlers.get_type_handler(jax.Array) is handler
