from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from XPolicyLab.policy.SAPolicy.inference_config import resolve_action_dataset


def action(name='real'):
    return dict(dataset_name=name, body_frame_actions=True, use_relative_actions=True,
                camera_names=['top', 'left', 'right'],
                transforms=[{'_target_': 'sapolicy.dataset.transform.Resize', 'height': 168, 'width': 224}])


def config(datasets, joint=False):
    train = {'dataset_opts': datasets}
    if joint:
        train['combined_loader_opts'] = {'mode': 'max_size_cycle'}
    return OmegaConf.create(dict(action_sequence_length=50, use_relative_actions=True,
                                 data={'train_dataset': train}))


@pytest.mark.parametrize('shape', ['single', 'flat', 'joint_flat', 'joint_nested'])
def test_selects_action_branch_and_preserves_training_config(shape):
    real, sim, tcp = action(), action('sim'), {'dataset_name': 'tcp', 'transforms': ['tcp-only']}
    datasets = {'single': real, 'flat': [real, sim], 'joint_flat': [tcp, sim],
                'joint_nested': [[tcp, deepcopy(tcp)], [real, sim]]}[shape]
    cfg = config(datasets, shape.startswith('joint'))
    before = OmegaConf.to_container(cfg)
    selected = resolve_action_dataset(cfg)
    assert selected.dataset_name == ('real' if shape == 'single' else 'sim')
    assert selected.transforms[0].height == 168
    assert OmegaConf.to_container(cfg) == before


@pytest.mark.parametrize('field,value', [
    ('body_frame_actions', False), ('use_relative_actions', False),
    ('transforms', []), ('camera_names', ['wrong']), ('action_sequence_length', 16),
])
def test_rejects_inconsistent_action_datasets(field, value):
    real, sim = action(), action('sim')
    real[field] = value
    with pytest.raises(ValueError, match='body-frame|inconsistent'):
        resolve_action_dataset(config([[{'dataset_name': 'tcp'}], [real, sim]], True))


def test_camera_pair_canonical_names_match_fixed_cameras():
    real = action()
    real['camera_pair_choices'] = {'canonical_names': real.pop('camera_names')}
    resolve_action_dataset(config([real, action('sim')]))


def test_shipped_mv51_recipe_remains_supported():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'mv51.yaml'
    selected = resolve_action_dataset(OmegaConf.load(path))
    assert selected.body_frame_actions


@pytest.mark.parametrize('datasets,joint', [([], False), ([[]], True), ({}, True)])
def test_rejects_empty_or_invalid_branches(datasets, joint):
    with pytest.raises(ValueError):
        resolve_action_dataset(config(datasets, joint))
