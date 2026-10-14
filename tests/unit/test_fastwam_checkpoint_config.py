from pathlib import Path

import pytest
from omegaconf import OmegaConf

from XPolicyLab.policy.FastWAM.deployment_config import use_training_config

TRAINING = Path(__file__).resolve().parents[3] / 'training/latest-models-local-20261013/fastwam-training-config.yaml'


def test_saved_contract_overrides_sim_defaults_without_mutating_them():
    base = OmegaConf.create({'model': {'action_scheduler': {'infer_shift': 5}},
                            'data': {'train': {'num_frames': 17}},
                            'EVALUATION': {'output_dir': '${unavailable:run}', 'num_inference_steps': 10}})
    saved = OmegaConf.load(TRAINING)
    result = use_training_config(base, TRAINING)
    assert result.model.action_scheduler == saved.model.action_scheduler
    assert result.model.action_scheduler.infer_shift == 1
    assert result.data == saved.data
    assert result.model.action_dit_config.action_dim == 14
    assert result.model.tokenizer_max_len == 128
    assert result.model._target_ == 'fastwam.runtime.create_fastwam'
    assert result.model.action_dit_pretrained_path is None
    assert result.model.skip_dit_load_from_pretrain
    assert result.model.load_text_encoder
    assert result.EVALUATION.num_inference_steps == 10
    assert base.model.action_scheduler.infer_shift == 5


@pytest.mark.parametrize('case', ['dimension', 'scheduler', 'factory', 'compiled'])
def test_invalid_saved_contract_fails_before_model_loading(tmp_path, case):
    saved = OmegaConf.load(TRAINING)
    if case == 'dimension':
        saved.model.action_dit_config.action_dim = 16
    elif case == 'scheduler':
        del saved.model.action_scheduler.infer_shift
    elif case == 'factory':
        saved.model._target_ = 'another.factory'
    else:
        saved.model.compile_training_denoise = True
    p = tmp_path / 'config.yaml'
    OmegaConf.save(saved, p)
    with pytest.raises(ValueError):
        use_training_config(OmegaConf.create({}), p)


def test_adapter_accepts_explicit_robot_layout_without_parent_registry():
    from XPolicyLab.policy.FastWAM.model import Model
    model = Model({'action_type': 'joint', 'env_cfg_type': 'yam_explicit_test',
                   'robot_action_dim_info': {'arm_dim': [6, 6], 'ee_dim': [1, 1]},
                   'num_envs': 1, 'allow_dummy_policy': True})
    assert model.robot_action_dim_info == {'arm_dim': [6, 6], 'ee_dim': [1, 1]}
