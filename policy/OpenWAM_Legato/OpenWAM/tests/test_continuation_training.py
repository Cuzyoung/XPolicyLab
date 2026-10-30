"""Continuation math, real tiny ActionDiT gradients and checkpoint integration."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from openwam.model.continuation import (
    ContinuationConfig, configure_architecture, prepare_action_training,
)
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.registry import build_architecture, resolve_architecture_config
from tests.test_openwam_trainer import _make_tiny_arch, _make_fake_loss_inputs


def batch():
    a = torch.arange(24).reshape(2, 4, 3).float() / 10
    eps = torch.zeros_like(a)
    sigma = torch.tensor([0.25, 0.75])
    t = sigma * 1000
    x = (1 - sigma[:, None, None]) * a + sigma[:, None, None] * eps
    return a, eps, sigma, t, x, eps - a


def test_disabled_is_identity_and_does_not_consume_rng():
    args = batch()
    before = torch.get_rng_state().clone()
    result = prepare_action_training(ContinuationConfig(), *args)
    assert result[0] is args[4] and result[1] is args[3] and result[2] is args[5]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('dimensions', [2, 3])
def test_rtc_input_target_padding_and_gradient(dimensions):
    args = batch(); a, eps, sigma, t, x, u = args
    pad = torch.zeros(a.shape if dimensions == 3 else a.shape[:2], dtype=torch.bool)
    pad[0, -1] = True
    original = pad.clone()
    inp, ts, target, mask = prepare_action_training(
        ContinuationConfig(method='train_time_rtc'), *args, pad,
        prefix_lengths=torch.tensor([1, 2]),
    )
    torch.testing.assert_close(inp[0, :1], a[0, :1]); torch.testing.assert_close(inp[1, :2], a[1, :2])
    torch.testing.assert_close(inp[1, 2:], x[1, 2:])
    assert ts.tolist() == [[0, 250, 250, 250], [0, 0, 750, 750]]
    assert target is u and torch.equal(original, pad)
    pred = torch.ones_like(a, requires_grad=True)
    arch = _make_tiny_arch()
    class Weights:
        def training_weight(self, ids): return torch.tensor([2., 3.])
    loss = arch._compute_action_loss(pred, target, torch.tensor([0, 1]), Weights(), {'action_is_pad': mask}, 'cpu')
    loss.backward()
    assert pred.grad[0, 0].abs().sum() == 0 and pred.grad[1, :2].abs().sum() == 0
    assert pred.grad[0, -1].abs().sum() == 0
    assert pred.grad[1, 2:].abs().sum() > 0
    valid = ~mask
    if valid.ndim == 2: valid = valid[..., None].expand_as(a)
    oracle = (((pred.detach()-target).square()*valid).sum((1,2))/valid.sum((1,2))*torch.tensor([2.,3.])).mean()
    torch.testing.assert_close(loss, oracle)


@pytest.mark.parametrize('reference,sign', [('official_20261020',-1), ('official_code',1), ('paper_v2',-1)])
def test_legato_target_matches_reference_after_time_reversal(reference, sign):
    args = batch(); a, eps, sigma, ts, x, _ = args
    cfg = ContinuationConfig(method='legato', reference=reference, num_denoise_steps=5)
    inp, _, target, _ = prepare_action_training(cfg, *args, prefix_lengths=torch.tensor([1,2]))
    omega = torch.tensor([[1,0,0,0],[1,1,0,0]],dtype=a.dtype)[...,None]
    # Independent reference uses increasing clean time and clean-minus-noise velocity.
    clean_time = 1 - sigma[:,None,None]
    ref_input = omega*a + (1-omega)*((1-clean_time)*eps+clean_time*a)
    ref_target = (1+sign*omega*5*(1-clean_time))*(a-eps)
    torch.testing.assert_close(inp[...,:3],ref_input)
    torch.testing.assert_close(inp[...,3:],1-omega)
    torch.testing.assert_close(target,-ref_target)
    assert target.shape == a.shape


def test_zero_prefix_legato_reduces_to_native_fm():
    args = batch()
    inp, ts, target, pad = prepare_action_training(
        ContinuationConfig(method='legato'), *args, prefix_lengths=torch.zeros(2,dtype=torch.long))
    torch.testing.assert_close(inp[...,:3], args[4]); assert (inp[...,3] == 1).all()
    torch.testing.assert_close(target,args[5]); assert ts is args[3] and pad is None


def test_linear_schedule_has_explicit_boundary_semantics():
    args=batch()
    cfg=ContinuationConfig(method='legato',schedule='linear',ramp_steps=2)
    inp,_,_,_=prepare_action_training(cfg,*args,prefix_lengths=torch.tensor([1,0]))
    torch.testing.assert_close(inp[0,:,-1],torch.tensor([0,1/3,2/3,1]))


@pytest.mark.parametrize('variant', ['joint_self_attn','joint_cross_attn'])
def test_real_action_backbone_token_time_equivalence_and_backward(variant):
    torch.manual_seed(4)
    dit=ActionDiT(action_dim=3,dim=16,ffn_dim=32,num_heads=2,num_layers=1,
                  video_dim=16,text_dim=8,bridge_layers=(0,),variant=variant,freq_dim=8)
    a=torch.randn(2,4,3);t=torch.tensor([100.,700.]);ctx=torch.randn(2,2,8)
    if variant=='joint_cross_attn':
        bridge={0:torch.randn(2,3,16)}
        y=dit(a,bridge,t,context=ctx)
        z=dit(a,bridge,t[:,None].expand(2,4),context=ctx)
        torch.testing.assert_close(y,z)
        tt=t[:,None].expand(2,4).clone();tt[:,:2]=0
        changed=dit(a,bridge,tt,context=ctx)
        assert not torch.allclose(z,changed)
        changed.square().mean().backward()
    else:
        s1=dit.prepare_state(a,t,context=ctx)
        s2=dit.prepare_state(a,t[:,None].expand(2,4),context=ctx)
        q1,k1,v1,_=dit.pre_attn_at_layer_for_compile(0,s1)
        q2,k2,v2,_=dit.pre_attn_at_layer_for_compile(0,s2)
        torch.testing.assert_close(q1,q2);torch.testing.assert_close(k1,k2);torch.testing.assert_close(v1,v2)
        def finish_self_attention(state):
            q, k, v, post = dit.pre_attn_at_layer_for_compile(0, state)
            def heads(x):
                return x.reshape(x.shape[0], x.shape[1], 2, -1).transpose(1, 2)
            attention = torch.nn.functional.scaled_dot_product_attention(heads(q), heads(k), heads(v))
            attention = attention.transpose(1, 2).reshape_as(q)
            state = dit.post_attn_at_layer_for_compile(0, state, attention, post)
            return dit.extract_prediction(state)
        scalar = finish_self_attention(s1)
        per_token = finish_self_attention(s2)
        torch.testing.assert_close(scalar, per_token)
        prefix_time = t[:, None].expand(2, 4).clone()
        prefix_time[:, :2] = 0
        conditioned = finish_self_attention(dit.prepare_state(a, prefix_time, context=ctx))
        assert not torch.allclose(scalar, conditioned)
        conditioned.square().mean().backward()
        assert dit.blocks[0].ffn[0].weight.grad is not None
    assert dit.action_encoder.weight.grad is not None
    assert torch.isfinite(dit.time_embedding.mlp[0].weight.grad).all()


@pytest.mark.parametrize('method', ['none','train_time_rtc','legato'])
def test_actual_compute_loss_optimizer_updates_and_checkpoint(tmp_path,method):
    torch.manual_seed(19)
    arch=_make_tiny_arch();arch.cfg={**arch.cfg,'shift_action':1.0}
    settings={'method':method,'prefix_min_steps':1,'prefix_max_steps':2}
    original=arch.action_backbone.action_encoder.weight.detach().clone()
    configure_architecture(arch,settings)
    arch.init_training_schedulers(1000)
    if method=='legato':
        assert torch.equal(arch.action_backbone.action_encoder.weight[:,:-1],original)
        assert (arch.action_backbone.action_encoder.weight[:,-1]==0).all()
    opt=torch.optim.AdamW(arch.parameters(),lr=1e-4)
    for _ in range(2):
        inputs=_make_fake_loss_inputs(B=2)
        inputs['actions']=torch.randn(2,5,7)
        inputs['action_is_pad']=torch.zeros(2,5,7,dtype=torch.bool)
        inputs['action_is_pad'][:,-1]=True
        result=arch.compute_loss(**inputs)
        assert torch.isfinite(result['loss'])
        opt.zero_grad();result['loss'].backward();opt.step()
    weight=arch.action_backbone.action_encoder.weight.detach()
    assert not torch.equal(weight[:,:7],original)
    if method=='legato':assert weight[:,-1].abs().sum()>0
    path=tmp_path/'checkpoint.safetensors';arch.save_checkpoint(str(path))
    reloaded=_make_tiny_arch();reloaded.cfg={**reloaded.cfg,'shift_action':1.0}
    configure_architecture(reloaded,settings);reloaded.load_checkpoint(str(path))
    for k,v in arch.state_dict().items():torch.testing.assert_close(v,reloaded.state_dict()[k],rtol=0,atol=0)


def test_base_checkpoint_migration_is_explicit_and_strict(tmp_path):
    base=_make_tiny_arch();path=tmp_path/'base.safetensors';base.save_checkpoint(str(path))
    target=_make_tiny_arch();target.cfg={**target.cfg,'shift_action':1.0}
    configure_architecture(target,{'method':'legato'})
    with pytest.raises(RuntimeError):target.load_checkpoint(str(path))
    target.load_checkpoint(str(path),expand_base_action_input=True)
    old=base.state_dict();new=target.state_dict()
    for k,v in old.items():
        if k=='action_backbone.action_encoder.weight':
            torch.testing.assert_close(v,new[k][:,:-1],rtol=0,atol=0);assert (new[k][:,-1]==0).all()
        else:torch.testing.assert_close(v,new[k],rtol=0,atol=0)


@pytest.mark.parametrize('bad',[{'method':'bad'},{'prefix_min_steps':-1},{'prefix_distribution':'bad'},
    {'num_denoise_steps':0},{'prefix_max_steps':1.5},{'reference':'unspecified'},
    {'schedule':'linear','ramp_steps':0},{'method':'train_time_rtc','schedule':'linear','ramp_steps':1}])
def test_bad_config_rejected(bad):
    with pytest.raises((ValueError,TypeError)):ContinuationConfig.parse(bad)


def test_prefix_must_leave_a_suffix():
    with pytest.raises(ValueError,match='free action'):
        ContinuationConfig(prefix_max_steps=4).sample_lengths(2,4,'cpu')


def test_factory_wires_config_before_loading_weights():
    model=OmegaConf.create({'architecture':{'framework':'dual_system','variant':'joint_cross_attn',
        'action_dim':3,'video_dim':16,'num_layers':1,'num_heads':2,'text_dim':8,'bridge_layers':[0],
        'continuation':{'method':'legato','prefix_max_steps':2}},
        'action_backbone':{'dim':16,'ffn_dim':32,'shift_action':1.0}})
    resolved=resolve_architecture_config(model)
    arch=build_architecture(resolved.registry_name,resolved.params)
    assert arch.continuation.method=='legato'
    assert arch.action_backbone.action_encoder.in_features==4 and arch.action_dim==3


@pytest.mark.parametrize('method',['train_time_rtc','legato'])
def test_joint_self_attention_full_training_path(method):
    from tests.test_openwam_trainer import _MockVideoBackbone
    cfg=dict(framework='dual_system',variant='joint_self_attn',action_dim=7,dim=64,
             ffn_dim=128,num_heads=4,num_layers=2,video_dim=64,text_dim=64,
             bridge_layers=(0,1),shift_action=1.0,mot_checkpoint_mixed_attn=False,
             continuation=dict(method=method,prefix_min_steps=1,prefix_max_steps=2))
    arch=build_architecture('dual_system_self_attn',cfg)
    arch.video_backbone=_MockVideoBackbone(dim=64,num_layers=2,num_heads=4)
    arch.build_mot_driver();arch._device=torch.device('cpu');arch._dtype=torch.float32
    arch.init_training_schedulers(1000)
    inputs=_make_fake_loss_inputs(B=2)
    inputs['actions']=torch.randn(2,5,7)
    inputs['context']=torch.randn(2,3,64)
    output=arch.compute_loss(**inputs)
    output['loss'].backward()
    assert torch.isfinite(output['loss'])
    assert arch.action_backbone.action_encoder.weight.grad.abs().sum()>0


def test_checkpoint_config_round_trip_and_real_loader_warmstart(tmp_path,monkeypatch):
    from openwam.model.architectures.base import BaseWAMArchitecture
    from tests.test_openwam_trainer import _MockVideoBackbone
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    from openwam.train.utils.checkpointing import save_config
    # Only replace heavyweight video construction; factory, ActionDiT and loader are real.
    monkeypatch.setattr(BaseWAMArchitecture,'_init_video_backbone',lambda self,cfg: setattr(self,'video_backbone',_MockVideoBackbone()))
    model=OmegaConf.create({'architecture':{'framework':'dual_system','variant':'joint_cross_attn',
        'action_dim':7,'dim':64,'ffn_dim':128,'num_heads':4,'video_dim':64,'text_dim':64,
        'bridge_layers':[0,1],'continuation':{'method':'none'}},
        'action_backbone':{'shift_action':1.0}, 'video_backbone':{}})
    cfg=OmegaConf.create({'model':model})
    resolved=resolve_architecture_config(model)
    arch=build_architecture(resolved.registry_name,resolved.params)
    arch.save_checkpoint(str(tmp_path/'checkpoint_step_1.safetensors'));save_config(str(tmp_path),cfg)
    live=OmegaConf.create({'model':OmegaConf.to_container(model)})
    live.model.architecture.continuation={'method':'legato','num_denoise_steps':7}
    _, loaded, _ = build_architecture_from_ckpt_dir(str(tmp_path),weights_required=True,override_cfg=live)
    assert loaded.continuation.num_denoise_steps==7
    assert loaded.action_backbone.action_encoder.in_features==8
    torch.testing.assert_close(loaded.action_backbone.action_encoder.weight[:,:7],arch.action_backbone.action_encoder.weight)
    newdir=tmp_path/'legato';newdir.mkdir();save_config(str(newdir),live)
    loaded.save_checkpoint(str(newdir/'checkpoint_step_1.safetensors'))
    _, reread, _=build_architecture_from_ckpt_dir(str(newdir),weights_required=True)
    assert reread.continuation==loaded.continuation
    for key,val in loaded.state_dict().items():torch.testing.assert_close(val,reread.state_dict()[key],rtol=0,atol=0)
