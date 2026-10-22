"""Offline reference arithmetic, native dispatch and XPolicyLab wire tests."""
import asyncio
import copy
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from openwam.model.continuation import ContinuationConfig, configure_architecture
from openwam.model.continuation_sampling import ContinuationFlowSampler
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.deploy.denoise_schedule import make_schedule
from tests.test_openwam_trainer import _make_tiny_arch, _make_fake_loss_inputs

POLICY=os.environ['CONTINUATION_POLICY']
adapter=importlib.import_module(f'XPolicyLab.policy.{POLICY}.model')
request=importlib.import_module(f'XPolicyLab.policy.{POLICY}.sampling')

class FakeArch:
    def __init__(self, method='legato', **kw):
        self.continuation=ContinuationConfig(method=method,prefix_max_steps=3,num_denoise_steps=4,**kw)
        self.action_scheduler=SimpleNamespace(num_train_timesteps=1000)
        self.video_scheduler=SimpleNamespace(num_train_timesteps=1000)
        self.normalizer=None
        self.seen=[]
    def forward(self,x,at,**inputs):
        self.seen.append((x.clone(),at.clone(),inputs['latents'].clone()))
        a=x[...,:2]
        # Coupling makes lost prefix/time/video updates observable in the suffix.
        velocity=.1*a + a.mean(dim=1,keepdim=True)*.03 + at.reshape(1,-1,1)/10000
        return inputs['latents']*.1+1, velocity

def grid():return [(1000.,1000.),(850.,750.),(600.,500.),(300.,250.),(0.,0.)]
def inputs():return {'latents':torch.ones(1,1,2,1,1),'first_frame_latents':torch.full((1,1,1,1,1),7.)}

@pytest.mark.parametrize('method',['train_time_rtc','legato'])
@pytest.mark.parametrize('prefix',[0,2])
def test_exact_reference_updates_and_native_video(method,prefix):
    arch=FakeArch(method); x=torch.arange(10.).reshape(1,5,2)/10; target=torch.full_like(x,3.)
    state=inputs(); refstate=copy.deepcopy(state); ref=x.clone(); oracle=FakeArch(method)
    mask=(torch.arange(5)<prefix)[None,:,None]
    for (tv,ta),(vnext,anext) in zip(grid(),grid()[1:]):
        guided=torch.where(mask,target,ref)
        t=torch.tensor([ta]) if method=='legato' else torch.where(mask[...,0],0.,ta)
        inp=torch.cat((guided,(~mask).float()),-1) if method=='legato' else guided
        vv,va=oracle.forward(inp,t,**refstate,timestep=torch.tensor([tv]))
        ref=guided+(anext-ta)/1000*va
        vid=refstate['latents']+(vnext-tv)/1000*vv
        vid[:,:,:1]=refstate['first_frame_latents'];refstate['latents']=vid
    loop=ContinuationFlowSampler(arch,target=target if prefix else None,prefix_steps=prefix)
    out=loop(grid(),state,x)
    torch.testing.assert_close(out,ref,rtol=0,atol=1e-7)
    torch.testing.assert_close(state['latents'],refstate['latents'])
    assert len(arch.seen)==4 and not out.requires_grad
    if prefix:assert not torch.equal(out[:,:prefix],target[:,:prefix]) # Reference has no final clamp.
    for seen,expected in zip(arch.seen,oracle.seen):
        torch.testing.assert_close(seen[0],expected[0]);torch.testing.assert_close(seen[1],expected[1])

@pytest.mark.parametrize('method',['train_time_rtc','legato'])
def test_native_default_dispatch_has_correct_input_width(method):
    arch=FakeArch(method)
    out=BaseWAMArchitecture._run_joint_denoising(arch,grid(),inputs(),torch.zeros(1,5,2))
    assert out.shape==(1,5,2)
    assert arch.seen[0][0].shape[-1]==(3 if method=='legato' else 2)

@pytest.mark.parametrize('bad', ['count','nonuniform','cache','cfg','no_target','full_prefix','wrong_shape'])
def test_invalid_sampling_contracts(bad):
    arch=FakeArch(); schedule=grid();kw={}; opts={}
    if bad=='count':schedule=schedule[:-2]+[schedule[-1]]
    if bad=='nonuniform':schedule[1]=(850.,800.)
    if bad=='cache':kw['dit_cache']=object()
    if bad=='cfg':kw['cfg_scale_f']=2.
    if bad=='no_target':opts['prefix_steps']=1
    if bad=='full_prefix':opts['prefix_steps']=5
    if bad=='wrong_shape':opts.update(prefix_steps=1,target=np.ones((5,3)))
    with pytest.raises(ValueError):ContinuationFlowSampler(arch,**opts)(schedule,inputs(),torch.zeros(1,5,2),**kw)
    assert arch.seen==[]

def test_inactive_dimensions_retain_analytic_noise_path():
    arch=FakeArch();noise=torch.ones(1,5,2)
    out=ContinuationFlowSampler(arch,target=torch.zeros_like(noise),prefix_steps=2)(
        grid(),inputs(),noise,inactive_action_dims=torch.tensor([False,True]))
    assert torch.equal(out[...,1],torch.zeros(1,5))
    # First call has a clean prefix. Free padding follows the noise path.
    assert arch.seen[1][0][0,3,1]==.75

@pytest.mark.parametrize('method',['train_time_rtc','legato'])
def test_real_action_backbone_train_save_load_then_generate(tmp_path,method,monkeypatch):
    torch.manual_seed(4)
    arch=_make_tiny_arch();arch.cfg={**arch.cfg,'shift_action':1.}
    cfg={'method':method,'prefix_max_steps':2,'num_denoise_steps':4}
    configure_architecture(arch,cfg);arch.init_training_schedulers(1000)
    train=_make_fake_loss_inputs(B=1);train['actions']=torch.randn(1,5,7)
    loss=arch.compute_loss(**train)['loss'];loss.backward()
    opt=torch.optim.AdamW(arch.parameters(),lr=1e-4);opt.step()
    ckpt=tmp_path/'checkpoint.safetensors';arch.save_checkpoint(str(ckpt))
    loaded=_make_tiny_arch();loaded.cfg={**loaded.cfg,'shift_action':1.};configure_architecture(loaded,cfg)
    loaded.load_checkpoint(str(ckpt));loaded.eval()
    def prep(**kw):
        return {'latents':train['input_latents'].clone(),'context':torch.zeros(1,2,64)}
    monkeypatch.setattr(loaded.video_backbone,'preprocess_input_for_inference',prep)
    target=np.ones((5,7),np.float32)
    result=loaded.generate(grid(),'test',num_frames=6,action_num_frames=6,decode_video=False,
        denoise_loop=ContinuationFlowSampler(loaded,target=target,prefix_steps=2))
    assert result['actions'].shape==(5,7) and np.isfinite(result['actions']).all()
    # Default generation must also supply Legato's additional channel.
    default=loaded.generate(grid(),'test',num_frames=6,decode_video=False)
    assert default['actions'].shape==(5,7) and np.isfinite(default['actions']).all()

@pytest.mark.parametrize('prefix',[0,2])
def test_existing_request_contract_and_normalization(prefix):
    arch=FakeArch();h=5
    payload={'mode':'train_time_rtc','prefix_steps':prefix}
    if prefix:payload.update(action_condition=np.ones((h,20)),condition_weights=np.arange(h)<prefix)
    target,count=request.parse_request(arch,payload,h)
    assert count==prefix
    if prefix:assert target.shape==(h,20)
    else:assert target is None
    bad={**payload,'beta':5}
    with pytest.raises(ValueError):request.parse_request(arch,bad,h)


def test_normalization_uses_action_mapping_not_proprio():
    from openwam.deploy.model_loader import _UnifyAwareNormalizer
    from openwam.dataloader.transforms.normalize import Normalizer
    inner=Normalizer('mean_std', {'mean':np.array([1.,3.]),'std':np.array([2.,4.]),
        '_normalize_stats':{'mean':np.array([10.,30.]),'std':np.array([20.,40.])}})
    norm=_UnifyAwareNormalizer(inner,np.array([2,0]),4,state_dst_index=np.array([1,3]))
    raw=np.array([[3.,11.]])
    result=request.normalize_action(SimpleNamespace(normalizer=norm),raw)
    np.testing.assert_array_equal(result,[[2.,0.,1.,0.]])
    np.testing.assert_allclose(norm.unnormalize(result),raw)
    assert not np.array_equal(result,norm.normalize(raw))


def test_real_adapter_and_shared_server_prefix_dispatch():
    from XPolicyLab.client_server.ws.model_server import PolicyServer
    from XPolicyLab.client_server.ws.protocol.messages import MessageType
    from XPolicyLab.client_server.ws.protocol.schemas import Frame
    from openwam.dataloader.utils import poses
    model=object.__new__(adapter.Model)
    model.model_cfg={};model.allow_dummy_policy=False;model.observation_profile='yam_base'
    model.replan_steps=5;model.eval_batch=False;model._calibration=None;model._poses=poses
    model.default_instruction='test';model._format_prompt=lambda x:x
    model._preprocessor=SimpleNamespace(preprocess=lambda x:{**x,'image':x['images']['head_camera']})
    cfg=ContinuationConfig(method=adapter.CONTINUATION_METHOD,prefix_max_steps=3,num_denoise_steps=4)
    arch=SimpleNamespace(continuation=cfg,normalizer=None)
    def generate(conditions):
        assert conditions['use_dit_cache'] is False
        loop=conditions['denoise_loop'];assert loop.cfg==cfg and loop.prefix_steps==2
        np.testing.assert_array_equal(loop.target[:2],np.ones((2,20)))
        loop.metadata={'method':cfg.method,'prefix_steps':2}
        physical=np.tile(np.array([0,0,0,1,0,0,0,1,0,.5]*2,np.float32),(5,1))
        return {'actions':physical}
    model._engine=SimpleNamespace(_dit_cache=None,architecture=arch,cfg=SimpleNamespace(inference=SimpleNamespace(num_frames=6)),generate=generate)
    model._wam_policy=SimpleNamespace(_project_binary_dims=lambda x:x)
    model.reset()
    observation={'vision':{k:{'color':np.zeros((8,8,3),np.uint8)} for k in ('cam_head','cam_left_wrist','cam_right_wrist')},
                 'state':{f'{arm}_ee_pose':np.array([0,0,0,1,0,0,0]) for arm in ('left','right')}}
    observation['state'].update({f'{arm}_ee_joint_state':np.array([.5]) for arm in ('left','right')})
    frame=Frame(message_type=MessageType.INFER,request_id='c',evaluation_id='c',payload={'observation':observation,
        'sampling':{'mode':'train_time_rtc','prefix_steps':2,'action_condition':np.ones((5,20)),
                    'condition_weights':np.array([1,1,0,0,0])}})
    reply=asyncio.run(PolicyServer(model)._handle_infer(frame))
    assert reply.payload['sampling_stats']['method']==cfg.method
    assert len(reply.payload['actions'])==5
    assert reply.payload['actions'][0]['left_ee_pose'].shape==(7,)
    assert model.sampling_modes()==['default','train_time_rtc']

@pytest.mark.parametrize('method',['train_time_rtc','legato'])
def test_actual_deployment_loader_reconstructs_trained_input_width(tmp_path,monkeypatch,method):
    from openwam.model.architectures.registry import resolve_architecture_config,build_architecture
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from tests.test_openwam_trainer import _MockVideoBackbone
    monkeypatch.setattr(BaseWAMArchitecture,'_init_video_backbone',
        lambda self,cfg:setattr(self,'video_backbone',_MockVideoBackbone()))
    cfg=OmegaConf.create({'model':{'architecture':{'framework':'dual_system','variant':'joint_cross_attn',
        'action_dim':7,'dim':64,'ffn_dim':128,'num_heads':4,'video_dim':64,'text_dim':64,
        'bridge_layers':[0,1],'continuation':{'method':method,'prefix_max_steps':2}},
        'action_backbone':{'shift_action':1.},'video_backbone':{}},
        'training':{'mixed_precision':'no'},'dataloader':{'normalize_mode':None}})
    resolved=resolve_architecture_config(cfg.model)
    trained=build_architecture(resolved.registry_name,resolved.params)
    trained.save_checkpoint(str(tmp_path/'checkpoint_step_10.safetensors'))
    OmegaConf.save(cfg,str(tmp_path/'config.yaml'))
    _,deployed=load_from_checkpoint_dir(str(tmp_path),device='cpu')
    assert deployed.continuation.method==method
    assert deployed.action_backbone.action_encoder.in_features==(8 if method=='legato' else 7)
    assert not any(p.is_meta for p in deployed.parameters())
    for k,v in trained.state_dict().items():torch.testing.assert_close(v,deployed.state_dict()[k],rtol=0,atol=0)


def test_native_shifted_video_and_uniform_legato_action_schedule():
    from openwam.model.action_backbone.scheduler import ActionScheduler
    arch=FakeArch()
    schedule=make_schedule('sync',video_scheduler=ActionScheduler(),action_scheduler=ActionScheduler(),
                          num_steps=4,shift=1.,shift_video=5.)
    assert schedule[1][0]!=schedule[1][1]
    result=ContinuationFlowSampler(arch)(schedule,inputs(),torch.zeros(1,5,2))
    assert torch.isfinite(result).all()

@pytest.mark.parametrize('method',['none','wrong_method'])
def test_deployment_rejects_mismatched_checkpoint_before_weights(tmp_path,method):
    (tmp_path/'checkpoint_step_10.safetensors').write_bytes(b'not read')
    cfg={'model':{'architecture':{'continuation':{'method':method}}},'dataloader':{'num_frames':6}}
    OmegaConf.save(OmegaConf.create(cfg),str(tmp_path/'config.yaml'))
    with pytest.raises(ValueError):adapter.validate_deployment({'observation_profile':'arx_x5_sim',
        'env_cfg_type':'arx_x5','ckpt_dir':str(tmp_path)})
