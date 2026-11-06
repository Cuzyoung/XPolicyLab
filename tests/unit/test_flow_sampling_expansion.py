from types import SimpleNamespace
import numpy as np
import pytest
import torch

from XPolicyLab.utils.flow_sampling import (EulerSampler, PaintSampler, RtcSampler,
    AutoHorizonSampler, action_attention)
from XPolicyLab.utils.sampling_contracts import validate_samples, validate_prefix
from XPolicyLab.policy.OpenWAM.sampling import JointFlowSampler, normalize_action
from XPolicyLab.policy.DP.rtc import guided_prediction


def test_paint_inverts_prefix_and_retains_free_suffix_with_three_passes():
    calls = []
    def velocity(x,t):
        calls.append(t)
        return torch.ones_like(x)*2
    noise = torch.arange(8.).reshape(1,4,2)
    mask = torch.zeros_like(noise,dtype=torch.bool);mask[:,:2]=True
    target = torch.full_like(noise,7)
    result=PaintSampler(4).sample(velocity,noise,target=target,mask=mask)
    assert len(calls)==12
    torch.testing.assert_close(result[:,:2],target[:,:2])
    torch.testing.assert_close(result[:,2:],noise[:,2:]+2)
    torch.testing.assert_close(noise,torch.arange(8.).reshape(1,4,2))


def test_rtc_zero_weights_is_native_flow_and_guidance_reduces_error():
    noise=torch.zeros(1,4,2);target=torch.ones_like(noise)
    vel=lambda x,t:x*0
    plain=EulerSampler(10).sample(vel,noise)
    unguided=RtcSampler(10).sample(vel,noise,target=target,weights=noise,beta=5)
    guided=RtcSampler(10).sample(vel,noise,target=target,weights=target,beta=5)
    torch.testing.assert_close(plain,unguided)
    assert torch.linalg.norm(guided-target)<torch.linalg.norm(noise-target)


def test_autohorizon_only_captures_third_call_and_keeps_actions_identical():
    owner=SimpleNamespace(); calls=[]
    def velocity(x,t):
        capture=getattr(owner,'_action_attention_capture',None)
        if capture:
            calls.append(t); q=torch.eye(4).reshape(1,1,4,4)
            capture(q,q)
        return x*.1+1
    noise=torch.zeros(1,4,2)
    actions,metadata=AutoHorizonSampler(5).sample(velocity,noise,attention_owners=[owner])
    torch.testing.assert_close(actions,EulerSampler(5).sample(velocity,noise))
    assert calls==[.4] and owner._action_attention_capture is None
    assert 1<=metadata['execution_steps']<=4


def test_attention_preserves_full_softmax_mass_and_mask():
    q=torch.ones(1,2,3,2);k=torch.ones(1,1,5,2)
    mask=torch.ones(1,1,3,5,dtype=torch.bool);mask[...,-1]=False
    result=action_attention(q,k,horizon=2,mask=mask)
    torch.testing.assert_close(result,torch.tensor([[.25,0],[.25,0]]))


@pytest.mark.parametrize('count',[True,1,0,2.2])
def test_bad_sample_count(count):
    with pytest.raises(ValueError):validate_samples({'num_samples':count})


@pytest.mark.parametrize('delay',[0,4,True,1.2])
def test_bad_paint_delay(delay):
    with pytest.raises(ValueError):validate_prefix({'delay_steps':delay,'action_prefix':np.zeros((1,2))},4,2)


@pytest.mark.parametrize('kind',['epsilon','sample','v_prediction'])
def test_dp_zero_weight_preserves_native_prediction_and_target_moves_clean(kind):
    alpha=torch.tensor([.5]); scheduler=SimpleNamespace(config=SimpleNamespace(prediction_type=kind),alphas_cumprod=alpha)
    x=torch.zeros(1,3,2);target=torch.ones_like(x)
    model=lambda sample,t,**kwargs:sample*.1
    native=model(x,0)
    zero=guided_prediction(model,x,0,scheduler,target=target,weights=x,beta=5)
    torch.testing.assert_close(zero,native)
    guided=guided_prediction(model,x,0,scheduler,target=target,weights=target,beta=5)
    if kind=='sample':clean=guided
    elif kind=='epsilon':clean=(x-(.5**.5)*guided)/(.5**.5)
    else:clean=(.5**.5)*(x-guided)
    assert torch.mean(clean)>0


class JointModel:
    device='cpu';dtype=torch.float32
    video_scheduler=SimpleNamespace(num_train_timesteps=1000)
    action_scheduler=SimpleNamespace(num_train_timesteps=1000)
    mot_driver=SimpleNamespace()
    def __init__(self):self.calls=[]
    def forward(self,x,a_timestep,*,latents,timestep,**kwargs):
        self.calls.append((float(a_timestep[0]),float(timestep[0])))
        capture=getattr(self.mot_driver,'_action_attention_capture',None)
        if capture:
            q=torch.eye(x.shape[1]).reshape(1,1,x.shape[1],x.shape[1]);capture(q,q)
        return latents*0+2,x*0+1


def test_wam_paint_inverts_both_streams_on_shifted_grid():
    arch=JointModel();noise=torch.zeros(1,4,2)
    inputs={'latents':torch.zeros(1,1,2,1,1),'first_frame_latents':torch.ones(1,1,1,1,1)*9}
    grid=[(1000.,1000.),(800.,600.),(300.,200.),(0.,0.)]
    loop=JointFlowSampler(arch,'paint',target=np.ones((4,2))*3,delay=1)
    result=loop(grid,inputs,noise)
    torch.testing.assert_close(result[:,:1],torch.ones(1,1,2)*3)
    torch.testing.assert_close(result[:,1:],torch.ones(1,3,2)*-1)
    assert len(arch.calls)==9 and arch.calls[3]==(0.,0.)
    assert inputs['latents'][0,0,0,0,0]==9
    assert loop.metadata['model_evaluations']==9


def test_wam_rtc_zero_weights_and_attention_keep_joint_loop():
    grid=[(1000.,1000.),(800.,600.),(300.,200.),(0.,0.)]
    noise=torch.zeros(1,4,2)
    for mode in ['rtc','autohorizon']:
        loop=JointFlowSampler(JointModel(),mode,target=np.zeros((4,2)) if mode=='rtc' else None,
                              weights=np.zeros(4) if mode=='rtc' else None)
        result=loop(grid,{'latents':torch.zeros(1,1,2,1,1)},noise)
        torch.testing.assert_close(result,torch.full_like(noise,-1))
        if mode=='autohorizon':assert loop.metadata['denoising_step']==3


def test_wam_normalizes_with_action_not_state_map():
    normalizer=SimpleNamespace(_action_dst_index=np.array([1,3]),_state_dst_index=np.array([0,2]),
                              _inner=None,_unify_dim=4)
    def scatter(x,index,width):
        out=np.zeros((*x.shape[:-1],width));out[...,index]=x
        return out,None
    normalizer._map_to_unify=scatter
    np.testing.assert_array_equal(normalize_action(SimpleNamespace(normalizer=normalizer),[[2,5]]),[[0,2,0,5]])


def test_dp_euler_codec_does_not_use_joint_width_for_quaternion_state():
    from XPolicyLab.policy.DP.model import Model, encode_obs
    from scipy.spatial.transform import Rotation
    state={}
    for side in ('left','right'):
        quat=Rotation.from_euler('xyz',[.1,-.2,.3]).as_quat()
        state[f'{side}_ee_pose']=np.r_[[.2,.3,.4],quat[3],quat[:3]]
        state[f'{side}_ee_joint_state']=np.array([.5])
    obs={'state':state,'vision':{name:{'color':np.zeros((6,6,3),dtype=np.uint8)}
        for name in ('cam_head','cam_left_wrist','cam_right_wrist')}}
    encoded=encode_obs(obs,'ee',{'arm_dim':[6,6],'ee_dim':[1,1]},'absolute_xyz_euler')
    assert encoded['agent_pos'].shape==(14,)
    model=object.__new__(Model); model.model_cfg={'eef_representation':'absolute_xyz_euler'}
    decoded=model._decode_actions(encoded['agent_pos'][None])[0]
    for key in state: np.testing.assert_allclose(decoded[key],state[key],atol=1e-8)


def test_lingbot_aac_samples_same_observation_without_updating_it():
    from XPolicyLab.policy.LingBot_VLA2.model import Model
    model=object.__new__(Model); observation={'test':'same'}; model._observations=[observation]
    calls=[]
    def infer(obs):
        assert obs is observation
        calls.append(obs)
        return {'actions':len(calls),'action_semantics':'anchor_relative_arm_absolute_gripper'}
    model._infer_one=infer
    result=model.get_action_aac({'mode':'aac','num_samples':4})
    assert [c['actions'] for c in result['actions']]==[1,2,3,4]
    assert model._observations==[observation]


def test_lingbot_compiled_attention_requires_functional_predictor_and_full_horizon():
    from XPolicyLab.policy.LingBot_VLA2.model import Model
    model=object.__new__(Model);model.action_horizon=4;model.model_cfg={'use_compile':False}
    model.model=SimpleNamespace(vla=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(num_steps=5,n_action_steps=4),get_predict_velocity_fn=lambda:None)))
    assert set(model.sampling_modes())=={'default','paint','aac','autohorizon'}
    assert not hasattr(model, 'get_action_rtc')
    model.model_cfg['use_compile']=True
    assert 'autohorizon' in model.sampling_modes()
    del model.model.vla.model.get_predict_velocity_fn
    assert 'autohorizon' not in model.sampling_modes()
    model.action_horizon=2
    assert model.sampling_modes()==['default','aac']


def test_openwam_specialized_seeds_are_independent_and_reset_reproducible():
    from XPolicyLab.policy.OpenWAM.model import Model
    model=object.__new__(Model);model._engine=None;model.model_cfg={'inference_seed':12};model._conditions=lambda obs:dict(obs)
    def run():
        model.reset();model._order=[0];model._batch={0:{'proprio':'same'}}
        return [model._sampling_conditions() for _ in range(5)]
    first=run();assert first==run()
    assert len({r['seed'] for r in first})==5
    assert all(r['proprio']=='same' for r in first)


def test_openwam_capabilities_match_cache_cfg_and_full_horizon():
    from XPolicyLab.policy.OpenWAM.model import Model
    model=object.__new__(Model)
    model.allow_dummy_policy=False;model.observation_profile='yam_base'
    model.model_cfg={'compile_enabled':False};model._compile_enabled=False;model.eval_batch=False;model.replan_steps=4
    model._engine=SimpleNamespace(_dit_cache=None,_cfg_scale=1.,
        architecture=SimpleNamespace(generate=lambda denoise_loop=None:None,mot_driver=object()),
        cfg=SimpleNamespace(inference=SimpleNamespace(num_frames=5,denoise_steps=5)))
    assert set(model.sampling_modes())=={'default','rtc','paint','aac','autohorizon'}
    model._engine._cfg_scale=2.
    assert model.sampling_modes()==['default','aac']
    model._engine._cfg_scale=1.;model._engine._dit_cache=object()
    assert set(model.sampling_modes())=={'default','rtc','paint','aac','autohorizon'}
    model._compile_enabled=True
    assert model.sampling_modes()==['default','aac','rtc']
    model.replan_steps=2
    assert model.sampling_modes()==['default']
