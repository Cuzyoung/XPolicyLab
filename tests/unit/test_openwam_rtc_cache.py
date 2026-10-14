"""Native cache/engine and synthetic flow regressions; no weights or hardware."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from XPolicyLab.policy.OpenWAM.model import Model, _parse_dit_cache_modes, _configure_deploy_runtime
from XPolicyLab.policy.OpenWAM.sampling import JointFlowSampler
from XPolicyLab.utils.sampling_contracts import validate_rtc
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
from openwam.model.architectures.base import BaseWAMArchitecture

ROOT = Path(__file__).resolve().parents[2]


def before(path, name):
    source = subprocess.check_output(['git', 'show', f'b3f170cb11d4dbeec94e9f67d7425ac6c52a70e8:{path}'], cwd=ROOT, text=True)
    # Validators moved out of flow_sampling; baseline samplers only use attention.
    ns = {'__name__': 'baseline_sampler'}
    exec(compile(source, str(path), 'exec'), ns)
    return ns[name]


class Arch:
    device = 'cpu'
    dtype = torch.float32
    def __init__(self, factor=.1):
        self.factor, self.calls = factor, 0
        self.video_scheduler = SimpleNamespace(num_train_timesteps=1000)
        self.action_scheduler = SimpleNamespace(num_train_timesteps=1000,
            flow_step=lambda v, s, n, x: x+(n-s)*v)
        self.mot_driver = SimpleNamespace()
    def forward(self, x, at, **kw):
        self.calls += 1
        return kw['latents'] * .01 + 1, x * self.factor + 1


def grid():
    return [(float(t), float(t)) for t in range(1000, -1, -100)]


def inputs():
    return {'latents': torch.ones(1,2,3,2,2), 'first_frame_latents': torch.ones(1,2,1,2,2)}


@pytest.mark.parametrize('mode', ['rtc','paint'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_disabled_cache_bitwise_matches_before_refactor(mode, dtype):
    old = before('policy/OpenWAM/sampling.py', 'JointFlowSampler')
    noise = torch.arange(16,dtype=dtype).reshape(1,8,2)/10
    kw = dict(target=np.ones((8,2)), weights=np.linspace(1,0,8), beta=5.) if mode=='rtc' else dict(target=np.ones((8,2)),delay=3)
    a,b=inputs(),inputs()
    x=old(Arch(), mode, **kw)(grid(),a,noise)
    y=JointFlowSampler(Arch(),mode,**kw)(grid(),b,noise)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(a['latents'],b['latents'],rtol=0,atol=0)


def test_corrected_cache_skips_both_forward_and_vjp_and_resets(monkeypatch):
    arch=Arch(factor=0)
    cache=DiTVelocityCache()
    seen=[]
    update=cache.update
    def spy(video,sigma,action):
        seen.append(action.detach().clone())
        update(video,sigma,action)
    monkeypatch.setattr(cache,'update',spy)
    vjps=[]; grad=torch.autograd.grad
    def grad_spy(*a,**kw):
        vjps.append(1);return grad(*a,**kw)
    monkeypatch.setattr(torch.autograd,'grad',grad_spy)
    loop=JointFlowSampler(arch,'rtc',target=np.full((8,2),3.),weights=np.ones(8),beta=.1)
    noise=torch.zeros(1,8,2)
    a=inputs();first=loop(grid(),a,noise,dit_cache=cache)
    assert arch.calls==len(vjps)==4
    assert loop.metadata['cache_hits']==6
    assert loop.metadata['guidance_evaluations']==4
    assert loop.metadata['dit_cache']['action_prediction']=='rtc_corrected'
    assert all(not torch.equal(x,torch.ones_like(x)) for x in seen)
    assert cache._cached_action_velocity.grad_fn is None
    assert cache._cached_velocity.grad_fn is None
    torch.testing.assert_close(cache.get_cached_action(),seen[-1],rtol=0,atol=0)
    torch.testing.assert_close(a['latents'][:,:,:1],inputs()['first_frame_latents'])
    assert not torch.equal(a['latents'][:,:,1:],inputs()['latents'][:,:,1:])
    second=loop(grid(),inputs(),noise,dit_cache=cache)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert loop.metadata['cache_hits']==6 and cache.stats['total_steps']==10
    assert arch.calls==8


def test_zero_weights_matches_native_cached_joint_loop():
    arch=Arch(factor=0)
    noise=torch.zeros(1,8,2)
    native_inputs=inputs();rtc_inputs=inputs()
    with torch.no_grad():
        expected=BaseWAMArchitecture._run_joint_denoising(arch,grid(),native_inputs,noise.clone(),
            dit_cache=DiTVelocityCache(),cfg_scale_f=1.,cfg_merge=False,inactive_action_dims=None)
    loop=JointFlowSampler(Arch(factor=0),'rtc',target=np.ones((8,2)),weights=np.zeros(8))
    actual=loop(grid(),rtc_inputs,noise,dit_cache=DiTVelocityCache())
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    torch.testing.assert_close(native_inputs['latents'],rtc_inputs['latents'],rtol=0,atol=0)


def test_cache_max_skip_and_similarity_still_native():
    cache=DiTVelocityCache()
    video=torch.ones(2);action=torch.ones(4)
    for sign in (1,-1):
        cache.update(video,1,sign*action)
    assert cache.should_recompute(.8,require_action=True) # action dissimilar, video similar
    cache.update(video,.8,-action)
    assert [cache.should_recompute(.7,require_action=True) for _ in range(4)]==[False,False,False,True]


@pytest.mark.parametrize('mode',['paint','autohorizon'])
def test_cached_unsupported_loop_rejected(mode):
    with pytest.raises(ValueError,match='Only RTC'):
        JointFlowSampler(Arch(),mode)(grid(),inputs(),torch.zeros(1,8,2),dit_cache=DiTVelocityCache())


@pytest.mark.parametrize('value', [None,'rtc',[None],['paint'],['autohorizon'],['rtc','rtc'],[{}]])
def test_invalid_cache_modes(value):
    with pytest.raises(ValueError,match='dit_cache_modes'):
        _parse_dit_cache_modes({'dit_cache_modes':value})


def test_config_default_and_compile_guard():
    assert _parse_dit_cache_modes({})==('default',)
    assert _parse_dit_cache_modes({'dit_cache_modes':[]})==()
    cfg=OmegaConf.create({'optimization':{'compile':{'enabled':True},'dit_cache':{'enabled':True}}})
    with pytest.raises(ValueError,match='compile_enabled'):
        _configure_deploy_runtime(cfg,{'dit_cache_modes':['rtc']})
    _configure_deploy_runtime(cfg,{'dit_cache_modes':['rtc'],'compile_enabled':False})
    assert cfg.optimization.dit_cache.enabled and not cfg.optimization.compile.enabled
    _configure_deploy_runtime(cfg,{'dit_cache_modes':['rtc'],'eval_batch':True})
    assert not cfg.optimization.dit_cache.enabled


@pytest.mark.parametrize('modes', [[],['default'],['rtc'],['default','rtc']])
def test_adapter_routes_each_mode_without_mutating_engine(modes):
    model=object.__new__(Model);model._dit_cache_modes=tuple(modes)
    engine=SimpleNamespace(generate=lambda x:x,_dit_cache=object())
    model._engine=engine; original=engine._dit_cache
    for mode in ('default','rtc','paint','autohorizon'):
        result=model._generate({'prompt':'test'},mode)
        assert result['use_dit_cache']==(mode in modes)
        assert engine._dit_cache is original


def test_engine_routes_and_resets_cache_without_profiling(monkeypatch):
    import openwam.deploy.engine as engine_module
    monkeypatch.setattr(engine_module,'make_schedule',lambda *a,**kw:grid())
    engine=object.__new__(JointInferenceEngine)
    engine.cfg=OmegaConf.create({'inference':{'denoise_steps':10,'num_frames':9}})
    engine._dit_cache=DiTVelocityCache();engine._decode_video=False;engine._profile=False
    engine._vace_cache=None;engine._prompt_embed_cache=None;engine._cfg_scale=1.;engine._cfg_merge=False
    seen=[]
    def generate(**kw):
        seen.append(kw['dit_cache'])
        assert engine._dit_cache.stats['total_steps']==0
        if kw['dit_cache'] is not None:kw['dit_cache'].should_recompute(1.)
        return {'actions':np.zeros((8,2))}
    engine.architecture=SimpleNamespace(generate=generate,video_scheduler=None,action_scheduler=None)
    # Real filter relies on inspecting generate; **kwargs is allowed by contract.
    for enabled in (True,False,True):
        result=engine.generate({'use_dit_cache':enabled})
        assert result['dit_cache_stats']['enabled']==enabled
        assert (seen[-1] is engine._dit_cache)==enabled
    with pytest.raises(ValueError,match='boolean'):engine.generate({'use_dit_cache':'false'})


def test_validation_is_cpu_only_and_preserves_writable_copy():
    cmd='from XPolicyLab.utils.sampling_contracts import validate_rtc; import sys; assert "torch" not in sys.modules; assert "jax" not in sys.modules'
    subprocess.run([sys.executable,'-c',cmd],cwd=ROOT.parent,check=True)
    x=np.zeros((8,2),dtype=np.float32);w=np.ones(8,dtype=np.float32)
    x.flags.writeable=w.flags.writeable=False
    request={'action_condition':x,'condition_weights':w,'beta':5}
    a,b,_=validate_rtc(request,8,2,copy=True)
    assert a.flags.writeable and b.flags.writeable
    assert not np.shares_memory(a,x) and not np.shares_memory(b,w)
    with pytest.raises(ValueError,match=r'\[0, 1\]'):
        validate_rtc({**request,'condition_weights':np.full(8,2)},8,2)


def test_adapter_ws_rtc_response_and_mode_switching():
    import asyncio
    from client_server.ws.model_server import PolicyServer
    from client_server.ws.protocol.messages import MessageType
    from client_server.ws.protocol.schemas import Frame
    arch=Arch(factor=0);arch.normalizer=None
    model=object.__new__(Model)
    model.model_cfg={'inference_seed':0};model.allow_dummy_policy=False
    model._metadata={};model._compile_enabled=False;model.eval_batch=False
    model._dit_cache_modes=('default','rtc');model._last_sampling_stats={}
    model.observation_profile='yam_base';model.replan_steps=8
    cache=DiTVelocityCache()
    def generate(conditions):
        loop=conditions['denoise_loop']
        actions=loop(grid(),inputs(),torch.zeros(1,8,20),dit_cache=cache if conditions['use_dit_cache'] else None)
        return {'actions':actions[0]}
    arch.generate=lambda denoise_loop=None:None
    model._engine=SimpleNamespace(generate=generate,architecture=arch,_dit_cache=cache,_cfg_scale=1.,
        cfg=SimpleNamespace(inference=SimpleNamespace(num_frames=9,denoise_steps=10)))
    model._wam_policy=SimpleNamespace(_project_binary_dims=lambda x:x)
    model._eef20_chunk_to_native=lambda x:x
    model._conditions=lambda obs:dict(obs)
    def update_obs(obs):model._order=[0];model._batch={0:obs}
    model.update_obs=update_obs
    async def run():
        server=PolicyServer(model)
        hello=await server.process_frame(Frame(message_type=MessageType.HELLO,request_id='h',evaluation_id='e',payload={}))
        assert 'rtc' in hello.payload['capabilities']['sampling_modes']
        reply=await server.process_frame(Frame(message_type=MessageType.INFER,request_id='i',evaluation_id='e',payload={
            'observation':{'state':{},'vision':{}},
            'sampling':{'mode':'rtc','action_condition':np.full((8,20),3.),'condition_weights':np.ones(8),'beta':.1}}))
        assert reply.message_type==MessageType.INFER_RESULT,reply.payload
        return reply
    reply=asyncio.run(run())
    assert reply.payload['sampling_stats']['cache_hits']==6
