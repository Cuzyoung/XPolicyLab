import dataclasses
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.linen as nn
from openpi.models.continuation import ContinuationConfig, training_inputs, loss_per_token, integrate, sample_prefix
from openpi.models import pi0, pi0_config, gemma, model as base
from openpi.training.weight_loaders import expand_legato_input


@pytest.mark.parametrize('method', ['none','train_time_rtc','legato'])
def test_training_equations_and_loss_gradient(method):
    cfg=ContinuationConfig(method=method, prefix_max_steps=2)
    actions=jnp.arange(24,dtype=jnp.float32).reshape(2,4,3)/7
    noise=-actions-.2; sigma=jnp.array([.2,.7]); omega=jnp.array([[1,0,0,0],[1,1,0,0]],dtype=jnp.float32)[...,None]
    x,t,target=training_inputs(actions,noise,sigma,omega,cfg)
    original=sigma[:,None,None]*noise+(1-sigma[:,None,None])*actions
    if method=='none':
        np.testing.assert_allclose(x,original);np.testing.assert_array_equal(target,noise-actions)
    elif method=='train_time_rtc':
        np.testing.assert_allclose(x,np.where(omega,actions,original))
        np.testing.assert_allclose(t,[[0,.2,.2,.2],[0,0,.7,.7]])
    else:
        # Independent oracle: public code time tau=1-sigma and velocity A-eps.
        tau=1-np.asarray(sigma)[:,None,None]; w=1-np.asarray(omega)
        expected=-(np.asarray(actions-noise)*(1+(1-w)*10*(1-tau)))
        np.testing.assert_allclose(target,expected,rtol=2e-6)
        np.testing.assert_array_equal(x[...,-1:],w)
    pred=target+jnp.arange(24,dtype=jnp.float32).reshape(2,4,3)/10
    loss=lambda v:loss_per_token(v,target,omega,cfg).mean()
    grad=jax.grad(loss)(pred)
    if method=='train_time_rtc':
        valid=np.broadcast_to(np.asarray(omega)==0,pred.shape)
        np.testing.assert_allclose(loss(pred),np.square(np.asarray(pred-target))[valid].mean(),rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(grad)[~valid],0)
        official=dataclasses.replace(cfg,rtc_loss_reduction='official_token_sum')
        np.testing.assert_allclose(loss_per_token(pred,target,omega,official).mean(),loss(pred)*3,rtol=1e-6)
    else:
        np.testing.assert_allclose(loss(pred),jnp.square(pred-target).mean(),rtol=1e-6)


@pytest.mark.parametrize('method',['train_time_rtc','legato'])
def test_sampler_matches_public_euler_updates(method):
    cfg=ContinuationConfig(method=method,num_denoise_steps=5)
    noise=jnp.arange(12,dtype=jnp.float32).reshape(1,4,3)/11
    target=jnp.full_like(noise,2.);omega=jnp.array([1,1,0,0],dtype=jnp.float32)[None,:,None]
    # Coupled velocity exposes whether suffix sees the guided prefix, and whether
    # the next state starts at guided_x rather than the previous free state.
    def velocity(x,t):return .2*x[...,:3]+.1*jnp.mean(x[...,:3],axis=1,keepdims=True)
    got=jax.jit(lambda x:integrate(velocity,x,target,omega,cfg,5))(noise)
    expected=np.asarray(noise)
    for _ in range(5):
        guided=np.asarray(omega*target)+(1-np.asarray(omega))*expected
        v=.2*guided+.1*guided.mean(axis=1,keepdims=True)
        expected=guided-v/5
    np.testing.assert_allclose(got,expected,rtol=1e-6)
    assert not np.array_equal(np.asarray(got[:,:2]),np.asarray(target[:,:2])) # No invented final clamp.
    if method=='legato':
        with pytest.raises(ValueError):integrate(velocity,noise,target,omega,cfg,4)


@pytest.mark.parametrize('distribution',['exp','bell','uniform'])
def test_prefix_range_and_per_example_sampling(distribution):
    cfg=ContinuationConfig(method='train_time_rtc',prefix_min_steps=1,prefix_max_steps=3,prefix_distribution=distribution)
    mask=np.asarray(sample_prefix(jax.random.key(5),500,8,cfg))[...,0]
    counts=mask.sum(1)
    assert set(counts)=={1,2,3}
    assert (np.diff(mask,axis=1)<=0).all()


def test_weight_migration_is_exact_and_strict():
    kernel=np.arange(15,dtype=np.float32).reshape(3,5)
    loaded={'action_in_proj':{'kernel':kernel,'bias':np.arange(5,dtype=np.float32)}}
    ref={'action_in_proj':{'kernel':np.empty((4,5),dtype=np.float32),'bias':np.empty(5,dtype=np.float32)}}
    out=expand_legato_input(loaded,ref)
    np.testing.assert_array_equal(out['action_in_proj']['kernel'][:3],kernel)
    np.testing.assert_array_equal(out['action_in_proj']['kernel'][3],0)
    np.testing.assert_array_equal(out['action_in_proj']['bias'],loaded['action_in_proj']['bias'])
    bad={'action_in_proj':{'kernel':np.empty((4,6)),'bias':np.empty(5)}}
    with pytest.raises(ValueError):expand_legato_input(loaded,bad)


def test_per_token_norm_and_time_embedding():
    x=jnp.arange(48,dtype=jnp.float32).reshape(2,3,8)/20
    cond=jnp.arange(16,dtype=jnp.float32).reshape(2,8)/10
    norm=gemma.RMSNorm()
    params=norm.init(jax.random.key(0),x,cond)
    params['params']['Dense_0']['kernel']=jnp.ones((8,24))*.03
    global_out=norm.apply(params,x,cond)[0]
    token_out=norm.apply(params,x,jnp.repeat(cond[:,None,:],3,axis=1))[0]
    np.testing.assert_allclose(global_out,token_out,rtol=1e-6)
    changed=norm.apply(params,x,jnp.repeat(cond[:,None,:],3,axis=1).at[:,0].set(0))[0]
    assert not np.allclose(changed[:,0],token_out[:,0])
    np.testing.assert_allclose(changed[:,1:],token_out[:,1:])
    time=jnp.array([.2,.7])
    emb=pi0.posemb_sincos(time,8,.004,4.)
    token=pi0.posemb_sincos(jnp.repeat(time[:,None],3,axis=1),8,.004,4.)
    np.testing.assert_allclose(token,jnp.repeat(emb[:,None,:],3,axis=1))


class TinyImage(nn.Module):
    @nn.compact
    def __call__(self,image,train=False):
        return nn.Dense(16)(jnp.mean(image,axis=(1,2)))[:,None,:],None


@pytest.mark.parametrize('method',['train_time_rtc','legato'])
@pytest.mark.parametrize('dtype',['float32','bfloat16'])
def test_real_pi05_attention_loss_jit_sampler_and_reload(monkeypatch,method,dtype):
    # Real Pi0, Gemma attention, FFN, adaRMS and KV-cache; only shrink dimensions
    # and replace the unchanged vision encoder to fit a CPU validation machine.
    monkeypatch.setattr(gemma,'PALIGEMMA_VOCAB_SIZE',32)
    monkeypatch.setattr(gemma,'get_config',lambda _:gemma.Config(16,1,32,2,1,8))
    monkeypatch.setattr(pi0._siglip,'Module',lambda **kw:TinyImage())
    cfg=pi0_config.Pi0Config(pi05=True,dtype=dtype,action_dim=3,action_horizon=4,max_token_len=2,
        continuation=ContinuationConfig(method=method,prefix_max_steps=2,num_denoise_steps=3))
    model=cfg.create(jax.random.key(1))
    obs=base.Observation(images={k:jnp.zeros((2,224,224,3)) for k in base.IMAGE_KEYS},image_masks={k:jnp.ones(2,dtype=bool) for k in base.IMAGE_KEYS},
        state=jnp.zeros((2,3)),tokenized_prompt=jnp.ones((2,2),dtype=jnp.int32),tokenized_prompt_mask=jnp.ones((2,2),dtype=bool))
    actions=jnp.arange(24,dtype=jnp.float32).reshape(2,4,3)/20
    @nnx.jit
    def update(m):
        loss,grads=nnx.value_and_grad(lambda net:net.compute_loss(jax.random.key(3),obs,actions).mean())(m)
        params=nnx.state(m,nnx.Param)
        nnx.update(m,jax.tree.map(lambda p,g:p-.001*g,params,grads))
        return loss,grads
    loss,grads=update(model)
    assert np.isfinite(loss)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    omega=jnp.array([1,0,0,0],dtype=jnp.float32)[None,:,None].repeat(2,axis=0)
    sample=nnx.jit(lambda m:m.sample_actions(jax.random.key(4),obs,num_steps=3,action_condition=actions,condition_weights=omega))
    out=sample(model)
    assert out.shape==(2,4,3) and np.isfinite(out).all()
    # Serialize parameter arrays, reconstruct the actual model and repeat sampler.
    import tempfile
    import flax.serialization
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'params.msgpack'
        state=nnx.state(model,nnx.Param).to_pure_dict()
        path.write_bytes(flax.serialization.msgpack_serialize(jax.tree.map(np.asarray,state)))
        restored=cfg.load(flax.serialization.msgpack_restore(path.read_bytes()))
        np.testing.assert_allclose(sample(restored),out,rtol=1e-5,atol=1e-6)
    unconditioned=nnx.jit(lambda m:m.sample_actions(jax.random.key(4),obs,num_steps=3))(model)
    assert np.isfinite(unconditioned).all()


def test_recipe_preserves_native_data_and_optimizer():
    from openpi.training import config as recipes
    original=recipes.get_config('pi05_yam')
    candidates=[c for c in recipes._CONFIGS if c.name in {'pi05_yam_train_time_rtc','pi05_yam_legato'}]
    assert len(candidates)==1
    new=candidates[0]
    for field in dataclasses.fields(original):
        if field.name not in {'name','model','data','weight_loader'}:
            assert getattr(new,field.name)==getattr(original,field.name)
    assert new.data.use_delta_joint_actions == original.data.use_delta_joint_actions
    assert new.data.assets.assets_dir==str(original.assets_dirs)
    assert new.model.continuation.method in {'train_time_rtc','legato'}
    assert new.model.action_horizon==50 and new.model.action_dim==32


def test_request_validation_and_physical_normalization():
    # Import the request parser without loading a checkpoint or model adapter.
    spec=importlib.util.spec_from_file_location('continuation_request',Path(__file__).resolve().parents[1]/'sampling.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=ContinuationConfig(method='train_time_rtc',prefix_max_steps=2)
    target=np.arange(56,dtype=np.float32).reshape(4,14)/20
    request={'mode':'train_time_rtc','prefix_steps':2,'action_condition':target,'condition_weights':[1,1,0,0]}
    raw,w=module.parse_request(request,4,14,cfg)
    np.testing.assert_array_equal(raw,target)
    with pytest.raises(ValueError):module.parse_request({**request,'condition_weights':[0,1,1,0]},4,14,cfg)
    with pytest.raises(ValueError):module.parse_request({**request,'prefix_steps':3},4,14,cfg)
    assert module.parse_request({'mode':'train_time_rtc','prefix_steps':0},4,14,cfg)==(None,None)
    from openpi.policies.policy import Policy
    from openpi import transforms
    from openpi.shared.normalize import NormStats
    class Echo(nnx.Module):
        action_dim=32
        action_horizon=4
        def sample_actions(self,rng,obs,*,action_condition,condition_weights,rtc_beta,num_steps=10):
            # Checking the trained model boundary after physical-to-model transforms.
            return action_condition
    state=np.arange(14,dtype=np.float32)/30
    stats={'state':NormStats(mean=np.zeros(14),std=np.ones(14)),
           'actions':NormStats(mean=np.ones(14)*.3,std=np.ones(14)*2)}
    mask=transforms.make_bool_mask(6,-1,6,-1)
    policy=Policy(Echo(),transforms=[transforms.DeltaActions(mask),transforms.Normalize(stats),transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(stats),transforms.AbsoluteActions(mask)])
    observation={'state':state,'image':{'base_0_rgb':np.zeros((4,4,3),dtype=np.uint8)},'image_mask':{'base_0_rgb':True}}
    result=policy.infer(observation,action_condition=raw,condition_weights=w,num_steps=10)
    np.testing.assert_allclose(result['actions'][...,:14],target,atol=1e-6)
