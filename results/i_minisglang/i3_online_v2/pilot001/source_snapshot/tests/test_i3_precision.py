"""Numerical dtype boundaries, tied head, reversibility and pre-capture ordering."""
from types import ModuleType,SimpleNamespace as NS
import sys

import pytest
import torch
import torch.nn.functional as F

from serving import i3_precision_v2 as policy


@pytest.fixture
def model(monkeypatch):
    context=NS(batch=NS(is_prefill=False,size=1))
    core=ModuleType('minisgl.core');core.get_global_ctx=lambda:context
    monkeypatch.setitem(sys.modules,'minisgl.core',core)
    def op():return NS(weight=torch.tensor([1.,1.],dtype=torch.float16),bias=None,eps=1e-6,
        forward=lambda x,residual=None:(x,residual),forward_inplace=lambda x:None)
    rotary=NS(head_size=2,_cos_sin_cache=torch.tensor([[1.,0.],[1.,0.]]),forward=lambda pos,q,k:(q,k))
    layers=[]
    for _ in range(4):
        layers.append(NS(input_layernorm=op(),post_attention_layernorm=op(),
            self_attn=NS(qkv_proj=op(),o_proj=op(),q_norm=op(),k_norm=op(),attn=NS(rotary=rotary)),
            mlp=NS(gate_up_proj=op(),down_proj=op(),act_fn=lambda x:x)))
    head=NS(weight=torch.zeros(5,2,dtype=torch.float16),bias=None,
        tied_embedding=NS(weight=torch.arange(10,dtype=torch.float16).reshape(5,2)),forward=lambda x:x)
    m=NS(model=NS(layers=NS(op_list=layers),norm=op()),lm_head=head)
    def attention(q,k,v,layer,batch):
        assert q.dtype==k.dtype==v.dtype==torch.float16
        return q
    return m,NS(forward=attention),context


def test_upper_weights_promote_and_restore_exact_original_objects(model):
    m,b,_=model;layer=m.model.layers.op_list[-1]
    weight=layer.self_attn.qkv_proj.weight;forward=layer.input_layernorm.forward
    restore,observed=policy.install(m,b,1)
    assert layer.self_attn.qkv_proj.weight.dtype==torch.float32
    assert m.model.layers.op_list[0].self_attn.qkv_proj.weight.dtype==torch.float16
    assert observed['upper_layers']==1 and observed['head_weight_dtype']=='torch.float32'
    restore();restore()
    assert layer.self_attn.qkv_proj.weight is weight and layer.input_layernorm.forward is forward


def test_tied_head_uses_promoted_embedding_and_attention_retains_FP16_boundary(model):
    m,b,_=model;original=m.lm_head.tied_embedding.weight
    restore,_=policy.install(m,b,1)
    x=torch.tensor([[.123456,.765432]],dtype=torch.float32)
    assert torch.equal(m.lm_head.forward(x),F.linear(x,original.float()))
    assert m.lm_head.tied_embedding.weight is original and original.dtype==torch.float16
    q=torch.ones(1,1,2,dtype=torch.float32)
    assert b.forward(q,q,q,3,None).dtype==torch.float32
    restore()


def test_upper_norm_preserves_FP32_residual_instead_of_rounding_to_half(model):
    m,b,_=model;restore,_=policy.install(m,b,1)
    x=torch.ones(1,2,dtype=torch.float16);residual=torch.tensor([[1e-5,2e-5]])
    value,total=m.model.layers.op_list[-1].input_layernorm.forward(x,residual)
    assert value.dtype==total.dtype==torch.float32
    assert torch.equal(total,x.float()+residual)
    assert not torch.equal(total,total.half().float())
    restore()


@pytest.mark.parametrize('count',[-1,5])
def test_invalid_layer_count_is_rejected(model,count):
    with pytest.raises(ValueError):policy.install(model[0],model[1],count)


@pytest.mark.parametrize('fail',[False,True])
def test_policy_is_installed_before_graph_and_rolls_back_on_failure(monkeypatch,fail):
    installed=[];restored=[]
    def install(model,backend,n):
        installed.append(n)
        return lambda:restored.append(True),dict(upper_layers=n,installed_before_graph=True)
    monkeypatch.setattr(policy,'install',install)
    def graph(**kw):
        assert installed==[2]
        if fail:raise RuntimeError('capture failed')
        return NS(graph_map={1:None,2:None,4:None})
    module=NS(GraphRunner=graph)
    def native(engine,config):engine.graph_runner=module.GraphRunner(model=object(),attn_backend=object())
    engine=NS();initialize=policy.engine_initializer(native,module,2)
    if fail:
        with pytest.raises(RuntimeError,match='capture failed'):initialize(engine,None)
        assert restored==[True]
    else:
        initialize(engine,None)
        assert engine._i3_precision_observed['captured_graph_keys']==[1,2,4]
        engine._i3_precision_restore();assert restored==[True]
    assert module.GraphRunner is graph
