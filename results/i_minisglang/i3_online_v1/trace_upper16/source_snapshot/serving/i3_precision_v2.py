"""Explicit, capture-compatible FP32 upper layers with FP16 FlashInfer/KV.

This is a new precision scope. It does not modify or retroactively qualify I2.
All intervention cost is part of the service being measured.
"""
import torch
import torch.nn.functional as F


def install(model, backend, upper_layers):
    from minisgl.core import get_global_ctx
    from serving.i2_external_alignment import math_norm
    body=model.model;layers=body.layers.op_list
    if not 0<=upper_layers<=len(layers):raise ValueError('Invalid upper layer count')
    saved=[]
    def bind(obj,name,value):
        saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def promote(op):
        bind(op,'weight',op.weight.float())
        if getattr(op,'bias',None) is not None:bind(op,'bias',op.bias.float())
    def norm32(op):
        promote(op)
        def forward(x,residual=None):
            total=x.float() if residual is None else x.float()+residual.float()
            return math_norm(total,op.weight,op.eps),total
        bind(op,'forward',forward)
    rotary_ops={}
    for layer in layers[len(layers)-upper_layers:] if upper_layers else []:
        for norm in (layer.input_layernorm,layer.post_attention_layernorm):norm32(norm)
        a=layer.self_attn
        for linear in (a.qkv_proj,a.o_proj,layer.mlp.gate_up_proj,layer.mlp.down_proj):promote(linear)
        for norm in (a.q_norm,a.k_norm):
            promote(norm)
            def inplace(x,op=norm):x.copy_(math_norm(x,op.weight,op.eps))
            bind(norm,'forward_inplace',inplace)
        bind(layer.mlp,'act_fn',lambda x:F.silu(x.chunk(2,-1)[0])*x.chunk(2,-1)[1])
        rotary_ops[id(a.attn.rotary)]=a.attn.rotary
    for rotary in rotary_ops.values():
        native=rotary.forward
        def rotate(positions,q,k,op=rotary,original=native):
            if q.dtype!=torch.float32:return original(positions,q,k)
            cos,sin=op._cos_sin_cache[positions].float().chunk(2,-1)
            def apply(x):
                shaped=x.view(len(positions),-1,op.head_size);first,second=shaped.chunk(2,-1)
                return torch.cat([first*cos[:,None]-second*sin[:,None],second*cos[:,None]+first*sin[:,None]],-1).reshape_as(x)
            return apply(q),apply(k)
        bind(rotary,'forward',rotate)
    native_attention=backend.forward
    def attention(q,k,v,layer_id,batch):
        dtype=q.dtype
        if dtype==torch.float32:
            return native_attention(q.half(),k.half(),v.half(),layer_id,batch).float()
        return native_attention(q,k,v,layer_id,batch)
    bind(backend,'forward',attention)
    norm32(body.norm)
    head=model.lm_head;module=head.tied_embedding or head;weight=module.weight.float()
    bias=head.bias.float() if head.bias is not None else None
    def project(x):
        batch=get_global_ctx().batch
        if batch.is_prefill:x=x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return F.linear(x.float(),weight,bias)
    bind(head,'forward',project)
    def restore():
        for op,name,value in reversed(saved):setattr(op,name,value)
        saved.clear()
    observed=dict(policy=f'fp16_lower_fp32_upper{upper_layers}_head_fp16_FI_KV',upper_layers=upper_layers,
        installed_before_graph=True,upper_linear_weight_dtype=str(layers[-1].self_attn.qkv_proj.weight.dtype) if upper_layers else None,
        head_weight_dtype=str(weight.dtype),attention_dtype='torch.float16',kv_dtype='torch.float16',
        TF32=False,FP32_rope_and_qk_norm=bool(upper_layers))
    return restore,observed


def engine_initializer(native_init,engine_module,upper_layers):
    from unittest.mock import patch
    def initialize(engine,config):
        native_graph=engine_module.GraphRunner
        restores=[];policies=[]
        def graph(*args,**kwargs):
            restore,observed=install(kwargs['model'],kwargs['attn_backend'],upper_layers)
            restores.append(restore);policies.append(observed)
            return native_graph(*args,**kwargs)
        try:
            with patch.object(engine_module,'GraphRunner',graph):native_init(engine,config)
            if len(policies)!=1:raise ValueError('Expected one pre-capture precision installation')
            engine._i3_precision_restore=restores[0]
            engine._i3_precision_observed=dict(policies[0],captured_graph_keys=sorted(engine.graph_runner.graph_map))
        except BaseException:
            for restore in reversed(restores):restore()
            raise
    return initialize
