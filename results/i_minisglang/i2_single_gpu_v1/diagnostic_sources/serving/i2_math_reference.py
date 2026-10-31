"""Independent PyTorch reference with explicit native BF16 rounding boundaries.

No Mini-SGLang, FlashInfer, paged-cache, or Graph kernels execute here. HF supplies
checkpoint tensors only. Attention uses PyTorch FP32 math SDPA and contiguous KV.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def rms(value, weight, eps):
    value=value.float()
    return (value*torch.rsqrt(value.square().mean(-1,keepdim=True)+eps)*weight.float()).to(weight.dtype)


def fused_add_norm(value, residual, weight, eps):
    total=value.float()+residual.float()
    return rms(total,weight,eps),total.to(value.dtype)


class BF16MathReference:
    def __init__(self,hf):
        self.hf=hf
        config=hf.config
        self.head_dim=config.head_dim
        self.qheads=config.num_attention_heads
        self.kheads=config.num_key_value_heads
        self.eps=config.rms_norm_eps
        self.qkv=[torch.cat([layer.self_attn.q_proj.weight,layer.self_attn.k_proj.weight,layer.self_attn.v_proj.weight])
                  for layer in hf.model.layers]
        self.gate_up=[torch.cat([layer.mlp.gate_proj.weight,layer.mlp.up_proj.weight]) for layer in hf.model.layers]
        inv=1/(config.rope_theta**(torch.arange(0,self.head_dim,2,device='cuda',dtype=torch.float32)/self.head_dim))
        positions=torch.arange(1024,device='cuda',dtype=torch.float32)
        angle=torch.outer(positions,inv)
        self.cos,self.sin=angle.cos(),angle.sin()
        self.reset()

    def reset(self):
        self.keys=[None]*len(self.hf.model.layers)
        self.values=[None]*len(self.hf.model.layers)
        self.length=0

    def rotate(self,value,positions):
        first,second=value.float().chunk(2,dim=-1)
        cosine=self.cos[positions,None,:]
        sine=self.sin[positions,None,:]
        return torch.cat([first*cosine-second*sine,second*cosine+first*sine],dim=-1).to(value.dtype)

    def __call__(self,ids):
        x=F.embedding(ids,self.hf.model.embed_tokens.weight)
        n=len(ids)
        positions=torch.arange(self.length,self.length+n,device=ids.device)
        residual=None
        for i,layer in enumerate(self.hf.model.layers):
            if residual is None:
                hidden=rms(x,layer.input_layernorm.weight,self.eps)
                residual=x
            else:
                hidden,residual=fused_add_norm(x,residual,layer.input_layernorm.weight,self.eps)
            projected=F.linear(hidden,self.qkv[i])
            q,k,v=projected.split([self.qheads*self.head_dim,self.kheads*self.head_dim,self.kheads*self.head_dim],dim=-1)
            q=rms(q.view(n,self.qheads,self.head_dim),layer.self_attn.q_norm.weight,self.eps)
            k=rms(k.view(n,self.kheads,self.head_dim),layer.self_attn.k_norm.weight,self.eps)
            q=self.rotate(q,positions).transpose(0,1).unsqueeze(0)
            k=self.rotate(k,positions).transpose(0,1).unsqueeze(0)
            v=v.view(n,self.kheads,self.head_dim).transpose(0,1).unsqueeze(0)
            if self.keys[i] is not None:
                k=torch.cat([self.keys[i],k],dim=2);v=torch.cat([self.values[i],v],dim=2)
            self.keys[i],self.values[i]=k,v
            with sdpa_kernel(SDPBackend.MATH):
                attention=F.scaled_dot_product_attention(q.float(),k.float(),v.float(),
                    is_causal=self.length==0,dropout_p=0,enable_gqa=True).to(x.dtype)
            attention=attention.squeeze(0).transpose(0,1).reshape(n,-1)
            x=F.linear(attention,layer.self_attn.o_proj.weight)
            hidden,residual=fused_add_norm(x,residual,layer.post_attention_layernorm.weight,self.eps)
            gate,up=F.linear(hidden,self.gate_up[i]).chunk(2,dim=-1)
            activated=(F.silu(gate.float())*up.float()).to(gate.dtype)
            x=F.linear(activated,layer.mlp.down_proj.weight)
        hidden,_=fused_add_norm(x,residual,self.hf.model.norm.weight,self.eps)
        self.length+=n
        return F.linear(hidden[-1],self.hf.lm_head.weight)
