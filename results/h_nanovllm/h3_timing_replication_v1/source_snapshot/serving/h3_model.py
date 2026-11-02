"""Architecture-preserving packed Dense GPT and an offline GPT-2 tokenizer."""

from __future__ import annotations

import torch
import math
from torch import nn
from torch.nn import functional as F


class PackedRouteScaleGPT(nn.Module):
    def __init__(self,native,attention_factory,get_context):
        super().__init__()
        if native.config.moe_num_experts or native.training:
            raise ValueError("Only eval Dense GPT is qualified")
        self.gpt,self.context=native,get_context
        config=native.config
        self.attentions=nn.ModuleList([attention_factory(config.n_head,config.n_embd//config.n_head,
            (config.n_embd//config.n_head)**-.5,config.n_head) for _ in range(config.n_layer)])

    def forward(self,token_ids,positions):
        if token_ids.ndim!=1 or positions.shape!=token_ids.shape:
            raise ValueError("Packed token IDs/positions must be matching one-dimensional arrays")
        gpt=self.gpt
        hidden=gpt.transformer.drop(gpt.transformer.wte(token_ids)+gpt.transformer.wpe(positions))
        heads,dim=gpt.config.n_head,gpt.config.n_embd//gpt.config.n_head
        for block,attention in zip(gpt.transformer.h,self.attentions):
            q,k,v=block.attn.c_attn(block.ln_1(hidden)).split(gpt.config.n_embd,dim=-1)
            output=attention(q.reshape(-1,heads,dim),k.reshape(-1,heads,dim),v.reshape(-1,heads,dim))
            hidden=hidden+block.attn.resid_dropout(block.attn.c_proj(output.reshape(hidden.shape)))
            hidden=hidden+block.mlp(block.ln_2(hidden))
        return gpt.transformer.ln_f(hidden)

    def compute_logits(self,hidden):
        context=self.context()
        if context.is_prefill:
            hidden=hidden[context.cu_seqlens_q[1:]-1].contiguous()
        return self.gpt.lm_head(hidden)


class SegmentedSDPA(nn.Module):
    """Prefill-only FP32 mapping reference, not the paged serving backend."""
    def __init__(self,heads,dim,scale,kv_heads,*,get_context):
        super().__init__()
        self.context=get_context

    def forward(self,q,k,v):
        context=self.context()
        if not context.is_prefill:
            raise ValueError("Segmented reference is Prefill-only")
        bounds=context.cu_seqlens_q.tolist()
        rows=[]
        for start,end in zip(bounds,bounds[1:]):
            shaped=[x[start:end].transpose(0,1).unsqueeze(0) for x in (q,k,v)]
            output=F.scaled_dot_product_attention(*shaped,is_causal=True,dropout_p=0)
            rows.append(output.squeeze(0).transpose(0,1))
        return torch.cat(rows,dim=0)


class OfflineGPT2Tokenizer:
    def __init__(self,data):
        from tokenizers import AddedToken,Regex,Tokenizer,decoders,models,pre_tokenizers
        self.eos_token_id=data["special_tokens"]["<|endoftext|>"]
        self.vocab_size=len(data["vocab"])+len(data["special_tokens"])
        self.special_tokens=list(data["special_tokens"])
        self.backend=Tokenizer(models.BPE(data["vocab"],[tuple(p) for p in data["merges"]],fuse_unk=False,ignore_merges=True))
        self.backend.pre_tokenizer=pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(data["pattern"]),behavior="isolated",invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
        self.backend.decoder=decoders.ByteLevel()
        self.backend.add_special_tokens([AddedToken(t,normalized=False,special=True) for t in self.special_tokens])
        if self.backend.token_to_id("<|endoftext|>")!=50256 or self.backend.get_vocab_size()!=50257:
            raise ValueError("Prepared GPT-2 tokenizer IDs changed")

    def encode(self,text):
        if any(token in text for token in self.special_tokens):
            raise ValueError("Special-token text is disallowed, matching source tiktoken default")
        return self.backend.encode(text,add_special_tokens=False).ids

    def decode(self,token_ids):
        if any(not isinstance(i,int) or not 0<=i<self.vocab_size for i in token_ids):
            raise ValueError("Token ID outside the original GPT-2 vocabulary")
        return self.backend.decode(token_ids,skip_special_tokens=False)


def validate_request(tokens,params,*,window=512,vocab_size=50257):
    if (not tokens or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<vocab_size for i in tokens)
            or not isinstance(params.max_tokens,int) or isinstance(params.max_tokens,bool)
            or params.max_tokens<1 or len(tokens)+params.max_tokens-1>window
            or not math.isfinite(params.temperature) or params.temperature<=1e-10):
        raise ValueError("Invalid token/output/temperature budget or absolute-position window overflow")
