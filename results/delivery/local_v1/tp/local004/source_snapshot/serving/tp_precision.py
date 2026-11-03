"""New TP policy: preserve FP32 upper layers AND gather the sharded vocabulary.

The TP1-only I2 initializer and tile guard are never bypassed or edited.
This separate initializer still needs real TP2/4 qualification on the rented host.
"""
from dataclasses import replace
from unittest.mock import patch

import torch


def gather_head(local,head):
    if head.tp_size==1:return local
    gathered=head._comm.all_gather(local)
    batch,width=local.shape
    return gathered.view(head.tp_size,batch,width).permute(1,0,2).contiguous().view(batch,-1)[:,:head.num_embeddings]


def initializer(native,engine_module):
    from serving.i2_precision_policy import install_residual_norms
    from serving.i2_numerical_policy import install_prefill_no_split
    from serving.i2_linear_tile_policy import tiled_linear
    from serving.i3_precision_v2 import install
    def initialize(engine,config):
        if config.tp_info.size not in (1,2,4):raise ValueError('TP1/2/4 only')
        config=replace(config,dtype=torch.float16,use_pynccl=False)
        restorers=[];observed=[];native_graph=engine_module.GraphRunner
        def graph(*args,**kwargs):
            model=kwargs['model'];restore,_=install_residual_norms(model);restorers.append(restore)
            restore,policy=install(model,kwargs['attn_backend'],16);restorers.append(restore);observed.append(policy)
            head=model.lm_head;project=head.forward
            head.forward=lambda x:gather_head(project(x),head)
            restorers.append(lambda:setattr(head,'forward',project))
            return native_graph(*args,**kwargs)
        try:
            with patch.object(engine_module,'GraphRunner',graph):native(engine,config)
            restorers.append(install_prefill_no_split(engine))
            from minisgl.core import get_global_ctx
            for layer in engine.model.model.layers.op_list:
                for op in (layer.self_attn.qkv_proj,layer.self_attn.o_proj,layer.mlp.gate_up_proj,layer.mlp.down_proj):
                    original=op.forward
                    def forward(x,fn=original):return tiled_linear(x,fn,512) if get_global_ctx().batch.is_prefill else fn(x)
                    op.forward=forward
                    restorers.append(lambda op=op,fn=original:setattr(op,'forward',fn))
            engine._tp_restorers=restorers
            engine._tp_observed=dict(observed[0],rank=config.tp_info.rank,world=config.tp_info.size,
                head_all_gather=True,collective='torch_nccl',captured_graph_keys=sorted(engine.graph_runner.graph_map))
        except BaseException:
            for restore in reversed(restorers):restore()
            raise
    return initialize
