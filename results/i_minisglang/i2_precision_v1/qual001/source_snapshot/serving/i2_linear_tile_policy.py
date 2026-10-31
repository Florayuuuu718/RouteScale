"""Common BF16 linear row geometry for explicit I2 qualification experiments."""
import torch
import torch.nn.functional as F


def tiled_linear(x, linear, tile=512):
    if x.ndim!=2 or tile<1 or len(x)==0:
        raise ValueError('Positive row tile and nonempty 2D activations required')
    pieces=[]
    for part in x.split(tile):
        padded=F.pad(part,(0,0,0,tile-len(part))) if len(part)!=tile else part
        pieces.append(linear(padded)[:len(part)])
    return pieces[0] if len(pieces)==1 else torch.cat(pieces)


def install_prefill_linear_tiles(engine,tile=512):
    from minisgl.core import get_global_ctx
    from minisgl.distributed import get_tp_info
    if get_tp_info().size!=1:raise ValueError('Linear geometry policy is TP1 only')
    restorers=[]
    for layer in engine.model.model.layers.op_list:
        for op in (layer.self_attn.qkv_proj,layer.self_attn.o_proj,layer.mlp.gate_up_proj,layer.mlp.down_proj):
            original=op.forward
            def forward(x,fn=original):
                return tiled_linear(x,fn,tile) if get_global_ctx().batch.is_prefill else fn(x)
            op.forward=forward;restorers.append((op,original))
    def restore():
        for op,original in restorers:op.forward=original
        restorers.clear()
    return restore
