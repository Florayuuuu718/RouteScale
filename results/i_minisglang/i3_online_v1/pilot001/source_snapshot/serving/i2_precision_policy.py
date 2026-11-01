"""Explicit TP1 policies installed before CUDA Graph construction.

Both policies use no-split prefill and linear tile512. Mixed precision retains
FP16 FlashInfer/linear/KV execution and FP32 residual plus layer/final RMSNorm.
No observer or sampler implements the policy, and no default engine is changed.
"""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import torch

MODES = ('bf16_control', 'mixed')
POLICY = 'fi_nosplit_linear512_fp16_main_fp32_residual_norm'
POLICIES = {'mixed': POLICY, 'bf16_control': 'fi_nosplit_linear512_bf16_control'}


def install_residual_norms(model):
    """Install before graph capture; tensor operations are capture compatible."""
    if getattr(model, '_i2_residual_policy', False):
        raise ValueError('Residual precision policy already installed')
    body = model.model
    norms = [op for layer in body.layers.op_list for op in
             (layer.input_layernorm, layer.post_attention_layernorm)] + [body.norm]
    if any(op.weight.dtype != torch.float16 for op in norms):
        raise ValueError('Mixed residual policy requires FP16 model weights')
    originals = []
    for op in norms:
        originals.append((op, op.forward))
        def forward(x, residual=None, norm=op):
            total = x.float() if residual is None else x.float() + residual.float()
            y = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + norm.eps)
            return (y * norm.weight.float()).to(x.dtype), total
        op.forward = forward
    model._i2_residual_policy = True

    def restore():
        for op, original in originals:
            op.forward = original
        originals.clear()
        model._i2_residual_policy = False
    return restore, len(norms)


def engine_initializer(native_init, engine_module, mode):
    if mode not in MODES:
        raise ValueError(mode)

    def init(engine, config):
        if config.tp_info.size != 1:
            raise ValueError('I2 precision qualification is TP1 only')
        dtype = torch.float16 if mode == 'mixed' else torch.bfloat16
        config = replace(config, dtype=dtype)
        restorers, graph_entries = [], []
        native_graph = engine_module.GraphRunner

        def graph_before_capture(*args, **kwargs):
            model = kwargs['model']
            count = 0
            if mode == 'mixed':
                restore, count = install_residual_norms(model)
                restorers.append(restore)
            graph_entries.append(dict(norms_installed=count, before_graph_constructor=True))
            return native_graph(*args, **kwargs)

        try:
            with patch.object(engine_module, 'GraphRunner', graph_before_capture):
                native_init(engine, config)
            if len(graph_entries) != 1:
                raise ValueError('Expected one graph constructor')
            from serving.i2_numerical_policy import install_prefill_no_split
            from serving.i2_linear_tile_policy import install_prefill_linear_tiles
            restorers += [install_prefill_no_split(engine), install_prefill_linear_tiles(engine)]
            engine._i2_precision_restorers = restorers
            engine._i2_precision_observed = dict(mode=mode, policy=POLICIES[mode],
                dtype=str(engine.dtype), weight_dtype=str(engine.model.model.embed_tokens.weight.dtype),
                kv_dtype=str(engine.kv_cache.dtype), residual_dtype='torch.float32' if mode == 'mixed' else 'torch.bfloat16',
                graph_install=graph_entries[0], captured_buckets=engine.graph_runner.graph_bs_list,
                captured_graph_keys=sorted(engine.graph_runner.graph_map),
                actual_backend=type(engine.attn_backend).__name__, prefill_no_split=True, linear_tile=512)
        except BaseException:
            for restore in reversed(restorers):
                restore()
            raise
    return init


def restore_engine_policy(engine):
    restorers = getattr(engine, '_i2_precision_restorers', [])
    for restore in reversed(restorers):
        restore()
    restorers.clear()


@contextmanager
def engine_construction(mode):
    from minisgl.engine import engine as engine_module
    native = engine_module.Engine.__init__
    with patch.object(engine_module.Engine, '__init__', engine_initializer(native, engine_module, mode)):
        yield
