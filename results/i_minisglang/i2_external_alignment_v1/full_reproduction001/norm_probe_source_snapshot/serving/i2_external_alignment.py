"""Read-only HF tracing and explicit precision interventions for I2 diagnostics.

The FP32 path retains Mini's model, merged weights, scheduler, page tables and
cache allocation. It replaces low-precision operators with PyTorch math; it is
not the FlashInfer serving path and must never be used as BF16 timing evidence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class HFTrace:
    """Canonical last-128 prefill rows, compatible with the native LayerTrace."""

    def __init__(self, model, rows=128):
        from transformers.models.qwen3 import modeling_qwen3 as qwen

        self.ops, self.order, self.handles = {}, [], []
        self.enabled = True
        self.rows = rows
        self.qwen = qwen
        self.original_rope = qwen.apply_rotary_pos_emb
        self.layer = None

        def save(name, tensor):
            if self.enabled:
                value = tensor.squeeze(0) if tensor.ndim >= 3 and tensor.shape[0] == 1 else tensor
                self.ops[name] = value[-rows:].detach().clone()
                self.order.append(name)

        self.save = save

        def post(module, name, transform=lambda v: v):
            self.handles.append(module.register_forward_hook(lambda m, a, out: save(name, transform(out))))

        def pre(module, name):
            self.handles.append(module.register_forward_pre_hook(lambda m, a: save(name, a[0])))

        post(model.model.embed_tokens, 'embedding')
        for i, layer in enumerate(model.model.layers):
            def active(m, a, index=i):
                self.layer = index
            self.handles.append(layer.register_forward_pre_hook(active))
            for attr in ('input_layernorm', 'post_attention_layernorm'):
                norm = getattr(layer, attr)
                pre(norm, f'{i}.{attr}.1')
                post(norm, f'{i}.{attr}.0')
            a = layer.self_attn
            parts = {}
            for attr in ('q_proj', 'k_proj', 'v_proj'):
                def projection(m, args, out, key=attr, index=i, values=parts):
                    if not self.enabled:
                        return
                    values[key] = out
                    if key == 'v_proj':
                        save(f'{index}.qkv_proj', torch.cat([values[n] for n in ('q_proj', 'k_proj', 'v_proj')], -1))
                        save(f'{index}.v', out.view(*out.shape[:-1], -1, a.head_dim))
                        values.clear()
                self.handles.append(getattr(a, attr).register_forward_hook(projection))
            post(a.q_norm, f'{i}.q_norm')
            post(a.k_norm, f'{i}.k_norm')
            self.handles.append(a.o_proj.register_forward_pre_hook(
                lambda m, args, index=i: save(f'{index}.attention', args[0].view(*args[0].shape[:-1], -1, model.config.head_dim))))
            post(a.o_proj, f'{i}.o_proj')
            gate = {}
            def gate_hook(m, args, out, value=gate):
                if self.enabled:
                    value['gate'] = out
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
            def up_hook(m, args, out, index=i, value=gate):
                if self.enabled:
                    save(f'{index}.gate_up_proj', torch.cat([value.pop('gate'), out], -1))
            self.handles.append(layer.mlp.up_proj.register_forward_hook(up_hook))
            pre(layer.mlp.down_proj, f'{i}.silu_mul')
            post(layer.mlp.down_proj, f'{i}.down_proj')
        pre(model.model.norm, 'final_norm.1')
        post(model.model.norm, 'final_norm.0')

        def rope(*args, **kwargs):
            q, k = self.original_rope(*args, **kwargs)
            save(f'{self.layer}.q_rope', q.transpose(1, 2))
            save(f'{self.layer}.k_rope', k.transpose(1, 2))
            return q, k
        qwen.apply_rotary_pos_emb = rope

    def cpu(self):
        return {name: value.cpu().contiguous() for name, value in self.ops.items()}

    def close(self):
        self.qwen.apply_rotary_pos_emb = self.original_rope
        for handle in self.handles:
            handle.remove()


def math_norm(x, weight, eps):
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def paged_math_attention(q, k, v, layer_id, batch, cache):
    """FP32 attention over the actual physical pages, with absolute causal mask."""
    heads, dim = cache._storage_shape[-2:]
    keys = cache.k_cache(layer_id).view(-1, heads, dim)
    values = cache.v_cache(layer_id).view(-1, heads, dim)
    keys.index_copy_(0, batch.out_loc.long(), k.view(-1, heads, dim))
    values.index_copy_(0, batch.out_loc.long(), v.view(-1, heads, dim))
    meta = batch.attn_metadata
    qi, ki = meta.cu_seqlens_q_cpu.tolist(), meta.cu_seqlens_k_cpu.tolist()
    outputs = []
    for n, req in enumerate(batch.reqs):
        query = q[qi[n]:qi[n+1]].transpose(0, 1)
        indices = meta.indices[ki[n]:ki[n+1]].long()
        key = keys.index_select(0, indices).transpose(0, 1)
        value = values.index_select(0, indices).transpose(0, 1)
        group = query.shape[0] // heads
        key, value = key.repeat_interleave(group, 0), value.repeat_interleave(group, 0)
        score = (query @ key.transpose(-2, -1)) * dim**-0.5
        positions = batch.positions[qi[n]:qi[n+1]]
        mask = torch.arange(len(indices), device=q.device)[None, :] <= positions[:, None]
        score = score.masked_fill(~mask[None], torch.finfo(score.dtype).min)
        outputs.append((score.softmax(-1, dtype=torch.float32) @ value).transpose(0, 1))
    return torch.cat(outputs, 0)


def install_precision_intervention(engine, mode):
    """Explicit, reversible diagnostics. Graph and overlap must be disabled."""
    if mode not in {'residual_fp32', 'fp32_math'}:
        raise ValueError(mode)
    from minisgl.layers import BaseOP
    if engine.graph_runner.graph_bs_list:
        raise ValueError('FP32 diagnostic cannot run captured graphs')
    restorers = []

    def bind(obj, attr, value):
        restorers.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    model = engine.model.model
    if mode == 'fp32_math':
        visited = set()
        def promote(obj):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            for name, value in list(vars(obj).items()):
                if name in {'weight', 'bias'} and isinstance(value, torch.Tensor):
                    bind(obj, name, value.float())
                elif isinstance(value, BaseOP):
                    promote(value)
                elif name == 'op_list':
                    for layer in value:
                        promote(layer)
        promote(engine.model)
        bind(model.embed_tokens, 'forward', lambda ids: F.embedding(ids, model.embed_tokens.weight))
        cache = engine.kv_cache
        buffer = cache._kv_buffer.float()
        bind(cache, '_kv_buffer', buffer)
        bind(cache, '_k_buffer', buffer[0])
        bind(cache, '_v_buffer', buffer[1])
        bind(engine.attn_backend, 'forward', lambda q, k, v, layer_id, batch:
             paged_math_attention(q, k, v, layer_id, batch, cache))

    for layer in [*model.layers.op_list, model]:
        norms = [layer.norm] if layer is model else [layer.input_layernorm, layer.post_attention_layernorm]
        for norm in norms:
            def forward(x, residual=None, op=norm):
                total = x.float() if residual is None else x.float() + residual.float()
                normalized = math_norm(total, op.weight, op.eps).to(x.dtype)
                return normalized, total
            bind(norm, 'forward', forward)
        if mode != 'fp32_math' or layer is model:
            continue
        a = layer.self_attn
        for norm in (a.q_norm, a.k_norm):
            def inplace(x, op=norm):
                x.copy_(math_norm(x, op.weight, op.eps))
            bind(norm, 'forward_inplace', inplace)
        bind(layer.mlp, 'act_fn', lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
    if mode == 'fp32_math':
        rotary = model.layers.op_list[0].self_attn.attn.rotary
        def rotate(positions, q, k):
            cos, sin = rotary._cos_sin_cache[positions].float().chunk(2, -1)
            def apply(x):
                shaped = x.view(len(positions), -1, rotary.head_size)
                first, second = shaped.chunk(2, -1)
                return torch.cat([first*cos[:, None]-second*sin[:, None],
                                  second*cos[:, None]+first*sin[:, None]], -1).reshape_as(x)
            return apply(q), apply(k)
        bind(rotary, 'forward', rotate)

    def restore():
        for obj, attr, value in reversed(restorers):
            setattr(obj, attr, value)
        restorers.clear()
    return restore
