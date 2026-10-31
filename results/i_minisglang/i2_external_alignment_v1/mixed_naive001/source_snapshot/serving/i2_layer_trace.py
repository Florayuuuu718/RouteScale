"""Observe native prefill operators using GPU clones, with no returned-value edits."""
from __future__ import annotations

import torch
from minisgl.core import get_global_ctx


class LayerTrace:
    def __init__(self, scheduler, prefix=384, suffix=128):
        self.scheduler = scheduler
        self.prefix, self.suffix = prefix, suffix
        self.phase = None
        self.ops, self.cold_kv, self.warm_kv, self.prefix_before = {}, {}, {}, {}
        self.metadata, self.order, self.restorers = {}, [], []
        self._install()

    def active(self):
        return self.phase in {'cold', 'warm'} and get_global_ctx().batch.is_prefill

    def save(self, name, tensor):
        if self.active() and self.phase == 'warm':
            if name in self.ops:
                raise AssertionError('More than one observed warm prefill')
            self.ops[name] = tensor[-self.suffix:].detach().clone()
            self.order.append(name)

    def bind(self, obj, name, replacement):
        original = getattr(obj, name)
        setattr(obj, name, replacement)
        self.restorers.append((obj, name, original))

    def wrap(self, obj, attr, name, inplace=False):
        original = getattr(obj, attr)
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if inplace:
                self.save(name, args[0])
            elif isinstance(result, tuple):
                for index, tensor in enumerate(result):
                    self.save(f'{name}.{index}', tensor)
            else:
                self.save(name, result)
            return result
        self.bind(obj, attr, observed)

    def _install(self):
        model = self.scheduler.engine.model.model
        self.wrap(model.embed_tokens, 'forward', 'embedding')
        for index, layer in enumerate(model.layers.op_list):
            for name in ('input_layernorm', 'post_attention_layernorm'):
                self.wrap(getattr(layer, name), 'forward', f'{index}.{name}')
            attention = layer.self_attn
            for name in ('qkv_proj', 'o_proj'):
                self.wrap(getattr(attention, name), 'forward', f'{index}.{name}')
            for name in ('q_norm', 'k_norm'):
                self.wrap(getattr(attention, name), 'forward_inplace', f'{index}.{name}', inplace=True)
            self.wrap(layer.mlp.gate_up_proj, 'forward', f'{index}.gate_up_proj')
            self.wrap(layer.mlp, 'act_fn', f'{index}.silu_mul')
            self.wrap(layer.mlp.down_proj, 'forward', f'{index}.down_proj')
        self.wrap(model.norm, 'forward', 'final_norm')
        backend = self.scheduler.engine.attn_backend
        original = backend.forward
        def attention(q, k, v, layer_id, batch):
            observe = self.active()
            if observe:
                assert len(batch.reqs) == 1
                req = batch.reqs[0]
                meta = batch.attn_metadata
                cache = backend.kvcache
                heads, dim = backend.kv_head_local, backend.config.head_dim
                keys = cache.k_cache(layer_id).view(-1, heads, dim)
                values = cache.v_cache(layer_id).view(-1, heads, dim)
                if layer_id == 0:
                    self.metadata[self.phase] = dict(cached=req.cached_len, extend=req.extend_len,
                        q_shape=list(q.shape), q_indptr=meta.cu_seqlens_q_cpu.tolist(),
                        kv_indptr=meta.cu_seqlens_k_cpu.tolist(), seq_lens=meta.seq_lens_cpu.tolist(),
                        indices=meta.indices.detach().clone(), positions=batch.positions.detach().clone(),
                        out_loc=batch.out_loc.detach().clone(), input_ids=batch.input_ids.detach().clone())
                if self.phase == 'warm':
                    self.save(f'{layer_id}.q_rope', q)
                    self.save(f'{layer_id}.k_rope', k.view(-1, heads, dim))
                    self.save(f'{layer_id}.v', v.view(-1, heads, dim))
                    # Preserve all Q rows for an independent fixed-input replay.
                    self.warm_kv[f'{layer_id}.q_all'] = q.detach().clone()
                    if req.cached_len:
                        self.prefix_before[f'{layer_id}.k'] = keys.index_select(0, meta.indices[:req.cached_len].long())
                        self.prefix_before[f'{layer_id}.v'] = values.index_select(0, meta.indices[:req.cached_len].long())
            result = original(q, k, v, layer_id, batch)
            if observe:
                length = self.prefix if self.phase == 'cold' else req.device_len
                indices = meta.indices[:length].long()
                target = self.cold_kv if self.phase == 'cold' else self.warm_kv
                target[f'{layer_id}.k'] = keys.index_select(0, indices)
                target[f'{layer_id}.v'] = values.index_select(0, indices)
                self.save(f'{layer_id}.attention', result)
            return result
        self.bind(backend, 'forward', attention)

    def cpu(self):
        def convert(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().contiguous()
            if isinstance(value, dict):
                return {k:convert(v) for k,v in value.items()}
            return value
        return convert(dict(ops=self.ops, cold_kv=self.cold_kv, warm_kv=self.warm_kv,
            prefix_before=self.prefix_before, metadata=self.metadata, order=self.order))

    def close(self):
        for obj, name, original in reversed(self.restorers):
            setattr(obj, name, original)
        self.restorers.clear()
