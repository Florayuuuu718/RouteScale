"""Isolate BF16 operators on identical native inputs; diagnostic, never a timing gate."""
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/vendor/mini-sglang-i2/python'))


def main():
    lock = (ROOT / 'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from serving.i2_workloads import CONFIG, digest
    from serving.i2_runtime import ReplayScheduler, flush_cache
    from serving.i2_math_reference import rms, fused_add_norm
    config = json.loads(CONFIG.read_text())
    output = ROOT / config['results_dir'] / 'kernel_diagnostic.json'
    if output.exists():
        raise ValueError('Preserve diagnostic')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    scheduler = None

    def compare(name, actual, expected):
        a, b = actual.float(), expected.float()
        diff = a - b
        rows.append(dict(name=name, shape=list(a.shape), exact=bool(torch.equal(a, b)),
            max_abs=float(diff.abs().max()), relative_rmse=float(diff.norm() / b.norm().clamp_min(1e-12)),
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())))

    with torch.inference_mode():
        try:
            scheduler = ReplayScheduler(config, 'naive_full')
            model = scheduler.engine.model
            for index, layer in enumerate(model.model.layers.op_list):
                for name in ('input_layernorm', 'post_attention_layernorm'):
                    norm = getattr(layer, name)
                    native = norm.forward
                    def wrapped(x, residual=None, *, op=norm, fn=native, name=f'{index}.{name}'):
                        # Native fused norm overwrites x and residual in place.
                        expected = (rms(x, op.weight, op.eps), x.clone()) if residual is None else fused_add_norm(x, residual, op.weight, op.eps)
                        actual = fn(x, residual)
                        compare(name, actual[0], expected[0])
                        compare(name+'.residual', actual[1], expected[1])
                        return actual
                    norm.forward = wrapped
                attn = layer.self_attn.attn
                rotary = attn.rotary
                # Rotary object is cached and shared by every layer. Wrap once.
                if index == 0:
                    native = rotary.forward
                    def rope(positions, query, key, fn=native, op=rotary):
                        expected = []
                        cos, sin = op._cos_sin_cache[positions].float().chunk(2, dim=-1)
                        for value in (query, key):
                            v = value.view(len(positions), -1, op.head_size).float()
                            a, b = v.chunk(2, dim=-1)
                            expected.append(torch.cat([a*cos[:,None]-b*sin[:,None], b*cos[:,None]+a*sin[:,None]], -1).reshape_as(value).to(value.dtype))
                        actual = fn(positions, query, key)
                        compare('rope.q', actual[0], expected[0]); compare('rope.k', actual[1], expected[1])
                        return actual
                    rotary.forward = rope
                for name in ('q_norm', 'k_norm'):
                    norm = getattr(layer.self_attn, name)
                    native = norm.forward_inplace
                    def qknorm(x, *, op=norm, fn=native, name=f'{index}.{name}'):
                        expected = rms(x, op.weight, op.eps)
                        result = fn(x)
                        compare(name, x, expected)
                        return result
                    norm.forward_inplace = qknorm
                native = layer.mlp.act_fn
                def activation(x, *, fn=native, name=f'{index}.silu_mul'):
                    a, b = x.float().chunk(2, dim=-1)
                    expected = (F.silu(a)*b).to(x.dtype)
                    actual = fn(x)
                    compare(name, actual, expected)
                    return actual
                layer.mlp.act_fn = activation
            backend = scheduler.engine.attn_backend
            native = backend.forward
            def attention(q, k, v, layer_id, batch):
                # Only a cold, single-request prefill is used by this diagnostic.
                if batch.is_decode:
                    return native(q, k, v, layer_id, batch)
                qh, kh = backend.qo_head_local, backend.kv_head_local
                size = backend.config.head_dim
                qt=q.transpose(0,1).unsqueeze(0).float()
                kt=k.view(-1,kh,size).transpose(0,1).unsqueeze(0).float()
                vt=v.view(-1,kh,size).transpose(0,1).unsqueeze(0).float()
                with sdpa_kernel(SDPBackend.MATH):
                    expected=F.scaled_dot_product_attention(qt,kt,vt,is_causal=True,enable_gqa=True).to(q.dtype)
                expected=expected.squeeze(0).transpose(0,1).contiguous()
                actual=native(q,k,v,layer_id,batch)
                compare(f'{layer_id}.attention',actual,expected)
                return actual
            backend.forward=attention
            workload=json.loads((ROOT/config['workload_path']).read_text())
            request=next(c for c in workload['cases'] if c['case_id']=='b1_zero')['waves']['cold'][0].copy()
            request['continuation_ids']=request['continuation_ids'][:1]
            flush_cache(scheduler,poison=True)
            scheduler.wave([request])
        finally:
            if scheduler is not None:
                scheduler.shutdown()
    result=dict(stage='I2',evidence_class='operator_rounding_diagnostic_only',performance_allowed=False,
        source_sha256=digest(Path(__file__)), rows=rows)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for kind in ('input_layernorm','post_attention_layernorm','q_norm','k_norm','silu_mul','attention','rope'):
        selected=[r for r in rows if kind in r['name'] and not r['name'].endswith('.residual')]
        print(kind, 'max_abs', max(r['max_abs'] for r in selected), 'max_relative_rmse',max(r['relative_rmse'] for r in selected))


if __name__=='__main__':
    main()
