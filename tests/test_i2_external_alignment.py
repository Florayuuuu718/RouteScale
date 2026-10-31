"""Causal paged attention must honor suffix positions, page indirection and GQA."""
from types import SimpleNamespace

import pytest
import torch

from serving.i2_external_alignment import install_precision_intervention, paged_math_attention


@pytest.mark.parametrize('prefixes,queries', [([0], [5]), ([3], [2]), ([3, 0], [2, 3]), ([4, 2], [1, 1])])
def test_paged_attention_matches_independent_tokenwise_reference(prefixes, queries):
    torch.manual_seed(410)
    heads, qheads, dim = 2, 4, 8
    lengths = [p+n for p, n in zip(prefixes, queries)]
    count = sum(lengths)
    mapping = torch.randperm(count)
    logical_k = torch.randn(count, heads, dim)
    logical_v = torch.randn_like(logical_k)
    stored_k = torch.full_like(logical_k, float('nan'))
    stored_v = torch.full_like(logical_v, float('nan'))
    out_locs, kv_indices, positions, current_k, current_v = [], [], [], [], []
    start = 0
    for prefix, n in zip(prefixes, queries):
        indices = mapping[start:start+prefix+n]
        stored_k[indices[:prefix]] = logical_k[start:start+prefix]
        stored_v[indices[:prefix]] = logical_v[start:start+prefix]
        out_locs.extend(indices[prefix:].tolist())
        kv_indices.extend(indices.tolist())
        positions.extend(range(prefix, prefix+n))
        current_k.append(logical_k[start+prefix:start+prefix+n])
        current_v.append(logical_v[start+prefix:start+prefix+n])
        start += prefix+n
    q = torch.randn(sum(queries), qheads, dim)
    batch = SimpleNamespace(reqs=[object() for _ in queries], out_loc=torch.tensor(out_locs),
        positions=torch.tensor(positions), attn_metadata=SimpleNamespace(
            cu_seqlens_q_cpu=torch.tensor([0, *queries]).cumsum(0),
            cu_seqlens_k_cpu=torch.tensor([0, *lengths]).cumsum(0), indices=torch.tensor(kv_indices)))
    cache = SimpleNamespace(_storage_shape=stored_k.shape, k_cache=lambda i: stored_k, v_cache=lambda i: stored_v)
    result = paged_math_attention(q, torch.cat(current_k), torch.cat(current_v), 0, batch, cache)
    expected = torch.empty_like(result)
    qi = ki = 0
    for prefix, n in zip(prefixes, queries):
        for t in range(n):
            for h in range(qheads):
                k = logical_k[ki:ki+prefix+t+1, h//2].double()
                v = logical_v[ki:ki+prefix+t+1, h//2].double()
                scores = k @ q[qi+t, h].double() / dim**.5
                expected[qi+t, h] = scores.softmax(0) @ v
        qi += n
        ki += prefix+n
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_residual_intervention_keeps_unrounded_sum_and_restores_norms(dtype):
    def untouched(x, residual=None):
        raise AssertionError('Original operator should not execute during intervention')
    def norm():
        return SimpleNamespace(weight=torch.tensor([1., 2.], dtype=dtype), eps=1e-6, forward=untouched)
    layer = SimpleNamespace(input_layernorm=norm(), post_attention_layernorm=norm())
    model = SimpleNamespace(layers=SimpleNamespace(op_list=[layer]), norm=norm())
    engine = SimpleNamespace(model=SimpleNamespace(model=model), graph_runner=SimpleNamespace(graph_bs_list=[]))
    restore = install_precision_intervention(engine, 'residual_fp32')
    try:
        half_ulp = .00390625 if dtype == torch.bfloat16 else .00048828125
        x = torch.tensor([[half_ulp, half_ulp * 2]], dtype=dtype)
        residual = torch.ones_like(x)
        actual, stored = layer.input_layernorm.forward(x, residual)
        total = x.float() + residual.float()
        expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) *
                    layer.input_layernorm.weight.float()).to(dtype)
        assert stored.dtype == torch.float32
        assert torch.equal(stored, total)
        assert not torch.equal(stored, total.to(dtype).float())
        assert torch.equal(actual, expected)
        assert actual.dtype == dtype
    finally:
        restore()
    assert layer.input_layernorm.forward is untouched
    assert layer.post_attention_layernorm.forward is untouched
    assert model.norm.forward is untouched


def test_precision_intervention_rejects_captured_graphs():
    engine = SimpleNamespace(graph_runner=SimpleNamespace(graph_bs_list=[1]))
    with pytest.raises(ValueError, match='captured graphs'):
        install_precision_intervention(engine, 'fp32_math')
