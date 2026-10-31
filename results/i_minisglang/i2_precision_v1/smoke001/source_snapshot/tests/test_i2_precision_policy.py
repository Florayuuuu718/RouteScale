"""State/precision contracts for pre-capture policy installation."""
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from serving.i2_precision_policy import engine_initializer, install_residual_norms, restore_engine_policy


def model(dtype=torch.float16):
    def original(x, residual=None):
        return x, residual
    norm = lambda: SimpleNamespace(weight=torch.tensor([1., 2.], dtype=dtype), eps=1e-6, forward=original)
    layer = SimpleNamespace(input_layernorm=norm(), post_attention_layernorm=norm())
    return SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[layer]), norm=norm(),
        embed_tokens=SimpleNamespace(weight=torch.zeros(2, 2, dtype=dtype))))


def test_norm_preserves_half_ulp_residual_and_restores():
    m = model()
    original = m.model.norm.forward
    restore, count = install_residual_norms(m)
    assert count == 3
    with pytest.raises(ValueError, match='already installed'):
        install_residual_norms(m)
    x = torch.tensor([[.00048828125, .0009765625]], dtype=torch.float16)
    r = torch.ones_like(x)
    actual, residual = m.model.norm.forward(x, r)
    total = x.float() + r.float()
    expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) * m.model.norm.weight.float()).half()
    assert torch.equal(actual, expected)
    assert torch.equal(residual, total) and residual.dtype == torch.float32
    assert not torch.equal(residual, total.half().float())
    restore()
    assert m.model.norm.forward is original


def test_norm_rejects_bf16_weights():
    with pytest.raises(ValueError, match='FP16'):
        install_residual_norms(model(torch.bfloat16))


@pytest.mark.parametrize('fail', [False, True])
def test_installation_precedes_graph_constructor_and_rolls_back(fail):
    m = model()
    original = m.model.norm.forward
    entries = []
    def graph(**kwargs):
        assert kwargs['model']._i2_residual_policy
        entries.append(True)
        if fail:
            raise RuntimeError('capture failed')
        return SimpleNamespace(graph_bs_list=[1, 2, 4])
    module = SimpleNamespace(GraphRunner=graph)
    @dataclass(frozen=True)
    class Config:
        dtype: object = torch.bfloat16
        tp_info: object = None
    config = Config(tp_info=SimpleNamespace(size=1))
    engine = SimpleNamespace()
    def native(e, cfg):
        assert cfg.dtype == torch.float16 and config.dtype == torch.bfloat16
        e.dtype, e.model = cfg.dtype, m
        e.kv_cache = SimpleNamespace(dtype=cfg.dtype)
        e.attn_backend = SimpleNamespace()
        e.graph_runner = module.GraphRunner(model=m)
    with patch('serving.i2_numerical_policy.install_prefill_no_split', return_value=lambda: None), \
         patch('serving.i2_linear_tile_policy.install_prefill_linear_tiles', return_value=lambda: None):
        init = engine_initializer(native, module, 'mixed')
        if fail:
            with pytest.raises(RuntimeError, match='capture failed'):
                init(engine, config)
        else:
            init(engine, config)
            assert engine._i2_precision_observed['graph_install']['norms_installed'] == 3
            assert engine._i2_precision_observed['captured_buckets'] == [1, 2, 4]
            restore_engine_policy(engine)
    assert entries == [True] and module.GraphRunner is graph
    assert m.model.norm.forward is original
