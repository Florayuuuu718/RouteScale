"""Explicit instance policy for I2 diagnostics; leaves the frozen native tree intact."""

POLICY = 'fi_fa2_prefill_no_split_v1'


def install_prefill_no_split(engine):
    backend=engine.attn_backend
    if type(backend).__name__ != 'FlashInferBackend':
        raise ValueError('This policy is qualified only for FlashInfer fa2')
    wrapper=backend.prefill_wrapper
    original=wrapper.plan
    def plan(*args,**kwargs):
        if kwargs.get('fixed_split_size') is not None:
            raise ValueError('A fixed split size conflicts with the no-split policy')
        return original(*args,**dict(kwargs,disable_split_kv=True))
    wrapper.plan=plan
    def restore():
        wrapper.plan=original
    return restore
