"""Opt-in, BF16-preserving linear tiling for the bounded scheduling pilot."""

from __future__ import annotations

from types import MethodType

LINEAR_POLICY = dict(policy_id="prefill_linear_tile512_v1", tile_rows=512,
    allowed_prefill_rows=[512, 1024], require_exact_logits=True)


def validate_linear_policy(policy):
    if policy is not None and policy != LINEAR_POLICY:
        raise ValueError("Unknown Prefill linear policy; qualify a separate scope instead")


def install_prefill_linear_policy(linears, get_context, policy):
    """Use identical 512-row GEMMs without padding, casting, or changing weights.

    Apply after upstream model warmup/Graph capture. Decode remains untouched;
    this controls only the 4/8-request, 128-token Prefill shapes in this pilot.
    The restorer must be called and dropped before destroying the runner.
    """
    validate_linear_policy(policy)
    if policy is None:
        raise ValueError("Installing a linear adapter requires an explicit policy")
    import torch
    saved = []

    def wrap(original):
        def forward(self, inputs):
            if not get_context().is_prefill:
                return original(inputs)
            if (inputs.ndim != 2 or inputs.shape[0] not in policy["allowed_prefill_rows"]
                    or inputs.dtype != torch.bfloat16):
                raise ValueError("Prefill linear shape/dtype is outside the qualified tile scope")
            chunks = [original(inputs[start:start+policy["tile_rows"]])
                for start in range(0, inputs.shape[0], policy["tile_rows"])]
            return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        return forward

    for module in linears:
        saved.append((module, module.forward))
        module.forward = MethodType(wrap(module.forward), module)
    if not saved:
        raise ValueError("No linear modules were selected")

    def restore():
        for module, original in saved:
            module.forward = original
        saved.clear()

    return restore, dict(**policy, linear_modules=len(saved), padding_rows=0,
        weight_dtype_unchanged=True, decode_unchanged=True)
