"""Explicit, opt-in H2 shape controls; historical native harness is untouched."""

from __future__ import annotations

import math
from types import MethodType


def install_shape_policy(runner, *, static_tables=False, paged_prefill=False):
    """Use identical page-table capacity and/or Prefill kernel family.

    These are adapter interventions, not upstream defaults. The returned callable
    restores the original methods. No precision, sampling or cache-reuse rule is
    changed. Install after initialization; Graph capture already uses full width.
    """
    from nanovllm.utils.context import get_context
    import torch.nn.functional as functional

    original_tables, original_prefill = runner.prepare_block_tables, runner.prepare_prefill
    width = math.ceil(runner.config.max_model_len / runner.block_size)

    def tables(self, seqs):
        result = original_tables(seqs)
        if result.size(1) > width:
            raise ValueError("Block table exceeds the configured model window")
        return functional.pad(result, (0, width - result.size(1)), value=0)

    def prefill(self, seqs):
        result = original_prefill(seqs)
        context = get_context()
        if all(seq.block_table for seq in seqs):
            context.block_tables = self.prepare_block_tables(seqs)
        return result

    if static_tables:
        runner.prepare_block_tables = MethodType(tables, runner)
    if paged_prefill:
        runner.prepare_prefill = MethodType(prefill, runner)

    def restore():
        runner.prepare_block_tables, runner.prepare_prefill = original_tables, original_prefill
    return restore
