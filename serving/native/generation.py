"""Greedy full-recompute and contiguous-KV generation paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model import GPT


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    step_logits: list[torch.Tensor]
    cache_rebuilds: int
    cache_allocated_bytes: int


@torch.inference_mode()
def greedy_generate_naive(
    model: GPT, prompt_ids: torch.Tensor, max_new_tokens: int
) -> GenerationResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = prompt_ids
    step_logits: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :]
        step_logits.append(logits.float().cpu())
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return GenerationResult(generated, step_logits, 0, 0)


@torch.inference_mode()
def greedy_generate_cached(
    model: GPT, prompt_ids: torch.Tensor, max_new_tokens: int
) -> GenerationResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape [batch, positive sequence]")
    generated = prompt_ids
    if max_new_tokens == 0:
        return GenerationResult(generated, [], 0, 0)

    cache = model.allocate_kv_cache(prompt_ids.shape[0])
    context = generated[:, -model.config.block_size :]
    logits = model.forward_with_kv_cache(context, cache)[:, -1, :]
    step_logits: list[torch.Tensor] = []
    rebuilds = 0
    for step in range(max_new_tokens):
        step_logits.append(logits.float().cpu())
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if step + 1 == max_new_tokens:
            break
        if cache.length == cache.capacity:
            cache = model.allocate_kv_cache(prompt_ids.shape[0])
            context = generated[:, -model.config.block_size :]
            logits = model.forward_with_kv_cache(context, cache)[:, -1, :]
            rebuilds += 1
        else:
            logits = model.forward_with_kv_cache(next_token, cache)[:, -1, :]
    return GenerationResult(
        generated,
        step_logits,
        rebuilds,
        cache.allocated_bytes,
    )

