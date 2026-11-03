"""Eager Top-1 expert parallelism; default process group, no hidden DDP averaging.

Collectives are synchronous and ordered. Backward reverses each variable-split
exchange. This deliberately small implementation establishes training semantics
before communication overlap or fixed-shape kernels are considered.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.profiler import record_function

from moe import Top1MoE


def exchange(x: torch.Tensor, send: list[int], receive: list[int]) -> torch.Tensor:
    output = x.new_empty((sum(receive), *x.shape[1:]))
    dist.all_to_all_single(output, x.contiguous(), receive, send)
    return output


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send, receive):
        ctx.send, ctx.receive = send, receive
        with record_function("ep/activation_all_to_all"):
            return exchange(x, send, receive)

    @staticmethod
    def backward(ctx, gradient):
        with record_function("ep/gradient_all_to_all"):
            return exchange(gradient, ctx.receive, ctx.send), None, None


@dataclass
class EPRouting:
    expert_indices: torch.Tensor
    dropped_mask: torch.Tensor
    global_counts: torch.Tensor
    global_processed_counts: torch.Tensor
    send_counts: list[int]
    receive_counts: list[int]
    capacity: int | None
    balance_loss: torch.Tensor


def global_routing(indices, num_experts, capacity_factor, drop_tokens):
    """Global counts and stable rank-concatenated admission, before exchange."""
    counts = torch.bincount(indices, minlength=num_experts)
    by_rank = [torch.empty_like(counts) for _ in range(dist.get_world_size())]
    with record_function("ep/routing_counts_all_gather"):
        dist.all_gather(by_rank, counts)
    by_rank = torch.stack(by_rank)
    totals = by_rank.sum(0)
    tokens = int(totals.sum().item())
    if tokens == 0:
        raise ValueError("EP global batch must contain at least one token")
    capacity = math.ceil(capacity_factor * tokens / num_experts) if capacity_factor else None
    prefix = by_rank[:dist.get_rank()].sum(0)
    assignments = torch.nn.functional.one_hot(indices, num_experts)
    positions = (assignments.cumsum(0) - 1).gather(1, indices[:, None]).squeeze(1)
    accepted = torch.ones_like(indices, dtype=torch.bool)
    if drop_tokens:
        accepted = positions + prefix[indices] < capacity
    return totals, accepted, capacity


def sum_replicated_gradients(parameters):
    """SUM, preserving globally absent gradients and optimizer state semantics."""
    for parameter in parameters:
        used = torch.tensor(int(parameter.grad is not None), device=parameter.device)
        dist.all_reduce(used)
        if used.item() == 0:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)


class ExpertParallelMoE(nn.Module):
    def __init__(self, source: Top1MoE):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("initialize a default process group before constructing EP")
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        if source.num_experts % self.world:
            raise ValueError("num_experts must be divisible by EP world size")
        if source.dropout_probability != 0:
            raise ValueError("F v1 requires dropout=0 for reproducible token exchanges")
        self.n_embd = source.n_embd
        self.num_experts = source.num_experts
        self.per_rank = self.num_experts // self.world
        self.capacity_factor = source.capacity_factor
        self.drop_tokens = source.drop_tokens
        self.router = copy.deepcopy(source.router)
        self.experts = nn.ModuleDict({
            str(i): copy.deepcopy(source.experts[i])
            for i in range(self.rank * self.per_rank, (self.rank + 1) * self.per_rank)
        })
        self.last_routing = None

    def forward(self, x):
        if x.ndim < 2 or x.shape[-1] != self.n_embd or not x.requires_grad:
            raise ValueError("training EP requires grad-enabled inputs ending in n_embd")
        flat = x.reshape(-1, self.n_embd)
        probabilities, weights, indices = self.router(flat)
        counts, accepted, capacity = global_routing(
            indices, self.num_experts, self.capacity_factor, self.drop_tokens
        )
        destinations = indices // self.per_rank
        selected = torch.nonzero(accepted, as_tuple=False).flatten()
        order = torch.argsort(destinations[selected], stable=True)
        packed_ids = selected[order]
        send_tensor = torch.bincount(destinations[packed_ids], minlength=self.world)
        receive_tensor = torch.empty_like(send_tensor)
        with record_function("ep/split_counts_all_to_all"):
            dist.all_to_all_single(receive_tensor, send_tensor)
        send, receive = send_tensor.tolist(), receive_tensor.tolist()
        with record_function("ep/expert_ids_all_to_all"):
            received_ids = exchange(indices[packed_ids], send, receive)
        with record_function("ep/dispatch"):
            received = _AllToAll.apply(flat[packed_ids], send, receive)
        compute_dtype = (torch.get_autocast_dtype(x.device.type)
                         if torch.is_autocast_enabled(x.device.type) else x.dtype)
        # Keep every rank in backward even if this owner receives no tokens.
        # No unselected expert parameter is attached to the graph.
        computed = received.to(compute_dtype) * 0
        with record_function("ep/local_experts"):
            for key, expert in self.experts.items():
                slots = torch.nonzero(received_ids == int(key), as_tuple=False).flatten()
                if slots.numel():
                    computed = computed.index_copy(0, slots, expert(received[slots]))
        with record_function("ep/combine"):
            returned = _AllToAll.apply(computed, receive, send)
        output = returned.new_zeros(flat.shape).index_copy(0, packed_ids, returned)
        output = output * weights.to(output.dtype)[:, None]
        frequencies = counts.to(probabilities.dtype) / counts.sum()
        balance = self.num_experts * (
            frequencies.detach() * probabilities.sum(0) / counts.sum()
        ).sum()
        processed = counts.clamp(max=capacity) if self.drop_tokens else counts
        self.last_routing = EPRouting(
            indices, ~accepted, counts, processed, send, receive, capacity, balance
        )
        return output.reshape(x.shape)

    def synchronize_router_gradients(self):
        sum_replicated_gradients(self.router.parameters())


def replicated_moe_control(source, x, global_counts, accepted):
    """All experts replicated, local token computation, globally shared drop mask.

    The caller explicitly SUM-reduces parameter gradients. It is a semantic
    DDP control rather than a wrapper around PyTorch's DDP reducer.
    """
    flat = x.reshape(-1, source.n_embd)
    probabilities, weights, indices = source.router(flat)
    output = flat * 0
    compute_dtype = (torch.get_autocast_dtype(x.device.type)
                     if torch.is_autocast_enabled(x.device.type) else x.dtype)
    output = output.to(compute_dtype)
    for i, expert in enumerate(source.experts):
        slots = torch.nonzero((indices == i) & accepted, as_tuple=False).flatten()
        if slots.numel():
            values = expert(flat[slots]) * weights[slots].to(compute_dtype)[:, None]
            output = output.index_copy(0, slots, values)
    balance = source.num_experts * (
        global_counts.to(probabilities.dtype) / global_counts.sum()
        * probabilities.sum(0) / global_counts.sum()
    ).sum()
    return output.reshape(x.shape), balance
