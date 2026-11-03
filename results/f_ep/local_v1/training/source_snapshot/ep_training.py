"""Global admission, loss scaling and clipping for F training preparation."""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from expert_parallel import (EPRouting, ExpertParallelMoE, global_routing,
                             replicated_moe_control, sum_replicated_gradients)


class ReplicatedGlobalMoE(nn.Module):
    """All experts on every rank, with the same global admission as EP."""
    def __init__(self, source):
        super().__init__()
        self.router, self.experts = source.router, source.experts
        for name in ("n_embd", "num_experts", "capacity_factor", "drop_tokens"):
            setattr(self, name, getattr(source, name))
        self.last_routing = None

    def forward(self, x):
        flat = x.reshape(-1, self.n_embd)
        probabilities, weights, indices = self.router(flat)
        counts, accepted, capacity = global_routing(indices, self.num_experts, self.capacity_factor, self.drop_tokens)
        compute_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        output = (flat * 0).to(compute_dtype)
        for index, expert in enumerate(self.experts):
            slots = torch.nonzero((indices == index) & accepted, as_tuple=False).flatten()
            if slots.numel():
                values = expert(flat[slots]) * weights[slots].to(compute_dtype)[:, None]
                output = output.index_copy(0, slots, values)
        balance = self.num_experts * (counts.to(probabilities.dtype) / counts.sum()
                                     * probabilities.sum(0) / counts.sum()).sum()
        self.last_routing = EPRouting(indices, ~accepted, counts,
                                     counts.clamp(max=capacity) if self.drop_tokens else counts,
                                     [], [], capacity, balance)
        return output.reshape(x.shape)


def parameter_groups(model, mode):
    shared, sharded = [], []
    prefix = f"transformer.h.{model.config.moe_layer_index}.mlp.experts."
    for name, parameter in model.named_parameters():
        (sharded if mode == "ep" and name.startswith(prefix) else shared).append(parameter)
    return shared, sharded


def synchronize(model, mode):
    if mode != "reference":
        shared, _ = parameter_groups(model, mode)
        sum_replicated_gradients(shared)


@torch.no_grad()
def clip_global_grad_norm(shared, sharded, max_norm):
    """Count replicated grads once and expert shards across ranks; keep None."""
    shared, sharded = list(shared), list(sharded)
    parameters = shared + sharded
    if not parameters or max_norm <= 0:
        raise ValueError("positive max_norm and a nonempty parameter set are required")
    device = parameters[0].device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in shared:
        if parameter.grad is not None:
            total += parameter.grad.double().square().sum() / dist.get_world_size()
    for parameter in sharded:
        if parameter.grad is not None:
            total += parameter.grad.double().square().sum()
    dist.all_reduce(total)
    norm = total.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite global gradient norm; optimizer step prohibited")
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return norm


def install(model, mode):
    if mode == "reference":
        return
    block = model.transformer.h[model.config.moe_layer_index]
    block.mlp = ExpertParallelMoE(block.mlp) if mode == "ep" else ReplicatedGlobalMoE(block.mlp)


def scaled_loss(model, logits, targets, global_tokens):
    routing = model.transformer.h[model.config.moe_layer_index].mlp.last_routing
    return (torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
            / global_tokens + model.config.moe_balance_loss_weight * routing.balance_loss)
