"""Single-device Top-1 Mixture-of-Experts building blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from triton_grouped_gemm import grouped_gemm


class ExpertMLP(nn.Module):
    """The same two-layer GELU MLP used by the dense Transformer block."""

    def __init__(self, n_embd: int, *, bias: bool, dropout: float) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Top1Router(nn.Module):
    """Produce one expert index and one differentiable gate weight per token."""

    def __init__(self, n_embd: int, num_experts: int, *, bias: bool) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least one")
        self.num_experts = num_experts
        self.proj = nn.Linear(n_embd, num_experts, bias=bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep the probability calculation in FP32 under autocast. The selected
        # probability scales the expert output, providing a task-loss gradient
        # to the router even though argmax itself is non-differentiable.
        probabilities = F.softmax(self.proj(x), dim=-1, dtype=torch.float32)
        weights, indices = probabilities.max(dim=-1)
        return probabilities, weights, indices


def load_balancing_loss(
    probabilities: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Switch-style auxiliary loss for encouraging balanced Top-1 routing."""
    if probabilities.ndim != 2 or probabilities.shape[-1] != num_experts:
        raise ValueError("probabilities must have shape (tokens, num_experts)")
    if expert_indices.shape != probabilities.shape[:-1]:
        raise ValueError("expert_indices must contain one entry per token")
    if probabilities.shape[0] == 0:
        return probabilities.sum()

    token_fractions = F.one_hot(
        expert_indices, num_classes=num_experts
    ).to(probabilities.dtype).mean(dim=0)
    mean_probabilities = probabilities.mean(dim=0)
    # The hard assignment is non-differentiable. Detaching its frequency makes
    # that boundary explicit while router probabilities carry the gradient.
    return num_experts * torch.sum(
        token_fractions.detach() * mean_probabilities
    )


@dataclass(frozen=True)
class Top1Routing:
    """Routing evidence returned by ``Top1MoE(..., return_routing=True)``."""

    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    expert_counts: torch.Tensor
    processed_counts: torch.Tensor
    overflow_counts: torch.Tensor
    dropped_mask: torch.Tensor
    router_probability_sums: torch.Tensor
    capacity: int | None
    balance_loss: torch.Tensor

    @property
    def token_count(self) -> int:
        return self.expert_indices.numel()

    @property
    def processed_token_count(self) -> int:
        return int(self.processed_counts.sum().item())

    @property
    def overflow_token_count(self) -> int:
        return int(self.overflow_counts.sum().item())

    @property
    def dropped_token_count(self) -> int:
        return int(self.dropped_mask.sum().item())


class Top1MoE(nn.Module):
    """Route each token to exactly one local expert and restore token order."""

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        *,
        bias: bool,
        dropout: float,
        capacity_factor: float = 0.0,
        drop_tokens: bool = False,
        dispatch_backend: str = "loop",
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least one")
        self.n_embd = n_embd
        self.num_experts = num_experts
        if capacity_factor < 0.0:
            raise ValueError("capacity_factor cannot be negative")
        if drop_tokens and capacity_factor <= 0.0:
            raise ValueError("drop_tokens requires a positive capacity_factor")
        if dispatch_backend not in {"loop", "padded_torch", "padded_triton"}:
            raise ValueError(
                "dispatch_backend must be loop, padded_torch, or padded_triton"
            )
        if dispatch_backend != "loop" and (
            capacity_factor <= 0.0 or not drop_tokens
        ):
            raise ValueError(
                "padded dispatch requires token drop and positive capacity"
            )
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.dispatch_backend = dispatch_backend
        self.dropout_probability = dropout
        self.router = Top1Router(n_embd, num_experts, bias=bias)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(n_embd, bias=bias, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        self.last_routing: Top1Routing | None = None

    def _forward_loop(
        self,
        flat_input: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        capacity: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dropped_mask = torch.zeros(
            flat_input.shape[0],
            dtype=torch.bool,
            device=flat_input.device,
        )
        processed_counts = []
        flat_output = None
        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                expert_indices == expert_index, as_tuple=False
            ).squeeze(-1)
            if self.drop_tokens and capacity is not None:
                dropped_indices = token_indices[capacity:]
                dropped_mask[dropped_indices] = True
                token_indices = token_indices[:capacity]
            processed_counts.append(token_indices.numel())
            if token_indices.numel() == 0:
                continue
            expert_input = flat_input.index_select(0, token_indices)
            expert_output = expert(expert_input)
            selected_weights = expert_weights.index_select(
                0, token_indices
            ).to(expert_output.dtype)
            weighted_output = expert_output * selected_weights.unsqueeze(-1)
            if flat_output is None:
                # Under autocast the normalized block input can be FP32 while
                # expert Linear outputs are BF16. Match the result dtype.
                flat_output = torch.zeros(
                    flat_input.shape,
                    dtype=expert_output.dtype,
                    device=flat_input.device,
                )
            flat_output = torch.index_copy(
                flat_output, 0, token_indices, weighted_output
            )

        if flat_output is None:
            flat_output = torch.zeros_like(flat_input)
        return (
            flat_output,
            torch.tensor(
                processed_counts,
                dtype=expert_indices.dtype,
                device=expert_indices.device,
            ),
            dropped_mask,
        )

    def _stacked_expert_parameters(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if not all(isinstance(expert, ExpertMLP) for expert in self.experts):
            raise TypeError("padded dispatch requires ExpertMLP experts")
        fc_weight = torch.stack(
            [expert.c_fc.weight for expert in self.experts]
        )
        proj_weight = torch.stack(
            [expert.c_proj.weight for expert in self.experts]
        )
        fc_bias = None
        proj_bias = None
        if self.experts[0].c_fc.bias is not None:
            fc_bias = torch.stack(
                [expert.c_fc.bias for expert in self.experts]
            )
            proj_bias = torch.stack(
                [expert.c_proj.bias for expert in self.experts]
            )
        return fc_weight, fc_bias, proj_weight, proj_bias

    def _expert_grouped_mlp(self, packed_input: torch.Tensor) -> torch.Tensor:
        fc_weight, fc_bias, proj_weight, proj_bias = (
            self._stacked_expert_parameters()
        )
        if self.dispatch_backend == "padded_triton":
            if not packed_input.is_cuda:
                raise ValueError("padded_triton dispatch requires CUDA")
            if torch.is_autocast_enabled("cuda"):
                compute_dtype = torch.get_autocast_dtype("cuda")
            else:
                compute_dtype = packed_input.dtype
            packed_input = packed_input.to(compute_dtype)
            fc_weight = fc_weight.to(compute_dtype)
            proj_weight = proj_weight.to(compute_dtype)
            hidden = grouped_gemm(
                packed_input,
                fc_weight.transpose(1, 2),
            )
        else:
            hidden = torch.bmm(
                packed_input,
                fc_weight.transpose(1, 2),
            )
        if fc_bias is not None:
            hidden = hidden + fc_bias[:, None, :].to(hidden.dtype)
        hidden = F.gelu(hidden)
        if self.dispatch_backend == "padded_triton":
            expert_output = grouped_gemm(
                hidden,
                proj_weight.transpose(1, 2),
            )
        else:
            expert_output = torch.bmm(
                hidden,
                proj_weight.transpose(1, 2),
            )
        if proj_bias is not None:
            expert_output = (
                expert_output + proj_bias[:, None, :].to(expert_output.dtype)
            )
        return F.dropout(
            expert_output,
            p=self.dropout_probability,
            training=self.training,
        )

    def _forward_padded(
        self,
        flat_input: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assignments = F.one_hot(
            expert_indices, num_classes=self.num_experts
        )
        expert_counts = assignments.sum(dim=0)
        positions = assignments.cumsum(dim=0) - 1
        token_positions = positions.gather(
            1, expert_indices[:, None]
        ).squeeze(1)
        accepted = token_positions < capacity
        sentinel = self.num_experts * capacity
        destinations = expert_indices * capacity + token_positions
        destinations = torch.where(
            accepted,
            destinations,
            torch.full_like(destinations, sentinel),
        )

        packed_with_sentinel = torch.zeros(
            (sentinel + 1, self.n_embd),
            dtype=flat_input.dtype,
            device=flat_input.device,
        ).index_add(
            0,
            destinations,
            flat_input * accepted[:, None],
        )
        packed_input = packed_with_sentinel[:sentinel].reshape(
            self.num_experts, capacity, self.n_embd
        )
        packed_output = self._expert_grouped_mlp(packed_input)
        output_with_sentinel = torch.cat(
            (
                packed_output.reshape(sentinel, self.n_embd),
                torch.zeros(
                    (1, self.n_embd),
                    dtype=packed_output.dtype,
                    device=packed_output.device,
                ),
            ),
            dim=0,
        )
        flat_output = output_with_sentinel.index_select(0, destinations)
        flat_output = flat_output * expert_weights.to(
            flat_output.dtype
        ).unsqueeze(-1)
        processed_counts = torch.clamp(expert_counts, max=capacity)
        return flat_output, expert_counts, processed_counts, ~accepted

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Top1Routing]:
        if x.ndim < 2 or x.shape[-1] != self.n_embd:
            raise ValueError(
                f"expected input ending in {self.n_embd}, got {tuple(x.shape)}"
            )

        original_shape = x.shape
        flat_input = x.reshape(-1, self.n_embd)
        probabilities, expert_weights, expert_indices = self.router(flat_input)
        token_count = flat_input.shape[0]
        capacity = None
        if self.capacity_factor > 0.0:
            capacity = math.ceil(
                self.capacity_factor * token_count / self.num_experts
            )
        if self.dispatch_backend == "loop":
            expert_counts = torch.bincount(
                expert_indices, minlength=self.num_experts
            )
            flat_output, processed_counts, dropped_mask = self._forward_loop(
                flat_input,
                expert_weights,
                expert_indices,
                capacity,
            )
        else:
            if capacity is None:
                raise RuntimeError("padded dispatch capacity is missing")
            (
                flat_output,
                expert_counts,
                processed_counts,
                dropped_mask,
            ) = self._forward_padded(
                flat_input,
                expert_weights,
                expert_indices,
                capacity,
            )
        overflow_counts = (
            torch.clamp(expert_counts - capacity, min=0)
            if capacity is not None
            else torch.zeros_like(expert_counts)
        )
        output = flat_output.reshape(original_shape)
        routing = Top1Routing(
            expert_indices=expert_indices.reshape(original_shape[:-1]),
            expert_weights=expert_weights.reshape(original_shape[:-1]),
            expert_counts=expert_counts,
            processed_counts=processed_counts,
            overflow_counts=overflow_counts,
            dropped_mask=dropped_mask.reshape(original_shape[:-1]),
            router_probability_sums=probabilities.sum(dim=0),
            capacity=capacity,
            balance_loss=load_balancing_loss(
                probabilities, expert_indices, self.num_experts
            ),
        )
        self.last_routing = routing
        if not return_routing:
            return output
        return output, routing


def aggregate_moe_metrics(records: list[dict]) -> dict | None:
    """Aggregate raw per-forward MoE metrics into JSON-friendly values."""
    if not records:
        return None

    expert_counts = torch.stack(
        [record["expert_counts"] for record in records]
    ).sum(dim=0)
    processed_counts = torch.stack(
        [record["processed_counts"] for record in records]
    ).sum(dim=0)
    overflow_counts = torch.stack(
        [record["overflow_counts"] for record in records]
    ).sum(dim=0)
    router_probability_sums = torch.stack(
        [record["router_probability_sums"] for record in records]
    ).sum(dim=0)
    token_count = sum(record["token_count"] for record in records)
    dropped_token_count = sum(
        record["dropped_token_count"] for record in records
    )

    loads = expert_counts.to(torch.float32)
    mean_load = loads.mean()
    max_load = loads.max()
    if mean_load.item() == 0.0:
        max_to_mean = 0.0
        coefficient_of_variation = 0.0
    else:
        max_to_mean = (max_load / mean_load).item()
        coefficient_of_variation = (
            loads.std(unbiased=False) / mean_load
        ).item()

    if token_count == 0 or expert_counts.numel() == 1:
        normalized_entropy = 1.0
    else:
        fractions = loads / token_count
        positive = fractions > 0
        entropy = -(fractions[positive] * fractions[positive].log()).sum()
        normalized_entropy = (
            entropy / math.log(expert_counts.numel())
        ).item()

    def mean_optional(name: str) -> float | None:
        values = [record[name] for record in records if record[name] is not None]
        if not values:
            return None
        return torch.stack(values).mean().item()

    return {
        "expert_counts": expert_counts.cpu().tolist(),
        "processed_counts": processed_counts.cpu().tolist(),
        "overflow_counts": overflow_counts.cpu().tolist(),
        "token_count": token_count,
        "processed_token_count": int(processed_counts.sum().item()),
        "overflow_token_count": int(overflow_counts.sum().item()),
        "dropped_token_count": dropped_token_count,
        "drop_rate": (
            dropped_token_count / token_count if token_count else 0.0
        ),
        "mean_load": mean_load.item(),
        "max_load": max_load.item(),
        "max_to_mean_load": max_to_mean,
        "load_coefficient_of_variation": coefficient_of_variation,
        "normalized_load_entropy": normalized_entropy,
        "mean_router_probabilities": (
            router_probability_sums / max(token_count, 1)
        ).cpu().tolist(),
        "capacity_per_micro_batch": [
            record["capacity"] for record in records
        ],
        "mean_lm_loss": mean_optional("lm_loss"),
        "mean_balance_loss": mean_optional("balance_loss"),
        "mean_total_loss": mean_optional("total_loss"),
    }
