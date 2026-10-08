from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from model import GPT, GPTConfig, MLP
from moe import Top1MoE, aggregate_moe_metrics, load_balancing_loss


class _ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Top1MoETests(unittest.TestCase):
    def test_routes_every_token_once_and_restores_order(self) -> None:
        moe = Top1MoE(2, 2, bias=False, dropout=0.0)
        moe.experts = nn.ModuleList([_ScaleExpert(2.0), _ScaleExpert(3.0)])
        with torch.no_grad():
            moe.router.proj.weight.copy_(
                torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
            )
        x = torch.tensor([[[2.0, 1.0], [-2.0, 4.0], [1.0, -3.0]]])

        output, routing = moe(x, return_routing=True)

        expected_scales = torch.tensor([2.0, 3.0, 2.0]).view(1, 3, 1)
        expected = x * expected_scales * routing.expert_weights.unsqueeze(-1)
        torch.testing.assert_close(output, expected)
        self.assertEqual(routing.token_count, x.shape[0] * x.shape[1])
        self.assertEqual(routing.expert_counts.tolist(), [2, 1])
        self.assertEqual(routing.expert_counts.sum().item(), routing.token_count)
        self.assertEqual(routing.expert_indices.tolist(), [[0, 1, 0]])

    def test_one_expert_matches_dense_output_and_gradients(self) -> None:
        config = SimpleNamespace(n_embd=8, bias=True, dropout=0.0)
        dense = MLP(config)
        moe = Top1MoE(8, 1, bias=True, dropout=0.0)
        moe.experts[0].load_state_dict(dense.state_dict())
        dense_input = torch.randn(2, 3, 8, requires_grad=True)
        moe_input = dense_input.detach().clone().requires_grad_(True)

        dense_output = dense(dense_input)
        moe_output = moe(moe_input)
        torch.testing.assert_close(moe_output, dense_output)

        gradient = torch.randn_like(dense_output)
        dense_output.backward(gradient)
        moe_output.backward(gradient)
        torch.testing.assert_close(moe_input.grad, dense_input.grad)
        for dense_parameter, expert_parameter in zip(
            dense.parameters(), moe.experts[0].parameters(), strict=True
        ):
            torch.testing.assert_close(expert_parameter.grad, dense_parameter.grad)

    def test_router_and_selected_experts_receive_gradients(self) -> None:
        moe = Top1MoE(2, 3, bias=False, dropout=0.0)
        with torch.no_grad():
            moe.router.proj.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [-2.0, 0.0],
                        [0.0, -10.0],
                    ]
                )
            )
        x = torch.tensor(
            [[[2.0, 1.0], [-2.0, 1.0], [3.0, 1.0], [-3.0, 1.0]]],
            requires_grad=True,
        )

        output, routing = moe(x, return_routing=True)
        output.square().sum().backward()

        self.assertEqual(routing.expert_counts.tolist(), [2, 2, 0])
        self.assertIsNotNone(moe.router.proj.weight.grad)
        self.assertGreater(moe.router.proj.weight.grad.abs().sum().item(), 0.0)
        for expert in moe.experts[:2]:
            gradients = [p.grad for p in expert.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in moe.experts[2].parameters())
        )

    def test_bfloat16_autocast_uses_expert_output_dtype(self) -> None:
        moe = Top1MoE(8, 2, bias=False, dropout=0.0)
        x = torch.randn(2, 3, 8)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = moe(x)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())

    def test_capacity_observes_overflow_without_dropping(self) -> None:
        moe = Top1MoE(
            2,
            2,
            bias=False,
            dropout=0.0,
            capacity_factor=0.5,
            drop_tokens=False,
        )
        with torch.no_grad():
            moe.router.proj.weight.zero_()
        x = torch.randn(1, 4, 2)

        _, routing = moe(x, return_routing=True)

        self.assertEqual(routing.capacity, 1)
        self.assertEqual(routing.expert_counts.tolist(), [4, 0])
        self.assertEqual(routing.processed_counts.tolist(), [4, 0])
        self.assertEqual(routing.overflow_counts.tolist(), [3, 0])
        self.assertEqual(routing.overflow_token_count, 3)
        self.assertEqual(routing.dropped_token_count, 0)

    def test_token_drop_is_deterministic_and_leaves_zero_moe_output(self) -> None:
        moe = Top1MoE(
            2,
            2,
            bias=False,
            dropout=0.0,
            capacity_factor=0.5,
            drop_tokens=True,
        )
        moe.experts = nn.ModuleList([_ScaleExpert(2.0), _ScaleExpert(3.0)])
        with torch.no_grad():
            moe.router.proj.weight.zero_()
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

        output, routing = moe(x, return_routing=True)

        torch.testing.assert_close(output[:, :1], x[:, :1])
        torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))
        self.assertEqual(routing.processed_counts.tolist(), [1, 0])
        self.assertEqual(routing.dropped_mask.tolist(), [[False, True, True, True]])
        self.assertEqual(routing.dropped_token_count, 3)

    def test_balance_loss_is_lower_for_balanced_routing(self) -> None:
        collapsed_probabilities = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
            requires_grad=True,
        )
        balanced_probabilities = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
            requires_grad=True,
        )

        collapsed = load_balancing_loss(
            collapsed_probabilities,
            torch.tensor([0, 0, 0, 0]),
            2,
        )
        balanced = load_balancing_loss(
            balanced_probabilities,
            torch.tensor([0, 0, 1, 1]),
            2,
        )

        self.assertLess(balanced.item(), collapsed.item())
        collapsed.backward()
        self.assertGreater(collapsed_probabilities.grad.abs().sum().item(), 0.0)

    def test_aggregate_metrics_reports_load_dispersion(self) -> None:
        record = {
            "expert_counts": torch.tensor([3, 1]),
            "processed_counts": torch.tensor([2, 1]),
            "overflow_counts": torch.tensor([1, 0]),
            "router_probability_sums": torch.tensor([2.5, 1.5]),
            "token_count": 4,
            "dropped_token_count": 1,
            "capacity": 2,
            "lm_loss": torch.tensor(2.0),
            "balance_loss": torch.tensor(1.25),
            "total_loss": torch.tensor(2.125),
        }

        metrics = aggregate_moe_metrics([record])

        self.assertEqual(metrics["expert_counts"], [3, 1])
        self.assertEqual(metrics["processed_token_count"], 3)
        self.assertEqual(metrics["overflow_token_count"], 1)
        self.assertEqual(metrics["drop_rate"], 0.25)
        self.assertEqual(metrics["max_to_mean_load"], 1.5)
        self.assertEqual(metrics["load_coefficient_of_variation"], 0.5)
        self.assertGreater(metrics["normalized_load_entropy"], 0.0)
        self.assertLess(metrics["normalized_load_entropy"], 1.0)

    def test_gpt_integration_preserves_public_interface(self) -> None:
        torch.manual_seed(7)
        config = GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            moe_num_experts=2,
            moe_layer_index=1,
        )
        model = GPT(config)
        tokens = torch.randint(0, config.vocab_size, (2, config.block_size))

        logits, loss = model(tokens, tokens)
        loss.backward()

        self.assertEqual(logits.shape, (2, config.block_size, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsInstance(model.transformer.h[1].mlp, Top1MoE)
        router_gradient = model.transformer.h[1].mlp.router.proj.weight.grad
        self.assertIsNotNone(router_gradient)
        self.assertTrue(torch.isfinite(router_gradient).all())

    def test_gpt_total_loss_includes_weighted_balance_loss(self) -> None:
        config = GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            moe_num_experts=2,
            moe_layer_index=1,
            moe_balance_loss_weight=0.25,
        )
        model = GPT(config)
        tokens = torch.randint(0, config.vocab_size, (2, config.block_size))

        _, total_loss = model(tokens, tokens)
        metrics = model.get_last_moe_metrics()

        expected = metrics["lm_loss"] + 0.25 * metrics["balance_loss"]
        torch.testing.assert_close(total_loss, expected)
        self.assertTrue(torch.isfinite(metrics["balance_loss"]))

    def test_drop_requires_positive_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive capacity"):
            Top1MoE(
                8,
                2,
                bias=False,
                dropout=0.0,
                capacity_factor=0.0,
                drop_tokens=True,
            )

    def test_dense_and_moe_shared_initialization_is_identical(self) -> None:
        common = dict(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            moe_layer_index=1,
        )
        torch.manual_seed(11)
        dense = GPT(GPTConfig(**common, moe_num_experts=0))
        torch.manual_seed(11)
        moe = GPT(GPTConfig(**common, moe_num_experts=2))

        dense_state = dense.state_dict()
        moe_state = moe.state_dict()
        for name, dense_value in dense_state.items():
            if name.startswith("transformer.h.1.mlp."):
                expert_name = name.replace(
                    "transformer.h.1.mlp.",
                    "transformer.h.1.mlp.experts.0.",
                )
                torch.testing.assert_close(moe_state[expert_name], dense_value)
            else:
                torch.testing.assert_close(moe_state[name], dense_value)

    def test_moe_state_dict_round_trip_is_strict(self) -> None:
        config = GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            moe_num_experts=2,
            moe_layer_index=1,
        )
        torch.manual_seed(19)
        source = GPT(config)
        torch.manual_seed(23)
        restored = GPT(config)

        incompatible = restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for name, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    def test_active_parameter_count_excludes_unselected_experts(self) -> None:
        common = dict(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            moe_layer_index=1,
        )
        dense = GPT(GPTConfig(**common, moe_num_experts=0))
        moe = GPT(GPTConfig(**common, moe_num_experts=4))
        one_expert = sum(
            parameter.numel()
            for parameter in moe.transformer.h[1].mlp.experts[0].parameters()
        )

        self.assertEqual(dense.get_num_active_params(), dense.get_num_params())
        self.assertEqual(
            moe.get_num_active_params(),
            moe.get_num_params() - 3 * one_expert,
        )


if __name__ == "__main__":
    unittest.main()
