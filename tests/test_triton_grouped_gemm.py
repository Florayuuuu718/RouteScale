from __future__ import annotations

import unittest

import torch

from moe import Top1MoE
from triton_grouped_gemm import grouped_gemm, triton_is_available


@unittest.skipUnless(
    triton_is_available(),
    "Triton grouped GEMM requires CUDA",
)
class TritonGroupedGemmTests(unittest.TestCase):
    def test_forward_and_backward_match_torch_bmm(self) -> None:
        torch.manual_seed(31)
        left = torch.randn(
            4,
            31,
            32,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        right = torch.randn(
            4,
            32,
            48,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        reference_left = left.detach().clone().requires_grad_(True)
        reference_right = right.detach().clone().requires_grad_(True)

        actual = grouped_gemm(left, right)
        expected = torch.bmm(reference_left, reference_right)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(
            left.grad, reference_left.grad, rtol=2e-2, atol=2e-1
        )
        torch.testing.assert_close(
            right.grad, reference_right.grad, rtol=2e-2, atol=2e-1
        )

    def test_moe_matches_padded_torch_under_bfloat16_autocast(self) -> None:
        common = dict(
            n_embd=32,
            num_experts=4,
            bias=False,
            dropout=0.0,
            capacity_factor=1.25,
            drop_tokens=True,
        )
        torch.manual_seed(37)
        reference = Top1MoE(
            **common, dispatch_backend="padded_torch"
        ).cuda()
        actual = Top1MoE(
            **common, dispatch_backend="padded_triton"
        ).cuda()
        actual.load_state_dict(reference.state_dict())
        reference_input = torch.randn(
            2, 16, 32, device="cuda", requires_grad=True
        )
        actual_input = reference_input.detach().clone().requires_grad_(True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected, expected_routing = reference(
                reference_input, return_routing=True
            )
            output, routing = actual(actual_input, return_routing=True)

        torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            routing.expert_counts, expected_routing.expert_counts
        )
        torch.testing.assert_close(
            routing.dropped_mask, expected_routing.dropped_mask
        )
        gradient = torch.randn_like(output)
        output.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(
            actual_input.grad,
            reference_input.grad,
            rtol=5e-2,
            atol=5e-2,
        )
