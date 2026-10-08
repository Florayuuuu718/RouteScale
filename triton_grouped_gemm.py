"""Fixed-shape grouped GEMM kernels used by the local Top-1 MoE path.

The forward kernel computes a batch of independent matrix multiplications in
one Triton launch:

    output[g] = left[g] @ right[g]

All groups share M/N/K. The MoE dispatcher pads each expert to a fixed
capacity, which keeps the launch shape stable and removes the Python expert
loop from the hot path. Backward deliberately uses PyTorch batched GEMMs for
the first implementation so training correctness can be established before
optimizing gradient kernels.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only installs
    triton = None
    tl = None


def triton_is_available() -> bool:
    """Return whether Triton is importable and a CUDA device is available."""
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_stages=3,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_stages=3,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                num_stages=3,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                num_stages=4,
                num_warps=8,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fixed_grouped_gemm_kernel(
        left_ptr,
        right_ptr,
        output_ptr,
        group_count: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_lg: tl.constexpr,
        stride_lm: tl.constexpr,
        stride_lk: tl.constexpr,
        stride_rg: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_rn: tl.constexpr,
        stride_og: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        programs_per_group = num_pid_m * num_pid_n
        group_index = pid // programs_per_group
        local_pid = pid % programs_per_group

        programs_in_m_group = GROUP_M * num_pid_n
        m_group_index = local_pid // programs_in_m_group
        first_pid_m = m_group_index * GROUP_M
        actual_group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (local_pid % actual_group_m)
        pid_n = (local_pid % programs_in_m_group) // actual_group_m

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)

        left = (
            left_ptr
            + group_index * stride_lg
            + offsets_m[:, None] * stride_lm
            + offsets_k[None, :] * stride_lk
        )
        right = (
            right_ptr
            + group_index * stride_rg
            + offsets_k[:, None] * stride_rk
            + offsets_n[None, :] * stride_rn
        )

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            left_values = tl.load(
                left,
                mask=(offsets_m[:, None] < M)
                & (k_start + offsets_k[None, :] < K),
                other=0.0,
            )
            right_values = tl.load(
                right,
                mask=(k_start + offsets_k[:, None] < K)
                & (offsets_n[None, :] < N),
                other=0.0,
            )
            accumulator = tl.dot(left_values, right_values, accumulator)
            left += BLOCK_K * stride_lk
            right += BLOCK_K * stride_rk

        output = (
            output_ptr
            + group_index * stride_og
            + offsets_m[:, None] * stride_om
            + offsets_n[None, :] * stride_on
        )
        tl.store(
            output,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


def _launch_grouped_gemm(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    group_count, rows, reduction = left.shape
    _, right_reduction, columns = right.shape
    if reduction != right_reduction:
        raise ValueError("grouped GEMM reduction dimensions do not match")
    output = torch.empty(
        (group_count, rows, columns),
        device=left.device,
        dtype=left.dtype,
    )
    grid = lambda meta: (
        group_count
        * triton.cdiv(rows, meta["BLOCK_M"])
        * triton.cdiv(columns, meta["BLOCK_N"]),
    )
    _fixed_grouped_gemm_kernel[grid](
        left,
        right,
        output,
        group_count,
        rows,
        columns,
        reduction,
        left.stride(0),
        left.stride(1),
        left.stride(2),
        right.stride(0),
        right.stride(1),
        right.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
    )
    return output


class _GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        contiguous_left = left.contiguous()
        contiguous_right = right.contiguous()
        ctx.save_for_backward(contiguous_left, contiguous_right)
        return _launch_grouped_gemm(contiguous_left, contiguous_right)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        left, right = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_left = torch.bmm(grad_output, right.transpose(1, 2))
        grad_right = torch.bmm(left.transpose(1, 2), grad_output)
        return grad_left, grad_right


def grouped_gemm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply equally shaped groups with one Triton forward launch.

    Args:
        left: CUDA tensor shaped ``[groups, M, K]``.
        right: CUDA tensor shaped ``[groups, K, N]``.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not left.is_cuda or not right.is_cuda:
        raise ValueError("Triton grouped GEMM requires CUDA tensors")
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("grouped GEMM inputs must both be rank three")
    if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[1]:
        raise ValueError("grouped GEMM input shapes are incompatible")
    if left.dtype != right.dtype:
        raise ValueError("grouped GEMM inputs must have the same dtype")
    if left.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("grouped GEMM supports float16 and bfloat16")
    return _GroupedGemm.apply(left, right)
