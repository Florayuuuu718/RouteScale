"""Check the actual FlashAttention kernels before launching the pinned engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from serving.backends.nanovllm import (
    DEFAULT_CONFIG, ROOT, base_result, digest, load_config, validate_environment, write_result,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "results/h_nanovllm/h1_attention.json")
    args = parser.parse_args()
    config = load_config(DEFAULT_CONFIG)
    validate_environment(config)
    report = base_result(config, "gpu_kernel_correctness")
    report["entrypoint_sha256"] = digest(Path(__file__))
    try:
        import torch
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        from nanovllm.layers.attention import store_kvcache

        torch.manual_seed(config["seed"])
        report["gpu"] = torch.cuda.get_device_name()
        report["compute_capability"] = list(torch.cuda.get_device_capability())
        report["checks"] = []
        # Qwen3 uses GQA and head_dim=128. Test both supported inference dtypes.
        for dtype in (torch.bfloat16, torch.float16):
            k = torch.randn(64, 8, 128, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            q = torch.randn(64, 16, 128, device="cuda", dtype=dtype)
            offsets = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
            out = flash_attn_varlen_func(q, k, v, offsets, offsets, 64, 64, causal=True)
            def reference(query, causal):
                return torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(0, 1).unsqueeze(0).float(),
                    k.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0).float(),
                    v.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0).float(),
                    is_causal=causal,
                ).squeeze(0).transpose(0, 1)
            ref = reference(q, True)
            torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.03)
            prefill_error = (out.float() - ref).abs().max().item()
            # Nonzero page ID verifies physical page indirection and the Triton KV writer.
            kc = torch.zeros(2, 256, 8, 128, device="cuda", dtype=dtype)
            vc = torch.zeros_like(kc)
            slots = torch.arange(256, 320, device="cuda", dtype=torch.int32)
            store_kvcache(k, v, kc, vc, slots)
            torch.testing.assert_close(kc[1, :64], k, atol=0, rtol=0)
            torch.testing.assert_close(vc[1, :64], v, atol=0, rtol=0)
            tables = torch.tensor([[1]], device="cuda", dtype=torch.int32)
            prefix_out = flash_attn_varlen_func(
                q[-16:], kc, vc,
                torch.tensor([0, 16], device="cuda", dtype=torch.int32), offsets,
                16, 64, causal=True, block_table=tables,
            )
            prefix_ref = reference(q, True)[-16:]
            torch.testing.assert_close(prefix_out.float(), prefix_ref, atol=0.03, rtol=0.03)
            out = flash_attn_with_kvcache(q[-1:].unsqueeze(0), kc, vc,
                cache_seqlens=torch.tensor([64], device="cuda", dtype=torch.int32),
                block_table=tables, causal=True)
            ref = reference(q[-1:], False)
            torch.testing.assert_close(out[0].float(), ref, atol=0.03, rtol=0.03)
            report["checks"].append({"dtype": str(dtype), "prefill_max_abs_error": prefill_error,
                "paged_prefix_prefill_max_abs_error": (prefix_out.float() - prefix_ref).abs().max().item(),
                "paged_decode_max_abs_error": (out[0].float() - ref).abs().max().item(),
                "kv_store_exact": True, "atol": 0.03, "rtol": 0.03})
        torch.cuda.synchronize()
        report["status"] = "pass"
    except Exception:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        write_result(report, args.result)
    print(f"H1 attention kernels: pass -> {args.result}")


if __name__ == "__main__":
    main()
