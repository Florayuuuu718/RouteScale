from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model import GPT, GPTConfig
from serving.native.benchmark import (
    build_prompt_batch,
    build_decode_batch,
    generate_timed,
    summarize_measurements,
)


class NativeServingBenchmarkTests(unittest.TestCase):
    def test_prompt_workloads_are_exact_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "val.bin"
            np.arange(2048, dtype=np.uint16).tofile(path)
            first, first_metadata = build_prompt_batch(
                path,
                batch_size=4,
                prompt_length=32,
                prefix_mode="high",
                seed=9,
            )
            second, second_metadata = build_prompt_batch(
                path,
                batch_size=4,
                prompt_length=32,
                prefix_mode="high",
                seed=9,
            )
            torch.testing.assert_close(first, second)
            self.assertEqual(first.shape, (4, 32))
            self.assertEqual(first_metadata["shared_prefix_tokens"], 24)
            self.assertEqual(
                first_metadata["prompt_token_ids_sha256"],
                second_metadata["prompt_token_ids_sha256"],
            )
            for row in first[1:]:
                torch.testing.assert_close(row[:24], first[0, :24])

    def test_cpu_timing_harness_preserves_naive_and_cache_outputs(self) -> None:
        torch.manual_seed(3)
        model = GPT(
            GPTConfig(
                block_size=16,
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_embd=16,
                dropout=0.0,
                bias=False,
            )
        ).eval()
        prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        forced = torch.tensor([[9, 10, 11], [12, 13, 14]])
        naive = generate_timed(
            model,
            prompt,
            output_length=3,
            backend="naive",
            forced_token_ids=forced,
        )
        cached = generate_timed(
            model,
            prompt,
            output_length=3,
            backend="kv_cache",
            forced_token_ids=forced,
        )
        torch.testing.assert_close(naive.token_ids, cached.token_ids)
        torch.testing.assert_close(naive.token_ids[:, -3:], forced)
        summary = summarize_measurements(
            [cached, cached], batch_size=2, prompt_length=4, output_length=3
        )
        self.assertEqual(summary["successful_requests"], 4)
        self.assertEqual(summary["output_tokens"], 12)
        self.assertTrue(summary["outputs_deterministic"])
        self.assertGreater(summary["cache_allocated_bytes"], 0)
        self.assertTrue(summary["predictions_deterministic"])

    def test_fixed_decode_tokens_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "val.bin"
            np.arange(2048, dtype=np.uint16).tofile(path)
            first, first_metadata = build_decode_batch(
                path, batch_size=3, output_length=16, seed=11
            )
            second, second_metadata = build_decode_batch(
                path, batch_size=3, output_length=16, seed=11
            )
            torch.testing.assert_close(first, second)
            self.assertEqual(first.shape, (3, 16))
            self.assertEqual(
                first_metadata["decode_token_ids_sha256"],
                second_metadata["decode_token_ids_sha256"],
            )


if __name__ == "__main__":
    unittest.main()

