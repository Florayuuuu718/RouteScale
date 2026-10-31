from __future__ import annotations

import unittest

import torch

from model import GPT, GPTConfig
from serving.native import greedy_generate_cached, greedy_generate_naive


class NativeKVCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_embd=16,
                dropout=0.0,
                bias=False,
            )
        ).eval()

    def test_prefill_and_decode_match_full_recompute_for_batch(self) -> None:
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        cache = self.model.allocate_kv_cache(batch_size=2)
        cached_logits = self.model.forward_with_kv_cache(prompt, cache)
        full_logits, _ = self.model(prompt)
        torch.testing.assert_close(cached_logits, full_logits, rtol=0.0, atol=0.0)
        self.assertEqual(cache.length, 3)
        self.assertTrue(all(tensor.is_contiguous() for tensor in cache.keys))
        self.assertTrue(all(tensor.is_contiguous() for tensor in cache.values))

        next_token = torch.argmax(full_logits[:, -1, :], dim=-1, keepdim=True)
        cached_next = self.model.forward_with_kv_cache(next_token, cache)
        full_next, _ = self.model(torch.cat((prompt, next_token), dim=1))
        torch.testing.assert_close(cached_next, full_next, rtol=1e-5, atol=1e-6)
        self.assertEqual(cache.length, 4)

    def test_cache_capacity_error_is_explicit(self) -> None:
        cache = self.model.allocate_kv_cache(batch_size=1, capacity=3)
        self.model.forward_with_kv_cache(torch.tensor([[1, 2, 3]]), cache)
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            self.model.forward_with_kv_cache(torch.tensor([[4]]), cache)

    def test_generation_rebuilds_at_block_boundary(self) -> None:
        prompt = torch.arange(8, dtype=torch.long).unsqueeze(0)
        naive = greedy_generate_naive(self.model, prompt, max_new_tokens=3)
        cached = greedy_generate_cached(self.model, prompt, max_new_tokens=3)
        torch.testing.assert_close(cached.token_ids, naive.token_ids)
        self.assertEqual(cached.cache_rebuilds, 2)
        self.assertGreater(cached.cache_allocated_bytes, 0)
        for cached_logits, naive_logits in zip(cached.step_logits, naive.step_logits):
            torch.testing.assert_close(cached_logits, naive_logits, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

