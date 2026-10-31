import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from serving.h2_kv_capacity import (
    PageAllocator, SpanAllocator, allocator_report, assert_conservation, static_window_manager,
)

ROOT = Path(__file__).resolve().parents[1]


def upstream_types():
    # Real pinned modules, with only eager GPU package initialization bypassed.
    # Restore sys.modules so this test cannot affect other serving tests.
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "nanovllm" or name.startswith("nanovllm."):
                del sys.modules[name]
        package = ModuleType("nanovllm")
        package.__path__ = [str(ROOT / "runs/vendor/nano-vllm/nanovllm")]
        sys.modules["nanovllm"] = package
        from nanovllm.engine.block_manager import BlockManager
        from nanovllm.engine.sequence import Sequence
        return BlockManager, Sequence


class KVCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.sequence = upstream_types()
        cls.cfg = json.loads((ROOT / "config/serving/h2_kv_capacity.json").read_text())

    def pages(self, static=False, blocks=32):
        return PageAllocator(self.base, self.sequence, blocks, 256, 1024, static=static)

    def test_homogeneous_capacity_has_both_positive_and_negative_controls(self):
        report = allocator_report(self.cfg, self.base, self.sequence)
        counts = {(r["policy"], r["live_kv_tokens_per_request"]): r["at_capacity"]["requests"] for r in report["capacity"]}
        self.assertEqual(len(counts), 18)
        self.assertEqual([counts["static_window", n] for n in self.cfg["capacity_lengths"]], [8]*6)
        self.assertEqual([counts["paged", n] for n in self.cfg["capacity_lengths"]], [32, 32, 16, 16, 10, 8])
        self.assertEqual([counts["contiguous_exact", n] for n in self.cfg["capacity_lengths"]], [64, 32, 31, 16, 10, 8])

    def test_every_replay_releases_the_entire_budget(self):
        report = allocator_report(self.cfg, self.base, self.sequence)
        for row in [*report["capacity"], *report["fragmentation"].values(), *report["growth"].values()]:
            state = row["after_release"]
            self.assertEqual(state["free_tokens"], 8192)
            self.assertEqual(state["reserved_tokens"], 0)
            self.assertEqual(state["requests"], 0)

    def test_external_fragmentation_is_not_total_capacity_exhaustion(self):
        result = allocator_report(self.cfg, self.base, self.sequence)["fragmentation"]
        contiguous, paged = result["contiguous_exact"], result["paged"]
        self.assertEqual(contiguous["before"]["free_tokens"], 4096)
        self.assertEqual(contiguous["before"]["largest_free_span_tokens"], 512)
        self.assertFalse(contiguous["accepted"])
        self.assertTrue(paged["accepted"])
        self.assertEqual(paged["after"]["free_tokens"], 4096-768)
        self.assertEqual(len(set(paged["page_tables"]["100"])), 3)

    def test_growth_relocation_is_counted_as_copy_requirement_not_gpu_timing(self):
        result = allocator_report(self.cfg, self.base, self.sequence)["growth"]
        self.assertTrue(all(r["accepted"] for r in result.values()))
        self.assertEqual(result["contiguous_exact"]["after"]["copied_tokens"], 256)
        self.assertEqual(result["paged"]["after"]["copied_tokens"], 0)

    def test_static_exposes_only_live_pages_for_upstream_decode_last_page(self):
        allocator = self.pages(static=True, blocks=8)
        self.assertTrue(allocator.allocate(0, 128))
        seq = allocator.live[0]
        self.assertEqual(seq.block_table, [0])
        self.assertEqual(allocator.manager.reservations[seq.seq_id], [0, 1, 2, 3])
        self.assertEqual(len(allocator.manager.used_block_ids), 4)
        self.assertTrue(allocator.grow(0, 257))
        self.assertEqual(seq.block_table, [0, 1])
        self.assertEqual(len(allocator.manager.used_block_ids), 4)
        assert_conservation(allocator)
        allocator.release(0)
        self.assertEqual(len(allocator.manager.free_block_ids), 8)
        self.assertFalse(allocator.manager.reservations)

    def test_static_future_reservation_and_tail_are_separate(self):
        allocator = self.pages(static=True)
        allocator.allocate(0, 128)
        state = assert_conservation(allocator)
        self.assertEqual(state["live_tokens"], 128)
        self.assertEqual(state["page_tail_tokens"], 128)
        self.assertEqual(state["future_reserved_tokens"], 768)
        self.assertEqual(state["reserved_tokens"], 1024)

    def test_paged_growth_crosses_boundary_without_moving_first_page(self):
        allocator = self.pages()
        allocator.allocate(0, 256)
        first = allocator.live[0].block_table[0]
        allocator.allocate(1, 256)
        allocator.grow(0, 257)
        self.assertEqual(allocator.live[0].block_table, [first, 2])
        assert_conservation(allocator)

    def test_failed_paged_growth_does_not_mutate_request(self):
        allocator = self.pages(blocks=1)
        allocator.allocate(0, 256)
        before = assert_conservation(allocator)
        self.assertFalse(allocator.grow(0, 257))
        self.assertEqual(before, assert_conservation(allocator))
        self.assertEqual(len(allocator.live[0]), 256)

    def test_failed_contiguous_growth_preserves_the_old_span(self):
        allocator = SpanAllocator(512, 1024)
        allocator.allocate(0, 256)
        allocator.allocate(1, 256)
        before = assert_conservation(allocator)
        self.assertFalse(allocator.grow(0, 257))
        self.assertEqual(before, assert_conservation(allocator))
        self.assertEqual(allocator.spans[0], (0, 256))

    def test_contiguous_holes_coalesce_after_release(self):
        allocator = SpanAllocator(1024, 1024)
        for key in range(4):
            allocator.allocate(key, 256)
        allocator.release(1)
        allocator.release(2)
        self.assertEqual(allocator.holes(), [(256, 512)])
        self.assertTrue(allocator.allocate(5, 512))

    def test_invalid_window_length_duplicate_key_and_shrink_fail_closed(self):
        for allocator in (self.pages(), self.pages(static=True), SpanAllocator(8192, 1024)):
            for tokens in (0, 1025):
                with self.assertRaises(ValueError):
                    allocator.allocate(0, tokens)
            allocator.allocate(0, 256)
            with self.assertRaises(ValueError):
                allocator.allocate(0, 256)
            with self.assertRaises(ValueError):
                allocator.grow(0, 255)
        with self.assertRaises(ValueError):
            static_window_manager(self.base, 1023)(32, 256)

    def test_corrupted_refcounts_fail_conservation_check(self):
        allocator = self.pages()
        allocator.allocate(0, 128)
        allocator.manager.blocks[0].ref_count = 2
        with self.assertRaisesRegex(AssertionError, "refcount"):
            assert_conservation(allocator)

    def test_reuse_static_pages_clears_old_hash_metadata(self):
        allocator = self.pages(static=True, blocks=4)
        allocator.allocate(0, 256)
        seq = allocator.live[0]
        seq.num_scheduled_tokens = 256
        allocator.manager.hash_blocks(seq)
        self.assertTrue(allocator.manager.hash_to_block_id)
        allocator.release(0)
        allocator.allocate(1, 128)
        self.assertFalse(allocator.manager.hash_to_block_id)
        self.assertEqual(allocator.live[1].num_cached_tokens, 0)
        assert_conservation(allocator)


if __name__ == "__main__":
    unittest.main()
