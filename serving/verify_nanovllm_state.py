"""Exercise pinned upstream scheduler/block lifecycles without model weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys
import unittest

from serving.backends.nanovllm import (
    DEFAULT_CONFIG, ROOT, base_result, digest, load_config, write_result,
)


def make_suite(upstream_path: Path):
    # Skip upstream's package __init__ (it eagerly imports the GPU engine).
    # The tested modules are loaded verbatim from the pinned source checkout.
    package = ModuleType("nanovllm")
    package.__path__ = [str(upstream_path / "nanovllm")]
    sys.modules["nanovllm"] = package
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence, SequenceStatus
    from nanovllm.sampling_params import SamplingParams

    class LifecycleTests(unittest.TestCase):
        def setUp(self):
            Sequence.block_size = 256

        def sequence(self, tokens, max_tokens=4):
            return Sequence(tokens, SamplingParams(max_tokens=max_tokens))

        def allocate(self, manager, seq):
            hit = manager.can_allocate(seq)
            self.assertGreaterEqual(hit, 0)
            manager.allocate(seq, hit)

        def publish(self, manager, seq):
            seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens
            manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

        def assert_partition(self, manager):
            free, used = set(manager.free_block_ids), manager.used_block_ids
            self.assertFalse(free & used)
            self.assertEqual(free | used, set(range(len(manager.blocks))))
            self.assertEqual(len(free), len(manager.free_block_ids))
            for block in manager.blocks:
                self.assertEqual(block.ref_count > 0, block.block_id in used)

        def scheduler(self, blocks=8, budget=1024):
            return Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=budget,
                eos=99, kvcache_block_size=256, num_kvcache_blocks=blocks))

        def test_allocate_release_conserves_blocks(self):
            manager = BlockManager(4, 256)
            seq = self.sequence([1] * 300)
            self.allocate(manager, seq)
            self.assertEqual(len(seq.block_table), 2)
            self.assert_partition(manager)
            manager.deallocate(seq)
            self.assertEqual(seq.block_table, [])
            self.assertEqual(len(manager.free_block_ids), 4)
            self.assert_partition(manager)

        def test_live_prefix_increments_refcount(self):
            manager = BlockManager(4, 256)
            a, b = self.sequence([1] * 300), self.sequence([1] * 256 + [2] * 44)
            self.allocate(manager, a)
            self.publish(manager, a)
            self.assertEqual(manager.can_allocate(b), 1)
            self.allocate(manager, b)
            shared = a.block_table[0]
            self.assertEqual(shared, b.block_table[0])
            self.assertEqual(manager.blocks[shared].ref_count, 2)
            manager.deallocate(a)
            self.assertEqual(manager.blocks[shared].ref_count, 1)
            manager.deallocate(b)
            self.assert_partition(manager)

        def test_released_prefix_can_be_reactivated(self):
            manager = BlockManager(4, 256)
            a = self.sequence([1] * 300)
            self.allocate(manager, a)
            self.publish(manager, a)
            shared = a.block_table[0]
            manager.deallocate(a)
            b = self.sequence([1] * 256 + [2] * 44)
            self.allocate(manager, b)
            self.assertEqual(b.num_cached_tokens, 256)
            self.assertEqual(b.block_table[0], shared)
            self.assert_partition(manager)

        def test_eviction_removes_stale_hash(self):
            manager = BlockManager(2, 256)
            a = self.sequence([1] * 300)
            self.allocate(manager, a)
            self.publish(manager, a)
            old_hash = manager.blocks[a.block_table[0]].hash
            manager.deallocate(a)
            b = self.sequence([2] * 300)
            self.allocate(manager, b)
            self.assertNotIn(old_hash, manager.hash_to_block_id)
            self.assertEqual(manager.can_allocate(self.sequence([1] * 300)), -1)
            manager.deallocate(b)
            self.assertEqual(manager.can_allocate(self.sequence([1] * 300)), 0)
            self.assert_partition(manager)

        def test_hash_chain_distinguishes_different_prefixes(self):
            suffix = [7] * 256
            g1 = BlockManager.compute_hash(suffix, BlockManager.compute_hash([1] * 256))
            g2 = BlockManager.compute_hash(suffix, BlockManager.compute_hash([2] * 256))
            self.assertNotEqual(g1, g2)

        def test_partial_tail_is_never_shared(self):
            manager = BlockManager(4, 256)
            a = self.sequence([1] * 100)
            self.allocate(manager, a)
            self.publish(manager, a)
            self.assertEqual(manager.can_allocate(self.sequence([1] * 100)), 0)

        def test_exhaustion_is_detected_without_mutation(self):
            manager = BlockManager(1, 256)
            seq = self.sequence([1] * 257)
            self.assertEqual(manager.can_allocate(seq), -1)
            self.assertFalse(seq.block_table)
            self.assert_partition(manager)

        def test_append_boundary_requires_a_free_block(self):
            manager = BlockManager(1, 256)
            seq = self.sequence([1] * 256)
            self.allocate(manager, seq)
            seq.append_token(2)
            self.assertFalse(manager.can_append(seq))
            manager.deallocate(seq)
            self.assertTrue(manager.can_append(seq))

        def test_scheduler_preempts_when_decode_needs_a_block(self):
            scheduler = self.scheduler(blocks=3)
            a, b = self.sequence([1] * 256), self.sequence([2] * 256)
            scheduler.add(a)
            scheduler.add(b)
            seqs, prefill = scheduler.schedule()
            scheduler.postprocess(seqs, [3, 4], prefill)
            seqs, prefill = scheduler.schedule()
            self.assertFalse(prefill)
            self.assertEqual(seqs, [a])
            self.assertEqual(list(scheduler.waiting), [b])
            self.assertEqual(b.status, SequenceStatus.WAITING)
            self.assertEqual(b.num_cached_tokens, 0)
            self.assertFalse(b.block_table)
            self.assert_partition(scheduler.block_manager)

        def test_chunked_prefill_emits_only_after_final_chunk(self):
            scheduler = self.scheduler(budget=128)
            seq = self.sequence([1] * 300, max_tokens=2)
            scheduler.add(seq)
            for cached in (128, 256, 300):
                seqs, prefill = scheduler.schedule()
                self.assertTrue(prefill)
                scheduler.postprocess(seqs, [2], prefill)
                self.assertEqual(seq.num_cached_tokens, cached)
                self.assertEqual(seq.num_completion_tokens, int(cached == 300))
            seqs, prefill = scheduler.schedule()
            scheduler.postprocess(seqs, [3], prefill)
            self.assertTrue(scheduler.is_finished())
            self.assertFalse(scheduler.block_manager.used_block_ids)

        def test_eos_releases_request_resources(self):
            scheduler = self.scheduler()
            seq = self.sequence([1] * 20)
            scheduler.add(seq)
            seqs, prefill = scheduler.schedule()
            scheduler.postprocess(seqs, [99], prefill)
            self.assertTrue(seq.is_finished)
            self.assertTrue(scheduler.is_finished())
            self.assert_partition(scheduler.block_manager)
            self.assertFalse(scheduler.block_manager.used_block_ids)

    return unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleTests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=ROOT / "results/h_nanovllm/h1_state_tests.json")
    args = parser.parse_args()
    config = load_config(args.config)
    report = base_result(config, "cpu_state_tests")
    suite = make_suite(ROOT / config["upstream"]["path"])
    report["test_names"] = [test.id().split(".")[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report.update(status="pass" if result.wasSuccessful() else "fail",
        entrypoint_sha256=digest(Path(__file__)), tests_run=result.testsRun,
        failures=[(test.id(), trace) for test, trace in result.failures + result.errors])
    write_result(report, args.result)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
