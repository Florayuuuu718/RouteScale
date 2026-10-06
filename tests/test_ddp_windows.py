from __future__ import annotations

import unittest

import numpy as np

from ddp_windows import GlobalWindowScheduler


class GlobalWindowSchedulerTests(unittest.TestCase):
    def make_scheduler(
        self,
        world_size: int,
        *,
        window_count: int = 101,
        global_windows: int = 32,
        batch_size: int = 8,
        tail_policy: str = "drop",
    ) -> GlobalWindowScheduler:
        return GlobalWindowScheduler(
            window_count=window_count,
            global_windows_per_update=global_windows,
            world_size=world_size,
            rank=0,
            batch_size=batch_size,
            seed=20260920,
            tail_policy=tail_policy,
        )

    def test_strong_scaling_mapping_has_same_global_order(self) -> None:
        schedulers = {world: self.make_scheduler(world) for world in (1, 2, 4)}
        expected_micro_steps = {1: 4, 2: 2, 4: 1}
        reference = schedulers[1].global_window_ids(0)

        for world_size, scheduler in schedulers.items():
            self.assertEqual(
                scheduler.local_micro_steps, expected_micro_steps[world_size]
            )
            rank_parts = [
                scheduler.rank_window_ids(0, rank).reshape(-1)
                for rank in range(world_size)
            ]
            reconstructed = np.concatenate(rank_parts)
            np.testing.assert_array_equal(reconstructed, reference)
            self.assertEqual(len(set(reconstructed.tolist())), len(reference))

        hashes = {
            scheduler.hash_updates(0, 3) for scheduler in schedulers.values()
        }
        self.assertEqual(len(hashes), 1)

    def test_weak_scaling_keeps_four_micro_steps_per_rank(self) -> None:
        for world_size in (1, 2, 4):
            scheduler = self.make_scheduler(
                world_size,
                window_count=513,
                global_windows=32 * world_size,
                batch_size=8,
            )
            self.assertEqual(scheduler.local_windows_per_update, 32)
            self.assertEqual(scheduler.local_micro_steps, 4)
            global_ids = scheduler.global_window_ids(0)
            self.assertEqual(len(global_ids), 32 * world_size)
            rank_ids = np.concatenate(
                [
                    scheduler.rank_window_ids(0, rank).reshape(-1)
                    for rank in range(world_size)
                ]
            )
            np.testing.assert_array_equal(rank_ids, global_ids)

    def test_same_seed_reconstructs_order_and_different_epoch_changes_it(self) -> None:
        first = self.make_scheduler(2)
        second = self.make_scheduler(2)
        np.testing.assert_array_equal(
            first.epoch_window_ids(0), second.epoch_window_ids(0)
        )
        self.assertFalse(
            np.array_equal(first.epoch_window_ids(0), first.epoch_window_ids(1))
        )

    def test_drop_tail_records_omissions_without_rank_duplicates(self) -> None:
        scheduler = self.make_scheduler(
            4, window_count=70, global_windows=32, batch_size=2, tail_policy="drop"
        )
        scheduled = scheduler.epoch_window_ids()
        dropped = scheduler.dropped_window_ids()
        self.assertEqual(scheduler.updates_per_epoch, 2)
        self.assertEqual(len(scheduled), 64)
        self.assertEqual(len(dropped), 6)
        self.assertEqual(len(set(scheduled.tolist())), 64)
        self.assertEqual(
            set(scheduled.tolist()) | set(dropped.tolist()), set(range(70))
        )
        self.assertFalse(set(scheduled.tolist()) & set(dropped.tolist()))

    def test_pad_tail_covers_every_window_and_records_repeats(self) -> None:
        scheduler = self.make_scheduler(
            2, window_count=35, global_windows=16, batch_size=4, tail_policy="pad"
        )
        scheduled = scheduler.epoch_window_ids()
        padded = scheduler.padded_window_ids()
        self.assertEqual(len(scheduled), 48)
        self.assertEqual(len(padded), 13)
        self.assertEqual(set(scheduled.tolist()), set(range(35)))
        self.assertEqual(len(scheduled) - len(set(scheduled.tolist())), 13)

    def test_invalid_rank_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by world_size"):
            self.make_scheduler(4, global_windows=30)
        with self.assertRaisesRegex(ValueError, "divisible by batch_size"):
            self.make_scheduler(2, global_windows=28, batch_size=8)


if __name__ == "__main__":
    unittest.main()
