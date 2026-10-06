from __future__ import annotations

import unittest

from distributed_benchmark import aggregate_rank_records


class AggregateRankRecordsTests(unittest.TestCase):
    def test_uses_slowest_rank_for_each_update(self) -> None:
        records = [
            {
                "rank": 1,
                "step_times_ms": [8.0, 12.0, 9.0],
                "mean_micro_batch_losses": [4.0, 3.0, 2.0],
                "all_losses_finite": True,
                "warmup_window_ids_sha256": "rank1-warmup",
                "expected_warmup_window_ids_sha256": "rank1-warmup",
                "measurement_window_ids_sha256": "rank1-measured",
                "expected_measurement_window_ids_sha256": "rank1-measured",
            },
            {
                "rank": 0,
                "step_times_ms": [10.0, 7.0, 11.0],
                "mean_micro_batch_losses": [6.0, 5.0, 4.0],
                "all_losses_finite": True,
                "warmup_window_ids_sha256": "rank0-warmup",
                "expected_warmup_window_ids_sha256": "rank0-warmup",
                "measurement_window_ids_sha256": "rank0-measured",
                "expected_measurement_window_ids_sha256": "rank0-measured",
            },
        ]

        result = aggregate_rank_records(
            records,
            measured_steps=3,
            global_tokens_per_update=1000,
        )

        self.assertEqual([item["rank"] for item in result["per_rank"]], [0, 1])
        self.assertEqual(result["slowest_rank_step_times_ms"], [10.0, 12.0, 11.0])
        self.assertEqual(result["global_mean_losses"], [5.0, 4.0, 3.0])
        self.assertEqual(result["median_slowest_rank_step_ms"], 11.0)
        self.assertAlmostEqual(
            result["global_tokens_per_second_from_median"],
            1000 / 0.011,
        )
        self.assertTrue(result["all_losses_finite"])
        self.assertTrue(result["all_window_hashes_match_plan"])

    def test_rejects_incomplete_rank_record(self) -> None:
        record = {
            "rank": 0,
            "step_times_ms": [1.0],
            "mean_micro_batch_losses": [1.0, 2.0],
            "all_losses_finite": True,
            "warmup_window_ids_sha256": "a",
            "expected_warmup_window_ids_sha256": "a",
            "measurement_window_ids_sha256": "b",
            "expected_measurement_window_ids_sha256": "b",
        }
        with self.assertRaisesRegex(ValueError, "step count"):
            aggregate_rank_records(
                [record], measured_steps=2, global_tokens_per_update=1000
            )


if __name__ == "__main__":
    unittest.main()
