from __future__ import annotations

import unittest

from scripts.summarize_d2_moe import distribution_metrics, summarize_run


class D2SummaryTests(unittest.TestCase):
    def test_distribution_metrics(self) -> None:
        metrics = distribution_metrics([3, 1])

        self.assertEqual(metrics["max_to_mean_load"], 1.5)
        self.assertEqual(metrics["load_coefficient_of_variation"], 0.5)
        self.assertGreater(metrics["normalized_load_entropy"], 0.0)
        self.assertLess(metrics["normalized_load_entropy"], 1.0)

    def test_summarize_run_aggregates_updates(self) -> None:
        update = {
            "expert_counts": [3, 1],
            "processed_counts": [2, 1],
            "token_count": 4,
            "overflow_token_count": 1,
            "dropped_token_count": 1,
            "max_to_mean_load": 1.5,
            "load_coefficient_of_variation": 0.5,
            "mean_lm_loss": 2.0,
            "mean_balance_loss": 1.25,
            "mean_total_loss": 2.125,
        }

        summary = summarize_run({"updates": [update, update]})

        self.assertEqual(summary["total_tokens"], 8)
        self.assertEqual(summary["aggregate_expert_counts"], [6, 2])
        self.assertEqual(summary["aggregate_processed_counts"], [4, 2])
        self.assertEqual(summary["overflow_rate"], 0.25)
        self.assertEqual(summary["drop_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
