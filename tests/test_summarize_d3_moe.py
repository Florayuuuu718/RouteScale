from __future__ import annotations

import unittest

from scripts.summarize_d3_moe import distribution_metrics, summarize_benchmark


class D3SummaryTests(unittest.TestCase):
    def test_distribution_metrics(self) -> None:
        metrics = distribution_metrics([3, 1])

        self.assertEqual(metrics["max_to_mean_load"], 1.5)
        self.assertEqual(metrics["load_coefficient_of_variation"], 0.5)
        self.assertGreater(metrics["normalized_load_entropy"], 0.0)
        self.assertLess(metrics["normalized_load_entropy"], 1.0)

    def test_benchmark_uses_median_of_independent_run_medians(self) -> None:
        runs = []
        for step_ms in (2.0, 1.0, 3.0):
            runs.append({
                "configuration": {
                    "tokens_per_update": 100,
                    "model_parameter_count": 20,
                    "active_parameter_count": 10,
                    "non_position_embedding_parameter_count": 19,
                    "active_non_position_embedding_parameter_count": 9,
                },
                "measurement": {
                    "median_step_ms": step_ms,
                    "tokens_per_second_from_median": 100 / (step_ms / 1000),
                    "peak_allocated_mib": step_ms,
                    "peak_reserved_mib": step_ms + 1,
                },
            })

        summary = summarize_benchmark(runs)

        self.assertEqual(summary["median_of_run_medians_ms"], 2.0)
        self.assertEqual(summary["tokens_per_second"], 50_000.0)
        self.assertEqual(summary["run_median_range_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()
