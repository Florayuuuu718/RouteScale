from __future__ import annotations

import unittest

from scripts.summarize_e_moe_profiler import count_percent_change
from scripts.summarize_e_triton_moe import compare, summarize_runs


def make_run(step_ms: float, allocated: float = 100.0) -> dict:
    return {
        "measurement": {
            "median_step_ms": step_ms,
            "peak_allocated_mib": allocated,
            "peak_reserved_mib": allocated + 20.0,
        },
        "configuration": {
            "tokens_per_update": 1000,
            "model_parameter_count": 200,
            "active_parameter_count": 150,
            "compile": False,
            "moe_dispatch_backend": "loop",
        },
    }


class EStageSummaryTests(unittest.TestCase):
    def test_summarize_uses_median_of_independent_runs(self) -> None:
        summary = summarize_runs(
            [make_run(12.0), make_run(10.0), make_run(11.0)]
        )
        self.assertEqual(summary["median_of_run_medians_ms"], 11.0)
        self.assertAlmostEqual(summary["tokens_per_second"], 1000 / 0.011)

    def test_compare_preserves_speedup_direction(self) -> None:
        baseline = summarize_runs(
            [make_run(20.0), make_run(20.0), make_run(20.0)]
        )
        candidate = summarize_runs(
            [make_run(10.0), make_run(10.0), make_run(10.0)]
        )
        result = compare(candidate, baseline)
        self.assertEqual(result["speedup"], 2.0)
        self.assertEqual(result["step_time_change_percent"], -50.0)
        self.assertEqual(result["throughput_change_percent"], 100.0)

    def test_new_profiler_operation_has_no_percentage(self) -> None:
        self.assertIsNone(count_percent_change(0, 10))
        self.assertEqual(count_percent_change(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
