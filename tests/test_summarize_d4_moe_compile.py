from __future__ import annotations

import unittest

from scripts.summarize_d4_moe_compile import compare, summarize_runs


def _run(step_ms: float, allocated: float, reserved: float) -> dict:
    return {
        "configuration": {"tokens_per_update": 100},
        "measurement": {
            "median_step_ms": step_ms,
            "peak_allocated_mib": allocated,
            "peak_reserved_mib": reserved,
        },
    }


class D4SummaryTests(unittest.TestCase):
    def test_summarize_runs_uses_median_of_run_medians(self) -> None:
        summary = summarize_runs(
            [_run(3.0, 12.0, 22.0), _run(1.0, 10.0, 20.0), _run(2.0, 11.0, 21.0)]
        )

        self.assertEqual(summary["median_of_run_medians_ms"], 2.0)
        self.assertEqual(summary["tokens_per_second"], 50_000.0)
        self.assertEqual(summary["run_median_range_ms"], 2.0)
        self.assertEqual(summary["median_peak_allocated_mib"], 11.0)

    def test_compare_reports_slowdown_and_memory_delta(self) -> None:
        eager = summarize_runs(
            [_run(2.0, 10.0, 20.0), _run(2.0, 10.0, 20.0), _run(2.0, 10.0, 20.0)]
        )
        compiled = summarize_runs(
            [_run(4.0, 13.0, 25.0), _run(4.0, 13.0, 25.0), _run(4.0, 13.0, 25.0)]
        )

        result = compare(eager, compiled)

        self.assertEqual(result["speedup"], 0.5)
        self.assertEqual(result["step_time_change_percent"], 100.0)
        self.assertEqual(result["throughput_change_percent"], -50.0)
        self.assertEqual(result["peak_allocated_change_mib"], 3.0)
        self.assertEqual(result["peak_reserved_change_mib"], 5.0)


if __name__ == "__main__":
    unittest.main()
