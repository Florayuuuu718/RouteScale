from __future__ import annotations

import unittest

from scripts.summarize_d4_moe_profiler import (
    aggregate_named,
    parse_time_total,
    percent_change,
)


class D4ProfilerSummaryTests(unittest.TestCase):
    def test_parse_time_total_converts_units(self) -> None:
        self.assertEqual(parse_time_total("Self CUDA time total: 2.5s", "Self CUDA time total"), 2500.0)
        self.assertEqual(parse_time_total("Self CPU time total: 25ms", "Self CPU time total"), 25.0)

    def test_aggregate_named_combines_duplicate_rows(self) -> None:
        rows = [
            {
                "name": "event",
                "count": 2,
                "self_cpu_time_us": 1000,
                "self_device_time_us": 2000,
                "device_time_total_us": 3000,
            },
            {
                "name": "event",
                "count": 3,
                "self_cpu_time_us": 4000,
                "self_device_time_us": 5000,
                "device_time_total_us": 6000,
            },
        ]

        result = aggregate_named(rows, "event")

        self.assertEqual(result["count"], 5)
        self.assertEqual(result["self_cpu_time_ms"], 5.0)
        self.assertEqual(result["self_device_time_ms"], 7.0)
        self.assertEqual(result["device_time_total_ms"], 9.0)

    def test_percent_change_preserves_direction(self) -> None:
        self.assertEqual(percent_change(100, 75), -25.0)
        self.assertEqual(percent_change(100, 125), 25.0)


if __name__ == "__main__":
    unittest.main()
