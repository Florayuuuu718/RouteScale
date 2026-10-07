from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_c_trace_overlap import (
    analyze_trace,
    intersection_duration,
    merge_intervals,
)


class TraceOverlapTests(unittest.TestCase):
    def test_interval_union_and_intersection(self) -> None:
        self.assertEqual(
            merge_intervals([(0, 4), (3, 7), (10, 12)]),
            [(0, 7), (10, 12)],
        )
        self.assertEqual(
            intersection_duration([(0, 5), (8, 10)], [(3, 9)]),
            3,
        )

    def test_analyzes_nccl_compute_and_backward_overlap(self) -> None:
        events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclDevKernel_AllReduce",
                    "ts": 2,
                    "dur": 6,
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "gemm_kernel",
                    "ts": 5,
                    "dur": 6,
                },
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "backward",
                    "ts": 0,
                    "dur": 7,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            result = analyze_trace(path)
        self.assertEqual(result["nccl_kernel_union_time_us"], 6)
        self.assertEqual(result["nccl_compute_overlap_us"], 3)
        self.assertEqual(result["nccl_compute_overlap_ratio"], 0.5)
        self.assertEqual(result["nccl_within_backward_scope_us"], 5)


if __name__ == "__main__":
    unittest.main()
