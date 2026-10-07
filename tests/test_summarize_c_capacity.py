from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_c_capacity import load_points, summarize_variant


class CapacitySummaryTests(unittest.TestCase):
    def test_discovers_success_and_failure_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success_path = root / "backend" / "deepspeed_zero2" / "4gpu" / "ok.json"
            success_path.parent.mkdir(parents=True)
            success_path.write_text(
                json.dumps(
                    {
                        "status": "benchmark_complete",
                        "variant": "deepspeed_zero2",
                        "run_id": "ok",
                        "configuration": {
                            "world_size": 4,
                            "model_parameter_count": 100,
                            "n_layer": 2,
                        },
                        "measurement": {
                            "median_slowest_rank_step_ms": 1.0,
                            "global_tokens_per_second_from_median": 2.0,
                            "all_losses_finite": True,
                            "all_window_hashes_match_plan": True,
                            "per_rank": [
                                {
                                    "peak_allocated_bytes": 3,
                                    "peak_reserved_bytes": 4,
                                    "peak_process_rss_bytes": 5,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            failure_path = root / "failures" / "failed.json"
            failure_path.parent.mkdir(parents=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "status": "capacity_failure",
                        "variant": "deepspeed_zero2",
                        "run_id": "failed",
                        "configuration": {"model_parameter_count": 200},
                        "failure": {
                            "type": "cuda_oom",
                            "phase": "backward",
                            "elapsed_seconds": 12.5,
                            "resource_snapshot": {
                                "gpu_index": 3,
                                "torch_allocated_bytes": 123,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            successes, failures = load_points(root)
            summary = summarize_variant(
                "deepspeed_zero2",
                successes["deepspeed_zero2"],
                failures["deepspeed_zero2"],
            )

            self.assertEqual(
                summary["maximum_tested_success"]["parameter_count"], 100
            )
            self.assertEqual(
                summary["first_tested_failure_above_success"]["parameter_count"],
                200,
            )
            self.assertEqual(
                summary["maximum_tested_success"]["configuration"]["n_layer"],
                2,
            )
            self.assertEqual(
                summary["first_tested_failure_above_success"][
                    "resource_snapshot"
                ]["torch_allocated_bytes"],
                123,
            )
            self.assertFalse(summary["success_is_lower_bound_without_failure"])

    def test_discovers_statusless_native_ddp_and_legacy_oom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = root / "ddp" / "strong" / "4gpu" / "success.json"
            success.parent.mkdir(parents=True)
            success.write_text(
                json.dumps(
                    {
                        "benchmark": "C deterministic DDP no-Profiler benchmark",
                        "run_id": "success",
                        "configuration": {
                            "world_size": 4,
                            "model_parameter_count": 100,
                        },
                        "measurement": {
                            "median_slowest_rank_step_ms": 1.0,
                            "global_tokens_per_second_from_median": 2.0,
                            "all_losses_finite": True,
                            "all_window_hashes_match_plan": True,
                            "per_rank": [
                                {
                                    "peak_allocated_mib": 3.0,
                                    "peak_reserved_mib": 4.0,
                                    "peak_process_rss_bytes": 5,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            failure = root / "ddp" / "strong" / "4gpu" / "oom.json"
            failure.write_text(
                json.dumps(
                    {
                        "status": "oom",
                        "backend": "native_ddp",
                        "world_size": 4,
                        "model": {"parameter_count": 200},
                        "training": {"dtype": "bfloat16"},
                        "failure": {
                            "phase": "forward",
                            "requested_mib": 8.0,
                            "device_total_gib": 24.0,
                            "pytorch_allocated_gib": 22.0,
                            "reserved_unallocated_mib": 512.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            successes, failures = load_points(root)
            summary = summarize_variant(
                "native_ddp",
                successes["native_ddp"],
                failures["native_ddp"],
            )
            self.assertEqual(
                summary["maximum_tested_success"]["parameter_count"], 100
            )
            failed = summary["first_tested_failure_above_success"]
            self.assertEqual(failed["parameter_count"], 200)
            self.assertEqual(
                failed["resource_snapshot"]["requested_allocation_bytes"],
                8 * 1024**2,
            )
            self.assertEqual(failed["configuration"]["world_size"], 4)


if __name__ == "__main__":
    unittest.main()
