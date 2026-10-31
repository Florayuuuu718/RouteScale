from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from scripts.summarize_g3_serving import summarize_runs


def make_run(backend: str, run_index: int, output_tps: float) -> dict:
    return {
        "stage": "G3",
        "benchmark": "native_serving",
        "status": "pass",
        "evidence_class": "formal_gpu",
        "run_id": f"run{run_index}",
        "version": {"source_sha256": {"model.py": "same"}},
        "environment": {"gpu": "test"},
        "artifact": {"sha256": "artifact"},
        "workload": {"prompt_token_ids_sha256": "workload"},
        "configuration": {
            "backend": backend,
            "prefix_mode": "none",
            "batch_size": 1,
            "prompt_length": 128,
            "output_length": 64,
            "dtype": "bfloat16",
            "decode_input_policy": "fixed",
            "context_overflow": False,
        },
        "measurement": {
            "outputs_deterministic": True,
            "predictions_deterministic": True,
            "output_token_ids_sha256": ["tokens"] * 5,
            "ttft_p50_ms": 10.0 + run_index,
            "tpot_p50_ms": 2.0 + run_index,
            "e2e_wall_p50_ms": 100.0 + run_index,
            "output_tokens_per_second": output_tps,
            "peak_allocated_bytes": 1000 + run_index,
            "peak_reserved_bytes": 2000 + run_index,
        },
    }


class H3SummaryTests(unittest.TestCase):
    def test_median_of_three_runs_and_backend_comparison(self) -> None:
        path_runs = []
        for backend, throughput in (("naive", 100.0), ("kv_cache", 250.0)):
            for run_index, delta in ((1, -10.0), (2, 0.0), (3, 10.0)):
                run = make_run(backend, run_index, throughput + delta)
                path_runs.append((Path(f"{backend}/run{run_index}.json"), run))
        summary = summarize_runs(path_runs, expected_runs=3)
        self.assertEqual(summary["variant_count"], 2)
        self.assertEqual(summary["comparison_count"], 1)
        comparison = summary["comparisons"][0]
        self.assertEqual(comparison["kv_to_naive_output_throughput_ratio"], 2.5)
        self.assertTrue(summary["invariants"]["median_of_independent_run_medians"])

    def test_rejects_mismatched_workload(self) -> None:
        path_runs = []
        for run_index in (1, 2, 3):
            run = make_run("naive", run_index, 100.0)
            if run_index == 3:
                run = deepcopy(run)
                run["workload"]["prompt_token_ids_sha256"] = "different"
            path_runs.append((Path(f"run{run_index}.json"), run))
        with self.assertRaisesRegex(ValueError, "invariants differ"):
            summarize_runs(path_runs, expected_runs=3)

    def test_rejects_naive_kv_workload_mismatch(self) -> None:
        path_runs = []
        for backend in ("naive", "kv_cache"):
            for run_index in (1, 2, 3):
                run = make_run(backend, run_index, 100.0)
                if backend == "kv_cache":
                    run["workload"]["prompt_token_ids_sha256"] = "other"
                path_runs.append((Path(f"{backend}/run{run_index}.json"), run))
        with self.assertRaisesRegex(ValueError, "naive/KV invariants differ"):
            summarize_runs(path_runs, expected_runs=3)


if __name__ == "__main__":
    unittest.main()

