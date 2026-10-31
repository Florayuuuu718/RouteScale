from pathlib import Path
import unittest

from serving.h2_workloads import build_case
from scripts.summarize_h2_nanovllm import summarize_runs, validate_run, VARIANTS, LATENCIES


def make_run(variant, index):
    phases = {}
    for phase in ("cold", "warm"):
        metrics = {key: 10 + {"p50": 0, "p95": 5, "p99": 10}[key.split("_")[1]]
                   for key in LATENCIES}
        metrics.update(output_tokens_per_second=100 * (2 if variant == "graph_no_prefix" else 1)
            + (-10, 0, 10)[index - 1], peak_allocated_bytes=1000,
            requests_completed=40, prompt_tokens=20480, output_tokens=2560,
            cached_prompt_tokens=10240 if variant == "eager_prefix" and phase == "warm" else 0)
        phases[phase] = metrics
    return {"schema_version": 1, "stage": "H2", "status": "pass", "evidence_class": "formal_gpu",
        "case_id": "b8_p512_o64_high", "variant": variant, "run_id": f"run{index}",
        "provenance": {"upstream_commit": "test-only", "model_revision": "test-only",
            "model_sha256": "test-only", "tokenizer_sha256": {"tokenizer": "test-only"},
            "source_sha256": {"harness": "test-only"}, "environment": {"gpu": "unit-test fixture"}},
        "engine": {**VARIANTS[variant], "dtype": "bfloat16", "num_kvcache_blocks": 128, "max_num_batched_tokens": 16384,
            "max_num_seqs": 32, "max_model_len": 1024, "kvcache_block_size": 256},
        "workload": {"sha256": "test-only", "batch_size": 8, "prompt_length": 512,
            "output_length": 64, "prefix_mode": "high", "measured_pairs": 5,
            "cache_reset_policy": "fresh_block_manager_per_pair", "sampling": {"ignore_eos": True}},
        "phases": phases}


def run_set():
    return [(Path(f"{variant}/run{i}.json"), make_run(variant, i))
            for variant in VARIANTS for i in (1, 2, 3)]


class I2WorkloadTests(unittest.TestCase):
    def case(self, **overrides):
        arguments = dict(batch=8, prompt=512, output=64, prefix_mode="high", seed=7)
        arguments.update(overrides)
        return build_case(list(range(100, 500)), lambda s: [10000 + ord(c) for c in s], **arguments)

    def test_reproducible_hash_and_exact_lengths(self):
        a, b = self.case(), self.case()
        self.assertEqual(a, b)
        self.assertEqual(len(a["waves"]["cold"]), 8)
        self.assertTrue(all(len(r["prompt_token_ids"]) == 512 for wave in a["waves"].values() for r in wave))

    def test_waves_have_distinct_requests_but_share_the_intended_prefix(self):
        case = self.case()
        requests = [r["prompt_token_ids"] for wave in case["waves"].values() for r in wave]
        self.assertEqual(len(set(map(tuple, requests))), 16)
        self.assertTrue(all(r[:384] == requests[0][:384] for r in requests))
        self.assertEqual(case["expected_warm_cached_tokens_per_request"], 256)

    def test_short_shared_prefix_is_a_negative_control(self):
        case = self.case(prompt=128)
        self.assertEqual(case["requested_shared_prefix_tokens"], 96)
        self.assertEqual(case["expected_warm_cached_tokens_per_request"], 0)

    def test_no_prefix_case_never_repeats_a_first_block(self):
        case = self.case(prefix_mode="none", batch=32)
        prompts = [r["prompt_token_ids"] for wave in case["waves"].values() for r in wave]
        self.assertEqual(len({tuple(p[:256]) for p in prompts}), 64)
        self.assertEqual(case["expected_warm_cached_tokens_per_request"], 0)

    def test_output_length_does_not_change_prompt_stream(self):
        short, long = self.case(output=64), self.case(output=256)
        self.assertEqual(short["prompt_stream_sha256"], long["prompt_stream_sha256"])
        self.assertNotEqual(short["workload_sha256"], long["workload_sha256"])

    def test_rejects_a_marker_that_cannot_fit(self):
        with self.assertRaisesRegex(ValueError, "marker"):
            self.case(prompt=8)


class I2SummaryTests(unittest.TestCase):
    def test_graph_only_scope_is_explicit_and_does_not_publish_prefix_results(self):
        runs = [(path, run) for path, run in run_set() if run["variant"] != "eager_prefix"]
        with self.assertRaisesRegex(ValueError, "Missing single-variable control"):
            summarize_runs(runs)
        summary = summarize_runs(runs, mechanism="cuda_graph")
        self.assertEqual(summary["scope"], "cuda_graph_only")
        self.assertEqual(summary["variant_count"], 2)
        self.assertEqual({row["mechanism"] for row in summary["comparisons"]}, {"cuda_graph"})

    def test_aggregates_independent_runs_and_single_variable_pairs(self):
        result = summarize_runs(run_set())
        self.assertEqual(result["variant_count"], 3)
        self.assertEqual(len(result["comparisons"]), 4)
        graph = next(c for c in result["comparisons"] if c["mechanism"] == "cuda_graph")
        self.assertEqual(graph["output_throughput_ratio"], 2)
        prefix = next(c for c in result["comparisons"] if c["mechanism"] == "prefix_cache" and c["phase"] == "warm")
        self.assertEqual(prefix["prompt_token_hit_rate"], 0.5)

    def test_rejects_cpu_evidence(self):
        run = make_run("eager_no_prefix", 1)
        run["evidence_class"] = "cpu_metadata_probe"
        with self.assertRaisesRegex(ValueError, "formal GPU"):
            validate_run(run)

    def test_rejects_cross_variant_model_and_cache_budget_changes(self):
        for kind in ("model", "cache"):
            runs = run_set()
            for _, run in runs:
                if run["variant"] == "graph_no_prefix":
                    if kind == "model":
                        run["provenance"]["model_revision"] = "another-model"
                    else:
                        run["engine"]["num_kvcache_blocks"] = 64
            with self.assertRaisesRegex(ValueError, "Cross-variant invariants"):
                summarize_runs(runs)

    def test_rejects_incomplete_or_duplicate_independent_runs(self):
        for runs in (run_set()[:-1], run_set() + [run_set()[0]]):
            with self.assertRaisesRegex(ValueError, "run IDs"):
                summarize_runs(runs)

    def test_rejects_workload_change_between_independent_runs(self):
        runs = run_set()
        runs[0][1]["workload"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Independent-run invariants"):
            summarize_runs(runs)

    def test_rejects_invalid_metric_and_incomplete_token_budget(self):
        for metric, value in (("ttft_p50_ms", float("nan")), ("output_tokens", 2559)):
            run = make_run("eager_no_prefix", 1)
            run["phases"]["warm"][metric] = value
            with self.assertRaises(ValueError):
                validate_run(run)

    def test_rejects_a_cache_hit_in_cold_or_disabled_controls(self):
        for variant, phase in (("eager_prefix", "cold"), ("eager_no_prefix", "warm")):
            run = make_run(variant, 1)
            run["phases"][phase]["cached_prompt_tokens"] = 256
            with self.assertRaisesRegex(ValueError, "Unexpected cache hits"):
                validate_run(run)

    def test_rejects_labels_that_do_not_match_engine_flags(self):
        run = make_run("graph_no_prefix", 1)
        run["engine"]["enforce_eager"] = True
        with self.assertRaisesRegex(ValueError, "engine switches"):
            validate_run(run)

    def test_rejects_prefill_budget_that_breaks_cold_bulk_admission(self):
        run = make_run("eager_no_prefix", 1)
        run["engine"]["max_num_batched_tokens"] = 1024
        with self.assertRaisesRegex(ValueError, "bulk admission"):
            validate_run(run)

    def test_rejects_pool_that_fits_prefill_but_cannot_cover_decode(self):
        run = make_run("eager_no_prefix", 1)
        run["engine"]["num_kvcache_blocks"] = 16
        with self.assertRaisesRegex(ValueError, "cold-wave decode"):
            validate_run(run)

    def test_rejects_an_unpinned_precision(self):
        run = make_run("eager_no_prefix", 1)
        run["engine"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "BF16"):
            validate_run(run)


if __name__ == "__main__":
    unittest.main()
