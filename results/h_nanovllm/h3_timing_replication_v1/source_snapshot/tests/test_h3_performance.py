from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch

from serving.h3_performance import compare_tensors, validate_config, value_hash, full_reference_logits, reference_name, reference_comparison_key, SCOPES, VARIANTS
from scripts.run_h3_performance import formal_jobs
from scripts.summarize_h3_performance import summarize_traces, summarize_runs, validate_run

ROOT = Path(__file__).resolve().parents[1]


class H3PerformanceContractTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "config/serving/h3_performance_v1.json").read_text())

    def case(self):
        return dict(case_id="tiny", batch_size=1, prompt_length=32, output_length=3,
                    continuation_ids=[[10, 11, 12]], workload_sha256="work")

    def raw(self, variant="native_kv", run_id="run1"):
        config, case = self.config(), self.case()
        trace = dict(ttft_ms=2., itl_ms=[1., 1.], tpot_ms=1., wall_ms=5.,
            model_execution_ms=4., output_tokens=3, output_sha256=value_hash(case["continuation_ids"]),
            argmax_sha256=None, cached_positions=34, max_used_blocks=0 if variant == "native_kv" else 1,
            prefix_tokens=0, released=True, prefill_calls=1, decode_calls=2)
        measurements = {}
        for scope in SCOPES:
            rows = [deepcopy(trace) for _ in range(config["measure_repeats"])]
            if scope == "replay_e2e":
                for row in rows:
                    row.update(model_execution_ms=None, argmax_sha256="prediction")
            measurements[scope] = dict(traces=rows, peak_allocated_bytes=100, peak_reserved_bytes=200)
        return dict(status="pass", stage="H3", benchmark="routescale_performance", evidence_class="formal_gpu",
            variant=variant, case_id="tiny", run_id=run_id, provenance={"source": "fixed"},
            qualification_gate_sha256="gate", workload_sha256="work", dtype="bfloat16", prefix_cache=False,
            token_policy=config["token_policy"], measure_repeats=config["measure_repeats"],
            warmup_repeats=config["warmup_repeats"], free_bytes_after_run=config["minimum_free_bytes"],
            kv_bytes=(8 if variant == "native_kv" else 64) * 2**20, measurements=measurements,
            initialization=dict(total_seconds=1., graph_capture_seconds=.5 if variant == "paged_graph" else 0.))

    def test_new_config_funds_whole_prefill_and_future_page_growth(self):
        config = self.config()
        self.assertIs(validate_config(config), config)
        config["engine"]["max_num_batched_tokens"] = 1024
        with self.assertRaisesRegex(ValueError, "complete Prefill"):
            validate_config(config)

    def test_rejects_window_overflow_and_precision_changes(self):
        config = self.config()
        config["cases"][0]["prompt_length"] = 512
        with self.assertRaisesRegex(ValueError, "window overflow"):
            validate_config(config)
        config = self.config()
        config["dtype"] = "float32"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_single_request_reference_does_not_batch_the_model_calls(self):
        class Model:
            def __init__(self):
                self.batches = []
            def __call__(self, inputs):
                self.batches.append(inputs.shape[0])
                return inputs.float().unsqueeze(-1).expand(-1, -1, 7), None
        model = Model()
        inputs = torch.arange(12).reshape(3, 4)
        result = full_reference_logits(model, inputs, "single_request_full_recompute")
        self.assertEqual(model.batches, [1, 1, 1])
        self.assertTrue(torch.equal(result[:, 0], inputs[:, -1].float()))
        with self.assertRaises(ValueError):
            full_reference_logits(model, inputs, "unknown")

    def test_three_references_have_distinct_reports_and_gate_keys(self):
        config = json.loads((ROOT / "config/serving/h3_performance_v4.json").read_text())
        validate_config(config)
        self.assertFalse(config["allow_bf16_reduced_precision_reduction"])
        policies = [config["reference_policy"], *config["additional_reference_policies"]]
        self.assertEqual(len({reference_name(config, p) for p in policies}), 3)
        self.assertEqual(len({reference_comparison_key(config, p, "native_kv", "page_growth") for p in policies}), 3)
        config["additional_reference_policies"].append(config["reference_policy"])
        with self.assertRaisesRegex(ValueError, "additional references"):
            validate_config(config)

    def test_rotates_path_order_and_has_exactly_36_distinct_jobs(self):
        jobs = list(formal_jobs(self.config()))
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(set(jobs)), 36)
        orders = [[v for c, v, r in jobs if c == "latency_short" and r == f"run{i}"] for i in (1, 2, 3)]
        self.assertEqual(orders, [list(VARIANTS), ["paged_eager", "paged_graph", "native_kv"], ["paged_graph", "native_kv", "paged_eager"]])

    def test_numerical_gate_checks_each_logit_not_just_average_rmse(self):
        reference = torch.ones(2, 1, 50257, dtype=torch.bfloat16)
        bad = reference.clone()
        bad[0, 0, 0] = 2
        result = compare_tensors(bad, reference, self.config()["tolerances"])
        self.assertLess(result["max_relative_rmse"], .005)
        self.assertFalse(result["passed"])

    def test_graph_exact_gate_rejects_small_tolerated_bf16_difference(self):
        reference = torch.ones(2, 1, 8, dtype=torch.bfloat16)
        near = reference.clone()
        near[0, 0, 0] += .0078125
        self.assertTrue(compare_tensors(near, reference, self.config()["tolerances"])["passed"])
        self.assertFalse(compare_tensors(near, reference, self.config()["tolerances"], exact=True)["passed"])

    def test_fp32_reference_is_explicit_and_does_not_loosen_error_budget(self):
        actual = torch.ones(2, 1, 8, dtype=torch.bfloat16)
        reference = actual.float()
        with self.assertRaisesRegex(ValueError, "dtype"):
            compare_tensors(actual, reference, self.config()["tolerances"])
        self.assertTrue(compare_tensors(actual, reference, self.config()["tolerances"], fp32_reference=True)["passed"])
        reference += .02
        self.assertFalse(compare_tensors(actual, reference, self.config()["tolerances"], fp32_reference=True)["passed"])

    def test_separate_scope_throughput_uses_correct_denominator(self):
        raw = self.raw()
        execution = summarize_traces(raw["measurements"]["model_execution"]["traces"], "model_execution")
        wall = summarize_traces(raw["measurements"]["replay_e2e"]["traces"], "replay_e2e")
        self.assertEqual(execution["output_tokens_per_second"], 750)
        self.assertEqual(wall["output_tokens_per_second"], 600)

    def test_rejects_nonformal_stale_or_missing_token_evidence(self):
        for change in (dict(evidence_class="cpu_smoke"), dict(qualification_gate_sha256="stale"), dict(prefix_cache=True)):
            raw = {**self.raw(), **change}
            with self.assertRaises(ValueError):
                validate_run(raw, self.config(), self.case(), {"source": "fixed"}, "gate")
        raw = self.raw()
        raw["measurements"]["replay_e2e"]["traces"][0]["output_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "Output budget"):
            validate_run(raw, self.config(), self.case(), {"source": "fixed"}, "gate")

    def test_summary_rejects_missing_or_duplicate_runs(self):
        config = self.config()
        records = [(ROOT / f"results/{v}/{r}.json", self.raw(v, f"run{r}")) for v in VARIANTS for r in (1, 2, 3)]
        good = summarize_runs(records, config, [self.case()], {"source": "fixed"}, "gate")
        self.assertEqual(good["formal_processes"], 9)
        for bad in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaisesRegex(ValueError, "Missing, duplicated"):
                summarize_runs(bad, config, [self.case()], {"source": "fixed"}, "gate")


if __name__ == "__main__":
    unittest.main()
