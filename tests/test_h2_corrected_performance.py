import copy
import json
from pathlib import Path
import unittest

from serving.benchmark_h2_corrected import validate_qualification, validate_timed_run
from serving.h2_runtime import aggregate_waves
from scripts.summarize_h2_nanovllm import summarize_runs

ROOT = Path(__file__).resolve().parents[1]


def qualification_fixture():
    # Read only small configuration/evidence metadata; these tests need no GPU,
    # downloaded weights, upstream imports, or generated token workloads.
    performance = json.loads((ROOT / "config/serving/h2_corrected_pilot.json").read_text())
    validation = json.loads((ROOT / performance["validation_config"]).read_text())
    pilot = json.loads((ROOT / validation["base_pilot"]).read_text())
    config = json.loads((ROOT / pilot["h2_config"]).read_text())
    cases = {}
    for case_id in performance["pilot_case_ids"]:
        batch, prompt, output, prefix = case_id.split("_")
        cases[case_id] = dict(batch_size=int(batch[1:]), prompt_length=int(prompt[1:]),
            output_length=int(output[1:]), prefix_mode=prefix)
    comparisons = [dict(case_id=case_id, variant=name, phase=phase,
        exact=True, compared_steps=64, mismatched_steps=[])
        for case_id in cases for name in performance["variants"][1:] for phase in ("cold", "warm")]
    gate = dict(status="pass", gpu_execution_allowed=True,
        evidence_class="corrected_policy_numerical_gate", policy_id=validation["policy_id"],
        policy=validation["policy"], common_engine=pilot["engine"],
        allowed_variants=performance["variants"], validated_case_ids=list(cases),
        comparisons=comparisons, compared_logit_tensors=64 * len(comparisons))
    return performance, validation, pilot, config, cases, gate


def timed_fixture():
    from scripts.summarize_h2_nanovllm import VARIANTS
    case_id = "fixture"
    case = dict(workload_sha256="unit-test", batch_size=1, prompt_length=512, output_length=64,
        expected_warm_cached_tokens_per_request=256,
        waves={phase: [{"request_id": phase + "_0"}] for phase in ("cold", "warm")})
    waves = {}
    for phase in ("cold", "warm"):
        hit = 256 if phase == "warm" else 0
        request = dict(request_id=phase + "_0", prompt_tokens=512, output_tokens=64,
            cached_prompt_tokens=hit, ttft_ms=10, e2e_ms=640, tpot_ms=10,
            output_token_ids_sha256="a" * 64)
        waves[phase] = [dict(wall_seconds=1, peak_allocated_bytes=1000,
            cached_prompt_tokens=hit, steps=64, requests=[copy.deepcopy(request)]) for _ in range(5)]
    run = dict(schema_version=1, stage="H2", status="pass", evidence_class="formal_gpu",
        variant="eager_prefix", case_id=case_id, run_id="run1",
        engine={**VARIANTS["eager_prefix"], "dtype": "bfloat16", "num_kvcache_blocks": 32,
            "kvcache_block_size": 256, "max_model_len": 1024, "max_num_seqs": 8, "max_num_batched_tokens": 4096},
        provenance={key: "test-only" for key in ("upstream_commit", "model_revision", "model_sha256",
            "tokenizer_sha256", "source_sha256", "environment")},
        workload=dict(sha256="unit-test", batch_size=1, prompt_length=512, output_length=64,
            prefix_mode="high", measured_pairs=5, sampling={"ignore_eos": True},
            cache_reset_policy="fresh_block_manager_per_pair"),
        compiler_settings=dict(recompile_limit=64, fail_on_recompile_limit_hit=True, suppress_errors=False),
        compile_counters=dict(before_measurement={"graphs": 1}, after_measurement={"graphs": 1}),
        released_state=dict(used_blocks=0), waves=waves,
        phases={phase: aggregate_waves(rows) for phase, rows in waves.items()})
    return run, {case_id: case}


class CorrectedPerformanceTests(unittest.TestCase):
    def test_complete_bounded_qualification(self):
        validate_qualification(*qualification_fixture())

    def test_rejects_failed_native_or_different_policy_gate(self):
        for key, value in (("status", "fail"), ("evidence_class", "native_gate"),
                ("policy_id", "different"), ("common_engine", {})):
            fixture = qualification_fixture()
            fixture[-1][key] = value
            with self.assertRaises(ValueError):
                validate_qualification(*fixture)

    def test_rejects_unqualified_or_expanded_scope(self):
        for kind in ("extra", "missing", "runs", "variants"):
            performance, validation, pilot, config, cases, gate = qualification_fixture()
            if kind == "extra":
                performance["pilot_case_ids"] = [*performance["pilot_case_ids"], "b32_p512_o256_high"]
            elif kind == "missing":
                gate["validated_case_ids"] = gate["validated_case_ids"][:-1]
            elif kind == "runs":
                performance["independent_runs"] = 4
            else:
                performance["variants"] = performance["variants"][:2]
            with self.assertRaises(ValueError):
                validate_qualification(performance, validation, pilot, config, cases, gate)

    def test_rejects_missing_duplicate_or_inexact_comparisons(self):
        for kind in ("missing", "duplicate", "mismatch", "steps"):
            fixture = qualification_fixture()
            rows = fixture[-1]["comparisons"]
            if kind == "missing":
                rows.pop()
            elif kind == "duplicate":
                rows[-1] = rows[0]
            elif kind == "mismatch":
                rows[0]["exact"] = False
            else:
                rows[0]["compared_steps"] = 4
            with self.assertRaises(ValueError):
                validate_qualification(*fixture)

    def test_timed_request_contract(self):
        validate_timed_run(*timed_fixture())

    def test_rejects_compile_fallback_recompile_and_leak(self):
        for kind in ("fallback", "recompile", "leak"):
            run, cases = timed_fixture()
            if kind == "fallback":
                run["compiler_settings"]["suppress_errors"] = True
            elif kind == "recompile":
                run["compile_counters"]["after_measurement"]["graphs"] = 2
            else:
                run["released_state"]["used_blocks"] = 1
            with self.assertRaises(ValueError):
                validate_timed_run(run, cases)

    def test_rejects_incomplete_waves_wrong_request_and_wrong_prefix_hits(self):
        for kind in ("missing", "request", "hit", "tpot"):
            run, cases = timed_fixture()
            waves = run["waves"]["warm"]
            if kind == "missing":
                waves.pop()
            elif kind == "request":
                waves[0]["requests"][0]["request_id"] = "not-this-request"
            elif kind == "hit":
                waves[0]["requests"][0]["cached_prompt_tokens"] = 0
            else:
                waves[0]["requests"][0]["tpot_ms"] = 20
            run["phases"]["warm"] = aggregate_waves(waves)
            with self.assertRaises(ValueError):
                validate_timed_run(run, cases)

    def test_summary_cannot_mix_native_and_corrected_provenance(self):
        from scripts.summarize_h2_nanovllm import VARIANTS
        runs = []
        for name, switches in VARIANTS.items():
            for index in range(1, 4):
                run, _ = timed_fixture()
                run.update(variant=name, run_id=f"run{index}")
                run["engine"].update(switches)
                for phase in ("cold", "warm"):
                    if not switches["prefix_cache"]:
                        run["phases"][phase]["cached_prompt_tokens"] = 0
                run["provenance"]["numerical_policy_id"] = "native" if name == "graph_no_prefix" else "corrected"
                runs.append((Path(f"{name}/run{index}.json"), run))
        with self.assertRaisesRegex(ValueError, "Cross-variant invariants"):
            summarize_runs(runs)


if __name__ == "__main__":
    unittest.main()
