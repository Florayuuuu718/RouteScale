import copy
import json
from pathlib import Path
import unittest

from serving.h2_scheduling import admission_count, build_cases, summarize_traces, validate_trace
from scripts.run_h2_scheduling import scheduling_jobs
from scripts.summarize_h2_scheduling import check_gate, compare_probes, engine_contract, summarize_runs

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    config = json.loads((ROOT / "config/serving/h2_scheduling.json").read_text())
    validation = json.loads((ROOT / config["base_validation"]).read_text())
    pilot = json.loads((ROOT / validation["base_pilot"]).read_text())
    base = json.loads((ROOT / pilot["h2_config"]).read_text())
    cases = build_cases(config, [[i+1] * 128 for i in range(16)])
    return config, pilot, base, cases


def simulate(case, config, policy):
    """CPU state/clock fixture, never evidence of GPU performance."""
    pending, active, states, steps, clock = list(case["requests"]), {}, {}, [], 0
    while pending or active:
        count = admission_count(policy, len(active), 8, pending, clock)
        new = []
        for _ in range(count):
            r = pending.pop(0)
            key = r["request_id"]
            states[key] = dict(request=r, admitted=clock, times=[])
            active[key] = states[key]
            new.append(key)
        if not active:
            clock = pending[0]["arrival_ms"]
            continue
        ids = new or list(active)
        check_ms, clock = clock, clock + 10
        done, before = [], len(active)
        for key in ids:
            active[key]["times"].append(clock)
            if len(active[key]["times"]) == active[key]["request"]["output_tokens"]:
                done.append(key)
        for key in done:
            del active[key]
        steps.append(dict(is_prefill=bool(new), request_ids=ids, batch_size=len(ids), active_before=before,
            completed_request_ids=done, admission_check_ms=check_ms, model_start_ms=check_ms, token_visible_ms=clock))
    rows = []
    for r in case["requests"]:
        s = states[r["request_id"]]
        times = s["times"]
        rows.append(dict(request_id=r["request_id"], output_tokens=r["output_tokens"], prompt_tokens=128,
            arrival_ms=r["arrival_ms"], admitted_ms=s["admitted"], first_scheduled_ms=s["admitted"],
            token_visible_ms=times, admission_delay_ms=s["admitted"] - r["arrival_ms"], scheduler_wait_ms=0,
            ttft_ms=times[0]-r["arrival_ms"], e2e_ms=times[-1]-r["arrival_ms"],
            tpot_ms=(times[-1]-times[0])/(len(times)-1), inter_token_ms=[b-a for a,b in zip(times,times[1:])],
            cached_prompt_tokens=0, output_token_ids_sha256="a"*64))
    return dict(wall_seconds=clock/1000, peak_allocated_bytes=1000, requests=rows, steps=steps,
        released_state=dict(used_blocks=0, nonzero_refcounts=0), finite_logits_checked=False)


def probe_fixture():
    config, pilot, base, cases = inputs()
    provenance = {"only": "unit-test fixture"}
    reports = {}
    for policy in config["policies"]:
        report = dict(status="pass", evidence_class="functional_gpu_scheduling_probe", policy=policy,
            provenance=provenance, engine=engine_contract(pilot, config), cases={}, kv_tensor_bytes=939524096,
            compiler_settings=dict(recompile_limit=64, fail_on_recompile_limit_hit=True, suppress_errors=False),
            free_bytes_after_run=2**30)
        for key, case in cases.items():
            trace = simulate(case, config, policy)
            trace["finite_logits_checked"] = True
            report["cases"][key] = dict(workload_sha256=case["workload_sha256"], pass_numerical=True,
                functional_trace=trace, comparisons={r["request_id"]: dict(steps=r["output_tokens"],
                    exact_steps=r["output_tokens"], allclose=True, max_absolute_error=0, max_relative_rmse=0)
                    for r in case["requests"]})
            report["cases"][key]["pass"] = True
        reports[policy] = report
    gate = dict(status="pass", gpu_execution_allowed=True, evidence_class="scheduling_numerical_gate",
        provenance=provenance, engine=engine_contract(pilot, config), allowed_policies=config["policies"],
        tolerances=dict(atol=.05, rtol=.02, max_relative_rmse=.005),
        case_ids=list(cases), comparisons=compare_probes(reports, config, pilot, cases), kv_tensor_bytes=939524096)
    return config, pilot, base, cases, gate, reports


def formal_fixture():
    config, pilot, base, cases, gate, _ = probe_fixture()
    runs = []
    for case, policy, run_id in scheduling_jobs(config):
        traces = [simulate(cases[case], config, policy) for _ in range(3)]
        run = dict(schema_version=1, stage="H2", status="pass", evidence_class="formal_gpu_scheduling",
            scope="admission_policy", case_id=case, policy=policy, run_id=run_id,
            provenance=gate["provenance"], engine=gate["engine"], kv_tensor_bytes=gate["kv_tensor_bytes"],
            workload_sha256=cases[case]["workload_sha256"], timing_policy=config["timing_policy"],
            sampling=base["sampling"], seed=base["seed"], warmup_traces=1, measured_traces=3,
            compiler_settings=dict(recompile_limit=64, fail_on_recompile_limit_hit=True, suppress_errors=False),
            compile_counters=dict(before={"graphs": 1}, after={"graphs": 1}), free_bytes_after_run=2**30,
            traces=traces, metrics=summarize_traces(traces))
        runs.append((Path(f"{case}/{policy}/{run_id}.json"), run))
    return runs, config, pilot, base, cases, gate


class SchedulingTests(unittest.TestCase):
    def test_tiled_gate_requires_exact_logits_not_just_old_tolerance(self):
        from serving.h2_linear_policy import LINEAR_POLICY
        config, pilot, _, cases, gate, reports = probe_fixture()
        config["linear_policy"] = copy.deepcopy(LINEAR_POLICY)
        for report in reports.values():
            report["engine"] = engine_contract(pilot, config)
        gate["engine"] = engine_contract(pilot, config)
        compare_probes(reports, config, pilot, cases)
        check_gate(gate, gate["provenance"], gate["engine"], cases)
        reports["continuous"]["cases"]["mixed_burst"]["comparisons"]["r00"]["exact_steps"] -= 1
        with self.assertRaisesRegex(ValueError, "exact"):
            compare_probes(reports, config, pilot, cases)
        gate["comparisons"][0]["exact_steps"] -= 1
        with self.assertRaisesRegex(ValueError, "exact"):
            check_gate(gate, gate["provenance"], gate["engine"], cases)

    def test_active_cap_is_not_the_upstream_per_step_cap(self):
        pending = [{"arrival_ms": 0}] * 16
        self.assertEqual(admission_count("fixed_cohort", 4, 8, pending, 0), 0)
        self.assertEqual(admission_count("continuous", 4, 8, pending, 0), 4)
        self.assertEqual(admission_count("continuous", 8, 8, pending, 0), 0)
        with self.assertRaises(ValueError):
            admission_count("continuous", 9, 8, pending, 0)

    def test_does_not_admit_future_arrivals(self):
        pending = [{"arrival_ms": 160}] * 8
        self.assertEqual(admission_count("continuous", 4, 8, pending, 159), 0)
        self.assertEqual(admission_count("continuous", 4, 8, pending, 160), 4)

    def test_fixed_inputs_and_bounded_rotated_jobs(self):
        config, _, _, cases = inputs()
        self.assertEqual(sum(sum(r["output_tokens"] for r in c["requests"]) for c in cases.values()), 1664)
        jobs = list(scheduling_jobs(config))
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len(set(jobs)), 18)
        self.assertEqual([jobs[i][1] for i in (0, 6, 12)], ["fixed_cohort", "continuous", "fixed_cohort"])
        config["max_active_requests"] = 16
        with self.assertRaises(ValueError):
            build_cases(config, [[1]*128]*16)

    def test_equal_burst_is_a_structural_negative_control(self):
        config, _, _, cases = inputs()
        fixed = simulate(cases["equal_burst"], config, "fixed_cohort")
        continuous = simulate(cases["equal_burst"], config, "continuous")
        self.assertEqual(fixed, continuous)
        self.assertEqual(len(fixed["steps"]), 64)

    def test_refill_reduces_steps_but_can_interrupt_old_requests(self):
        config, _, _, cases = inputs()
        metrics = {}
        for policy in config["policies"]:
            for case in cases.values():
                trace = simulate(case, config, policy)
                validate_trace(trace, case, config, policy)
            trace = simulate(cases["mixed_burst"], config, policy)
            metrics[policy] = summarize_traces([trace])
        self.assertEqual(metrics["fixed_cohort"]["decode_steps"], 126)
        self.assertEqual(metrics["continuous"]["decode_steps"], 77)
        self.assertGreater(metrics["continuous"]["mean_decode_batch_size"], metrics["fixed_cohort"]["mean_decode_batch_size"])
        self.assertGreater(metrics["continuous"]["itl_max_ms"], metrics["fixed_cohort"]["itl_max_ms"])

    def test_rejects_missing_requests_budget_changes_and_bad_clock(self):
        config, _, _, cases = inputs()
        case = cases["mixed_staggered"]
        for kind in ("missing", "budget", "arrival", "release", "itl"):
            trace = simulate(case, config, "continuous")
            if kind == "missing":
                trace["requests"].pop()
            elif kind == "budget":
                trace["requests"][0]["output_tokens"] += 1
            elif kind == "arrival":
                trace["requests"][-1]["admitted_ms"] = 0
            elif kind == "release":
                trace["released_state"]["used_blocks"] = 1
            else:
                trace["requests"][0]["inter_token_ms"][0] = 999
            with self.assertRaises(ValueError):
                validate_trace(trace, case, config, "continuous")

    def test_rejects_refill_mislabeled_as_fixed_cohort(self):
        config, _, _, cases = inputs()
        case = cases["mixed_burst"]
        with self.assertRaises(ValueError):
            validate_trace(simulate(case, config, "continuous"), case, config, "fixed_cohort")

    def test_qualified_gate_and_1664_full_vocabulary_steps(self):
        config, _, _, cases, gate, _ = probe_fixture()
        check_gate(gate, gate["provenance"], gate["engine"], cases)
        self.assertEqual(sum(r["compared_steps"] for r in gate["comparisons"]), 1664)

    def test_rejects_numerical_differences_missing_steps_and_mixed_provenance(self):
        for kind in ("difference", "missing", "provenance"):
            config, pilot, _, cases, _, reports = probe_fixture()
            if kind == "provenance":
                reports["continuous"]["provenance"] = {"changed": True}
            else:
                row = reports["continuous"]["cases"]["mixed_burst"]["comparisons"]["r00"]
                row["max_relative_rmse"] = .02 if kind == "difference" else 0
                if kind == "missing":
                    row["steps"] -= 1
            with self.assertRaises(ValueError):
                compare_probes(reports, config, pilot, cases)

    def test_numerical_gate_does_not_release_native_or_other_scopes(self):
        _, _, _, cases, gate, _ = probe_fixture()
        for field, value in (("status", "fail"), ("evidence_class", "other"), ("allowed_policies", ["fixed_cohort"])):
            other = copy.deepcopy(gate)
            other[field] = value
            with self.assertRaises(ValueError):
                check_gate(other, gate["provenance"], gate["engine"], cases)

    def test_formal_summary_uses_all_18_processes_with_single_policy_control(self):
        report = summarize_runs(*formal_fixture())
        self.assertEqual(len(report["variants"]), 6)
        self.assertEqual(len(report["comparisons"]), 3)
        equal = next(r for r in report["comparisons"] if r["case_id"] == "equal_burst")
        self.assertEqual(equal["output_throughput_ratio"], 1)

    def test_summary_rejects_missing_run_mixed_provenance_and_compile_contamination(self):
        for kind in ("missing", "provenance", "compile", "metric"):
            runs, *rest = formal_fixture()
            if kind == "missing":
                runs.pop()
            elif kind == "provenance":
                runs[0][1]["provenance"] = {"different": True}
            elif kind == "compile":
                runs[0][1]["compile_counters"]["after"]["graphs"] = 2
            else:
                runs[0][1]["metrics"]["output_tokens_per_second"] = 99999
            with self.assertRaises(ValueError):
                summarize_runs(runs, *rest)


if __name__ == "__main__":
    unittest.main()
