import copy
import json
from pathlib import Path
import unittest

from serving.h2_kv_pressure import (
    build_cases, effective_pilot, engine_contract, funded_cohort, request_credit,
    summarize_pressure_traces, validate_pressure_trace,
)
from serving.h2_linear_policy import LINEAR_POLICY
from scripts.run_h2_kv_pressure import formal_jobs
from scripts.summarize_h2_kv_pressure import COMPILER, check_gate, check_report, gate_contract, summarize_runs

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    config = json.loads((ROOT/"config/serving/h2_kv_pressure.json").read_text())
    pilot = json.loads((ROOT/"config/serving/h2_pilot.json").read_text())
    base = json.loads((ROOT/pilot["h2_config"]).read_text())
    cases = build_cases(config, [[i+1]*128 for i in range(16)])
    return config, pilot, base, cases


def simulate(case, config, policy):
    """CPU state/clock fixture. Not GPU timing or numerical evidence."""
    pending, rows, steps, clock = list(case["requests"]), {}, [], 0
    while pending:
        cohort, committed = funded_cohort(pending,policy,case["num_blocks"])
        pending = pending[len(cohort):]
        for request in cohort:
            rows[request["request_id"]] = dict(admitted_ms=clock, first_scheduled_ms=clock, times=[])
        for index in range(case["output_tokens"]):
            start, clock = clock, clock+10
            keys = [r["request_id"] for r in cohort]
            used = len(cohort)*4 if policy == "static_window" else len(cohort)*((128+index+255)//256)
            for key in keys:
                rows[key]["times"].append(clock)
            steps.append(dict(is_prefill=index==0, request_ids=keys, batch_size=len(keys), active_before=len(keys),
                completed_request_ids=keys if index==case["output_tokens"]-1 else [], admission_check_ms=start,
                model_start_ms=start, token_visible_ms=clock, used_kv_blocks=used,
                free_kv_blocks=case["num_blocks"]-used, committed_kv_blocks=committed,
                cached_tokens_per_request=[128+index]*len(keys)))
    requests = []
    for request in case["requests"]:
        state = rows[request["request_id"]]
        times = state["times"]
        requests.append(dict(request_id=request["request_id"], prompt_tokens=128, output_tokens=len(times),
            arrival_ms=0, admitted_ms=state["admitted_ms"], first_scheduled_ms=state["first_scheduled_ms"],
            token_visible_ms=times, admission_delay_ms=state["admitted_ms"], scheduler_wait_ms=0,
            ttft_ms=times[0], e2e_ms=times[-1], tpot_ms=(times[-1]-times[0])/(len(times)-1),
            inter_token_ms=[b-a for a,b in zip(times,times[1:])], cached_prompt_tokens=0, output_token_ids_sha256="a"*64))
    return dict(wall_seconds=clock/1000, peak_allocated_bytes=1000, requests=requests, steps=steps,
        finite_logits_checked=False, released_state=dict(running=0,waiting=0,used_blocks=0,nonzero_refcounts=0))


def fixtures():
    config,pilot,base,cases = inputs()
    provenance = {"test_only":"CPU fixtures, never measured GPU evidence"}
    gate = dict(status="pass",evidence_class="kv_pressure_numerical_gate",gpu_performance_execution_allowed=True,
        contract=gate_contract(config,cases,provenance,pilot), compared_vectors=4160, exact_vectors=4160,
        report_sha256={f"preflight/{key}_{p}.json":"b"*64 for key in cases for p in config["policies"]})
    reports=[]
    for key,policy,run_id in formal_jobs(config):
        case=cases[key]
        traces=[simulate(case,config,policy) for _ in range(3)]
        report=dict(status="pass",scope="kv_funded_admission",evidence_class="formal_gpu_kv_pressure",
            case_id=key,policy=policy,run_id=run_id,provenance=provenance,engine=engine_contract(pilot,case),
            workload_sha256=case["workload_sha256"],kv_tensor_bytes=case["num_blocks"]*29360128,
            compiler_settings=COMPILER,free_bytes_after_run=2**30,sampling=base["sampling"],seed=base["seed"],
            timing_policy=config["timing_policy"],admission_policy=config["admission_policy"],warmup_traces=1,measured_traces=3,
            traces=traces,compile_counters=dict(before={"graphs":1},after={"graphs":1}),
            initialization=dict(initialization_seconds=10,prefill_linear_policy=dict(**LINEAR_POLICY,linear_modules=112,
                padding_rows=0,weight_dtype_unchanged=True,decode_unchanged=True)),
            metrics=summarize_pressure_traces(traces),qualification_gate_sha256="c"*64)
        reports.append((Path(f"{key}/{policy}/{run_id}.json"),report))
    return config,pilot,base,cases,provenance,gate,reports


class KVPressureTests(unittest.TestCase):
    def test_peak_credit_accounts_for_last_output_not_yet_cached(self):
        request=dict(prompt_token_ids=[1]*128,output_tokens=129)
        self.assertEqual(request_credit(request,"paged"),1)
        request["output_tokens"]=130
        self.assertEqual(request_credit(request,"paged"),2)
        self.assertEqual(request_credit(request,"static_window"),4)

    def test_current_free_pages_are_not_a_safe_growth_budget(self):
        pending=[dict(prompt_token_ids=[i+1]*128,output_tokens=450) for i in range(8)]
        cohort,credit=funded_cohort(pending,"paged",16)
        self.assertEqual(len(cohort),5)  # Eight prompt pages fit, but 24 peak pages do not.
        self.assertEqual(credit,15)

    def test_fcfs_cannot_skip_an_unfundable_head_request(self):
        pending=[dict(prompt_token_ids=[1]*128,output_tokens=n) for n in (700,64)]
        with self.assertRaisesRegex(ValueError,"head request"):
            funded_cohort(pending,"paged",3)

    def test_tight_and_ample_capacity_is_derived_not_hardcoded_in_scheduler(self):
        config,_,_,cases=inputs()
        for key in cases:
            case=cases[key]
            sizes=[len(funded_cohort(case["requests"],p,case["num_blocks"])[0]) for p in config["policies"]]
            self.assertEqual(sizes,[8,8] if key=="ample_short" else [4,8])

    def test_invalid_request_budget_policy_and_configuration_fail(self):
        for request,policy in [(dict(prompt_token_ids=[1]*128,output_tokens=1),"paged"),
                (dict(prompt_token_ids=[1]*128,output_tokens=897),"paged"),
                (dict(prompt_token_ids=[1]*128,output_tokens=64),"unknown")]:
            with self.assertRaises(ValueError):
                request_credit(request,policy)
        config,_,_,_=inputs()
        config["requests_per_trace"]=32
        with self.assertRaises(ValueError):
            build_cases(config,[[1]*128]*16)

    def test_physical_pool_override_does_not_mutate_qualified_pilot(self):
        _,pilot,_,cases=inputs()
        override=effective_pilot(pilot,cases["tight_short"])
        self.assertEqual(override["engine"]["num_kvcache_blocks"],16)
        self.assertEqual(pilot["engine"]["num_kvcache_blocks"],32)
        self.assertEqual(override["engine"]["max_num_seqs"],8)

    def test_all_cpu_trace_states_and_timing_formulas_replay(self):
        config,_,_,cases=inputs()
        for case in cases.values():
            for policy in config["policies"]:
                validate_pressure_trace(simulate(case,config,policy),case,config,policy)

    def test_growth_uses_all_sixteen_pages_at_257_tokens(self):
        config,_,_,cases=inputs()
        trace=simulate(cases["tight_growth"],config,"paged")
        self.assertEqual(trace["steps"][128]["used_kv_blocks"],8)
        self.assertEqual(trace["steps"][129]["used_kv_blocks"],16)
        self.assertEqual(trace["steps"][129]["cached_tokens_per_request"],[257]*8)

    def test_ample_budget_has_identical_model_call_structure(self):
        config,_,_,cases=inputs()
        left=simulate(cases["ample_short"],config,"static_window")
        right=simulate(cases["ample_short"],config,"paged")
        self.assertEqual([(s["is_prefill"],s["request_ids"]) for s in left["steps"]],
            [(s["is_prefill"],s["request_ids"]) for s in right["steps"]])

    def test_corrupt_page_growth_budget_or_positions_are_rejected(self):
        config,_,_,cases=inputs()
        for key,value in (("used_kv_blocks",9),("free_kv_blocks",0),("committed_kv_blocks",17),("cached_tokens_per_request",[129]*8)):
            trace=simulate(cases["tight_growth"],config,"paged")
            trace["steps"][0][key]=value
            with self.assertRaisesRegex(ValueError,"KV growth"):
                validate_pressure_trace(trace,cases["tight_growth"],config,"paged")

    def test_formal_job_order_is_bounded_and_rotated(self):
        config,_,_,_=inputs()
        jobs=list(formal_jobs(config))
        self.assertEqual(len(jobs),18)
        self.assertEqual(len(set(jobs)),18)
        self.assertEqual([jobs[i][1] for i in (0,6,12)],["static_window","paged","static_window"])

    def test_formal_summary_recomputes_complete_matrix(self):
        config,pilot,base,cases,provenance,gate,reports=fixtures()
        summary=summarize_runs(reports,config,cases,provenance,pilot,base,gate,"c"*64)
        self.assertEqual(summary["measured_requests"],864)
        self.assertEqual(len(summary["variants"]),6)
        self.assertEqual([p["resident_requests"] for p in summary["comparisons"]],[[4,8],[4,8],[8,8]])

    def test_summary_rejects_missing_duplicate_and_changed_raw_metrics(self):
        config,pilot,base,cases,provenance,gate,reports=fixtures()
        for rows in (reports[:-1],reports+[reports[0]]):
            with self.assertRaises(ValueError):
                summarize_runs(rows,config,cases,provenance,pilot,base,gate,"c"*64)
        report=copy.deepcopy(reports[0][1])
        report["metrics"]["output_tokens_per_second"]+=1
        with self.assertRaisesRegex(ValueError,"recompute"):
            check_report(report,config,cases[report["case_id"]],provenance,pilot,base,probe=False)

    def test_gate_does_not_inherit_old_or_inexact_qualification(self):
        config,pilot,_,cases,provenance,gate,_=fixtures()
        check_gate(gate,config,cases,provenance,pilot)
        for key,value in (("exact_vectors",4159),("gpu_performance_execution_allowed",False),("evidence_class","cpu_allocator_replay")):
            broken=copy.deepcopy(gate)
            broken[key]=value
            with self.assertRaises(ValueError):
                check_gate(broken,config,cases,provenance,pilot)
        with self.assertRaises(ValueError):
            check_gate(gate,config,cases,{"changed":"source"},pilot)

    def test_timing_rejects_cpu_teacher_probes_compile_and_pool_changes(self):
        config,pilot,base,cases,provenance,_,reports=fixtures()
        original=reports[0][1]
        for mutation in ("evidence","probe","compile","pool"):
            report=copy.deepcopy(original)
            if mutation=="evidence": report["evidence_class"]="cpu_allocator_replay"
            if mutation=="probe": report["traces"][0]["finite_logits_checked"]=True
            if mutation=="compile": report["compile_counters"]["after"]["graphs"]=2
            if mutation=="pool": report["kv_tensor_bytes"]*=2
            with self.assertRaises(ValueError):
                check_report(report,config,cases[report["case_id"]],provenance,pilot,base,probe=False)

    def test_numerical_report_requires_every_request_and_exact_step(self):
        config,pilot,base,cases,provenance,_,reports=fixtures()
        report=copy.deepcopy(next(r for _,r in reports if r["policy"]=="paged"))
        case=cases[report["case_id"]]
        trace=report.pop("traces")[0]
        trace["finite_logits_checked"]=True
        report["evidence_class"]="functional_gpu_kv_pressure"
        report["numerical"]=dict(passed=True,finite_logits_all_steps=True,kv_poisoned_before_run=True,
            full_vocabulary_vectors=16*case["output_tokens"],saved_logits=dict(shape=[16,case["output_tokens"],151936],dtype="bfloat16"),
            counters={r["request_id"]:dict(steps=case["output_tokens"],exact_steps=case["output_tokens"],
                max_absolute_error=0,max_relative_rmse=0) for r in case["requests"]},functional_trace=trace)
        check_report(report,config,case,provenance,pilot,base,probe=True)
        report["numerical"]["counters"]["r00"]["exact_steps"]-=1
        with self.assertRaisesRegex(ValueError,"exact"):
            check_report(report,config,case,provenance,pilot,base,probe=True)


if __name__=="__main__":
    unittest.main()
