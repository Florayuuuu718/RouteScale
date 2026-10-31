import copy
import json
from pathlib import Path
import unittest

from serving.h2_kv_diag_stats import (
    LAYOUTS,layout_metrics,layout_order,layout_pages,paired_order,summarize,validate_report,wall_components,
)
from tests.test_h2_kv_pressure import inputs,simulate

ROOT=Path(__file__).resolve().parents[1]


def stream_fixture(layout):
    return dict(evidence_class="teacher_forced_cuda_stream_probe",layout=layout,pages=layout_pages(layout),
        released=True,sampler_executed=False,steps=[dict(is_prefill=i==0,cached_tokens=128+i,
            stream_ms=1+i/100,host_model_ms=2+i/100) for i in range(64)])


def fixtures(run=1):
    pressure,_,_,cases=inputs()
    config=json.loads((ROOT/"config/serving/h2_kv_diagnostic.json").read_text())
    case=cases["ample_short"]
    provenance={"test_only":"not GPU measurements"}
    report=dict(status="pass",evidence_class="gpu_kv_negative_control_diagnostic",provenance=provenance,
        run_id=run,seed=20261023,kv_tensor_bytes=939524096,free_bytes_after_run=2**30,
        compiler_settings=dict(recompile_limit=64,fail_on_recompile_limit_hit=True,suppress_errors=False),
        compile_counters={name:dict(before={"graphs":21},after={"graphs":21}) for name in ("paired","layouts")},
        paired_blocks=[],layout_timings=[],layout_numerics={})
    for block in range(4):
        rows=[]
        for policy in paired_order(run,block):
            trace=simulate(case,pressure,policy)
            rows.append(dict(policy=policy,seed=20261023+block,trace=trace,components=wall_components(trace)))
        report["paired_blocks"].append(dict(block_id=block,traces=rows))
    for round_id in range(3):
        for layout in layout_order(run,round_id):
            trace=stream_fixture(layout)
            report["layout_timings"].append(dict(round_id=round_id,trace=trace,metrics=layout_metrics(trace)))
    for layout in LAYOUTS:
        report["layout_numerics"][layout]=dict(vectors=512,exact_vectors=0 if layout=="stride4" else 512,
            finite=True,kv_poisoned=True,released=True,pages=layout_pages(layout),shape=[64,8,151936],
            dtype="bfloat16",free_pages_remain_nan=True)
    return config,pressure,case,provenance,report


class KVDiagnosticTests(unittest.TestCase):
    def test_paired_order_balances_positions_and_reverses_outer_policy(self):
        for run in (1,2,3):
            for block in range(4):
                order=paired_order(run,block)
                self.assertEqual([order.count(p) for p in ("static_window","paged")],[2,2])
                self.assertEqual(sum(i for i,p in enumerate(order) if p=="paged"),3)
                self.assertEqual(order,tuple(reversed(order)))
        self.assertNotEqual(paired_order(1,0),paired_order(1,1))

    def test_layout_latin_square_covers_each_position(self):
        for run in (1,2,3):
            for position in range(3):
                self.assertEqual({layout_order(run,i)[position] for i in range(3)},set(LAYOUTS))

    def test_controlled_pages_are_unique_real_pool_addresses(self):
        self.assertEqual(layout_pages("stride4"),[0,4,8,12,16,20,24,28])
        self.assertEqual(layout_pages("dense0"),list(range(8)))
        self.assertEqual(layout_pages("dense8"),list(range(8,16)))
        for layout in LAYOUTS:
            pages=layout_pages(layout)
            self.assertEqual(len(set(pages)),8)
            self.assertTrue(all(0<=p<32 for p in pages))
        for callback,args in ((paired_order,(4,0)),(layout_order,(1,3)),(layout_pages,("unknown",))):
            with self.assertRaises(ValueError): callback(*args)

    def test_wall_accounting_partitions_host_visible_trace(self):
        pressure,_,_,cases=inputs()
        trace=simulate(cases["ample_short"],pressure,"paged")
        components=wall_components(trace)
        self.assertEqual(components["prefill_visible_ms"],20)
        self.assertAlmostEqual(sum(components.values()),trace["wall_seconds"]*1000)
        trace["steps"][0]["model_start_ms"]=11
        with self.assertRaises(ValueError): wall_components(trace)

    def test_stream_metrics_are_not_sampled_serving_throughput(self):
        trace=stream_fixture("dense0")
        metrics=layout_metrics(trace)
        self.assertAlmostEqual(metrics["decode_stream_mean_ms"],1.32)
        self.assertNotIn("output_tokens_per_second",metrics)
        trace["sampler_executed"]=True
        with self.assertRaises(ValueError): layout_metrics(trace)

    def test_stream_timing_rejects_missing_events_wrong_pages_and_cache_positions(self):
        for kind in ("missing","pages","tokens","event"):
            trace=stream_fixture("dense0")
            if kind=="missing": trace["steps"].pop()
            if kind=="pages": trace["pages"]=[0]*8
            if kind=="tokens": trace["steps"][1]["cached_tokens"]=130
            if kind=="event": trace["steps"][0]["stream_ms"]=0
            with self.assertRaises(ValueError): layout_metrics(trace)

    def test_full_cpu_fixture_replays_and_aggregates_three_runs(self):
        config,pressure,case,provenance,_=fixtures()
        reports=[fixtures(i)[-1] for i in (1,2,3)]
        result=summarize(reports,config,case,pressure,provenance)
        self.assertEqual(result["paired_sampled_traces"],48)
        self.assertEqual(result["median_of_process_paired_ratios"],1)
        self.assertEqual(result["full_vocabulary_layout_comparisons"],3072)
        with self.assertRaises(ValueError): summarize(reports[:2],config,case,pressure,provenance)

    def test_diagnostic_requires_exact_poisoned_numerical_gate_and_no_compile(self):
        config,pressure,case,provenance,original=fixtures()
        for kind in ("exact","poison","compile","empty","isolation"):
            report=copy.deepcopy(original)
            if kind=="exact": report["layout_numerics"]["dense8"]["exact_vectors"]=511
            if kind=="poison": report["layout_numerics"]["dense8"]["kv_poisoned"]=False
            if kind=="compile": report["compile_counters"]["paired"]["after"]["graphs"]=22
            if kind=="empty": report["compile_counters"]={}
            if kind=="isolation": report["layout_numerics"]["dense8"]["free_pages_remain_nan"]=False
            with self.assertRaises(ValueError): validate_report(report,config,case,pressure,provenance)

    def test_changed_seed_order_sampled_tokens_and_provenance_fail_closed(self):
        config,pressure,case,provenance,original=fixtures()
        for kind in ("seed","order","hash","source"):
            report=copy.deepcopy(original)
            row=report["paired_blocks"][0]["traces"][0]
            if kind=="seed": row["seed"]+=1
            if kind=="order": row["policy"]="paged"
            if kind=="hash": row["trace"]["requests"][0]["output_token_ids_sha256"]="b"*64
            if kind=="source": report["provenance"]={}
            with self.assertRaises(ValueError): validate_report(report,config,case,pressure,provenance)

    def test_saved_metrics_must_recompute_from_events_and_trace_intervals(self):
        config,pressure,case,provenance,original=fixtures()
        for kind in ("event_metric","wall_metric"):
            report=copy.deepcopy(original)
            if kind=="event_metric": report["layout_timings"][0]["metrics"]["decode_stream_mean_ms"]+=1
            else: report["paired_blocks"][0]["traces"][0]["components"]["decode_visible_ms"]+=1
            with self.assertRaises(ValueError): validate_report(report,config,case,pressure,provenance)


if __name__=="__main__":
    unittest.main()
