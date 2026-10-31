"""Read historical evidence; run opt-in paired/layout diagnostics without fixes."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import time
import traceback

from serving.backends.nanovllm import ROOT,digest,write_result
from serving.h2_kv_diag_stats import (
    LAYOUTS,layout_metrics,layout_order,layout_pages,paired_order,summarize,wall_components,
)
from serving.h2_kv_pressure import create_engine,execute_trace,load_inputs as load_pressure,reset_manager,validate_pressure_trace
from serving.h2_runtime import assert_released,compile_counts,now
from serving.h2_scheduling import close_scheduling_engine
from scripts.summarize_h2_kv_pressure import check_gate

DEFAULT_CONFIG=ROOT/"config/serving/h2_kv_diagnostic.json"
SOURCES=("serving/h2_kv_diagnostic.py","serving/h2_kv_diag_stats.py","scripts/run_h2_kv_diagnostic.py")


def load_inputs(path=DEFAULT_CONFIG):
    config=json.loads(path.read_text())
    if (config["schema_version"]!=1 or config["case_id"]!="ample_short" or config["independent_runs"]!=3
            or config["paired_blocks"]!=4 or config["layout_rounds"]!=3
            or config["layouts"]!=list(LAYOUTS) or config["output_tokens"]!=64):
        raise ValueError("Use a new scope for a different diagnostic")
    pressure,inputs,cases,previous=load_pressure(ROOT/config["base_pressure_config"])
    directory=ROOT/pressure["results_dir"]
    audit_path,gate_path=directory/"audit.json",directory/"preflight.json"
    if digest(audit_path)!=config["base_audit_sha256"] or digest(gate_path)!=config["base_gate_sha256"]:
        raise ValueError("Historical pressure gate/audit changed")
    old=json.loads(audit_path.read_text())
    if old["status"]!="pass" or old["provenance"]!=previous:
        raise ValueError("Qualified source/model/environment changed")
    check_gate(json.loads(gate_path.read_text()),pressure,cases,previous,inputs[2])
    for name,checksum in old["raw_sha256"].items():
        if digest(directory/name)!=checksum:
            raise ValueError("Never diagnose against changed historical raw evidence")
    provenance={**previous,"diagnostic_config_sha256":digest(path),
        "diagnostic_source_sha256":{name:digest(ROOT/name) for name in SOURCES},
        "diagnostic_base_audit_sha256":digest(audit_path),"diagnostic_base_gate_sha256":digest(gate_path)}
    return config,pressure,inputs,cases["ample_short"],provenance


def telemetry():
    query="clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu"
    result=subprocess.run(["nvidia-smi",f"--query-gpu={query}","--format=csv,noheader,nounits"],
        capture_output=True,text=True,timeout=10)
    return dict(at_utc=now(),query=query,returncode=result.returncode,
        values=result.stdout.strip(),error=result.stderr.strip(),sampling="between blocks, not during traces")


def layout_wave(llm,case,layout,*,capture=False,reference=None,timed=False):
    """One allocator, one 8-request cohort, controlled pages and teacher tokens.

    Events bracket prepare-completed ModelRunner.run_model, including graph buffer
    copies and LM head, excluding sampler. Stream elapsed time can include idle
    gaps caused by host dispatch; it is NOT attention-kernel-only time.
    """
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context
    if capture==timed:
        raise ValueError("Choose numerical capture OR CUDA stream timing")
    reset_manager(llm,"paged")
    manager=llm.scheduler.block_manager
    pages=layout_pages(layout)
    manager.free_block_ids=deque(pages+[p for p in range(32) if p not in pages])
    pool=llm.model_runner.kv_cache
    if capture:
        pool.fill_(float("nan"))
    requests=case["requests"][:8]
    for request in requests:
        llm.add_request(request["prompt_token_ids"],SamplingParams(temperature=.6,ignore_eos=True,max_tokens=64))
    expected=list(llm.scheduler.waiting)
    steps,rows,exact=[],[],0
    for index in range(64):
        seqs,prefill=llm.scheduler.schedule()
        if seqs!=expected or prefill!=(index==0) or [s.block_table for s in seqs]!=[[p] for p in pages]:
            raise AssertionError("Controlled layout or complete-cohort scheduling changed")
        if manager.used_block_ids!=set(pages) or any(b.ref_count!=int(b.block_id in pages) for b in manager.blocks):
            raise AssertionError("Layout allocation/refcounts changed")
        runner=llm.model_runner
        tokens,positions=runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
        begin,end=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) if timed else (None,None)
        try:
            started=time.perf_counter()
            if timed:
                begin.record()
            with torch.inference_mode():
                logits=runner.run_model(tokens,positions,prefill)
            if timed:
                end.record()
                end.synchronize()
                host_ms=(time.perf_counter()-started)*1000
                steps.append(dict(is_prefill=prefill,cached_tokens=128+index,
                    stream_ms=begin.elapsed_time(end),host_model_ms=host_ms))
            else:
                cpu=logits.detach().cpu().contiguous()
                if cpu.dtype!=torch.bfloat16 or cpu.shape!=(8,151936) or not bool(torch.isfinite(cpu).all()):
                    raise AssertionError("Nonfinite or changed full-vocabulary layout logits")
                rows.append(cpu)
                if reference is not None:
                    exact+=int((cpu==reference[index]).all(dim=1).sum())
        finally:
            reset_context()
        forced=[s.prompt_token_ids[-1-index] for s in seqs]
        llm.scheduler.postprocess(seqs,forced,prefill)
    assert_released(llm)
    if capture:
        for page in range(32):
            good=torch.isfinite(pool[:,:,page,:191]).all() if page in pages else torch.isnan(pool[:,:,page]).all()
            if not bool(good):
                raise AssertionError("Layout writes did not isolate valid KV from free pages")
        return torch.stack(rows),dict(vectors=512,exact_vectors=exact,finite=True,kv_poisoned=True,
            released=True,pages=pages,shape=[64,8,151936],dtype="bfloat16",free_pages_remain_nan=True)
    trace=dict(evidence_class="teacher_forced_cuda_stream_probe",layout=layout,pages=pages,
        sampler_executed=False,released=True,steps=steps)
    return trace,layout_metrics(trace)


def run(path,run_id,result_path):
    import torch
    config,pressure,inputs,case,provenance=load_inputs(path)
    if run_id not in (1,2,3) or result_path.exists():
        raise ValueError("Fresh output and a bounded independent run are required")
    report=dict(schema_version=1,stage="H2",status="running",run_id=run_id,
        evidence_class="gpu_kv_negative_control_diagnostic",created_at_utc=now(),provenance=provenance,
        seed=inputs[3]["seed"],paired_blocks=[],layout_numerics={},layout_timings=[],compile_counters={})
    llm,phase=None,"engine_initialization"
    try:
        torch.manual_seed(report["seed"])
        llm,report["initialization"]=create_engine(inputs,case)
        report["kv_tensor_bytes"]=llm.model_runner.kv_cache.numel()*llm.model_runner.kv_cache.element_size()
        for policy in ("static_window","paged"):
            execute_trace(llm,case,pressure,inputs[3],policy)
        phase="balanced_full_path_pairs"
        before=compile_counts()
        for block_id in range(4):
            block=dict(block_id=block_id,telemetry_before=telemetry(),traces=[])
            for policy in paired_order(run_id,block_id):
                seed=report["seed"]+block_id
                torch.manual_seed(seed)
                trace=execute_trace(llm,case,pressure,inputs[3],policy)
                validate_pressure_trace(trace,case,pressure,policy)
                block["traces"].append(dict(policy=policy,seed=seed,trace=trace,components=wall_components(trace)))
            block["telemetry_after"]=telemetry()
            report["paired_blocks"].append(block)
            write_result(report,result_path)
            print(f"diagnostic run{run_id}: paired block {block_id+1}/4",flush=True)
        report["compile_counters"]["paired"]=dict(before=before,after=compile_counts())
        phase="controlled_page_layout_numerics"
        directory=ROOT/"runs/validation/h2_kv_diagnostic"/Path(config["results_dir"]).name
        directory.mkdir(parents=True,exist_ok=True)
        baseline=None
        for layout in LAYOUTS:
            tensor,metadata=layout_wave(llm,case,layout,capture=True,reference=baseline)
            target=directory/f"run{run_id}_{layout}.pt"
            if target.exists():
                raise ValueError("Preserve existing full numerical layout tensors")
            torch.save(tensor,target)
            metadata.update(path=str(target.relative_to(ROOT)),sha256=digest(target))
            report["layout_numerics"][layout]=metadata
            if layout=="stride4":
                baseline=tensor
            elif metadata["exact_vectors"]!=512:
                raise ValueError("New layout requires fully exact logits before CUDA timing")
        del baseline,tensor
        phase="controlled_page_layout_stream_timings"
        before=compile_counts()
        for round_id in range(3):
            for layout in layout_order(run_id,round_id):
                trace,metrics=layout_wave(llm,case,layout,timed=True)
                report["layout_timings"].append(dict(round_id=round_id,trace=trace,metrics=metrics))
        report["compile_counters"]["layouts"]=dict(before=before,after=compile_counts())
        if any(row["before"]!=row["after"] for row in report["compile_counters"].values()):
            raise ValueError("Compilation changed inside a measurement phase")
        assert_released(llm)
        report["free_bytes_after_run"]=torch.cuda.mem_get_info()[0]
        report["compiler_settings"]=dict(recompile_limit=torch._dynamo.config.recompile_limit,
            fail_on_recompile_limit_hit=torch._dynamo.config.fail_on_recompile_limit_hit,suppress_errors=torch._dynamo.config.suppress_errors)
        report["status"]="pass"
    except BaseException:
        report.update(status="fail",failure_phase=phase,traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                close_scheduling_engine(llm)
        finally:
            report["finished_at_utc"]=now()
            write_result(report,result_path)


def audit(path,check_only=False):
    import torch
    config,pressure,inputs,case,provenance=load_inputs(path)
    directory=ROOT/config["results_dir"]
    paths=[directory/"raw"/f"run{i}.json" for i in (1,2,3)]
    reports=[json.loads(p.read_text()) for p in paths]
    summary=summarize(reports,config,case,pressure,provenance)
    exact=0
    for report in reports:
        tensors={}
        for layout,metadata in report["layout_numerics"].items():
            target=ROOT/metadata["path"]
            if digest(target)!=metadata["sha256"]:
                raise ValueError("Full-vocabulary diagnostic tensor checksum changed")
            tensor=torch.load(target,weights_only=True,mmap=True,map_location="cpu")
            if tensor.shape!=(64,8,151936) or tensor.dtype!=torch.bfloat16:
                raise ValueError("Saved full-vocabulary shape or dtype changed")
            tensors[layout]=tensor
        for layout in ("dense0","dense8"):
            for left,right in zip(tensors["stride4"],tensors[layout]):
                if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                    raise ValueError("Saved layout logits contain nonfinite values")
                exact+=int((left==right).all(dim=1).sum())
    if exact!=3072:
        raise ValueError("Saved layout vectors did not independently compare exact")
    old_directory=ROOT/pressure["results_dir"]
    old=json.loads((old_directory/"audit.json").read_text())
    historical={}
    for name in ("h2_graph_b8","h2_corrected_pilot","h2_scheduling_tile512","h2_kv_pressure"):
        old_root=ROOT/"results/h_nanovllm"/name
        hashes=json.loads((old_root/"audit.json").read_text())["raw_sha256"]
        for relative,checksum in hashes.items():
            if digest(old_root/relative)!=checksum:
                raise ValueError("Historical raw evidence changed")
        historical[name]=len(hashes)
    if subprocess.check_output(["git","diff","--","docs/RouteScale_实施教程大纲.md"],cwd=ROOT,text=True):
        raise ValueError("Unexpected implementation-outline change")
    record=dict(schema_version=1,stage="H2",status="pass",evidence_class="kv_diagnostic_integrity_audit",
        created_at_utc=now(),provenance=provenance,full_vocabulary_vectors_recompared=exact,
        historical_formal_raw_hashes_match=historical,outline_git_diff_empty=True,
        raw_sha256={str(p.relative_to(directory)):digest(p) for p in paths},
        original_ample_throughput_ratio=0.9310083884434223,
        limitations=config["limitations"])
    if check_only:
        if json.loads((directory/"summary.json").read_text())!=summary:
            raise ValueError("Saved diagnostic summary did not recompute")
        saved=json.loads((directory/"audit.json").read_text())
        if saved["raw_sha256"]!=record["raw_sha256"] or saved["provenance"]!=provenance:
            raise ValueError("Saved diagnostic audit bindings changed")
    else:
        if (directory/"audit.json").exists() or (directory/"summary.json").exists():
            raise ValueError("Use --check-only; never replace completed diagnostic evidence")
        write_result(summary,directory/"summary.json")
        write_result(record,directory/"audit.json")
    print(f"KV diagnostic audit pass: 48 full-path traces, 27 stream probes, {exact} exact vectors",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--run-id",type=int)
    parser.add_argument("--result",type=Path)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.run_id is not None:
        if args.result is None:
            parser.error("--result is required with --run-id")
        run(args.config,args.run_id,args.result)
    else:
        audit(args.config,args.check_only)


if __name__=="__main__":
    main()
