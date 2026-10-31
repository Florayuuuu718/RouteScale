"""H3 mapping/native-reference/paged/Graph/Prefix functional acceptance, no timing."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import traceback

from serving.backends.nanovllm import ROOT,digest,load_config,validate_environment,write_result
from serving.h3_prepare import DEFAULT_CONFIG
from serving.h2_runtime import assert_released,now

SOURCES=("model.py","moe.py","distributed_common.py","serving/artifact.py","serving/h3_prepare.py",
    "serving/h3_model.py","serving/backends/routescale_nanovllm.py","serving/verify_h3_routescale.py",
    "serving/h2_numerical_policy.py","serving/h2_prefix_probe.py","serving/h2_runtime.py","scripts/run_h3_routescale.py")


def load_inputs(path):
    config=json.loads(path.read_text())
    pin=load_config(ROOT/config["h1_config"])
    validate_environment(pin)
    prep_path=ROOT/config["results_dir"]/"preparation.json"
    prep=json.loads(prep_path.read_text())
    directory=ROOT/config["workload_dir"]
    if (prep["status"]!="pass" or prep["config_sha256"]!=digest(path)
            or prep["builder_sha256"]!=digest(ROOT/"serving/h3_prepare.py")
            or prep["tokenizer_sha256"]!=digest(directory/"tokenizer.json")
            or prep["workload_sha256"]!=digest(directory/"requests.json")):
        raise ValueError("Prepared offline tokenizer or workload changed")
    manifest=json.loads((ROOT/config["artifact_manifest"]).read_text())
    artifact=ROOT/manifest["artifact_path"]
    if digest(artifact)!=config["artifact_sha256"] or config["artifact_sha256"]!=manifest["artifact_sha256"]:
        raise ValueError("Original G1 inference artifact changed")
    import torch
    provenance=dict(config_sha256=digest(path),artifact_sha256=digest(artifact),
        artifact_state_dict_sha256=manifest["state_dict_sha256"],artifact_model_config=manifest["model_config"],
        original_manifest_sha256=digest(ROOT/config["artifact_manifest"]),preparation_sha256=digest(prep_path),
        tokenizer_sha256=prep["tokenizer_sha256"],workload_sha256=prep["workload_sha256"],
        upstream_commit=pin["upstream"]["commit"],
        upstream_source_sha256={str(p.relative_to(ROOT/pin["upstream"]["path"])):digest(p)
            for p in sorted((ROOT/pin["upstream"]["path"]).rglob("*.py"))},
        source_sha256={name:digest(ROOT/name) for name in SOURCES},
        environment=dict(python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            packages=json.loads(subprocess.check_output([sys.executable,"-m","pip","list","--format=json"],text=True))))
    return config,artifact,json.loads((directory/"tokenizer.json").read_text()),json.loads((directory/"requests.json").read_text())["cases"],provenance


def difference(left,right,tolerance,*,fp32=False):
    import torch
    a,b=left.float(),right.float()
    delta=a-b
    relative=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))
    return dict(exact=torch.equal(left,right),max_absolute_error=float(delta.abs().max()),relative_rmse=relative,
        passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all()) and torch.allclose(a,b,
            atol=tolerance["fp32_atol" if fp32 else "bf16_atol"],rtol=tolerance["fp32_rtol" if fp32 else "bf16_rtol"])
            and (fp32 or relative<=tolerance["max_relative_rmse"]))


def mapping_probe(artifact,cases,tolerance):
    import torch
    from serving.artifact import load_inference_artifact
    from serving.h3_model import PackedRouteScaleGPT,SegmentedSDPA
    native,_=load_inference_artifact(artifact,device="cuda",dtype=torch.float32)
    results=[]
    for case in cases:
        prompts=[r["prompt_token_ids"] for r in case["requests"]]
        lengths=[len(p) for p in prompts]
        bounds=[0]
        for n in lengths: bounds.append(bounds[-1]+n)
        context=SimpleNamespace(is_prefill=True,cu_seqlens_q=torch.tensor(bounds,dtype=torch.long))
        getter=lambda:context
        packed=PackedRouteScaleGPT(native,partial(SegmentedSDPA,get_context=getter),getter).eval()
        tokens=torch.tensor([i for p in prompts for i in p],device="cuda")
        positions=torch.tensor([i for p in prompts for i in range(len(p))],device="cuda")
        with torch.inference_mode():
            actual=packed.compute_logits(packed(tokens,positions))
            reference=torch.cat([native(torch.tensor([p],device="cuda"))[0][:,-1] for p in prompts])
        checks=[difference(a,b,tolerance,fp32=True) for a,b in zip(actual,reference)]
        results.append(dict(case_id=case["case_id"],checks=checks,passed=all(c["passed"] for c in checks)))
    return results


def execute_wave(engine,case,native,tolerance,*,expected_hit=0):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.utils.context import reset_context
    requests=case["requests"]
    for r in requests:
        engine.add_request(r["prompt_token_ids"],SamplingParams(temperature=.6,ignore_eos=True,max_tokens=r["output_tokens"]))
    sequences=list(engine.scheduler.waiting)
    generated=[list(r["prompt_token_ids"]) for r in requests]
    rows,checks,usage=[],[],[]
    for step in range(requests[0]["output_tokens"]):
        seqs,prefill=engine.scheduler.schedule()
        if seqs!=sequences or prefill!=(step==0):
            raise ValueError("Expected one bulk Prefill and full cohort Decode")
        if prefill and [s.num_cached_tokens for s in seqs]!=[expected_hit]*len(seqs):
            raise ValueError("Prefix hit count changed")
        usage.append(dict(step=step,cached_tokens=[s.num_cached_tokens+s.num_scheduled_tokens for s in seqs],
            used_blocks=len(engine.scheduler.block_manager.used_block_ids),tables=[s.block_table.copy() for s in seqs]))
        runner=engine.model_runner
        tokens,positions=runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
        try:
            with torch.inference_mode():
                logits=runner.run_model(tokens,positions,prefill).detach().cpu().contiguous()
                # Full recomputation in the SAME serving environment and with
                # the SAME BF16 weights; per-request references avoid padding.
                reference=torch.cat([native(torch.tensor([ids],device="cuda"))[0][:,-1].cpu() for ids in generated])
            if logits.shape!=(len(requests),50257) or logits.dtype!=torch.bfloat16 or not bool(torch.isfinite(logits).all()):
                raise ValueError("Paged full-vocabulary logits are nonfinite or changed shape/dtype")
            rows.append(logits)
            checks.extend(difference(a,b,tolerance) for a,b in zip(logits,reference))
        finally:
            reset_context()
        forced=[r["prompt_token_ids"][-1-step%len(r["prompt_token_ids"])] for r in requests]
        for ids,token in zip(generated,forced): ids.append(token)
        engine.scheduler.postprocess(seqs,forced,prefill)
    assert_released(engine)
    return torch.stack(rows),dict(vectors=len(checks),passed=all(c["passed"] for c in checks),
        max_absolute_error=max(c["max_absolute_error"] for c in checks),max_relative_rmse=max(c["relative_rmse"] for c in checks),
        exact_native_vectors=sum(c["exact"] for c in checks),expected_cached_tokens_per_request=expected_hit,
        usage=usage,released=True,finite=True)


def run_variant(path,variant):
    import torch
    from serving.artifact import load_inference_artifact
    from serving.h3_model import OfflineGPT2Tokenizer
    from serving.backends.routescale_nanovllm import create_engine,close_engine
    config,artifact,tokenizer_data,cases,provenance=load_inputs(path)
    output=ROOT/config["results_dir"]/f"{variant}.json"
    if variant not in config["variants"] or output.exists():
        raise ValueError("Fresh result and a bounded variant required")
    report=dict(schema_version=1,stage="H3",status="running",variant=variant,created_at_utc=now(),
        evidence_class="routescale_nanovllm_integration_correctness",provenance=provenance,cases={})
    engine,phase=None,"offline_tokenizer"
    try:
        tokenizer=OfflineGPT2Tokenizer(tokenizer_data)
        if any(tokenizer.encode(t["text"])!=t["token_ids"] or tokenizer.decode(t["token_ids"])!=t["decoded"] for t in tokenizer_data["tests"]):
            raise ValueError("Offline fast tokenizer differs from original tiktoken")
        report["tokenizer_checks"]=len(tokenizer_data["tests"])
        phase="fp32_mapping"
        report["mapping_checks"]=mapping_probe(artifact,cases,config["tolerances"])
        if not all(c["passed"] for c in report["mapping_checks"]):
            raise ValueError("FP32 architecture/weight mapping gate failed")
        phase="engine_initialization"
        torch._dynamo.config.recompile_limit=64
        torch._dynamo.config.fail_on_recompile_limit_hit=True
        torch._dynamo.config.suppress_errors=False
        torch.manual_seed(config["seed"])
        engine=create_engine(config["engine"],artifact,tokenizer,enforce_eager=variant!="graph_no_prefix",prefix_cache=variant=="eager_prefix")
        native,metadata=load_inference_artifact(artifact,device="cuda",dtype=torch.bfloat16)
        report.update(kv_tensor_bytes=engine.model_runner.kv_cache.numel()*2,
            tied_weights=engine.model_runner.model.gpt.transformer.wte.weight.data_ptr()==engine.model_runner.model.gpt.lm_head.weight.data_ptr(),
            parameter_count=sum(p.numel() for p in native.parameters()))
        directory=ROOT/"runs/validation/h3_routescale"/Path(config["results_dir"]).name
        directory.mkdir(parents=True,exist_ok=True)
        phase="native_bf16_full_output_validation"
        for case in cases:
            engine.reset_cache()
            engine.model_runner.kv_cache.fill_(float("nan"))
            for wave in (["cold","warm"] if case["case_id"]=="prefix_reuse" else ["cold"]):
                expected=256 if variant=="eager_prefix" and wave=="warm" else 0
                tensor,result=execute_wave(engine,case,native,config["tolerances"],expected_hit=expected)
                target=directory/f"{variant}_{case['case_id']}_{wave}.pt"
                if target.exists(): raise ValueError("Preserve existing numerical tensors")
                torch.save(tensor,target)
                result.update(path=str(target.relative_to(ROOT)),sha256=digest(target),shape=list(tensor.shape))
                report["cases"][f"{case['case_id']}_{wave}"]=result
                write_result(report,output)
        if not all(c["passed"] for c in report["cases"].values()):
            raise ValueError("Native BF16 versus paged reference gate failed; preserve this scope")
        phase="public_generation_smoke"
        from nanovllm import SamplingParams
        engine.reset_cache()
        report["smoke"]=engine.generate(["Once upon a time","A little fox"],SamplingParams(temperature=.6,ignore_eos=True,max_tokens=8),use_tqdm=False)
        if any(len(r["token_ids"])!=8 or not r["text"] for r in report["smoke"]):
            raise ValueError("Public text generation smoke failed")
        assert_released(engine)
        report["free_bytes_after_run"]=torch.cuda.mem_get_info()[0]
        if report["free_bytes_after_run"]<config["minimum_free_bytes"]: raise ValueError("Insufficient headroom")
        report["status"]="pass"
    except BaseException:
        report.update(status="fail",failure_phase=phase,traceback=traceback.format_exc())
        raise
    finally:
        try:
            if engine is not None: close_engine(engine)
        finally:
            report["finished_at_utc"]=now()
            write_result(report,output)


def compare(path,check_only=False):
    import torch
    config,_,_,cases,provenance=load_inputs(path)
    directory=ROOT/config["results_dir"]
    reports={v:json.loads((directory/f"{v}.json").read_text()) for v in config["variants"]}
    expected={f"{c['case_id']}_{w}" for c in cases for w in (["cold","warm"] if c["case_id"]=="prefix_reuse" else ["cold"])}
    tensors={}
    for variant,r in reports.items():
        if (r["status"]!="pass" or r["provenance"]!=provenance or set(r["cases"])!=expected
                or not r["tied_weights"] or r["kv_tensor_bytes"]!=67108864):
            raise ValueError("Incomplete/changed integration source, numerical gate or KV pool")
        tensors[variant]={}
        for key,metadata in r["cases"].items():
            if not metadata["passed"] or not metadata["released"] or not metadata["finite"] or digest(ROOT/metadata["path"])!=metadata["sha256"]:
                raise ValueError("Failed/stale case or saved logits")
            tensor=torch.load(ROOT/metadata["path"],weights_only=True,mmap=True,map_location="cpu")
            if tensor.dtype!=torch.bfloat16 or list(tensor.shape)!=metadata["shape"] or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Saved BF16 logits malformed")
            tensors[variant][key]=tensor
    graph_exact=0
    for key,left in tensors["eager_no_prefix"].items():
        right=tensors["graph_no_prefix"][key]
        if not torch.equal(left,right): raise ValueError(f"Eager/Graph exact gate failed: {key}")
        graph_exact+=left.shape[0]*left.shape[1]
    historical={}
    for name in ("h2_graph_b8","h2_corrected_pilot","h2_scheduling_tile512","h2_kv_pressure"):
        old=ROOT/"results/h_nanovllm"/name
        hashes=json.loads((old/"audit.json").read_text())["raw_sha256"]
        if any(digest(old/relative)!=sha for relative,sha in hashes.items()): raise ValueError("Historical raw changed")
        historical[name]=len(hashes)
    report=dict(schema_version=1,stage="H3",status="pass",evidence_class="routescale_integration_correctness_gate",
        created_at_utc=now(),provenance=provenance,independent_variant_processes=3,
        native_bf16_vectors=sum(c["vectors"] for r in reports.values() for c in r["cases"].values()),
        eager_graph_exact_vectors=graph_exact,historical_formal_raw_hashes_match=historical,
        report_sha256={f"{v}.json":digest(directory/f"{v}.json") for v in reports},
        performance_execution_allowed=False,limitations=config["limitations"])
    if check_only:
        saved=json.loads((directory/"gate.json").read_text())
        if saved["provenance"]!=provenance or saved["report_sha256"]!=report["report_sha256"]: raise ValueError("Saved gate changed")
    else:
        if (directory/"gate.json").exists(): raise ValueError("Use --check-only to preserve the gate")
        write_result(report,directory/"gate.json")
    print(f"H3 integration gate: pass, {report['native_bf16_vectors']} native vectors, {graph_exact} exact Graph vectors",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--variant")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.variant: run_variant(args.config,args.variant)
    else: compare(args.config,args.check_only)
