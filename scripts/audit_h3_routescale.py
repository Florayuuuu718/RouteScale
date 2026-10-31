"""Fresh native-reference/weight/API/lifetime audit of the frozen H3 bridge."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.backends.nanovllm import digest,write_result
from serving.verify_h3_routescale import DEFAULT_CONFIG,difference,load_inputs
from serving.h2_runtime import assert_released,now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    import torch
    from serving.artifact import load_inference_artifact
    from serving.h3_model import OfflineGPT2Tokenizer
    from serving.backends.routescale_nanovllm import create_engine,close_engine
    config,artifact,data,cases,provenance=load_inputs(args.config)
    directory=ROOT/config["results_dir"]
    output=directory/"verification.json"
    if output.exists() and not args.check_only:
        raise ValueError("Preserve this verification; use --check-only")
    gate_path=directory/"gate.json"
    gate=json.loads(gate_path.read_text())
    if gate["status"]!="pass" or gate["provenance"]!=provenance:
        raise ValueError("Original H3 qualification missing or stale")
    for relative,sha in gate["report_sha256"].items():
        if digest(directory/relative)!=sha: raise ValueError("H3 numerical report changed")
    reports={v:json.loads((directory/f"{v}.json").read_text()) for v in config["variants"]}
    references=ROOT/"runs/validation/h3_routescale"/directory.name/"independent_native"
    old=json.loads(output.read_text()) if args.check_only else None
    native=None
    if not args.check_only:
        native,_=load_inference_artifact(artifact,device="cuda",dtype=torch.bfloat16)
    reference_metadata,comparisons={},{}
    for case in cases:
        name=case["case_id"]
        path=references/f"{name}.pt"
        count=case["requests"][0]["output_tokens"]
        if args.check_only:
            metadata=old["native_reference_tensors"][name]
            if digest(path)!=metadata["sha256"]: raise ValueError("Native reference changed")
            reference=torch.load(path,weights_only=True,mmap=True,map_location="cpu")
        else:
            if path.exists(): raise ValueError("Preserve existing independent native tensors")
            # Per-request full recomputation, separately from the serving loop.
            request_rows=[]
            with torch.inference_mode():
                for request in case["requests"]:
                    ids=list(request["prompt_token_ids"])
                    rows=[]
                    for step in range(count):
                        rows.append(native(torch.tensor([ids],device="cuda"))[0][0,-1].cpu())
                        ids.append(request["prompt_token_ids"][-1-step%len(request["prompt_token_ids"])])
                    request_rows.append(torch.stack(rows))
            reference=torch.stack(request_rows,dim=1)
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(reference,path)
        reference_metadata[name]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
        if reference.shape!=(count,len(case["requests"]),50257) or reference.dtype!=torch.bfloat16:
            raise ValueError("Independent native shape/dtype changed")
        for variant,report in reports.items():
            for wave in (["cold","warm"] if name=="prefix_reuse" else ["cold"]):
                metadata=report["cases"][f"{name}_{wave}"]
                if digest(ROOT/metadata["path"])!=metadata["sha256"]: raise ValueError("Target logits changed")
                actual=torch.load(ROOT/metadata["path"],weights_only=True,mmap=True,map_location="cpu")
                if actual.shape!=reference.shape or actual.dtype!=reference.dtype: raise ValueError("Target shape/dtype changed")
                checks=[difference(a,b,config["tolerances"]) for step_a,step_b in zip(actual,reference) for a,b in zip(step_a,step_b)]
                if not all(c["passed"] for c in checks): raise ValueError("Independent native BF16 comparison failed")
                comparisons[f"{variant}_{name}_{wave}"]=dict(vectors=len(checks),
                    max_absolute_error=max(c["max_absolute_error"] for c in checks),
                    max_relative_rmse=max(c["relative_rmse"] for c in checks))
    if sum(v["vectors"] for v in comparisons.values())!=792:
        raise ValueError("Incomplete independent native vector coverage")
    lifetime,weights,guard_count=None,0,0
    if not args.check_only:
        torch._dynamo.config.recompile_limit=64
        torch._dynamo.config.fail_on_recompile_limit_hit=True
        torch._dynamo.config.suppress_errors=False
        before=torch.cuda.memory_allocated()
        engine=create_engine(config["engine"],artifact,OfflineGPT2Tokenizer(data))
        try:
            source=native.state_dict()
            target=engine.model_runner.model.gpt.state_dict()
            if set(source)!=set(target): raise ValueError("Weight names changed")
            for name in source:
                if source[name].dtype!=torch.bfloat16 or not torch.equal(source[name],target[name]):
                    raise ValueError(f"Runtime BF16 weight mismatch: {name}")
                weights+=1
            del source,target
            if engine.model_runner.model.gpt.lm_head.weight is not engine.model_runner.model.gpt.transformer.wte.weight:
                raise ValueError("Embedding/LM head tying broken")
            invalid=[([],8,.6),([50257],8,.6),([1]*512,2,.6),([1],8,0),([1],8,float("nan")),([1],1.5,.6)]
            for tokens,count,temp in invalid:
                try: engine.add_request(tokens,SimpleNamespace(max_tokens=count,temperature=temp,ignore_eos=True))
                except ValueError: guard_count+=1
                else: raise ValueError("Invalid public request was accepted")
                assert_released(engine)
            from nanovllm import SamplingParams
            # Exactly 512 cached positions and one final sample are valid.
            boundary=engine.generate([[1]*512],SamplingParams(temperature=.6,max_tokens=1,ignore_eos=True),use_tqdm=False)
            if len(boundary[0]["token_ids"])!=1: raise ValueError("Public final-cache-position boundary failed")
            assert_released(engine)
            runner_ref=weakref.ref(engine.model_runner)
            model_ref=weakref.ref(engine.model_runner.model)
            kv_ref=weakref.ref(engine.model_runner.kv_cache)
        finally:
            close_engine(engine)
        gc.collect()
        torch.cuda.synchronize()
        lifetime=dict(runner_collected=runner_ref() is None,model_collected=model_ref() is None,
            kv_tensor_collected=kv_ref() is None,allocated_bytes_before=before,allocated_bytes_after=torch.cuda.memory_allocated())
        if not all(lifetime[k] for k in ("runner_collected","model_collected","kv_tensor_collected")):
            raise ValueError("Closed engine retains model/runner/KV tensors")
    report=dict(schema_version=1,stage="H3",status="pass",evidence_class="independent_routescale_bridge_verification",
        created_at_utc=now(),verifier_sha256=digest(Path(__file__)),gate_sha256=digest(gate_path),provenance=provenance,
        native_bf16_vectors_recompared=792,native_reference_tensors=reference_metadata,comparisons=comparisons,
        runtime_bf16_weight_tensors_equal=weights,invalid_public_requests_rejected=guard_count,
        public_512_cached_token_boundary_passed=True,closed_engine_lifetime=lifetime,
        performance_execution_allowed=False,limitations=config["limitations"])
    if args.check_only:
        for key in ("provenance","gate_sha256","native_reference_tensors","comparisons"):
            if old[key]!=report[key]: raise ValueError(f"Saved independent verification changed: {key}")
    else:
        write_result(report,output)
    print("H3 independent audit: pass, 792 native vectors; weights/API/lifetime checked on initial audit",flush=True)


if __name__=="__main__":
    main()
