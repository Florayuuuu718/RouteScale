"""Per-rank collective, fixed-token TP qualification, timing or separate trace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import read,sha,write
from serving.tp_preparation import load_plan


def fixture(c):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/c['model_path'],local_files_only=True)
    pool=tokenizer.encode(' A little girl walked through the green forest. She found a small house and told her friends a happy story.',add_special_tokens=False)
    rows=[]
    for name,batch,length in [('b1_p128',1,128),('b3_p512_shared',3,512)]:
        phases={}
        for phase in ('cold','warm'):
            requests=[]
            for index in range(batch):
                ids=(pool*((length+len(pool)-1)//len(pool)))[:length]
                suffix=tokenizer.encode(f' Story number {index}: the child smiled.',add_special_tokens=False)
                ids[-len(suffix):]=suffix
                requests.append(dict(request_id=f'{name}_{phase}_{index}',input_ids=ids,
                    continuation_ids=(pool[index:]+pool[:index])[:c['qualification_output_tokens']]))
            phases[phase]=requests
        rows.append(dict(case_id=name,waves=phases))
    return dict(seed=c['seed'],cases=rows)


def collective(directory,world,rank):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(seconds=60))
    try:
        tensor=torch.full((4096,),rank+1,dtype=torch.float32,device=f'cuda:{rank}')
        dist.all_reduce(tensor);torch.cuda.synchronize()
        expected=world*(world+1)/2
        passed=bool(torch.all(tensor==expected))
        write(directory/f'collective_tp{world}_rank{rank}.json',dict(passed=passed,rank=rank,world=world,
            backend=dist.get_backend(),nccl_version=list(torch.cuda.nccl.version()),GPU=torch.cuda.get_device_name(rank),
            GPU_uuid=str(getattr(torch.cuda.get_device_properties(rank),'uuid','unavailable')),
            allocated_bytes=torch.cuda.memory_allocated(rank),reserved_bytes=torch.cuda.memory_reserved(rank),
            expected=expected,elements=tensor.numel()))
        if not passed:raise ValueError('NCCL collective values differ')
    finally:dist.destroy_process_group()


def reference(directory,c):
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(ROOT/c['model_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='eager').float().cuda().eval()
    values={}
    with torch.inference_mode():
        for case in read(directory/'workload.json')['cases']:
            for phase in ('cold','warm'):
                for row in case['waves'][phase]:
                    inputs=torch.tensor([row['input_ids']],device='cuda');past=None;parts=[]
                    for forced in row['continuation_ids']:
                        output=model(input_ids=inputs,past_key_values=past,use_cache=True,logits_to_keep=1)
                        parts.append(output.logits[0,-1].cpu());past=output.past_key_values
                        inputs=torch.tensor([[forced]],device='cuda')
                    values[row['request_id']]=torch.stack(parts)
    target=directory/'hf_reference.pt';torch.save(values,target)
    write(directory/'hf_reference.json',dict(passed=True,policy='BF16_checkpoint_promoted_FP32_eager',
        tensor_sha256=sha(target),vectors=sum(v.shape[0] for v in values.values())))


def replay(directory,c,framework,variant,world,rank,mode,label):
    import torch
    settings=c['frameworks'][framework]
    sys.path.insert(0,str(ROOT/settings['checkout']/('python' if framework=='mini' else '')))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(c['seed']);torch.set_num_threads(4)
    numeric=mode=='qualify'
    variants=dict(naive_full=dict(cache='naive',chunk=4096,overlap=False,graph_bs=[]),
        radix_full=dict(cache='radix',chunk=4096,overlap=False,graph_bs=[]),
        radix_chunk256=dict(cache='radix',chunk=256,overlap=False,graph_bs=[]),
        radix_chunk128=dict(cache='radix',chunk=128,overlap=False,graph_bs=[]),
        radix_overlap=dict(cache='radix',chunk=256,overlap=True,graph_bs=[]),
        radix_graph124=dict(cache='radix',chunk=4096,overlap=False,graph_bs=[1,2,4]),
        radix_graph13=dict(cache='radix',chunk=4096,overlap=False,graph_bs=[1,3]))
    cfg=dict(model_path=str(ROOT/c['model_path']),engine=dict(attention_backend='fi',page_size=1,num_pages=4096,
        max_running_requests=4,max_seq_len=1024),variants=variants)
    if framework=='mini':
        from serving.tp_mini_runtime import create
    else:
        from serving.backends.nanovllm import load_config,validate_environment
        validate_environment(load_config())
        from serving.tp_nano_runtime import create
    runtime=create(cfg,variant,world,rank,numeric)
    target=directory/'raw'/label/f'rank{rank}.json';tensors={};waves=[]
    record=dict(passed=False,framework=framework,variant=variant,world=world,rank=rank,mode=mode,label=label,
        config_sha256=sha(ROOT/'config/delivery/tp_matrix.json'),workload_sha256=sha(directory/'workload.json'),
        policy=settings['policy'],pid=os.getpid(),waves=waves)
    try:
        if framework=='mini':record['observed']=runtime.engine._tp_observed
        else:record['observed']=dict(dtype='torch.bfloat16',KV_shape=list(runtime.engine.kv_cache.shape),
            graph_keys=sorted(runtime.engine.graphs) if not runtime.engine.enforce_eager else [],head_reference='TP1_internal_only')
        repeats=1 if numeric or mode=='trace' else c['warmup_repeats']+c['measure_repeats']
        from contextlib import nullcontext
        profiling=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if mode=='trace' else nullcontext()
        with profiling as prof:
            for repeat in range(repeats):
                for case in read(directory/'workload.json')['cases']:
                    if framework=='mini':
                        from serving.i2_runtime import flush_cache
                        flush_cache(runtime,poison=numeric)
                    else:runtime.reset()
                    for phase in ('cold','warm'):
                        row,values,_=runtime.wave(case['waves'][phase]);tensors.update(values)
                        row.update(case_id=case['case_id'],phase=phase,repeat=repeat,
                            measured=mode=='benchmark' and repeat>=c['warmup_repeats'])
                        waves.append(row)
        if mode=='trace':
            trace=directory/'traces'/f'{label}_rank{rank}.json';trace.parent.mkdir(parents=True,exist_ok=True)
            prof.export_chrome_trace(str(trace));record['trace_sha256']=sha(trace)
        if numeric and tensors:
            tensor_path=directory/'tensors'/f'{label}_rank{rank}.pt';tensor_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(tensors,tensor_path);record.update(tensor_file=str(tensor_path.relative_to(directory)),tensor_sha256=sha(tensor_path),vectors=sum(t.shape[0] for t in tensors.values()))
        record.update(passed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(rank))
    finally:
        started=time.monotonic();runtime.shutdown();record['shutdown_s']=time.monotonic()-started
        write(target,record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--mode',choices=['collective','fixture','reference','qualify','benchmark','trace'],required=True)
    p.add_argument('--framework');p.add_argument('--variant');p.add_argument('--label')
    args=p.parse_args();c=load_plan();world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('LOCAL_RANK','0'))
    if args.mode=='fixture':write(args.directory/'workload.json',fixture(c));return
    if args.mode=='reference':reference(args.directory,c);return
    if args.mode=='collective':collective(args.directory,world,rank);return
    from serving.tp_preparation import engine_rendezvous,numerical_gate
    engine_rendezvous()
    if args.mode in ('benchmark','trace'):numerical_gate(args.directory,c)
    meta=read(args.directory/'provenance.json')
    if any(sha(ROOT/n)!=s for n,s in meta['source_sha256'].items()):raise ValueError('Frozen TP sources changed')
    replay(args.directory,c,args.framework,args.variant,world,rank,args.mode,args.label)


if __name__=='__main__':main()
