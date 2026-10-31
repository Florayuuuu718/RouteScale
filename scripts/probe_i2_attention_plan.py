"""Replay fixed native Q/K/V while changing only FA2 planning policy."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def worker(directory):
    import torch
    import flashinfer
    from serving.i2_attention_reference import causal_math,tensor_metrics
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    meta=json.loads((directory/'provenance.json').read_text())
    for name,sha in meta['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen probe source changed')
    traces={}
    for variant,record in meta['inputs'].items():
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError('Captured input changed')
        traces[variant]=torch.load(ROOT/record['path'],mmap=True,weights_only=True)
    naive,radix=traces['naive_full'],traces['radix_full']
    assert torch.equal(naive['warm_kv']['0.q_all'][-128:],radix['warm_kv']['0.q_all'])
    for name in ('0.k','0.v'):assert torch.equal(naive['warm_kv'][name],radix['warm_kv'][name])
    q=naive['warm_kv']['0.q_all'].cuda()
    k,v=[naive['warm_kv'][n].cuda() for n in ('0.k','0.v')]
    with torch.inference_mode():
        math32=causal_math(q[-128:],k,v,384)
        math64=causal_math(q[-128:],k,v,384,dtype=torch.float64)
        rows=[];outputs={};original={}
        for policy,extra in (('auto',{}),('no_split',{'disable_split_kv':True}),('fixed128',{'fixed_split_size':128})):
            selected={};plans={}
            for variant,query in (('naive_full',q),('radix_full',q[-128:])):
                n,h,d=query.shape;kh=k.shape[1]
                # Preserve the original qkv.split Q stride, without changing values.
                packed=torch.empty(n,(h+2*kh)*d,device='cuda',dtype=q.dtype)
                view=packed[:,:h*d].view(n,h,d);view.copy_(query)
                indices=traces[variant]['metadata']['warm']['indices'].cuda()
                keys=torch.full((2049,1,kh,d),float('nan'),device='cuda',dtype=k.dtype)
                values=torch.full_like(keys,float('nan'))
                keys[indices.long(),0]=k;values[indices.long(),0]=v
                workspace=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
                wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace,kv_layout='NHD',backend='fa2')
                cpu=dict(device='cpu',dtype=torch.int32,pin_memory=True)
                wrapper.plan(qo_indptr=torch.tensor([0,n],**cpu),paged_kv_indptr=torch.tensor([0,512],**cpu),
                    paged_kv_indices=indices,paged_kv_last_page_len=torch.ones(1,**cpu),
                    num_qo_heads=h,num_kv_heads=kh,head_dim_qk=d,page_size=1,pos_encoding_mode='NONE',
                    seq_lens=torch.tensor([512],**cpu),q_data_type=q.dtype,kv_data_type=k.dtype,
                    non_blocking=True,causal=True,**extra)
                result=wrapper.run(q=view,paged_kv_cache=(keys,values))[-128:].clone()
                torch.cuda.synchronize()
                vector=list(wrapper._plan_info)
                offset=vector[9]
                chunk=int(wrapper._int_workspace_buffer[offset:offset+4].view(torch.int32).cpu()[0])
                plans[variant]=dict(vector=vector,cta_tile_q=vector[3],split_kv=bool(vector[14]),kv_chunk_size=chunk,
                    q_stride=list(view.stride()),policy=extra)
                selected[variant]=result
                outputs[f'{policy}.{variant}']=result.cpu()
                if policy=='auto':original[variant]=tensor_metrics(result.cpu(),traces[variant]['ops']['0.attention'])
                del wrapper,keys,values,workspace
            rows.append(dict(policy=policy,plans=plans,full_vs_suffix=tensor_metrics(selected['radix_full'],selected['naive_full']),
                full_vs_math32=tensor_metrics(selected['naive_full'],math32),suffix_vs_math32=tensor_metrics(selected['radix_full'],math32)))
        outputs['math32']=math32.cpu();outputs['math64']=math64.cpu()
        tensor=ROOT/meta['validation_dir']/'outputs.pt';torch.save(outputs,tensor)
        result=dict(evidence_class='fixed_QKV_FA2_planning_intervention',performance_allowed=False,
            original_trace_reproduced=original,rows=rows,math32_vs_math64=tensor_metrics(math32,math64),
            output=dict(path=str(tensor.relative_to(ROOT)),sha256=digest(tensor)),
            packages={n:importlib.metadata.version(n) for n in ('torch','flashinfer-python')})
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path);parser.add_argument('--name',default='probe001')
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    directory=args.trace/args.name;directory.mkdir(parents=True,exist_ok=False)
    trace_meta=json.loads((args.trace/'provenance.json').read_text())
    if not json.loads((args.trace/'audit.json').read_text())['localization_valid']:raise ValueError('Valid native trace first')
    validation=ROOT/'runs/validation/i2_numerical_trace_v1'/args.trace.name/args.name
    validation.mkdir(parents=True,exist_ok=False)
    names=['scripts/probe_i2_attention_plan.py','serving/i2_attention_reference.py','docs/i2_attention_probe_protocol.md']
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    lib=ROOT/'runs/envs/mini-sglang/lib/python3.11/site-packages/flashinfer'
    meta=dict(source_sha256={n:digest(ROOT/n) for n in names},validation_dir=str(validation.relative_to(ROOT)),
        inputs={v:json.loads((args.trace/f'{v}.json').read_text())['tensors']['trace'] for v in ('naive_full','radix_full')},
        trace_audit_sha256=digest(args.trace/'audit.json'),performance_allowed=False,
        library_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (lib/'prefill.py',lib/'data/include/flashinfer/attention/scheduler.cuh')})
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    runtime=dict(trace_meta['config'],runtime_dir='runs/logs/i_minisglang/i2_numerical_trace_v1',subprocess_timeout_s=180)
    launch([sys.executable,__file__,'--worker',str(directory)],args.name,runtime,args.trace.name,directory,[])


if __name__=='__main__':main()
