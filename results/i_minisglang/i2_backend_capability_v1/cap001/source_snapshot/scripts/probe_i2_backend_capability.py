"""Bounded explicit kernel-API probes; availability is not model acceptance."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import CONFIG,digest


def worker(directory,backend):
    import torch
    from serving.i2_attention_reference import causal_math,tensor_metrics
    torch.set_num_threads(4);torch.manual_seed(20261101)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    q=torch.randn(32,16,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(128,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    keys=k.view(2,64,8,128);values=v.view_as(keys)
    table=torch.tensor([[0,1]],device='cuda',dtype=torch.int32)
    seq=torch.tensor([128],device='cuda',dtype=torch.int32)
    cu_q=torch.tensor([0,32],device='cuda',dtype=torch.int32);cu_k=torch.tensor([0,128],device='cuda',dtype=torch.int32)
    record=dict(backend=backend,evidence_class='explicit_kernel_API_capability_only',fallback_used=False,
        full_model_qualified=False,performance_allowed=False,capability=list(torch.cuda.get_device_capability()),
        shape=dict(query=32,kv=128,qheads=16,kvheads=8,head_dim=128,page_size=64,dtype='bfloat16'),
        packages={n:importlib.metadata.version(n) for n in ('torch','flashinfer-python','sgl-kernel')})
    try:
        if backend=='fa4':
            sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
            from minisgl.attention.fa import _fa_sgl_impl
            result=_fa_sgl_impl(q,keys,values,table,seq,cu_q,cu_k,32,128**-.5,version=4)
        else:
            from flashinfer.prefill import trtllm_batch_context_with_kv_cache
            workspace=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
            result=trtllm_batch_context_with_kv_cache(query=q,kv_cache=(keys,values),workspace_buffer=workspace,
                block_tables=table,seq_lens=seq,max_q_len=32,max_kv_len=128,bmm1_scale=128**-.5,bmm2_scale=1.,
                cum_seq_lens_q=cu_q,cum_seq_lens_kv=cu_k,kv_layout='NHD',batch_size=1,out_dtype=q.dtype)
        torch.cuda.synchronize()
        expected=causal_math(q,k,v,96)
        metrics=tensor_metrics(result.reshape_as(q),expected)
        record.update(status='kernel_API_available' if metrics['finite'] else 'kernel_returned_nonfinite',metrics=metrics)
    except Exception:
        record.update(status='kernel_API_unavailable_in_this_pinned_stack',error=traceback.format_exc())
    (directory/f'{backend}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt')
    parser.add_argument('--worker',type=Path);parser.add_argument('--backend',choices=['fa4','trtllm'])
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.backend)
    if not args.attempt or not args.attempt.replace('_','').isalnum():raise ValueError('Simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    directory=ROOT/'results/i_minisglang/i2_backend_capability_v1'/args.attempt;directory.mkdir(parents=True,exist_ok=False)
    names=['scripts/probe_i2_backend_capability.py','serving/i2_attention_reference.py']
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    lib=ROOT/'runs/envs/mini-sglang/lib/python3.11/site-packages'
    files=[ROOT/'runs/vendor/mini-sglang-i2/python/minisgl/attention/fa.py',ROOT/'runs/vendor/mini-sglang-i2/python/minisgl/attention/trtllm.py',
        lib/'sgl_kernel/flash_attn.py',lib/'flashinfer/prefill.py']
    meta=dict(source_sha256={n:digest(ROOT/n) for n in names},backend_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in files},
        performance_allowed=False,full_model_qualified=False,comparison_eligible=False)
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    config=dict(json.loads(CONFIG.read_text()),runtime_dir='runs/logs/i_minisglang/i2_backend_capability_v1',subprocess_timeout_s=180)
    journal=[]
    for backend in ('fa4','trtllm'):
        launch([sys.executable,__file__,'--worker',str(directory),'--backend',backend],backend,config,args.attempt,directory,journal)


if __name__=='__main__':main()
