"""Common BF16 prefill row-shape intervention; no numerical/performance gate."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def child(variant, directory):
    sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
    import torch
    import torch.nn.functional as F
    from minisgl.core import get_global_ctx
    from serving.i2_runtime import ReplayScheduler,flush_cache
    from serving.i2_workloads import CONFIG,digest
    config=json.loads(CONFIG.read_text())
    workload=json.loads((ROOT/config['workload_path']).read_text())
    case=next(c for c in workload['cases'] if c['case_id']=='b1_shared')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    native=F.linear
    scheduler=None
    with torch.inference_mode():
        try:
            scheduler=ReplayScheduler(config,variant,capture_logits=True)
            def linear(x,weight,bias=None):
                if get_global_ctx().batch.is_prefill and x.ndim==2:
                    pieces=[]
                    for tile in x.split(512):
                        padded=F.pad(tile,(0,0,0,512-len(tile)))
                        pieces.append(native(padded,weight,bias)[:len(tile)])
                    return torch.cat(pieces)
                return native(x,weight,bias)
            F.linear=linear
            flush_cache(scheduler,poison=True)
            rows=[];tensors={}
            for phase in ('cold','warm'):
                row,values,_=scheduler.wave(case['waves'][phase])
                rows.append(dict(phase=phase,requests=row['requests'],released=row['released']))
                tensors.update(values)
            tensor=directory/f'{variant}.pt'
            torch.save(tensors,tensor)
            (directory/f'{variant}.json').write_text(json.dumps(dict(variant=variant,rows=rows,
                intervention='all_prefill_F_linear_BF16_tile512_both_sides',tensor_sha256=digest(tensor)),indent=2)+'\n')
        finally:
            F.linear=native
            if scheduler is not None:scheduler.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',choices=['naive_full','radix_full'])
    args=parser.parse_args()
    from serving.i2_workloads import CONFIG,digest
    config=json.loads(CONFIG.read_text())
    directory=ROOT/config['validation_dir']/'linear_shape_diagnostic'
    if args.child:
        return child(args.child,directory)
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=ROOT/config['results_dir']/'linear_shape_diagnostic.json'
    if output.exists() or directory.exists():raise ValueError('Preserve existing diagnostic')
    directory.mkdir(parents=True)
    from scripts.run_i2_single_gpu import environment
    processes=[]
    for variant in ('naive_full','radix_full'):
        log=directory/f'{variant}.log'
        with log.open('w') as stream:
            proc=subprocess.run([sys.executable,__file__,'--child',variant],env=environment(config),
                stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        processes.append(dict(variant=variant,returncode=proc.returncode,log_sha256=digest(log),
            gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()))
        if proc.returncode:raise RuntimeError(f'Child failed: {log}')
    import torch
    from serving.i2_worker import error_metrics
    baseline=torch.load(directory/'naive_full.pt',mmap=True,weights_only=True)
    candidate=torch.load(directory/'radix_full.pt',mmap=True,weights_only=True)
    rows=[]
    for rid in baseline:
        errors=error_metrics(candidate[rid],baseline[rid],config['numerics'])
        rows.append(dict(request_id=rid,vectors=len(errors),passed=all(e['passed'] for e in errors),
            exact=torch.equal(candidate[rid],baseline[rid]),max_abs=max(e['max_abs'] for e in errors),
            max_relative_rmse=max(e['relative_rmse'] for e in errors)))
    output.write_text(json.dumps(dict(evidence_class='common_row_shape_intervention_diagnostic_only',
        performance_allowed=False,policy_unchanged=config['numerics'],rows=rows,processes=processes,
        sources={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
        child_evidence={v:json.loads((directory/f'{v}.json').read_text()) for v in ('naive_full','radix_full')}),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
