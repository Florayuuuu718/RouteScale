"""One isolated native I2 layer-trace process; never a performance run."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--variant',choices=['naive_full','radix_full'],required=True)
    args=parser.parse_args()
    import torch
    from serving.i2_layer_trace import LayerTrace
    from serving.i2_runtime import ReplayScheduler,flush_cache
    from serving.i2_workloads import digest
    from scripts.prepare_i2_minisglang import validate
    from serving.i2_worker import telemetry
    meta=json.loads((args.directory/'provenance.json').read_text())
    config=meta['config']
    workload=json.loads((args.directory/'workload.json').read_text())
    case=next(c for c in workload['cases'] if c['case_id']=='b1_shared')
    validate(ROOT/config['checkout'])
    for name,sha in meta['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen sources changed')
    validation=ROOT/meta['validation_dir']
    output=args.directory/f'{args.variant}.json'
    if output.exists():raise ValueError('Preserve trace')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    scheduler=trace=None
    rows=[];tensors={}
    with torch.inference_mode():
        try:
            scheduler=ReplayScheduler(config,args.variant,capture_logits=True)
            trace=LayerTrace(scheduler)
            flush_cache(scheduler,poison=True)
            for phase in ('cold','warm'):
                trace.phase=phase
                row,values,_=scheduler.wave(case['waves'][phase])
                rows.append(dict(phase=phase,requests=row['requests'],released=row['released']))
                tensors.update(values)
            result=trace.cpu()
            torch.save(result,validation/f'{args.variant}_trace.pt')
            torch.save(tensors,validation/f'{args.variant}_logits.pt')
        finally:
            if trace is not None:trace.close()
            if scheduler is not None:scheduler.shutdown()
    record=dict(variant=args.variant,evidence_class='unmodified_native_layer_trace',performance_allowed=False,
        rows=rows,tree=meta['series']['tree'],packages={n:importlib.metadata.version(n) for n in ('torch','flashinfer-python','sgl-kernel')},
        tensors={name:dict(path=str((validation/f'{args.variant}_{name}.pt').relative_to(ROOT)),
            sha256=digest(validation/f'{args.variant}_{name}.pt')) for name in ('trace','logits')},
        gpu_end=telemetry())
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(args.variant,'trace completed',flush=True)


if __name__=='__main__':main()
