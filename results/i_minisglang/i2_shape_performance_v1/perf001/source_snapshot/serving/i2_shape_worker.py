"""One fresh performance process for the qualified BF16 shape-control scope."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
from serving.i2_workloads import digest,validate_workload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--variant',required=True);parser.add_argument('--run',type=int,required=True)
    args=parser.parse_args();meta=json.loads((args.directory/'provenance.json').read_text());config=meta['config']
    for n,sha in meta['source_sha256'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Frozen performance sources changed')
    from scripts.prepare_i2_minisglang import validate
    validate(ROOT/config['checkout'])
    workload=json.loads((args.directory/'workload.json').read_text());validate_workload(workload,config)
    label=f'benchmark_{args.variant}_run{args.run}';output=args.directory/f'{label}.json'
    if output.exists():raise ValueError('Preserve performance result')
    raw=ROOT/config['runtime_dir']/args.directory.name/label;raw.mkdir(parents=True,exist_ok=False)
    validation=ROOT/config['validation_dir']/args.directory.name;validation.mkdir(parents=True,exist_ok=True)
    import torch
    import serving.i2_runtime as runtime
    from serving.i2_shape_runtime import ShapeScheduler
    from serving.i2_worker import replay,telemetry
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    record=dict(status='fail',variant=args.variant,run_id=args.run,config_sha256=meta['config_sha256'],
        workload_sha256=meta['workload_sha256'],patched_tree=meta['series']['tree'],runtime_dir=str(raw.relative_to(ROOT)),
        precision={'bf16_reduced_reduction':False,'tf32':False},policy=config['policy'],reference_scope=config['reference_scope'],
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','flashinfer-python','sgl-kernel')},gpu_before=telemetry())
    native=runtime.ReplayScheduler
    try:
        # The legacy replay loop accepts its runtime through this module factory.
        # Keep the old v1 worker and loop unchanged; name the replacement here.
        runtime.ReplayScheduler=ShapeScheduler
        with torch.inference_mode():
            record['replay']=replay(config,workload,args.variant,'benchmark',validation,raw,args.run)
        record.update(status=record['replay']['status'],gpu_end=telemetry(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException:record['error']=traceback.format_exc();print(record['error'],flush=True)
    finally:
        runtime.ReplayScheduler=native
        output.write_text(json.dumps(record,indent=2)+'\n')
    print(label,record['status'],flush=True)
    return 0 if record['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
