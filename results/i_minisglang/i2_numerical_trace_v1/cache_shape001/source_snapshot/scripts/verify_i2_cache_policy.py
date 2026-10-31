"""Full-model native cache pair under one explicit common planning intervention."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import CONFIG,digest


def worker(directory,variant):
    sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
    import torch
    from serving.i2_numerical_policy import install_prefill_no_split,POLICY
    from serving.i2_runtime import ReplayScheduler,flush_cache
    from scripts.prepare_i2_minisglang import validate
    meta=json.loads((directory/'provenance.json').read_text());config=meta['config']
    for name,sha in meta['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen sources changed')
    workload=json.loads((directory/'workload.json').read_text())
    case=next(c for c in workload['cases'] if c['case_id']=='b1_shared')
    validate(ROOT/config['checkout'])
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    scheduler=restore=restore_linear=None;rows=[];tensors={}
    with torch.inference_mode():
        try:
            scheduler=ReplayScheduler(config,variant,capture_logits=True)
            restore=install_prefill_no_split(scheduler.engine)
            if meta.get('linear_tile512'):
                from serving.i2_linear_tile_policy import install_prefill_linear_tiles
                restore_linear=install_prefill_linear_tiles(scheduler.engine)
            flush_cache(scheduler,poison=True)
            for phase in ('cold','warm'):
                row,values,_=scheduler.wave(case['waves'][phase]);tensors.update(values)
                rows.append(dict(phase=phase,requests=row['requests'],released=row['released']))
            path=ROOT/meta['validation_dir']/f'{variant}.pt';torch.save(tensors,path)
        finally:
            if restore_linear is not None:restore_linear()
            if restore is not None:restore()
            if scheduler is not None:scheduler.shutdown()
    (directory/f'{variant}.json').write_text(json.dumps(dict(variant=variant,policy=POLICY,linear_tile512=meta.get('linear_tile512',False),rows=rows,
        tensor=dict(path=str(path.relative_to(ROOT)),sha256=digest(path)),performance_allowed=False),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt');parser.add_argument('--worker',type=Path)
    parser.add_argument('--variant',choices=['naive_full','radix_full'])
    parser.add_argument('--linear-tile512',action='store_true')
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.variant)
    if not args.attempt or not args.attempt.replace('_','').isalnum():raise ValueError('Simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(CONFIG.read_text())
    directory=ROOT/'results/i_minisglang/i2_numerical_trace_v1'/args.attempt
    directory.mkdir(parents=True,exist_ok=False)
    validation=ROOT/'runs/validation/i2_numerical_trace_v1'/args.attempt
    validation.mkdir(parents=True,exist_ok=False)
    names=['scripts/verify_i2_cache_policy.py','serving/i2_numerical_policy.py','serving/i2_runtime.py',
        'serving/i2_workloads.py','serving/i2_worker.py','docs/i2_prefill_policy_protocol.md',str(CONFIG.relative_to(ROOT))]
    if args.linear_tile512:names+=['serving/i2_linear_tile_policy.py','docs/i2_shape_policy_protocol.md']
    from scripts.prepare_i2_minisglang import prepare_i2
    meta=dict(config=config,series=prepare_i2(),source_sha256={n:digest(ROOT/n) for n in names},
        workload_sha256=digest(ROOT/config['workload_path']),validation_dir=str(validation.relative_to(ROOT)),
        linear_tile512=args.linear_tile512,performance_allowed=False)
    for name in names:
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(ROOT/config['workload_path'],directory/'workload.json')
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    runtime=dict(config,runtime_dir='runs/logs/i_minisglang/i2_numerical_trace_v1',subprocess_timeout_s=300)
    journal=[]
    for variant in ('naive_full','radix_full'):
        launch([sys.executable,__file__,'--worker',str(directory),'--variant',variant],variant,runtime,args.attempt,directory,journal)
    import torch
    from serving.i2_worker import error_metrics
    torch.set_num_threads(4)
    records={v:json.loads((directory/f'{v}.json').read_text()) for v in ('naive_full','radix_full')}
    values={v:torch.load(ROOT/r['tensor']['path'],mmap=True,weights_only=True) for v,r in records.items()}
    rows=[]
    for rid,reference in values['naive_full'].items():
        target=values['radix_full'][rid]
        metrics=error_metrics(target,reference,config['numerics'])
        rows.append(dict(request_id=rid,exact=torch.equal(target,reference),vectors=len(metrics),
            passed=all(m['passed'] for m in metrics),max_relative_rmse=max(m['relative_rmse'] for m in metrics),
            max_abs=max(m['max_abs'] for m in metrics)))
    result=dict(evidence_class='common_prefill_policy_full_model_diagnostic',performance_allowed=False,
        within_native_pair_passed=all(r['passed'] for r in rows),rows=rows,
        tensor_sha256={v:r['tensor']['sha256'] for v,r in records.items()},policy_not_HF_equivalence=True)
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
