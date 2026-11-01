"""Bounded arithmetic probe on retained failed prefill inputs; never formal timing."""
import argparse
import fcntl
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest


def main():
    import torch
    from minisgl.engine import engine as engine_module
    from serving.i2_precision_runtime import PrecisionScheduler
    from serving.i2_runtime import flush_cache
    from serving.i2_worker import error_metrics
    from scripts.run_i2_precision import precision_setup
    from serving.i3_precision_v2 import install
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upper',type=int,required=True)
    p.add_argument('--attempt');args=p.parse_args()
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from scripts.run_i2_precision_followup import require_idle_gpu
    require_idle_gpu();precision_setup()
    source=ROOT/'results/i_minisglang/i3_online_v1/qual001';meta=read(source/'provenance.json');config=meta['config']
    workload=read(source/'workload.json');lookup={r['request_id']:r for r in workload['requests']}
    refs=read(source/'hf_radix.json')['requests'];directory=source.parent/(args.attempt or f'precision_probe_upper{args.upper}')
    directory.mkdir(exist_ok=False);restorers=[];observed=[]
    native_graph=engine_module.GraphRunner
    def graph(*a,**kw):
        if args.upper>=0:
            restore,policy=install(kw['model'],kw['attn_backend'],args.upper);restorers.append(restore);observed.append(policy)
        return native_graph(*a,**kw)
    scheduler=None;rows=[]
    try:
        with patch.object(engine_module,'GraphRunner',graph):
            scheduler=PrecisionScheduler(config,'radix',capture_logits=True,mode='mixed')
        def run(names,label):
            requests=[dict(request_id=rid,input_ids=lookup[rid]['input_ids'],continuation_ids=[0]) for rid in names]
            row,values,batches=scheduler.wave(requests)
            errors={rid:error_metrics(value,torch.load(ROOT/refs[rid]['reference_path'],weights_only=True)[:1],config['numerics']) for rid,value in values.items()}
            rows.append(dict(label=label,requests=names,work=row,batches=batches,errors=errors))
        with torch.inference_mode():
            failures=['load_015','load_023'];shorts=[r['request_id'] for r in workload['requests'] if r['kind']=='short'][:2]
            seed=next(r for r in workload['requests'] if r['kind']=='shared' and r['request_id'] not in failures)
            for rid in failures:
                flush_cache(scheduler,poison=True);run([rid],'cold_single')
                flush_cache(scheduler,poison=True);scheduler.wave([dict(request_id='seed',input_ids=seed['input_ids'],continuation_ids=[0])]);run([rid],'warm_single')
            flush_cache(scheduler,poison=True);run(failures+shorts,'cold_batch4')
            flush_cache(scheduler,poison=True);scheduler.wave([dict(request_id='seed',input_ids=seed['input_ids'],continuation_ids=[0])]);run(failures+shorts,'warm_batch4')
    finally:
        for restore in reversed(restorers):restore()
        if scheduler is not None:scheduler.shutdown()
    all_errors=[e for row in rows for values in row['errors'].values() for e in values]
    result=dict(upper_layers=args.upper,passed=all(e['passed'] for e in all_errors),max_relative_rmse=max(e['relative_rmse'] for e in all_errors),
        vectors=len(all_errors),observed=observed,rows=rows,diagnostic_only=True,performance_allowed=False,
        source_sha256={n:digest(ROOT/n) for n in ['scripts/probe_i3_precision.py','serving/i3_precision_v2.py']})
    write(directory/'probe.json',result)
    for name in result['source_sha256']:
        import shutil
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    print({k:result[k] for k in ('upper_layers','passed','max_relative_rmse','vectors')},flush=True)


if __name__=='__main__':main()
