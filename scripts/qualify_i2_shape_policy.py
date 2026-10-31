"""New-input, seven-profile BF16 within-engine qualification; no timing gate."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import CONFIG,digest,token_hash


def worker(directory,variant,reference=False):
    sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
    import torch
    from serving.i2_runtime import ReplayScheduler,flush_cache
    from serving.i2_numerical_policy import install_prefill_no_split
    from serving.i2_linear_tile_policy import install_prefill_linear_tiles
    from serving.i2_worker import error_metrics
    meta=json.loads((directory/'provenance.json').read_text());config=meta['config']
    for n,sha in meta['source_sha256'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Frozen qualification sources changed')
    workload=json.loads((directory/'workload.json').read_text())
    validation=ROOT/meta['validation_dir']
    label='reference' if reference else variant
    output=directory/f'{label}.json'
    if output.exists():raise ValueError('Preserve qualification')
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    scheduler=restore=restore_linear=None
    rows=[];errors={};snapshots={};tensors={}
    with torch.inference_mode():
        try:
            scheduler=ReplayScheduler(config,variant,capture_logits=True)
            restore=install_prefill_no_split(scheduler.engine)
            restore_linear=install_prefill_linear_tiles(scheduler.engine)
            capture=scheduler.engine.graph_runner.graph_bs_list
            for suite in workload['suites']:
                for case in suite['workload']['cases']:
                    flush_cache(scheduler,poison=True)
                    for phase in ('cold','warm'):
                        row,values,batches=scheduler.wave(case['waves'][phase]);tensors.update(values)
                        expected_hit=suite['config']['shared_prefix_tokens'] if phase=='warm' and case['prefix_mode']=='shared' and config['variants'][variant]['cache']=='radix' else 0
                        if phase=='warm' and any(r['cached_tokens']!=expected_hit for r in row['requests']):raise AssertionError('Prefix hit mismatch')
                        row.update(suite=suite['name'],case_id=case['case_id'],phase=phase,batches=batches)
                        rows.append(row)
            path=validation/f'{label}.pt';torch.save(tensors,path)
        finally:
            if restore_linear is not None:restore_linear()
            if restore is not None:restore()
            if scheduler is not None:scheduler.shutdown()
    if not reference:
        expected=torch.load(validation/'reference.pt',mmap=True,weights_only=True)
        for rid,target in tensors.items():errors[rid]=error_metrics(target,expected[rid],config['numerics'])
    passed=reference or all(e['passed'] for row in errors.values() for e in row)
    record=dict(variant=variant,reference=reference,within_native_passed=passed,performance_allowed=False,
        captured_buckets=capture,rows=rows,errors=errors,
        tensor=dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(label,'within_native_passed',passed,flush=True)
    # Complete all seven numerical diagnostics even on mismatch. No benchmark
    # code exists in this controller. Failures stay explicit in each result.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt');parser.add_argument('--worker',type=Path)
    parser.add_argument('--variant');parser.add_argument('--reference',action='store_true')
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.variant,args.reference)
    if not args.attempt or not args.attempt.replace('_','').isalnum():raise ValueError('Simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(CONFIG.read_text())
    directory=ROOT/'results/i_minisglang/i2_shape_qualification_v1'/args.attempt
    validation=ROOT/'runs/validation/i2_shape_qualification_v1'/args.attempt
    directory.mkdir(parents=True,exist_ok=False);validation.mkdir(parents=True,exist_ok=False)
    from transformers import AutoTokenizer
    from serving.i2_workloads import build_workload,validate_workload
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config['model_path'],local_files_only=True)
    heldout=dict(config,seed=20261101,batch_sizes=[1,2],prompt_tokens=768,shared_prefix_tokens=512)
    suites=[]
    for name,spec in (('control512',config),('holdout768',heldout)):
        w=build_workload(spec,tokenizer)
        for case in w['cases']:
            for phase in ('cold','warm'):
                for request in case['waves'][phase]:request['request_id']=name+':'+request['request_id']
            case['sha256']=token_hash({k:v for k,v in case.items() if k!='sha256'})
        w['config_sha256']=token_hash(spec);validate_workload(w,spec)
        suites.append(dict(name=name,config=spec,workload=w))
    workload=dict(suites=suites)
    (directory/'workload.json').write_text(json.dumps(workload,indent=2)+'\n')
    names=['scripts/qualify_i2_shape_policy.py','serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py',
        'serving/i2_runtime.py','serving/i2_workloads.py','serving/i2_worker.py','serving/h2_workloads.py',
        'scripts/run_i2_single_gpu.py','scripts/prepare_i2_minisglang.py','scripts/prepare_i1_lifecycle.py',
        'docs/i2_policy_qualification_protocol.md',str(CONFIG.relative_to(ROOT))]
    from scripts.prepare_i2_minisglang import prepare_i2
    meta=dict(config=config,series=prepare_i2(),validation_dir=str(validation.relative_to(ROOT)),
        source_sha256={n:digest(ROOT/n) for n in names},workload_sha256=digest(directory/'workload.json'),
        model_sha256={p.name:digest(p) for p in (ROOT/config['model_path']).iterdir() if p.is_file()},
        performance_allowed=False,reference_scope='within_native_BF16_strategy_invariance',paid_resources_used=False)
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    runtime=dict(config,runtime_dir='runs/logs/i_minisglang/i2_shape_qualification_v1',subprocess_timeout_s=900)
    journal=[]
    launch([sys.executable,__file__,'--worker',str(directory),'--variant','naive_full','--reference'],
        'reference',runtime,args.attempt,directory,journal)
    for name in config['variants']:
        launch([sys.executable,__file__,'--worker',str(directory),'--variant',name],name,runtime,args.attempt,directory,journal)


if __name__=='__main__':main()
