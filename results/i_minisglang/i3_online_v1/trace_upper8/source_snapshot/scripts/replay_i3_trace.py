"""Exact recorded HTTP batch/prefix-history replay; diagnostic, never timing."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest


def main():
    import torch
    from minisgl.engine import engine as engine_module
    from minisgl.core import SamplingParams
    from minisgl.message import UserMsg
    from serving.i2_precision_runtime import PrecisionScheduler
    from serving.i2_runtime import assert_idle
    from serving.i2_worker import error_metrics
    from scripts.run_i2_precision import precision_setup
    from serving.i3_precision_v2 import install
    from scripts.run_i2_precision_followup import require_idle_gpu
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upper',type=int,required=True);args=p.parse_args()
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_idle_gpu();precision_setup();torch.set_num_threads(4)
    source=ROOT/'results/i_minisglang/i3_online_v1/qual001';meta=read(source/'provenance.json');config=meta['config']
    workload=read(source/'workload.json');events=read(source/'qualification_radix/events.json')
    original=read(source/'hf_radix.json')['requests']
    assignments={e['external_id']:e['uid'] for e in events if e['event']=='frontend_new_user'}
    lookup={assignments[r['request_id']]:r for r in workload['warmup']+workload['requests']}
    last=max(e['monotonic_ns'] for e in events if e['event']=='scheduler_token' and e['uid'] in lookup)
    trace=[e for e in events if e['event']=='i3_batch' and e['monotonic_ns']<=last]
    directory=source.parent/f'trace_upper{args.upper}';directory.mkdir(exist_ok=False)
    native_graph=engine_module.GraphRunner;restorers=[];observed=[];scheduler=None
    def graph(*a,**kw):
        if args.upper>=0:
            restore,policy=install(kw['model'],kw['attn_backend'],args.upper);restorers.append(restore);observed.append(policy)
        return native_graph(*a,**kw)
    try:
        with patch.object(engine_module,'GraphRunner',graph):scheduler=PrecisionScheduler(config,'radix',capture_logits=True,mode='mixed')
        scheduler.hits={};scheduler.frees=Counter();scheduler.sample_steps=Counter();scheduler.batches=[]
        scheduler.control_replies=[];scheduler.outputs={};scheduler.logits={};scheduler.replays=0;scheduler.forced={}
        scheduler.zero_token=torch.tensor(0,device=scheduler.device,dtype=torch.int32)
        for uid,r in lookup.items():
            scheduler.outputs[uid]=dict(tokens=[],first_ns=None,last_ns=None,finished=False)
            scheduler.logits[uid]=[]
            scheduler.forced[uid]=torch.tensor(original[r['request_id']]['tokens'],device=scheduler.device,dtype=torch.int32)
        expected=None;prepare=scheduler._prepare_batch;sent=set();checked=0
        def prepare_exact(batch):
            nonlocal checked
            desired=[r['uid'] for r in expected['requests']]
            if set(desired)!={r.uid for r in batch.reqs} or batch.phase!=expected['phase']:
                raise ValueError(f'Batch membership/phase drift at {checked}: expected {desired}, got {[r.uid for r in batch.reqs]}')
            objects={r.uid:r for r in batch.reqs};batch.reqs=[objects[uid] for uid in desired]
            actual=[dict(uid=r.uid,cached=r.cached_len,extend=r.extend_len,device_len=r.device_len) for r in batch.reqs]
            if actual!=expected['requests']:raise ValueError(f'Prefix/history geometry drift at {checked}: {actual} != {expected["requests"]}')
            checked+=1
            return prepare(batch)
        scheduler._prepare_batch=prepare_exact
        with torch.inference_mode():
            for expected in trace:
                for r in expected['requests']:
                    uid=r['uid']
                    if uid not in sent:
                        req=lookup[uid];sent.add(uid)
                        scheduler.messages.append(UserMsg(uid,torch.tensor(req['input_ids'],dtype=torch.int32),
                            SamplingParams(max_tokens=req['output_budget'],ignore_eos=True,temperature=0,top_k=1)))
                with scheduler.engine_stream_ctx:
                    scheduler.engine.stream.wait_stream(scheduler.stream);scheduler.normal_loop()
            torch.cuda.synchronize();released=assert_idle(scheduler)
            tensors={lookup[uid]['request_id']:torch.stack(values).cpu().float() for uid,values in scheduler.logits.items()}
        errors={};exact=0
        for rid,value in tensors.items():
            old=original[rid];ref=torch.load(ROOT/old['reference_path'],weights_only=True)
            errors[rid]=error_metrics(value,ref,config['numerics'])
            native=torch.load(ROOT/old['actual_path'],weights_only=True)['logits']
            exact+=sum(bool(torch.equal(x,y)) for x,y in zip(value,native,strict=True))
        validation=ROOT/'runs/validation/i3_precision_diagnostic';validation.mkdir(parents=True,exist_ok=True)
        path=validation/f'trace_upper{args.upper}.pt'
        if path.exists():raise ValueError('Preserve diagnostic tensors')
        torch.save(tensors,path)
        all_errors=[e for row in errors.values() for e in row]
        result=dict(upper_layers=args.upper,passed=all(e['passed'] for e in all_errors),vectors=len(all_errors),
            failed_vectors=sum(not e['passed'] for e in all_errors),max_relative_rmse=max(e['relative_rmse'] for e in all_errors),
            observed=observed,exact_original_vectors=exact,recorded_batches=len(trace),checked_batches=checked,
            errors=errors,released=released,tensor=dict(path=str(path.relative_to(ROOT)),sha256=digest(path)),
            diagnostic_only=True,performance_allowed=False,source_sha256={n:digest(ROOT/n) for n in
                ('scripts/replay_i3_trace.py','serving/i3_precision_v2.py')},original_reference_sha256=digest(source/'hf_radix.json'))
        write(directory/'probe.json',result)
        for name in result['source_sha256']:
            dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        print({k:result[k] for k in ('upper_layers','passed','vectors','failed_vectors','max_relative_rmse','exact_original_vectors','checked_batches')},flush=True)
    finally:
        for restore in reversed(restorers):restore()
        if scheduler is not None:scheduler.shutdown()


if __name__=='__main__':main()
