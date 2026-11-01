"""One isolated, bounded real HTTP service process for I3."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read, write
from serving.i2_workloads import digest, token_hash
from serving.online_client import OnlineClient, OnlineRequest
from scripts.run_i1_lifecycle import events
from scripts.run_i1_repaired import living_children


async def exercise(config, workload, raw, output, stage, rate_name, schedule):
    import httpx
    ledger, snapshots = [], []
    def sink(event):
        with (raw/'client.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
    def request(row, **options):
        return OnlineRequest(row['request_id'],row['payload'],timeout_s=options.pop('timeout_s',config['client_timeout_s']),**options)
    async def run(rows, concurrency=64):
        result = await OnlineClient('http://127.0.0.1:1949',concurrency=concurrency,sink=sink).run(rows)
        ledger.extend(result)
        write(output/'requests.json',ledger)
        return result
    async with httpx.AsyncClient(base_url='http://127.0.0.1:1949',timeout=2,trust_env=False) as http:
        async def settle(label):
            end=time.monotonic()+config['cleanup_s']
            while True:
                front=(await http.get('/debug/i1/frontend')).json()
                rows=events(raw)
                idle=next((e for e in reversed(rows) if e['event']=='scheduler_idle'),None)
                detok=next((e for e in reversed(rows) if e['event'] in ('detokenizer_state','detokenizer_end')),None)
                last=max((e['monotonic_ns'] for e in rows if e['event'] in ('scheduler_token','scheduler_end')),default=0)
                keys=('ack_uids','event_uids','deadline_uids','admitted_uids','aborting_uids','backend_done_uids')
                if (all(not front[k] for k in keys) and idle and idle['monotonic_ns']>=last
                    and not idle['pending_uids'] and not idle['running_uids'] and detok and not detok['active_uids']):
                    snapshots.append(dict(label=label,frontend=front,idle=idle,detokenizer=detok))
                    write(output/'snapshots.json',snapshots)
                    return
                if time.monotonic()>end:
                    raise TimeoutError(f'Unreleased frontend/backend references after {label}')
                await asyncio.sleep(.05)
        warm=await run([request(r,timeout_s=600) for r in workload['warmup']],concurrency=4)
        if not all(r['state']=='finished' for r in warm):
            raise RuntimeError('Warmup failed')
        await settle('warmup')
        if stage=='pilot':
            rows=await run([request(r,timeout_s=600) for r in workload['requests']],concurrency=4)
            if not all(r['state']=='finished' for r in rows):
                raise RuntimeError('Closed-loop capacity pilot failed')
            await settle('pilot')
            sends=[e['monotonic_ns'] for r in rows for e in r['events'] if e['event']=='http_send']
            terminals=[e['monotonic_ns'] for r in rows for e in r['events'] if e['event']=='finished']
            write(output/'capacity.json',dict(requests=len(rows),seconds=(max(terminals)-min(sends))/1e9,
                requests_per_s=len(rows)/((max(terminals)-min(sends))/1e9),
                boundary='diagnostic closed-loop concurrency4, all48 unique payloads; not formal online evidence'))
        elif stage=='qualification':
            pending=list(workload['requests'])+workload.get('qualification_extra',[]);sizes=[4,1,2,3];index=0
            while pending:
                n=sizes[index%4];index+=1
                group,pending=pending[:n],pending[n:]
                result=await run([request(r,timeout_s=600) for r in group],concurrency=4)
                if not all(r['state']=='finished' for r in result):
                    raise RuntimeError('Numerical/lifecycle normal request failed')
                await settle(f'qualification_{index}')
            template=next(r for r in workload['requests'] if r['kind']=='short')
            def extra(rid,budget=256,**options):
                row=dict(template,request_id=rid,payload=dict(template['payload'],max_tokens=budget))
                return request(row,**options)
            blockers=asyncio.create_task(run([extra(f'life_block_{i}') for i in range(4)],concurrency=4))
            end=time.monotonic()+15
            while True:
                rows=events(raw);uids={e['uid'] for e in rows if e['event']=='frontend_new_user' and str(e.get('external_id','')).startswith('life_block_')}
                if len(uids)==4 and any(e['event']=='batch_prepare' and uids<=set(e['uids']) for e in rows):
                    break
                if time.monotonic()>end:
                    raise TimeoutError('Four lifecycle blockers never ran together')
                await asyncio.sleep(.02)
            await run([extra('life_queued_cancel',cancel_after_s=.25)])
            await blockers
            await settle('queued_cancel')
            await run([extra('life_decode_cancel',disconnect_after_chunks=4)])
            await settle('decode_cancel')
            await run([extra('life_deadline',timeout_s=.25)])
            await settle('deadline')
            probe=await run([extra('life_probe',budget=16)])
            if probe[0]['state']!='finished':
                raise RuntimeError('Post-cancellation probe failed')
            await settle('final')
        else:
            arrivals={r['request_id']:r['arrival_s'] for r in schedule['traces'][rate_name]}
            await run([request(r,arrival_s=arrivals[r['request_id']]) for r in workload['requests']],concurrency=len(workload['requests']))
            await settle('formal')
    return ledger


def main():
    import httpx
    import psutil
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--variant',required=True);p.add_argument('--stage',choices=['qualification','pilot','formal'],required=True)
    p.add_argument('--rate',default='');p.add_argument('--run',type=int,default=0)
    args=p.parse_args();meta=read(args.directory/'provenance.json');config=meta['config']
    output=args.directory/args.label;raw=ROOT/config['runtime_dir']/args.directory.name/args.label
    output.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    settings=config['variants'][args.variant]
    command=[sys.executable,'scripts/launch_i3_service.py','--model',str(ROOT/config['model_path']),
        '--host','127.0.0.1','--port','1949','--dtype','float16','--attention-backend','fi',
        '--page-size','1','--num-pages','2048','--max-running-requests','4','--max-seq-len-override','1024',
        '--max-prefill-length',str(settings['chunk']),'--cache-type','radix','--num-tokenizer','0',
        '--max-total-requests',str(config['max_total_requests']),'--request-timeout-s',str(config['server_timeout_s']),
        '--cuda-graph-max-bs','0']
    if settings['graph_bs']:
        command+=['--cuda-graph-bs',*map(str,settings['graph_bs'])]
    tensors=ROOT/config['validation_dir']/args.directory.name/args.label
    env=dict(os.environ,ROUTESCALE_I1_EVENTS=str(raw),ROUTESCALE_I3_NUMERIC='1' if args.stage=='qualification' else '0',
        ROUTESCALE_I3_TENSORS=str(tensors),MINISGL_DISABLE_OVERLAP_SCHEDULING='1',
        ROUTESCALE_I3_UPPER_LAYERS=str(config['precision']['upper_fp32_layers']))
    record=dict(stage=args.stage,variant=args.variant,rate=args.rate,run=args.run,status='fail',
        runtime_dir=str(raw.relative_to(ROOT)),server_command=command,config_sha256=meta['config_sha256'],
        workload_sha256=meta['workload_sha256'],server_pid=None,tensors_dir=str(tensors.relative_to(ROOT)))
    process=None;children=[]
    try:
        with (raw/'server.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['server_pid']=process.pid;start=time.monotonic()
            with httpx.Client(timeout=1,trust_env=False) as http:
                while time.monotonic()-start<600:
                    if process.poll() is not None:
                        raise RuntimeError('Server exited during startup')
                    try:
                        if http.get('http://127.0.0.1:1949/ready').status_code==200 and http.get('http://127.0.0.1:1949/debug/i1/frontend').json()['pid']==process.pid:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.2)
                else:
                    raise TimeoutError('Startup deadline')
            children=psutil.Process(process.pid).children(recursive=True)
            schedule=read(args.directory/'schedule.json') if args.stage=='formal' else None
            asyncio.run(exercise(config,read(args.directory/'workload.json'),raw,output,args.stage,args.rate,schedule))
            record['status']='exercised'
    except BaseException:
        record['error']=traceback.format_exc();print(record['error'],flush=True)
    finally:
        if process is not None:
            start=time.monotonic();forced=False
            if process.poll() is None:
                record['shutdown_request']='SIGTERM'
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=config['shutdown_s'])
            except subprocess.TimeoutExpired:
                forced=True;process.kill();process.wait(timeout=3)
            alive=living_children(children)
            if alive:
                _,alive=psutil.wait_procs(alive,timeout=.5)
            for child in alive:
                forced=True
                try:child.kill()
                except psutil.NoSuchProcess:pass
            record.update(shutdown_s=time.monotonic()-start,forced_cleanup=forced,
                surviving_pids=[c.pid for c in living_children(children)],returncode=process.returncode)
        write(output/'events.json',events(raw))
        record['raw_sha256']={p.name:digest(p) for p in raw.iterdir() if p.is_file()}
        if tensors.exists():
            record['tensor_sha256']={p.name:digest(p) for p in tensors.glob('*.pt')}
        write(output/'run.json',record)
    if args.stage=='formal' and record['status']=='exercised':
        from serving.i3_metrics import analyze
        requests=[r for r in read(output/'requests.json') if r['request_id'].startswith('load_')]
        report=analyze(read(args.directory/'workload.json'),requests,read(output/'events.json'),
            read(args.directory/'schedule.json')['rates_rps'][args.rate],config)
        write(output/'metrics.json',report)
    return 0 if record['status']=='exercised' and not record['forced_cleanup'] and not record['surviving_pids'] else 1


if __name__=='__main__':
    raise SystemExit(main())
