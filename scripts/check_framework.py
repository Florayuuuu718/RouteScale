"""Real, bounded full-SGLang checks using the unchanged I3 text inputs."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.delivery import read,sha,write
from serving.framework_client import CompletionClient
from serving.framework_metrics import HTML,parse
from serving.online_client import OnlineRequest,canonical_hash
from serving.sglang_framework import load

WORKLOAD=Path('results/i_minisglang/i3_online_v2/workload002/workload.json')
SCHEDULE=Path('results/i_minisglang/i3_online_v2/online001/schedule.json')


def descendants(pid):
    """Only signal a worker whose parent chain leads to our own server."""
    found=[]
    for child in Path('/proc').iterdir():
        if not child.name.isdecimal():continue
        try:
            stat=(child/'stat').read_text();parent=int(stat[stat.rfind(')')+2:].split()[1])
            if parent==pid:found.append(int(child.name))
        except (OSError,ValueError,IndexError):continue
    return found+[g for child in found for g in descendants(child)]


def select_worker(server_pid):
    candidates=[]
    for pid in descendants(server_pid):
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except OSError:continue
        if 'scheduler' in cmd.lower():candidates.append(dict(pid=pid,command=cmd))
    if len(candidates)!=1:raise ValueError(f'Cannot uniquely identify owned scheduler: {candidates}')
    return candidates[0]


async def exercise(c,run,out,scenario):
    import httpx
    base=f'http://127.0.0.1:{c["port"]}';ledger=[];snapshots=[]
    workload=read(ROOT/WORKLOAD);schedule=read(ROOT/SCHEDULE)
    def request(rid,tokens=32,**options):
        payload=dict(workload['requests'][0]['payload'],model=c['server_args']['served_model_name'],max_tokens=tokens)
        return OnlineRequest(rid,payload,timeout_s=options.pop('timeout_s',c['request_timeout_s']),**options)
    async def send(reqs):
        rows=await CompletionClient(base,concurrency=len(reqs)).run(reqs);ledger.extend(rows);write(out/'requests.json',ledger);return rows
    async with httpx.AsyncClient(base_url=base,timeout=5,trust_env=False) as client:
        async def snap(label):
            r=await client.get('/metrics');r.raise_for_status();native=parse(r.text)
            p=out/'metrics'/f'{len(snapshots):03d}_{label}.txt';p.parent.mkdir(exist_ok=True);p.write_text(r.text)
            row=dict(label=label,time_ns=time.time_ns(),native=native,metrics_file=str(p.relative_to(out)))
            snapshots.append(row);write(out/'snapshots.json',snapshots);return native['metrics']
        async def settle(label):
            end=time.monotonic()+15
            while time.monotonic()<end:
                m=await snap(label)
                if m['running']==0 and m['queued']==0:
                    flush=await client.get('/flush_cache',params={'timeout':3});flush.raise_for_status()
                    write(out/f'flush_{label}.json',dict(status=flush.status_code,body=flush.text))
                    await asyncio.sleep(.3);after=await snap(label+'_flushed')
                    if after['KV_used']==0 and after['KV_usage']==0:return
                await asyncio.sleep(.2)
            raise TimeoutError('Native queue/KV did not release: '+label)
        for route in ('/ready','/health','/health_generate','/v1/models','/server_info'):
            r=await client.get(route);r.raise_for_status()
            write(out/(route.strip('/').replace('/','_')+'.json'),r.json() if r.headers.get('content-type','').startswith('application/json') else dict(status=r.status_code,body=r.text))
        view=await client.get(f'http://127.0.0.1:{c["dashboard_port"]}/');view.raise_for_status();(out/'dashboard.html').write_text(view.text)
        first=await send([request('warmup')]);assert first[0]['state']=='finished';await settle('warmup')
        if scenario=='healthy':
            r=await client.post('/v1/completions',json=dict(request('plain').payload,stream=False));r.raise_for_status();write(out/'nonstream.json',r.json())
            assert r.json()['choices'][0]['finish_reason']=='length'
            bad=await client.post('/v1/completions',json=dict(request('bad').payload,max_tokens=-1))
            over=await client.post('/v1/completions',json=dict(request('over').payload,prompt='hello '*2000))
            write(out/'http_checks.json',dict(nonstream_status=r.status_code,invalid_budget_status=bad.status_code,over_context_status=over.status_code))
            assert 400<=bad.status_code<500 and 400<=over.status_code<500
            await settle('validation')
            cancel=await send([request('disconnect',512,disconnect_after_chunks=4)]);assert cancel[0]['state']=='cancelled';await settle('cancel')
            deadline=await send([request('deadline',512,timeout_s=.1)]);assert deadline[0]['state']=='cancelled' and deadline[0]['reason']=='deadline';await settle('deadline')
            # An explicit native RID makes the queued abort reproducible. The
            # disconnect and administrative abort are recorded as separate acts.
            blockers=asyncio.create_task(send([request(f'queue_blocker_{i}',900) for i in range(4)]))
            end=time.monotonic()+10
            while time.monotonic()<end:
                before=await snap('queue_blockers')
                if before['running']==4:break
                if blockers.done():raise RuntimeError('Queue blockers finished before occupying the native running slots')
                await asyncio.sleep(.03)
            else:raise TimeoutError('Did not observe four native running requests')
            queued=request('queued_disconnect',64,cancel_after_s=.75)
            queued=replace(queued,payload=dict(queued.payload,rid='routescale_queued_cancel'))
            pending=asyncio.create_task(send([queued]));end=time.monotonic()+.7
            while time.monotonic()<end:
                waiting=await snap('queued_cancel_waiting')
                if waiting['queued'] and waiting['queued']>0:break
                await asyncio.sleep(.02)
            else:raise TimeoutError('Cancellation request was not observed in the native queue')
            closed=(await pending)[0];assert closed['state']=='cancelled' and not closed['content']
            abort=await client.post('/abort_request',json={'rid':'routescale_queued_cancel'});abort.raise_for_status()
            end=time.monotonic()+5
            while time.monotonic()<end:
                after=await snap('queued_cancel_aborted')
                if after['queued']==0 and after['running']==4 and after['aborted']>before['aborted']:break
                await asyncio.sleep(.03)
            else:raise TimeoutError('Native queued abort was not observed while blockers were still running')
            write(out/'queued_abort.json',dict(rid='routescale_queued_cancel',client_id='queued_disconnect',
                client_closed_before_native_abort=True,status=abort.status_code,waiting=waiting,after=after))
            held=await blockers;assert all(r['state']=='finished' for r in held);await settle('queued_cancel')
            tasks=asyncio.create_task(send([request(f'burst_{i}',512) for i in range(14)]))
            observed=[]
            while not tasks.done():
                observed.append(await snap('burst'));await asyncio.sleep(.05)
            rows=await tasks;write(out/'queue_observation.json',observed)
            assert any(m['queued'] is not None and m['queued']>0 for m in observed)
            assert all(m['running'] is None or m['running']<=4 for m in observed)
            assert any(r['http_status'] in (429,503) for r in rows)
            await settle('burst')
            # Same 48 prompt strings, output budgets and frozen arrival order as I3.
            source={r['request_id']:r for r in workload['requests']};requests=[];mapping=[]
            for slot in schedule['traces']['near']:
                row=source[slot['request_id']];payload=dict(row['payload'],model=c['server_args']['served_model_name'])
                requests.append(OnlineRequest(row['request_id'],payload,arrival_s=slot['arrival_s'],timeout_s=30))
                mapping.append(dict(request_id=row['request_id'],original_payload_sha256=row['payload_sha256'],mapped_payload_sha256=canonical_hash(payload),prompt_sha256=sha_text(payload['prompt']),arrival_s=slot['arrival_s']))
            write(out/'workload_binding.json',dict(workload_sha256=sha(ROOT/WORKLOAD),schedule_sha256=sha(ROOT/SCHEDULE),rate=schedule['rates_rps']['near'],rate_is_not_SGLang_calibration=True,requests=mapping))
            normal=await send(requests);assert len(normal)==48 and all(r['state']=='finished' for r in normal)
            await settle('trace')
            probe=await send([request('post_cancel_probe')]);assert probe[0]['state']=='finished';await settle('final')
        else:
            active=asyncio.create_task(send([request('inflight',512)]));end=time.monotonic()+10
            while time.monotonic()<end:
                m=await snap('inflight')
                if m['running'] and m['running']>0:break
                if active.done():raise RuntimeError('Request finished before fault trigger')
                await asyncio.sleep(.02)
            else:raise TimeoutError('No actual running request before controlled fault')
            status=read(run/'status.json')
            target=select_worker(status['server_pid']) if scenario=='worker_fault' else dict(pid=status['supervisor_pid'],command='owned supervisor')
            write(out/'fault.json',dict(scenario=scenario,signal='SIGTERM',target=target,time_ns=time.time_ns(),observed_native=m))
            os.kill(target['pid'],signal.SIGTERM);rows=await active
            assert len(rows)==1 and rows[0]['state'] in ('finished','failed','cancelled')
        write(out/'client_summary.json',summarize(ledger))
        if scenario=='healthy':
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                view=(await client.get(f'http://127.0.0.1:{c["dashboard_port"]}/snapshot')).json()
                if view.get('client',{}).get('offered')==len(ledger):break
                await asyncio.sleep(.1)
            else:raise TimeoutError('Dashboard did not collect the final observed client ledger')
            write(out/'dashboard_snapshot.json',view)
    return ledger


def sha_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def summarize(rows):
    ttft=[];intervals=[]
    for row in rows:
        start=next((e['monotonic_ns'] for e in row['events'] if e['event']=='http_send'),None)
        first=next((e['monotonic_ns'] for e in row['events'] if e['event']=='first_text'),None)
        if start and first:ttft.append((first-start)/1e9)
        chunks=[e['monotonic_ns'] for e in row['events'] if e['event']=='sse_choice' and e.get('text')]
        intervals.extend((b-a)/1e9 for a,b in zip(chunks,chunks[1:]))
    def stats(values):
        values=sorted(values)
        return dict(count=len(values),median=values[len(values)//2] if values else None,p95=values[min(len(values)-1,int(.95*len(values)))] if values else None)
    return dict(offered=len(rows),terminal_counts=dict(Counter(r['state'] for r in rows)),first_text_TTFT_s=stats(ttft),SSE_nonempty_chunk_interval_s=stats(intervals),native_token_ITL_claimed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--scenario',choices=['healthy','worker_fault','shutdown_active'],required=True);a=p.parse_args()
    out=a.directory.resolve();out.mkdir(parents=True,exist_ok=False);run=out/'runtime';c=load();parent=None
    names=['serving/sglang_framework.py','serving/framework_metrics.py','serving/framework_client.py','scripts/framework_service.py','scripts/check_framework.py','scripts/audit_framework.py','serving/delivery.py','serving/online_client.py','deploy/local/sglang/runtime.json','deploy/local/sglang/upstream_sources.json','deploy/local/sglang/requirements.lock','docs/sglang_delivery_protocol.md',str(WORKLOAD),str(SCHEDULE)]
    sources={n:sha(ROOT/n) for n in names if (ROOT/n).exists()}
    container=os.environ.get('ROUTESCALE_IN_CONTAINER')=='1'
    commit=os.environ.get('ROUTESCALE_SOURCE_COMMIT','unrecorded') if container else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    clean=False if container else not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    report=dict(passed=False,scenario=a.scenario,root_commit=commit,checkout_clean=clean,container=container,source_sha256=sources,scope=c['scope'],native_BF16_not_I3_numeric_qualification=True)
    import shutil
    for n in sources:
        t=out/'source_snapshot'/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,t)
    try:
        with (out/'supervisor.log').open('x') as log:
            parent=subprocess.Popen([sys.executable,str(ROOT/'scripts/framework_service.py'),'run','--run-dir',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,ROUTESCALE_CLIENT_LEDGER=str(out/'requests.json')))
            start=time.monotonic()
            while time.monotonic()-start<c['startup_s']+15:
                if parent.poll() is not None:raise RuntimeError('SGLang supervisor exited before readiness')
                if (run/'status.json').exists() and read(run/'status.json')['state']=='ready':break
                time.sleep(.2)
            else:raise TimeoutError('Framework readiness deadline')
            asyncio.run(exercise(c,run,out,a.scenario))
            if a.scenario=='healthy':parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=c['shutdown_s']+15);state=read(run/'status.json')
            expected='failed' if a.scenario=='worker_fault' else 'stopped'
            assert state['state']==expected and state['GPU_end']=='0 MiB' and not state['forced_cleanup']
            assert all(sha(ROOT/n)==v for n,v in sources.items())
            report.update(passed=True,supervisor_returncode=parent.returncode)
    except BaseException:report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        if parent is not None and parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            try:parent.wait(timeout=c['shutdown_s']+15)
            except subprocess.TimeoutExpired:parent.kill();parent.wait()
        if (run/'status.json').exists():write(out/'status.json',read(run/'status.json'))
        if (run/'snapshot.json').exists():write(out/'dashboard_snapshot.json',read(run/'snapshot.json'))
        if (run/'metrics_samples.jsonl').exists():shutil.copy2(run/'metrics_samples.jsonl',out/'native_metrics_samples.jsonl')
        write(out/'run.json',report)
    print('Full SGLang',a.scenario,report['passed'],flush=True);return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
