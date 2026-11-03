"""Real service delivery smoke; leaves every request, fault and cleanup record."""
from __future__ import annotations

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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import DEFAULT,http,load,read,sha,write
from serving.online_client import OnlineClient,OnlineRequest


def events(run):
    return sorted([json.loads(line) for p in (run/'events').glob('events.*.jsonl') for line in p.read_text().splitlines() if line],key=lambda e:e['monotonic_ns'])


async def exercise(c,run,output,scenario):
    import httpx
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/c['model_path'],local_files_only=True)
    base=f'http://{c["host"]}:{c["port"]}';ledger=[];snapshots=[]
    def request(rid,tokens=16,**opts):
        prompt=tok.apply_chat_template([dict(role='user',content='Tell a short story about a girl and a forest.')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return OnlineRequest(rid,dict(model=str(ROOT/c['model_path']),prompt=prompt,max_tokens=tokens,
            ignore_eos=True,temperature=0,top_p=1,top_k=1),timeout_s=opts.pop('timeout_s',30),**opts)
    async def run_requests(reqs):
        rows=await OnlineClient(base,concurrency=len(reqs)).run(reqs);ledger.extend(rows);write(output/'requests.json',ledger);return rows
    async with httpx.AsyncClient(base_url=base,timeout=2,trust_env=False) as client:
        async def settle(label):
            deadline=time.monotonic()+10
            while True:
                f=(await client.get('/debug/i1/frontend')).json();es=events(run)
                idle=next((r for r in reversed(es) if r['event']=='scheduler_idle'),None)
                detok=next((r for r in reversed(es) if r['event'] in ('detokenizer_end','detokenizer_state')),None)
                last=max((r['monotonic_ns'] for r in es if r['event'] in ('scheduler_token','scheduler_end')),default=0)
                if all(not f[k] for k in ('ack_uids','event_uids','deadline_uids','admitted_uids','aborting_uids','backend_done_uids')) and idle and idle['monotonic_ns']>=last and not idle['pending_uids'] and not idle['running_uids'] and detok and not detok['active_uids']:
                    snapshots.append(dict(label=label,frontend=f,idle=idle,detokenizer=detok));write(output/'snapshots.json',snapshots);return
                if time.monotonic()>deadline:raise TimeoutError('Unreleased resources after '+label)
                await asyncio.sleep(.05)
        async def running(rid):
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                es=events(run);uids=[e['uid'] for e in es if e['event']=='frontend_new_user' and e['external_id']==rid]
                if uids and sum(e['event']=='scheduler_token' and e.get('uid')==uids[0] for e in es)>=4:return uids[0]
                await asyncio.sleep(.03)
            raise TimeoutError('Actual decode never observed: '+rid)
        for route in ('/live','/ready','/health','/v1/models'):
            r=await client.get(route);r.raise_for_status()
        first=await run_requests([request('warmup',timeout_s=600)]);assert first[0]['state']=='finished'
        await settle('warmup')
        if scenario=='healthy':
            normal=await run_requests([request('stream')]);assert normal[0]['state']=='finished'
            payload=request('plain').payload
            r=await client.post('/v1/chat/completions',json=dict(payload,stream=False),headers={'X-Request-ID':'plain'});r.raise_for_status()
            write(output/'nonstream.json',r.json());assert r.json()['choices'][0]['finish_reason']=='length'
            await settle('normal')
            bad=await client.post('/v1/chat/completions',json=dict(payload,max_tokens=0),headers={'X-Request-ID':'invalid_budget'});assert bad.status_code==422
            over=await client.post('/v1/chat/completions',json=dict(payload,prompt='hello '*2000),headers={'X-Request-ID':'over_context'});assert over.status_code==400
            write(output/'http_checks.json',dict(nonstream_status=r.status_code,invalid_budget_status=bad.status_code,over_context_status=over.status_code))
            await settle('validation')
            cancelled=await run_requests([request('disconnect',tokens=128,disconnect_after_chunks=4)])
            assert cancelled[0]['state']=='cancelled';await settle('cancel')
            deadline=await run_requests([request('client_deadline',tokens=256,timeout_s=.1)])
            assert deadline[0]['state']=='cancelled' and deadline[0]['reason']=='deadline';await settle('deadline')
            load_rows=await run_requests([request(f'overload_{i}',tokens=128) for i in range(12)])
            assert any(r['state']=='rejected' for r in load_rows)
            assert all(r['state'] in ('finished','rejected') for r in load_rows)
            await settle('overload')
            response=await client.get('/metrics');response.raise_for_status();(output/'metrics.txt').write_text(response.text)
            assert 'routescale_inference_admitted_current 0' in response.text
            assert 'status="429"' in response.text
            result=await run_requests([request('post_cancel_probe')]);assert result[0]['state']=='finished'
            await settle('final')
        else:
            active=asyncio.create_task(run_requests([request('inflight',tokens=512)]));await running('inflight')
            if scenario=='worker_fault':
                engine=[r for r in events(run) if r['event']=='engine_ready'][-1]
                target=engine['pid'];os.kill(target,signal.SIGTERM)
            else:
                target=read(run/'status.json')['supervisor_pid'];os.kill(target,signal.SIGTERM)
            write(output/'fault.json',dict(scenario=scenario,target_pid=target,signal='SIGTERM',time_ns=time.monotonic_ns()))
            rows=await active;assert all(r['state'] in ('failed','finished') for r in rows)
        return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--scenario',choices=['healthy','worker_fault','shutdown_active'],required=True);args=p.parse_args()
    out=args.directory.resolve();out.mkdir(parents=True,exist_ok=False);run=out/'runtime';c=load();parent=None
    report=dict(scenario=args.scenario,passed=False,source_root=str(ROOT),root_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        checkout_clean=not subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    try:
        with (out/'supervisor.log').open('x') as log:
            parent=subprocess.Popen([sys.executable,str(ROOT/'scripts/routeserve.py'),'run','--run-dir',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            started=time.monotonic()
            while time.monotonic()-started<c['startup_s']+15:
                if parent.poll() is not None:raise RuntimeError('Supervisor exited before ready')
                if (run/'status.json').exists() and read(run/'status.json')['state']=='ready':break
                time.sleep(.2)
            else:raise TimeoutError('Supervisor did not become ready')
            asyncio.run(exercise(c,run,out,args.scenario))
            if args.scenario=='healthy':parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=c['shutdown_s']+5)
            state=read(run/'status.json');write(out/'status.json',state)
            expected='failed' if args.scenario=='worker_fault' else 'stopped'
            report.update(supervisor_returncode=parent.returncode,status=state['state'],forced_cleanup=state['forced_cleanup'],
                shutdown_s=state['shutdown_s'],GPU_end=state['gpu_after'])
            assert state['state']==expected and state['gpu_after']=='0 MiB' and not state['forced_cleanup']
            report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        if parent is not None and parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            try:parent.wait(timeout=15)
            except subprocess.TimeoutExpired:parent.kill();parent.wait()
        write(out/'events.json',events(run))
        if (run/'status.json').exists():write(out/'status.json',read(run/'status.json'))
        write(out/'run.json',report)
    print('Delivery smoke',args.scenario,report['passed'],flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
