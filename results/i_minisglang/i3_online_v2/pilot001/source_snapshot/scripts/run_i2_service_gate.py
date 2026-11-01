"""Real HTTP/SSE cancellation and cleanup checks for each I2 configuration."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
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
from serving.i2_workloads import CONFIG, digest
from scripts.run_i1_lifecycle import events
from scripts.run_i1_repaired import living_children
from serving.online_client import OnlineClient, OnlineRequest, TERMINAL


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+"\n")


async def exercise(config, raw, output):
    import httpx
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config['model_path'],local_files_only=True)
    ledger,workload,snapshots=[],[],[]
    def sink(row):
        with (raw/'client.jsonl').open('a') as stream:stream.write(json.dumps(row)+"\n")
    client=OnlineClient('http://127.0.0.1:1929',concurrency=8,sink=sink)
    def req(rid,message='Reply with the single word OK.',max_tokens=64,**options):
        prompt=tokenizer.apply_chat_template([dict(role='user',content=message)],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        payload=dict(model=str(ROOT/config['model_path']),prompt=prompt,max_tokens=max_tokens,
            ignore_eos=options.pop('ignore_eos',False),temperature=0,top_p=1,top_k=1)
        workload.append(dict(request_id=rid,payload=payload,input_ids=tokenizer.encode(prompt)))
        return OnlineRequest(rid,payload,timeout_s=options.pop('timeout_s',30),**options)
    async def run(requests):
        result=await client.run(requests)
        ledger.extend(result)
        write(output/'requests.json',ledger)
        write(output/'workload.json',workload)
        return result
    async with httpx.AsyncClient(base_url='http://127.0.0.1:1929',timeout=2,trust_env=False) as http:
        async def settle(label):
            deadline=time.monotonic()+config['cleanup_s']
            while True:
                snapshot=(await http.get('/debug/i1/frontend')).json()
                rows=events(raw)
                idle=next((r for r in reversed(rows) if r['event']=='scheduler_idle'),None)
                detok=next((r for r in reversed(rows) if r['event'] in {'detokenizer_state','detokenizer_end'}),None)
                keys=('ack_uids','event_uids','deadline_uids','admitted_uids','aborting_uids','backend_done_uids')
                if (all(not snapshot[k] for k in keys) and idle and detok and not idle['pending_uids'] and not idle['running_uids'] and not detok['active_uids']):
                    snapshots.append(dict(label=label,**snapshot,idle=idle,detokenizer=detok))
                    write(output/'snapshots.json',snapshots)
                    return
                if time.monotonic()>deadline:raise TimeoutError(f'Unreleased refs after {label}')
                await asyncio.sleep(.05)
        first,=await run([req('eos',timeout_s=600)])
        if first['state']!='finished':raise RuntimeError('Real model EOS request failed')
        await settle('eos')
        await run([req('length',max_tokens=32,ignore_eos=True)])
        await settle('length')
        await run([req('mixed_short',max_tokens=64),req('mixed_medium',max_tokens=32,ignore_eos=True),
            req('decode_cancel',message='Write a long list of integers.',max_tokens=384,ignore_eos=True,disconnect_after_chunks=4)])
        await settle('mixed')
        blockers=[req(f'blocker_{i}',message='List successive integers separated by commas.',max_tokens=384,ignore_eos=True) for i in range(4)]
        task=asyncio.create_task(run(blockers))
        deadline=time.monotonic()+10
        while True:
            rows=events(raw)
            uids={r['uid'] for r in rows if r['event']=='frontend_new_user' and (r.get('external_id') or '').startswith('blocker_')}
            if len(uids)==4 and any(r['event']=='batch_prepare' and uids<=set(r['uids']) for r in rows):break
            if time.monotonic()>deadline:raise TimeoutError('Four blocker requests never ran together')
            await asyncio.sleep(.02)
        await run([req('queued_cancel',cancel_after_s=.25)])
        await task
        await settle('queued_cancel')
        await run([req('deadline_cancel',message='Write a long list of integers.',max_tokens=384,ignore_eos=True,timeout_s=.25)])
        await settle('deadline')
        await run([req('post_cancel_probe',max_tokens=32)])
        await settle('final')
    return ledger


def audit_service(output, config):
    run=json.loads((output/'run.json').read_text())
    if run['status']!='exercised':return dict(passed=False,reason='service execution failed')
    rows=json.loads((output/'events.json').read_text())
    requests=json.loads((output/'requests.json').read_text())
    snapshots=json.loads((output/'snapshots.json').read_text())
    by_name={r['request_id']:r for r in requests}
    assigned={r['external_id']:r['uid'] for r in rows if r['event']=='frontend_new_user'}
    local=defaultdict(list)
    for row in rows:
        if 'uid' in row:local[row['uid']].append(row)
    checks={}
    raw=ROOT/run['runtime_dir']
    checks['raw_logs']=all(digest(raw/name)==value for name,value in run['raw_sha256'].items())
    checks['events_match_raw']=rows==events(raw)
    checks['one_terminal']=len(by_name)==len(requests)==12 and all(sum(e['event'] in TERMINAL for e in r['events'])==1 for r in requests)
    for rid,record in by_name.items():
        uid=assigned[rid]
        tokens=[r for r in local[uid] if r['event']=='scheduler_token']
        free=[r for r in local[uid] if r['event']=='request_free']
        if rid in ('decode_cancel','queued_cancel','deadline_cancel'):
            aborts=[r for r in local[uid] if r['event']=='scheduler_receive' and r['kind']=='AbortBackendMsg']
            ends=[r for r in local[uid] if r['event']=='detokenizer_end']
            terminal=next(e['monotonic_ns'] for e in record['events'] if e['event'] in TERMINAL)
            checks[rid]=record['state']=='cancelled' and len(aborts)==len(ends)==1 and (ends[0]['monotonic_ns']-terminal)/1e9<=config['cleanup_s'] and not any(r['event']=='scheduler_token' and r['monotonic_ns']>aborts[0]['monotonic_ns'] for r in local[uid])
            checks[rid+'_no_further_batches']=bool(aborts) and not any(r['event']=='batch_prepare' and uid in r['uids'] and r['monotonic_ns']>aborts[0]['monotonic_ns'] for r in rows)
            if rid=='queued_cancel':checks['queued_never_executes']=not tokens and not any(r['event']=='batch_prepare' and uid in r['uids'] for r in rows)
            else:checks[rid+'_free_once']=len(free)==1
        else:
            checks[rid]=record['state']=='finished' and len(free)==1 and sum(r['finished'] for r in tokens)==1
            expected='length' if rid=='length' or rid=='mixed_medium' or rid.startswith('blocker_') else 'stop'
            checks[rid+'_reason']=record['finish_reason']==expected
            if expected=='length':checks[rid+'_budget']=len(tokens)==(32 if rid in ('length','mixed_medium') else 384)
    checks['clean_snapshots']=len(snapshots)==6 and all(not s[k] for s in snapshots for k in ('ack_uids','event_uids','deadline_uids','admitted_uids','aborting_uids','backend_done_uids'))
    checks['idle_resources']=all(s['idle']['available_tables']==4 and s['idle']['prefix_total_tokens']==s['idle']['prefix_evictable_tokens'] and s['idle']['free_pages']+s['idle']['prefix_total_tokens']==s['idle']['num_pages'] and not s['detokenizer']['active_uids'] for s in snapshots)
    stopped=[r for r in rows if r['event']=='workers_stopped']
    checks['no_escalation']=bool(stopped) and all(not r['escalated'] and not r['surviving_pids'] and all(v==0 for v in r['exitcodes'].values()) for r in stopped)
    checks['shutdown_bounded']=not run['forced_cleanup'] and not run['surviving_pids'] and run['shutdown_s']<=config['shutdown_s']
    return dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],variant=run['variant'],evidence_sha256={p.name:digest(p) for p in output.glob('*.json') if p.name!='audit.json'})


def worker(config,attempt,variant):
    import httpx
    import psutil
    output=ROOT/config['results_dir']/attempt/f'service_{variant}'
    raw=ROOT/config['runtime_dir']/attempt/f'service_{variant}'
    output.mkdir(parents=True,exist_ok=False)
    raw.mkdir(parents=True,exist_ok=False)
    settings=config['variants'][variant]
    args=[sys.executable,config.get('service_launcher','scripts/launch_i2_service.py'),'--model',str(ROOT/config['model_path']),
        '--host','127.0.0.1','--port','1929','--dtype','bfloat16','--attention-backend','fi','--page-size','1',
        '--num-pages','2048','--max-running-requests','4','--max-seq-len-override','1024',
        '--max-prefill-length',str(settings['chunk']),'--cache-type',settings['cache'],'--num-tokenizer','0',
        '--max-total-requests','8','--cuda-graph-max-bs','0']
    if settings['graph_bs']:args+=['--cuda-graph-bs',*map(str,settings['graph_bs'])]
    env=dict(os.environ,ROUTESCALE_I1_EVENTS=str(raw),MINISGL_DISABLE_OVERLAP_SCHEDULING='0' if settings['overlap'] else '1')
    record=dict(variant=variant,status='fail',runtime_dir=str(raw.relative_to(ROOT)),server_command=args)
    process=None;children=[]
    try:
        with (raw/'server.log').open('w') as log:
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            with httpx.Client(timeout=1,trust_env=False) as http:
                while time.monotonic()-start<600:
                    if process.poll() is not None:raise RuntimeError('Server exited during startup')
                    try:
                        if http.get('http://127.0.0.1:1929/ready').status_code==200 and http.get('http://127.0.0.1:1929/debug/i1/frontend').json()['pid']==process.pid:break
                    except httpx.HTTPError:pass
                    time.sleep(.2)
                else:raise TimeoutError('Service startup budget exceeded')
            children=psutil.Process(process.pid).children(recursive=True)
            asyncio.run(exercise(config,raw,output))
            record['status']='exercised'
    except BaseException:
        record['error']=traceback.format_exc()
        (raw/'error.log').write_text(record['error'])
    finally:
        if process is not None:
            start=time.monotonic();forced=False
            if process.poll() is None:process.send_signal(signal.SIGTERM)
            try:process.wait(timeout=config['shutdown_s'])
            except subprocess.TimeoutExpired:forced=True;process.kill();process.wait(timeout=3)
            alive=living_children(children)
            if alive:_,alive=psutil.wait_procs(alive,timeout=.5)
            for child in alive:
                forced=True
                try:child.kill()
                except psutil.NoSuchProcess:pass
            record.update(shutdown_s=time.monotonic()-start,forced_cleanup=forced,surviving_pids=[p.pid for p in living_children(children)])
        write(output/'events.json',events(raw))
        record['raw_sha256']={p.name:digest(p) for p in raw.iterdir() if p.is_file()}
        write(output/'run.json',record)
    result=audit_service(output,config)
    write(output/'audit.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


def run_service_gate(config,attempt,output,journal,launch):
    if not json.loads((output/'numerical_gate.json').read_text())['passed']:raise ValueError('Numerical gate first')
    results={}
    for variant in config['variants']:
        path=output/f'service_{variant}'/'audit.json'
        if not path.exists():
            launch([sys.executable,'scripts/run_i2_service_gate.py','--attempt',attempt,'--variant',variant],
                f'service_{variant}',config,attempt,output,journal)
        stored=json.loads(path.read_text())
        actual=audit_service(path.parent,config)
        if stored!=actual:raise ValueError('Service audit changed')
        results[variant]=dict(passed=actual['passed'],checks=len(actual.get('checks',{})),audit_sha256=digest(path))
    passed=all(r['passed'] for r in results.values())
    write(output/'service_gate.json',dict(passed=passed,variants=results,source_sha256=digest(__file__)))
    if not passed:raise ValueError('Configuration lifecycle gate failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True);parser.add_argument('--variant',required=True)
    args=parser.parse_args()
    raise SystemExit(worker(json.loads(CONFIG.read_text()),args.attempt,args.variant))
