"""CPU-verifiable TP plan, evidence binding and bounded process launcher."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from serving.delivery import ROOT,read,sha,write

CONFIG=ROOT/'config/delivery/tp_matrix.json'


def load_plan(path=CONFIG):
    c=read(path)
    if (c['world_sizes'] != [1,2,4] or c['independent_runs']!=3 or c['subprocess_timeout_s']>900
        or c['paid_resources_authorized'] is not False or c['port']!=2333
        or c['qualification_output_tokens']!=8 or c['numerics']!={'atol':.125,'rtol':.02,'max_relative_rmse':.005}
        or c['performance_requires_all_world_sizes_numerically_qualified'] is not True):
        raise ValueError('Unsupported TP preparation contract')
    return c


def jobs(c,stage):
    rows=[]
    for framework,settings in c['frameworks'].items():
        for world in c['world_sizes']:
            variants=settings['variants']
            rounds=range(1,c['independent_runs']+1) if stage=='benchmark' else [0]
            for run in rounds:
                order=variants[(run-1)%len(variants):]+variants[:(run-1)%len(variants)] if run else variants
                for variant in order:
                    rows.append(dict(framework=framework,world=world,variant=variant,run=run,stage=stage,
                        label=f'{framework}_tp{world}_{variant}_{stage}_run{run}'))
    return rows


def require_gpus(world):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in text.strip().splitlines()]
    if len(rows)<world:raise ValueError(f'{world} GPUs required, only {len(rows)} available; no subprocess started')
    if any(float(r[1]) != 0 for r in rows[:world]):raise ValueError('Selected GPUs must be exclusively idle')
    return text


def numerical_gate(directory,c):
    gate=read(directory/'qualification_gate.json')
    if (not gate.get('passed') or gate.get('config_sha256') != sha(CONFIG)
        or gate.get('model_manifest_sha256') != sha(ROOT/c['model_manifest'])):
        raise ValueError('All-world-size independent qualification required; no timing/trace allowed')
    planned={j['label'] for j in jobs(c,'qualify')}
    if set(gate['qualified_labels'])!=planned:raise ValueError('Incomplete TP qualification matrix')
    if gate.get('local_only') is not False or gate.get('multigpu_accepted') is not True:
        raise ValueError('Local-only probes cannot authorize timing')
    for name,digest in gate['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('TP gate source changed')
    for name,digest in gate['evidence_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('TP gate evidence changed')
    return gate


def launch(command,log,timeout,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic();escalated=[]
        record=dict(pid=process.pid,command=command,status='running')
        write(log.with_suffix('.process.json'),record)
        try:rc=process.wait(timeout=timeout)
        except BaseException:
            for sig,budget in ((signal.SIGINT,5),(signal.SIGTERM,3),(signal.SIGKILL,2)):
                if process.poll() is not None:break
                escalated.append(sig.name)
                try:os.killpg(process.pid,sig)
                except ProcessLookupError:break
                try:process.wait(timeout=budget)
                except subprocess.TimeoutExpired:pass
            record.update(status='failed',returncode=process.poll(),escalated=escalated,
                seconds=time.monotonic()-start,log_sha256=sha(log))
            write(log.with_suffix('.process.json'),record)
            raise
    record.update(status='pass' if rc==0 else 'failed',returncode=rc,seconds=time.monotonic()-start,
        escalated=escalated,log_sha256=sha(log))
    write(log.with_suffix('.process.json'),record)
    if rc:raise RuntimeError(f'TP child rc={rc}: {log}')
    return dict(pid=process.pid,returncode=rc,seconds=time.monotonic()-start,escalated=escalated,log_sha256=sha(log))


def inventory(directory):
    directory.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name,command in [('inventory',['nvidia-smi','-q','-x']),('topology',['nvidia-smi','topo','-m'])]:
        r=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        p=directory/f'{name}.txt'
        with p.open('x') as f:f.write(r.stdout)
        outputs[name]=dict(returncode=r.returncode,sha256=sha(p))
    write(directory/'inventory.json',outputs);return outputs
