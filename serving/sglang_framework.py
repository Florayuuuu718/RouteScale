"""Formal SGLang delivery; distinct from the teaching Mini-SGLang runtime."""
from __future__ import annotations

import fcntl
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from serving.delivery import ROOT,check_ports,gpu_idle,http,identity,read,sha,stop_child,write
from serving.framework_metrics import HTML,client_counts,parse

CONFIG=ROOT/'deploy/local/sglang/runtime.json'


def load(path=CONFIG):
    c=read(path)
    if c['version']!='0.5.20' or c['host']!='127.0.0.1' or c['server_args']['context_length']!=1024:
        raise ValueError('Unsupported formal framework profile')
    if c['server_args']['max_running_requests']!=4 or c['server_args']['max_queued_requests']!=8:raise ValueError('Bounded profile required')
    c['model_path']=os.environ.get('ROUTESCALE_MODEL_PATH',c['model_path'])
    if os.environ.get('ROUTESCALE_IN_CONTAINER')=='1':c['host']='0.0.0.0'
    return c


def command(c):
    python=Path(os.environ.get('ROUTESCALE_FRAMEWORK_PYTHON',str(ROOT/c['environment']/'bin/python')))
    args=[str(python),'-m','sglang.launch_server','--model-path',str(ROOT/c['model_path']),'--host',c['host'],'--port',str(c['port'])]
    for key,value in c['server_args'].items():args+=['--'+key.replace('_','-'),str(value)]
    return args+['--'+key.replace('_','-') for key in c['server_flags']]


def doctor(c):
    model=ROOT/c['model_path'];manifest=read(ROOT/c['model_manifest'])
    if any(sha(model/name)!=digest for name,digest in manifest['files'].items()):raise ValueError('Same pinned Qwen checkpoint/tokenizer required')
    python=command(c)[0]
    code='import importlib.metadata as m,json,platform;print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()})))'
    env=json.loads(subprocess.check_output([python,'-c',code],text=True))
    if env['packages'].get('sglang')!=c['version']:raise ValueError('SGLang version differs from the selected release')
    pinned=read(ROOT/'deploy/local/sglang/upstream_sources.json')
    if pinned['commit']!=c['source_commit']:raise ValueError('Upstream source pin differs from the profile')
    program='import importlib.metadata as m,hashlib,json,sys;d=m.distribution("sglang");print(json.dumps({n:hashlib.sha256(d.locate_file(n).read_bytes()).hexdigest() for n in json.loads(sys.argv[1])}))'
    actual=json.loads(subprocess.check_output([python,'-c',program,json.dumps(list(pinned['files']))],text=True))
    if actual!=pinned['files']:raise ValueError('Installed SGLang source differs from the clean pinned release')
    lock=ROOT/'deploy/local/sglang/requirements.lock'
    if lock.exists() and os.environ.get('ROUTESCALE_IN_CONTAINER')!='1':
        expected=dict(line.split('==',1) for line in lock.read_text().splitlines() if line and not line.startswith('#'))
        bad={k:(v,env['packages'].get(k)) for k,v in expected.items() if env['packages'].get(k)!=v}
        if bad:raise ValueError(f'Locked formal environment changed: {bad}')
    toolkit=Path(os.environ.get('ROUTESCALE_CUDA_HOME') or os.environ.get('CUDA_HOME') or str(ROOT/'runs/toolchains/cuda-13.0-i1')).resolve()
    nvcc=subprocess.check_output([str(toolkit/'bin/nvcc'),'--version'],text=True)
    return dict(passed=True,model_revision=manifest['revision'],environment=env,model_manifest_sha256=sha(ROOT/c['model_manifest']),
        upstream_source_sha256=actual,source_commit=pinned['commit'],toolchain=dict(path=str(toolkit),nvcc=nvcc,
            math_header_sha256=sha(toolkit/'targets/x86_64-linux/include/crt/math_functions.h')))


def serve(c,directory):
    directory=Path(directory).resolve()
    if directory.exists():raise ValueError('Fresh framework run directory required')
    checked=doctor(c);check_ports('127.0.0.1',c['port']);check_ports('127.0.0.1',c['dashboard_port'])
    lock=ROOT/'runs/tmp/h2_runner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    stop=[];handlers={s:signal.signal(s,lambda sig,frame:stop.append(sig)) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with lock.open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle();directory.mkdir(parents=True)
            state=dict(state='starting',config=c,doctor=checked,supervisor_pid=os.getpid(),supervisor_identity=identity(os.getpid()),command=command(c))
            write(directory/'status.json',state);write(directory/'snapshot.json',dict(state='starting'))
            class Dashboard(BaseHTTPRequestHandler):
                def do_GET(self):
                    body=(directory/'snapshot.json').read_bytes() if self.path=='/snapshot' else HTML.encode()
                    self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/snapshot' else 'text/html; charset=utf-8');self.end_headers();self.wfile.write(body)
                def log_message(self,*args):pass
            viewer=ThreadingHTTPServer((c['host'],c['dashboard_port']),Dashboard)
            threading.Thread(target=viewer.serve_forever,daemon=True).start()
            toolkit=Path(os.environ.get('ROUTESCALE_CUDA_HOME') or os.environ.get('CUDA_HOME') or str(ROOT/'runs/toolchains/cuda-13.0-i1')).resolve()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MAX_JOBS='4',
                PYTHONUNBUFFERED='1',CUDA_HOME=str(toolkit),PATH=str(toolkit/'bin')+':'+os.environ['PATH'])
            process=None;base=f'http://127.0.0.1:{c["port"]}';started=time.monotonic();sample=0
            try:
                with (directory/'server.log').open('x') as log:
                    process=subprocess.Popen(state['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state['server_pid']=process.pid;write(directory/'status.json',state)
                    while not stop:
                        if process.poll() is not None:raise RuntimeError('SGLang process exited unexpectedly')
                        try:ready=http(base,'/ready')[0]==200
                        except OSError:ready=False
                        if state['state']=='starting':
                            if ready:
                                try:healthy=http(base,'/health_generate',timeout=30)[0]==200
                                except OSError:healthy=False
                                if healthy:
                                    state.update(state='ready',startup_s=time.monotonic()-started);write(directory/'status.json',state)
                                    print('SGLang model ready:',base,'dashboard:',c['dashboard_port'],flush=True)
                            if time.monotonic()-started>c['startup_s']:raise TimeoutError('Formal framework startup deadline')
                        elif not ready:raise RuntimeError('SGLang readiness lost')
                        if ready:
                            try:
                                code,text=http(base,'/metrics',timeout=3)
                                if code==200:
                                    native=parse(text)
                                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
                                    snapshot=dict(state=state['state'],time_ns=time.time_ns(),native=native,
                                        client=client_counts(os.environ.get('ROUTESCALE_CLIENT_LEDGER')),GPU_used_MiB=float(gpu))
                                    write(directory/'snapshot.json',snapshot)
                                    with (directory/'metrics_samples.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
                                    (directory/'metrics.latest.txt').write_text(text)
                            except (OSError,ValueError) as exc:
                                with (directory/'monitor_errors.txt').open('a') as f:f.write(str(exc)+'\n')
                        time.sleep(1)
                state['stop_requested']=True
            except BaseException as exc:state['error']=f'{type(exc).__name__}: {exc}'
            finally:
                viewer.shutdown();viewer.server_close()
                if process is not None:state.update(stop_child(process,c['shutdown_s']))
                state['GPU_end']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()
                state['state']='stopped' if state.get('stop_requested') and not state.get('forced_cleanup') and state['GPU_end']=='0 MiB' else 'failed'
                write(directory/'status.json',state)
            return int(state['state']=='failed')
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
