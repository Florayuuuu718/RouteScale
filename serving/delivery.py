"""Local delivery contract and supervisor; imports require only the stdlib."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'config/delivery/local.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


def load(path=DEFAULT):
    c = read(path)
    fixed = dict(schema_version=1, host='127.0.0.1', tensor_parallel_size=1, dtype='float16',
        upper_fp32_layers=16, attention_backend='fi', graph_bs=[1,2,4], page_size=1,
        num_pages=2048, max_running_requests=4, max_total_requests=8, max_seq_len=1024,
        max_prefill_length=4096, request_timeout_s=20, shutdown_s=10)
    if any(c.get(k) != v for k,v in fixed.items()): raise ValueError('Unsupported qualified delivery scope')
    if type(c['port']) is not int or not 1024 <= c['port'] <= 65534: raise ValueError('Invalid local port')
    if not 0 < c['startup_s'] <= 600 or not 0 < c['health_poll_s'] <= 2 or not 1 <= c['health_failures'] <= 5:
        raise ValueError('Bounded startup/health settings required')
    return c


def http(base, path, timeout=1):
    request = urllib.request.Request(base + path)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()


def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rfind(')')+2:].split()[19]  # Linux starttime, field22
    except (FileNotFoundError, ProcessLookupError): return None


def check_ports(host, port):
    sockets = []
    try:
        for number in (port, port+1):
            s = socket.socket(); sockets.append(s)
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind((host,number))
    finally:
        for s in sockets: s.close()


def gpu_idle():
    value = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()
    if value != '0 MiB': raise ValueError('Exclusive GPU must be idle: ' + value)
    return value


def doctor(c):
    qualified = read(ROOT / c['qualified_sources'])
    if not read(ROOT / c['I3_acceptance'])['local_single_gpu_complete']: raise ValueError('I3 acceptance required')
    for name,digest in qualified.items():
        if sha(ROOT/name) != digest: raise ValueError('Qualified source changed: ' + name)
    from scripts.prepare_i2_minisglang import validate
    tree = validate(ROOT / c['checkout'])['tree']
    manifest = read(ROOT / c['model_manifest'])
    for name,digest in manifest['files'].items():
        if sha(ROOT/c['model_path']/name) != digest: raise ValueError('Model pin changed: ' + name)
    python = ROOT/c['environment']/'bin/python'
    program = 'import importlib.metadata as m,json,platform; print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()})))'
    observed = json.loads(subprocess.check_output([str(python),'-c',program],text=True))
    if observed['python'] != '3.11.16': raise ValueError('Python 3.11.16 required')
    expected = dict(line.split('==',1) for line in (ROOT/c['requirements']).read_text().splitlines() if line and not line.startswith('#'))
    bad = {k:(v,observed['packages'].get(k)) for k,v in expected.items() if observed['packages'].get(k) != v}
    if bad: raise ValueError(f'Locked environment differs: {bad}')
    if not (ROOT/c['toolkit']/'bin/nvcc').is_file(): raise ValueError('Recorded CUDA toolkit required')
    return dict(passed=True, vendor_tree=tree, model_revision=manifest['revision'], packages=len(expected),
        qualified_sources=len(qualified), python=observed['python'], requirements_sha256=sha(ROOT/c['requirements']))


def command(c):
    result = [str(ROOT/c['environment']/'bin/python'),str(ROOT/'scripts/launch_delivery_service.py'),
        '--model',str(ROOT/c['model_path']),'--host',c['host'],'--port',str(c['port'])]
    for key,value in dict(dtype=c['dtype'],attention_backend=c['attention_backend'],page_size=c['page_size'],
        num_pages=c['num_pages'],max_running_requests=c['max_running_requests'],max_seq_len_override=c['max_seq_len'],
        max_prefill_length=c['max_prefill_length'],cache_type='radix',num_tokenizer=0,
        max_total_requests=c['max_total_requests'],request_timeout_s=c['request_timeout_s'],startup_timeout_s=c['startup_s'],cuda_graph_max_bs=0).items():
        result += ['--'+key.replace('_','-'),str(value)]
    return result + ['--cuda-graph-bs',*map(str,c['graph_bs'])]


def environment(c, run):
    toolkit = ROOT/c['toolkit']; cache = ROOT/'runs/cache/delivery'
    return dict(os.environ, PATH=f'{ROOT/c["environment"]}/bin:{toolkit}/bin:' + os.environ['PATH'],
        CUDA_HOME=str(toolkit), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MAX_JOBS='4',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
        MINISGL_DISABLE_OVERLAP_SCHEDULING='1',ROUTESCALE_I3_NUMERIC='0',ROUTESCALE_I3_UPPER_LAYERS='16',
        ROUTESCALE_I1_EVENTS=str(run/'events'), TVM_FFI_CACHE_DIR=str(cache/'tvm'),
        FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))


def stop_child(process, timeout):
    started = time.monotonic(); forced = False
    if process.poll() is None: process.send_signal(signal.SIGTERM)
    try: process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        forced = True
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=2)
    # Give the multiprocessing resource tracker a bounded exit interval too.
    settle = min(started+timeout,time.monotonic()+.5)
    while True:
        try: os.killpg(process.pid,0)
        except ProcessLookupError: break
        if time.monotonic()>=settle:
            forced=True
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            break
        time.sleep(.02)
    return dict(returncode=process.returncode, shutdown_s=time.monotonic()-started, forced_cleanup=forced)


def serve(c, run):
    run = Path(run).resolve()
    if run.exists(): raise ValueError('Fresh run directory required')
    checked = doctor(c); check_ports(c['host'],c['port'])
    lock_path = ROOT/'runs/tmp/h2_runner.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    stop = []; old_handlers = {}
    for sig in (signal.SIGINT,signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig,lambda number,frame: stop.append(number))
    try:
        with lock_path.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); gpu_idle()
            run.mkdir(parents=True); status = run/'status.json'; process = None
            report = dict(state='starting',supervisor_pid=os.getpid(),supervisor_identity=identity(os.getpid()),
                config=c,doctor=checked,command=command(c),stop_requested=False)
            write(status,report)
            try:
                with (run/'server.log').open('x') as log:
                    process = subprocess.Popen(report['command'],cwd=ROOT,env=environment(c,run),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    report['server_pid']=process.pid;write(status,report)
                    base=f'http://{c["host"]}:{c["port"]}'; started=time.monotonic(); failures=0
                    while not stop:
                        if process.poll() is not None: raise RuntimeError('API process exited unexpectedly')
                        try:
                            ready = http(base,'/ready')[0] == 200
                            match = ready and json.loads(http(base,'/debug/i1/frontend')[1])['pid'] == process.pid
                        except (OSError,ValueError): match=False
                        if report['state']=='starting':
                            if match:
                                report.update(state='ready',startup_s=time.monotonic()-started);write(status,report)
                                print('RouteScale ready:',base,flush=True)
                            elif time.monotonic()-started >= c['startup_s']: raise TimeoutError('Startup deadline')
                        else:
                            failures=0 if match else failures+1
                            if failures >= c['health_failures']: raise RuntimeError('Worker readiness lost')
                        time.sleep(c['health_poll_s'])
                report['stop_requested']=True
            except BaseException as exc:
                report['error']=f'{type(exc).__name__}: {exc}'
            finally:
                report['state']='stopping';write(status,report)
                if process is not None: report.update(stop_child(process,c['shutdown_s']))
                report['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()
                report['state']='stopped' if (report.get('stop_requested') and not report.get('forced_cleanup')
                    and report.get('returncode') in (0,-signal.SIGTERM) and report['gpu_after']=='0 MiB') else 'failed'
                write(status,report)
            return 0 if report['state']=='stopped' else 1
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)


def request_stop(run):
    path=Path(run)/'status.json'; r=read(path)
    if r['state'] in ('stopped','failed'): return r
    if identity(r['supervisor_pid']) != r['supervisor_identity']: raise ValueError('Stale supervisor PID; refusing signal')
    os.kill(r['supervisor_pid'],signal.SIGTERM)
    deadline=time.monotonic()+r['config']['shutdown_s']+4
    while time.monotonic()<deadline:
        r=read(path)
        if r['state'] in ('stopped','failed'): return r
        time.sleep(.1)
    raise TimeoutError('Supervisor did not finish bounded shutdown')
