"""Bounded ready-then-shutdown probe with worker stacks before escalation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import fcntl
    import httpx
    import psutil
    from scripts.run_i2_single_gpu import environment
    from serving.i2_workloads import digest
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    directory=ROOT/'results/i_minisglang/i2_precision_v1'/args.attempt
    raw=ROOT/'runs/logs/i_minisglang/i2_precision_v1'/args.attempt
    directory.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'scripts/launch_i2_precision_shutdown_trace.py','--model',str(ROOT/'runs/models/Qwen3-0.6B'),
        '--port','1939','--dtype','bfloat16','--attention-backend','fi','--page-size','1','--num-pages','2048',
        '--max-running-requests','4','--max-seq-len-override','1024','--max-prefill-length','4096',
        '--cache-type','naive','--num-tokenizer','0','--max-total-requests','8','--cuda-graph-max-bs','0']
    env=dict(environment({}),ROUTESCALE_I2_PRECISION_MODE='bf16_control',ROUTESCALE_I1_EVENTS=str(raw))
    record=dict(command=command,performance_allowed=False,source_sha256={n:digest(ROOT/n) for n in
        ('scripts/diagnose_i2_precision_shutdown.py','scripts/launch_i2_precision_shutdown_trace.py',
         'scripts/launch_i2_precision_service.py','serving/i2_precision_policy.py')})
    children=[];process=None
    try:
        with (raw/'server.log').open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            start=time.monotonic()
            with httpx.Client(timeout=1,trust_env=False) as client:
                while time.monotonic()-start<180:
                    if process.poll() is not None:raise RuntimeError('Startup failed')
                    try:
                        if client.get('http://127.0.0.1:1939/ready').status_code==200:break
                    except httpx.HTTPError:pass
                    time.sleep(.2)
                else:raise TimeoutError('Startup budget')
            children=psutil.Process(process.pid).children(recursive=True)
            targets=[p for p in children if 'spawn_main' in ' '.join(p.cmdline())]
            record['worker_pids']=[p.pid for p in targets]
            started=time.monotonic();process.send_signal(signal.SIGTERM)
            for _ in range(3):
                time.sleep(.8)
                for child in targets:
                    if child.is_running():
                        try:os.kill(child.pid,signal.SIGUSR1)
                        except ProcessLookupError:pass
            process.wait(timeout=10)
            record.update(shutdown_s=time.monotonic()-started,returncode=process.returncode)
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait()
        for child in children:
            if child.is_running():
                try:child.kill()
                except psutil.NoSuchProcess:pass
        record['log_sha256']=digest(raw/'server.log')
        record['runtime_dir']=str(raw.relative_to(ROOT))
        (directory/'diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
