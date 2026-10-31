"""Real CPU processes ignore SIGINT/TERM; verify one shared bounded deadline."""
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from serving.i2_worker_shutdown import stop_workers
    from serving.i2_workloads import digest
    output=ROOT/'results/i_minisglang/i2_precision_v1/shutdown_fault_probe.json'
    if output.exists():raise ValueError('Preserve fault probe')
    code="import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(3600)"
    processes=[]
    class Child:
        def __init__(self,p):self.p=p;self.pid=p.pid
        @property
        def exitcode(self):return self.p.poll()
        def is_alive(self):return self.p.poll() is None
        def join(self,timeout):
            try:self.p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:pass
    try:
        for _ in range(2):
            p=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,text=True)
            processes.append(p)
            if p.stdout.readline().strip()!='ready':raise RuntimeError('Fault child startup')
        group=SimpleNamespace(processes=[Child(p) for p in processes],_stop_lock=threading.Lock(),shutdown_report=None)
        report=stop_workers(group)
        passed=report['escalated'] and not report['surviving_pids'] and report['seconds']<10 and all(v==-9 for v in report['exitcodes'].values())
        record=dict(passed=passed,evidence_class='CPU_process_fault_budget_not_model_inference',
            report=report,external_shutdown_budget_s=10,source_sha256={n:digest(ROOT/n) for n in
                ('scripts/check_i2_shutdown_budget.py','serving/i2_worker_shutdown.py')})
        output.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))
        if not passed:raise ValueError('Fault shutdown budget failed')
    finally:
        for p in processes:
            if p.poll() is None:p.kill()
            p.wait()
            if p.stdout:p.stdout.close()


if __name__=='__main__':main()
