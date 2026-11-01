"""Bounded graceful worker teardown, including Python interpreter GC latency.

The external 10-second shutdown criterion is unchanged. Phase budgets are shared
across all workers: 6 s SIGINT, 1.5 s SIGTERM, 0.5 s SIGKILL (8 s total).
"""
import os
import signal
import time

BUDGETS=((signal.SIGINT,6.0),(signal.SIGTERM,1.5),(signal.SIGKILL,.5))


def stop_workers(group):
    with group._stop_lock:
        if group.shutdown_report is not None:
            return group.shutdown_report
        started=time.monotonic();signals=[]
        for sig,budget in BUDGETS:
            for process in [p for p in group.processes if p.is_alive()]:
                try:
                    os.kill(process.pid,sig)
                    signals.append(dict(pid=process.pid,signal=sig.name))
                except ProcessLookupError:
                    pass
            deadline=time.monotonic()+budget
            for process in group.processes:
                process.join(timeout=max(0,deadline-time.monotonic()))
            if not any(p.is_alive() for p in group.processes):
                break
        group.shutdown_report=dict(seconds=time.monotonic()-started,signals=signals,
            escalated=any(r['signal']!='SIGINT' for r in signals),
            surviving_pids=[p.pid for p in group.processes if p.is_alive()],
            exitcodes={str(p.pid):p.exitcode for p in group.processes})
        return group.shutdown_report
