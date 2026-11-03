"""Own worker handles; readiness is based on acknowledged, live processes."""
from __future__ import annotations

import multiprocessing as mp
import queue
import os
import signal
import threading
import time


class WorkerGroup:
    def __init__(self):
        self.processes: list[mp.Process] = []
        self.ready_messages: list[str] = []
        self._stop_lock = threading.Lock()
        self.shutdown_report: dict | None = None

    def start(self, **kwargs):
        process = mp.Process(**kwargs)
        process.start()
        self.processes.append(process)

    def failure(self) -> str | None:
        for process in self.processes:
            if not process.is_alive():
                return f"Worker {process.name} (pid={process.pid}) exited ({process.exitcode})"
        return None

    def wait_ready(self, ack_queue, count: int, timeout: float):
        deadline = time.monotonic() + timeout
        while len(self.ready_messages) < count:
            if failure := self.failure():
                raise RuntimeError(failure)
            if time.monotonic() >= deadline:
                raise TimeoutError("Worker startup deadline exceeded")
            try:
                self.ready_messages.append(ack_queue.get(timeout=0.1))
            except queue.Empty:
                pass

    def stop(self) -> dict:
        # One shared deadline per phase, never a timeout per child. Also reap
        # children that had already died before the shutdown request.
        with self._stop_lock:
            if self.shutdown_report is not None:
                return self.shutdown_report
            started = time.monotonic()
            signals = []
            for sig, budget in ((signal.SIGINT, 4.0), (signal.SIGTERM, 1.5), (signal.SIGKILL, 0.5)):
                alive = [p for p in self.processes if p.is_alive()]
                for process in alive:
                    try:
                        os.kill(process.pid, sig)
                        signals.append(dict(pid=process.pid, signal=sig.name))
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + budget
                for process in self.processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                if not any(p.is_alive() for p in self.processes):
                    break
            self.shutdown_report = dict(seconds=time.monotonic() - started,
                signals=signals, escalated=any(s['signal'] != 'SIGINT' for s in signals),
                surviving_pids=[p.pid for p in self.processes if p.is_alive()],
                exitcodes={str(p.pid): p.exitcode for p in self.processes})
            return self.shutdown_report
