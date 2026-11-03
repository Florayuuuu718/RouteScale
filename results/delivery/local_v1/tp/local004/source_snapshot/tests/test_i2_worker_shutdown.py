"""Global teardown deadlines, graceful GC allowance, escalation and idempotency."""
from types import SimpleNamespace
import signal
import threading

import pytest

from serving import i2_worker_shutdown as shutdown


@pytest.mark.parametrize('slow_gc', [False,True])
def test_shared_global_deadline_and_escalation(monkeypatch,slow_gc):
    now=[0.];sent=[];children={}
    class Child:
        def __init__(self,pid,delay):
            self.pid=pid;self.delay=delay;self.exit_at=float('inf');self.exitcode=None;children[pid]=self
        def is_alive(self):
            if now[0]>=self.exit_at:
                self.exitcode=0 if slow_gc else -int(signal.SIGKILL)
                return False
            return True
        def join(self,timeout):
            now[0]+=min(timeout,max(0,self.exit_at-now[0]))
            self.is_alive()
    def kill(pid,sig):
        sent.append((pid,sig))
        if slow_gc and sig==signal.SIGINT:
            children[pid].exit_at=now[0]+children[pid].delay
        elif sig==signal.SIGKILL:
            children[pid].exit_at=now[0]
    monkeypatch.setattr(shutdown.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(shutdown.os,'kill',kill)
    group=SimpleNamespace(processes=[Child(101,5.2),Child(102,5.5)],_stop_lock=threading.Lock(),shutdown_report=None)
    result=shutdown.stop_workers(group)
    assert not result['surviving_pids'] and result['seconds']<=8
    if slow_gc:
        assert result['seconds']==5.5 and not result['escalated']
        assert sent==[(101,signal.SIGINT),(102,signal.SIGINT)]
    else:
        assert result['seconds']==7.5 and result['escalated']
        assert [s for _,s in sent]==[signal.SIGINT]*2+[signal.SIGTERM]*2+[signal.SIGKILL]*2
    count=len(sent);elapsed=now[0]
    assert shutdown.stop_workers(group) is result
    assert len(sent)==count and now[0]==elapsed
