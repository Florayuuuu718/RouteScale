"""Refresh native gauges from real pools after a successful pinned cache flush."""
from functools import wraps
import importlib.metadata
import json
import os
import time


def apply(scheduler_class):
    if getattr(scheduler_class,'_routescale_flush_metrics',False):return
    original=scheduler_class.flush_cache
    @wraps(original)
    def flush(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if result:
            reporter=self.metrics_reporter
            if reporter.current_scheduler_metrics_enabled:
                # Upstream rate-limits idle publication for 30s even after pool
                # reset. Force its own observer/collector to read actual state.
                reporter.metrics_collector.last_log_time=time.perf_counter()-31
                reporter._maybe_log_idle_metrics()
            pool=self.pool_stats_observer.get_pool_stats()
            print(json.dumps(dict(event='routescale_native_flush',pid=os.getpid(),monotonic_ns=time.monotonic_ns(),
                used=pool.full_num_used,available=pool.full_available_size,evictable=pool.full_evictable_size,
                capacity=self.max_total_num_tokens)),flush=True)
        return result
    scheduler_class.flush_cache=flush
    scheduler_class._routescale_flush_metrics=True


def install():
    if importlib.metadata.version('sglang')!='0.5.20':raise ValueError('Flush hook is pinned to SGLang0.5.20')
    from sglang.srt.managers.scheduler import Scheduler
    apply(Scheduler)
