"""Opt-in real service launcher using the same pre-capture precision policy."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/vendor/mini-sglang-i2/python'))

import torch
from minisgl.engine import engine as engine_module
from serving.i2_precision_policy import engine_initializer, restore_engine_policy
from serving.i2_workloads import digest
from serving.minisgl_observer import emit
from serving.i2_worker_shutdown import stop_workers, BUDGETS
from minisgl.server.workers import WorkerGroup

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
mode = os.environ['ROUTESCALE_I2_PRECISION_MODE']
initialize = engine_initializer(engine_module.Engine.__init__, engine_module, mode)
native_shutdown = engine_module.Engine.shutdown


def init(self, config):
    initialize(self, config)
    emit('i2_precision_policy', **self._i2_precision_observed,
        source_sha256={n: digest(ROOT/n) for n in ('serving/i2_precision_policy.py',
            'serving/i2_numerical_policy.py', 'serving/i2_linear_tile_policy.py')})


def shutdown(self):
    restore_engine_policy(self)
    native_shutdown(self)


engine_module.Engine.__init__ = init
engine_module.Engine.shutdown = shutdown


def stop(self):
    if self.shutdown_report is None:
        emit('i2_shutdown_policy', budgets=[dict(signal=s.name,seconds=t) for s,t in BUDGETS],
            total_budget_s=sum(t for _,t in BUDGETS), external_shutdown_budget_s=10,
            source_sha256=digest(ROOT/'serving/i2_worker_shutdown.py'))
    return stop_workers(self)


WorkerGroup.stop = stop

# Install the read-only observer after the core teardown policy so its original
# workers_stopped event observes this implementation rather than being replaced.
import scripts.launch_i2_service

if __name__ == '__main__':
    from scripts.prepare_i2_minisglang import validate
    validate(ROOT/'runs/vendor/mini-sglang-i2')
    from minisgl.server import launch_server
    launch_server()
