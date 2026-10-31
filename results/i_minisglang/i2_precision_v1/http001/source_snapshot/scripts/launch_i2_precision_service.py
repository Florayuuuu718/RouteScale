"""Opt-in real service launcher using the same pre-capture precision policy."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/vendor/mini-sglang-i2/python'))

import scripts.launch_i2_service
import torch
from minisgl.engine import engine as engine_module
from serving.i2_precision_policy import engine_initializer, restore_engine_policy
from serving.i2_workloads import digest
from serving.minisgl_observer import emit

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

if __name__ == '__main__':
    from scripts.prepare_i2_minisglang import validate
    validate(ROOT/'runs/vendor/mini-sglang-i2')
    from minisgl.server import launch_server
    launch_server()
