"""Explicit I2-qualified precision policy plus read-only I3 observation."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['ROUTESCALE_I2_PRECISION_MODE'] = 'mixed'
import scripts.launch_i2_precision_service
from minisgl.engine import engine as engine_module
from serving.i3_precision_v2 import engine_initializer
from serving.i2_workloads import digest
from serving.minisgl_observer import emit
native_init=engine_module.Engine.__init__
native_shutdown=engine_module.Engine.shutdown
initialize=engine_initializer(native_init,engine_module,int(os.environ['ROUTESCALE_I3_UPPER_LAYERS']))


def init(self,config):
    initialize(self,config)
    emit('i3_precision_policy',**self._i3_precision_observed,source_sha256=digest(ROOT/'serving/i3_precision_v2.py'))


def shutdown(self):
    restore=getattr(self,'_i3_precision_restore',None)
    if restore is not None:
        restore();self._i3_precision_restore=None
    return native_shutdown(self)


engine_module.Engine.__init__=init
engine_module.Engine.shutdown=shutdown
from serving.i3_observer import install
install()

if __name__ == '__main__':
    from minisgl.server import launch_server
    launch_server()
