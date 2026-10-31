"""Explicit I2 shape policy on every spawned real service engine."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/vendor/mini-sglang-i2/python'))

from minisgl.engine.engine import Engine
from serving.i2_numerical_policy import install_prefill_no_split
from serving.i2_linear_tile_policy import install_prefill_linear_tiles
from serving.i2_workloads import digest

native_init,native_shutdown=Engine.__init__,Engine.shutdown


def init(self,*args,**kwargs):
    native_init(self,*args,**kwargs)
    self._shape_policy_restorers=[install_prefill_no_split(self),install_prefill_linear_tiles(self)]
    from serving.minisgl_observer import emit
    emit('i2_shape_policy',name='fi_prefill_no_split_linear512',source_sha256={n:digest(ROOT/n) for n in
        ('serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py')})


def shutdown(self):
    restorers=getattr(self,'_shape_policy_restorers',[])
    for restore in reversed(restorers):restore()
    restorers.clear()
    native_shutdown(self)


Engine.__init__,Engine.shutdown=init,shutdown
# Also installs the same read-only lifecycle observer in spawn children and
# restores the native asynchronous sampler exactly as in the I2 baseline.
import scripts.launch_i2_service

if __name__=='__main__':
    from scripts.prepare_i2_minisglang import validate
    validate(ROOT/'runs/vendor/mini-sglang-i2')
    from minisgl.server import launch_server
    launch_server()
