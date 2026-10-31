"""Explicit factory for the independently qualified common I2 compute policy."""
from serving.i2_runtime import ReplayScheduler as NativeReplayScheduler
from serving.i2_numerical_policy import install_prefill_no_split
from serving.i2_linear_tile_policy import install_prefill_linear_tiles


class ShapeScheduler(NativeReplayScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy_restorers=[install_prefill_no_split(self.engine),install_prefill_linear_tiles(self.engine)]

    def shutdown(self):
        for restore in reversed(self.policy_restorers):restore()
        self.policy_restorers.clear()
        super().shutdown()
