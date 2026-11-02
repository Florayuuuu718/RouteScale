"""New per-rank offline TP replay on the patched native scheduler."""
from dataclasses import replace
from unittest.mock import patch

import torch


def create(c,variant,world,rank,numeric):
    import serving.i2_runtime as replay
    from minisgl.engine import engine as module
    from minisgl.distributed import DistributedInfo
    from serving.tp_precision import initializer
    from serving.i2_runtime import ReplayScheduler
    original=replay.engine_config
    def cfg(config,name):
        return replace(original(config,name),tp_info=DistributedInfo(rank,world),dtype=torch.float16,
            use_pynccl=False,distributed_timeout=60)
    class TPReplay(ReplayScheduler):
        def shutdown(self):
            for restore in reversed(getattr(self.engine,'_tp_restorers',[])):restore()
            self.engine._tp_restorers=[]
            return super().shutdown()
    with patch.object(replay,'engine_config',cfg),patch.object(module.Engine,'__init__',initializer(module.Engine.__init__,module)):
        return TPReplay(c,variant,capture_logits=numeric)
