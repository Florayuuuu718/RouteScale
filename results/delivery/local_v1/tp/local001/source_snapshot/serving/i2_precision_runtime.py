"""Scheduler factory shared by numerical qualification and formal timing."""
from serving.i2_runtime import ReplayScheduler
from serving.i2_precision_policy import engine_construction, restore_engine_policy


class PrecisionScheduler(ReplayScheduler):
    def __init__(self, config, variant, capture_logits=False, mode='mixed'):
        with engine_construction(mode):
            super().__init__(config, variant, capture_logits=capture_logits)

    def shutdown(self):
        restore_engine_policy(self.engine)
        super().shutdown()
