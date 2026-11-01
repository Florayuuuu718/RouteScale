"""Read-only scheduler measurements; optional logits are qualification-only."""
import os
from pathlib import Path

from serving.minisgl_observer import emit


def install():
    import torch
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.cache import CacheManager
    from minisgl.message import UserMsg
    from minisgl.engine.sample import Sampler
    from minisgl.engine.graph import GraphRunner
    numeric = os.getenv('ROUTESCALE_I3_NUMERIC') == '1'
    if getattr(Scheduler, '_i3_observed', False):
        return
    Scheduler._i3_observed = True
    native_init = Scheduler.__init__
    def initialize(self, config):
        self._i3_logits, self._i3_inputs = {}, {}
        result = native_init(self, config)
        if numeric:
            native_sample = self.engine.sampler.sample
            def sample(logits, args):
                result = native_sample(logits, args)
                for index, req in enumerate(self._i3_batch.reqs):
                    self._i3_logits.setdefault(req.uid, []).append(logits[index].detach().clone())
                return result
            self.engine.sampler.sample = sample
        return result
    Scheduler.__init__ = initialize
    native_process = Scheduler._process_one_msg
    def process(self, msg):
        if isinstance(msg, UserMsg):
            emit('i3_input', uid=msg.uid, prompt_tokens=len(msg.input_ids), output_budget=msg.sampling_params.max_tokens)
            if numeric:
                self._i3_inputs[msg.uid] = msg.input_ids.tolist()
        return native_process(self, msg)
    Scheduler._process_one_msg = process
    native_prepare = Scheduler._prepare_batch
    def prepare(self, batch):
        details = [dict(uid=r.uid, cached=r.cached_len, extend=r.extend_len, device_len=r.device_len) for r in batch.reqs]
        result = native_prepare(self, batch)
        emit('i3_batch', phase=batch.phase, requests=details, padded=batch.padded_size,
            graph_eligible=self.engine.graph_runner.can_use_cuda_graph(batch))
        return result
    Scheduler._prepare_batch = prepare
    native_forward = Scheduler._forward
    def forward(self, data):
        self._i3_batch = data.batch
        return native_forward(self, data)
    Scheduler._forward = forward
    native_replay = GraphRunner.replay
    def replay(self, batch):
        emit('i3_graph_replay', uids=[r.uid for r in batch.reqs], padded=batch.padded_size)
        return native_replay(self, batch)
    GraphRunner.replay = replay
    native_allocate = CacheManager._allocate
    def allocate(self, needed):
        before = len(self.free_slots)
        result = native_allocate(self, needed)
        emit('i3_allocate', requested_pages=needed, prefix_evicted_pages=len(self.free_slots)+needed-before)
        return result
    CacheManager._allocate = allocate
    native_free = Scheduler._free_req_resources
    def free(self, req):
        if numeric and req.uid in self._i3_logits:
            values = self._i3_logits.pop(req.uid)
            path = Path(os.environ['ROUTESCALE_I3_TENSORS'])/f'uid_{req.uid}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise ValueError('Duplicate numeric request dump')
            torch.save(dict(input_ids=self._i3_inputs.pop(req.uid), logits=torch.stack(values).cpu().float()), path)
            emit('i3_numeric_saved', uid=req.uid, vectors=len(values))
        return native_free(self, req)
    Scheduler._free_req_resources = free
