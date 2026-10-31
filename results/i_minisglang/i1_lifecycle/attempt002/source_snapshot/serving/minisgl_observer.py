"""I1 observation hooks for pinned Mini-SGLang; no scheduling/cleanup fixes.

Logs are local, per-process JSONL. Host timestamps do not measure CUDA execution.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import sys


def emit(event, **data):
    directory = os.getenv("ROUTESCALE_I1_EVENTS")
    if directory:
        path = Path(directory) / f"events.{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(dict(event=event, pid=os.getpid(),
                monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), **data)) + "\n")


def scheduler_state(self):
    size = self.cache_manager.prefix_cache.size_info
    return dict(pending_uids=[p.uid for p in self.prefill_manager.pending_list],
        running_uids=sorted(r.uid for r in self.decode_manager.running_reqs),
        available_tables=self.table_manager.available_size,
        free_pages=len(self.cache_manager.free_slots), num_pages=self.cache_manager.num_pages,
        prefix_total_tokens=size.total_size, prefix_evictable_tokens=size.evictable_size)


def install():
    if not os.getenv("ROUTESCALE_I1_EVENTS"):
        return
    from minisgl.server import api_server as api
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.io import SchedulerIOMixin
    from minisgl.message import UserMsg, AbortBackendMsg
    from minisgl.tokenizer.detokenize import DetokenizeManager
    from minisgl.tokenizer.tokenize import TokenizeManager
    from minisgl.engine.sample import Sampler
    import torch

    if getattr(Scheduler, "_routescale_i1_observed", False):
        return
    Scheduler._routescale_i1_observed = True
    emit("observer_installed")

    original_init = Scheduler.__init__
    def initialize(self, config):
        result = original_init(self, config)
        emit("engine_ready", backend=type(self.engine.attn_backend).__name__,
             torch=torch.__version__, cuda=torch.version.cuda,
             capability=list(torch.cuda.get_device_capability()),
             dtype=str(self.engine.dtype),
             free_bytes=torch.cuda.mem_get_info()[0], **scheduler_state(self))
        return result
    Scheduler.__init__ = initialize

    original_sample = Sampler.sample
    def sample(self, logits, args):
        emit("logits_check", shape=list(logits.shape), finite=bool(torch.isfinite(logits).all().item()),
             allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        return original_sample(self, logits, args)
    Sampler.sample = sample

    original_process = Scheduler._process_one_msg
    def process(self, msg):
        if isinstance(msg, (UserMsg, AbortBackendMsg)):
            emit("scheduler_receive", uid=msg.uid, kind=type(msg).__name__)
        result = original_process(self, msg)
        if isinstance(msg, (UserMsg, AbortBackendMsg)):
            emit("scheduler_state", trigger=type(msg).__name__, uid=msg.uid, **scheduler_state(self))
        return result
    Scheduler._process_one_msg = process

    original_prepare = Scheduler._prepare_batch
    def prepare(self, batch):
        emit("batch_prepare", phase=batch.phase, uids=[r.uid for r in batch.reqs])
        return original_prepare(self, batch)
    Scheduler._prepare_batch = prepare

    original_reply = SchedulerIOMixin._reply_tokenizer_rank0
    def reply(self, messages):
        for msg in messages:
            emit("scheduler_token", uid=msg.uid, token_id=msg.next_token, finished=msg.finished)
        return original_reply(self, messages)
    SchedulerIOMixin._reply_tokenizer_rank0 = reply

    original_free = Scheduler._free_req_resources
    def free(self, req):
        result = original_free(self, req)
        emit("request_free", uid=req.uid, **scheduler_state(self))
        return result
    Scheduler._free_req_resources = free

    original_idle = Scheduler.run_when_idle
    def idle(self):
        result = original_idle(self)
        emit("scheduler_idle", integrity="pass", **scheduler_state(self))
        return result
    Scheduler.run_when_idle = idle

    original_tokenize = TokenizeManager.tokenize
    def tokenize(self, msgs):
        result = original_tokenize(self, msgs)
        for msg, ids in zip(msgs, result, strict=True):
            emit("tokenized", uid=msg.uid, input_ids=ids.tolist())
        return result
    TokenizeManager.tokenize = tokenize

    original_detokenize = DetokenizeManager.detokenize
    def detokenize(self, msgs):
        result = original_detokenize(self, msgs)
        emit("detokenizer_state", uids=[m.uid for m in msgs],
             active_uids=sorted(self.decode_map))
        return result
    DetokenizeManager.detokenize = detokenize

    original_abort = api.FrontendManager.abort_user
    async def abort(self, uid):
        emit("frontend_abort_start", uid=uid)
        result = await original_abort(self, uid)
        emit("frontend_abort_sent", uid=uid, ack_uids=sorted(self.ack_map),
             event_uids=sorted(self.event_map))
        return result
    api.FrontendManager.abort_user = abort

    original_new_user = api.FrontendManager.new_user
    def new_user(self):
        uid = original_new_user(self)
        # Pinned API calls new_user directly from its endpoint; inspect only that caller.
        request = sys._getframe(1).f_locals.get("request")
        emit("frontend_new_user", uid=uid,
             external_id=request.headers.get("X-Request-ID") if request is not None else None)
        return uid
    api.FrontendManager.new_user = new_user

    @api.app.get("/debug/i1/frontend")
    async def frontend_snapshot():
        state = api.get_global_state()
        snapshot = dict(pid=os.getpid(), uid_counter=state.uid_counter,
                        ack_uids=sorted(state.ack_map), event_uids=sorted(state.event_map))
        emit("frontend_snapshot", **{k: v for k, v in snapshot.items() if k != "pid"})
        return snapshot
