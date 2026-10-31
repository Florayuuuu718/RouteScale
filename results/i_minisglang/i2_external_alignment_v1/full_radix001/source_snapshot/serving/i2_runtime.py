"""Controlled fixed-token replay on the real Mini-SGLang scheduler/engine.

This is an explicit experiment controller, not a production decoding policy.
The native sampler executes; its result is replaced by the common continuation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import DetokenizeMsg, UserMsg
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.prefill import ChunkedReq

from serving.i2_workloads import ROOT, token_hash


def engine_config(config, variant):
    settings = config["variants"][variant]
    engine = config["engine"]
    return SchedulerConfig(model_path=str(ROOT / config["model_path"]),
        tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, attention_backend=engine["attention_backend"],
        page_size=engine["page_size"], num_page_override=engine["num_pages"],
        max_running_req=engine["max_running_requests"], max_seq_len_override=engine["max_seq_len"],
        max_extend_tokens=settings["chunk"], cache_type=settings["cache"],
        cuda_graph_bs=settings["graph_bs"], cuda_graph_max_bs=0, offline_mode=True)


def state(scheduler):
    cache = scheduler.cache_manager
    size = cache.prefix_cache.size_info
    return dict(pending=[r.uid for r in scheduler.prefill_manager.pending_list],
        running=sorted(r.uid for r in scheduler.decode_manager.running_reqs),
        tables=scheduler.table_manager.available_size, free_pages=len(cache.free_slots),
        prefix_total=size.total_size, prefix_protected=size.protected_size, num_pages=cache.num_pages,
        retired=sorted(scheduler._retired_uids))


def assert_idle(scheduler):
    snapshot = state(scheduler)
    if (snapshot["pending"] or snapshot["running"] or snapshot["retired"] or snapshot["prefix_protected"] or
        snapshot["tables"] != scheduler.capacity or snapshot["free_pages"] + snapshot["prefix_total"] != snapshot["num_pages"]):
        raise AssertionError(f"Unreleased scheduler resources: {snapshot}")
    scheduler.cache_manager.check_integrity()
    slots = scheduler.table_manager._free_slots
    if sorted(slots) != list(range(scheduler.capacity)):
        raise AssertionError("Duplicate or missing request-table slots")
    cache = scheduler.cache_manager
    indices = cache.free_slots.cpu().tolist()
    root = getattr(cache.prefix_cache, "root_node", None)
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        if not node.is_root():
            indices.extend(node.value.cpu().tolist())
        stack.extend(node.children.values())
    if sorted(indices) != list(range(cache.num_pages)):
        raise AssertionError("Duplicate, missing, or out-of-range KV page indices")
    snapshot.update(table_slots_unique=True, page_indices_unique=True,
                    page_indices_sha256=token_hash(sorted(indices)))
    return snapshot


def flush_cache(scheduler, poison=False):
    assert_idle(scheduler)
    cache = scheduler.cache_manager
    size = cache.prefix_cache.size_info.total_size
    if size:
        cache._free(cache.prefix_cache.evict(size))
    if poison:
        scheduler.engine.kv_cache._kv_buffer.fill_(float("nan"))
    torch.cuda.synchronize()
    if state(scheduler)["free_pages"] != cache.num_pages:
        raise AssertionError("Flush did not return every page")


class ReplayScheduler(Scheduler):
    def __init__(self, config, variant, capture_logits=False):
        self.capacity = config["engine"]["max_running_requests"]
        self.overlap = config["variants"][variant]["overlap"]
        self.capture_logits = capture_logits
        self.uid_counter = 0
        self.messages = []
        self.batch_in_forward = None
        super().__init__(engine_config(config, variant))
        self._native_sample = self.engine.sampler.sample
        self.engine.sampler.sample = self._sample
        replay = self.engine.graph_runner.replay
        def counted_replay(batch):
            self.replays += 1
            return replay(batch)
        self.engine.graph_runner.replay = counted_replay

    def offline_receive_msg(self, blocking=False):
        result, self.messages = self.messages, []
        return result

    def offline_send_result(self, replies):
        now = time.perf_counter_ns()
        for message in replies:
            if not isinstance(message, DetokenizeMsg):
                self.control_replies.append(dict(uid=message.uid, kind=type(message).__name__, time_ns=now))
                continue
            row = self.outputs[message.uid]
            if row["finished"]:
                raise AssertionError("Duplicate backend output after terminal")
            row["tokens"].append(message.next_token)
            row["first_ns"] = row["first_ns"] or now
            if message.finished:
                row.update(finished=True, last_ns=now, reason=message.finish_reason)

    def _prepare_batch(self, batch):
        details = []
        for req in batch.reqs:
            self.hits.setdefault(req.uid, req.cached_len)
            details.append(dict(uid=req.uid, cached=req.cached_len, extend=req.extend_len,
                                chunked=isinstance(req, ChunkedReq)))
        result = super()._prepare_batch(batch)
        self.batches.append(dict(phase=batch.phase, requests=details, padded=batch.padded_size,
                                 graph_eligible=self.engine.graph_runner.can_use_cuda_graph(batch)))
        return result

    def _forward(self, forward_input):
        self.batch_in_forward = forward_input.batch
        return super()._forward(forward_input)

    def _sample(self, logits, args):
        # No isfinite/item/CPU tensor copy in the performance mode.
        self._native_sample(logits, args)
        forced = []
        for index, req in enumerate(self.batch_in_forward.reqs):
            if isinstance(req, ChunkedReq):
                forced.append(self.zero_token)
                continue
            step = self.sample_steps[req.uid]
            if step >= len(self.forced[req.uid]):
                # An EOS path can speculatively execute one extra step. The
                # controlled streams ignore EOS, so this is always a bug here.
                raise AssertionError("Speculative execution exceeded the fixed output budget")
            if self.capture_logits:
                # Keep async GPU clones until the wave is finished. A per-step
                # .cpu()/isfinite().item() would serialize overlap and mask races.
                self.logits[req.uid].append(logits[index].detach().clone())
            forced.append(self.forced[req.uid][step])
            self.sample_steps[req.uid] += 1
        return torch.stack(forced)

    def _free_req_resources(self, req):
        self.frees[req.uid] += 1
        if self.frees[req.uid] != 1:
            raise AssertionError("Request resources freed more than once")
        return super()._free_req_resources(req)

    def wave(self, requests, max_steps=4096):
        assert_idle(self)
        self.hits, self.frees, self.sample_steps = {}, Counter(), Counter()
        self.batches, self.control_replies, self.outputs, self.logits = [], [], {}, {}
        self.replays = 0
        self.forced, assignment = {}, {}
        self.zero_token = torch.tensor(0, device=self.device, dtype=torch.int32)
        for request in requests:
            uid = self.uid_counter
            self.uid_counter += 1
            assignment[uid] = request
            self.outputs[uid] = dict(tokens=[], first_ns=None, last_ns=None, finished=False)
            self.logits[uid] = []
            self.forced[uid] = torch.tensor(request["continuation_ids"], device=self.device, dtype=torch.int32)
            self.messages.append(UserMsg(uid, torch.tensor(request["input_ids"], dtype=torch.int32),
                SamplingParams(max_tokens=len(request["continuation_ids"]), ignore_eos=True, temperature=0, top_k=1)))
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        data = None
        for step in range(max_steps):
            if self.overlap:
                data = self.overlap_loop(data)
            else:
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    self.normal_loop()
            if (all(r["finished"] for r in self.outputs.values()) and not self.prefill_manager.runnable
                and not self.decode_manager.runnable and data is None):
                break
        else:
            raise TimeoutError("Scheduler replay exceeded bounded loop count")
        torch.cuda.synchronize()
        ended = time.perf_counter_ns()
        released = assert_idle(self)
        rows, tensors = [], {}
        for uid, request in assignment.items():
            out = self.outputs[uid]
            if out["tokens"] != request["continuation_ids"] or out["reason"] != "length" or self.frees[uid] != 1:
                raise AssertionError("Teacher-forced output/finish/free contract changed")
            rows.append(dict(request_id=request["request_id"], uid=uid, cached_tokens=self.hits[uid],
                output_tokens=len(out["tokens"]), output_sha256=token_hash(out["tokens"]),
                first_result_ms=(out["first_ns"] - started) / 1e6,
                terminal_ms=(out["last_ns"] - started) / 1e6, reason=out["reason"], free_count=self.frees[uid]))
            if self.capture_logits:
                value = torch.stack(self.logits[uid]).cpu().float().contiguous()
                if not torch.isfinite(value).all():
                    raise AssertionError("Nonfinite real model logits (poisoned KV read?)")
                tensors[request["request_id"]] = value
        return dict(seconds=(ended - started) / 1e9, requests=rows, released=released,
            prefill_batches=sum(b["phase"] == "prefill" for b in self.batches),
            decode_batches=sum(b["phase"] == "decode" for b in self.batches),
            computed_prefill_tokens=sum(r["extend"] for b in self.batches if b["phase"] == "prefill" for r in b["requests"]),
            graph_replays=self.replays, batch_trace_sha256=token_hash(self.batches)), tensors, self.batches
