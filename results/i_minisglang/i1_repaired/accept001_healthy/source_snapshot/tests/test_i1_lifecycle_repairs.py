"""CPU control-plane tests against the actual replayed upstream modules."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs/vendor/mini-sglang-i1/python"))
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import set_tp_info, try_get_tp_info
from minisgl.message import (AbortBackendMsg, BaseFrontendMsg, BaseTokenizerMsg, BatchTokenizerMsg,
    DetokenizeMsg, RequestEndMsg, TokenizeMsg, UserMsg, UserReply)
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.decode import DecodeManager
from minisgl.server import api_server as api
from minisgl.server.workers import WorkerGroup
from minisgl.tokenizer.detokenize import DetokenizeManager
from minisgl.utils import ZmqAsyncPushQueue
import httpx
import torch


class Queue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.closed = False

    async def put(self, item):
        await self.queue.put(item)

    async def get(self):
        return await self.queue.get()

    def stop(self):
        self.closed = True


class Tokenizer:
    eos_token_id = 99

    def batch_decode(self, ids):
        return [" ".join(str(x) for x in row) for row in ids]


class Request:
    def __init__(self, disconnected=False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


def idle_worker(ready):
    def exit_cleanly(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGINT, exit_cleanly)
    ready.set()
    while True:
        time.sleep(0.1)


class FrontendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tx, self.rx = Queue(), Queue()
        self.state = api.FrontendManager(NS(request_timeout_s=1, max_total_requests=2,
            model_path="cpu-test"), self.tx, self.rx)
        api._GLOBAL_STATE = self.state
        self.state._create_listener_once()

    async def asyncTearDown(self):
        await self.state.shutdown()
        api._GLOBAL_STATE = None

    async def deliver(self, uid, reason="length", text="hello", error=None, status=200):
        await self.rx.put(UserReply(uid, text, True, reason, error, status))
        await asyncio.sleep(0.01)

    def assert_clean(self):
        self.assertFalse(self.state.ack_map)
        self.assertFalse(self.state.event_map)
        self.assertFalse(self.state.deadline_map)

    async def test_stream_finish_reason_and_immediate_cleanup(self):
        for reason in ("stop", "length"):
            uid = self.state.new_user()
            await self.deliver(uid, reason)
            body = b"".join([chunk async for chunk in self.state.stream_chat_completions(uid)])
            self.assertIn(json.dumps({"delta": {}, "index": 0, "finish_reason": reason}).encode(), body)
            self.assertEqual(body.count(b"[DONE]"), 1)
            self.assert_clean()
            self.assertFalse(self.state.admitted)

    async def test_explicit_early_close_aborts_but_keeps_admission_until_ack(self):
        uid = self.state.new_user()
        await self.rx.put(UserReply(uid, "first", False))
        stream = self.state.stream_with_cancellation(self.state.stream_chat_completions(uid), Request(), uid)
        await anext(stream)
        await stream.aclose()
        self.assert_clean()
        self.assertEqual(self.state.admitted, {uid})
        self.assertEqual((await self.tx.get()).uid, uid)
        await self.state.abort_user(uid)  # repeated cleanup cannot duplicate aborts
        self.assertTrue(self.tx.queue.empty())
        await self.deliver(uid, reason=None, text="")
        self.assertFalse(self.state.admitted)
        self.assertFalse(self.state.aborting)

    async def test_disconnect_before_first_token(self):
        uid = self.state.new_user()
        with self.assertRaises(asyncio.CancelledError):
            await self.state.start_request(uid, TokenizeMsg(uid, "p", SamplingParams()), Request(True))
        self.assert_clean()
        self.assertIn(uid, self.state.aborting)

    async def test_task_cancellation_and_generator_exception(self):
        uid = self.state.new_user()
        task = asyncio.create_task(self.state.start_request(uid, TokenizeMsg(uid, "p", SamplingParams())))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_clean()
        await self.deliver(uid)

        uid = self.state.new_user()
        async def broken():
            yield b"chunk"
            raise ValueError("broken consumer")
        with self.assertRaisesRegex(ValueError, "broken consumer"):
            async for _ in self.state.stream_with_cancellation(broken(), Request(), uid):
                pass
        self.assert_clean()
        self.assertIn(uid, self.state.aborting)

    async def test_bounded_admission_cannot_be_bypassed_by_cancelling(self):
        first, second = self.state.new_user(), self.state.new_user()
        await self.state.abort_user(first)
        with self.assertRaises(api.HTTPException) as error:
            self.state.new_user()
        self.assertEqual(error.exception.status_code, 429)
        await self.deliver(first)
        replacement = self.state.new_user()
        self.assertEqual(self.state.admitted, {second, replacement})

    async def test_error_is_http_rejection_before_sse_headers(self):
        async def reject():
            msg = await self.tx.get()
            await self.deliver(msg.uid, error="context limit", status=400)
        responder = asyncio.create_task(reject())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
            response = await http.post("/v1/chat/completions", json={"model": "m", "prompt": "p", "stream": True})
        await responder
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("text/event-stream", response.headers["content-type"])
        self.assert_clean()
        self.assertFalse(self.state.admitted)

    async def test_nonstream_finish_reason_and_cleanup(self):
        async def respond():
            msg = await self.tx.get()
            await self.deliver(msg.uid, "length")
        responder = asyncio.create_task(respond())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
            response = await http.post("/v1/chat/completions", json={"model": "m", "prompt": "p"})
        await responder
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")
        self.assert_clean()

    async def test_worker_exit_fails_inflight_and_readiness_and_stops_admission(self):
        uid = self.state.new_user()
        worker = NS(reason=None)
        self.state.workers = NS(failure=lambda: worker.reason, stop=lambda: {})
        await self.rx.put(UserReply(uid, "first", False))
        stream = self.state.stream_with_cancellation(self.state.stream_chat_completions(uid), Request(), uid)
        await anext(stream)
        worker.reason = "Worker scheduler exited (-15)"
        await asyncio.sleep(0.07)
        body = b"".join([chunk async for chunk in stream])
        self.assertIn(b"error", body)
        self.assertNotIn(b"[DONE]", body)
        with self.assertRaises(api.HTTPException) as error:
            await api.v1_root()
        self.assertEqual(error.exception.status_code, 503)
        with self.assertRaises(api.HTTPException):
            self.state.new_user()
        self.assert_clean()

    async def test_server_deadline_cleans_and_aborts(self):
        self.state.config.request_timeout_s = 0.03
        uid = self.state.new_user()
        with self.assertRaises(api.HTTPException) as error:
            await self.state.start_request(uid, TokenizeMsg(uid, "p", SamplingParams()))
        self.assertEqual(error.exception.status_code, 504)
        self.assert_clean()
        self.assertIn(uid, self.state.aborting)

    async def test_shutdown_cancels_owned_tasks_before_closing_queues(self):
        self.state.new_user()
        await self.state.shutdown()
        self.assertTrue(self.tx.closed and self.rx.closed)
        self.assertTrue(self.state.listener_task.done() and self.state.monitor_task.done())
        self.assert_clean()
        await self.state.shutdown()


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if try_get_tp_info() is None:
            set_tp_info(0, 1)

    def test_new_wire_fields_round_trip_in_batches(self):
        messages = BatchTokenizerMsg([DetokenizeMsg(1, 2, True, "length"),
                                      RequestEndMsg(3, "context", 400)])
        self.assertEqual(BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(messages)), messages)
        reply = UserReply(7, "", True, error="failed", status_code=503)
        self.assertEqual(BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply)), reply)

    def test_detokenizer_cancellation_ack_discards_state_without_tombstones(self):
        detok = DetokenizeManager(Tokenizer())
        detok.detokenize([DetokenizeMsg(7, 1, False)])
        detok.detokenize([DetokenizeMsg(7, 2, False)])  # token already in transit before abort
        self.assertIn(7, detok.decode_map)
        detok.forget(7)  # ordered backend ack follows all earlier tokens
        detok.forget(7)
        self.assertFalse(detok.decode_map)
        for uid in range(100):
            detok.detokenize([DetokenizeMsg(uid, 2, True, "length")])
        self.assertFalse(detok.decode_map)

    def test_scheduler_context_error_never_enqueues(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.engine = NS(max_seq_len=8)
        reply, pending = [], []
        scheduler.send_result = reply.extend
        scheduler.prefill_manager = NS(add_one_req=pending.append)
        for length in (8, 9):
            scheduler._process_one_msg(UserMsg(length, torch.zeros(length, dtype=torch.int32), SamplingParams()))
        self.assertFalse(pending)
        self.assertEqual([msg.status_code for msg in reply], [400, 400])

    def test_scheduler_releases_before_abort_ack_and_stops_scheduling(self):
        scheduler = Scheduler.__new__(Scheduler)
        events = []
        req = NS(uid=4)
        pending = {4: req}
        scheduler.prefill_manager = NS(abort_req=lambda uid: pending.pop(uid, None))
        scheduler.decode_manager = NS(abort_req=lambda uid: None)
        scheduler._free_req_resources = lambda req: events.append(("free", req.uid))
        scheduler.send_result = lambda msgs: events.extend(("ack", msg.uid) for msg in msgs)
        scheduler._process_one_msg(AbortBackendMsg(4))
        scheduler._process_one_msg(AbortBackendMsg(4))
        self.assertEqual(events, [("free", 4), ("ack", 4), ("ack", 4)])
        self.assertFalse(pending)

    def test_scheduler_eos_precedence_and_length_ignore_eos(self):
        for token, ignore_eos, expected in ((99, False, "stop"), (2, False, "length"), (99, True, "length")):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.eos_token_id = 99
            scheduler.finished_reqs = set()
            scheduler.cache_manager = NS(lazy_free_region=nullcontext)
            scheduler.decode_manager = DecodeManager(1)
            freed, replies = [], []
            scheduler._free_req_resources = freed.append
            scheduler.send_result = replies.extend
            req = Req(torch.tensor([1, 2]), 0, 0, 1, 5, SamplingParams(ignore_eos=ignore_eos), None)
            req.complete_one()
            scheduler._process_last_data((NS(batch=Batch([req], "decode")),
                (None, torch.tensor([token]), NS(synchronize=lambda: None))))
            self.assertEqual(replies[0].finish_reason, expected)
            self.assertEqual(freed, [req])

    def test_worker_group_reaps_healthy_and_already_dead_children(self):
        ctx = mp.get_context("fork")
        ready = ctx.Event()
        process = ctx.Process(target=idle_worker, args=(ready,))
        process.start()
        self.assertTrue(ready.wait(3))
        workers = WorkerGroup()
        workers.processes = [process]
        self.assertIsNone(workers.failure())
        report = workers.stop()
        self.assertFalse(report["escalated"] or report["surviving_pids"])
        self.assertEqual(process.exitcode, 0)
        self.assertIs(workers.stop(), report)
        self.assertIn("exited", workers.failure())

    def test_startup_observes_dead_worker_instead_of_waiting_forever(self):
        ctx = mp.get_context("fork")
        process = ctx.Process(target=int)
        process.start()
        process.join(2)
        workers = WorkerGroup()
        workers.processes = [process]
        with self.assertRaisesRegex(RuntimeError, "exited"):
            workers.wait_ready(ctx.Queue(), 1, timeout=3)
        self.assertFalse(workers.stop()["surviving_pids"])


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_linger_shutdown_when_peer_died_with_buffered_sends(self):
        queue = ZmqAsyncPushQueue(f"ipc:///tmp/routescale-cpu-absent-{os.getpid()}", False, lambda value: value)
        await queue.put({"buffered": True})
        started = time.monotonic()
        queue.stop()
        self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":
    unittest.main()
