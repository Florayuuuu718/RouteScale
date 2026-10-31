import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs/vendor/mini-sglang-i2/python"))
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import set_tp_info, try_get_tp_info
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.decode import DecodeManager
from serving.i2_workloads import CONFIG, build_workload, token_hash, validate_workload
from serving.i2_worker import error_metrics


class Tokenizer:
    all_special_ids = [-1]
    def encode(self, text, **kwargs):
        return list(range(10, 110))


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_exact_prefix_boundaries_and_fixed_continuations(self):
        workload = build_workload(self.config, Tokenizer())
        validate_workload(workload, self.config)
        self.assertEqual(len(workload["cases"]), 4)
        self.assertEqual(sum(len(c["waves"][p]) for c in workload["cases"] for p in ("cold", "warm")), 16)

    def test_semantic_prefix_tamper_fails_even_with_recomputed_hash(self):
        workload = build_workload(self.config, Tokenizer())
        case = next(c for c in workload["cases"] if c["prefix_mode"] == "shared")
        row = case["waves"]["warm"][0]
        row["input_ids"][0] += 1000
        row["input_sha256"] = token_hash(row["input_ids"])
        case["sha256"] = token_hash({k:v for k,v in case.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "shared-prefix"):
            validate_workload(workload, self.config)

    def test_full_vocabulary_gate_rejects_one_bad_value_or_nonfinite(self):
        policy = self.config["numerics"]
        reference = torch.ones(2, 100)
        self.assertTrue(all(r["passed"] for r in error_metrics(reference, reference, policy)))
        target = reference.clone()
        target[1, -1] = 5
        self.assertFalse(error_metrics(target, reference, policy)[1]["passed"])
        target[1, -1] = float("nan")
        self.assertFalse(error_metrics(target, reference, policy)[1]["finite"])
        with self.assertRaises(ValueError):
            error_metrics(reference[:1], reference, policy)


class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if try_get_tp_info() is None:
            set_tp_info(0, 1)

    def make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._overlap_data = None
        scheduler._retired_uids = set()
        scheduler.finished_reqs = set()
        scheduler.eos_token_id = 99
        scheduler.cache_manager = NS(lazy_free_region=nullcontext)
        scheduler.prefill_manager = NS(abort_req=lambda uid: None, runnable=False)
        scheduler.decode_manager = DecodeManager(1)
        self.events, self.replies = [], []
        scheduler._free_req_resources = lambda req: self.events.append(("free", req.uid))
        def send(msgs):
            self.replies.extend(msgs)
            self.events.extend(("reply", msg.uid) for msg in msgs)
        scheduler.send_result = send
        return scheduler

    def request(self, uid=7, output=4):
        return Req(torch.tensor([1, 2]), 0, 0, output, uid, SamplingParams(), None)

    def data(self, req, token, name):
        event = NS(synchronize=lambda: self.events.append(("sync", name)))
        output = (None, torch.tensor([token]), event)
        return (NS(batch=Batch([req], "decode")), output)

    def inflight(self, req):
        return (NS(batch=Batch([req], "decode")), NS(copy_done_event=NS(synchronize=lambda: self.events.append(("wait", req.uid)))))

    def test_eos_waits_before_free_and_suppresses_speculative_reply(self):
        scheduler = self.make_scheduler()
        req = self.request()
        req.complete_one()
        req.complete_one()
        scheduler.decode_manager.running_reqs.add(req)
        scheduler._overlap_data = self.inflight(req)
        scheduler._process_last_data(self.data(req, 99, "last"))
        self.assertLess(self.events.index(("wait", 7)), self.events.index(("free", 7)))
        scheduler._process_last_data(self.data(req, 2, "speculative"))
        self.assertEqual(len(self.replies), 1)
        self.assertEqual(self.replies[0].finish_reason, "stop")
        self.assertEqual(self.events.count(("free", 7)), 1)

    def test_length_uses_host_tokens_not_ahead_device_length(self):
        scheduler = self.make_scheduler()
        req = self.request(output=2)
        req.complete_one()
        req.complete_one()
        self.assertFalse(req.can_decode)
        scheduler._process_last_data(self.data(req, 3, "first"))
        self.assertFalse(self.replies[-1].finished)
        scheduler._process_last_data(self.data(req, 4, "last"))
        self.assertTrue(self.replies[-1].finished)
        self.assertEqual(self.replies[-1].finish_reason, "length")

    def test_abort_finds_final_step_after_request_left_running_set(self):
        scheduler = self.make_scheduler()
        req = self.request(output=1)
        req.complete_one()
        scheduler._overlap_data = self.inflight(req)
        scheduler._process_one_msg(AbortBackendMsg(7))
        self.assertEqual(self.events[:3], [("wait", 7), ("free", 7), ("reply", 7)])
        scheduler._process_last_data(self.data(req, 3, "late"))
        self.assertEqual(len(self.replies), 1)
        scheduler._process_one_msg(AbortBackendMsg(7))
        self.assertEqual(self.events.count(("free", 7)), 1)

    def test_retirement_set_is_bounded_to_actual_inflight_requests(self):
        scheduler = self.make_scheduler()
        for uid in range(100):
            req = self.request(uid, output=1)
            req.complete_one()
            scheduler._process_last_data(self.data(req, 3, str(uid)))
        self.assertFalse(scheduler._retired_uids)
        scheduler._retired_uids = {7}
        req = self.request()
        scheduler.receive_msg = lambda **kwargs: []
        scheduler._schedule_next_batch = lambda: None
        scheduler.overlap_loop(self.data(req, 4, "discard"))
        self.assertFalse(scheduler._retired_uids)


if __name__ == "__main__":
    unittest.main()
