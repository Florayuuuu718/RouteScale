import time
from types import SimpleNamespace
import unittest

from serving.delivery_warmup import prewarm


class State:
    def __init__(self,error=False):
        self.config=SimpleNamespace(request_timeout_s=20);self.deadline_map={};self.admitted=set();self.error=error;self.aborted=[]
    def new_user(self):self.deadline_map[0]=time.monotonic()+20;self.admitted.add(0);return 0
    async def start_request(self,uid,message):self.internal_budget=self.deadline_map[uid]-time.monotonic()
    async def wait_for_ack(self,uid):
        yield SimpleNamespace(error='broken' if self.error else None,finished=False)
        yield SimpleNamespace(error=None,finished=True)
        self.admitted.discard(uid)
    def cleanup_user(self,uid):self.deadline_map.pop(uid,None)
    async def abort_user(self,uid):self.aborted.append(uid);self.admitted.discard(uid)


class WarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_startup_request_uses_startup_budget(self):
        s=State();events=[]
        await prewarm(s,lambda uid:None,lambda name,**data:events.append((name,data)),600)
        self.assertGreater(s.internal_budget,590);self.assertEqual(s.config.request_timeout_s,20)
        self.assertFalse(s.admitted);self.assertFalse(s.deadline_map);self.assertEqual(len(events),2)

    async def test_warmup_error_aborts_and_does_not_publish_end(self):
        s=State(error=True);events=[]
        with self.assertRaisesRegex(RuntimeError,'broken'):await prewarm(s,lambda uid:None,lambda name,**data:events.append(name),600)
        self.assertEqual(s.aborted,[0]);self.assertNotIn('delivery_warmup_end',events);self.assertFalse(s.deadline_map)


if __name__=='__main__':unittest.main()
