import asyncio
import httpx
import unittest

from serving.framework_gate_client import GateClient
from serving.online_client import OnlineRequest


class Stream(httpx.AsyncByteStream):
    def __init__(self, finish=False):
        self.finish = finish
        self.entered = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.entered.set()
        if self.finish:
            yield b'data: {"choices":[{"text":"hi","finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
        else:
            await asyncio.Event().wait()
            yield b''

    async def aclose(self):
        self.closed = True


class GateClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, stream, gate):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream))
        return GateClient('http://fixture', concurrency=1, close_gate=gate, transport=transport)

    async def test_observation_gate_closes_real_stream_once(self):
        stream = Stream()
        gate = asyncio.Event()
        task = asyncio.create_task(self.client(stream, gate).run([OnlineRequest('queued', {'prompt': 'x'}, timeout_s=1)]))
        await stream.entered.wait()
        self.assertFalse(task.done())
        gate.set()
        row = (await task)[0]
        self.assertTrue(stream.closed)
        self.assertEqual(row['state'], 'cancelled')
        self.assertEqual(row['reason'], 'observed_queue_connection_close')
        self.assertEqual(sum(e['event'] == 'cancelled' for e in row['events']), 1)

    async def test_native_finish_before_gate_is_preserved(self):
        stream = Stream(finish=True)
        row = (await self.client(stream, asyncio.Event()).run([OnlineRequest('finished', {'prompt': 'x'})]))[0]
        self.assertEqual(row['state'], 'finished')
        self.assertEqual(row['content'], 'hi')
        self.assertTrue(stream.closed)

    async def test_deadline_closes_stream_without_observation(self):
        stream = Stream()
        row = (await self.client(stream, asyncio.Event()).run([OnlineRequest('deadline', {'prompt': 'x'}, timeout_s=.03)]))[0]
        self.assertEqual(row['state'], 'cancelled')
        self.assertEqual(row['reason'], 'deadline')
        self.assertTrue(stream.closed)
