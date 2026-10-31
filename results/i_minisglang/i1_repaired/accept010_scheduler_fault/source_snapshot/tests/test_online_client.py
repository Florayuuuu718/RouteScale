import asyncio
import json
import unittest

import httpx

from serving.online_client import OnlineClient, OnlineRequest, RequestRecord, TERMINAL, sse_messages


def frame(text="Hi", finish=None):
    return ("data: " + json.dumps({"id": "cmpl-7", "choices": [{"delta": {"content": text},
             "finish_reason": finish}]}) + "\n\n").encode()


class FakeStream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks, self.delay, self.closed = chunks, delay, False
    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk
    async def aclose(self):
        self.closed = True


class OnlineClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, chunks=None, delay=0, status=200, concurrency=2):
        streams = []
        def handle(request):
            stream = FakeStream(chunks if chunks is not None else
                                [frame(), frame("", "stop"), b"data: [DONE]\n\n"], delay)
            streams.append(stream)
            return httpx.Response(status, stream=stream, headers={"content-type": "text/event-stream"})
        return OnlineClient("http://fake", concurrency, transport=httpx.MockTransport(handle)), streams

    def assert_terminal_once(self, record, expected):
        self.assertEqual(record["state"], expected)
        self.assertEqual(sum(e["event"] in TERMINAL for e in record["events"]), 1)

    async def test_finished_requires_done_and_finish_reason(self):
        client, streams = self.client()
        record, = await client.run([OnlineRequest("ok", {})])
        self.assert_terminal_once(record, "finished")
        self.assertEqual(record["content"], "Hi")
        self.assertEqual(record["response_id"], "cmpl-7")
        self.assertTrue(streams[0].closed)

    async def test_truncated_and_malformed_stream_fail(self):
        for chunks in ([frame()], [b"data: [DONE]\n\n"], [b"data: {bad}\n\n"]):
            client, _ = self.client(chunks)
            record, = await client.run([OnlineRequest("broken", {})])
            self.assert_terminal_once(record, "failed")

    async def test_http_rejections_and_server_failure(self):
        for status, terminal in ((422, "rejected"), (503, "failed")):
            client, _ = self.client([b"bad input"], status=status)
            record, = await client.run([OnlineRequest("bad", {})])
            self.assert_terminal_once(record, terminal)
            self.assertEqual(record["http_status"], status)

    async def test_wrong_sse_json_shapes_are_failed_terminals(self):
        for item in (None, [], {"choices": {}}, {"choices": [None]},
                     {"choices": [{"delta": {"content": [1]}}]}):
            client, _ = self.client([("data: " + json.dumps(item) + "\n\n").encode()])
            record, = await client.run([OnlineRequest("shape", {})])
            self.assert_terminal_once(record, "failed")

    async def test_explicit_sse_error_cannot_be_a_success(self):
        client, _ = self.client([b'data: {"error": {"message": "worker exited", "code": 503}}\n\n'])
        record, = await client.run([OnlineRequest("fault", {})])
        self.assert_terminal_once(record, "failed")
        self.assertIn("worker exited", record["reason"])
        self.assertEqual(sum(e["event"] == "server_error" for e in record["events"]), 1)

    async def test_nonstream_completion_and_malformed_finish(self):
        for reason, expected in (("length", "finished"), (None, "failed")):
            def handle(request):
                self.assertFalse(json.loads(request.content)["stream"])
                return httpx.Response(200, json={"id": "chatcmpl-2", "choices": [
                    {"message": {"content": "OK"}, "finish_reason": reason}]})
            client = OnlineClient("http://fake", transport=httpx.MockTransport(handle))
            record, = await client.run([OnlineRequest("plain", {}, stream=False)])
            self.assert_terminal_once(record, expected)

    async def test_nonstream_bad_shapes_still_produce_one_terminal(self):
        for item in (None, [], {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]}):
            client = OnlineClient("http://fake", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=item)))
            record, = await client.run([OnlineRequest("bad_plain", {}, stream=False)])
            self.assert_terminal_once(record, "failed")

    async def test_disconnect_closes_response(self):
        client, streams = self.client(delay=.005)
        record, = await client.run([OnlineRequest("close", {}, disconnect_after_chunks=1)])
        self.assert_terminal_once(record, "cancelled")
        self.assertEqual(record["reason"], "intentional_disconnect")
        self.assertTrue(streams[0].closed)

    async def test_cancel_before_first_chunk(self):
        client, streams = self.client(delay=.2)
        record, = await client.run([OnlineRequest("cancel", {}, cancel_after_s=.02)])
        self.assert_terminal_once(record, "cancelled")
        self.assertEqual(record["sse_chunks"], 0)
        self.assertTrue(streams[0].closed)

    async def test_deadline_includes_semaphore_wait(self):
        client, streams = self.client(delay=.05, concurrency=1)
        records = await client.run([OnlineRequest("first", {}, timeout_s=1),
                                    OnlineRequest("queued", {}, timeout_s=.01)])
        self.assert_terminal_once(records[0], "finished")
        self.assert_terminal_once(records[1], "cancelled")
        self.assertFalse(any(e["event"] == "http_send" for e in records[1]["events"]))
        self.assertEqual(len(streams), 1)

    async def test_duplicate_ids_rejected_before_http(self):
        client, streams = self.client()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            await client.run([OnlineRequest("x", {}), OnlineRequest("x", {})])
        self.assertEqual(streams, [])

    async def test_arrival_schedule(self):
        client, _ = self.client()
        record, = await client.run([OnlineRequest("later", {}, arrival_s=.015)])
        created, queued = record["events"][:2]
        self.assertGreaterEqual(queued["monotonic_ns"] / 1e9, created["planned_arrival_monotonic_s"])

    async def test_sse_fragmentation_comments_multiline(self):
        stream = FakeStream([b": comment\r\nda", b"ta: one\r\ndata: two\r", b"\n\r\n"])
        response = httpx.Response(200, stream=stream)
        self.assertEqual([item async for item in sse_messages(response)], ["one\ntwo"])
        await response.aclose()

    def test_terminal_state_cannot_change(self):
        record = RequestRecord("x", "hash")
        record.transition("failed")
        with self.assertRaises(ValueError):
            record.transition("finished")


if __name__ == "__main__":
    unittest.main()
