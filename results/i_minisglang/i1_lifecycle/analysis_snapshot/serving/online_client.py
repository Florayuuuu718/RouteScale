"""Bounded, event-logged SSE client. Client states never stand in for GPU states."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Callable

import httpx

TERMINAL = frozenset({"finished", "cancelled", "failed", "rejected"})


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OnlineRequest:
    request_id: str
    payload: dict
    arrival_s: float = 0
    timeout_s: float = 30
    cancel_after_s: float | None = None
    disconnect_after_chunks: int | None = None
    external_id: str | None = None


@dataclass
class RequestRecord:
    request_id: str
    payload_sha256: str
    state: str = "created"
    events: list[dict] = field(default_factory=list)
    content: str = ""
    sse_chunks: int = 0
    response_id: str | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    reason: str | None = None
    _sink: Callable | None = field(default=None, repr=False)

    def event(self, name: str, **data):
        row = dict(request_id=self.request_id, event=name,
                   monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(), **data)
        self.events.append(row)
        if self._sink:
            self._sink(row)

    def transition(self, state: str, **data):
        allowed = {"created": {"queued", *TERMINAL}, "queued": {"running", *TERMINAL},
                   "running": TERMINAL}
        if state not in allowed.get(self.state, set()):
            raise ValueError(f"Invalid lifecycle: {self.state} -> {state}")
        self.state = state
        self.reason = data.get("reason", self.reason)
        self.event(state, **data)

    def export(self):
        return {key: value for key, value in vars(self).items() if key != "_sink"}


async def sse_messages(response):
    """Honor SSE blank lines, multi-line data and comments across network chunks."""
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


class OnlineClient:
    def __init__(self, base_url: str, concurrency: int = 4, sink=None, transport=None):
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.base_url, self.concurrency = base_url.rstrip("/"), concurrency
        self.sink, self.transport = sink, transport

    async def run(self, requests: list[OnlineRequest]):
        ids = [r.request_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate request ID in workload")
        if any(r.arrival_s < 0 or r.timeout_s <= 0 or
               (r.cancel_after_s is not None and r.cancel_after_s <= 0) or
               (r.disconnect_after_chunks is not None and r.disconnect_after_chunks < 1)
               for r in requests):
            raise ValueError("Invalid arrival, deadline or cancellation budget")
        semaphore = asyncio.Semaphore(self.concurrency)
        origin = time.monotonic()
        async with httpx.AsyncClient(base_url=self.base_url, timeout=None,
                                    transport=self.transport, trust_env=False) as http:
            async def one(req):
                record = RequestRecord(req.request_id, canonical_hash(req.payload), _sink=self.sink)
                record.event("created", planned_arrival_monotonic_s=origin + req.arrival_s)
                await asyncio.sleep(max(0, origin + req.arrival_s - time.monotonic()))
                record.transition("queued", boundary="client_semaphore")
                try:
                    # Budget includes semaphore wait and HTTP; scheduled arrival is the origin.
                    async with asyncio.timeout_at(origin + req.arrival_s + req.timeout_s):
                        async with semaphore:
                            record.event("http_send")
                            if req.cancel_after_s is not None:
                                try:
                                    async with asyncio.timeout(req.cancel_after_s):
                                        await self._stream(http, req, record)
                                except TimeoutError:
                                    record.transition("cancelled", reason="scheduled_connection_close")
                            else:
                                await self._stream(http, req, record)
                except TimeoutError:
                    if record.state not in TERMINAL:
                        record.transition("cancelled", reason="deadline")
                except asyncio.CancelledError:
                    if record.state not in TERMINAL:
                        record.transition("cancelled", reason="task_cancel")
                    raise
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    if record.state not in TERMINAL:
                        record.transition("failed", reason=f"{type(exc).__name__}: {exc}")
                return record.export()
            return await asyncio.gather(*(one(req) for req in requests))

    async def _stream(self, http, req, record):
        async with http.stream("POST", "/v1/chat/completions", json={**req.payload, "stream": True},
                               headers={"X-Request-ID": req.external_id or req.request_id}) as response:
            record.http_status = response.status_code
            record.event("http_headers", status=response.status_code)
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:2048]
                record.transition("rejected" if response.status_code < 500 else "failed",
                                  reason=f"HTTP {response.status_code}", response=body)
                return
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ValueError("Expected text/event-stream")
            async for data in sse_messages(response):
                if data == "[DONE]":
                    if record.finish_reason is None:
                        raise ValueError("DONE without finish_reason")
                    record.transition("finished")
                    return
                item = json.loads(data)
                if not isinstance(item, dict):
                    raise ValueError("SSE JSON must be an object")
                choices = item.get("choices", [])
                if not isinstance(choices, list):
                    raise ValueError("SSE choices must be a list")
                if not choices:  # e.g. an optional usage-only frame
                    continue
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                    raise ValueError("Malformed SSE choice/delta")
                text = choice.get("delta", {}).get("content") or ""
                if not isinstance(text, str):
                    raise ValueError("SSE content must be text")
                if record.state == "queued":
                    record.transition("running", boundary="first_sse_choice_not_gpu_start")
                record.response_id = item.get("id", record.response_id)
                record.sse_chunks += 1
                if text and not record.content:
                    record.event("first_text", boundary="first_nonempty_text_not_token")
                record.content += text
                record.finish_reason = choice.get("finish_reason") or record.finish_reason
                record.event("sse_choice", text=text, finish_reason=choice.get("finish_reason"),
                             response_id=record.response_id)
                if req.disconnect_after_chunks and record.sse_chunks >= req.disconnect_after_chunks:
                    record.transition("cancelled", reason="intentional_disconnect")
                    return
            raise ValueError("SSE ended without DONE")
