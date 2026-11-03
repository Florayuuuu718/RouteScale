from __future__ import annotations

import asyncio
import json
import time
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(ge=1)
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = Field(default=16, ge=1)
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    deadline_map: Dict[int, float] = field(default_factory=dict)
    workers: object | None = None
    failure: str | None = None
    accepting: bool = True
    listener_task: asyncio.Task | None = None
    monitor_task: asyncio.Task | None = None
    closed: bool = False
    admitted: set[int] = field(default_factory=set)
    aborting: set[int] = field(default_factory=set)
    backend_done: set[int] = field(default_factory=set)

    def backend_terminal(self, uid: int):
        if uid in self.admitted:
            self.backend_done.add(uid)
            self.release_if_complete(uid)

    def release_if_complete(self, uid: int):
        # Slow HTTP consumers still own buffers after backend completion.
        # Cancellation closes the frontend first; normal completion can reverse
        # that order. Reuse admission only after BOTH sides have released it.
        if uid in self.backend_done and uid not in self.ack_map:
            self.admitted.discard(uid)
            self.aborting.discard(uid)
            self.backend_done.discard(uid)

    def ensure_available(self):
        if self.workers is not None and (failure := self.workers.failure()):
            self.mark_failed(failure)
        if not self.accepting:
            raise HTTPException(503, self.failure or "Service is shutting down")

    def mark_failed(self, reason: str):
        if self.failure is not None:
            return
        self.failure = reason
        self.accepting = False
        for uid in self.ack_map:
            self.ack_map[uid] = [UserReply(uid, "", True, error=reason, status_code=503)]
            self.event_map[uid].set()

    async def monitor(self):
        while self.accepting:
            if self.workers is not None and (failure := self.workers.failure()):
                self.mark_failed(failure)
                break
            await asyncio.sleep(0.05)
        if self.failure is not None and self.workers is not None:
            report = await asyncio.to_thread(self.workers.stop)
            logger.info("Worker fault cleanup: %s", report)
            self.admitted.clear()
            self.aborting.clear()
            self.backend_done.clear()

    def new_user(self) -> int:
        self.ensure_available()
        if len(self.admitted) >= self.config.max_total_requests:
            raise HTTPException(429, "Inference admission limit reached", headers={"Retry-After": "1"})
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.deadline_map[uid] = time.monotonic() + self.config.request_timeout_s
        self.admitted.add(uid)
        return uid

    async def listen(self):
        try:
            while True:
                msg = await self.recv_tokenizer.get()
                for msg in _unwrap_msg(msg):
                    # Normal replies are sent after resource release; cancellation
                    # replies are sent after the ordered detokenizer cleanup.
                    if msg.finished:
                        self.backend_terminal(msg.uid)
                    if msg.uid not in self.ack_map:
                        continue
                    self.ack_map[msg.uid].append(msg)
                    self.event_map[msg.uid].set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.mark_failed(f"Tokenizer channel failed: {exc}")

    def _create_listener_once(self):
        if not self.initialized:
            self.listener_task = asyncio.create_task(self.listen())
            self.monitor_task = asyncio.create_task(self.monitor())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        try:
            await asyncio.wait_for(self.send_tokenizer.put(msg), timeout=1.0)
        except Exception as exc:
            self.mark_failed(f"Tokenizer send failed: {exc}")
            raise HTTPException(503, self.failure) from exc

    async def wait_event(self, uid: int, request: Request | None = None):
        event = self.event_map[uid]
        while True:
            if time.monotonic() >= self.deadline_map[uid]:
                raise HTTPException(504, "Request deadline exceeded")
            if event.is_set():
                return
            if request is not None and await request.is_disconnected():
                raise asyncio.CancelledError
            try:
                await asyncio.wait_for(event.wait(), timeout=0.05)
            except TimeoutError:
                pass

    async def start_request(self, uid: int, msg: TokenizeMsg, request: Request | None = None):
        # Delay response headers until a token or an explicit validation error.
        # A disconnect while tokenizing/prefilling must also reach the scheduler.
        try:
            await self.send_one(msg)
            await self.wait_event(uid, request)
            first = self.ack_map[uid][0]
            if first.error:
                raise HTTPException(first.status_code, first.error)
        except BaseException:
            await asyncio.shield(self.abort_user(uid))
            raise

    async def wait_for_ack(self, uid: int, request: Request | None = None):
        event = self.event_map[uid]
        try:
            while True:
                await self.wait_event(uid, request)
                event.clear()
                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                for ack in pending:
                    yield ack
                    if ack.finished:
                        return
        finally:
            self.cleanup_user(uid)

    def cleanup_user(self, uid: int):
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        self.deadline_map.pop(uid, None)
        self.release_if_complete(uid)

    async def stream_generate(self, uid: int):
        async with aclosing(self.wait_for_ack(uid)) as acks:
            async for ack in acks:
                if ack.error:
                    yield f"data: {json.dumps({'error': {'message': ack.error, 'code': ack.status_code}})}\n\n".encode()
                    return
                yield f"data: {ack.incremental_output}\n".encode()
        yield b"data: [DONE]\n"

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        finish_reason = None
        async with aclosing(self.wait_for_ack(uid)) as acks:
            async for ack in acks:
                if ack.error:
                    yield f"data: {json.dumps({'error': {'message': ack.error, 'code': ack.status_code}})}\n\n".encode()
                    return
                delta = {}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                if ack.incremental_output:
                    delta["content"] = ack.incremental_output
                chunk = {
                    "id": f"cmpl-{uid}",
                    "object": "text_completion.chunk",
                    "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                if ack.finished:
                    finish_reason = ack.finish_reason
                    break
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        complete = False
        try:
            async with aclosing(generator):
                async for chunk in generator:
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    yield chunk
                complete = True
        except HTTPException as exc:
            yield f"data: {json.dumps({'error': {'message': exc.detail, 'code': exc.status_code}})}\n\n".encode()
        finally:
            self.cleanup_user(uid)
            if not complete:
                await asyncio.shield(self.abort_user(uid))

    async def abort_user(self, uid: int):
        self.cleanup_user(uid)
        if uid not in self.admitted or uid in self.aborting or not self.accepting or self.closed:
            return
        self.aborting.add(uid)
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    async def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.mark_failed("Service is shutting down")
        for task in (self.listener_task, self.monitor_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (self.listener_task, self.monitor_task)
                               if task is not None), return_exceptions=True)
        # Close sockets only after their consumers have left; linger=0 prevents
        # context.term from waiting forever for a dead tokenizer peer.
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()
        if self.workers is not None:
            report = await asyncio.to_thread(self.workers.stop)
            logger.info("Worker shutdown: %s", report)
        for uid in list(self.ack_map):
            self.cleanup_user(uid)
        self.admitted.clear()
        self.aborting.clear()
        self.backend_done.clear()


class ManagedStreamingResponse(StreamingResponse):
    """Own the request even when headers fail before the body iterator starts."""

    def __init__(self, *args, state: FrontendManager, uid: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.state, self.uid = state, uid

    async def __call__(self, scope, receive, send):
        try:
            return await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(self.state.abort_user(self.uid))


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_global_state()._create_listener_once()
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        await _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.start_request(uid,
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        ), request
    )

    return ManagedStreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        state=state, uid=uid,
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    get_global_state().ensure_available()
    return {"status": "ok"}


@app.get("/health")
@app.get("/ready")
async def readiness():
    return await v1_root()


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        if req.prompt is None:
            raise HTTPException(400, "Either 'messages' or 'prompt' must be provided")
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.start_request(uid,
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        ), request
    )

    if req.stream:
        return ManagedStreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            state=state, uid=uid,
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    finish_reason = None
    complete = False
    try:
        async with aclosing(state.wait_for_ack(uid, request)) as acks:
            async for ack in acks:
                if ack.error:
                    raise HTTPException(ack.status_code, ack.error)
                full_content += ack.incremental_output
                if ack.finished:
                    finish_reason = ack.finish_reason
        complete = True
    finally:
        state.cleanup_user(uid)
        if not complete:
            await asyncio.shield(state.abort_user(uid))

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    state.ensure_available()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.start_request(uid,
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )



async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        await get_global_state().shutdown()


def run_api_server(config: ServerArgs, start_backend: Callable, run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    try:
        _GLOBAL_STATE.workers = start_backend()
        logger.info(f"API server is ready to serve on {host}:{port}")
        if not run_shell:
            uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=1)
        else:
            asyncio.run(shell())
    finally:
        # Includes startup failure and interruption before ASGI lifespan starts.
        _GLOBAL_STATE.send_tokenizer.stop()
        _GLOBAL_STATE.recv_tokenizer.stop()
        if _GLOBAL_STATE.workers is not None:
            _GLOBAL_STATE.workers.stop()
