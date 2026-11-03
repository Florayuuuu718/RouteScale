from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    RequestEndMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # Preserve wire order, including end acknowledgments in mixed batches.
            for msg in pending_msg:
                if isinstance(msg, DetokenizeMsg):
                    reply, = detokenize_manager.detokenize([msg])
                    send_frontend.put(UserReply(msg.uid, reply, msg.finished, msg.finish_reason))
                elif isinstance(msg, RequestEndMsg):
                    detokenize_manager.forget(msg.uid)
                    send_frontend.put(UserReply(msg.uid, "", True,
                                               error=msg.error, status_code=msg.status_code))
                elif isinstance(msg, TokenizeMsg):
                    ids, = tokenize_manager.tokenize([msg])
                    send_backend.put(UserMsg(msg.uid, ids, msg.sampling_params))
                elif isinstance(msg, AbortMsg):
                    # Cleanup must wait for the scheduler's ordered acknowledgment;
                    # a local delete here could be undone by an in-flight token.
                    send_backend.put(AbortBackendMsg(uid=msg.uid))
                else:
                    raise TypeError(f"Unexpected tokenizer message: {type(msg)}")
    except KeyboardInterrupt:
        pass
    finally:
        recv_listener.stop()
        send_backend.stop()
        send_frontend.stop()
