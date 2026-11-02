"""Internal startup request before the ASGI lifespan publishes readiness."""
import asyncio
from contextlib import aclosing
import time


async def prewarm(state,make_message,emit,budget):
    uid=state.new_user();complete=False;tokens=0
    # Only this internal request uses the startup budget. Public requests keep
    # ServerArgs.request_timeout_s (20 s); no global configuration is changed.
    state.deadline_map[uid]=time.monotonic()+budget
    emit('delivery_warmup_start',uid=uid,budget_s=budget,public_request_timeout_s=state.config.request_timeout_s)
    try:
        async with asyncio.timeout(budget):
            await state.start_request(uid,make_message(uid))
            async with aclosing(state.wait_for_ack(uid)) as acks:
                async for ack in acks:
                    if ack.error:raise RuntimeError(ack.error)
                    tokens+=1
                    if ack.finished:complete=True
        if not complete:raise RuntimeError('Startup prefill did not finish')
    finally:
        state.cleanup_user(uid)
        if not complete:await asyncio.shield(state.abort_user(uid))
    if uid in state.admitted:raise RuntimeError('Internal startup request retained admission')
    emit('delivery_warmup_end',uid=uid,tokens=tokens,public_request_timeout_s=state.config.request_timeout_s)
