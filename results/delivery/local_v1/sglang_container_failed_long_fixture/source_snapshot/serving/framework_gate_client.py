"""Close a real diagnostic HTTP stream only after an observed queue signal."""
import asyncio
from serving.framework_client import CompletionClient


class GateClient(CompletionClient):
    def __init__(self, *args, close_gate, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_gate = close_gate

    async def _stream(self, http, req, record):
        stream = asyncio.create_task(super()._stream(http, req, record))
        closer = asyncio.create_task(self.close_gate.wait())
        try:
            done, _ = await asyncio.wait((stream, closer), return_when=asyncio.FIRST_COMPLETED)
            if stream in done:
                await stream
            else:
                stream.cancel()
                try:
                    await stream
                except asyncio.CancelledError:
                    pass
                if record.state not in ('finished', 'failed', 'cancelled', 'rejected'):
                    record.transition('cancelled', reason='observed_queue_connection_close')
        finally:
            closer.cancel()
            if not stream.done():
                stream.cancel()
            await asyncio.gather(stream, closer, return_exceptions=True)
