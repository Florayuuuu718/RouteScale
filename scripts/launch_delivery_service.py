"""Qualified I3 computation with separate HTTP monitoring routes."""
from pathlib import Path
from contextlib import asynccontextmanager
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.launch_i3_service
from minisgl.server import api_server as api
from serving.delivery_metrics import HTTPMetrics,Registry
from serving.delivery_warmup import prewarm
from serving.minisgl_observer import emit
from minisgl.message import TokenizeMsg
from minisgl.core import SamplingParams
from fastapi.responses import PlainTextResponse
registry=Registry()
api.app.add_middleware(HTTPMetrics,registry=registry)
native_lifespan=api.app.router.lifespan_context


@asynccontextmanager
async def lifespan(app):
    async with native_lifespan(app):
        state=api.get_global_state()
        def message(uid):
            return TokenizeMsg(uid=uid,text='A small girl walked into a forest and told her friends a happy story. '*8,
                sampling_params=SamplingParams(max_tokens=2,ignore_eos=True,temperature=0,top_k=1))
        await prewarm(state,message,emit,state.config.startup_timeout_s)
        yield


api.app.router.lifespan_context=lifespan


@api.app.get('/live')
async def live():return dict(status='alive')


@api.app.get('/metrics',response_class=PlainTextResponse)
async def metrics():return registry.render(api.get_global_state())


if __name__=='__main__':
    from minisgl.server import launch_server
    launch_server()
