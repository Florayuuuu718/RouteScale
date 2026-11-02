"""Qualified I3 computation with separate HTTP monitoring routes."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.launch_i3_service
from minisgl.server import api_server as api
from serving.delivery_metrics import HTTPMetrics,Registry
from fastapi.responses import PlainTextResponse
registry=Registry()
api.app.add_middleware(HTTPMetrics,registry=registry)


@api.app.get('/live')
async def live():return dict(status='alive')


@api.app.get('/metrics',response_class=PlainTextResponse)
async def metrics():return registry.render(api.get_global_state())


if __name__=='__main__':
    from minisgl.server import launch_server
    launch_server()
