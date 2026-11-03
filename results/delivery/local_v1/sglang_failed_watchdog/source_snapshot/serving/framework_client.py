"""Adapt native completion chunks to the frozen I3 transport-event schema.

The original prompt is sent to /v1/completions unchanged. Raw native frames are
preserved before mapping choices.text to the legacy delta.content reader.
"""
from contextlib import asynccontextmanager
from copy import deepcopy
import json

from serving.online_client import OnlineClient


class CompletionClient(OnlineClient):
    async def _stream(self,http,req,record):
        if not req.stream:raise ValueError('Use the official nonstream endpoint directly')
        class Response:
            def __init__(self,inner):self.inner=inner
            def __getattr__(self,name):return getattr(self.inner,name)
            async def aiter_lines(self):
                async for line in self.inner.aiter_lines():
                    if line.startswith('data:') and line[5:].strip()!='[DONE]':
                        item=json.loads(line[5:])
                        if not isinstance(item,dict) or not isinstance(item.get('choices',[]),list):
                            raise ValueError('Malformed native completion frame')
                        record.event('framework_frame',native=deepcopy(item))
                        if any(not isinstance(choice,dict) for choice in item.get('choices',[])):
                            raise ValueError('Malformed native completion choice')
                        for choice in item.get('choices',[]):choice['delta']=dict(content=choice.get('text') or '')
                        line='data: '+json.dumps(item)
                    yield line
        class Bridge:
            @asynccontextmanager
            async def stream(self,method,path,**options):
                async with http.stream(method,'/v1/completions',**options) as response:yield Response(response)
        return await super()._stream(Bridge(),req,record)
