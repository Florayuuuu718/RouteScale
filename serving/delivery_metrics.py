"""Small ASGI HTTP metrics; frontend gauges retain backend admission ownership."""
from collections import Counter
import time

ROUTES = {'/v1/chat/completions','/generate','/ready','/health','/live','/v1/models','/metrics'}


class Registry:
    def __init__(self):
        self.requests=Counter();self.seconds=Counter();self.active=0;self.started=time.monotonic()

    def render(self, state=None):
        rows=['# HTTP completion is transport completion, not a backend terminal.',
            f'routescale_http_active {self.active}',f'routescale_uptime_seconds {time.monotonic()-self.started:.6f}']
        for key,count in sorted(self.requests.items()):
            route,method,status,result=key;labels=f'route="{route}",method="{method}",status="{status}",result="{result}"'
            rows += [f'routescale_http_requests_total{{{labels}}} {count}',
                f'routescale_http_duration_seconds_sum{{{labels}}} {self.seconds[key]:.9f}']
        if state is not None:
            for field,name in [('admitted','admitted'),('aborting','aborting'),('ack_map','frontend_buffers'),('deadline_map','deadlines')]:
                rows.append(f'routescale_inference_{name}_current {len(getattr(state,field,()))}')
            workers=getattr(state,'workers',None)
            alive=sum(p.is_alive() for p in workers.processes) if workers else 0
            rows.append(f'routescale_workers_alive {alive}')
            rows.append(f'routescale_ready {int(bool(state.accepting and workers and not workers.failure()))}')
        return '\n'.join(rows)+'\n'


class HTTPMetrics:
    def __init__(self,app,registry): self.app=app;self.registry=registry

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['path']=='/metrics':return await self.app(scope,receive,send)
        route=scope['path'] if scope['path'] in ROUTES else 'other'
        method=scope.get('method','GET');method=method if method in ('GET','POST','HEAD','OPTIONS') else 'other'
        started=time.monotonic();status=500;result='error';disconnected=False
        self.registry.active+=1
        async def recv():
            nonlocal disconnected
            message=await receive()
            if message['type']=='http.disconnect':disconnected=True
            return message
        async def sent(message):
            nonlocal status,result
            if message['type']=='http.response.start':status=message['status']
            await send(message)
            if message['type']=='http.response.body' and not message.get('more_body',False):result='completed'
        try: await self.app(scope,recv,sent)
        finally:
            if disconnected:result='client_disconnected'
            key=(route,method,str(status),result);self.registry.active-=1
            self.registry.requests[key]+=1;self.registry.seconds[key]+=time.monotonic()-started
