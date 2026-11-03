import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from serving.delivery import DEFAULT,check_ports,command,identity,load,read,request_stop,sha,stop_child,write
from serving.delivery_metrics import HTTPMetrics,Registry


class DeliveryTests(unittest.TestCase):
    def test_qualified_scope_rejects_unvalidated_tp_dtype_and_admission(self):
        original=read(DEFAULT)
        for key,value in [('tensor_parallel_size',2),('upper_fp32_layers',8),('dtype','bfloat16'),
                ('max_total_requests',999),('host','0.0.0.0'),('shutdown_s',60),('port',65535)]:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'c.json';write(p,dict(original,**{key:value}))
                with self.assertRaises(ValueError):load(p)

    def test_bounded_health_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.json';write(p,dict(read(DEFAULT),startup_s=0))
            with self.assertRaises(ValueError):load(p)

    def test_command_reuses_qualified_launcher_flags(self):
        args=command(load())
        self.assertIn('launch_delivery_service.py',Path(args[1]).name)
        self.assertEqual(args[args.index('--dtype')+1],'float16')
        self.assertEqual(args[args.index('--max-total-requests')+1],'8')
        self.assertEqual(args[-4:],['--cuda-graph-bs','1','2','4'])

    def test_ports_reject_another_listener_without_touching_it(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            if port==65535:self.skipTest('ephemeral range edge')
            with self.assertRaises(OSError):check_ports('127.0.0.1',port)
            self.assertEqual(s.getsockname()[1],port)

    def test_identity_detects_absent_pid(self):
        self.assertIsNotNone(identity(os.getpid()))
        self.assertIsNone(identity(99999999))

    def test_stale_stop_refuses_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp)/'status.json',dict(state='ready',supervisor_pid=os.getpid(),supervisor_identity='wrong',config=load()))
            with patch('serving.delivery.os.kill') as kill:
                with self.assertRaisesRegex(ValueError,'Stale'):request_stop(tmp)
                kill.assert_not_called()

    def test_stop_is_idempotent_after_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=dict(state='stopped');write(Path(tmp)/'status.json',r)
            self.assertEqual(request_stop(tmp),r)

    def test_graceful_owned_subprocess_teardown(self):
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        r=stop_child(p,1)
        self.assertFalse(r['forced_cleanup']);self.assertEqual(r['returncode'],-signal.SIGTERM)

    def test_ignored_sigterm_is_killed_within_budget(self):
        p=subprocess.Popen([sys.executable,'-u','-c',
            'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(30)'],
            start_new_session=True,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(),'ready');start=time.monotonic();r=stop_child(p,.1)
            self.assertTrue(r['forced_cleanup']);self.assertEqual(r['returncode'],-signal.SIGKILL)
            self.assertLess(time.monotonic()-start,2)
        finally:
            p.stdout.close()
            if p.poll() is None:p.kill();p.wait()

    def test_atomic_json_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'status.json';write(p,{'state':'ready'});before=sha(p)
            write(p,{'state':'stopped'});self.assertNotEqual(sha(p),before)
            self.assertFalse(p.with_suffix('.json.tmp').exists())


class MetricsTests(unittest.IsolatedAsyncioTestCase):
    async def call_app(self,app,path='/v1/chat/completions',messages=None):
        registry=Registry();wrapper=HTTPMetrics(app,registry);sent=[]
        messages=list(messages or [{'type':'http.request','body':b''}])
        async def receive():return messages.pop(0)
        async def send(msg):sent.append(msg)
        await wrapper(dict(type='http',path=path,method='POST'),receive,send)
        return registry,sent

    async def test_sse_entire_transport_lifetime_and_429(self):
        async def app(scope,receive,send):
            await send(dict(type='http.response.start',status=429))
            await send(dict(type='http.response.body',body=b'busy'))
        registry,_=await self.call_app(app)
        self.assertEqual(registry.active,0)
        self.assertEqual(registry.requests[('/v1/chat/completions','POST','429','completed')],1)

    async def test_disconnect_is_not_labeled_successful_inference(self):
        async def app(scope,receive,send):
            await send(dict(type='http.response.start',status=200));await receive()
        r,_=await self.call_app(app,messages=[dict(type='http.disconnect')])
        self.assertEqual(r.requests[('/v1/chat/completions','POST','200','client_disconnected')],1)

    async def test_exception_releases_active_metric(self):
        registry=Registry()
        async def bad(*args):raise RuntimeError('boom')
        async def unused():return None
        with self.assertRaises(RuntimeError):await HTTPMetrics(bad,registry)(dict(type='http',path='/generate',method='POST'),unused,unused)
        self.assertEqual(registry.active,0);self.assertEqual(sum(registry.requests.values()),1)

    async def test_metric_route_has_no_self_scrape_feedback(self):
        async def app(scope,receive,send):return None
        r,_=await self.call_app(app,path='/metrics');self.assertFalse(r.requests)

    async def test_cardinality_is_bounded_for_arbitrary_paths(self):
        async def app(scope,receive,send):
            await send(dict(type='http.response.start',status=404));await send(dict(type='http.response.body',body=b''))
        r,_=await self.call_app(app,path='/user-specific-random-id')
        self.assertEqual(next(iter(r.requests))[0],'other')

    async def test_backend_admission_gauge_is_independent_of_http(self):
        class State:
            admitted={1,2};aborting={2};ack_map={};deadline_map={};workers=None;accepting=False
        text=Registry().render(State())
        self.assertIn('routescale_inference_admitted_current 2',text)
        self.assertIn('routescale_inference_frontend_buffers_current 0',text)
        self.assertIn('routescale_ready 0',text)


if __name__=='__main__':unittest.main()
