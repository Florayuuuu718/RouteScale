from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
import unittest

from serving.framework_metrics import parse,NAMES
from serving.framework_client import CompletionClient
from serving.online_client import OnlineRequest,RequestRecord
from serving.sglang_framework import command,environment,load,runtime_bin


class NativeMetricsTests(unittest.TestCase):
    def test_missing_is_not_zero(self):
        out=parse('sglang:num_running_reqs{model_name="Qwen"} 0\n')
        self.assertEqual(out['metrics']['running'],0);self.assertIsNone(out['metrics']['queued'])
        self.assertIn('sglang:num_queue_reqs',out['missing'])
    def test_ttft_mean_uses_native_count_not_chunk_count(self):
        out=parse('sglang:time_to_first_token_seconds_sum 2\nsglang:time_to_first_token_seconds_count 4\n')
        self.assertEqual(out['metrics']['TTFT_mean_s'],.5)
    def test_labels_are_preserved_and_replicas_sum(self):
        out=parse('sglang:num_queue_reqs{tp_rank="0"} 1\nsglang:num_queue_reqs{tp_rank="1"} 2\n')
        self.assertEqual(out['metrics']['queued'],3);self.assertIn('tp_rank',out['samples'][0]['labels'])
    def test_nonfinite_is_not_reported_as_a_valid_gauge(self):
        self.assertIsNone(parse('sglang:num_running_reqs NaN\n')['metrics']['running'])
    def test_malformed_exporter_is_rejected(self):
        with self.assertRaises(ValueError):parse('unexpected metrics body')
    def test_fixed_native_profile_is_not_the_I3_mixed_precision_policy(self):
        c=load();args=command(c)
        self.assertEqual(c['version'],'0.5.20');self.assertEqual(c['server_args']['dtype'],'bfloat16')
        self.assertIn('--disable-prefill-cuda-graph',args);self.assertIn('--disable-decode-cuda-graph',args)
        self.assertNotIn('--disable-cuda-graph',args)
    def test_native_bin_and_current_unified_cache_are_explicit(self):
        c=load();env=environment(c)
        self.assertIn(str(runtime_bin(c)),env['PATH'].split(':'))
        self.assertTrue(env['SGLANG_CACHE_DIR'].endswith(c['cache_path']))
    def test_optional_crash_diagnostics_cannot_extend_the_shutdown_budget(self):
        c=load();env=environment(c)
        self.assertEqual(env['SGLANG_PYSPY_DUMP_BEFORE_CRASH'],'0')
        self.assertEqual(env['SGLANG_CUDA_COREDUMP_BEFORE_CRASH'],'0')
        self.assertEqual(c['shutdown_s'],20)
    def test_OCI_image_pinned_and_model_mount_readonly(self):
        root=Path(__file__).resolve().parents[2];base=root/'deploy/local/sglang'
        image=json.loads((base/'image.lock.json').read_text())
        self.assertEqual(len(image['linux_amd64_digest'].split(':')[1]),64)
        self.assertIn(image['linux_amd64_digest'],(base/'Dockerfile').read_text())
        volumes=json.loads((base/'compose.json').read_text())['services']['sglang']['volumes']
        self.assertTrue(next(v for v in volumes if v['target']=='/models/Qwen3-0.6B')['read_only'])


class CompletionSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self,frames):
        class Response:
            status_code=200;headers={'content-type':'text/event-stream'}
            async def aiter_lines(self):
                for frame in frames:
                    yield 'data: '+(frame if isinstance(frame,str) else json.dumps(frame));yield ''
        class HTTP:
            @asynccontextmanager
            async def stream(self,method,path,**opts):
                self.path=path;self.payload=opts['json'];yield Response()
        req=OnlineRequest('native',dict(model='test',prompt='unchanged prompt'));record=RequestRecord('native','hash');record.transition('queued')
        http=HTTP();await CompletionClient('http://127.0.0.1')._stream(http,req,record)
        return http,record
    async def test_native_text_maps_to_canonical_schema_and_raw_frame_stays_raw(self):
        frames=[dict(id='cmpl-1',choices=[dict(text='Hello',finish_reason=None)]),
            dict(id='cmpl-1',choices=[dict(text='',finish_reason='length')]),'[DONE]']
        http,r=await self.invoke(frames)
        self.assertEqual(http.path,'/v1/completions');self.assertEqual(http.payload['prompt'],'unchanged prompt')
        self.assertEqual(r.state,'finished');self.assertEqual(r.content,'Hello')
        native=next(e['native'] for e in r.events if e['event']=='framework_frame')
        self.assertNotIn('delta',native['choices'][0]);self.assertEqual(native['choices'][0]['text'],'Hello')
    async def test_incomplete_native_stream_is_not_finished(self):
        with self.assertRaisesRegex(ValueError,'without DONE'):await self.invoke([dict(choices=[dict(text='partial',finish_reason=None)])])
    async def test_malformed_native_choice_is_a_bounded_client_error(self):
        with self.assertRaisesRegex(ValueError,'Malformed native completion choice'):await self.invoke([dict(choices=['bad'])])


class SummaryTests(unittest.TestCase):
    def test_offer_denominator_and_chunk_boundary_fixture(self):
        from scripts.check_framework import summarize
        rows=[dict(state='finished',events=[dict(event='http_send',monotonic_ns=1_000_000_000),
            dict(event='first_text',monotonic_ns=1_500_000_000),dict(event='sse_choice',monotonic_ns=1_500_000_000,text='two tokens'),
            dict(event='sse_choice',monotonic_ns=1_800_000_000,text='another block')]),
            dict(state='failed',events=[]),dict(state='cancelled',events=[])]
        out=summarize(rows)
        self.assertEqual(out['offered'],3);self.assertEqual(out['terminal_counts'],dict(finished=1,failed=1,cancelled=1))
        self.assertEqual(out['first_text_TTFT_s']['median'],.5)
        self.assertAlmostEqual(out['SSE_nonempty_chunk_interval_s']['median'],.3)
        self.assertFalse(out['native_token_ITL_claimed'])
    def test_client_dashboard_is_missing_without_a_ledger(self):
        from serving.framework_metrics import client_counts
        self.assertIsNone(client_counts(None))
    def test_native_processed_is_not_used_to_invent_client_success(self):
        import tempfile
        from serving.framework_metrics import client_counts
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json';p.write_text(json.dumps([dict(state='failed'),dict(state='finished'),dict(state='cancelled')]))
            self.assertEqual(client_counts(p),dict(offered=3,finished=1,failed=1,rejected=0,cancelled=1))


if __name__=='__main__':unittest.main()
