import asyncio
import copy

import httpx
import pytest

from serving.i3_metrics import analyze, distribution, percentile
from serving.i3_workloads import schedules
from serving.online_client import OnlineClient, OnlineRequest


def fixture_record(state='finished', late_ms=0):
    ns=lambda ms:int(ms*1e6)
    events=[dict(event='created',planned_arrival_monotonic_s=1.,monotonic_ns=ns(900)),
        dict(event='queued',monotonic_ns=ns(1000)),dict(event='http_send',monotonic_ns=ns(1000+late_ms))]
    if state=='finished':
        events += [dict(event='first_text',monotonic_ns=ns(1100)),
            dict(event='sse_choice',text='ab',monotonic_ns=ns(1100)),
            dict(event='sse_choice',text='cd',monotonic_ns=ns(1120))]
    events += [dict(event=state,monotonic_ns=ns(1200))]
    record=dict(request_id='r',state=state,events=events,http_status=200 if state=='finished' else 429,
        reason=None,finish_reason='length' if state=='finished' else None)
    backend=[dict(event='frontend_new_user',external_id='r',uid=0,monotonic_ns=ns(1001)),
        dict(event='scheduler_receive',kind='UserMsg',uid=0,monotonic_ns=ns(1010)),
        dict(event='i3_batch',phase='prefill',requests=[dict(uid=0,cached=40,extend=60)],monotonic_ns=ns(1030)),
        *[dict(event='scheduler_token',uid=0,monotonic_ns=ns(t)) for t in (1090,1100,1110,1120)]]
    workload=dict(requests=[dict(request_id='r',kind='short',prompt_tokens=100,output_budget=4)])
    config=dict(latency_objectives_s=dict(first_text=1,e2e=8,amortized_tpot=.1),arrival_jitter_s=dict(p95_max=.05,max=.25))
    return workload,[record],backend,config


def test_percentiles_preserve_empty_and_single_samples():
    assert distribution([])['p99'] is None
    assert percentile([7],.99)==7
    assert percentile([0,10],.95)==pytest.approx(9.5)


def test_goodput_includes_offered_window_and_actual_tokens():
    w,r,e,c=fixture_record();a=analyze(w,r,e,.5,c)
    m=a['metrics'];d=a['requests'][0]
    assert m['drain_inclusive_window_s']==2
    assert m['goodput_rps']==.5 and m['offered_SLO_success_fraction']==1
    assert m['completed_output_tokens']==4 and m['computed_prefill_queries']==60
    assert d['scheduler_queue_ms']==20 and m['prompt_token_cache_hit_fraction']==.4
    assert d['backend_emission_itl_ms']==[10,10,10]
    assert d['sse_text_chunk_intervals_ms']==[20]  # Text chunks are not token timestamps.


def test_rejected_requests_cannot_improve_SLO_fraction_or_supply_zero_latency():
    w,r,e,c=fixture_record('rejected');m=analyze(w,r,[],1,c)['metrics']
    assert m['terminal_counts']['rejected']==1 and m['goodput_rps']==0
    assert m['client_text_ttft_ms']['n']==0 and m['client_text_ttft_ms']['p50'] is None
    assert m['completed_output_token_throughput']==0


def test_drain_is_included_in_rate_denominator():
    w,r,e,c=fixture_record();r[0]['events'][-1]['monotonic_ns']=5_000_000_000
    m=analyze(w,r,e,10,c)['metrics']
    assert m['drain_inclusive_window_s']==4
    assert m['completed_request_throughput']==.25
    assert m['goodput_rps']==0  # Long amortized TPOT fails even though completion succeeded.


@pytest.mark.parametrize('mutation',['missing','duplicate','terminal'])
def test_lost_or_duplicate_accounting_is_rejected(mutation):
    w,r,e,c=fixture_record()
    if mutation=='missing':r=[]
    elif mutation=='duplicate':r=r*2
    else:r[0]['events'].append(dict(event='failed',monotonic_ns=1_300_000_000))
    with pytest.raises(ValueError):analyze(w,r,e,1,c)


def test_late_open_arrivals_invalidate_measurement():
    w,r,e,c=fixture_record(late_ms=100)
    assert not analyze(w,r,e,1,c)['metrics']['arrival_schedule_valid']


def test_rate_scaled_schedules_share_payload_order_and_random_trace():
    c=dict(seed=3,rate_rule=dict(low_min_capacity_multiplier=.25,near_min_capacity_multiplier=.9,
        overload_max_capacity_multiplier=1.5),latency_objectives_s=dict(first_text=1,e2e=8,amortized_tpot=.1))
    w=dict(requests=[dict(request_id=str(i)) for i in range(48)])
    d=schedules(w,dict(radix=2,graph124=8),c)
    assert d==schedules(w,dict(radix=2,graph124=8),c)
    for name,rows in d['traces'].items():
        assert [r['request_id'] for r in rows]==[str(i) for i in range(48)]
        assert all(a['arrival_s']<b['arrival_s'] for a,b in zip(rows,rows[1:]))
        assert rows[-1]['arrival_s']*d['rates_rps'][name]==pytest.approx(47)
    assert d['rates_rps']['overload']>8


def test_open_client_does_not_hold_requests_behind_a_completion():
    async def exercise():
        entered=0;barrier=asyncio.Event();n=8
        async def respond(request):
            nonlocal entered
            entered+=1
            if entered==n:barrier.set()
            await asyncio.wait_for(barrier.wait(),1)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=
                b'data: {"id":"x","choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
        rows=await OnlineClient('http://unit',concurrency=n,transport=httpx.MockTransport(respond)).run(
            [OnlineRequest(str(i),dict(prompt='x'),timeout_s=2) for i in range(n)])
        assert all(r['state']=='finished' for r in rows) and entered==n
    asyncio.run(exercise())
