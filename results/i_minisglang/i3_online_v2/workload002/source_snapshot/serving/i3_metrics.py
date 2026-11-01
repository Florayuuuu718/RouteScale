"""Join offered requests with HTTP and backend evidence; no missing-as-zero latency."""
from collections import Counter, defaultdict
import math
import statistics

from serving.online_client import TERMINAL


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo]*(hi-x)+values[hi]*(x-lo) if lo != hi else values[lo]


def distribution(values):
    return dict(n=len(values), p50=percentile(values,.5), p95=percentile(values,.95),
                p99=percentile(values,.99), max=max(values) if values else None)


def analyze(workload, records, events, rate, config):
    expected = {r['request_id']:r for r in workload['requests']}
    if len(expected) != len(records) or set(expected) != {r['request_id'] for r in records}:
        raise ValueError('Every offered request must be represented exactly once')
    assigned = {e['external_id']:e['uid'] for e in events if e['event']=='frontend_new_user'}
    local = defaultdict(list)
    for e in events:
        if 'uid' in e:
            local[e['uid']].append(e)
    first_batch, hits, prefill_count = {}, {}, Counter()
    computed_queries = 0
    for e in events:
        if e['event'] != 'i3_batch':
            continue
        for r in e['requests']:
            first_batch.setdefault(r['uid'], e['monotonic_ns'])
            if e['phase']=='prefill':
                hits.setdefault(r['uid'], r['cached'])
                prefill_count[r['uid']] += 1
    detail = []
    for record in records:
        rid = record['request_id']; request = expected[rid]
        ev = record['events']; terminal = [e for e in ev if e['event'] in TERMINAL]
        if len(terminal)!=1 or record['state'] not in TERMINAL:
            raise ValueError('Exactly one terminal per offered request required')
        created = next(e for e in ev if e['event']=='created')
        planned = round(created['planned_arrival_monotonic_s']*1e9)
        send = next((e['monotonic_ns'] for e in ev if e['event']=='http_send'), None)
        text = next((e['monotonic_ns'] for e in ev if e['event']=='first_text'), None)
        uid = assigned.get(rid)
        tokens = [e for e in local[uid] if e['event']=='scheduler_token'] if uid is not None else []
        received = next((e['monotonic_ns'] for e in local[uid]
            if e['event']=='scheduler_receive' and e['kind']=='UserMsg'), None)
        chunk_times = [e['monotonic_ns'] for e in ev if e['event']=='sse_choice' and e.get('text')]
        end = terminal[0]['monotonic_ns']
        completed = record['state']=='finished'
        e2e = (end-planned)/1e6 if completed else None
        ttft = (text-planned)/1e6 if text is not None and completed else None
        tpot = (end-text)/1e6/(len(tokens)-1) if completed and text is not None and len(tokens)>1 else None
        slo = config['latency_objectives_s']
        good = completed and ttft is not None and tpot is not None and (
            ttft<=slo['first_text']*1000 and e2e<=slo['e2e']*1000 and tpot<=slo['amortized_tpot']*1000)
        detail.append(dict(request_id=rid, kind=request['kind'], state=record['state'], uid=uid,
            http_status=record['http_status'], reason=record['reason'], finish_reason=record['finish_reason'],
            planned_ns=planned, send_ns=send, terminal_ns=end,
            send_lateness_ms=(send-planned)/1e6 if send is not None else None,
            client_text_ttft_ms=ttft, e2e_ms=e2e, amortized_tpot_ms=tpot, satisfies_all_SLOs=bool(good),
            scheduler_queue_ms=(first_batch[uid]-received)/1e6 if uid in first_batch and received is not None else None,
            backend_emission_itl_ms=[(b['monotonic_ns']-a['monotonic_ns'])/1e6 for a,b in zip(tokens,tokens[1:])],
            sse_text_chunk_intervals_ms=[(b-a)/1e6 for a,b in zip(chunk_times,chunk_times[1:])],
            backend_generated_tokens=len(tokens), completed_output_tokens=len(tokens) if completed else 0,
            admitted_input_tokens=request['prompt_tokens'] if uid is not None else 0,
            cached_prompt_tokens=hits.get(uid,0), prefill_batches=prefill_count[uid] if uid is not None else 0))
    offered_uids={r['uid'] for r in detail if r['uid'] is not None}
    for e in events:
        if e['event']=='i3_batch' and e['phase']=='prefill':
            computed_queries += sum(r['extend'] for r in e['requests'] if r['uid'] in offered_uids)
    origin=min(r['planned_ns'] for r in detail)
    window=len(detail)/rate
    makespan=max(window,(max(r['terminal_ns'] for r in detail)-origin)/1e9)
    counts=Counter(r['state'] for r in detail)
    admitted=sum(r['admitted_input_tokens'] for r in detail)
    cached=sum(r['cached_prompt_tokens'] for r in detail)
    measured_batches=[e for e in events if e['event']=='i3_batch' and any(r['uid'] in offered_uids for r in e['requests'])]
    jitter=[r['send_lateness_ms'] for r in detail if r['send_lateness_ms'] is not None]
    metrics=dict(offered=len(detail), terminal_counts={s:counts[s] for s in sorted(TERMINAL)},
        offered_rate_rps=rate, offered_window_s=window, drain_inclusive_window_s=makespan,
        completed_request_throughput=counts['finished']/makespan,
        goodput_rps=sum(r['satisfies_all_SLOs'] for r in detail)/makespan,
        offered_SLO_success_fraction=sum(r['satisfies_all_SLOs'] for r in detail)/len(detail),
        admitted_input_tokens=admitted, admitted_input_token_throughput=admitted/makespan,
        completed_output_tokens=sum(r['completed_output_tokens'] for r in detail),
        completed_output_token_throughput=sum(r['completed_output_tokens'] for r in detail)/makespan,
        backend_generated_tokens_including_partial=sum(r['backend_generated_tokens'] for r in detail),
        computed_prefill_queries=computed_queries, cached_prompt_tokens=cached,
        prompt_token_cache_hit_fraction=cached/admitted if admitted else None,
        repeated_prefill_requests=sum(r['prefill_batches']>1 for r in detail),
        preemption='unsupported by pinned scheduler; no running-request preemption mechanism',
        batch_sizes=sorted({len(e['requests']) for e in measured_batches}),
        graph_replay_batches=sum(e['event']=='i3_graph_replay' and bool(set(e['uids'])&offered_uids) for e in events),
        arrival_send_lateness_ms=distribution(jitter),
        client_text_ttft_ms=distribution([r['client_text_ttft_ms'] for r in detail if r['client_text_ttft_ms'] is not None]),
        e2e_ms=distribution([r['e2e_ms'] for r in detail if r['e2e_ms'] is not None]),
        amortized_tpot_ms=distribution([r['amortized_tpot_ms'] for r in detail if r['amortized_tpot_ms'] is not None]),
        scheduler_queue_ms=distribution([r['scheduler_queue_ms'] for r in detail if r['scheduler_queue_ms'] is not None]),
        backend_emission_itl_ms=distribution([v for r in detail for v in r['backend_emission_itl_ms']]),
        sse_text_chunk_intervals_ms=distribution([v for r in detail for v in r['sse_text_chunk_intervals_ms']]))
    metrics['arrival_schedule_valid']=(len(jitter)==len(detail) and max(jitter)<=config['arrival_jitter_s']['max']*1000
        and percentile(jitter,.95)<=config['arrival_jitter_s']['p95_max']*1000)
    return dict(metrics=metrics, requests=detail)
