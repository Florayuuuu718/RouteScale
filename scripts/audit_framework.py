"""Independently re-open full SGLang transport, queue, KV and shutdown evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.delivery import identity,read,sha,write
from serving.framework_metrics import parse
from serving.online_client import TERMINAL
from scripts.check_framework import SCHEDULE,WORKLOAD,sha_text,summarize


def audit(directory):
    d=Path(directory);r=read(d/'run.json');s=read(d/'status.json');rows=read(d/'requests.json')
    checks=dict(run_passed=r['passed'],clean_checkout=r['checkout_clean'],doctor=s['doctor']['passed'],
        native_version=s['doctor']['environment']['packages']['sglang']=='0.5.20',
        GPU_released=s['GPU_end']=='0 MiB',bounded_shutdown=s['shutdown_s']<=20.5,
        no_forced_cleanup=not s['forced_cleanup'],supervisor_dead=identity(s['supervisor_pid']) is None,
        native_server_dead=identity(s['server_pid']) is None,
        expected_state=s['state']==('failed' if r['scenario']=='worker_fault' else 'stopped'),
        unique_ledger=len({x['request_id'] for x in rows})==len(rows),
        all_offered_terminal=all(x['state'] in TERMINAL for x in rows),
        actual_BF16=s['config']['server_args']['dtype']=='bfloat16',
        separate_numeric_scope=r['native_BF16_not_I3_numeric_qualification'])
    checks['actual_unified_radix_cache']='impl=UnifiedRadixCache' in (d/'native_server.txt').read_text()
    native_flushes=[]
    for line in (d/'native_server.txt').read_text().splitlines():
        if '{"event": "routescale_native_flush"' in line:native_flushes.append(json.loads(line[line.index('{'):]))
    if r.get('container'):
        # A built image has no .git. Host build/inspect evidence is required instead.
        build=read(d/'container_provenance.json')
        checks['clean_checkout']=build['clean_source_checkout'] and build['source_commit']==r['root_commit']
        bases={'sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4',
            'sha256:00f0ecbf74ff8f915020d5a40c4bc6a83f46cd7b83f47db51c7e204f0d8a3ec2'}
        checks['immutable_container_image']=build['image_id'].startswith('sha256:') and build['base_digest'] in bases
        checks['readonly_container_weights']=build['model_mount_readonly']
    for n,h in r['source_sha256'].items():checks['source_snapshot/'+n]=sha(d/'source_snapshot'/n)==h
    tokenization=read(d/'tokenization.json');original=read(d/'source_snapshot'/WORKLOAD)
    frozen={q['request_id']:q for q in original['requests']}
    checks['tokenizer_48_unchanged_IDs']=tokenization['passed'] and len(tokenization['requests'])==48 and all(
        q['input_ids']==frozen[q['request_id']]['input_ids'] and q['prompt_sha256']==sha_text(frozen[q['request_id']]['payload']['prompt']) for q in tokenization['requests'])
    for row in rows:
        rid=row['request_id'];es=row['events']
        checks[rid+'/one_terminal']=len([e for e in es if e['event'] in TERMINAL])==1
        frames=[e['native'] for e in es if e['event']=='framework_frame']
        content=''.join((f.get('choices') or [{}])[0].get('text') or '' for f in frames)
        checks[rid+'/native_text_binding']=content==row['content']
        if row['state']=='finished':
            checks[rid+'/native_finish']=any((f.get('choices') or [{}])[0].get('finish_reason') for f in frames)
    checks['independent_summary']=read(d/'client_summary.json')==summarize(rows)
    snapshots=read(d/'snapshots.json')
    for i,snap in enumerate(snapshots):
        actual=parse((d/snap['metrics_file']).read_text())
        checks[f'raw_metrics/{i}']=actual==snap['native']
    if r['scenario']=='healthy':
        byid={x['request_id']:x for x in rows};binding=read(d/'workload_binding.json')
        w=read(d/'source_snapshot'/WORKLOAD);schedule=read(d/'source_snapshot'/SCHEDULE);source={q['request_id']:q for q in w['requests']}
        checks['workload_binding']=binding['workload_sha256']==sha(d/'source_snapshot'/WORKLOAD) and binding['schedule_sha256']==sha(d/'source_snapshot'/SCHEDULE)
        checks['trace_has_all_48']=len(binding['requests'])==48 and {x['request_id'] for x in binding['requests']}==set(source)
        checks['arrival_table']=all(a['arrival_s']==b['arrival_s'] and a['request_id']==b['request_id'] for a,b in zip(binding['requests'],schedule['traces']['near']))
        checks['not_SGLang_saturation_claim']=binding['rate_is_not_SGLang_calibration'] is True
        for mapped in binding['requests']:
            rid=mapped['request_id'];original=source[rid]
            checks[rid+'/unchanged_prompt']=sha_text(original['payload']['prompt'])==mapped['prompt_sha256']
            checks[rid+'/payload']=mapped['original_payload_sha256']==original['payload_sha256'] and mapped['mapped_payload_sha256']==byid[rid]['payload_sha256']
            checks[rid+'/trace_completed']=byid[rid]['state']=='finished' and byid[rid]['finish_reason']=='length'
        checks['disconnect']=byid['disconnect']['state']=='cancelled'
        checks['deadline']=byid['deadline']['state']=='cancelled' and byid['deadline']['reason']=='deadline'
        burst=[v for k,v in byid.items() if k.startswith('burst_')]
        checks['overload_accounted']=len(burst)==14 and any(x['http_status'] in (429,503) for x in burst)
        observed=read(d/'queue_observation.json')
        checks['actual_queue']=any(x['queued'] is not None and x['queued']>0 for x in observed)
        checks['running_bound']=all(x['running'] is None or x['running']<=4 for x in observed)
        queued=read(d/'queued_abort.json')
        checks['queued_client_closed']=byid['queued_disconnect']['state']=='cancelled' and not byid['queued_disconnect']['content']
        checks['native_queued_abort']=queued['status']==200 and queued['waiting']['queued']>0 and queued['after']['queued']==0 and queued['after']['running']==4
        for label in ('warmup','validation','cancel','deadline','queued_cancel','burst','trace','final'):
            final=[x['native']['metrics'] for x in snapshots if x['label']==label+'_flushed'][-1]
            checks[label+'/native_released']=final['running']==0 and final['queued']==0 and final['KV_used']==0 and final['KV_usage']==0 and final['KV_available']==4096
            flush=read(d/f'flush_{label}.json');checks[label+'/native_flush']=flush['status']==200 and 'Cache flushed' in flush['body']
            observed=[e for e in native_flushes if flush['start_ns']<=e['monotonic_ns']<=flush['end_ns']]
            checks[label+'/actual_pool_state']=len(observed)==1 and observed[0]['used']==0 and observed[0]['available']==4096 and observed[0]['evictable']==0 and observed[0]['capacity']==4096
        http=read(d/'http_checks.json')
        checks['validation_errors']=http['nonstream_status']==200 and all(400<=http[k]<500 for k in ('invalid_budget_status','over_context_status'))
        checks['probe_after_cancellations']=byid['post_cancel_probe']['state']=='finished'
        final=snapshots[-1]['native'];checks['actual_metrics_present']=not final['missing']
        checks['native_aborts_observed']=final['metrics']['aborted']>=2
        from serving.framework_metrics import client_counts
        view=read(d/'dashboard_snapshot.json')
        checks['dashboard_observed_client_counts']=view['client']==client_counts(d/'requests.json')
        checks['dashboard_actual_GPU_memory']=view['GPU_used_MiB']>0
    else:
        fault=read(d/'fault.json');checks['owned_fault']=fault['scenario']==r['scenario'] and fault['signal']=='SIGTERM' and fault['observed_native']['running']>0
        checks['active_request_accounted']=any(x['request_id']=='inflight' for x in rows)
    evidence={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(d).parts and p.name!='audit.json'}
    return dict(passed=all(checks.values()),checks=checks,scenario=r['scenario'],requests=len(rows),source_commit=r['root_commit'],evidence_sha256=evidence,
        HF_FP32_equivalence_claimed=False,TP2_4_accepted=False,paid_resources_used=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    target=a.directory/'audit.json'
    if target.exists():raise ValueError('Preserve framework audit')
    result=audit(a.directory);write(target,result)
    print('Full-framework audit',result['passed'],len(result['checks']),'failed',[k for k,v in result['checks'].items() if not v])
    if not result['passed']:raise SystemExit(1)
