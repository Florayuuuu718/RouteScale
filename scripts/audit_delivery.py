"""Reopen clean-checkout service evidence, request lifetimes and worker cleanup."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import identity,read,sha,write


def audit(directory):
    report=read(directory/'run.json');state=read(directory/'status.json');events=read(directory/'events.json')
    scenario=report['scenario'];checks={};source=Path(report['source_root'])
    checks['scenario_passed']=report['passed']
    checks['clean_checkout']=report['checkout_clean'] and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    checks['source_commit']=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==report['root_commit']
    checks['state']=state['state']==('failed' if scenario=='worker_fault' else 'stopped')
    checks['no_forced_cleanup']=not state['forced_cleanup'] and state['shutdown_s']<=10 and state['gpu_after']=='0 MiB'
    checks['supervisor_dead']=identity(state['supervisor_pid']) is None
    checks['API_dead']=identity(state['server_pid']) is None
    policies=[r for r in events if r['event']=='i3_precision_policy'];i2=[r for r in events if r['event']=='i2_precision_policy']
    checks['I3_actual_policy']=len(policies)==1 and policies[0]['upper_layers']==16 and policies[0]['head_weight_dtype']=='torch.float32' and policies[0]['captured_graph_keys']==[1,2,4] and policies[0]['TF32'] is False
    checks['I2_actual_policy']=len(i2)==1 and i2[0]['graph_install']['norms_installed']==57 and i2[0]['dtype']=='torch.float16' and i2[0]['kv_dtype']=='torch.float16'
    warm_start=[r for r in events if r['event']=='delivery_warmup_start'];warm_end=[r for r in events if r['event']=='delivery_warmup_end']
    first_external=min((r['monotonic_ns'] for r in events if r['event']=='frontend_new_user' and r['external_id']),default=0)
    checks['startup_prefill_before_traffic']=(len(warm_start)==len(warm_end)==1 and warm_end[0]['monotonic_ns']<first_external
        and warm_end[0]['public_request_timeout_s']==20 and warm_end[0]['tokens']==2)
    stopped=[r for r in events if r['event']=='workers_stopped']
    checks['workers_stopped']=bool(stopped) and all(not r['escalated'] and not r['surviving_pids'] for r in stopped)
    if stopped:
        checks['worker_PIDs_dead']=all(identity(w['pid']) is None for r in stopped for w in r['workers'])
        if scenario!='worker_fault':checks['worker_exitcodes']=all(v==0 for r in stopped for v in r['exitcodes'].values())
    requests=read(directory/'requests.json');checks['unique_request_ledger']=len({r['request_id'] for r in requests})==len(requests)
    checks['client_terminal']=all(r['state'] in ('finished','failed','rejected','cancelled') and
        sum(e['event'] in ('finished','failed','rejected','cancelled') for e in r['events'])==1 for r in requests)
    for r in requests:
        rid=r['request_id'];users=[e for e in events if e['event']=='frontend_new_user' and e['external_id']==rid]
        if r['state']=='rejected':
            checks[rid+'/not_admitted']=not users and any(e['event']=='frontend_rejected' and e['external_id']==rid and e['status_code']==429 for e in events)
            continue
        checks[rid+'/admitted_once']=len(users)==1
        if not users:continue
        uid=users[0]['uid']
        if scenario=='healthy' or rid=='warmup':
            checks[rid+'/backend_free_once']=sum(e['event']=='request_free' and e.get('uid')==uid for e in events)==1
            checks[rid+'/admission_released']=sum(e['event']=='admission_release' and e['uid']==uid for e in events)==1
        if r['state']=='cancelled':
            checks[rid+'/backend_abort']=any(e['event']=='frontend_abort_sent' and e['uid']==uid for e in events)
    if scenario=='healthy':
        checks['17_client_requests']=len(requests)==17
        checks['overload_rejected']=any(r['state']=='rejected' for r in requests)
        checks['client_disconnect']=any(r['request_id']=='disconnect' and r['state']=='cancelled' for r in requests)
        checks['client_deadline']=any(r['request_id']=='client_deadline' and r['reason']=='deadline' for r in requests)
        checks['HTTP_validation']=read(directory/'http_checks.json')==dict(nonstream_status=200,invalid_budget_status=422,over_context_status=400)
        snapshots=read(directory/'snapshots.json');checks['seven_clean_settlements']=len(snapshots)==7
        for snapshot in snapshots:
            label=snapshot['label'];f=snapshot['frontend'];idle=snapshot['idle']
            checks[label+'/frontend_clean']=all(not f[k] for k in ('ack_uids','event_uids','deadline_uids','admitted_uids','aborting_uids','backend_done_uids'))
            checks[label+'/KV_table_clean']=not idle['pending_uids'] and not idle['running_uids'] and idle['available_tables']==4 and idle['free_pages']+idle['prefix_total_tokens']==2048
            checks[label+'/detokenizer_clean']=not snapshot['detokenizer']['active_uids']
        text=(directory/'metrics.txt').read_text();checks['monitoring_gauges']='routescale_inference_admitted_current 0' in text and 'status="429"' in text
        checks['actual_graph_replay']=any(e['event']=='i3_graph_replay' for e in events)
    else:
        fault=read(directory/'fault.json');checks['owned_fault']=fault['scenario']==scenario and fault['signal']=='SIGTERM'
        checks['active_request_terminal']=any(r['request_id']=='inflight' for r in requests)
        if scenario=='worker_fault':checks['worker_health_failure']=any(e['event']=='frontend_failed' for e in events) and 'readiness' in state.get('error','').lower()
    return dict(passed=all(checks.values()),checks=checks,scenario=scenario,requests=len(requests),source_commit=report['root_commit'],
        evidence_sha256={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='audit.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    target=a.directory/'audit.json'
    if target.exists():raise ValueError('Preserve service audit')
    r=audit(a.directory);write(target,r);print('Delivery independent audit',r['passed'],len(r['checks']),'failed',[k for k,v in r['checks'].items() if not v])
    if not r['passed']:raise SystemExit(1)
