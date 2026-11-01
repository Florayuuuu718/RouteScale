"""Independent raw-service checks shared by qualification and online audits."""
from collections import defaultdict
from pathlib import Path

from serving.i3_workloads import ROOT, read
from serving.i2_workloads import digest, token_hash
from scripts.run_i1_lifecycle import events as raw_events


def service_checks(path, config, workload, variant, stage):
    run=read(path/'run.json');events=read(path/'events.json');records=read(path/'requests.json')
    snapshots=read(path/'snapshots.json');raw=ROOT/run['runtime_dir']
    checks=dict(executed=run['status']=='exercised', raw_hashes=all(digest(raw/n)==sha for n,sha in run['raw_sha256'].items()),
        raw_events=events==raw_events(raw), normal_exit=run['returncode']==0 and not run['forced_cleanup']
        and not run['surviving_pids'] and run['shutdown_s']<=config['shutdown_s'])
    stopped=[e for e in events if e['event']=='workers_stopped']
    checks['workers_exit']=len(stopped)==1 and not stopped[0]['escalated'] and not stopped[0]['surviving_pids'] and all(v==0 for v in stopped[0]['exitcodes'].values())
    from scripts.audit_i2_precision import policy_ok
    policies=[e for e in events if e['event']=='i2_precision_policy']
    checks['actual_precision_capture']=len(policies)==1 and policy_ok(policies[0],'mixed',config['variants'][variant])
    shutdown=[e for e in events if e['event']=='i2_shutdown_policy']
    checks['bounded_shutdown_policy']=len(shutdown)==1 and shutdown[0]['total_budget_s']==8 and shutdown[0]['external_shutdown_budget_s']==10
    keys=('ack_uids','event_uids','deadline_uids','admitted_uids','aborting_uids','backend_done_uids')
    checks['frontend_clean']=bool(snapshots) and all(not s['frontend'][k] for s in snapshots for k in keys)
    checks['resources_clean']=all(s['idle']['available_tables']==4 and not s['idle']['pending_uids']
        and not s['idle']['running_uids'] and s['idle']['prefix_total_tokens']==s['idle']['prefix_evictable_tokens']
        and s['idle']['free_pages']+s['idle']['prefix_total_tokens']==2048 and not s['detokenizer']['active_uids'] for s in snapshots)
    known={r['request_id']:r for r in workload['requests']+workload['warmup']}
    assigned={e['external_id']:e['uid'] for e in events if e['event']=='frontend_new_user'}
    local=defaultdict(list)
    for e in events:
        if 'uid' in e:local[e['uid']].append(e)
    ids=[r['request_id'] for r in records]
    checks['unique_requests']=len(ids)==len(set(ids))
    checks['one_client_terminal']=all(sum(e['event'] in ('finished','failed','cancelled','rejected') for e in r['events'])==1 for r in records)
    normals=[]
    for record in records:
        rid=record['request_id'];uid=assigned.get(rid)
        if rid not in known:continue
        req=known[rid]
        checks[rid+'_payload']=record['payload_sha256']==token_hash(req['payload'])
        if record['state']=='finished':
            tokens=[e for e in local[uid] if e['event']=='scheduler_token']
            inputs=[e for e in local[uid] if e['event']=='tokenized']
            checks[rid+'_tokens']=len(tokens)==req['output_budget'] and sum(e['finished'] for e in tokens)==1 and tokens[-1]['finish_reason']=='length' and record['finish_reason']=='length'
            checks[rid+'_input']=len(inputs)==1 and inputs[0]['input_ids']==req['input_ids']
            checks[rid+'_free']=sum(e['event']=='request_free' for e in local[uid])==1
            normals.append(rid)
        elif stage!='formal':
            checks[rid+'_completed']=False
        elif record['state']=='rejected':
            checks[rid+'_rejection']=record['http_status']==429 and uid is None and any(e['event']=='frontend_rejected' and e.get('external_id')==rid for e in events)
        elif record['state']=='cancelled':
            checks[rid+'_cancel']=uid is not None and any(e['event']=='scheduler_end' for e in local[uid]) and any(e['event']=='backend_terminal' for e in local[uid])
        else:
            checks[rid+'_unexpected_failure']=False
    checks['all_offered_present']=set(known)<=set(ids)
    if stage=='qualification':
        records_by_id={r['request_id']:r for r in records}
        for rid in ('life_queued_cancel','life_decode_cancel','life_deadline'):
            uid=assigned.get(rid)
            checks[rid]=(records_by_id[rid]['state']=='cancelled' and uid is not None
                and any(e['event']=='scheduler_end' for e in local[uid])
                and any(e['event']=='backend_terminal' for e in local[uid]))
        uid=assigned['life_queued_cancel']
        checks['queued_no_compute']=not any(e['event']=='batch_prepare' and uid in e['uids'] for e in events)
        uid=assigned['life_decode_cancel']
        checks['decode_did_compute']=any(e['event']=='scheduler_token' for e in local[uid])
        checks['post_cancel_probe']=records_by_id['life_probe']['state']=='finished'
        sizes={len(e['requests']) for e in events if e['event']=='i3_batch'}
        checks['active_batch_1_2_3_4']={1,2,3,4}<=sizes
    replays=[e for e in events if e['event']=='i3_graph_replay']
    checks['actual_graph_replay']=bool(replays) if config['variants'][variant]['graph_bs'] else not replays
    return checks


def source_checks(directory, meta):
    return (digest(directory/'workload.json')==meta['workload_sha256']
        and token_hash(meta['config'])==meta['config_sha256']
        and all(digest(ROOT/n)==sha==digest(directory/'source_snapshot'/n) for n,sha in meta['source_sha256'].items())
        and all(digest(ROOT/meta['config']['model_path']/n)==sha for n,sha in meta['model_sha256'].items()))
