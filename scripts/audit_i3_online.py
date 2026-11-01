"""Final independent audit of qualification, pilot design and all offered HTTP work."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write
from serving.i3_audit import source_checks,service_checks
from serving.i3_metrics import analyze
from serving.i2_workloads import digest


def audit(directory):
    from scripts.audit_i3_qualification import audit as qualify
    from scripts.audit_i3_pilot import audit as pilot_audit
    from scripts.run_i3_formal import gates
    meta=read(directory/'provenance.json');config=meta['config'];binding=meta['artifact_bindings']
    q=ROOT/binding['qualification_directory'];pilot=ROOT/binding['pilot_directory']
    qa=qualify(q);pa=pilot_audit(pilot)
    workload=read(directory/'workload.json');schedule=read(directory/'schedule.json')
    checks=dict(qualification_recomputed=qa==read(q/'audit.json') and qa['passed'],
        pilot_recomputed=pa==read(pilot/'audit.json') and pa['passed'],
        gate_binding=gates(q,pilot,config)==binding==read(directory/'gate.json'),
        frozen_sources=source_checks(directory,meta),qualified_payloads=workload==read(q/'workload.json'),
        frozen_arrivals=schedule==read(pilot/'schedule.json'),
        qualified_I2=meta['I2_acceptance_sha256']==digest(ROOT/config['I2_acceptance']) and read(ROOT/config['I2_acceptance'])['local_single_gpu_complete'])
    # Extra independent lifecycle checks include normal blocker token budgets.
    for variant in config['variants']:
        path=q/f'qualification_{variant}';records={r['request_id']:r for r in read(path/'requests.json')}
        ev=read(path/'events.json');assigned={e['external_id']:e['uid'] for e in ev if e['event']=='frontend_new_user'}
        for rid in [f'life_block_{i}' for i in range(4)]+['life_probe']:
            tokens=[e for e in ev if e['event']=='scheduler_token' and e['uid']==assigned[rid]]
            checks[variant+'_'+rid+'_healthy']=records[rid]['state']=='finished' and len(tokens)==(16 if rid=='life_probe' else config['lifecycle_block_tokens'])
    labels=[];runs=[];details=[];groups=defaultdict(list)
    rates=['low','near','overload'];variants=list(config['variants'])
    for run in range(1,4):
        order=rates if run==1 else list(reversed(rates)) if run==2 else rates[1:]+rates[:1]
        for index,rate in enumerate(order):
            policies=variants if run==1 or (run==3 and index%2==0) else list(reversed(variants))
            for variant in policies:
                label=f'formal_{variant}_{rate}_run{run}';labels.append(label);path=directory/label
                actual=read(path/'run.json');ev=read(path/'events.json')
                checks.update({label+'_'+k:v for k,v in service_checks(path,config,workload,variant,'formal').items()})
                checks[label+'_identity']=(actual['variant']==variant and actual['rate']==rate and actual['run']==run
                    and actual['config_sha256']==meta['config_sha256'] and actual['workload_sha256']==meta['workload_sha256'])
                checks[label+'_no_numeric_copy']=not any(e['event']=='i3_numeric_saved' for e in ev) and not actual.get('tensor_sha256')
                records=[r for r in read(path/'requests.json') if r['request_id'].startswith('load_')]
                computed=analyze(workload,records,ev,schedule['rates_rps'][rate],config)
                checks[label+'_raw_metrics']=computed==read(path/'metrics.json')
                checks[label+'_open_arrivals']=computed['metrics']['arrival_schedule_valid']
                offsets={r['request_id']:r['arrival_s'] for r in schedule['traces'][rate]}
                origins=[r['planned_ns']/1e9-offsets[r['request_id']] for r in computed['requests']]
                checks[label+'_identical_trace']=max(origins)-min(origins)<1e-6
                checks[label+'_work_budget']=all(r['completed_output_tokens']==next(w['output_budget'] for w in workload['requests'] if w['request_id']==r['request_id'])
                    for r in computed['requests'] if r['state']=='finished')
                m=computed['metrics']
                start=min(r['planned_ns'] for r in computed['requests'])
                end=max(r['terminal_ns'] for r in computed['requests'])
                evicted=sum(e['prefix_evicted_pages'] for e in ev if e['event']=='i3_allocate' and start<=e['monotonic_ns']<=end)
                runs.append(dict(label=label,variant=variant,rate=rate,run=run,prefix_evicted_pages=evicted,**m))
                groups[(variant,rate)].append(m)
                details += [dict(variant=variant,rate=rate,run=run,**r) for r in computed['requests']]
                print('I3 independent formal audit',label,'terminals',m['terminal_counts'],'jitter_valid',m['arrival_schedule_valid'],flush=True)
    journal=read(directory/'processes.json')
    checks['18_clean_ordered_processes']=len(journal)==18 and [r['label'] for r in journal]==labels and all(
        r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal)
    checks['864_offered_records']=len(details)==864
    stats=[]
    scalar=['completed_request_throughput','goodput_rps','offered_SLO_success_fraction',
        'completed_output_token_throughput','admitted_input_token_throughput','prompt_token_cache_hit_fraction']
    latency=['client_text_ttft_ms','e2e_ms','amortized_tpot_ms','scheduler_queue_ms','backend_emission_itl_ms','sse_text_chunk_intervals_ms']
    for (variant,rate),values in sorted(groups.items()):
        row=dict(variant=variant,rate=rate,processes=len(values))
        for key in scalar:
            points=[v[key] for v in values if v[key] is not None]
            row[key]=dict(n=len(points),median=statistics.median(points) if points else None,
                min=min(points) if points else None,max=max(points) if points else None)
        for key in latency:
            row[key]={}
            for quantile in ('p50','p95','p99'):
                points=[v[key][quantile] for v in values if v[key][quantile] is not None]
                row[key][quantile]=dict(n=len(points),median=statistics.median(points) if points else None,
                    min=min(points) if points else None,max=max(points) if points else None)
        stats.append(row)
    checks['six_three_process_cells']=len(stats)==6 and all(r['processes']==3 for r in stats)
    comparisons=[]
    for rate in rates:
        base=[r for r in runs if r['variant']=='radix' and r['rate']==rate]
        target=[r for r in runs if r['variant']=='graph124' and r['rate']==rate]
        base.sort(key=lambda r:r['run']);target.sort(key=lambda r:r['run'])
        ratios=[b['goodput_rps']/a['goodput_rps'] if a['goodput_rps'] else None for a,b in zip(base,target,strict=True)]
        differences=[b['goodput_rps']-a['goodput_rps'] for a,b in zip(base,target,strict=True)]
        comparisons.append(dict(rate=rate,paired_goodput_ratios=ratios,
            ratio_median=statistics.median(ratios) if all(r is not None for r in ratios) else None,
            ratio_min=min(ratios) if all(r is not None for r in ratios) else None,
            ratio_max=max(ratios) if all(r is not None for r in ratios) else None,
            paired_goodput_differences=differences,difference_median=statistics.median(differences)))
    # These frozen labels are operationally checked, not inferred solely from pilot rates.
    for variant in variants:
        low=groups[variant,'low'];high=groups[variant,'overload']
        checks[variant+'_low_no_rejection']=all(m['terminal_counts']['rejected']==0 for m in low)
        checks[variant+'_overload_observed']=all(m['terminal_counts']['rejected']>0 or m['scheduler_queue_ms']['p95']>500 for m in high)
    checks['baseline_near_queue_observed']=any(m['scheduler_queue_ms']['p95']>50 for m in groups['radix','near'])
    passed=all(checks.values())
    return dict(stage='I3',scope=config['scope'],passed=passed,local_single_gpu_complete=passed,
        checks=checks,runs=runs,requests=details,statistics=stats,comparisons=comparisons,
        formal_processes=18,offered_requests=864,qualification_vectors=qa['vectors'],
        latency_objectives_s=config['latency_objectives_s'],rates_rps=schedule['rates_rps'],
        finite_trace=True,steady_state_claimed=False,TP2_4_complete=False,paid_resources_used=False,
        provenance_sha256=digest(directory/'provenance.json'),gate_sha256=digest(directory/'gate.json'),
        qualification_audit_sha256=digest(q/'audit.json'),pilot_audit_sha256=digest(pilot/'audit.json'),
        auditor_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    target=args.directory/'audit.json'
    if target.exists():raise ValueError('Preserve independent final audit')
    result=audit(args.directory);write(target,result)
    dest=args.directory/'audit_source_snapshot/scripts';dest.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),dest/Path(__file__).name)
    print('Final I3 passed:',result['passed'],'failed:',[k for k,v in result['checks'].items() if not v],flush=True)
    raise SystemExit(0 if result['passed'] else 1)
