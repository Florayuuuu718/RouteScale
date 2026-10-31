"""Recompute I2 coverage, real batch work, lifecycle gates and process statistics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest,token_hash
from scripts.audit_i2_numerics import audit_numerics
from scripts.run_i2_service_gate import audit_service


def audit(directory):
    read=lambda name:json.loads((directory/name).read_text())
    meta=read('provenance.json');config=meta['config'];checks={}
    numeric=audit_numerics(directory)
    checks['independent_numerical_recompute']=numeric==read('numerical_gate.json') and numeric['passed']
    checks['live_source_matches_frozen']=all(digest(ROOT/name)==value for name,value in meta['source_sha256'].items())
    for name in config['variants']:
        stored=json.loads((directory/f'service_{name}'/'audit.json').read_text())
        actual=audit_service(directory/f'service_{name}',config)
        checks[f'lifecycle_{name}']=actual==stored and actual['passed']
    journal=read('processes.json')
    checks['all_processes_successful']=all(r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal)
    checks['expected_process_count']=len(journal)==1+2*len(config['variants'])+config['independent_runs']*len(config['variants'])
    metrics=defaultdict(list);reports=[]
    for name,settings in config['variants'].items():
        for run in range(1,config['independent_runs']+1):
            report=read(f'benchmark_{name}_run{run}.json');reports.append(report)
            label=f'{name}_r{run}'
            checks[f'provenance_{label}']=report['status']=='pass' and report['config_sha256']==numeric['config_sha256'] and report['workload_sha256']==numeric['workload_sha256'] and report['patched_tree']==numeric['patched_tree'] and report['precision']=={'bf16_reduced_reduction':False,'tf32':False}
            data=report['replay'];raw=ROOT/report['runtime_dir']/'batches.jsonl'
            trace_rows=[json.loads(line) for line in raw.read_text().splitlines()]
            traces={(r['case_id'],r['phase'],r['repeat']):r['batches'] for r in trace_rows}
            checks[f'raw_trace_{label}']=digest(raw)==data['trace_sha256'] and len(traces)==len(trace_rows)==len(data['cases'])
            expected={(c,p,k) for c in ('b1_zero','b1_shared','b3_zero','b3_shared') for p in ('cold','warm') for k in range(config['warmup_repeats']+config['measure_repeats'])}
            checks[f'coverage_{label}']={(r['case_id'],r['phase'],r['repeat']) for r in data['cases']}==expected
            grouped=defaultdict(list)
            for row in data['cases']:
                batch=1 if row['case_id'].startswith('b1') else 3
                trace=traces[(row['case_id'],row['phase'],row['repeat'])]
                correct=token_hash(trace)==row['batch_trace_sha256']
                correct &= row['computed_prefill_tokens']==sum(r['extend'] for b in trace if b['phase']=='prefill' for r in b['requests'])
                correct &= row['computed_prefill_tokens']==batch*config['prompt_tokens']-sum(r['cached_tokens'] for r in row['requests'])
                replays=0
                for b in trace:
                    size=len(b['requests'])
                    eligible=b['phase']=='decode' and bool(settings['graph_bs']) and size<=max(settings['graph_bs'])
                    padded=next(x for x in settings['graph_bs'] if x>=size) if eligible else size
                    correct &= b['graph_eligible']==eligible and b['padded']==padded
                    replays+=eligible
                correct &= row['graph_replays']==replays
                correct &= row['seconds']>0 and len(row['requests'])==batch and all(
                    r['output_tokens']==64 and r['reason']=='length' and r['free_count']==1 and
                    0<r['first_result_ms']<=r['terminal_ms']<=row['seconds']*1000 for r in row['requests'])
                released=row['released']
                correct &= released['tables']==4 and released['page_indices_unique'] and released['table_slots_unique'] and not released['retired'] and not released['pending'] and not released['running'] and not released['prefix_protected']
                checks[f'actual_work_{label}']=checks.get(f'actual_work_{label}',True) and bool(correct)
                if row['measured']:
                    grouped[(row['case_id'],row['phase'])].append(dict(
                        throughput=batch*64/row['seconds'],first_result_ms=statistics.mean(r['first_result_ms'] for r in row['requests']),
                        seconds=row['seconds'],cached=statistics.mean(r['cached_tokens'] for r in row['requests']),
                        prefill_queries=row['computed_prefill_tokens']))
            for (case,phase),values in grouped.items():
                checks[f'measured_repeats_{label}']=checks.get(f'measured_repeats_{label}',True) and len(values)==config['measure_repeats']
                metrics[(name,case,phase)].append(dict(run=run,throughput=statistics.mean(v['throughput'] for v in values),
                    first_result_ms=statistics.mean(v['first_result_ms'] for v in values),
                    throughput_repeat_cv=statistics.stdev(v['throughput'] for v in values)/statistics.mean(v['throughput'] for v in values),
                    cached_tokens=statistics.mean(v['cached'] for v in values),prefill_queries=statistics.mean(v['prefill_queries'] for v in values)))
    stats=[]
    for (name,case,phase),values in sorted(metrics.items()):
        throughputs=[v['throughput'] for v in values]
        stats.append(dict(variant=name,case=case,phase=phase,processes=values,
            throughput_median=statistics.median(throughputs),throughput_min=min(throughputs),throughput_max=max(throughputs),
            first_result_ms_median=statistics.median(v['first_result_ms'] for v in values),
            process_cv=statistics.stdev(throughputs)/statistics.mean(throughputs)))
    comparisons=[]
    indexed={(r['variant'],r['case'],r['phase']):r for r in stats}
    for case in ('b1_zero','b1_shared','b3_zero','b3_shared'):
        for phase in ('cold','warm'):
            for name in config['variants']:
                if name=='radix_full':continue
                baseline='naive_full' if name=='naive_full' else 'radix_full'
                candidate='radix_full' if name=='naive_full' else name
                a=indexed[(baseline,case,phase)];b=indexed[(candidate,case,phase)]
                paired=[y['throughput']/x['throughput'] for x,y in zip(a['processes'],b['processes'],strict=True)]
                comparisons.append(dict(case=case,phase=phase,baseline=baseline,candidate=candidate,
                    paired_ratio_median=statistics.median(paired),paired_ratio_min=min(paired),paired_ratio_max=max(paired),
                    baseline_process_cv=a['process_cv'],candidate_process_cv=b['process_cv']))
    passed=all(checks.values())
    return dict(stage='I2',local_single_gpu_complete=passed,passed=passed,checks=checks,
        numerical_vectors=numeric['compared_vectors'],service_profiles=len(config['variants']),
        formal_processes=len(reports),statistics=stats,comparisons=comparisons,
        measurement_scope=config['performance_boundary'],multi_gpu_complete=False,paid_resources_used=False,
        numerical_gate_sha256=digest(directory/'numerical_gate.json'),service_gate_sha256=digest(directory/'service_gate.json'),
        audit_source_sha256=digest(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();result=audit(args.directory);path=args.directory/'audit.json'
    if args.check_only:
        if json.loads(path.read_text())!=result:raise ValueError('Independent I2 audit differs')
    elif path.exists():raise ValueError('Preserve existing audit')
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','local_single_gpu_complete','numerical_vectors','formal_processes')}))
    raise SystemExit(0 if result['passed'] else 1)
