"""Independent acceptance for the explicit shape-controlled I2 timing scope."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest,token_hash


def audit(directory):
    from scripts.audit_i2_shape_qualification import audit as audit_qualification
    from scripts.run_i2_service_gate import audit_service
    from scripts.run_i2_shape_performance import prerequisites
    read=lambda name:json.loads((directory/name).read_text())
    meta=read('provenance.json');config=meta['config'];checks={}
    q=ROOT/config['qualification_directory'];l=ROOT/config['lifecycle_directory']
    qualification=audit_qualification(q)
    checks['independent_numeric_recompute']=qualification==json.loads((q/'audit.json').read_text()) and qualification['qualification_passed']
    checks['admission_binding']=prerequisites(config,config,read('workload.json'))==meta['gates']==read('gate.json')
    checks['frozen_sources']=all(digest(ROOT/n)==sha==digest(directory/'source_snapshot'/n) for n,sha in meta['source_sha256'].items())
    checks['frozen_config']=token_hash(config)==meta['config_sha256'] and read('config.json')==config
    checks['frozen_workload']=digest(directory/'workload.json')==meta['workload_sha256']
    lm=json.loads((l/'provenance.json').read_text())
    for variant in config['variants']:
        actual=audit_service(l/f'service_{variant}',lm['config'])
        checks[f'HTTP_{variant}']=actual['passed'] and actual==json.loads((l/f'service_{variant}/audit.json').read_text())
    journal=read('processes.json')
    checks['21_clean_processes']=len(journal)==21 and len({r['label'] for r in journal})==21 and all(r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal)
    metrics=defaultdict(list);reports=[]
    for name,settings in config['variants'].items():
        for run in range(1,4):
            report=read(f'benchmark_{name}_run{run}.json');reports.append(report);label=f'{name}_r{run}'
            checks[f'provenance_{label}']=report['status']=='pass' and report['config_sha256']==meta['config_sha256'] and report['workload_sha256']==meta['workload_sha256'] and report['patched_tree']==meta['series']['tree'] and report['policy']==config['policy'] and report['reference_scope']==config['reference_scope'] and report['precision']=={'bf16_reduced_reduction':False,'tf32':False}
            data=report['replay'];raw=ROOT/report['runtime_dir']/'batches.jsonl'
            trace_rows=[json.loads(s) for s in raw.read_text().splitlines()]
            traces={(r['case_id'],r['phase'],r['repeat']):r['batches'] for r in trace_rows}
            checks[f'traces_{label}']=digest(raw)==data['trace_sha256'] and len(traces)==len(trace_rows)==len(data['cases'])==64
            checks[f'capture_{label}']=data['capture']['buckets']==settings['graph_bs'] and data['engine']['actual_backend']=='FlashInferBackend' and data['engine']['num_pages']==2048
            expected={(case,phase,k) for case in ('b1_zero','b1_shared','b3_zero','b3_shared') for phase in ('cold','warm') for k in range(8)}
            checks[f'coverage_{label}']={(r['case_id'],r['phase'],r['repeat']) for r in data['cases']}==expected
            grouped=defaultdict(list)
            for row in data['cases']:
                batch=1 if row['case_id'].startswith('b1') else 3
                trace=traces[(row['case_id'],row['phase'],row['repeat'])]
                correct=token_hash(trace)==row['batch_trace_sha256']
                queries=sum(r['extend'] for b in trace if b['phase']=='prefill' for r in b['requests'])
                padded_rows=sum(((sum(r['extend'] for r in b['requests'])+511)//512)*512 for b in trace if b['phase']=='prefill')
                correct &= queries==row['computed_prefill_tokens']==batch*512-sum(r['cached_tokens'] for r in row['requests'])
                replays=0
                for b in trace:
                    size=len(b['requests']);eligible=b['phase']=='decode' and bool(settings['graph_bs']) and size<=max(settings['graph_bs'])
                    padded=next(x for x in settings['graph_bs'] if x>=size) if eligible else size
                    correct &= b['graph_eligible']==eligible and b['padded']==padded;replays+=eligible
                correct &= row['graph_replays']==replays
                correct &= len(row['requests'])==batch and row['seconds']>0 and all(r['output_tokens']==64 and r['free_count']==1 and r['reason']=='length' and 0<r['first_result_ms']<=r['terminal_ms']<=row['seconds']*1000 for r in row['requests'])
                if row['phase']=='warm':
                    hit=384 if row['case_id'].endswith('shared') and settings['cache']=='radix' else 0
                    correct &= all(r['cached_tokens']==hit for r in row['requests'])
                state=row['released']
                correct &= not state['pending'] and not state['running'] and not state['retired'] and not state['prefix_protected'] and state['tables']==4 and state['free_pages']+state['prefix_total']==2048 and state['table_slots_unique'] and state['page_indices_unique'] and state['page_indices_sha256']==token_hash(list(range(2048)))
                checks[f'work_and_resources_{label}']=checks.get(f'work_and_resources_{label}',True) and bool(correct)
                correct_measured=row['repeat']>=3
                checks[f'measured_flags_{label}']=checks.get(f'measured_flags_{label}',True) and row['measured']==correct_measured
                if row['measured']:
                    grouped[(row['case_id'],row['phase'])].append(dict(throughput=batch*64/row['seconds'],
                        first_result_ms=statistics.mean(r['first_result_ms'] for r in row['requests']),queries=queries,padded_rows=padded_rows))
            for (case,phase),values in grouped.items():
                checks[f'five_repeats_{label}']=checks.get(f'five_repeats_{label}',True) and len(values)==5
                metrics[(name,case,phase)].append(dict(run=run,throughput=statistics.mean(r['throughput'] for r in values),
                    first_result_ms=statistics.mean(r['first_result_ms'] for r in values),
                    prefill_queries=statistics.mean(r['queries'] for r in values),
                    prefill_linear_padded_rows_per_projection=statistics.mean(r['padded_rows'] for r in values),
                    throughput_repeat_cv=statistics.stdev(r['throughput'] for r in values)/statistics.mean(r['throughput'] for r in values)))
    stats=[]
    for (name,case,phase),values in sorted(metrics.items()):
        throughputs=[r['throughput'] for r in values]
        stats.append(dict(variant=name,case=case,phase=phase,processes=values,
            throughput_median=statistics.median(throughputs),throughput_min=min(throughputs),throughput_max=max(throughputs),
            first_result_ms_median=statistics.median(r['first_result_ms'] for r in values),
            process_cv=statistics.stdev(throughputs)/statistics.mean(throughputs)))
    index={(r['variant'],r['case'],r['phase']):r for r in stats};comparisons=[]
    for case in ('b1_zero','b1_shared','b3_zero','b3_shared'):
        for phase in ('cold','warm'):
            for name in config['variants']:
                if name=='radix_full':continue
                baseline='naive_full' if name=='naive_full' else 'radix_full'
                candidate='radix_full' if name=='naive_full' else name
                a=index[(baseline,case,phase)];b=index[(candidate,case,phase)]
                paired=[y['throughput']/x['throughput'] for x,y in zip(a['processes'],b['processes'],strict=True)]
                comparisons.append(dict(case=case,phase=phase,baseline=baseline,candidate=candidate,
                    paired_ratio_median=statistics.median(paired),paired_ratio_min=min(paired),paired_ratio_max=max(paired),
                    baseline_process_cv=a['process_cv'],candidate_process_cv=b['process_cv']))
    passed=all(checks.values())
    return dict(stage='I2',passed=passed,local_single_gpu_complete=passed,scope='shape_controlled_native_BF16_systems_ablation',
        reference_scope=config['reference_scope'],original_HF_FP32_equivalence_passed=False,checks=checks,
        numerical_vectors=qualification['compared_vectors'],service_profiles=7,formal_processes=len(reports),
        statistics=stats,comparisons=comparisons,measurement_scope=config['performance_boundary'],
        multi_gpu_complete=False,paid_resources_used=False,source_sha256=digest(Path(__file__)),
        gate_sha256=digest(directory/'gate.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();result=audit(args.directory);path=args.directory/'audit.json'
    if path.exists():raise ValueError('Preserve timing audit')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','local_single_gpu_complete','numerical_vectors','formal_processes')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
