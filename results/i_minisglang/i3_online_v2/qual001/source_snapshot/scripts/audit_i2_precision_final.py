"""Final timing audit with explicit retry-artifact bindings.

The frozen numerical/HTTP auditors are reused unchanged. Timing computations
retain the original independent audit algorithm; artifact lookup is explicit.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest,token_hash
from scripts.audit_i2_precision import (read,audit_qualification,audit_http,check_sources,policy_ok,work_ok,journal_ok)


def audit_performance(directory):
    from scripts.run_i2_precision_followup import prerequisites
    from serving.i2_precision_policy import POLICIES
    meta = read(directory/'provenance.json')
    config = meta['config']
    q, l = ROOT/meta['artifact_bindings']['qualification_directory'], ROOT/meta['artifact_bindings']['lifecycle_directory']
    qualification = audit_qualification(q)
    http = audit_http(l)
    from scripts.audit_i2_precision_shutdown import audit as audit_shutdown
    shutdown = audit_shutdown(l)
    checks = dict(independent_shutdown_recompute=shutdown == read(l/'shutdown_audit.json') and shutdown['passed'],
        independent_numerical_recompute=qualification == read(q/'audit.json') and qualification['qualification_passed'],
        independent_HTTP_recompute=http == read(l/'audit.json') and http['HTTP_lifecycle_passed'],
        gate_binding=prerequisites(config,meta['artifact_bindings']) == read(directory/'gate.json'),
        frozen_sources=check_sources(directory,meta), config_hash=token_hash(config) == meta['config_sha256'],
        workload_hash=digest(directory/'workload.json') == meta['workload_sha256'])
    workload = read(directory/'workload.json')
    control = read(q/'workload.json')['suites'][0]['workload']
    checks['qualified_performance_input'] = workload == control
    cases = {c['case_id']:c for c in workload['cases']}
    grouped = defaultdict(list)
    for mode in config['modes']:
        for variant, settings in config['variants'].items():
            for run in range(1,4):
                key = f'{mode}_{variant}_run{run}'
                report = read(directory/f'benchmark_{key}.json')
                data = report['replay']
                checks[key+'_identity'] = (report['status'] == 'pass' and report['mode'] == mode and report['variant'] == variant
                    and report['run_id'] == run and report['policy'] == POLICIES[mode]
                    and report['config_sha256'] == meta['config_sha256'] and report['workload_sha256'] == meta['workload_sha256']
                    and report['patched_tree'] == meta['series']['tree']
                    and report['precision'] == dict(tf32=False,bf16_reduced_reduction=False,fp16_reduced_reduction=False))
                checks[key+'_precision_policy'] = policy_ok(report['observed'],mode,settings)
                checks[key+'_actual_engine'] = (data['engine']['dtype'] == ('torch.float16' if mode == 'mixed' else 'torch.bfloat16')
                    and data['engine']['actual_backend'] == 'FlashInferBackend' and data['engine']['num_pages'] == 2048
                    and data['capture']['buckets'] == settings['graph_bs'])
                trace_path = ROOT/report['runtime_dir']/'batches.jsonl'
                traces_list = [json.loads(s) for s in trace_path.read_text().splitlines()]
                traces = {(r['case_id'],r['phase'],r['repeat']):r['batches'] for r in traces_list}
                expected = {(c,p,n) for c in cases for p in ('cold','warm') for n in range(8)}
                checks[key+'_trace_hash'] = digest(trace_path) == data['trace_sha256']
                checks[key+'_coverage'] = (set(traces) == {(r['case_id'],r['phase'],r['repeat']) for r in data['cases']} == expected
                    and len(traces_list) == len(data['cases']) == 64)
                per_process = defaultdict(list)
                work, flags = True, True
                for row in data['cases']:
                    trace = traces[(row['case_id'],row['phase'],row['repeat'])]
                    with_batches = dict(row,batches=trace)
                    work &= work_ok(with_batches,cases[row['case_id']],config,settings)
                    measured = row['repeat'] >= 3
                    flags &= row['measured'] == measured
                    padded_rows = sum(((sum(r['extend'] for r in b['requests'])+511)//512)*512 for b in trace if b['phase'] == 'prefill')
                    if measured:
                        batch = cases[row['case_id']]['batch_size']
                        per_process[(row['case_id'],row['phase'])].append(dict(throughput=batch*64/row['seconds'],
                            first_result_ms=statistics.mean(r['first_result_ms'] for r in row['requests']),
                            prefill_queries=row['computed_prefill_tokens'], padded_rows=padded_rows))
                checks[key+'_work_resources'] = bool(work)
                checks[key+'_measured_flags'] = bool(flags)
                checks[key+'_five_measured_repeats'] = all(len(values) == 5 for values in per_process.values()) and len(per_process) == 8
                for (case,phase),values in per_process.items():
                    ts = [r['throughput'] for r in values]
                    grouped[(mode,variant,case,phase)].append(dict(run=run,throughput=statistics.mean(ts),
                        first_result_ms=statistics.mean(r['first_result_ms'] for r in values),
                        prefill_queries=statistics.mean(r['prefill_queries'] for r in values),
                        prefill_linear_padded_rows_per_projection=statistics.mean(r['padded_rows'] for r in values),
                        throughput_repeat_cv=statistics.stdev(ts)/statistics.mean(ts)))
                print('performance audit',key,'work/resources',bool(work),flush=True)
    names = list(config['variants']); labels=[]
    for run in range(1,4):
        order = names if run == 1 else list(reversed(names)) if run == 2 else names[3:]+names[:3]
        for index,variant in enumerate(order):
            modes = config['modes'] if run == 1 or (run == 3 and index%2 == 0) else list(reversed(config['modes']))
            labels += [f'benchmark_{m}_{variant}_run{run}' for m in modes]
    checks['42_clean_processes'] = journal_ok(directory,labels)
    checks['balanced_execution_order'] = [r['label'] for r in read(directory/'processes.json')] == labels
    stats=[]
    for (mode,variant,case,phase),values in sorted(grouped.items()):
        ts=[r['throughput'] for r in values]
        stats.append(dict(mode=mode,variant=variant,case=case,phase=phase,processes=values,
            throughput_median=statistics.median(ts),throughput_min=min(ts),throughput_max=max(ts),
            first_result_ms_median=statistics.median(r['first_result_ms'] for r in values),
            process_cv=statistics.stdev(ts)/statistics.mean(ts)))
    checks['112_complete_cells'] = len(stats) == 112 and all(len(r['processes']) == 3 for r in stats)
    index={(r['mode'],r['variant'],r['case'],r['phase']):r for r in stats}
    costs=[]; comparisons=[]
    for case in cases:
        for phase in ('cold','warm'):
            for variant in config['variants']:
                a=index[('bf16_control',variant,case,phase)]; b=index[('mixed',variant,case,phase)]
                ratios=[y['throughput']/x['throughput'] for x,y in zip(a['processes'],b['processes'],strict=True)]
                costs.append(dict(variant=variant,case=case,phase=phase,mixed_over_BF16_throughput_median=statistics.median(ratios),
                    ratio_min=min(ratios),ratio_max=max(ratios),baseline_external_HF_passed=False))
            for mode in config['modes']:
                for variant in config['variants']:
                    if variant == 'radix_full':
                        continue
                    baseline='naive_full' if variant == 'naive_full' else 'radix_full'
                    candidate='radix_full' if variant == 'naive_full' else variant
                    a=index[(mode,baseline,case,phase)]; b=index[(mode,candidate,case,phase)]
                    ratios=[y['throughput']/x['throughput'] for x,y in zip(a['processes'],b['processes'],strict=True)]
                    comparisons.append(dict(mode=mode,case=case,phase=phase,baseline=baseline,candidate=candidate,
                        paired_ratio_median=statistics.median(ratios),paired_ratio_min=min(ratios),paired_ratio_max=max(ratios)))
    passed=all(checks.values())
    return dict(stage='I2',scope='local_FI_TP1_mixed_precision_and_same_shape_BF16_cost_control',
        passed=passed,local_single_gpu_complete=passed,checks=checks,statistics=stats,precision_costs=costs,comparisons=comparisons,
        mixed_external_HF_vectors=12544,baseline_internal_vectors=12544,HTTP_requests=168,HTTP_scenario_checks=560,
        formal_processes=42,original_BF16_external_HF_passed=False,multi_gpu_complete=False,paid_resources_used=False,
        measurement_scope=config['performance_boundary'],auditor_sha256=digest(Path(__file__)),
        gate_sha256=digest(directory/'gate.json'),provenance_sha256=digest(directory/'provenance.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();output=args.directory/'audit.json'
    if output.exists():raise ValueError('Preserve final audit')
    result=audit_performance(args.directory)
    output.write_text(json.dumps(result,indent=2)+'\n')
    snapshot=args.directory/'audit_source_snapshot/scripts';snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print(json.dumps({k:result[k] for k in ('passed','local_single_gpu_complete','formal_processes')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
