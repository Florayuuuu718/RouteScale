"""Independent numerical, HTTP and timing acceptance for the final I2 TP1 scope."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.i2_workloads import digest, token_hash, validate_workload


def read(path):
    return json.loads(Path(path).read_text())


def check_sources(directory, meta):
    return all(digest(directory/'source_snapshot'/n) == sha == digest(ROOT/n)
               for n, sha in meta['source_sha256'].items())


def policy_ok(observed, mode, settings):
    from serving.i2_precision_policy import POLICIES
    dtype = 'torch.float16' if mode == 'mixed' else 'torch.bfloat16'
    return (observed['mode'] == mode and observed['policy'] == POLICIES[mode]
        and observed['dtype'] == observed['weight_dtype'] == observed['kv_dtype'] == dtype
        and observed['residual_dtype'] == ('torch.float32' if mode == 'mixed' else 'torch.bfloat16')
        and observed['graph_install'] == dict(norms_installed=57 if mode == 'mixed' else 0, before_graph_constructor=True)
        and observed['captured_buckets'] == observed['captured_graph_keys'] == settings['graph_bs']
        and observed['actual_backend'] == 'FlashInferBackend' and observed['prefill_no_split']
        and observed['linear_tile'] == 512)


def resources_ok(state):
    return (not state['pending'] and not state['running'] and not state['retired'] and not state['prefix_protected']
        and state['tables'] == 4 and state['free_pages'] + state['prefix_total'] == 2048
        and state['page_indices_unique'] and state['table_slots_unique']
        and state['page_indices_sha256'] == token_hash(list(range(2048))))


def work_ok(row, case, suite_config, settings):
    trace = row['batches']
    batch = case['batch_size']
    request_map = {r['request_id']: r for r in case['waves'][row['phase']]}
    good = set(request_map) == {r['request_id'] for r in row['requests']} and len(row['requests']) == batch
    good &= row['batch_trace_sha256'] == token_hash(trace)
    good &= all(r['output_tokens'] == 64 and r['free_count'] == 1 and r['reason'] == 'length'
        and r['output_sha256'] == token_hash(request_map[r['request_id']]['continuation_ids'])
        and 0 < r['first_result_ms'] <= r['terminal_ms'] <= row['seconds'] * 1000 for r in row['requests'])
    queries = sum(r['extend'] for b in trace if b['phase'] == 'prefill' for r in b['requests'])
    good &= queries == row['computed_prefill_tokens'] == batch*suite_config['prompt_tokens'] - sum(r['cached_tokens'] for r in row['requests'])
    if row['phase'] == 'warm':
        hit = suite_config['shared_prefix_tokens'] if case['prefix_mode'] == 'shared' and settings['cache'] == 'radix' else 0
        good &= all(r['cached_tokens'] == hit for r in row['requests'])
    replays = 0
    for b in trace:
        size = len(b['requests'])
        eligible = b['phase'] == 'decode' and bool(settings['graph_bs']) and size <= max(settings['graph_bs'])
        padded = next(x for x in settings['graph_bs'] if x >= size) if eligible else size
        good &= b['graph_eligible'] == eligible and b['padded'] == padded
        replays += eligible
    return bool(good and row['graph_replays'] == replays and resources_ok(row['released']))


def journal_ok(directory, expected_labels):
    journal = read(directory/'processes.json')
    return (len(journal) == len(expected_labels) and {r['label'] for r in journal} == set(expected_labels)
        and all(r['status'] == 'pass' and r['returncode'] == 0 and r['gpu_after'] == '0 MiB'
            and digest(ROOT/r['log']) == r['log_sha256'] for r in journal))


def audit_qualification(directory):
    import torch
    torch.set_num_threads(4)
    from safetensors import safe_open
    meta = read(directory/'provenance.json')
    config = meta['config']
    workload = read(directory/'workload.json')
    checks = dict(frozen_sources=check_sources(directory, meta),
        workload_hash=digest(directory/'workload.json') == meta['workload_sha256'],
        unchanged_threshold=config['numerics']['atol'] == .125 and config['numerics']['rtol'] == .02 and config['numerics']['max_relative_rmse'] == .005,
        model_hashes=all(digest(ROOT/config['model_path']/n) == sha for n, sha in meta['model_sha256'].items()),
        diagnostic_only=meta['performance_allowed'] is False and meta['stage'] == 'qualify')
    dtypes = set()
    for file in (ROOT/config['model_path']).glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as f:
            dtypes.update(f.get_slice(n).get_dtype() for n in f.keys())
    checks['BF16_checkpoint'] = dtypes == {'BF16'}
    requests, cases = {}, {}
    for suite in workload['suites']:
        validate_workload(suite['workload'], suite['config'])
        for case in suite['workload']['cases']:
            cases[(suite['name'], case['case_id'])] = (suite, case)
            for phase in ('cold', 'warm'):
                for r in case['waves'][phase]:
                    requests[r['request_id']] = r
    checks['heldout_scope'] = (len(requests) == 28 and len(cases) == 8 and len(workload['suites']) == 2
        and workload['suites'][1]['config']['prompt_tokens'] == 768
        and workload['suites'][1]['config']['seed'] == 20261101 and workload['suites'][1]['config']['batch_sizes'] == [1, 2]
        and all(r['continuation_ids'] == list(reversed(r['input_ids'][-64:])) for r in requests.values()))
    references = {}
    for mode, label in [('mixed', 'hf_reference'), ('bf16_control', 'native_reference')]:
        record = read(directory/f'{label}.json')
        path = ROOT/record['tensor']['path']
        checks[label+'_hash'] = digest(path) == record['tensor']['sha256'] and record['kind'] == label
        values = torch.load(path, weights_only=True, mmap=True)
        checks[label+'_coverage'] = set(values) == set(requests) and all(t.shape == (64,151936) and bool(torch.isfinite(t).all()) for t in values.values())
        references[mode] = values
        if mode == 'bf16_control':
            checks['native_reference_policy'] = policy_ok(record['observed'], mode, config['variants']['naive_full'])
    original_record = read(ROOT/'results/i_minisglang/i2_single_gpu_v1/attempt001/reference.json')
    original_path = ROOT/original_record['reference']['tensor_file']
    checks['original_HF_reference_hash'] = digest(original_path) == original_record['reference']['tensor_sha256']
    original = torch.load(original_path, weights_only=True, mmap=True)
    control = {rid: value for rid, value in references['mixed'].items() if rid.startswith('control512:')}
    checks['HF_control_reproduces_original'] = len(control) == 16 and all(torch.equal(v, original[rid.split(':',1)[1]]) for rid, v in control.items())
    old_native = read((ROOT/config['suite_source']).parent/'reference.json')['tensor']
    checks['historical_BF16_reference_hash'] = digest(ROOT/old_native['path']) == old_native['sha256']
    old_values = torch.load(ROOT/old_native['path'], weights_only=True, mmap=True)
    checks['BF16_control_reproduces_accepted_scope'] = set(old_values) == set(references['bf16_control']) and all(torch.equal(v, old_values[rid]) for rid,v in references['bf16_control'].items())
    reports, compared = {}, 0
    for mode in config['modes']:
        for variant, settings in config['variants'].items():
            key = f'{mode}_{variant}'
            record = read(directory/f'{key}.json')
            path = ROOT/record['tensor']['path']
            checks[key+'_tensor_hash'] = digest(path) == record['tensor']['sha256']
            values = torch.load(path, weights_only=True, mmap=True)
            checks[key+'_coverage'] = set(values) == set(record['errors']) == set(requests) and all(t.shape == (64,151936) for t in values.values())
            checks[key+'_policy'] = policy_ok(record['observed'], mode, settings)
            failures, exact, top1, maximum, max_abs, consistent = [], 0, 0, 0., 0., True
            for rid in sorted(requests):
                if len(record['errors'][rid]) != 64:
                    raise ValueError('Wrong number of step metrics')
                for step in range(64):
                    x, y = values[rid][step].float(), references[mode][rid][step].float()
                    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    diff = (x-y).abs()
                    rel = float((x-y).square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))
                    violations = int((diff > .125 + .02*y.abs()).sum())
                    equal = bool(torch.equal(x,y))
                    same_top1 = int(x.argmax()) == int(y.argmax())
                    passed = finite and violations == 0 and rel <= .005
                    if mode == 'bf16_control':
                        passed &= equal
                    old = record['errors'][rid][step]
                    consistent &= old['finite'] == finite and old['tolerance_violations'] == violations and old['top1_equal'] == same_top1
                    consistent &= math.isclose(old['relative_rmse'], rel, rel_tol=1e-6, abs_tol=1e-9) and math.isclose(old['max_abs'],float(diff.max()),rel_tol=1e-6,abs_tol=1e-9)
                    consistent &= old['passed'] == (finite and violations == 0 and rel <= .005)
                    compared += 1; exact += equal; top1 += same_top1
                    maximum = max(maximum, rel); max_abs = max(max_abs, float(diff.max()))
                    if not passed:
                        failures.append(dict(request_id=rid, step=step, finite=finite, relative_rmse=rel, violations=violations, exact=equal))
            checks[key+'_metrics'] = bool(consistent) and record['passed'] == all(e['passed'] for row in record['errors'].values() for e in row)
            seen, work = set(), True
            for row in record['rows']:
                pair = (row['suite'], row['case_id'])
                suite, case = cases[pair]
                seen.add((*pair,row['phase']))
                work &= work_ok(row, case, suite['config'], settings)
            checks[key+'_actual_work_and_resources'] = bool(work) and len(seen) == len(record['rows']) == 16
            reports[key] = dict(passed=not failures, vectors=1792, exact_vectors=exact, top1_equal_vectors=top1,
                max_relative_rmse=maximum, max_abs=max_abs, failures=failures, evidence_sha256=digest(directory/f'{key}.json'))
            print('numerical audit',key,'passed',not failures,'max_relative_rmse',maximum,flush=True)
    labels = ['hf_reference','native_reference'] + [f'numeric_{m}_{v}' for m in config['modes'] for v in config['variants']]
    checks['sixteen_clean_processes'] = journal_ok(directory, labels)
    checks['all_target_vectors'] = compared == 25088
    integrity = all(checks.values())
    return dict(stage='qualification', integrity_passed=integrity,
        qualification_passed=integrity and all(r['passed'] for r in reports.values()), checks=checks, profiles=reports,
        compared_vectors=compared, mixed_external_HF_vectors=12544, baseline_internal_vectors=12544,
        original_BF16_external_HF_passed=False, performance_allowed=False, paid_resources_used=False,
        auditor_sha256=digest(Path(__file__)), provenance_sha256=digest(directory/'provenance.json'))


def audit_http(directory):
    from scripts.run_i2_service_gate import audit_service
    meta = read(directory/'provenance.json')
    config = meta['config']
    checks = dict(frozen_sources=check_sources(directory,meta),
        numerical_binding=meta['qualification_audit_sha256'] == digest(ROOT/config['qualification_directory']/'audit.json'))
    reports = {}
    for mode in config['modes']:
        for variant, settings in config['variants'].items():
            path = directory/mode/f'service_{variant}'
            actual = audit_service(path,config)
            key = f'{mode}_{variant}'
            checks[key+'_lifecycle'] = actual['passed'] and actual == read(path/'audit.json')
            events = read(path/'events.json')
            policy = [r for r in events if r['event'] == 'i2_precision_policy']
            checks[key+'_policy'] = len(policy) == 1 and policy_ok(policy[0],mode,settings)
            checks[key+'_policy_sources'] = len(policy) == 1 and policy[0]['source_sha256'] == {n:meta['source_sha256'][n] for n in
                ('serving/i2_precision_policy.py','serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py')}
            reports[key] = dict(passed=actual['passed'], scenario_checks=len(actual['checks']),
                requests=len(read(path/'requests.json')), audit_sha256=digest(path/'audit.json'))
            print('HTTP audit',key,'passed',actual['passed'],flush=True)
    labels = [f'service_{m}_{v}' for m in config['modes'] for v in config['variants']]
    checks['fourteen_clean_service_processes'] = journal_ok(directory,labels)
    checks['scope_counts'] = sum(r['scenario_checks'] for r in reports.values()) == 560 and sum(r['requests'] for r in reports.values()) == 168
    return dict(stage='http', HTTP_lifecycle_passed=all(checks.values()), checks=checks, profiles=reports,
        requests=sum(r['requests'] for r in reports.values()), scenario_checks=sum(r['scenario_checks'] for r in reports.values()),
        performance_allowed=False, auditor_sha256=digest(Path(__file__)), provenance_sha256=digest(directory/'provenance.json'))


def audit_performance(directory):
    from scripts.run_i2_precision import prerequisites
    from serving.i2_precision_policy import POLICIES
    meta = read(directory/'provenance.json')
    config = meta['config']
    q, l = ROOT/config['qualification_directory'], ROOT/config['lifecycle_directory']
    qualification = audit_qualification(q)
    http = audit_http(l)
    checks = dict(independent_numerical_recompute=qualification == read(q/'audit.json') and qualification['qualification_passed'],
        independent_HTTP_recompute=http == read(l/'audit.json') and http['HTTP_lifecycle_passed'],
        gate_binding=prerequisites(config) == read(directory/'gate.json'),
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['qualify','http','performance'],required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    path=args.directory/'audit.json'
    if path.exists():
        raise ValueError('Preserve independent audit; call audit function for read-only rechecks')
    result={'qualify':audit_qualification,'http':audit_http,'performance':audit_performance}[args.stage](args.directory)
    path.write_text(json.dumps(result,indent=2)+'\n')
    dest=args.directory/'audit_source_snapshot/scripts'
    dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),dest/Path(__file__).name)
    success=result.get('qualification_passed',result.get('HTTP_lifecycle_passed',result.get('passed',False)))
    print(json.dumps({k:v for k,v in result.items() if k not in {'checks','statistics','precision_costs','comparisons','profiles'}},indent=2))
    raise SystemExit(0 if success else 1)


if __name__=='__main__':
    main()
