"""Independent timing accounting. Each independent process is one observation."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics

from rental.common import ROOT, positive, read, sha, validate_hashes, write
from serving.tp_preparation import jobs, load_plan, numerical_gate
from scripts.prepare_f_training import performance_gate, jobs as training_jobs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    values = [positive(v) for v in values]
    mean = statistics.mean(values)
    return dict(n=len(values), mean=mean, median=statistics.median(values), min=min(values), max=max(values),
                cv=statistics.stdev(values) / mean if len(values) > 1 else None)


def measured_waves(record, job, workload, config):
    require(record['passed'] and record['mode'] == job['stage'] and record['world'] == job['world'] and
            record['label'] == job['label'] and record['framework'] == job['framework'] and
            record['variant'] == job['variant'], 'TP run identity or status differs')
    repeats = config['warmup_repeats'] + config['measure_repeats'] if job['stage'] == 'benchmark' else 1
    expected = {(repeat, case['case_id'], phase): case['waves'][phase]
                for repeat in range(repeats) for case in workload['cases'] for phase in ('cold', 'warm')}
    actual = {}
    for wave in record['waves']:
        key = (wave['repeat'], wave['case_id'], wave['phase'])
        require(key in expected and key not in actual, 'Missing/duplicate/unexpected wave')
        require(wave['released'] is True, 'TP request resources not released')
        positive(wave['seconds'])
        measured = job['stage'] == 'benchmark' and key[0] >= config['warmup_repeats']
        require(wave['measured'] is measured, 'Warmup incorrectly included in timing')
        requests = {r['request_id']: r for r in wave['requests']}
        require(len(requests) == len(wave['requests']) and set(requests) == {r['request_id'] for r in expected[key]},
                'TP request accounting mismatch')
        for request in expected[key]:
            row = requests[request['request_id']]
            require(row['output_tokens'] == len(request['continuation_ids']), 'Wrong generated work')
            if 'actual_tokens' in row:
                require(row['actual_tokens'] == request['continuation_ids'], 'Wrong forced tokens')
            if 'free_count' in row:
                require(row['free_count'] == 1 and row['reason'] == 'length', 'Wrong terminal state')
        actual[key] = wave
    require(set(actual) == set(expected), 'Incomplete timing wave matrix')
    return {key: wave for key, wave in actual.items() if wave['measured']}


def summarize_tp(directory):
    config = load_plan()
    numerical_gate(directory, config)
    workload = read(directory / 'workload.json')
    journal = read(directory / 'processes.json')
    counts = {}
    for process in journal:
        counts.setdefault(process['label'], []).append(process)
    rows, evidence = [], {}
    for job in jobs(config, 'benchmark'):
        launched = counts.get(job['label'], [])
        require(len(launched) == 1 and launched[0]['returncode'] == 0 and not launched[0]['escalated'],
                'Missing/duplicate/failed TP performance process')
        log = directory / 'logs' / f'{job["label"]}.log'
        require(sha(log) == launched[0]['log_sha256'], 'TP worker log changed')
        ranks = []
        for rank in range(job['world']):
            path = directory / 'raw' / job['label'] / f'rank{rank}.json'
            report = read(path); evidence[str(path.relative_to(directory))] = sha(path)
            require(report['rank'] == rank and report['workload_sha256'] == sha(directory / 'workload.json') and
                    report['config_sha256'] == sha(ROOT / 'config/delivery/tp_matrix.json'), 'TP report provenance differs')
            require(report['shutdown_s'] <= config['shutdown_grace_s'], 'TP shutdown budget exceeded')
            ranks.append((report, measured_waves(report, job, workload, config)))
        for case in workload['cases']:
            for phase in ('cold', 'warm'):
                keys = [(repeat, case['case_id'], phase) for repeat in range(config['warmup_repeats'],
                        config['warmup_repeats'] + config['measure_repeats'])]
                # Concurrent ranks process the same global requests. Count work
                # once and use the maximum rank duration for each aligned wave.
                seconds = [max(waves[key]['seconds'] for _, waves in ranks) for key in keys]
                tokens = sum(len(r['continuation_ids']) for r in case['waves'][phase])
                rows.append(dict(framework=job['framework'], variant=job['variant'], world=job['world'],
                                 run=job['run'], case=case['case_id'], phase=phase,
                                 output_tokens_per_s=tokens * len(seconds) / sum(seconds),
                                 mean_wave_s=statistics.mean(seconds), wave_cv=distribution(seconds)['cv'],
                                 peak_rank_allocated=max(r['peak_allocated_bytes'] for r, _ in ranks),
                                 peak_rank_reserved=max(r['peak_reserved_bytes'] for r, _ in ranks)))
    return rows, evidence


def training_record(report, job, rank, config):
    t = config['training']
    require(report['rank'] == rank and report['world'] == job['world'] and report['tiny'] is False and
            report['dtype'] == 'bfloat16' and report['accumulation'] == t['accumulation_steps'] and
            report['global_tokens_per_update'] == t['global_tokens_per_update'], 'Wrong F training work/precision')
    require(len(report['entries']) == 1, 'Unexpected training modes')
    entry = report['entries'][0]
    require(entry['mode'] == job['mode'] and entry['warmup'] == t['warmup_updates'] and
            len(entry['update_seconds']) == t['measured_updates'], 'Incomplete F timing samples')
    for value in entry['update_seconds']:
        positive(value)
    require(math.isfinite(entry['last_loss']) and math.isfinite(entry['last_global_grad_norm']), 'Nonfinite training state')
    require(report['peak_allocated'] > 0, 'Missing rank memory')
    return entry['update_seconds']


def summarize_training(directory):
    performance_gate(directory)
    config = read(ROOT / 'config/f_ep_rental_requirements.json')
    rows, evidence, windows = [], {}, set()
    for job in training_jobs(benchmark=True):
        path = directory / 'performance' / job['label']
        manifest = read(path / 'job.json')
        require(manifest['returncode'] == 0 and not manifest['timed_out'] and
                all(manifest[k] == v for k, v in job.items()), 'Failed/mismatched F performance process')
        require(manifest['host'] == read(directory / 'training/cuda1_full/job.json')['host'], 'F performance host differs')
        validate_hashes(ROOT, manifest['source_sha256'])
        times, memory = [], []
        for rank in range(job['world']):
            report = read(path / f'rank{rank}.json')
            times.append(training_record(report, job, rank, config))
            memory.append(report['peak_allocated'])
            windows.add(report['window_sha256'])
        require(all(values == times[0] for values in times), 'F rank-max timings differ across ranks')
        elapsed = sum(times[0])
        rows.append(dict(mode=job['mode'], world=job['world'], run=int(job['label'].rsplit('run', 1)[1]),
                         tokens_per_s=config['training']['global_tokens_per_update'] * len(times[0]) / elapsed,
                         mean_update_s=statistics.mean(times[0]), update_cv=distribution(times[0])['cv'],
                         peak_rank_allocated=max(memory)))
        for file in [path / 'job.json', path / 'worker.log', *path.glob('rank*.json')]:
            evidence[str(file.relative_to(directory))] = sha(file)
    require(len(windows) == 1, 'Different global training windows across modes/runs')
    return rows, evidence


def comparisons(rows, keys, value, baseline):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in grouped.items():
        require(len(group) == 3 and {r['run'] for r in group} == {1, 2, 3}, 'Three independent processes required')
        bases = baseline(group[0], rows)
        ratios = [r[value] / bases[r['run']][value] for r in group]
        result.append(dict(zip(keys, key), performance=distribution([r[value] for r in group]),
                           paired_speedup=distribution(ratios),
                           peak_rank_allocated=max(r['peak_rank_allocated'] for r in group)))
    return result


def interval_union(intervals):
    total = 0.0
    end = -math.inf
    for begin, stop in sorted(intervals):
        total += max(0, stop - max(begin, end))
        end = max(stop, end)
    return total


def trace_summary(path):
    events = read(path)['traceEvents']
    communication, computation = [], []
    for event in events:
        if event.get('ph') != 'X' or 'kernel' not in event.get('cat', '').lower():
            continue
        start, duration = float(event['ts']), float(event['dur'])
        require(math.isfinite(start) and math.isfinite(duration) and duration >= 0, 'Invalid trace interval')
        (communication if 'nccl' in event.get('name', '').lower() else computation).append((start, start + duration))
    require(bool(communication or computation), 'No CUDA kernels in trace')
    comm, compute = interval_union(communication), interval_union(computation)
    union = interval_union(communication + computation)
    return dict(sha256=sha(path), nccl_kernels=len(communication), other_kernels=len(computation),
                nccl_union_us=comm, other_kernel_union_us=compute, kernel_union_us=union,
                overlap_us=max(0, comm + compute - union),
                note='Per-rank profiled CUDA kernel unions; other kernels include copies/helpers. Not unprofiled speedup or network bytes.')


def summarize_traces(directory):
    result = {}
    tp = directory / 'tp'
    config = load_plan()
    workload = read(tp / 'workload.json')
    for job in jobs(config, 'trace'):
        for rank in range(job['world']):
            report = read(tp / 'raw' / job['label'] / f'rank{rank}.json')
            measured_waves(report, job, workload, config)
            path = tp / 'traces' / f'{job["label"]}_rank{rank}.json'
            require(sha(path) == report['trace_sha256'], 'TP trace changed')
            summary = trace_summary(path)
            require(job['world'] == 1 or summary['nccl_kernels'] > 0, 'Missing multi-GPU NCCL trace')
            result[str(path.relative_to(directory))] = summary
    f = directory / 'f/trace'
    audit = read(f / 'audit.json')
    require(audit['passed'] and audit['job_sha256'] == sha(f / 'job.json'), 'F trace numerical audit failed')
    require(audit['world'] == 2 and audit['device'] == 'cuda' and audit['cross_gpu_accepted'], 'Wrong F trace audit scope')
    validate_hashes(ROOT, read(f / 'job.json')['source_sha256'])
    for name, digest in audit['raw_sha256'].items():
        require(sha(name) == digest, 'Changed F trace numerical tensors')
    validate_hashes(f, audit['report_sha256'])
    for rank in (0, 1):
        path = f / f'trace_rank{rank}.json'
        summary = trace_summary(path)
        require(summary['nccl_kernels'] > 0, 'Missing EP2 NCCL trace')
        result[str(path.relative_to(directory))] = dict(summary,
            scope='D/E module qualification harness: reference + replicated + EP; not isolated EP or full GPT timing',
            logical_traffic=audit['traffic'], ep_cpu_scopes=read(f / f'rank{rank}.json').get('events', []))
    return result


def export(directory, destination):
    require(directory not in destination.parents and destination != directory, 'Export outside raw session to preserve evidence')
    destination.mkdir(parents=True, exist_ok=False)
    session = read(directory / 'session.json')
    report = dict(passed=False, multigpu_accepted=False, session_status=session['status'],
                  offered_steps=len(read(directory / 'plan.json')['steps']), recorded_steps=session['steps'])
    write(destination / 'summary.json', report)
    try:
        require(session['passed'] and session['completed_through'] == 'trace', 'Incomplete session; failures retained, no accepted ranking')
        require(sha(directory / 'plan.json') == session['plan_sha256'] and
                sha(directory / 'host_contract.json') == session['host_contract_sha256'] and
                sha(directory / 'preflight.json') == session['preflight_sha256'], 'Session manifest changed')
        expected = [s['name'] for s in read(directory / 'plan.json')['steps']]
        require([s['name'] for s in session['steps']] == expected and all(s['passed'] for s in session['steps']),
                'Missing or failed stage')
        validate_hashes(ROOT, read(directory / 'preflight.json')['source_sha256'])
        preflight = read(directory / 'preflight.json')
        if 'deployment_sha256' in preflight:
            require(sha(ROOT / 'runs/rental/deployment.json') == preflight['deployment_sha256'], 'Deployment manifest changed')
        for step in session['steps']:
            require(sha(directory / f'{step["name"]}.log') == step['log_sha256'], 'Stage log changed')
        tp, tp_hashes = summarize_tp(directory / 'tp')
        training, f_hashes = summarize_training(directory / 'f')
        traces = summarize_traces(directory)
        tp_stats = comparisons(tp, ['framework', 'variant', 'world', 'case', 'phase'], 'output_tokens_per_s',
                    lambda row, rows: {r['run']: r for r in rows if r['world'] == 1 and
                                      all(r[k] == row[k] for k in ('framework', 'variant', 'case', 'phase'))})
        f_stats = comparisons(training, ['mode', 'world'], 'tokens_per_s',
                              lambda row, rows: {r['run']: r for r in rows if r['mode'] == 'reference'})
        report.update(passed=True, multigpu_accepted=True, tp=tp_stats, training=f_stats, traces=traces,
                      scope='prepared Qwen3-0.6B TP and manual-reduction MoE matrix only',
                      inference_latency='fixed-token scheduler replay; no HTTP TTFT/SLO claim',
                      training_control='manual replicated reduction, not native DDP reducer',
                      uncertainty='Three independent processes; publish CV and min/max, no universal scaling claim')
        for name, rows in (('tp_processes', tp), ('training_processes', training)):
            with (destination / f'{name}.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        evidence = {**{'tp/' + k: v for k, v in tp_hashes.items()}, **{'f/' + k: v for k, v in f_hashes.items()}}
        evidence.update({k: v['sha256'] for k, v in traces.items()})
        # Bind all compact manifests, logs and gates; raw numerical tensor hashes
        # are transitively bound by the original independent qualification gates.
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.log', '.txt'):
                evidence[str(path.relative_to(directory))] = sha(path)
        report['evidence_sha256'] = evidence
    except Exception as exc:
        report.update(passed=False, multigpu_accepted=False, error=f'{type(exc).__name__}: {exc}')
    write(destination / 'summary.json', report)
    return report
