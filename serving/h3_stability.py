"""Separate H3 timing replication; the frozen v5 computation is reused verbatim."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path
from statistics import median, pstdev
import subprocess
import time

from serving.backends.nanovllm import ROOT, digest
from serving.h3_performance import SCOPES, SOURCES, VARIANTS, value_hash
from scripts.summarize_h3_performance import summarize_traces, validate_run

CONFIG = ROOT / 'config/serving/h3_stability_v1.json'
EXTRA_SOURCES = ('serving/h3_stability.py', 'scripts/run_h3_stability.py',
    'scripts/audit_h3_stability.py', 'scripts/export_h3_stability.py', 'scripts/accept_h3_stability.py',
    'tests/test_h3_stability.py', 'tests/test_h3_performance.py', 'config/serving/h3_stability_v1.json',
    'config/serving/h3_performance_v5.json', 'config/serving/h1_nanovllm.json',
    'docs/h3_stability_protocol.md', 'docs/h3_performance_v5_protocol.md')


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def config(path=CONFIG):
    c = read(path)
    if (c['stage'] != 'H3' or c['schema_version'] != 1 or c['rounds'] != 6
            or tuple(c['variants']) != VARIANTS or c['case_ids'] != ['long_prefill', 'batch_short']
            or c['warmup_repeats'] != 20 or c['measure_repeats'] != 20
            or c['sensor_every_waves'] != 5 or c['require_AC'] is not True
            or c['max_within_process_cv'] != .15
            or c['max_between_process_throughput_ratio'] != 1.25 or c['max_paired_ratio_cv'] != .15):
        raise ValueError('Unsupported frozen replication contract')
    return c


def plan(c):
    permutations = list(itertools.permutations(VARIANTS))
    # All six orders, with cyclic orders first and the other permutations after.
    orders = [permutations[i] for i in (0, 3, 4, 5, 2, 1)]
    jobs = []
    for index, order in enumerate(orders):
        case_order = c['case_ids'] if index % 2 == 0 else list(reversed(c['case_ids']))
        for case_id in case_order:
            for position, variant in enumerate(order):
                jobs.append(dict(case_id=case_id, variant=variant, round=index + 1,
                    position=position, label=f'{case_id}_{variant}_run{index + 1}',
                    scopes=list(SCOPES if index % 2 == 0 else reversed(SCOPES))))
    return jobs


def ac_state():
    return {str(p): p.read_text().strip() for p in sorted(Path('/sys/class/power_supply').glob('*/online'))}


def require_ac(c):
    states = ac_state()
    if c['require_AC'] and not any(v == '1' for v in states.values()):
        raise ValueError(f'AC power required for this protocol: {states}')
    return states


def gpu_idle():
    # Utilization is a lagging sample. Allow it to settle only when memory is
    # already zero; external memory use is rejected immediately.
    deadline = time.monotonic() + 5
    while True:
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader'], text=True).strip()
        if value == '0 MiB, 0 %': return value
        if not value.startswith('0 MiB,') or time.monotonic() >= deadline:
            raise ValueError(f'Exclusive idle GPU required: {value}')
        time.sleep(.25)


def sensors(label):
    fields = ['pstate', 'temperature.gpu', 'power.draw', 'power.limit', 'clocks.sm', 'clocks.mem',
        'memory.used', 'utilization.gpu']
    start = time.monotonic()
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=' + ','.join(fields),
        '--format=csv,noheader,nounits'], text=True).strip()
    values = next(csv.reader([raw]))
    return dict(label=label, monotonic_s=start, collection_s=time.monotonic() - start,
        gpu=dict(zip(fields, [v.strip() for v in values])), cpu_load=list(os.getloadavg()),
        AC=ac_state(), timing_boundary='outside_run_wave')


def check_sources(directory):
    meta = read(directory / 'provenance.json')
    if not all(digest(ROOT / name) == sha == digest(directory / 'source_snapshot' / name)
               for name, sha in meta['source_sha256'].items()):
        raise ValueError('Frozen replication source changed')
    if not all(digest(ROOT / name) == sha for name, sha in meta['evidence_sha256'].items()):
        raise ValueError('Frozen v5 qualification/configuration/workload changed')
    if not all(digest(ROOT / name) == sha for name, sha in read(directory / 'historical.json').items()):
        raise ValueError('Historical evidence changed')
    return meta


def qualification_audit(directory):
    """Reopen every new and reference tensor; admission never trusts pass flags alone."""
    import torch
    from serving.h3_performance import compare_tensors
    meta = check_sources(directory)
    c = meta['config']; old = read(ROOT / c['base_config']); old_dir = ROOT / old['results_dir']
    old_reference = read(old_dir / 'reference.json')
    old_check = read(directory / 'baseline_check.json')
    if old_check['returncode'] != 0 or digest(ROOT / old_check['log']) != old_check['log_sha256']:
        raise ValueError('Read-only original v5 full audit did not complete')
    tensors = {}; checks = {}; hashes = {}; comparisons = {}; argmax = {}; vectors = exact_vectors = 0
    for variant in VARIANTS:
        p = directory / 'qualification' / f'{variant}.json'; record = read(p)
        hashes[str(p.relative_to(ROOT))] = digest(p)
        if record['status'] != 'pass' or record['base_provenance'] != meta['base_provenance']:
            raise ValueError('New qualification process failed or provenance differs')
        if record['replication_source_sha256'] != meta['source_sha256']:
            raise ValueError('New qualification source differs')
        if set(record['cases']) != set(c['case_ids']):
            raise ValueError('Incomplete numerical case set')
        for case_id in c['case_ids']:
            q = record['cases'][case_id]; ref = old_reference['cases'][case_id]
            for item in (q, ref):
                path = ROOT / item['path']; hashes[item['path']] = digest(path)
                if hashes[item['path']] != item['sha256']:
                    raise ValueError('New qualification or original reference tensor hash differs')
            t = torch.load(ROOT / q['path'], weights_only=True, mmap=True, map_location='cpu')
            reference = torch.load(ROOT / ref['path'], weights_only=True, mmap=True, map_location='cpu')
            result = compare_tensors(t, reference, old['tolerances'], fp32_reference=True)
            comparisons[f'{variant}/{case_id}'] = result
            checks[f'{variant}/{case_id}/numerical'] = result['passed']; vectors += result['vectors']
            checks[f'{variant}/{case_id}/weights'] = q['weight_sha256'] == old_reference['weight_sha256']
            case = next(row for row in meta['cases'] if row['case_id'] == case_id)
            trace = q['trace']; positions = case['prompt_length'] + case['output_length'] - 1
            checks[f'{variant}/{case_id}/resources'] = (trace['released'] and trace['cached_positions'] == positions
                and trace['max_used_blocks'] == (0 if variant == 'native_kv' else case['batch_size'] * math.ceil(positions / 256))
                and trace['output_tokens'] == case['batch_size'] * case['output_length']
                and trace['output_sha256'] == value_hash(case['continuation_ids'])
                and trace['prefill_calls'] == 1 and trace['decode_calls'] == case['output_length'] - 1
                and trace['prefix_tokens'] == 0)
            argmax[f'{variant}/{case_id}'] = value_hash(t.argmax(-1).tolist())
            tensors[variant, case_id] = t
    for case_id in c['case_ids']:
        result = compare_tensors(tensors['paged_graph', case_id], tensors['paged_eager', case_id],
            old['tolerances'], exact=True)
        comparisons[f'graph_exact/{case_id}'] = result
        checks[f'graph_exact/{case_id}'] = result['passed']; exact_vectors += result['vectors']
    checks.update(vectors=vectors == 1728, exact_vectors=exact_vectors == 576)
    return dict(passed=all(checks.values()), checks=checks, vectors=vectors, exact_vectors=exact_vectors,
        comparisons=comparisons, argmax_sha256=argmax, artifact_sha256=hashes, baseline_check_sha256=digest(directory / 'baseline_check.json'),
        provenance_sha256=digest(directory / 'provenance.json'), auditor_sha256=digest(ROOT / 'scripts/audit_h3_stability.py'))


def validate_record(record, meta, c):
    """Apply the frozen v5 validator to actual new traces and memory values."""
    case = next(row for row in meta['cases'] if row['case_id'] == record['job']['case_id'])
    old = dict(read(ROOT / c['base_config']), warmup_repeats=c['warmup_repeats'], measure_repeats=c['measure_repeats'])
    run = dict(record, benchmark='routescale_performance', evidence_class='formal_gpu',
        variant=record['job']['variant'], case_id=case['case_id'], provenance=record['base_provenance'],
        qualification_gate_sha256=meta['base_gate_sha256'])
    validate_run(run, old, case, meta['base_provenance'], meta['base_gate_sha256'])
    warm = {scope: dict(m, traces=m['warmup_traces']) for scope, m in run['measurements'].items()}
    validate_run(dict(run, measurements=warm), old, case, meta['base_provenance'], meta['base_gate_sha256'])


def admission(directory):
    check_sources(directory)
    recomputed = qualification_audit(directory)
    saved = read(directory / 'qualification_gate.json')
    if not recomputed['passed'] or recomputed != saved:
        raise ValueError('Qualification gate failed or is stale')
    return saved


def cv(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Timing samples must be finite and positive')
    return pstdev(values) / (sum(values) / len(values))


def summarize(records, c):
    expected = {r['label'] for r in plan(c)}
    if len(records) != len(expected) or {r['job']['label'] for r in records} != expected:
        raise ValueError('Missing, duplicate or unexpected formal process')
    rows = []; groups = []; pairs = []
    for r in records:
        for scope in SCOPES:
            rows.append(dict(**r['job'], measurement_scope=scope,
                **summarize_traces(r['measurements'][scope]['traces'], scope)))
    for case_id in c['case_ids']:
        for scope in SCOPES:
            for variant in VARIANTS:
                members = sorted((r for r in rows if r['case_id'] == case_id and r['variant'] == variant
                    and r['measurement_scope'] == scope), key=lambda r: r['round'])
                values = [r['output_tokens_per_second'] for r in members]
                stable = (all(max(r['wall_cv'], r['measurement_cv']) <= c['max_within_process_cv'] for r in members)
                    and max(values) / min(values) <= c['max_between_process_throughput_ratio'])
                metrics = {k: median(r[k] for r in members) if members[0][k] is not None else None
                    for k in summarize_traces(records[0]['measurements'][scope]['traces'], scope)}
                groups.append(dict(case_id=case_id, variant=variant, measurement_scope=scope,
                    **metrics, throughput_values=values, throughput_min=min(values), throughput_max=max(values),
                    max_min_ratio=max(values) / min(values), between_process_cv=cv(values),
                    maximum_within_process_cv=max(max(r['wall_cv'], r['measurement_cv']) for r in members), stable=stable))
            for base, candidate in (('native_kv', 'paged_eager'), ('paged_eager', 'paged_graph'), ('native_kv', 'paged_graph')):
                lookup = {(r['variant'], r['round']): r['output_tokens_per_second'] for r in rows
                    if r['case_id'] == case_id and r['measurement_scope'] == scope}
                ratios = [lookup[candidate, i] / lookup[base, i] for i in range(1, c['rounds'] + 1)]
                pairs.append(dict(case_id=case_id, measurement_scope=scope, baseline=base, candidate=candidate,
                    paired_ratios=ratios, median_ratio=median(ratios), minimum_ratio=min(ratios), maximum_ratio=max(ratios),
                    ratio_cv=cv(ratios), all_six_above_one=all(v > 1 for v in ratios), stable=cv(ratios) <= c['max_paired_ratio_cv']))
    stable = all(g['stable'] for g in groups) and all(p['stable'] for p in pairs)
    return dict(processes=len(records), process_scope_rows=rows, statistics=groups, comparisons=pairs,
        timing_stable=stable, timing_status='stable_in_this_sample' if stable else 'variability_remains',
        finite_offline_replay=True, steady_state_claimed=False)
