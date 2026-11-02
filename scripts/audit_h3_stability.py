"""Independent numerical admission and complete raw-trace H3 replication audit."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_performance import SCOPES, VARIANTS, value_hash
from serving.h3_stability import admission, check_sources, config, gpu_idle, plan, qualification_audit, read, summarize, validate_record


def pid_alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    return True


def audit(directory):
    meta = check_sources(directory); c = meta['config']; gate = admission(directory)
    checks = dict(frozen_sources=True, historical_evidence=True, qualification=gate['passed'])
    rows = read(directory / 'processes.json'); expected_jobs = plan(c)
    labels = ['original_v5_readonly_audit', *['qualification_' + v for v in VARIANTS],
        *['formal_' + j['label'] for j in expected_jobs]]
    checks['serial_complete_journal'] = len(rows) == 40 and [r['label'] for r in rows] == labels
    for r in rows:
        checks[r['label'] + '/process'] = (r['status'] == 'pass' and r['returncode'] == 0
            and r['gpu_before'] == r['gpu_after'] == '0 MiB, 0 %' and not r['escalations'] and not pid_alive(r['pid'])
            and digest(ROOT / r['log']) == r['log_sha256'] and 0 < r['duration_s'] <= c['subprocess_timeout_seconds'])
    records = []; hashes = {}
    raw_paths = list((directory / 'raw').glob('*.json'))
    checks['exact_raw_set'] = {p.stem for p in raw_paths} == {j['label'] for j in expected_jobs} and len(raw_paths) == 36
    ref = read(ROOT / read(ROOT / c['base_config'])['results_dir'] / 'reference.json')
    for job in expected_jobs:
        p = directory / 'raw' / f'{job["label"]}.json'; r = read(p); records.append(r)
        hashes[str(p.relative_to(directory))] = digest(p); prefix = job['label']
        checks[prefix + '/job'] = r['job'] == job and list(r['measurements']) == job['scopes']
        checks[prefix + '/source'] = r['replication_source_sha256'] == meta['source_sha256']
        checks[prefix + '/gate'] = r['gate_sha256'] == digest(directory / 'qualification_gate.json')
        checks[prefix + '/actual_policy'] = (r['actual_reduction_permission'] is False
            and r['weight_sha256'] == ref['weight_sha256']
            and r['graph_keys'] == ([1,2,4,8] if job['variant'] == 'paged_graph' else [])
            and (r['initialization']['graph_capture_seconds'] > 0 if job['variant'] == 'paged_graph'
                 else r['initialization']['graph_capture_seconds'] == 0))
        validate_record(r, meta, c); checks[prefix + '/v5_timing_resource_contract'] = True
        # Fresh numerical logits determine the expected prediction checksum per path.
        expected_argmax = gate['argmax_sha256'][f'{job["variant"]}/{job["case_id"]}']
        e2e = r['measurements']['replay_e2e']
        checks[prefix + '/actual_argmax'] = all(t['argmax_sha256'] == expected_argmax for t in [*e2e['warmup_traces'], *e2e['traces']])
        labels_expected = [f'{scope}/{suffix}' for scope in job['scopes'] for suffix in
            ('before_warmup', 'after_warmup', 'after_measure_5', 'after_measure_10', 'after_measure_15', 'after_measure_20')]
        checks[prefix + '/sensor_boundaries'] = ([s['label'] for s in r['sensors']] == labels_expected
            and all(s['timing_boundary'] == 'outside_run_wave' and s['collection_s'] > 0
                and any(v == '1' for v in s['AC'].values()) for s in r['sensors'])
            and all(b['monotonic_s'] > a['monotonic_s'] + a['collection_s'] for a,b in zip(r['sensors'],r['sensors'][1:])))
    summary = summarize(records, c)
    checks['all_formal_pids_dead'] = all(not pid_alive(r['pid']) for r in records)
    cpu = read(directory / 'cpu_tests.json')
    checks['CPU_tests'] = cpu['passed'] and cpu['tests'] >= 12 and digest(ROOT / cpu['log']) == cpu['log_sha256']
    checks['CPU_sources'] = all(digest(ROOT / n) == sha == meta['source_sha256'][n] for n,sha in cpu['source_sha256'].items())
    checks['GPU_idle'] = gpu_idle() == '0 MiB, 0 %'
    return dict(stage='H3', scope=c['scope'], investigation_complete=all(checks.values()), checks=checks,
        qualification_vectors=gate['vectors'], graph_exact_vectors=gate['exact_vectors'], formal_processes=36,
        measured_waves=36 * 2 * c['measure_repeats'], warmup_waves=36 * 2 * c['warmup_repeats'],
        **summary, raw_sha256=hashes, provenance_sha256=digest(directory / 'provenance.json'),
        gate_sha256=digest(directory / 'qualification_gate.json'), journal_sha256=digest(directory / 'processes.json'),
        auditor_sha256=digest(Path(__file__)), paid_resources_used=False, TP2_4_complete=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--stage', choices=['qualification','final'], required=True)
    p.add_argument('--check-only', action='store_true'); args = p.parse_args()
    directory = ROOT / config()['results_dir']; name = 'qualification_gate.json' if args.stage == 'qualification' else 'audit.json'
    target = directory / name
    if target.exists() and not args.check_only: raise ValueError('Preserve original audit; use --check-only')
    result = qualification_audit(directory) if args.stage == 'qualification' else audit(directory)
    passed = result['passed'] if args.stage == 'qualification' else result['investigation_complete']
    if args.check_only:
        if result != read(target): raise ValueError('Independent recomputation differs')
    else: write_result(result, target)
    print('H3', args.stage, 'audit:', passed, 'checks:', len(result['checks']), flush=True)
    if not passed: raise ValueError('H3 audit failed')


if __name__ == '__main__': main()
